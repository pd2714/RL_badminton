from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from badminton.config import SimulationConfig
from badminton.movement import advance_player_toward, can_arrive_by_time, intercept_body_target_after_reaction
from badminton.state import ShotAction, Side, StageRecord, StageState, ValidatedShotAction
from badminton.trajectory import (
    TrajectoryResult,
    ballistic_landing_point,
    ballistic_landing_time,
    ballistic_net_crossing,
    ballistic_position,
    simulate_trajectory,
)
from badminton.utils import (
    opponent_side,
    player_position,
    player_velocity,
    recovery_bounds,
    service_target_bounds_for_receiver_state,
    target_bounds_for_receiver,
)


REACTION_MISS_FLIGHT_TIME_THRESHOLD = 0.1
REACTION_MISS_PROBABILITY = 0.8
REACTION_MISS_SECONDARY_FLIGHT_TIME_THRESHOLD = 0.5
REACTION_MISS_SECONDARY_PROBABILITY = 0.0
REACTION_MISS_ZERO_FLIGHT_TIME_THRESHOLD = 0.5


@dataclass(frozen=True)
class PreparedShot:
    validated_action: ValidatedShotAction
    trajectory: TrajectoryResult
    candidate_times: np.ndarray
    candidate_xs: np.ndarray
    candidate_ys: np.ndarray
    candidate_zs: np.ndarray
    feasible_indices: list[int]


def reaction_time_for_side(state: StageState, side: str) -> float:
    if side == "left":
        return float(state.reaction_time_left)
    return float(state.reaction_time_right)


def reaction_miss_probability(flight_time: float, config: SimulationConfig | None = None) -> float:
    action_config = SimulationConfig().action if config is None else config.action
    t = float(flight_time)
    fast_t = action_config.reaction_miss_fast_threshold
    secondary_t = action_config.reaction_miss_secondary_threshold
    zero_t = action_config.reaction_miss_zero_threshold
    fast_p = action_config.reaction_miss_fast_probability
    secondary_p = action_config.reaction_miss_secondary_probability
    if t < fast_t:
        return fast_p
    if t <= secondary_t:
        if secondary_t <= fast_t:
            return secondary_p
        ratio = (t - fast_t) / (secondary_t - fast_t)
        return float(fast_p + ratio * (secondary_p - fast_p))
    if t < zero_t:
        if zero_t <= secondary_t:
            return 0.0
        ratio = (t - secondary_t) / (zero_t - secondary_t)
        return float(secondary_p * (1.0 - ratio))
    return 0.0


def vy_bounds_for_hitter(side: str, config: SimulationConfig) -> tuple[float, float]:
    if side == "left":
        return config.action.vy_min_forward, config.action.vy_max_forward
    return -config.action.vy_max_forward, -config.action.vy_min_forward


def _candidate_is_feasible(
    receiver_start: tuple[float, float],
    receiver_velocity: tuple[float, float],
    receiver: str,
    t: float,
    x_pos: float,
    y_pos: float,
    z_pos: float,
    config: SimulationConfig,
    *,
    reaction_time: float = 0.0,
) -> bool:
    ground_reach = can_arrive_by_time(
        receiver_start,
        receiver_velocity,
        (float(x_pos), float(y_pos)),
        float(t),
        config,
        reach_side=receiver,
        target_z=float(z_pos),
        reaction_time=reaction_time,
    )
    height_reach = config.player.z_min <= float(z_pos) <= config.player.z_max
    on_receiver_side = y_pos < config.court.net_y if receiver == "left" else y_pos > config.court.net_y
    return ground_reach and height_reach and on_receiver_side


def _dedupe_sorted(values: np.ndarray, *, atol: float = 1e-6) -> np.ndarray:
    if values.size == 0:
        return values
    sorted_values = np.sort(values.astype(float, copy=False))
    kept = [float(sorted_values[0])]
    for value in sorted_values[1:]:
        if not np.isclose(value, kept[-1], atol=atol, rtol=0.0):
            kept.append(float(value))
    return np.asarray(kept, dtype=float)


def _feasible_time_representatives(
    dense_times: np.ndarray,
    dense_xs: np.ndarray,
    dense_ys: np.ndarray,
    dense_zs: np.ndarray,
    state: StageState,
    config: SimulationConfig,
) -> np.ndarray:
    if dense_times.size == 0:
        return np.asarray([], dtype=float)

    receiver = opponent_side(state.current_hitter)
    receiver_start = player_position(state, receiver)
    receiver_speed = player_velocity(state, receiver)
    receiver_reaction_time = reaction_time_for_side(state, receiver)
    feasible_mask = np.asarray(
        [
            _candidate_is_feasible(
                receiver_start,
                receiver_speed,
                receiver,
                t,
                x_pos,
                y_pos,
                z_pos,
                config,
                reaction_time=receiver_reaction_time,
            )
            for t, x_pos, y_pos, z_pos in zip(dense_times, dense_xs, dense_ys, dense_zs)
        ],
        dtype=bool,
    )
    if not feasible_mask.any():
        return np.asarray([], dtype=float)

    indices = np.flatnonzero(feasible_mask)
    groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    representatives: list[float] = []
    for group in groups:
        center = group[len(group) // 2]
        representatives.extend(
            [
                float(dense_times[group[0]]),
                float(dense_times[center]),
                float(dense_times[group[-1]]),
            ]
        )
    return _dedupe_sorted(np.asarray(representatives, dtype=float))


def _select_candidate_times(
    base_times: np.ndarray,
    dense_times: np.ndarray,
    dense_xs: np.ndarray,
    dense_ys: np.ndarray,
    dense_zs: np.ndarray,
    state: StageState,
    config: SimulationConfig,
) -> np.ndarray:
    target_count = config.action.intercept_count
    if target_count <= 0 or dense_times.size == 0:
        return np.asarray([], dtype=float)

    priority_times = _feasible_time_representatives(dense_times, dense_xs, dense_ys, dense_zs, state, config)
    selected = _dedupe_sorted(np.concatenate([priority_times, base_times]))

    if selected.size < target_count:
        selected = _dedupe_sorted(np.concatenate([selected, dense_times]))

    if selected.size <= target_count:
        return selected

    if priority_times.size >= target_count:
        indices = np.linspace(0, priority_times.size - 1, target_count)
        return _dedupe_sorted(np.asarray([priority_times[int(round(index))] for index in indices], dtype=float))

    remaining_slots = target_count - priority_times.size
    selected_list = [float(value) for value in priority_times]
    selected_mask = np.asarray(
        [np.any(np.isclose(value, priority_times, atol=1e-6, rtol=0.0)) for value in selected],
        dtype=bool,
    )
    non_priority = selected[~selected_mask]
    if remaining_slots > 0 and non_priority.size > 0:
        indices = np.linspace(0, non_priority.size - 1, remaining_slots)
        selected_list.extend(float(non_priority[int(round(index))]) for index in indices)
    return _dedupe_sorted(np.asarray(selected_list, dtype=float))


def landing_time(state: StageState, action: ShotAction, config: SimulationConfig) -> float:
    if config.action.effective_trajectory_mode == "ballistic":
        return ballistic_landing_time(state.z0, action.v_z, config.action.gravity)
    return simulate_trajectory(state.x0, state.y0, state.z0, action.v_x, action.v_y, action.v_z, config).landing_time


def landing_position(state: StageState, action: ShotAction, config: SimulationConfig) -> tuple[float, float]:
    if config.action.effective_trajectory_mode == "ballistic":
        return ballistic_landing_point(
            state.x0,
            state.y0,
            state.z0,
            action.v_x,
            action.v_y,
            action.v_z,
            config.action.gravity,
        )
    result = simulate_trajectory(state.x0, state.y0, state.z0, action.v_x, action.v_y, action.v_z, config)
    return result.landing_x, result.landing_y


def effective_flight_time(state: StageState, action: ShotAction, config: SimulationConfig) -> float:
    return landing_time(state, action, config)


def trajectory_result(state: StageState, action: ShotAction, config: SimulationConfig) -> TrajectoryResult:
    return simulate_trajectory(state.x0, state.y0, state.z0, action.v_x, action.v_y, action.v_z, config)


def trajectory_samples(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    result = trajectory_result(state, action, config)
    ts = np.asarray([point.t for point in result.samples], dtype=float)
    xs = np.asarray([point.x for point in result.samples], dtype=float)
    ys = np.asarray([point.y for point in result.samples], dtype=float)
    zs = np.asarray([point.z for point in result.samples], dtype=float)
    return ts, xs, ys, zs


def ballistic_net_crossing_valid(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
) -> bool:
    net_crossing = ballistic_net_crossing(
        state.x0,
        state.y0,
        state.z0,
        action.v_x,
        action.v_y,
        action.v_z,
        config.court.net_y,
        g=config.action.gravity,
    )
    if net_crossing is None:
        return False
    total_time = ballistic_landing_time(state.z0, action.v_z, config.action.gravity)
    if not (0.0 < net_crossing.t < total_time):
        return False
    required = config.court.net_height + config.action.net_clearance_margin
    return net_crossing.z >= required


def _crosses_net_from_hitter_side(state: StageState, action: ShotAction, config: SimulationConfig) -> bool:
    if state.current_hitter == "left":
        return state.y0 < config.court.net_y and action.v_y > 0.0
    return state.y0 > config.court.net_y and action.v_y < 0.0


def _lands_on_opponent_side(landing_y: float, receiver: str, config: SimulationConfig) -> bool:
    if receiver == "left":
        return landing_y < config.court.net_y
    return landing_y > config.court.net_y


def _lands_in_bounds(landing_x: float, landing_y: float, receiver: str, config: SimulationConfig) -> bool:
    x_bounds, y_bounds = target_bounds_for_receiver(receiver, config)
    return x_bounds[0] <= landing_x <= x_bounds[1] and y_bounds[0] <= landing_y <= y_bounds[1]


def valid_hitter_action(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
    *,
    result: TrajectoryResult | None = None,
) -> bool:
    receiver = opponent_side(state.current_hitter)
    if not np.all(np.isfinite([action.v_x, action.v_y, action.v_z, action.x_rec, action.y_rec])):
        return False
    if not _crosses_net_from_hitter_side(state, action, config):
        return False

    if config.action.effective_trajectory_mode == "ballistic":
        if not ballistic_net_crossing_valid(state, action, config):
            return False
        x_land, y_land = ballistic_landing_point(
            state.x0,
            state.y0,
            state.z0,
            action.v_x,
            action.v_y,
            action.v_z,
            config.action.gravity,
        )
    else:
        active = result or trajectory_result(state, action, config)
        if active.net_crossing is None:
            return False
        required = config.court.net_height + config.action.net_clearance_margin
        if not (0.0 < active.net_crossing.t < active.landing_time):
            return False
        if active.net_crossing.z < required:
            return False
        x_land = active.landing_x
        y_land = active.landing_y

    return _lands_on_opponent_side(y_land, receiver, config) and _lands_in_bounds(x_land, y_land, receiver, config)


def validate_and_clip_shot_action(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
) -> ValidatedShotAction:
    validated, _ = validate_and_clip_shot_action_with_result(state, action, config)
    return validated


def validate_and_clip_shot_action_with_result(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
) -> tuple[ValidatedShotAction, TrajectoryResult]:
    (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, config)
    vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, config)
    vx_low = config.action.vx_min if config.court.lateral_motion_enabled else 0.0
    vx_high = config.action.vx_max if config.court.lateral_motion_enabled else 0.0

    speed_cap = float(config.action.vy_max_forward)
    velocity = np.asarray([action.v_x, action.v_y, action.v_z], dtype=float)
    speed = float(np.linalg.norm(velocity))
    if np.isfinite(speed) and speed > speed_cap > 0.0:
        velocity = velocity * (speed_cap / speed)

    v_x = float(np.clip(velocity[0], vx_low, vx_high))
    v_y = float(np.clip(velocity[1], vy_low, vy_high))
    v_z = float(np.clip(velocity[2], config.action.vz_min, config.action.vz_max))
    clipped_speed = float(np.sqrt(v_x * v_x + v_y * v_y + v_z * v_z))
    if np.isfinite(clipped_speed) and clipped_speed > speed_cap > 0.0:
        scale = speed_cap / clipped_speed
        v_x *= scale
        v_y *= scale
        v_z *= scale

    applied = ShotAction(
        v_x=v_x,
        v_y=v_y,
        v_z=v_z,
        x_rec=float(np.clip(action.x_rec, rec_x_low, rec_x_high)),
        y_rec=float(np.clip(action.y_rec, rec_y_low, rec_y_high)),
    )
    projected = not (
        np.isclose(applied.v_x, action.v_x)
        and np.isclose(applied.v_y, action.v_y)
        and np.isclose(applied.v_z, action.v_z)
        and np.isclose(applied.x_rec, action.x_rec)
        and np.isclose(applied.y_rec, action.y_rec)
    )

    active = trajectory_result(state, applied, config)
    if not valid_hitter_action(state, applied, config, result=active):
        raise ValueError("Shot action is not physically valid for the current stage.")

    return ValidatedShotAction(requested=action, applied=applied, projected=projected), active


def _ballistic_candidate_points(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_time = ballistic_landing_time(state.z0, action.v_z, config.action.gravity)
    net_crossing = ballistic_net_crossing(
        state.x0,
        state.y0,
        state.z0,
        action.v_x,
        action.v_y,
        action.v_z,
        config.court.net_y,
        g=config.action.gravity,
    )
    lower_bound = None if net_crossing is None else float(net_crossing.t + 1e-6)
    base_times = config.candidate_times(total_time, lower_bound=lower_bound)
    dense_count = max(config.action.intercept_count * 20, 200)
    dense_lower = config.action.intercept_time_min if lower_bound is None else lower_bound
    dense_upper = total_time - config.action.intercept_margin_before_landing
    if dense_upper <= dense_lower:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    dense_times = np.linspace(
        dense_lower,
        dense_upper,
        dense_count,
    )
    dense_xs: list[float] = []
    dense_ys: list[float] = []
    dense_zs: list[float] = []
    for t in dense_times:
        x, y, z = ballistic_position(
            state.x0,
            state.y0,
            state.z0,
            action.v_x,
            action.v_y,
            action.v_z,
            float(t),
            config.action.gravity,
        )
        dense_xs.append(float(x))
        dense_ys.append(float(y))
        dense_zs.append(float(z))
    dense_xs_array = np.asarray(dense_xs, dtype=float)
    dense_ys_array = np.asarray(dense_ys, dtype=float)
    dense_zs_array = np.asarray(dense_zs, dtype=float)
    times = _select_candidate_times(base_times, dense_times, dense_xs_array, dense_ys_array, dense_zs_array, state, config)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for t in times:
        x, y, z = ballistic_position(
            state.x0,
            state.y0,
            state.z0,
            action.v_x,
            action.v_y,
            action.v_z,
            float(t),
            config.action.gravity,
        )
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
    return times, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)


def _sample_drag_candidates(
    state: StageState,
    result: TrajectoryResult,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(result.samples) <= 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    lower_bound = (
        config.action.intercept_time_min
        if result.net_crossing is None
        else float(result.net_crossing.t + 1e-6)
    )
    upper_bound = result.landing_time - config.action.intercept_margin_before_landing
    points = [
        point
        for point in result.samples[1:-1]
        if lower_bound <= point.t <= upper_bound
    ]
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    dense_times = np.asarray([point.t for point in points], dtype=float)
    dense_xs = np.asarray([point.x for point in points], dtype=float)
    dense_ys = np.asarray([point.y for point in points], dtype=float)
    dense_zs = np.asarray([point.z for point in points], dtype=float)
    if dense_times.size <= config.action.intercept_count:
        return dense_times, dense_xs, dense_ys, dense_zs

    base_indices = np.linspace(0, dense_times.size - 1, config.action.intercept_count)
    base_times = np.asarray([dense_times[int(round(index))] for index in base_indices], dtype=float)
    times = _select_candidate_times(
        base_times,
        dense_times,
        dense_xs,
        dense_ys,
        dense_zs,
        state,
        config,
    )
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for t in times:
        point_index = int(np.argmin(np.abs(dense_times - t)))
        xs.append(float(dense_xs[point_index]))
        ys.append(float(dense_ys[point_index]))
        zs.append(float(dense_zs[point_index]))
    return times, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)


def candidate_intercept_points(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
    *,
    result: TrajectoryResult | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if config.action.effective_trajectory_mode == "ballistic":
        return _ballistic_candidate_points(state, action, config)
    active = result or trajectory_result(state, action, config)
    return _sample_drag_candidates(state, active, config)


def feasible_intercept_indices(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
    *,
    result: TrajectoryResult | None = None,
    candidates: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[int]:
    receiver = opponent_side(state.current_hitter)
    receiver_start = player_position(state, receiver)
    receiver_speed = player_velocity(state, receiver)
    receiver_reaction_time = reaction_time_for_side(state, receiver)
    if candidates is None:
        candidates = candidate_intercept_points(state, action, config, result=result)
    times, xs, ys, zs = candidates

    feasible: list[int] = []
    for index, (t, x_pos, y_pos, z_pos) in enumerate(zip(times, xs, ys, zs)):
        if _candidate_is_feasible(
            receiver_start,
            receiver_speed,
            receiver,
            t,
            x_pos,
            y_pos,
            z_pos,
            config,
            reaction_time=receiver_reaction_time,
        ):
            feasible.append(index)
    return feasible


def prepare_shot(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
) -> PreparedShot:
    validated, active = validate_and_clip_shot_action_with_result(state, action, config)
    candidates = candidate_intercept_points(
        state,
        validated.applied,
        config,
        result=active,
    )
    feasible_indices = feasible_intercept_indices(
        state,
        validated.applied,
        config,
        candidates=candidates,
    )
    candidate_times, candidate_xs, candidate_ys, candidate_zs = candidates
    return PreparedShot(
        validated_action=validated,
        trajectory=active,
        candidate_times=candidate_times,
        candidate_xs=candidate_xs,
        candidate_ys=candidate_ys,
        candidate_zs=candidate_zs,
        feasible_indices=feasible_indices,
    )


def sample_trajectory(
    state: StageState,
    action: ShotAction,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return trajectory_samples(state, action, config)


def _terminal_rewards(winner: str | None) -> tuple[float, float]:
    if winner == "left":
        return 1.0, -1.0
    if winner == "right":
        return -1.0, 1.0
    return 0.0, 0.0


def _is_legal_service_landing(
    state: StageState,
    landing_x: float,
    landing_y: float,
    receiver: Side,
    config: SimulationConfig,
) -> bool:
    x_bounds, y_bounds = service_target_bounds_for_receiver_state(state, receiver, config)
    return x_bounds[0] <= landing_x <= x_bounds[1] and y_bounds[0] <= landing_y <= y_bounds[1]


def step_stage(
    state: StageState,
    action: ShotAction,
    intercept_index: int | None,
    config: SimulationConfig,
    *,
    enable_reaction_miss: bool = True,
    prepared_shot: PreparedShot | None = None,
) -> StageRecord:
    if state.rally_done:
        raise ValueError("Cannot step a finished rally.")

    prepared = prepared_shot or prepare_shot(state, action, config)
    validated = prepared.validated_action
    receiver_side = opponent_side(state.current_hitter)
    active = prepared.trajectory
    candidate_times = prepared.candidate_times
    candidate_xs = prepared.candidate_xs
    candidate_ys = prepared.candidate_ys
    candidate_zs = prepared.candidate_zs
    feasible_indices = list(prepared.feasible_indices)

    chosen_time = None
    intercept_point = None
    notes: list[str] = []
    if validated.projected:
        notes.append(
            "Action projected into valid range "
            f"(vx={validated.applied.v_x:.2f}, vy={validated.applied.v_y:.2f}, "
            f"vz={validated.applied.v_z:.2f}, x_rec={validated.applied.x_rec:.2f}, y_rec={validated.applied.y_rec:.2f})"
        )

    if state.stage_index == 0 and not _is_legal_service_landing(state, active.landing_x, active.landing_y, receiver_side, config):
        notes.append("Serve landed outside the legal cross-court service box.")
        next_state = replace(
            state,
            rally_done=True,
            winner=receiver_side,
            stage_index=state.stage_index + 1,
        )
        reward_left, reward_right = _terminal_rewards(next_state.winner)
        return StageRecord(
            stage_index=state.stage_index,
            state_before=state,
            validated_action=validated,
            receiver_side=receiver_side,
            candidate_times=candidate_times,
            feasible_indices=[],
            chosen_index=intercept_index,
            chosen_time=None,
            intercept_point=None,
            next_state=next_state,
            reward_left=reward_left,
            reward_right=reward_right,
            intended_intercept_time=None,
            intended_intercept_point=None,
            terminal_reason="invalid_serve_target",
            notes=notes,
        )

    if not feasible_indices:
        next_state = replace(
            state,
            rally_done=True,
            winner=state.current_hitter,
            stage_index=state.stage_index + 1,
        )
        reward_left, reward_right = _terminal_rewards(next_state.winner)
        return StageRecord(
            stage_index=state.stage_index,
            state_before=state,
            validated_action=validated,
            receiver_side=receiver_side,
            candidate_times=candidate_times,
            feasible_indices=feasible_indices,
            chosen_index=intercept_index,
            chosen_time=None,
            intercept_point=None,
            next_state=next_state,
            reward_left=reward_left,
            reward_right=reward_right,
            intended_intercept_time=None,
            intended_intercept_point=None,
            terminal_reason="no_feasible_intercept",
            notes=notes,
        )

    if intercept_index is not None and 0 <= intercept_index < len(candidate_times):
        chosen_time = float(candidate_times[intercept_index])
        intercept_point = (
            float(candidate_xs[intercept_index]),
            float(candidate_ys[intercept_index]),
            float(candidate_zs[intercept_index]),
        )

    if intercept_index not in feasible_indices:
        next_state = replace(
            state,
            rally_done=True,
            winner=state.current_hitter,
            stage_index=state.stage_index + 1,
        )
        reward_left, reward_right = _terminal_rewards(next_state.winner)
        return StageRecord(
            stage_index=state.stage_index,
            state_before=state,
            validated_action=validated,
            receiver_side=receiver_side,
            candidate_times=candidate_times,
            feasible_indices=feasible_indices,
            chosen_index=intercept_index,
            chosen_time=chosen_time,
            intercept_point=intercept_point,
            next_state=next_state,
            reward_left=reward_left,
            reward_right=reward_right,
            intended_intercept_time=chosen_time,
            intended_intercept_point=intercept_point,
            terminal_reason="invalid_intercept_choice",
            notes=notes,
        )

    assert intercept_index is not None
    t_int = float(candidate_times[intercept_index])
    x_int = float(candidate_xs[intercept_index])
    y_int = float(candidate_ys[intercept_index])
    z_int = float(candidate_zs[intercept_index])
    miss_probability = reaction_miss_probability(t_int, config)
    if enable_reaction_miss and miss_probability > 0.0 and float(np.random.random()) < miss_probability:
        notes.append(
            "Receiver missed a fast shuttle "
            f"(flight={t_int:.2f}s, miss_probability={miss_probability:.2f})."
        )
        next_state = replace(
            state,
            rally_done=True,
            winner=state.current_hitter,
            stage_index=state.stage_index + 1,
        )
        reward_left, reward_right = _terminal_rewards(next_state.winner)
        return StageRecord(
            stage_index=state.stage_index,
            state_before=state,
            validated_action=validated,
            receiver_side=receiver_side,
            candidate_times=candidate_times,
            feasible_indices=feasible_indices,
            chosen_index=intercept_index,
            chosen_time=None,
            intercept_point=None,
            next_state=next_state,
            reward_left=reward_left,
            reward_right=reward_right,
            intended_intercept_time=t_int,
            intended_intercept_point=(x_int, y_int, z_int),
            terminal_reason="reaction_miss",
            notes=notes,
        )

    hitter_start = player_position(state, state.current_hitter)
    hitter_speed = player_velocity(state, state.current_hitter)
    receiver_start = player_position(state, receiver_side)
    receiver_speed = player_velocity(state, receiver_side)
    receiver_reaction_time = reaction_time_for_side(state, receiver_side)
    receiver_target = intercept_body_target_after_reaction(
        receiver_start,
        receiver_speed,
        (x_int, y_int),
        receiver_side,
        config,
        target_z=z_int,
        reaction_time=receiver_reaction_time,
    )
    hitter_motion = advance_player_toward(
        hitter_start,
        hitter_speed,
        (validated.applied.x_rec, validated.applied.y_rec),
        t_int,
        config,
        stop_when_early=True,
    )
    receiver_motion = advance_player_toward(
        receiver_start,
        receiver_speed,
        receiver_target,
        t_int,
        config,
        reaction_time=receiver_reaction_time,
        stop_when_early=True,
    )
    hitter_end = hitter_motion.position
    hitter_end_speed = hitter_motion.velocity
    receiver_end_speed = receiver_motion.velocity

    receiver_end = receiver_motion.position

    if receiver_side == "left":
        next_x_left, next_y_left = receiver_end
        next_x_right, next_y_right = hitter_end
        next_v_x_left, next_v_y_left = receiver_end_speed
        next_v_x_right, next_v_y_right = hitter_end_speed
    else:
        next_x_left, next_y_left = hitter_end
        next_x_right, next_y_right = receiver_end
        next_v_x_left, next_v_y_left = hitter_end_speed
        next_v_x_right, next_v_y_right = receiver_end_speed

    next_state = StageState(
        x_left=next_x_left,
        y_left=next_y_left,
        x_right=next_x_right,
        y_right=next_y_right,
        current_hitter=receiver_side,
        x0=x_int,
        y0=y_int,
        z0=z_int,
        v_x_left=next_v_x_left,
        v_y_left=next_v_y_left,
        v_x_right=next_v_x_right,
        v_y_right=next_v_y_right,
        reaction_time_left=state.reaction_time_left,
        reaction_time_right=state.reaction_time_right,
        rally_done=False,
        winner=None,
        stage_index=state.stage_index + 1,
    )
    return StageRecord(
        stage_index=state.stage_index,
        state_before=state,
        validated_action=validated,
        receiver_side=receiver_side,
        candidate_times=candidate_times,
        feasible_indices=feasible_indices,
        chosen_index=intercept_index,
        chosen_time=t_int,
        intercept_point=(x_int, y_int, z_int),
        next_state=next_state,
        reward_left=0.0,
        reward_right=0.0,
        intended_intercept_time=t_int,
        intended_intercept_point=(x_int, y_int, z_int),
        terminal_reason=None,
        notes=notes,
    )
