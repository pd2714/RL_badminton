from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite

import numpy as np

from badminton.config import SimulationConfig
from badminton.dynamics import candidate_intercept_points, reaction_miss_probability, reaction_time_for_side
from badminton.movement import closest_intercept_body_target
from badminton.playback import MatchTrace, StageTrace
from badminton.state import ShotAction, Side, StageRecord, StageState
from badminton.utils import player_position, player_velocity


@dataclass(frozen=True)
class ShotPressureWeights:
    required_speed: float = 0.35
    intercept_scarcity: float = 0.30
    low_contact: float = 0.15
    reaction_miss: float = 0.20


@dataclass(frozen=True)
class ShotPressureIndex:
    pressure: float
    required_speed_score: float
    intercept_scarcity_score: float
    low_contact_score: float
    reaction_miss_score: float
    required_speed: float
    required_speed_ratio: float
    feasible_intercept_count: int
    candidate_intercept_count: int
    best_contact_height: float | None
    chosen_contact_height: float | None
    chosen_required_speed: float | None
    chosen_reaction_miss_probability: float | None
    terminal_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MatchShotPressure:
    rally_index: int
    rally_number: int | None
    stage_index: int
    hitter_side: Side
    receiver_side: Side
    pressure: ShotPressureIndex

    def to_dict(self) -> dict[str, object]:
        payload = {
            "rally_index": self.rally_index,
            "rally_number": self.rally_number,
            "stage_index": self.stage_index,
            "hitter_side": self.hitter_side,
            "receiver_side": self.receiver_side,
        }
        payload.update(self.pressure.to_dict())
        return payload


def shot_pressure_from_record(
    record: StageRecord,
    config: SimulationConfig,
    *,
    weights: ShotPressureWeights | None = None,
) -> ShotPressureIndex:
    state = record.state_before
    action = record.validated_action.applied
    times, xs, ys, zs = candidate_intercept_points(state, action, config)
    receiver_start = player_position(state, record.receiver_side)
    receiver_velocity = player_velocity(state, record.receiver_side)
    receiver_reaction_time = reaction_time_for_side(state, record.receiver_side)
    return shot_pressure_from_candidates(
        receiver_side=record.receiver_side,
        receiver_start=receiver_start,
        receiver_velocity=receiver_velocity,
        receiver_reaction_time=receiver_reaction_time,
        candidate_times=times,
        candidate_xs=xs,
        candidate_ys=ys,
        candidate_zs=zs,
        feasible_indices=record.feasible_indices,
        chosen_index=record.chosen_index,
        chosen_time=record.chosen_time,
        chosen_point=record.intended_intercept_point or record.intercept_point,
        terminal_reason=record.terminal_reason,
        config=config,
        weights=weights,
    )


def shot_pressure_from_stage_trace(
    stage: StageTrace,
    config: SimulationConfig,
    *,
    receiver_reaction_time: float | None = None,
    weights: ShotPressureWeights | None = None,
) -> ShotPressureIndex:
    local_config = _config_for_stage(stage, config)
    reaction_time = (
        float(stage.receiver_reaction_time)
        if receiver_reaction_time is None
        else float(receiver_reaction_time)
    )
    state = _state_from_stage_trace(stage, reaction_time=reaction_time)
    action = ShotAction(
        v_x=float(stage.shuttle_velocity[0]),
        v_y=float(stage.shuttle_velocity[1]),
        v_z=float(stage.shuttle_velocity[2]),
        x_rec=float(stage.recovery_target[0]),
        y_rec=float(stage.recovery_target[1]),
    )
    times, xs, ys, zs = candidate_intercept_points(state, action, local_config)
    feasible_indices = _feasible_indices_from_candidates(
        receiver_side=stage.receiver_side,
        receiver_start=stage.receiver_start,
        receiver_velocity=(0.0, 0.0),
        receiver_reaction_time=reaction_time,
        candidate_times=times,
        candidate_xs=xs,
        candidate_ys=ys,
        candidate_zs=zs,
        config=local_config,
    )
    chosen_index = _nearest_time_index(times, stage.intended_intercept_time)
    return shot_pressure_from_candidates(
        receiver_side=stage.receiver_side,
        receiver_start=stage.receiver_start,
        receiver_velocity=(0.0, 0.0),
        receiver_reaction_time=reaction_time,
        candidate_times=times,
        candidate_xs=xs,
        candidate_ys=ys,
        candidate_zs=zs,
        feasible_indices=feasible_indices,
        chosen_index=chosen_index,
        chosen_time=stage.intended_intercept_time,
        chosen_point=stage.intended_intercept_point or stage.intercept_point,
        terminal_reason=stage.terminal_reason,
        config=local_config,
        weights=weights,
    )


def shot_pressure_from_candidates(
    *,
    receiver_side: Side,
    receiver_start: tuple[float, float],
    receiver_velocity: tuple[float, float],
    receiver_reaction_time: float,
    candidate_times: np.ndarray,
    candidate_xs: np.ndarray,
    candidate_ys: np.ndarray,
    candidate_zs: np.ndarray,
    feasible_indices: list[int],
    config: SimulationConfig,
    weights: ShotPressureWeights | None = None,
    chosen_index: int | None = None,
    chosen_time: float | None = None,
    chosen_point: tuple[float, float, float] | None = None,
    terminal_reason: str | None = None,
) -> ShotPressureIndex:
    active_weights = weights or ShotPressureWeights()
    candidate_count = int(len(candidate_times))
    feasible = [int(index) for index in feasible_indices if 0 <= int(index) < candidate_count]
    feasible_count = len(feasible)

    required_speeds = _candidate_required_speeds(
        receiver_start,
        receiver_velocity,
        receiver_reaction_time,
        candidate_times,
        candidate_xs,
        candidate_ys,
        candidate_zs,
        receiver_side,
        config,
    )
    best_required_speed = _best_required_speed(required_speeds)
    speed_ratio = best_required_speed / max(float(config.player.v_max), 1e-9)
    required_speed_score = float(np.clip(speed_ratio, 0.0, 1.0))

    if candidate_count <= 0:
        scarcity_score = 1.0
    else:
        scarcity_score = 1.0 - float(feasible_count / candidate_count)

    best_contact_height = _best_feasible_contact_height(candidate_zs, feasible)
    low_contact_score = _low_contact_score(best_contact_height, config)
    reaction_miss_score = _reaction_miss_score(candidate_times, feasible, config)

    chosen_height = None
    chosen_required_speed = None
    chosen_reaction_miss_probability = None
    if chosen_point is not None:
        chosen_height = float(chosen_point[2])
        time_for_choice = chosen_time
        if time_for_choice is None and chosen_index is not None and 0 <= chosen_index < candidate_count:
            time_for_choice = float(candidate_times[chosen_index])
        if time_for_choice is not None:
            chosen_reaction_miss_probability = float(reaction_miss_probability(float(time_for_choice), config))
            chosen_required_speed = _required_horizontal_speed(
                receiver_start,
                receiver_velocity,
                receiver_reaction_time,
                float(time_for_choice),
                (float(chosen_point[0]), float(chosen_point[1])),
                float(chosen_point[2]),
                receiver_side,
                config,
            )

    pressure = _weighted_score(
        required_speed_score=required_speed_score,
        intercept_scarcity_score=scarcity_score,
        low_contact_score=low_contact_score,
        reaction_miss_score=reaction_miss_score,
        weights=active_weights,
    )
    return ShotPressureIndex(
        pressure=pressure,
        required_speed_score=required_speed_score,
        intercept_scarcity_score=scarcity_score,
        low_contact_score=low_contact_score,
        reaction_miss_score=reaction_miss_score,
        required_speed=float(best_required_speed),
        required_speed_ratio=float(speed_ratio),
        feasible_intercept_count=feasible_count,
        candidate_intercept_count=candidate_count,
        best_contact_height=best_contact_height,
        chosen_contact_height=chosen_height,
        chosen_required_speed=chosen_required_speed,
        chosen_reaction_miss_probability=chosen_reaction_miss_probability,
        terminal_reason=terminal_reason,
    )


def evaluate_match_pressure(
    trace: MatchTrace,
    config: SimulationConfig,
    *,
    receiver_reaction_time: float | None = None,
    weights: ShotPressureWeights | None = None,
) -> list[MatchShotPressure]:
    rows: list[MatchShotPressure] = []
    for rally_index, rally in enumerate(trace.rallies):
        for stage in rally.stages:
            rows.append(
                MatchShotPressure(
                    rally_index=rally_index,
                    rally_number=rally.rally_number,
                    stage_index=stage.stage_index,
                    hitter_side=stage.hitter_side,
                    receiver_side=stage.receiver_side,
                    pressure=shot_pressure_from_stage_trace(
                        stage,
                        config,
                        receiver_reaction_time=receiver_reaction_time,
                        weights=weights,
                    ),
                )
            )
    return rows


def summarize_match_pressure(rows: list[MatchShotPressure]) -> dict[str, object]:
    if not rows:
        return {
            "shot_count": 0,
            "avg_pressure": 0.0,
            "max_pressure": 0.0,
            "avg_required_speed": 0.0,
            "avg_feasible_intercepts": 0.0,
        }
    pressure_values = np.asarray([row.pressure.pressure for row in rows], dtype=float)
    speed_values = np.asarray([row.pressure.required_speed for row in rows], dtype=float)
    feasible_values = np.asarray([row.pressure.feasible_intercept_count for row in rows], dtype=float)
    return {
        "shot_count": len(rows),
        "avg_pressure": float(np.mean(pressure_values)),
        "max_pressure": float(np.max(pressure_values)),
        "avg_required_speed": float(np.mean(speed_values[np.isfinite(speed_values)])) if np.isfinite(speed_values).any() else float("inf"),
        "avg_feasible_intercepts": float(np.mean(feasible_values)),
    }


def resolve_match_trace_path(path: Path) -> Path:
    if path.is_dir():
        trace_path = path / "match_trace.json"
    elif path.suffix.lower() == ".mp4":
        trace_path = path.with_name("match_trace.json")
    else:
        trace_path = path
    if not trace_path.exists():
        raise FileNotFoundError(f"Could not find match trace at {trace_path}")
    return trace_path


def _config_for_stage(stage: StageTrace, config: SimulationConfig) -> SimulationConfig:
    return SimulationConfig(
        court=config.court,
        player=config.player,
        render=config.render,
        action=replace(
            config.action,
            trajectory_mode=stage.trajectory_mode,
            gravity=stage.gravity,
            drag_coefficient=stage.drag_coefficient,
            horizontal_drag_coefficient=stage.horizontal_drag_coefficient,
            vertical_drag_coefficient=stage.vertical_drag_coefficient,
            drag_dt=stage.drag_dt,
        ),
    )


def _state_from_stage_trace(stage: StageTrace, *, reaction_time: float) -> StageState:
    left_reaction = reaction_time if stage.receiver_side == "left" else 0.0
    right_reaction = reaction_time if stage.receiver_side == "right" else 0.0
    return StageState(
        x_left=float(stage.left_start[0]),
        y_left=float(stage.left_start[1]),
        x_right=float(stage.right_start[0]),
        y_right=float(stage.right_start[1]),
        current_hitter=stage.hitter_side,
        x0=float(stage.shuttle_start[0]),
        y0=float(stage.shuttle_start[1]),
        z0=float(stage.shuttle_start[2]),
        reaction_time_left=left_reaction,
        reaction_time_right=right_reaction,
        stage_index=stage.stage_index,
    )


def _feasible_indices_from_candidates(
    *,
    receiver_side: Side,
    receiver_start: tuple[float, float],
    receiver_velocity: tuple[float, float],
    receiver_reaction_time: float,
    candidate_times: np.ndarray,
    candidate_xs: np.ndarray,
    candidate_ys: np.ndarray,
    candidate_zs: np.ndarray,
    config: SimulationConfig,
) -> list[int]:
    feasible: list[int] = []
    for index, (time_value, x_pos, y_pos, z_pos) in enumerate(zip(candidate_times, candidate_xs, candidate_ys, candidate_zs)):
        if not _candidate_has_legal_contact(receiver_side, float(y_pos), float(z_pos), config):
            continue
        required_speed = _required_horizontal_speed(
            receiver_start,
            receiver_velocity,
            receiver_reaction_time,
            float(time_value),
            (float(x_pos), float(y_pos)),
            float(z_pos),
            receiver_side,
            config,
        )
        if required_speed <= float(config.player.v_max) + 1e-9:
            feasible.append(index)
    return feasible


def _candidate_required_speeds(
    receiver_start: tuple[float, float],
    receiver_velocity: tuple[float, float],
    receiver_reaction_time: float,
    candidate_times: np.ndarray,
    candidate_xs: np.ndarray,
    candidate_ys: np.ndarray,
    candidate_zs: np.ndarray,
    receiver_side: Side,
    config: SimulationConfig,
) -> list[float]:
    legal_speeds: list[float] = []
    fallback_speeds: list[float] = []
    for time_value, x_pos, y_pos, z_pos in zip(candidate_times, candidate_xs, candidate_ys, candidate_zs):
        speed = _required_horizontal_speed(
            receiver_start,
            receiver_velocity,
            receiver_reaction_time,
            float(time_value),
            (float(x_pos), float(y_pos)),
            float(z_pos),
            receiver_side,
            config,
        )
        fallback_speeds.append(speed)
        if _candidate_has_legal_contact(receiver_side, float(y_pos), float(z_pos), config):
            legal_speeds.append(speed)
    return legal_speeds or fallback_speeds


def _required_horizontal_speed(
    start: tuple[float, float],
    velocity: tuple[float, float],
    reaction_time: float,
    target_time: float,
    target_xy: tuple[float, float],
    target_z: float,
    receiver_side: Side,
    config: SimulationConfig,
) -> float:
    del velocity
    available_time = float(target_time) - max(float(reaction_time), 0.0)
    body_target = closest_intercept_body_target(start, target_xy, receiver_side, config, target_z=target_z)
    distance = float(np.linalg.norm(np.asarray(body_target, dtype=float) - np.asarray(start, dtype=float)))
    if distance <= 1e-9:
        return 0.0
    if available_time <= 1e-9:
        return float("inf")
    return distance / available_time


def _best_required_speed(required_speeds: list[float]) -> float:
    finite = [speed for speed in required_speeds if isfinite(speed)]
    if finite:
        return min(finite)
    return float("inf")


def _best_feasible_contact_height(candidate_zs: np.ndarray, feasible_indices: list[int]) -> float | None:
    if not feasible_indices:
        return None
    return float(max(float(candidate_zs[index]) for index in feasible_indices))


def _low_contact_score(best_contact_height: float | None, config: SimulationConfig) -> float:
    if best_contact_height is None:
        return 1.0
    z_span = max(float(config.player.z_max - config.player.z_min), 1e-9)
    return float(1.0 - np.clip((best_contact_height - config.player.z_min) / z_span, 0.0, 1.0))


def _reaction_miss_score(candidate_times: np.ndarray, feasible_indices: list[int], config: SimulationConfig) -> float:
    if not feasible_indices:
        return 1.0
    probabilities = [
        float(reaction_miss_probability(float(candidate_times[index]), config))
        for index in feasible_indices
        if 0 <= int(index) < len(candidate_times)
    ]
    if not probabilities:
        return 1.0
    return float(np.clip(np.mean(probabilities), 0.0, 1.0))


def _weighted_score(
    *,
    required_speed_score: float,
    intercept_scarcity_score: float,
    low_contact_score: float,
    reaction_miss_score: float,
    weights: ShotPressureWeights,
) -> float:
    weight_values = [
        max(float(weights.required_speed), 0.0),
        max(float(weights.intercept_scarcity), 0.0),
        max(float(weights.low_contact), 0.0),
        max(float(weights.reaction_miss), 0.0),
    ]
    total = sum(weight_values)
    if total <= 0.0:
        return 0.0
    value = (
        weight_values[0] * required_speed_score
        + weight_values[1] * intercept_scarcity_score
        + weight_values[2] * low_contact_score
        + weight_values[3] * reaction_miss_score
    ) / total
    return float(np.clip(value, 0.0, 1.0))


def _candidate_has_legal_contact(receiver_side: Side, y_pos: float, z_pos: float, config: SimulationConfig) -> bool:
    on_receiver_side = y_pos < config.court.net_y if receiver_side == "left" else y_pos > config.court.net_y
    height_reach = config.player.z_min <= z_pos <= config.player.z_max
    return bool(on_receiver_side and height_reach)


def _nearest_time_index(times: np.ndarray, target_time: float | None) -> int | None:
    if target_time is None or len(times) == 0:
        return None
    return int(np.argmin(np.abs(times - float(target_time))))
