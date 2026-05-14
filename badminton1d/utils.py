from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.state import ShotAction, Side, StageState


def opponent_side(side: Side) -> Side:
    return "right" if side == "left" else "left"


def side_to_agent_canonical(side: Side, agent_side: Side) -> Side:
    return "left" if side == agent_side else "right"


def canonicalize_state_for_agent(state: StageState, agent_side: Side) -> StageState:
    if agent_side == "left":
        return state
    winner = None if state.winner is None else side_to_agent_canonical(state.winner, agent_side)
    return StageState(
        x_left=state.x_right,
        y_left=-state.y_right,
        x_right=state.x_left,
        y_right=-state.y_left,
        current_hitter=side_to_agent_canonical(state.current_hitter, agent_side),
        x0=state.x0,
        y0=-state.y0,
        z0=state.z0,
        v_x_left=state.v_x_right,
        v_y_left=-state.v_y_right,
        v_x_right=state.v_x_left,
        v_y_right=-state.v_y_left,
        reaction_time_left=state.reaction_time_right,
        reaction_time_right=state.reaction_time_left,
        rally_done=state.rally_done,
        winner=winner,
        stage_index=state.stage_index,
    )


def canonicalize_shot_action_for_agent(action: ShotAction, agent_side: Side) -> ShotAction:
    if agent_side == "left":
        return action
    return ShotAction(
        v_x=action.v_x,
        v_y=-action.v_y,
        v_z=action.v_z,
        x_rec=action.x_rec,
        y_rec=-action.y_rec,
    )


def restore_shot_action_from_agent_canonical(action: ShotAction, agent_side: Side) -> ShotAction:
    return canonicalize_shot_action_for_agent(action, agent_side)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def player_position(state: StageState, side: Side) -> tuple[float, float]:
    if side == "left":
        return float(state.x_left), float(state.y_left)
    return float(state.x_right), float(state.y_right)


def player_velocity(state: StageState, side: Side) -> tuple[float, float]:
    if side == "left":
        return float(state.v_x_left), float(state.v_y_left)
    return float(state.v_x_right), float(state.v_y_right)


def shuttle_position(state: StageState) -> tuple[float, float, float]:
    return float(state.x0), float(state.y0), float(state.z0)


def x_bounds(config: SimulationConfig, *, margin: float | None = None) -> tuple[float, float]:
    if not config.court.lateral_motion_enabled:
        x_center = float(config.court.default_player_start_x)
        return x_center, x_center
    pad = config.court.boundary_margin if margin is None else margin
    return (
        -config.court.half_width + pad,
        config.court.half_width - pad,
    )


def half_court_center_x(side: Side, config: SimulationConfig) -> float:
    if not config.court.lateral_motion_enabled:
        return float(config.court.default_player_start_x)
    offset = 0.5 * config.court.half_width
    return -offset if side == "left" else offset


def service_court_x(side: Side, config: SimulationConfig) -> float:
    if not config.court.lateral_motion_enabled:
        return float(config.court.default_player_start_x)
    offset = float(config.court.service_x_offset_from_center_line)
    return -offset if side == "left" else offset


def side_x_bounds(
    side: Side,
    config: SimulationConfig,
    *,
    margin: float | None = None,
) -> tuple[float, float]:
    if not config.court.lateral_motion_enabled:
        x_center = float(config.court.default_player_start_x)
        return x_center, x_center
    pad = config.court.boundary_margin if margin is None else margin
    if side == "left":
        return -config.court.half_width + pad, 0.0
    return 0.0, config.court.half_width - pad


def side_y_bounds(
    side: Side,
    config: SimulationConfig,
    *,
    net_margin: float | None = None,
    back_margin: float | None = None,
) -> tuple[float, float]:
    inner = config.court.boundary_margin if net_margin is None else net_margin
    outer = config.court.boundary_margin if back_margin is None else back_margin
    if side == "left":
        return (
            -config.court.half_length + outer,
            config.court.net_y - inner,
        )
    return (
        config.court.net_y + inner,
        config.court.half_length - outer,
    )


def target_bounds_for_receiver(
    side: Side,
    config: SimulationConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return x_bounds(config), side_y_bounds(side, config)


def service_target_bounds_for_receiver(
    side: Side,
    config: SimulationConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    line_distance = config.court.service_line_distance_from_net
    if side == "left":
        y_bounds = (-config.court.half_length + config.court.boundary_margin, config.court.net_y - line_distance)
    else:
        y_bounds = (config.court.net_y + line_distance, config.court.half_length - config.court.boundary_margin)
    return side_x_bounds(side, config), y_bounds


def service_target_bounds_for_receiver_state(
    state: StageState,
    receiver: Side,
    config: SimulationConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    _, y_range = service_target_bounds_for_receiver(receiver, config)
    if not config.court.lateral_motion_enabled:
        return x_bounds(config), y_range
    receiver_x, _ = player_position(state, receiver)
    if receiver_x < 0.0:
        return side_x_bounds("left", config), y_range
    if receiver_x > 0.0:
        return side_x_bounds("right", config), y_range
    return x_bounds(config), y_range


def recovery_bounds(
    side: Side,
    config: SimulationConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        x_bounds(config, margin=config.action.recovery_x_margin),
        side_y_bounds(
            side,
            config,
            net_margin=config.action.recovery_net_margin,
            back_margin=config.action.recovery_back_margin,
        ),
    )


def side_center_y(side: Side, config: SimulationConfig) -> float:
    low, high = side_y_bounds(side, config)
    return 0.5 * (low + high)


def default_player_position(side: Side, config: SimulationConfig) -> tuple[float, float]:
    if side == "left":
        return (
            half_court_center_x("left", config),
            config.court.net_y - config.court.default_player_start_distance_from_net,
        )
    return (
        half_court_center_x("right", config),
        config.court.net_y + config.court.default_player_start_distance_from_net,
    )


def move_toward(
    start: float | Iterable[float],
    target: float | Iterable[float],
    max_distance: float,
) -> float | tuple[float, ...]:
    if isinstance(start, (int, float)) and isinstance(target, (int, float)):
        delta = float(target) - float(start)
        if abs(delta) <= max_distance:
            return float(target)
        step = max_distance if delta >= 0.0 else -max_distance
        return float(start) + step

    start_vec = np.asarray(tuple(start), dtype=float)
    target_vec = np.asarray(tuple(target), dtype=float)
    delta = target_vec - start_vec
    distance = float(np.linalg.norm(delta))
    if distance <= max_distance or np.isclose(distance, 0.0):
        return tuple(float(value) for value in target_vec)
    ratio = max_distance / distance
    moved = start_vec + ratio * delta
    return tuple(float(value) for value in moved)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
