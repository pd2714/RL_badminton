from __future__ import annotations

from typing import Literal

from badminton.config import SimulationConfig
from badminton.state import Side
from badminton.shot_generators.tactic_lookup_common import ANGLE_BIN_NAMES, POWER_BIN_NAMES
from badminton.utils import opponent_side, side_y_bounds


CourtDepth = Literal["front", "middle", "back"]
FRONT_COURT_DEPTH_M = 3.0


def _contact_is_front(contact_y: float, config: SimulationConfig) -> bool:
    low, high = side_y_bounds("left", config)
    threshold = low + 0.72 * (high - low)
    return float(contact_y) >= float(threshold)


def _court_depth_third(y: float, side: Side, config: SimulationConfig) -> CourtDepth:
    low, high = side_y_bounds(side, config)
    max_depth_from_net = max(abs(float(high) - config.court.net_y), abs(config.court.net_y - float(low)))
    if side == "left":
        depth_from_net = float(config.court.net_y) - float(y)
    else:
        depth_from_net = float(y) - float(config.court.net_y)
    if depth_from_net <= FRONT_COURT_DEPTH_M:
        return "front"
    middle_back_threshold = FRONT_COURT_DEPTH_M + 0.5 * max(max_depth_from_net - FRONT_COURT_DEPTH_M, 0.0)
    if depth_from_net <= middle_back_threshold:
        return "middle"
    return "back"


def name_velocity_shot(
    *,
    hitter: Side,
    contact_x: float,
    contact_y: float,
    landing_x: float,
    landing_y: float,
    theta_degrees: float,
    config: SimulationConfig,
    cross_court_min_delta_x: float = 2.4,
) -> str:
    """Name a velocity-oriented shot from contact, landing, and launch angle."""
    receiver = opponent_side(hitter)
    landing_depth = _court_depth_third(landing_y, receiver, config)
    contact_depth = _court_depth_third(contact_y, hitter, config)
    theta = float(theta_degrees)

    if landing_depth == "front":
        base_name = "drop"
    elif contact_depth in {"middle", "back"}:
        if theta > 15.0:
            base_name = "clear"
        elif theta < 0.0:
            base_name = "smash"
        else:
            base_name = "drive"
    elif theta > 45.0:
        base_name = "lift"
    elif theta < 0.0:
        base_name = "net kill"
    else:
        base_name = "push"

    crosses_lateral_halves = float(contact_x) * float(landing_x) < 0.0
    wide_enough = abs(float(landing_x) - float(contact_x)) > float(cross_court_min_delta_x)
    if crosses_lateral_halves and wide_enough:
        return f"cross-court {base_name}"
    return base_name


def infer_shot_name(
    *,
    contact_y: float,
    landing_row: int,
    angle_bin: int,
    power_bin: int,
    config: SimulationConfig,
    angle_names: tuple[str, ...] = ANGLE_BIN_NAMES,
    power_names: tuple[str, ...] = POWER_BIN_NAMES,
    landing_row_count: int = 3,
    angle_degrees: float | None = None,
) -> str:
    angle_name = angle_names[int(angle_bin)] if int(angle_bin) < len(angle_names) else "generic"
    power_name = power_names[int(power_bin)] if int(power_bin) < len(power_names) else "normal"
    contact_is_front = _contact_is_front(contact_y, config)
    row_count = max(int(landing_row_count), 1)
    row_ratio = 0.5 if row_count == 1 else float(landing_row) / float(row_count - 1)
    is_front_target = row_ratio <= 0.25
    is_back_target = row_ratio >= 0.75
    is_mid_target = not is_front_target and not is_back_target
    if angle_degrees is None:
        downward_angle = angle_name in {"very_down", "down", "steep_down", "very_steep_down"}
        upward_angle = angle_name in {"up", "high_up", "very_high_up"}
        flat_angle = angle_name == "flat"
    else:
        downward_angle = angle_degrees <= -8.0
        upward_angle = angle_degrees >= 18.0
        flat_angle = -8.0 < angle_degrees < 18.0

    if downward_angle and (is_mid_target or is_back_target):
        return "smash"
    if is_front_target and (downward_angle or (flat_angle and power_name == "soft")):
        return "drop"
    if upward_angle and is_back_target:
        return "clear"
    if contact_is_front and is_back_target and upward_angle:
        return "lift"
    if flat_angle and is_mid_target:
        return "drive"
    if is_front_target and power_name == "soft":
        return "net"
    return "generic"
