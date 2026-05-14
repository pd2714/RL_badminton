from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.state import StageState
from badminton1d.utils import recovery_bounds, side_y_bounds, x_bounds

ANGLE_BIN_NAMES = ("very_down", "down", "flat", "up", "high_up")
ANGLE_BIN_NAMES_2D = ANGLE_BIN_NAMES
ANGLE_BIN_COUNT_1D = 15
ANGLE_BIN_NAMES_1D = tuple(f"angle_bin_{index:02d}" for index in range(ANGLE_BIN_COUNT_1D))
POWER_BIN_NAMES = ("soft", "normal", "hard")
POWER_BIN_NAMES_1D = ("very_soft", "soft", "normal", "hard", "very_hard", "max")
SHOT_NAME_ORDER = ("smash", "drop", "clear", "lift", "drive", "net", "generic")
TACTIC_POLICY_TYPES = (
    "velocity_oriented",
    "tactic_oriented",
    "conditional_prob",
    "continuous_action",
    "mixed_discrete_continous",
)
LANDING_ZONE_COUNT_1D = 5


@dataclass(frozen=True)
class TacticAction1D:
    landing_zone: int
    angle_bin: int
    power_bin: int


@dataclass(frozen=True)
class TacticAction2D:
    landing_row: int
    landing_col: int
    angle_bin: int
    power_bin: int

    @property
    def landing_zone(self) -> int:
        return self.landing_row * 3 + self.landing_col


@dataclass(frozen=True)
class TacticRuntimeConfig:
    regenerate_lookup_table: bool = False
    lookup_dir: Path = field(default_factory=lambda: Path("lookup_tables"))
    fallback_penalty: float = 0.0


@dataclass(frozen=True)
class CanonicalState:
    x0: float
    y0: float
    z0: float
    mirrored_longitudinal: bool


@dataclass(frozen=True)
class LookupQueryResult:
    velocity: tuple[float, ...]
    contact_bins: tuple[int, ...]
    valid: bool
    fallback_used: bool
    landing_position: tuple[float, float]
    net_crossing_height: float | None
    flight_time: float
    score: float
    inferred_shot_name: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_lookup_dir(runtime_config: TacticRuntimeConfig | None = None) -> Path:
    config = runtime_config or TacticRuntimeConfig()
    base = Path(config.lookup_dir)
    if base.is_absolute():
        return base
    return repo_root() / base


def canonicalize_state(state: StageState) -> CanonicalState:
    mirrored = state.current_hitter == "right"
    if mirrored:
        return CanonicalState(
            x0=float(state.x0),
            y0=float(-state.y0),
            z0=float(state.z0),
            mirrored_longitudinal=True,
        )
    return CanonicalState(
        x0=float(state.x0),
        y0=float(state.y0),
        z0=float(state.z0),
        mirrored_longitudinal=False,
    )


def restore_velocity_from_canonical(
    velocity: tuple[float, ...],
    *,
    mirrored_longitudinal: bool,
) -> tuple[float, ...]:
    if not mirrored_longitudinal:
        return velocity
    if len(velocity) == 2:
        return (velocity[0], -velocity[1])
    return (velocity[0], -velocity[1], velocity[2])


def contact_x_centers(config: SimulationConfig, bins: int = 5) -> np.ndarray:
    low, high = x_bounds(config)
    return np.linspace(low, high, bins, dtype=float)


def contact_y_centers(config: SimulationConfig, bins: int = 5) -> np.ndarray:
    low, high = side_y_bounds("left", config)
    return np.linspace(low, high, bins, dtype=float)


def contact_y_centers_1d(
    config: SimulationConfig,
    bins: int = 5,
    *,
    min_net_distance: float = 0.5,
) -> np.ndarray:
    low, high = side_y_bounds("left", config)
    closest_contact_y = float(config.court.net_y - min_net_distance)
    high = min(high, closest_contact_y)
    return np.linspace(low, high, bins, dtype=float)


def contact_height_centers(config: SimulationConfig, bins: int = 5) -> np.ndarray:
    return np.linspace(float(config.player.z_min), float(config.player.z_max), bins, dtype=float)


def landing_row_centers(config: SimulationConfig, bins: int = 3) -> np.ndarray:
    low, high = side_y_bounds("right", config)
    edges = np.linspace(low, high, bins + 1, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def landing_col_centers(config: SimulationConfig, bins: int = 3) -> np.ndarray:
    low, high = x_bounds(config)
    edges = np.linspace(low, high, bins + 1, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def landing_zone_names_2d() -> tuple[str, ...]:
    rows = ("front", "mid", "back")
    cols = ("left", "center", "right")
    return tuple(f"{row}_{col}" for row in rows for col in cols)


def landing_zone_names_1d() -> tuple[str, ...]:
    return ("front", "front_mid", "mid", "back_mid", "back")


def nearest_bin(value: float, centers: np.ndarray) -> int:
    return int(np.argmin(np.abs(centers - float(value))))


def default_recovery_target(
    state: StageState,
    *,
    landing_row: int,
    landing_col: int | None,
    config: SimulationConfig,
    landing_row_count: int = 3,
) -> tuple[float, float]:
    (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, config)
    span_y = rec_y_high - rec_y_low
    row_count = max(int(landing_row_count), 1)
    row_ratio = (float(landing_row) + 0.5) / float(row_count)
    if state.current_hitter == "left":
        y_target = rec_y_low + (0.35 + 0.4 * row_ratio) * span_y
    else:
        y_target = rec_y_high - (0.35 + 0.4 * row_ratio) * span_y

    if landing_col is None or not config.court.lateral_motion_enabled:
        x_target = 0.5 * (rec_x_low + rec_x_high)
    else:
        col_ratio = (float(landing_col) + 0.5) / 3.0
        x_target = rec_x_low + col_ratio * (rec_x_high - rec_x_low)
    return float(np.clip(x_target, rec_x_low, rec_x_high)), float(np.clip(y_target, rec_y_low, rec_y_high))


def angle_bin_centers_deg() -> np.ndarray:
    return np.asarray([-28.0, -10.0, 8.0, 28.0, 48.0], dtype=float)


def angle_bin_centers_deg_1d(
    contact_y: float,
    contact_z: float,
    config: SimulationConfig,
    *,
    bins: int = ANGLE_BIN_COUNT_1D,
    upper_limit_deg: float = 80.0,
) -> np.ndarray:
    horizontal_distance = max(float(abs(config.court.net_y - contact_y)), 1e-6)
    lower_limit_deg = float(np.degrees(np.arctan2(config.court.net_height - contact_z, horizontal_distance)))
    lower_limit_deg = min(lower_limit_deg, upper_limit_deg)
    if lower_limit_deg >= upper_limit_deg:
        return np.full(bins, lower_limit_deg, dtype=float)
    return np.linspace(lower_limit_deg, upper_limit_deg, bins, dtype=float)


def power_speed_targets_2d() -> np.ndarray:
    return np.asarray([8.0, 13.0, 18.0], dtype=float)


def power_speed_targets_1d() -> np.ndarray:
    return np.linspace(4.0, 22.0, len(POWER_BIN_NAMES_1D), dtype=float)


def validate_policy_type(
    policy_type: str,
) -> Literal["velocity_oriented", "tactic_oriented", "conditional_prob", "continuous_action", "mixed_discrete_continous"]:
    normalized = policy_type.strip().lower()
    if normalized not in TACTIC_POLICY_TYPES:
        raise ValueError(f"Unsupported policy_type: {policy_type}")
    return normalized  # type: ignore[return-value]
