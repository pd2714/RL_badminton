from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Any

import numpy as np
from gymnasium import spaces

from badminton.config import SimulationConfig
from badminton.dynamics import (
    PreparedShot,
    landing_position,
    prepare_shot,
    validate_and_clip_shot_action_with_result,
    vy_bounds_for_hitter,
)
from badminton.trajectory import ballistic_landing_point, ballistic_landing_time, ballistic_net_crossing
from badminton.shot_generators import (
    POWER_BIN_NAMES,
    SHOT_NAME_ORDER,
    TacticAction1D,
    TacticAction2D,
    TacticLookup1D,
    TacticLookup2D,
    TacticRuntimeConfig,
)
from badminton.shot_generators.tactic_lookup_common import default_recovery_target, validate_policy_type
from badminton.state import ShotAction, Side, StageState, ValidatedShotAction
from badminton.utils import (
    canonicalize_state_for_agent,
    opponent_side,
    recovery_bounds,
    restore_shot_action_from_agent_canonical,
    service_target_bounds_for_receiver_state,
    side_y_bounds,
    target_bounds_for_receiver,
    x_bounds,
)

_LEGAL_SERVE_MASK_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
_LEGAL_HITTER_MASK_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
_VELOCITY_SPEED_RANGE_CACHE: dict[tuple[Any, ...], tuple[float, float] | None] = {}
CONTINUOUS_ACTION_DIM = 6
MIXED_DISCRETE_CONTINOUS_POLICY_TYPE = "mixed_discrete_continous"
CONDITIONAL_VELOCITY_POLICY_TYPES = {
    "velocity_oriented",
    "conditional_prob",
    "continuous_action",
    MIXED_DISCRETE_CONTINOUS_POLICY_TYPE,
}

VELOCITY_ORIENTED_THETA_HIGH_DEG = 65.0
VELOCITY_ORIENTED_SERVICE_PHI_BOUNDARY_TRIM_DEG = 5.0
VELOCITY_ORIENTED_MIDRALLY_NEAR_BOUNDARY_PHI_TRIM_DEG = 5.0
VELOCITY_ORIENTED_MIDRALLY_FAR_BOUNDARY_PHI_TRIM_DEG = 10.0
VELOCITY_ORIENTED_MIDRALLY_LEFT_COURT_PHI_TRIM_DEG = VELOCITY_ORIENTED_MIDRALLY_NEAR_BOUNDARY_PHI_TRIM_DEG
VELOCITY_ORIENTED_MIDRALLY_RIGHT_COURT_PHI_TRIM_DEG = VELOCITY_ORIENTED_MIDRALLY_FAR_BOUNDARY_PHI_TRIM_DEG
VELOCITY_ORIENTED_PHI_CLUSTER_POWER = 2.2
VELOCITY_ORIENTED_SPEED_UPPER = 100.0
VELOCITY_ORIENTED_SPEED_RANGE_SCAN_COUNT = 17


@dataclass(frozen=True, init=False)
class DiscreteActionConfig:
    phi_bins: int = 11
    theta_bins: int = 8
    speed_bins: int = 5
    x_rec_bins: int = 5
    y_rec_bins: int = 5

    def __init__(
        self,
        phi_bins: int = 11,
        theta_bins: int = 8,
        speed_bins: int = 5,
        x_rec_bins: int = 5,
        y_rec_bins: int = 5,
        **legacy_bins: int,
    ) -> None:
        if "v_x_bins" in legacy_bins:
            phi_bins = legacy_bins.pop("v_x_bins")
        if "v_y_bins" in legacy_bins:
            theta_bins = legacy_bins.pop("v_y_bins")
        if "v_z_bins" in legacy_bins:
            speed_bins = legacy_bins.pop("v_z_bins")
        if legacy_bins:
            unknown = ", ".join(sorted(legacy_bins))
            raise TypeError(f"Unexpected discrete action bin option(s): {unknown}")
        object.__setattr__(self, "phi_bins", int(phi_bins))
        object.__setattr__(self, "theta_bins", int(theta_bins))
        object.__setattr__(self, "speed_bins", int(speed_bins))
        object.__setattr__(self, "x_rec_bins", int(x_rec_bins))
        object.__setattr__(self, "y_rec_bins", int(y_rec_bins))
        self.__post_init__()

    def __post_init__(self) -> None:
        if min(self.phi_bins, self.theta_bins, self.speed_bins, self.x_rec_bins, self.y_rec_bins) <= 0:
            raise ValueError("All discrete action bin counts must be positive.")

    def __setstate__(self, state: dict[str, int]) -> None:
        phi_bins = state.get("phi_bins", state.get("v_x_bins", 11))
        theta_bins = state.get("theta_bins", state.get("v_y_bins", 8))
        speed_bins = state.get("speed_bins", state.get("v_z_bins", 5))
        object.__setattr__(self, "phi_bins", int(phi_bins))
        object.__setattr__(self, "theta_bins", int(theta_bins))
        object.__setattr__(self, "speed_bins", int(speed_bins))
        object.__setattr__(self, "x_rec_bins", int(state.get("x_rec_bins", 5)))
        object.__setattr__(self, "y_rec_bins", int(state.get("y_rec_bins", 5)))
        self.__post_init__()


@dataclass(frozen=True)
class HitterActionDecode:
    flat_index: int
    shot_action: ShotAction
    phi_index: int | None = None
    theta_index: int | None = None
    speed_index: int | None = None
    x_rec_index: int | None = None
    y_rec_index: int | None = None
    tactic_action: TacticAction1D | TacticAction2D | None = None
    landing_zone_index: int | None = None
    angle_bin_index: int | None = None
    power_bin_index: int | None = None
    tactic_shot_name: str | None = None
    lookup_valid: bool | None = None
    lookup_fallback_used: bool | None = None
    lookup_score: float | None = None
    lookup_contact_bins: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ProjectedHitterAction:
    shot_action: ShotAction
    projected: bool
    prepared_shot: PreparedShot


def _prepared_shot_for_projected_action(prepared: PreparedShot) -> PreparedShot:
    applied = prepared.validated_action.applied
    return replace(
        prepared,
        validated_action=ValidatedShotAction(
            requested=applied,
            applied=applied,
            projected=False,
        ),
    )


@dataclass(frozen=True)
class _SpeedValidationContext:
    state: StageState
    cos_phi: float
    sin_phi: float
    cos_theta: float
    sin_theta: float
    required_net_z: float
    target_x_low: float
    target_x_high: float
    target_y_low: float
    target_y_high: float


@dataclass(frozen=True)
class _DragLandingMetrics:
    landing_x: float
    landing_y: float
    net_z: float | None


class VelocityOrientedActionMapper:
    """Maps a single discrete action id to hitter or receiver decisions."""

    policy_type = "velocity_oriented"
    tactic_zone_names: tuple[str, ...] = ()
    tactic_angle_names: tuple[str, ...] = ()
    tactic_power_names: tuple[str, ...] = ()
    tactic_shot_names: tuple[str, ...] = ()

    def __init__(
        self,
        config: SimulationConfig,
        discrete_config: DiscreteActionConfig | None = None,
    ) -> None:
        self.config = config
        self.discrete_config = discrete_config or DiscreteActionConfig()
        self._effective_phi_bins = self.discrete_config.phi_bins if self.config.court.lateral_motion_enabled else 1
        self._effective_x_rec_bins = self.discrete_config.x_rec_bins if self.config.court.lateral_motion_enabled else 1
        self._phi_grid_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._theta_grid_cache: dict[tuple[Any, ...], np.ndarray] = {}

        self.hitter_action_count = (
            self._effective_phi_bins
            * self.discrete_config.theta_bins
            * self.discrete_config.speed_bins
            * self._effective_x_rec_bins
            * self.discrete_config.y_rec_bins
        )
        self.receiver_action_count = self.config.action.intercept_count
        self.action_count = max(self.hitter_action_count, self.receiver_action_count)
        self.action_space = spaces.Discrete(self.action_count)

    def _linspace_value(self, lower: float, upper: float, index: int, count: int) -> float:
        if count == 1:
            return 0.5 * (lower + upper)
        return float(np.linspace(lower, upper, count)[index])

    def decode_hitter(self, action: int, state: StageState) -> HitterActionDecode:
        if action < 0:
            raise ValueError("Discrete action must be non-negative.")

        flat_index = action % self.hitter_action_count
        y_rec_index = flat_index % self.discrete_config.y_rec_bins
        rem = flat_index // self.discrete_config.y_rec_bins
        x_rec_index = rem % self._effective_x_rec_bins
        rem //= self._effective_x_rec_bins
        speed_index = rem % self.discrete_config.speed_bins
        rem //= self.discrete_config.speed_bins
        theta_index = rem % self.discrete_config.theta_bins
        phi_index = rem // self.discrete_config.theta_bins

        if self.config.court.lateral_motion_enabled:
            phi = self._phi_grid(state)[phi_index]
            theta = self._theta_grid_for_phi(state, phi)[theta_index]
            speed = self._speed_grid_for_phi_theta(state, phi, theta)[speed_index]
            vh = float(speed * np.cos(theta))
            v_x = float(vh * np.cos(phi))
            v_y = float(vh * np.sin(phi))
            v_z = float(speed * np.sin(theta))
        else:
            vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, self.config)
            v_x = 0.0
            v_y = self._linspace_value(vy_low, vy_high, theta_index, self.discrete_config.theta_bins)
            v_z = self._linspace_value(self.config.action.vz_min, self.config.action.vz_max, speed_index, self.discrete_config.speed_bins)
            phi, theta, speed = self._shot_context_from_velocity(v_x, v_y, v_z, state.current_hitter)

        landing_x = landing_y = 0.0
        if self.config.action.conditional_recovery_grid:
            landing_x, landing_y = landing_position(
                state,
                ShotAction(v_x=v_x, v_y=v_y, v_z=v_z, x_rec=0.0, y_rec=0.0),
                self.config,
            )
        x_rec_grid, y_rec_grid = self._recovery_grid_for_shot_context(
            state,
            phi=float(phi),
            theta=float(theta),
            speed=float(speed),
            landing_x=float(landing_x),
            landing_y=float(landing_y),
        )

        shot_action = ShotAction(
            v_x=v_x,
            v_y=v_y,
            v_z=v_z,
            x_rec=float(x_rec_grid[x_rec_index]),
            y_rec=float(y_rec_grid[y_rec_index]),
        )
        return HitterActionDecode(
            flat_index=flat_index,
            phi_index=phi_index,
            theta_index=theta_index,
            speed_index=speed_index,
            x_rec_index=x_rec_index,
            y_rec_index=y_rec_index,
            shot_action=shot_action,
        )

    def decode_hitter_for_agent(self, action: int, state: StageState, agent_side: Side) -> HitterActionDecode:
        canonical_state = canonicalize_state_for_agent(state, agent_side)
        decoded = self.decode_hitter(action, canonical_state)
        return replace(
            decoded,
            shot_action=restore_shot_action_from_agent_canonical(decoded.shot_action, agent_side),
        )

    def encode_hitter(self, shot_action: ShotAction, state: StageState) -> int:
        if self.config.court.lateral_motion_enabled:
            phi = float(np.arctan2(shot_action.v_y, shot_action.v_x))
            phi_grid = self._phi_grid(state)
            phi_index = self._nearest_angle_index(phi, phi_grid)
            vh = float(np.hypot(shot_action.v_x, shot_action.v_y))
            theta = float(np.arctan2(shot_action.v_z, vh))
            theta_grid = self._theta_grid_for_phi(state, float(phi_grid[phi_index]))
            theta_index = self._nearest_angle_index(theta, theta_grid)
            speed = float(np.hypot(vh, shot_action.v_z))
            speed_index = self._nearest_value_index(
                speed,
                self._speed_grid_for_phi_theta(
                    state,
                    float(phi_grid[phi_index]),
                    float(theta_grid[theta_index]),
                ),
            )
        else:
            vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, self.config)
            phi_index = 0
            theta_index = self._nearest_index(shot_action.v_y, vy_low, vy_high, self.discrete_config.theta_bins)
            speed_index = self._nearest_index(shot_action.v_z, self.config.action.vz_min, self.config.action.vz_max, self.discrete_config.speed_bins)

        decoded_velocity = self.decode_hitter(
            (
                ((phi_index * self.discrete_config.theta_bins + theta_index) * self.discrete_config.speed_bins + speed_index)
                * self._effective_x_rec_bins
                * self.discrete_config.y_rec_bins
            ),
            state,
        ).shot_action
        x_rec_grid, y_rec_grid = self._recovery_grid_for_shot_action(state, decoded_velocity)
        x_rec_index = self._nearest_value_index(shot_action.x_rec, x_rec_grid)
        y_rec_index = self._nearest_value_index(shot_action.y_rec, y_rec_grid)

        return (
            ((((phi_index * self.discrete_config.theta_bins) + theta_index) * self.discrete_config.speed_bins + speed_index)
             * self._effective_x_rec_bins + x_rec_index)
            * self.discrete_config.y_rec_bins
            + y_rec_index
        )

    def decode_receiver(self, action: int) -> int:
        if self.receiver_action_count <= 0:
            raise ValueError("Receiver action count must be positive.")
        return int(action) % self.receiver_action_count

    def encode_receiver(self, intercept_index: int) -> int:
        return int(intercept_index)

    def project_hitter_action(self, state: StageState, shot_action: ShotAction) -> ProjectedHitterAction:
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, self.config)
        vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, self.config)
        vx_low, vx_high = self._vx_range()

        clipped = ShotAction(
            v_x=float(np.clip(shot_action.v_x, vx_low, vx_high)),
            v_y=float(np.clip(shot_action.v_y, vy_low, vy_high)),
            v_z=float(np.clip(shot_action.v_z, self.config.action.vz_min, self.config.action.vz_max)),
            x_rec=float(np.clip(shot_action.x_rec, rec_x_low, rec_x_high)),
            y_rec=float(np.clip(shot_action.y_rec, rec_y_low, rec_y_high)),
        )
        prepared = prepare_shot(state, clipped, self.config)
        validated = prepared.validated_action
        if state.stage_index == 0 and not self._is_legal_serve_landing(
            state,
            prepared.trajectory.landing_x,
            prepared.trajectory.landing_y,
        ):
            raise ValueError("Decoded hitter action is not legal for the current serve target.")
        return ProjectedHitterAction(
            shot_action=validated.applied,
            projected=not self._actions_close(shot_action, validated.applied),
            prepared_shot=_prepared_shot_for_projected_action(prepared),
        )

    def vx_grid(self) -> np.ndarray:
        vx_low, vx_high = self._vx_range()
        return np.linspace(vx_low, vx_high, self._effective_phi_bins)

    def vy_grid(self, state: StageState) -> np.ndarray:
        return np.linspace(*vy_bounds_for_hitter(state.current_hitter, self.config), self.discrete_config.theta_bins)

    def phi_grid(self, state: StageState) -> np.ndarray:
        return self._phi_grid(state)

    def vh_grid(self, state: StageState, phi_index: int) -> np.ndarray:
        phi = float(self._phi_grid(state)[phi_index])
        theta = float(self._theta_grid_for_phi(state, phi)[0])
        return self._speed_grid_for_phi_theta(state, phi, theta) * np.cos(theta)

    def vz_grid(self) -> np.ndarray:
        return np.linspace(self.config.action.vz_min, self.config.action.vz_max, self.discrete_config.speed_bins)

    def theta_grid(self, state: StageState, phi_index: int) -> np.ndarray:
        phi = float(self._phi_grid(state)[phi_index])
        return self._theta_grid_for_phi(state, phi)

    def speed_grid(self, state: StageState, phi_index: int, theta_index: int) -> np.ndarray:
        phi = float(self._phi_grid(state)[phi_index])
        theta = float(self._theta_grid_for_phi(state, phi)[theta_index])
        return self._speed_grid_for_phi_theta(state, phi, theta)

    def x_rec_grid(self, state: StageState) -> np.ndarray:
        low, high = recovery_bounds(state.current_hitter, self.config)[0]
        return self._recovery_axis_grid(low, high, self._effective_x_rec_bins)

    def _vx_range(self) -> tuple[float, float]:
        if self.config.court.lateral_motion_enabled:
            return self.config.action.vx_min, self.config.action.vx_max
        return 0.0, 0.0

    def y_rec_grid(self, state: StageState) -> np.ndarray:
        low, high = recovery_bounds(state.current_hitter, self.config)[1]
        return self._recovery_axis_grid(low, high, self.discrete_config.y_rec_bins)

    def legal_serve_hitter_mask(self, state: StageState) -> np.ndarray:
        if state.stage_index != 0:
            return np.ones(self.hitter_action_count, dtype=bool)

        cache_key = self._serve_mask_cache_key(state)
        cached = _LEGAL_SERVE_MASK_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

        mask = np.zeros(self.hitter_action_count, dtype=bool)
        for action_index in range(self.hitter_action_count):
            decoded = self.decode_hitter(action_index, state).shot_action
            try:
                validated, trajectory = validate_and_clip_shot_action_with_result(state, decoded, self.config)
            except ValueError:
                continue
            if self._is_legal_serve_landing(state, trajectory.landing_x, trajectory.landing_y):
                mask[action_index] = True
        _LEGAL_SERVE_MASK_CACHE[cache_key] = mask.copy()
        return mask

    def legal_hitter_mask(self, state: StageState) -> np.ndarray:
        if state.stage_index == 0:
            return self.legal_serve_hitter_mask(state)

        cache_key = self._hitter_mask_cache_key(state)
        cached = _LEGAL_HITTER_MASK_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

        mask = np.zeros(self.hitter_action_count, dtype=bool)
        for action_index in range(self.hitter_action_count):
            decoded = self.decode_hitter(action_index, state).shot_action
            try:
                validate_and_clip_shot_action_with_result(state, decoded, self.config)
            except ValueError:
                continue
            mask[action_index] = True
        _LEGAL_HITTER_MASK_CACHE[cache_key] = mask.copy()
        return mask

    def _is_legal_serve_action(self, state: StageState, action: ShotAction) -> bool:
        if state.stage_index != 0:
            return True
        receiver = opponent_side(state.current_hitter)
        x_bounds, y_bounds = service_target_bounds_for_receiver_state(state, receiver, self.config)
        landing_x, landing_y = landing_position(state, action, self.config)
        return x_bounds[0] <= landing_x <= x_bounds[1] and y_bounds[0] <= landing_y <= y_bounds[1]

    def _is_legal_serve_landing(self, state: StageState, landing_x: float, landing_y: float) -> bool:
        receiver = opponent_side(state.current_hitter)
        x_bounds, y_bounds = service_target_bounds_for_receiver_state(state, receiver, self.config)
        return x_bounds[0] <= landing_x <= x_bounds[1] and y_bounds[0] <= landing_y <= y_bounds[1]

    def _serve_mask_cache_key(self, state: StageState) -> tuple[Any, ...]:
        rounded = tuple(
            round(float(value), 6)
            for value in (
                state.x_left,
                state.y_left,
                state.x_right,
                state.y_right,
                state.x0,
                state.y0,
                state.z0,
            )
        )
        return (
            state.current_hitter,
            self.policy_type,
            rounded,
            self.discrete_config,
            self.config.court,
            self.config.player,
            self.config.action,
        )

    def _hitter_mask_cache_key(self, state: StageState) -> tuple[Any, ...]:
        return (*self._serve_mask_cache_key(state), int(state.stage_index))

    def _nearest_index(self, value: float, lower: float, upper: float, count: int) -> int:
        if count == 1:
            return 0
        grid = np.linspace(lower, upper, count)
        return int(np.argmin(np.abs(grid - value)))

    def _nearest_value_index(self, value: float, grid: np.ndarray) -> int:
        if grid.size == 1:
            return 0
        return int(np.argmin(np.abs(grid - value)))

    def _nearest_angle_index(self, value: float, grid: np.ndarray) -> int:
        if grid.size == 1:
            return 0
        diffs = np.arctan2(np.sin(grid - value), np.cos(grid - value))
        return int(np.argmin(np.abs(diffs)))

    def _clustered_angle_grid(self, lower: float, upper: float, center: float, count: int) -> np.ndarray:
        if count <= 1:
            return np.asarray([float(np.clip(center, lower, upper))], dtype=float)

        center = float(np.clip(center, lower, upper))
        if np.isclose(center, lower) or np.isclose(center, upper):
            return np.linspace(lower, upper, count)

        interval_count = count - 1
        center_fraction = (center - lower) / (upper - lower)
        left_intervals = int(round(interval_count * center_fraction))
        left_intervals = int(np.clip(left_intervals, 1, interval_count - 1))
        right_intervals = interval_count - left_intervals

        power = VELOCITY_ORIENTED_PHI_CLUSTER_POWER
        left_span = center - lower
        right_span = upper - center
        left = center - left_span * (np.linspace(1.0, 0.0, left_intervals + 1) ** power)
        right = center + right_span * (np.linspace(0.0, 1.0, right_intervals + 1) ** power)
        return np.concatenate((left[:-1], right)).astype(float)

    def _phi_grid(self, state: StageState) -> np.ndarray:
        cache_key = self._angle_grid_state_key(state)
        cached = self._phi_grid_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.config.court.lateral_motion_enabled:
            grid = np.asarray([np.pi / 2.0 if state.current_hitter == "left" else -np.pi / 2.0], dtype=float)
            self._phi_grid_cache[cache_key] = grid
            return grid

        net_y = float(self.config.court.net_y)
        if state.stage_index == 0:
            receiver = opponent_side(state.current_hitter)
            (x_low, x_high), (y_low, y_high) = service_target_bounds_for_receiver_state(state, receiver, self.config)
            corner_angles = [
                float(np.arctan2(corner_y - state.y0, corner_x - state.x0))
                for corner_x in (x_low, x_high)
                for corner_y in (y_low, y_high)
            ]
            phi_low, phi_high = min(corner_angles), max(corner_angles)
        else:
            x_low = -self.config.court.half_width
            x_high = self.config.court.half_width
            left_angle = float(np.arctan2(net_y - state.y0, x_low - state.x0))
            right_angle = float(np.arctan2(net_y - state.y0, x_high - state.x0))
            phi_low, phi_high = sorted((left_angle, right_angle))
        if state.stage_index == 0:
            lower_trim = upper_trim = float(np.deg2rad(VELOCITY_ORIENTED_SERVICE_PHI_BOUNDARY_TRIM_DEG))
        else:
            lower_trim, upper_trim = self._midrally_phi_boundary_trims(state, left_angle, right_angle)
        if phi_high - phi_low > lower_trim + upper_trim:
            phi_low += lower_trim
            phi_high -= upper_trim
        if self._effective_phi_bins <= 2:
            grid = np.linspace(phi_low, phi_high, self._effective_phi_bins)
            self._phi_grid_cache[cache_key] = grid
            return grid
        forward_phi = np.pi / 2.0 if state.current_hitter == "left" else -np.pi / 2.0
        grid = self._clustered_angle_grid(phi_low, phi_high, forward_phi, self._effective_phi_bins)
        self._phi_grid_cache[cache_key] = grid
        return grid

    def _midrally_phi_boundary_trims(
        self,
        state: StageState,
        left_boundary_angle: float,
        right_boundary_angle: float,
    ) -> tuple[float, float]:
        near_trim = float(np.deg2rad(VELOCITY_ORIENTED_MIDRALLY_NEAR_BOUNDARY_PHI_TRIM_DEG))
        far_trim = float(np.deg2rad(VELOCITY_ORIENTED_MIDRALLY_FAR_BOUNDARY_PHI_TRIM_DEG))
        if state.x0 < 0.0:
            left_trim, right_trim = near_trim, far_trim
        elif state.x0 > 0.0:
            left_trim, right_trim = far_trim, near_trim
        else:
            left_trim = right_trim = near_trim

        if left_boundary_angle <= right_boundary_angle:
            return left_trim, right_trim
        return right_trim, left_trim

    def _theta_grid_for_phi(self, state: StageState, phi: float) -> np.ndarray:
        cache_key = (*self._angle_grid_state_key(state), float(phi), self.discrete_config.theta_bins)
        cached = self._theta_grid_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.config.court.lateral_motion_enabled:
            grid = np.linspace(
                np.arctan2(self.config.action.vz_min, self.config.action.vy_max_forward),
                np.arctan2(self.config.action.vz_max, self.config.action.vy_max_forward),
                self.discrete_config.theta_bins,
            )
            self._theta_grid_cache[cache_key] = grid
            return grid

        sin_phi = float(np.sin(phi))
        forward_sin = sin_phi if state.current_hitter == "left" else -sin_phi
        required_z = self.config.court.net_height + self.config.action.net_clearance_margin
        if forward_sin <= 1e-6:
            theta_low = 0.0
        else:
            horizontal_distance_to_net = abs((self.config.court.net_y - state.y0) / sin_phi)
            theta_low = float(np.arctan2(required_z - state.z0, max(horizontal_distance_to_net, 1e-6)))
        theta_high = float(np.deg2rad(VELOCITY_ORIENTED_THETA_HIGH_DEG))
        theta_low = float(np.clip(theta_low, np.deg2rad(-80.0), theta_high))
        raw_theta_bins = self.discrete_config.theta_bins + 2
        eased = np.linspace(0.0, 1.0, raw_theta_bins) ** 2.0
        grid = theta_low + (theta_high - theta_low) * eased[2:]
        self._theta_grid_cache[cache_key] = grid
        return grid

    def _angle_grid_state_key(self, state: StageState) -> tuple[Any, ...]:
        return (
            state.current_hitter,
            int(state.stage_index),
            float(state.x_left),
            float(state.y_left),
            float(state.x_right),
            float(state.y_right),
            float(state.x0),
            float(state.y0),
            float(state.z0),
            self.discrete_config,
            self.config.court,
            self.config.action,
        )

    def _intersect_speed_interval(
        self,
        lower: float,
        upper: float,
        factor: float,
        component_low: float,
        component_high: float,
    ) -> tuple[float, float] | None:
        if abs(factor) <= 1e-9:
            if component_low <= 0.0 <= component_high:
                return lower, upper
            return None
        scaled_low = component_low / factor
        scaled_high = component_high / factor
        component_speed_low = min(scaled_low, scaled_high)
        component_speed_high = max(scaled_low, scaled_high)
        lower = max(lower, float(component_speed_low))
        upper = min(upper, float(component_speed_high))
        if upper < lower:
            return None
        return lower, upper

    def _component_speed_interval_for_phi_theta(
        self,
        state: StageState,
        phi: float,
        theta: float,
    ) -> tuple[float, float] | None:
        lower = 0.0
        upper = VELOCITY_ORIENTED_SPEED_UPPER
        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))

        vx_factor = cos_theta * float(np.cos(phi))
        interval = self._intersect_speed_interval(lower, upper, vx_factor, self.config.action.vx_min, self.config.action.vx_max)
        if interval is None:
            return None
        lower, upper = interval

        forward_factor = cos_theta * (float(np.sin(phi)) if state.current_hitter == "left" else -float(np.sin(phi)))
        interval = self._intersect_speed_interval(
            lower,
            upper,
            forward_factor,
            self.config.action.vy_min_forward,
            self.config.action.vy_max_forward,
        )
        if interval is None:
            return None
        lower, upper = interval

        interval = self._intersect_speed_interval(lower, upper, sin_theta, self.config.action.vz_min, self.config.action.vz_max)
        if interval is None:
            return None
        lower, upper = interval

        lower = max(lower, 0.0)
        if upper < lower or not np.all(np.isfinite([lower, upper])):
            return None
        return float(lower), float(upper)

    def _shot_action_from_phi_theta_speed(
        self,
        state: StageState,
        phi: float,
        theta: float,
        speed: float,
        *,
        x_rec: float = 0.0,
        y_rec: float = 0.0,
    ) -> ShotAction:
        vh = float(speed * np.cos(theta))
        v_x = float(vh * np.cos(phi))
        v_y = float(vh * np.sin(phi))
        v_z = float(speed * np.sin(theta))
        if state.current_hitter == "right":
            v_y = -abs(v_y)
        return ShotAction(v_x=v_x, v_y=v_y, v_z=v_z, x_rec=x_rec, y_rec=y_rec)

    def _speed_valid_for_phi_theta(self, state: StageState, phi: float, theta: float, speed: float) -> bool:
        return self._speed_valid_for_context(self._speed_validation_context(state, phi, theta), speed)

    def _speed_validation_context(self, state: StageState, phi: float, theta: float) -> _SpeedValidationContext:
        receiver = opponent_side(state.current_hitter)
        if state.stage_index == 0:
            (x_low, x_high), (y_low, y_high) = service_target_bounds_for_receiver_state(state, receiver, self.config)
        else:
            (x_low, x_high), (y_low, y_high) = target_bounds_for_receiver(receiver, self.config)
        return _SpeedValidationContext(
            state=state,
            cos_phi=float(np.cos(phi)),
            sin_phi=float(np.sin(phi)),
            cos_theta=float(np.cos(theta)),
            sin_theta=float(np.sin(theta)),
            required_net_z=float(self.config.court.net_height + self.config.action.net_clearance_margin),
            target_x_low=float(x_low),
            target_x_high=float(x_high),
            target_y_low=float(y_low),
            target_y_high=float(y_high),
        )

    def _shot_action_from_speed_context(
        self,
        context: _SpeedValidationContext,
        speed: float,
    ) -> ShotAction:
        vh = float(speed * context.cos_theta)
        v_x = float(vh * context.cos_phi)
        v_y = float(vh * context.sin_phi)
        v_z = float(speed * context.sin_theta)
        if context.state.current_hitter == "right":
            v_y = -abs(v_y)
        return ShotAction(v_x=v_x, v_y=v_y, v_z=v_z, x_rec=0.0, y_rec=0.0)

    def _speed_valid_for_context(self, context: _SpeedValidationContext, speed: float) -> bool:
        action = self._shot_action_from_speed_context(context, speed)
        outcome = self._fast_landing_and_net_crossing(context.state, action)
        if outcome is None:
            return False
        landing_x, landing_y, net_z = outcome
        if net_z < context.required_net_z:
            return False
        return (
            context.target_x_low <= landing_x <= context.target_x_high
            and context.target_y_low <= landing_y <= context.target_y_high
        )

    def _fast_landing_and_net_crossing(
        self,
        state: StageState,
        action: ShotAction,
    ) -> tuple[float, float, float] | None:
        if not np.all(np.isfinite([action.v_x, action.v_y, action.v_z])):
            return None
        if state.current_hitter == "left":
            if not (state.y0 < self.config.court.net_y and action.v_y > 0.0):
                return None
        elif not (state.y0 > self.config.court.net_y and action.v_y < 0.0):
            return None

        if self.config.action.effective_trajectory_mode == "ballistic":
            net_crossing = ballistic_net_crossing(
                state.x0,
                state.y0,
                state.z0,
                action.v_x,
                action.v_y,
                action.v_z,
                self.config.court.net_y,
                g=self.config.action.gravity,
            )
            if net_crossing is None:
                return None
            landing_time = ballistic_landing_time(state.z0, action.v_z, self.config.action.gravity)
            if not (0.0 < net_crossing.t < landing_time):
                return None
            landing_x, landing_y = ballistic_landing_point(
                state.x0,
                state.y0,
                state.z0,
                action.v_x,
                action.v_y,
                action.v_z,
                self.config.action.gravity,
            )
            return float(landing_x), float(landing_y), float(net_crossing.z)

        return self._fast_drag_landing_and_net_crossing(state, action)

    def _fast_drag_landing_and_net_crossing(
        self,
        state: StageState,
        action: ShotAction,
    ) -> tuple[float, float, float] | None:
        metrics = self._fast_drag_landing_metrics(state, action)
        if metrics is None or metrics.net_z is None:
            return None
        return metrics.landing_x, metrics.landing_y, metrics.net_z

    def _fast_drag_landing_metrics(
        self,
        state: StageState,
        action: ShotAction,
    ) -> _DragLandingMetrics | None:
        dt = float(self.config.action.drag_dt)
        if dt <= 0.0:
            return None
        horizontal_drag = self.config.action.effective_horizontal_drag_coefficient
        vertical_drag = self.config.action.effective_vertical_drag_coefficient
        g = float(self.config.action.gravity)
        net_y = float(self.config.court.net_y)
        t = 0.0
        x = float(state.x0)
        y = float(state.y0)
        z = float(state.z0)
        vx = float(action.v_x)
        vy = float(action.v_y)
        vz = float(action.v_z)
        net_z: float | None = None
        max_time = 10.0

        while t < max_time:
            prev_t, prev_x, prev_y, prev_z = t, x, y, z
            x = x + vx * dt
            y = y + vy * dt
            z = z + vz * dt
            speed = sqrt(vx * vx + vy * vy + vz * vz)
            vx = vx + (-horizontal_drag * speed * vx) * dt
            vy = vy + (-horizontal_drag * speed * vy) * dt
            vz = vz + (-g - vertical_drag * speed * vz) * dt
            t = t + dt

            if net_z is None and (prev_y - net_y) * (y - net_y) <= 0.0 and abs(prev_y - y) > 1e-12:
                ratio = (net_y - prev_y) / (y - prev_y)
                ratio = min(max(ratio, 0.0), 1.0)
                net_t = prev_t + ratio * (t - prev_t)
                if net_t <= 0.0:
                    return None
                net_z = float(prev_z + ratio * (z - prev_z))

            if z <= 0.0:
                ratio = 1.0 if abs(prev_z - z) <= 1e-12 else min(max(prev_z / (prev_z - z), 0.0), 1.0)
                landing_x = float(prev_x + ratio * (x - prev_x))
                landing_y = float(prev_y + ratio * (y - prev_y))
                return _DragLandingMetrics(landing_x=landing_x, landing_y=landing_y, net_z=net_z)

        return None

    def valid_speed_range(self, state: StageState, phi_index: int, theta_index: int) -> tuple[float, float] | None:
        phi = float(self._phi_grid(state)[phi_index])
        theta = float(self._theta_grid_for_phi(state, phi)[theta_index])
        return self._valid_speed_range_for_phi_theta(state, phi, theta)

    def _valid_speed_range_for_phi_theta(
        self,
        state: StageState,
        phi: float,
        theta: float,
    ) -> tuple[float, float] | None:
        if not self.config.court.lateral_motion_enabled:
            return None

        cache_key = self._speed_range_cache_key(state, phi, theta)
        if cache_key in _VELOCITY_SPEED_RANGE_CACHE:
            cached = _VELOCITY_SPEED_RANGE_CACHE[cache_key]
            return None if cached is None else (float(cached[0]), float(cached[1]))

        result = self._physical_speed_interval_for_phi_theta(
            state,
            phi,
            theta,
            (0.0, VELOCITY_ORIENTED_SPEED_UPPER),
        )
        _VELOCITY_SPEED_RANGE_CACHE[cache_key] = result
        return result

    def _physical_speed_interval_for_phi_theta(
        self,
        state: StageState,
        phi: float,
        theta: float,
        bounds: tuple[float, float],
    ) -> tuple[float, float] | None:
        lower, upper = bounds
        if upper <= lower:
            return None

        context = self._speed_validation_context(state, phi, theta)
        if self.config.action.effective_trajectory_mode == "drag_square":
            return self._drag_speed_interval_for_context(context, lower, upper)

        speeds = np.linspace(lower, upper, VELOCITY_ORIENTED_SPEED_RANGE_SCAN_COUNT)
        valid = np.asarray(
            [self._speed_valid_for_context(context, float(speed)) for speed in speeds],
            dtype=bool,
        )
        if not valid.any():
            return None

        indices = np.flatnonzero(valid)
        groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        group = max(groups, key=len)
        first = int(group[0])
        last = int(group[-1])

        valid_low = float(speeds[first])
        valid_high = float(speeds[last])
        if first > 0:
            invalid_low = float(speeds[first - 1])
            for _ in range(16):
                midpoint = 0.5 * (invalid_low + valid_low)
                if self._speed_valid_for_context(context, midpoint):
                    valid_low = midpoint
                else:
                    invalid_low = midpoint
        else:
            valid_low = lower

        if last < speeds.size - 1:
            invalid_high = float(speeds[last + 1])
            for _ in range(16):
                midpoint = 0.5 * (valid_high + invalid_high)
                if self._speed_valid_for_context(context, midpoint):
                    valid_high = midpoint
                else:
                    invalid_high = midpoint
        else:
            valid_high = upper

        if valid_high < valid_low:
            return None
        return float(valid_low), float(valid_high)

    def _drag_speed_interval_for_context(
        self,
        context: _SpeedValidationContext,
        lower: float,
        upper: float,
    ) -> tuple[float, float] | None:
        target_interval = self._target_ray_progress_interval(context)
        if target_interval is None:
            return None
        target_low, target_high = target_interval

        lower_status = self._drag_speed_status(context, lower, target_low, target_high)
        upper_status = self._drag_speed_status(context, upper, target_low, target_high)
        if lower_status > 0 or upper_status < 0:
            return None

        if lower_status == 0:
            valid_low = lower
        else:
            invalid_low = lower
            valid_or_long = upper
            for _ in range(16):
                midpoint = 0.5 * (invalid_low + valid_or_long)
                if self._drag_speed_status(context, midpoint, target_low, target_high) < 0:
                    invalid_low = midpoint
                else:
                    valid_or_long = midpoint
            if self._drag_speed_status(context, valid_or_long, target_low, target_high) != 0:
                return None
            valid_low = valid_or_long

        if upper_status == 0:
            valid_high = upper
        else:
            valid_or_short = valid_low
            invalid_high = upper
            for _ in range(16):
                midpoint = 0.5 * (valid_or_short + invalid_high)
                if self._drag_speed_status(context, midpoint, target_low, target_high) == 0:
                    valid_or_short = midpoint
                else:
                    invalid_high = midpoint
            valid_high = valid_or_short

        if valid_high < valid_low:
            return None
        return float(valid_low), float(valid_high)

    def _target_ray_progress_interval(self, context: _SpeedValidationContext) -> tuple[float, float] | None:
        state = context.state
        dx = context.cos_phi
        dy = context.sin_phi
        if state.current_hitter == "right":
            dy = -abs(dy)

        lower = 0.0
        upper = float("inf")
        for origin, direction, bound_low, bound_high in (
            (state.x0, dx, context.target_x_low, context.target_x_high),
            (state.y0, dy, context.target_y_low, context.target_y_high),
        ):
            if abs(direction) <= 1e-9:
                if bound_low <= origin <= bound_high:
                    continue
                return None
            axis_low = (bound_low - origin) / direction
            axis_high = (bound_high - origin) / direction
            lower = max(lower, float(min(axis_low, axis_high)))
            upper = min(upper, float(max(axis_low, axis_high)))
            if upper < lower:
                return None
        if not np.all(np.isfinite([lower, upper])):
            return None
        return float(lower), float(upper)

    def _drag_speed_status(
        self,
        context: _SpeedValidationContext,
        speed: float,
        target_progress_low: float,
        target_progress_high: float,
    ) -> int:
        action = self._shot_action_from_speed_context(context, speed)
        horizontal_speed = float(np.hypot(action.v_x, action.v_y))
        if horizontal_speed <= 1e-9:
            return -1
        return self._fast_drag_speed_status(
            context,
            action,
            target_progress_low,
            target_progress_high,
            horizontal_speed,
        )

    def _fast_drag_speed_status(
        self,
        context: _SpeedValidationContext,
        action: ShotAction,
        target_progress_low: float,
        target_progress_high: float,
        horizontal_speed: float,
    ) -> int:
        state = context.state
        dt = float(self.config.action.drag_dt)
        if dt <= 0.0:
            return -1
        horizontal_drag = self.config.action.effective_horizontal_drag_coefficient
        vertical_drag = self.config.action.effective_vertical_drag_coefficient
        g = float(self.config.action.gravity)
        net_y = float(self.config.court.net_y)
        x0 = float(state.x0)
        y0 = float(state.y0)
        dx = float(action.v_x) / horizontal_speed
        dy = float(action.v_y) / horizontal_speed
        t = 0.0
        x = x0
        y = y0
        z = float(state.z0)
        vx = float(action.v_x)
        vy = float(action.v_y)
        vz = float(action.v_z)
        net_ok = False
        max_time = 10.0

        while t < max_time:
            prev_t, prev_x, prev_y, prev_z = t, x, y, z
            x = x + vx * dt
            y = y + vy * dt
            z = z + vz * dt
            speed = sqrt(vx * vx + vy * vy + vz * vz)
            vx = vx + (-horizontal_drag * speed * vx) * dt
            vy = vy + (-horizontal_drag * speed * vy) * dt
            vz = vz + (-g - vertical_drag * speed * vz) * dt
            t = t + dt

            if not net_ok and (prev_y - net_y) * (y - net_y) <= 0.0 and abs(prev_y - y) > 1e-12:
                ratio = (net_y - prev_y) / (y - prev_y)
                ratio = min(max(ratio, 0.0), 1.0)
                net_t = prev_t + ratio * (t - prev_t)
                if net_t <= 0.0:
                    return -1
                net_z = prev_z + ratio * (z - prev_z)
                if net_z < context.required_net_z:
                    return -1
                net_ok = True

            if z <= 0.0:
                if not net_ok:
                    return -1
                ratio = 1.0 if abs(prev_z - z) <= 1e-12 else min(max(prev_z / (prev_z - z), 0.0), 1.0)
                landing_x = prev_x + ratio * (x - prev_x)
                landing_y = prev_y + ratio * (y - prev_y)
                landing_progress = (landing_x - x0) * dx + (landing_y - y0) * dy
                if landing_progress < target_progress_low:
                    return -1
                if landing_progress > target_progress_high:
                    return 1
                return 0

            progress = (x - x0) * dx + (y - y0) * dy
            if net_ok and progress > target_progress_high:
                return 1

        return -1

    def _speed_range_cache_key(self, state: StageState, phi: float, theta: float) -> tuple[Any, ...]:
        rounded_state = tuple(
            round(float(value), 6)
            for value in (
                state.x_left,
                state.y_left,
                state.x_right,
                state.y_right,
                state.x0,
                state.y0,
                state.z0,
            )
        )
        return (
            state.current_hitter,
            int(state.stage_index),
            rounded_state,
            round(float(phi), 9),
            round(float(theta), 9),
            self.discrete_config.speed_bins,
            self.config.court,
            self.config.player,
            self.config.action,
        )

    def _speed_grid_for_phi_theta(self, state: StageState, phi: float, theta: float) -> np.ndarray:
        if not self.config.court.lateral_motion_enabled:
            return np.abs(np.linspace(*vy_bounds_for_hitter(state.current_hitter, self.config), self.discrete_config.speed_bins))

        valid_range = self._valid_speed_range_for_phi_theta(state, phi, theta)
        if valid_range is None:
            lower = 0.0
            upper = VELOCITY_ORIENTED_SPEED_UPPER
        else:
            lower, upper = valid_range
        lower = min(float(lower), float(upper))
        if self.discrete_config.speed_bins == 1:
            return np.asarray([upper], dtype=float)
        return np.linspace(lower, upper, self.discrete_config.speed_bins)

    def _shot_context_from_velocity(self, v_x: float, v_y: float, v_z: float, side: Side) -> tuple[float, float, float]:
        if self.config.court.lateral_motion_enabled:
            phi = float(np.arctan2(v_y, v_x))
        else:
            phi = float(np.pi / 2.0 if side == "left" else -np.pi / 2.0)
        horizontal_speed = float(np.hypot(v_x, v_y))
        theta = float(np.arctan2(v_z, horizontal_speed))
        speed = float(np.hypot(horizontal_speed, v_z))
        return phi, theta, speed

    def _recovery_grid_for_shot_action(self, state: StageState, shot_action: ShotAction) -> tuple[np.ndarray, np.ndarray]:
        phi, theta, speed = self._shot_context_from_velocity(
            shot_action.v_x,
            shot_action.v_y,
            shot_action.v_z,
            state.current_hitter,
        )
        landing_x = landing_y = 0.0
        if self.config.action.conditional_recovery_grid:
            landing_x, landing_y = landing_position(state, shot_action, self.config)
        return self._recovery_grid_for_shot_context(
            state,
            phi=phi,
            theta=theta,
            speed=speed,
            landing_x=float(landing_x),
            landing_y=float(landing_y),
        )

    def _recovery_grid_for_shot_context(
        self,
        state: StageState,
        *,
        phi: float,
        theta: float,
        speed: float,
        landing_x: float,
        landing_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, self.config)
        if not self.config.action.conditional_recovery_grid:
            if self.config.court.lateral_motion_enabled:
                x_grid = self._recovery_axis_grid(rec_x_low, rec_x_high, self._effective_x_rec_bins)
            else:
                x_grid = np.asarray([0.0], dtype=float)
            y_grid = self._recovery_axis_grid(rec_y_low, rec_y_high, self.discrete_config.y_rec_bins)
            return x_grid, y_grid

        x_anchor, y_anchor = self._conditional_recovery_anchor(
            state,
            phi=phi,
            theta=theta,
            speed=speed,
            landing_x=landing_x,
            landing_y=landing_y,
        )
        if self.config.court.lateral_motion_enabled:
            x_grid = self._conditional_recovery_axis_grid(rec_x_low, rec_x_high, self._effective_x_rec_bins, x_anchor)
        else:
            x_grid = np.asarray([0.0], dtype=float)
        y_grid = self._conditional_recovery_axis_grid(rec_y_low, rec_y_high, self.discrete_config.y_rec_bins, y_anchor)
        return x_grid, y_grid

    def _conditional_recovery_axis_grid(self, lower: float, upper: float, count: int, anchor: float) -> np.ndarray:
        anchor = float(np.clip(anchor, lower, upper))
        if count == 1:
            return np.asarray([anchor], dtype=float)
        units = np.linspace(0.0, 1.0, count + 2)[1:-1]
        return np.asarray([self._interp_around_anchor(float(unit), lower, upper, anchor) for unit in units], dtype=float)

    def _recovery_axis_grid(self, lower: float, upper: float, count: int) -> np.ndarray:
        if count == 1:
            return np.asarray([0.5 * (lower + upper)], dtype=float)
        if self.config.court.lateral_motion_enabled and count in {3, 5}:
            return np.linspace(lower, upper, count + 2)[1:-1]
        return np.linspace(lower, upper, count)

    def _conditional_recovery_anchor(
        self,
        state: StageState,
        *,
        phi: float,
        theta: float,
        speed: float,
        landing_x: float,
        landing_y: float,
    ) -> tuple[float, float]:
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, self.config)
        receiver = opponent_side(state.current_hitter)
        (target_x_low, target_x_high), (target_y_low, target_y_high) = target_bounds_for_receiver(receiver, self.config)

        if not np.isfinite(landing_x):
            landing_x = 0.5 * (target_x_low + target_x_high)
        if not np.isfinite(landing_y):
            landing_y = 0.5 * (target_y_low + target_y_high)

        landing_x = float(np.clip(landing_x, target_x_low, target_x_high))
        landing_y = float(np.clip(landing_y, target_y_low, target_y_high))
        x_ratio = self._ratio(landing_x, target_x_low, target_x_high)
        if receiver == "right":
            depth_ratio = self._ratio(landing_y, target_y_low, target_y_high)
        else:
            depth_ratio = self._ratio(target_y_high - landing_y, 0.0, target_y_high - target_y_low)

        if self.config.court.lateral_motion_enabled:
            phi_grid = self._phi_grid(state)
            phi_ratio = self._ratio(phi, float(phi_grid[0]), float(phi_grid[-1]))
            x_ratio = 0.85 * x_ratio + 0.15 * phi_ratio

        forward_sin = float(np.sin(phi)) if state.current_hitter == "left" else -float(np.sin(phi))
        forward_speed = float(speed * max(np.cos(theta), 0.0) * max(forward_sin, 0.0))
        speed_depth_ratio = self._ratio(
            forward_speed,
            self.config.action.vy_min_forward,
            self.config.action.vy_max_forward,
        )
        depth_ratio = 0.85 * depth_ratio + 0.15 * speed_depth_ratio

        x_anchor = rec_x_low + x_ratio * (rec_x_high - rec_x_low)
        depth_fraction = 0.35 + 0.4 * depth_ratio
        if state.current_hitter == "left":
            y_anchor = rec_y_low + depth_fraction * (rec_y_high - rec_y_low)
        else:
            y_anchor = rec_y_high - depth_fraction * (rec_y_high - rec_y_low)
        return (
            float(np.clip(x_anchor, rec_x_low, rec_x_high)),
            float(np.clip(y_anchor, rec_y_low, rec_y_high)),
        )

    def _interp_around_anchor(self, unit_value: float, lower: float, upper: float, anchor: float) -> float:
        unit_value = float(np.clip(unit_value, 0.0, 1.0))
        anchor = float(np.clip(anchor, lower, upper))
        if unit_value <= 0.5:
            return float(lower + unit_value * 2.0 * (anchor - lower))
        return float(anchor + (unit_value - 0.5) * 2.0 * (upper - anchor))

    def _ratio(self, value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.5
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    def _actions_close(self, left: ShotAction, right: ShotAction) -> bool:
        return (
            np.isclose(left.v_x, right.v_x)
            and np.isclose(left.v_y, right.v_y)
            and np.isclose(left.v_z, right.v_z)
            and np.isclose(left.x_rec, right.x_rec)
            and np.isclose(left.y_rec, right.y_rec)
        )


class TacticOrientedActionMapper:
    policy_type = "tactic_oriented"
    tactic_power_names = POWER_BIN_NAMES
    tactic_shot_names = SHOT_NAME_ORDER

    def __init__(
        self,
        config: SimulationConfig,
        tactic_runtime_config: TacticRuntimeConfig | None = None,
    ) -> None:
        self.config = config
        self.tactic_runtime_config = tactic_runtime_config or TacticRuntimeConfig()
        if config.court.lateral_motion_enabled:
            self.lookup = TacticLookup2D(config, self.tactic_runtime_config)
            self.tactic_zone_names = self.lookup.zone_names
        else:
            self.lookup = TacticLookup1D(config, self.tactic_runtime_config)
            self.tactic_zone_names = self.lookup.zone_names
        self.tactic_angle_names = tuple(self.lookup.angle_names)
        self.tactic_power_names = tuple(self.lookup.power_names)
        self.hitter_action_count = self.lookup.action_count
        self.receiver_action_count = self.config.action.intercept_count
        self.action_count = max(self.hitter_action_count, self.receiver_action_count)
        self.action_space = spaces.Discrete(self.action_count)

    def decode_hitter(self, action: int, state: StageState) -> HitterActionDecode:
        flat_index = int(action) % self.hitter_action_count
        if self.config.court.lateral_motion_enabled:
            tactic_action = self.lookup.flat_to_action(flat_index)
            entry = self.lookup.lookup(state, tactic_action)
            x_rec, y_rec = default_recovery_target(
                state,
                landing_row=tactic_action.landing_row,
                landing_col=tactic_action.landing_col,
                config=self.config,
                landing_row_count=len(self.lookup.landing_row_centers),
            )
            shot_action = ShotAction(
                v_x=float(entry.velocity[0]),
                v_y=float(entry.velocity[1]),
                v_z=float(entry.velocity[2]),
                x_rec=x_rec,
                y_rec=y_rec,
            )
            return HitterActionDecode(
                flat_index=flat_index,
                shot_action=shot_action,
                tactic_action=tactic_action,
                landing_zone_index=tactic_action.landing_zone,
                angle_bin_index=tactic_action.angle_bin,
                power_bin_index=tactic_action.power_bin,
                tactic_shot_name=entry.inferred_shot_name,
                lookup_valid=entry.valid,
                lookup_fallback_used=entry.fallback_used,
                lookup_score=entry.score,
                lookup_contact_bins=entry.contact_bins,
            )

        tactic_action = self.lookup.flat_to_action(flat_index)
        entry = self.lookup.lookup(state, tactic_action)
        x_rec, y_rec = default_recovery_target(
            state,
            landing_row=tactic_action.landing_zone,
            landing_col=None,
            config=self.config,
            landing_row_count=len(self.lookup.landing_zone_centers),
        )
        shot_action = ShotAction(
            v_x=0.0,
            v_y=float(entry.velocity[0]),
            v_z=float(entry.velocity[1]),
            x_rec=x_rec,
            y_rec=y_rec,
        )
        return HitterActionDecode(
            flat_index=flat_index,
            shot_action=shot_action,
            tactic_action=tactic_action,
            landing_zone_index=tactic_action.landing_zone,
            angle_bin_index=tactic_action.angle_bin,
            power_bin_index=tactic_action.power_bin,
            tactic_shot_name=entry.inferred_shot_name,
            lookup_valid=entry.valid,
            lookup_fallback_used=entry.fallback_used,
            lookup_score=entry.score,
            lookup_contact_bins=entry.contact_bins,
        )

    def decode_hitter_for_agent(self, action: int, state: StageState, agent_side: Side) -> HitterActionDecode:
        canonical_state = canonicalize_state_for_agent(state, agent_side)
        decoded = self.decode_hitter(action, canonical_state)
        return replace(
            decoded,
            shot_action=restore_shot_action_from_agent_canonical(decoded.shot_action, agent_side),
        )

    def encode_hitter(self, shot_action: ShotAction, state: StageState) -> int:
        return int(self.lookup.best_flat_action_for_velocity(state, shot_action))

    def decode_receiver(self, action: int) -> int:
        if self.receiver_action_count <= 0:
            raise ValueError("Receiver action count must be positive.")
        return int(action) % self.receiver_action_count

    def encode_receiver(self, intercept_index: int) -> int:
        return int(intercept_index)

    def project_hitter_action(self, state: StageState, shot_action: ShotAction) -> ProjectedHitterAction:
        try:
            prepared = prepare_shot(state, shot_action, self.config)
            validated = prepared.validated_action
            return ProjectedHitterAction(
                shot_action=validated.applied,
                projected=not self._actions_close(shot_action, validated.applied),
                prepared_shot=_prepared_shot_for_projected_action(prepared),
            )
        except ValueError:
            from badminton.agents import SafeHitter

            safe_action = SafeHitter().choose_action(state, self.config)
            prepared = prepare_shot(state, safe_action, self.config)
            return ProjectedHitterAction(
                shot_action=prepared.validated_action.applied,
                projected=True,
                prepared_shot=_prepared_shot_for_projected_action(prepared),
            )

    def legal_serve_hitter_mask(self, state: StageState) -> np.ndarray:
        if state.stage_index != 0:
            return np.ones(self.hitter_action_count, dtype=bool)

        cache_key = self._serve_mask_cache_key(state)
        cached = _LEGAL_SERVE_MASK_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

        mask = np.zeros(self.hitter_action_count, dtype=bool)
        for flat_index in range(self.hitter_action_count):
            decoded = self.decode_hitter(flat_index, state)
            try:
                validated, trajectory = validate_and_clip_shot_action_with_result(state, decoded.shot_action, self.config)
            except ValueError:
                continue
            if self._is_legal_serve_landing(state, trajectory.landing_x, trajectory.landing_y):
                mask[flat_index] = True
        _LEGAL_SERVE_MASK_CACHE[cache_key] = mask.copy()
        return mask

    def legal_hitter_mask(self, state: StageState) -> np.ndarray:
        if state.stage_index == 0:
            return self.legal_serve_hitter_mask(state)

        cache_key = self._hitter_mask_cache_key(state)
        cached = _LEGAL_HITTER_MASK_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

        mask = np.zeros(self.hitter_action_count, dtype=bool)
        for flat_index in range(self.hitter_action_count):
            decoded = self.decode_hitter(flat_index, state)
            try:
                validate_and_clip_shot_action_with_result(state, decoded.shot_action, self.config)
            except ValueError:
                continue
            mask[flat_index] = True
        _LEGAL_HITTER_MASK_CACHE[cache_key] = mask.copy()
        return mask

    def _is_legal_serve_action(self, state: StageState, action: ShotAction) -> bool:
        if state.stage_index != 0:
            return True
        receiver = opponent_side(state.current_hitter)
        x_bounds, y_bounds = service_target_bounds_for_receiver_state(state, receiver, self.config)
        landing_x, landing_y = landing_position(state, action, self.config)
        return x_bounds[0] <= landing_x <= x_bounds[1] and y_bounds[0] <= landing_y <= y_bounds[1]

    def _is_legal_serve_landing(self, state: StageState, landing_x: float, landing_y: float) -> bool:
        receiver = opponent_side(state.current_hitter)
        x_bounds, y_bounds = service_target_bounds_for_receiver_state(state, receiver, self.config)
        return x_bounds[0] <= landing_x <= x_bounds[1] and y_bounds[0] <= landing_y <= y_bounds[1]

    def _serve_mask_cache_key(self, state: StageState) -> tuple[Any, ...]:
        rounded = tuple(
            round(float(value), 6)
            for value in (
                state.x_left,
                state.y_left,
                state.x_right,
                state.y_right,
                state.x0,
                state.y0,
                state.z0,
            )
        )
        return (
            state.current_hitter,
            rounded,
            self.policy_type,
            self.tactic_runtime_config,
            self.config.court,
            self.config.player,
            self.config.action,
        )

    def _hitter_mask_cache_key(self, state: StageState) -> tuple[Any, ...]:
        return (*self._serve_mask_cache_key(state), int(state.stage_index))

    def _actions_close(self, left: ShotAction, right: ShotAction) -> bool:
        return (
            np.isclose(left.v_x, right.v_x)
            and np.isclose(left.v_y, right.v_y)
            and np.isclose(left.v_z, right.v_z)
            and np.isclose(left.x_rec, right.x_rec)
            and np.isclose(left.y_rec, right.y_rec)
        )

    def lookup_summary(self) -> dict[str, float]:
        return self.lookup.summary()


class DiscreteActionMapper:
    def __init__(
        self,
        config: SimulationConfig,
        discrete_config: DiscreteActionConfig | None = None,
        *,
        policy_type: str = "velocity_oriented",
        tactic_runtime_config: TacticRuntimeConfig | None = None,
    ) -> None:
        self.config = config
        self.discrete_config = discrete_config or DiscreteActionConfig()
        self.policy_type = validate_policy_type(policy_type)
        if self.policy_type in CONDITIONAL_VELOCITY_POLICY_TYPES:
            self._impl: Any = VelocityOrientedActionMapper(config, self.discrete_config)
        else:
            self._impl = TacticOrientedActionMapper(config, tactic_runtime_config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)

    @property
    def hitter_action_count(self) -> int:
        return int(self._impl.hitter_action_count)

    @property
    def receiver_action_count(self) -> int:
        return int(self._impl.receiver_action_count)

    @property
    def action_count(self) -> int:
        return int(self._impl.action_count)

    @property
    def action_space(self):
        if self.policy_type in {"continuous_action", MIXED_DISCRETE_CONTINOUS_POLICY_TYPE}:
            return spaces.Box(low=-1.0, high=1.0, shape=(CONTINUOUS_ACTION_DIM,), dtype=np.float32)
        return self._impl.action_space

    @property
    def tactic_zone_names(self) -> tuple[str, ...]:
        return tuple(getattr(self._impl, "tactic_zone_names", ()))

    @property
    def tactic_angle_names(self) -> tuple[str, ...]:
        return tuple(getattr(self._impl, "tactic_angle_names", ()))

    @property
    def tactic_power_names(self) -> tuple[str, ...]:
        return tuple(getattr(self._impl, "tactic_power_names", ()))

    @property
    def tactic_shot_names(self) -> tuple[str, ...]:
        return tuple(getattr(self._impl, "tactic_shot_names", ()))

    def decode_hitter(self, action: int, state: StageState) -> HitterActionDecode:
        if self.policy_type == "continuous_action":
            if self._is_scalar_integer_action(action):
                return self._impl.decode_hitter(int(action), state)
            return self._decode_continuous_hitter(action, state)
        if self.policy_type == MIXED_DISCRETE_CONTINOUS_POLICY_TYPE:
            if self._is_scalar_integer_action(action):
                return self._impl.decode_hitter(int(action), state)
            return self._decode_mixed_hitter(action, state)
        return self._impl.decode_hitter(action, state)

    def decode_hitter_for_agent(self, action: int, state: StageState, agent_side: Side) -> HitterActionDecode:
        if self.policy_type == "continuous_action":
            if self._is_scalar_integer_action(action):
                return self._impl.decode_hitter_for_agent(int(action), state, agent_side)
            canonical_state = canonicalize_state_for_agent(state, agent_side)
            decoded = self._decode_continuous_hitter(action, canonical_state)
            return replace(
                decoded,
                shot_action=restore_shot_action_from_agent_canonical(decoded.shot_action, agent_side),
            )
        if self.policy_type == MIXED_DISCRETE_CONTINOUS_POLICY_TYPE:
            if self._is_scalar_integer_action(action):
                return self._impl.decode_hitter_for_agent(int(action), state, agent_side)
            canonical_state = canonicalize_state_for_agent(state, agent_side)
            decoded = self._decode_mixed_hitter(action, canonical_state)
            return replace(
                decoded,
                shot_action=restore_shot_action_from_agent_canonical(decoded.shot_action, agent_side),
            )
        if hasattr(self._impl, "decode_hitter_for_agent"):
            return self._impl.decode_hitter_for_agent(action, state, agent_side)
        canonical_state = canonicalize_state_for_agent(state, agent_side)
        decoded = self._impl.decode_hitter(action, canonical_state)
        return replace(
            decoded,
            shot_action=restore_shot_action_from_agent_canonical(decoded.shot_action, agent_side),
        )

    def encode_hitter(self, shot_action: ShotAction, state: StageState) -> int:
        return int(self._impl.encode_hitter(shot_action, state))

    def decode_receiver(self, action: int) -> int:
        if self.policy_type in {"continuous_action", MIXED_DISCRETE_CONTINOUS_POLICY_TYPE}:
            if self._is_scalar_integer_action(action):
                return int(self._impl.decode_receiver(int(action)))
            values = np.asarray(action, dtype=float).reshape(-1)
            receiver_value = float(values[5]) if values.size > 5 else 0.0
            unit = float(np.clip(0.5 * (receiver_value + 1.0), 0.0, 1.0))
            return int(np.clip(round(unit * (self.receiver_action_count - 1)), 0, self.receiver_action_count - 1))
        return int(self._impl.decode_receiver(action))

    def encode_receiver(self, intercept_index: int) -> int:
        return int(self._impl.encode_receiver(intercept_index))

    def project_hitter_action(self, state: StageState, shot_action: ShotAction) -> ProjectedHitterAction:
        return self._impl.project_hitter_action(state, shot_action)

    def legal_serve_hitter_mask(self, state: StageState) -> np.ndarray:
        return np.asarray(self._impl.legal_serve_hitter_mask(state), dtype=bool)

    def legal_hitter_mask(self, state: StageState) -> np.ndarray:
        return np.asarray(self._impl.legal_hitter_mask(state), dtype=bool)

    def _decode_continuous_hitter(self, action: Any, state: StageState) -> HitterActionDecode:
        values = np.asarray(action, dtype=float).reshape(-1)
        if values.size < 5:
            padded = np.zeros(5, dtype=float)
            padded[: values.size] = values
            values = padded
        values = np.clip(values[:5], -1.0, 1.0)
        unit = 0.5 * (values + 1.0)

        if self.config.court.lateral_motion_enabled:
            phi_grid = self._impl.phi_grid(state)
            phi = self._interp(float(unit[0]), float(phi_grid[0]), float(phi_grid[-1]))
            theta_grid = self._impl._theta_grid_for_phi(state, phi)
            theta = self._interp(float(unit[1]), float(theta_grid[0]), float(theta_grid[-1]))
            speed_grid = self._impl._speed_grid_for_phi_theta(state, phi, theta)
            speed = self._interp(float(unit[2]), float(speed_grid[0]), float(speed_grid[-1]))
            vh = float(speed * np.cos(theta))
            v_x = float(vh * np.cos(phi))
            v_y = float(vh * np.sin(phi))
            v_z = float(speed * np.sin(theta))
        else:
            vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, self.config)
            phi = np.pi / 2.0 if state.current_hitter == "left" else -np.pi / 2.0
            theta = self._interp(float(unit[1]), self.config.action.vz_min, self.config.action.vz_max)
            speed = self._interp(float(unit[2]), vy_low, vy_high)
            v_x = 0.0
            v_y = speed
            v_z = theta

        provisional = ShotAction(v_x=v_x, v_y=v_y, v_z=v_z, x_rec=0.0, y_rec=0.0)
        landing_x, landing_y = landing_position(state, provisional, self.config)
        x_rec, y_rec = self._decode_conditional_recovery(
            state,
            x_unit=float(unit[3]),
            y_unit=float(unit[4]),
            phi=float(phi),
            theta=float(theta),
            speed=float(speed),
            landing_x=float(landing_x),
            landing_y=float(landing_y),
        )
        shot_action = ShotAction(v_x=v_x, v_y=v_y, v_z=v_z, x_rec=x_rec, y_rec=y_rec)
        flat_index = int(self._impl.encode_hitter(shot_action, state))
        return HitterActionDecode(flat_index=flat_index, shot_action=shot_action)

    def _decode_mixed_hitter(self, action: Any, state: StageState) -> HitterActionDecode:
        values = np.asarray(action, dtype=float).reshape(-1)
        if values.size < 5:
            padded = np.zeros(5, dtype=float)
            padded[: values.size] = values
            values = padded
        values = np.clip(values[:5], -1.0, 1.0)
        phi_index = self._signed_to_index(float(values[0]), self._impl._effective_phi_bins)
        theta_index = self._signed_to_index(float(values[1]), self.discrete_config.theta_bins)
        speed_index = self._signed_to_index(float(values[2]), self.discrete_config.speed_bins)
        recovery_count = self._impl._effective_x_rec_bins * self.discrete_config.y_rec_bins
        velocity_flat_index = (
            ((phi_index * self.discrete_config.theta_bins + theta_index) * self.discrete_config.speed_bins + speed_index)
            * recovery_count
        )
        decoded_velocity = self._impl.decode_hitter(velocity_flat_index, state)
        (x_low, x_high), (y_low, y_high) = self._full_recovery_bounds(state.current_hitter)
        x_rec = self._interp_signed(float(values[3]), x_low, x_high)
        y_rec = self._interp_signed(float(values[4]), y_low, y_high)
        if not self.config.court.lateral_motion_enabled:
            x_rec = 0.0

        shot_action = replace(decoded_velocity.shot_action, x_rec=x_rec, y_rec=y_rec)
        flat_index = self._mixed_flat_index(
            phi_index=phi_index,
            theta_index=theta_index,
            speed_index=speed_index,
            x_rec=x_rec,
            y_rec=y_rec,
            state=state,
        )
        return HitterActionDecode(
            flat_index=flat_index,
            phi_index=phi_index,
            theta_index=theta_index,
            speed_index=speed_index,
            shot_action=shot_action,
        )

    def _mixed_flat_index(
        self,
        *,
        phi_index: int,
        theta_index: int,
        speed_index: int,
        x_rec: float,
        y_rec: float,
        state: StageState,
    ) -> int:
        (x_low, x_high), (y_low, y_high) = self._full_recovery_bounds(state.current_hitter)
        x_grid = self._impl._recovery_axis_grid(x_low, x_high, self._impl._effective_x_rec_bins)
        y_grid = self._impl._recovery_axis_grid(y_low, y_high, self.discrete_config.y_rec_bins)
        x_rec_index = self._impl._nearest_value_index(x_rec, x_grid)
        y_rec_index = self._impl._nearest_value_index(y_rec, y_grid)
        return (
            ((((phi_index * self.discrete_config.theta_bins) + theta_index) * self.discrete_config.speed_bins + speed_index)
             * self._impl._effective_x_rec_bins + x_rec_index)
            * self.discrete_config.y_rec_bins
            + y_rec_index
        )

    def _interp(self, unit_value: float, lower: float, upper: float) -> float:
        return float(lower + float(np.clip(unit_value, 0.0, 1.0)) * (upper - lower))

    def _interp_signed(self, signed_value: float, lower: float, upper: float) -> float:
        return self._interp(0.5 * (float(np.clip(signed_value, -1.0, 1.0)) + 1.0), lower, upper)

    def _signed_to_index(self, signed_value: float, count: int) -> int:
        if count <= 1:
            return 0
        unit = 0.5 * (float(np.clip(signed_value, -1.0, 1.0)) + 1.0)
        return int(np.clip(round(unit * (count - 1)), 0, count - 1))

    def _full_recovery_bounds(self, side: Side) -> tuple[tuple[float, float], tuple[float, float]]:
        return x_bounds(self.config, margin=0.0), side_y_bounds(side, self.config, net_margin=0.0, back_margin=0.0)

    def _decode_conditional_recovery(
        self,
        state: StageState,
        *,
        x_unit: float,
        y_unit: float,
        phi: float,
        theta: float,
        speed: float,
        landing_x: float,
        landing_y: float,
    ) -> tuple[float, float]:
        rec_bounds = recovery_bounds(state.current_hitter, self.config)
        x_anchor, y_anchor = self._conditional_recovery_anchor(
            state,
            phi=phi,
            theta=theta,
            speed=speed,
            landing_x=landing_x,
            landing_y=landing_y,
        )
        x_rec = self._interp_around_anchor(float(x_unit), rec_bounds[0][0], rec_bounds[0][1], x_anchor)
        y_rec = self._interp_around_anchor(float(y_unit), rec_bounds[1][0], rec_bounds[1][1], y_anchor)
        if not self.config.court.lateral_motion_enabled:
            x_rec = 0.0
        return x_rec, y_rec

    def _conditional_recovery_anchor(
        self,
        state: StageState,
        *,
        phi: float,
        theta: float,
        speed: float,
        landing_x: float,
        landing_y: float,
    ) -> tuple[float, float]:
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, self.config)
        receiver = opponent_side(state.current_hitter)
        (target_x_low, target_x_high), (target_y_low, target_y_high) = target_bounds_for_receiver(receiver, self.config)

        if not np.isfinite(landing_x):
            landing_x = 0.5 * (target_x_low + target_x_high)
        if not np.isfinite(landing_y):
            landing_y = 0.5 * (target_y_low + target_y_high)

        landing_x = float(np.clip(landing_x, target_x_low, target_x_high))
        landing_y = float(np.clip(landing_y, target_y_low, target_y_high))
        x_ratio = self._ratio(landing_x, target_x_low, target_x_high)
        if receiver == "right":
            depth_ratio = self._ratio(landing_y, target_y_low, target_y_high)
        else:
            depth_ratio = self._ratio(target_y_high - landing_y, 0.0, target_y_high - target_y_low)

        if self.config.court.lateral_motion_enabled:
            phi_grid = self._impl.phi_grid(state)
            phi_ratio = self._ratio(phi, float(phi_grid[0]), float(phi_grid[-1]))
            x_ratio = 0.85 * x_ratio + 0.15 * phi_ratio

        forward_sin = float(np.sin(phi)) if state.current_hitter == "left" else -float(np.sin(phi))
        forward_speed = float(speed * max(np.cos(theta), 0.0) * max(forward_sin, 0.0))
        speed_depth_ratio = self._ratio(
            forward_speed,
            self.config.action.vy_min_forward,
            self.config.action.vy_max_forward,
        )
        depth_ratio = 0.85 * depth_ratio + 0.15 * speed_depth_ratio

        x_anchor = rec_x_low + x_ratio * (rec_x_high - rec_x_low)
        depth_fraction = 0.35 + 0.4 * depth_ratio
        if state.current_hitter == "left":
            y_anchor = rec_y_low + depth_fraction * (rec_y_high - rec_y_low)
        else:
            y_anchor = rec_y_high - depth_fraction * (rec_y_high - rec_y_low)
        return (
            float(np.clip(x_anchor, rec_x_low, rec_x_high)),
            float(np.clip(y_anchor, rec_y_low, rec_y_high)),
        )

    def _interp_around_anchor(self, unit_value: float, lower: float, upper: float, anchor: float) -> float:
        unit_value = float(np.clip(unit_value, 0.0, 1.0))
        anchor = float(np.clip(anchor, lower, upper))
        if unit_value <= 0.5:
            return self._interp(unit_value * 2.0, lower, anchor)
        return self._interp((unit_value - 0.5) * 2.0, anchor, upper)

    def _ratio(self, value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.5
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    def _is_scalar_integer_action(self, action: Any) -> bool:
        if isinstance(action, (int, np.integer)):
            return True
        arr = np.asarray(action)
        return arr.shape == () and np.issubdtype(arr.dtype, np.integer)
