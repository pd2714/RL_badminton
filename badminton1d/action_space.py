from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import landing_position, validate_and_clip_shot_action, vy_bounds_for_hitter
from badminton1d.state import ShotAction, StageState
from badminton1d.utils import opponent_side, recovery_bounds, service_target_bounds_for_receiver


@dataclass(frozen=True)
class DiscreteActionConfig:
    v_x_bins: int = 11
    v_y_bins: int = 15
    v_z_bins: int = 11
    x_rec_bins: int = 5
    y_rec_bins: int = 5

    def __post_init__(self) -> None:
        if min(self.v_x_bins, self.v_y_bins, self.v_z_bins, self.x_rec_bins, self.y_rec_bins) <= 0:
            raise ValueError("All discrete action bin counts must be positive.")


@dataclass(frozen=True)
class HitterActionDecode:
    flat_index: int
    v_x_index: int
    v_y_index: int
    v_z_index: int
    x_rec_index: int
    y_rec_index: int
    shot_action: ShotAction


@dataclass(frozen=True)
class ProjectedHitterAction:
    shot_action: ShotAction
    projected: bool


class DiscreteActionMapper:
    """Maps a single discrete action id to hitter or receiver decisions."""

    def __init__(
        self,
        config: SimulationConfig,
        discrete_config: DiscreteActionConfig | None = None,
    ) -> None:
        self.config = config
        self.discrete_config = discrete_config or DiscreteActionConfig()
        self._effective_v_x_bins = self.discrete_config.v_x_bins if self.config.court.lateral_motion_enabled else 1
        self._effective_x_rec_bins = self.discrete_config.x_rec_bins if self.config.court.lateral_motion_enabled else 1

        self.hitter_action_count = (
            self._effective_v_x_bins
            * self.discrete_config.v_y_bins
            * self.discrete_config.v_z_bins
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
        v_z_index = rem % self.discrete_config.v_z_bins
        rem //= self.discrete_config.v_z_bins
        v_y_index = rem % self.discrete_config.v_y_bins
        v_x_index = rem // self.discrete_config.v_y_bins

        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, self.config)
        vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, self.config)
        vx_low, vx_high = self._vx_range()

        shot_action = ShotAction(
            v_x=self._linspace_value(vx_low, vx_high, v_x_index, self._effective_v_x_bins),
            v_y=self._linspace_value(vy_low, vy_high, v_y_index, self.discrete_config.v_y_bins),
            v_z=self._linspace_value(self.config.action.vz_min, self.config.action.vz_max, v_z_index, self.discrete_config.v_z_bins),
            x_rec=self._linspace_value(rec_x_low, rec_x_high, x_rec_index, self._effective_x_rec_bins),
            y_rec=self._linspace_value(rec_y_low, rec_y_high, y_rec_index, self.discrete_config.y_rec_bins),
        )
        return HitterActionDecode(
            flat_index=flat_index,
            v_x_index=v_x_index,
            v_y_index=v_y_index,
            v_z_index=v_z_index,
            x_rec_index=x_rec_index,
            y_rec_index=y_rec_index,
            shot_action=shot_action,
        )

    def encode_hitter(self, shot_action: ShotAction, state: StageState) -> int:
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, self.config)
        vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, self.config)
        vx_low, vx_high = self._vx_range()

        v_x_index = self._nearest_index(shot_action.v_x, vx_low, vx_high, self._effective_v_x_bins)
        v_y_index = self._nearest_index(shot_action.v_y, vy_low, vy_high, self.discrete_config.v_y_bins)
        v_z_index = self._nearest_index(shot_action.v_z, self.config.action.vz_min, self.config.action.vz_max, self.discrete_config.v_z_bins)
        x_rec_index = self._nearest_index(shot_action.x_rec, rec_x_low, rec_x_high, self._effective_x_rec_bins)
        y_rec_index = self._nearest_index(shot_action.y_rec, rec_y_low, rec_y_high, self.discrete_config.y_rec_bins)

        return (
            ((((v_x_index * self.discrete_config.v_y_bins) + v_y_index) * self.discrete_config.v_z_bins + v_z_index)
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
        try:
            validated = validate_and_clip_shot_action(state, clipped, self.config)
            if self._is_legal_serve_action(state, validated.applied):
                return ProjectedHitterAction(shot_action=validated.applied, projected=not self._actions_close(shot_action, validated.applied))
            if state.stage_index == 0:
                from badminton1d.agents import SafeHitter

                safe_action = SafeHitter().choose_action(state, self.config)
                safe_validated = validate_and_clip_shot_action(state, safe_action, self.config)
                return ProjectedHitterAction(shot_action=safe_validated.applied, projected=True)
        except ValueError:
            if state.stage_index == 0:
                from badminton1d.agents import SafeHitter

                safe_action = SafeHitter().choose_action(state, self.config)
                safe_validated = validate_and_clip_shot_action(state, safe_action, self.config)
                return ProjectedHitterAction(shot_action=safe_validated.applied, projected=True)

        vx_low, vx_high = self._vx_range()
        vx_grid = np.linspace(vx_low, vx_high, self._effective_v_x_bins)
        vy_grid = np.linspace(vy_low, vy_high, self.discrete_config.v_y_bins)
        vz_grid = np.linspace(self.config.action.vz_min, self.config.action.vz_max, self.discrete_config.v_z_bins)
        x_rec_grid = np.linspace(rec_x_low, rec_x_high, self._effective_x_rec_bins)
        y_rec_grid = np.linspace(rec_y_low, rec_y_high, self.discrete_config.y_rec_bins)
        vx_order = np.argsort(np.abs(vx_grid - clipped.v_x))
        vy_order = np.argsort(np.abs(vy_grid - clipped.v_y))
        vz_order = np.argsort(np.abs(vz_grid - clipped.v_z))
        x_rec_order = np.argsort(np.abs(x_rec_grid - clipped.x_rec))
        y_rec_order = np.argsort(np.abs(y_rec_grid - clipped.y_rec))
        for y_rec_index in y_rec_order:
            for x_rec_index in x_rec_order:
                for vx_index in vx_order:
                    for vy_index in vy_order:
                        for vz_index in vz_order:
                            candidate = ShotAction(
                                v_x=float(vx_grid[int(vx_index)]),
                                v_y=float(vy_grid[int(vy_index)]),
                                v_z=float(vz_grid[int(vz_index)]),
                                x_rec=float(x_rec_grid[int(x_rec_index)]),
                                y_rec=float(y_rec_grid[int(y_rec_index)]),
                            )
                            try:
                                validated = validate_and_clip_shot_action(state, candidate, self.config)
                                if not self._is_legal_serve_action(state, validated.applied):
                                    continue
                                return ProjectedHitterAction(shot_action=validated.applied, projected=True)
                            except ValueError:
                                continue

        from badminton1d.agents import SafeHitter

        safe_action = SafeHitter().choose_action(state, self.config)
        validated = validate_and_clip_shot_action(state, safe_action, self.config)
        return ProjectedHitterAction(shot_action=validated.applied, projected=True)

    def vx_grid(self) -> np.ndarray:
        vx_low, vx_high = self._vx_range()
        return np.linspace(vx_low, vx_high, self._effective_v_x_bins)

    def vy_grid(self, state: StageState) -> np.ndarray:
        return np.linspace(*vy_bounds_for_hitter(state.current_hitter, self.config), self.discrete_config.v_y_bins)

    def vz_grid(self) -> np.ndarray:
        return np.linspace(self.config.action.vz_min, self.config.action.vz_max, self.discrete_config.v_z_bins)

    def x_rec_grid(self, state: StageState) -> np.ndarray:
        return np.linspace(*recovery_bounds(state.current_hitter, self.config)[0], self._effective_x_rec_bins)

    def _vx_range(self) -> tuple[float, float]:
        if self.config.court.lateral_motion_enabled:
            return self.config.action.vx_min, self.config.action.vx_max
        return 0.0, 0.0

    def y_rec_grid(self, state: StageState) -> np.ndarray:
        return np.linspace(*recovery_bounds(state.current_hitter, self.config)[1], self.discrete_config.y_rec_bins)

    def legal_serve_hitter_mask(self, state: StageState) -> np.ndarray:
        if state.stage_index != 0:
            return np.ones(self.hitter_action_count, dtype=bool)

        mask = np.zeros(self.hitter_action_count, dtype=bool)
        for action_index in range(self.hitter_action_count):
            decoded = self.decode_hitter(action_index, state).shot_action
            try:
                validated = validate_and_clip_shot_action(state, decoded, self.config)
            except ValueError:
                continue
            if self._is_legal_serve_action(state, validated.applied):
                mask[action_index] = True
        return mask

    def _is_legal_serve_action(self, state: StageState, action: ShotAction) -> bool:
        if state.stage_index != 0:
            return True
        receiver = opponent_side(state.current_hitter)
        x_bounds, y_bounds = service_target_bounds_for_receiver(receiver, self.config)
        landing_x, landing_y = landing_position(state, action, self.config)
        return x_bounds[0] <= landing_x <= x_bounds[1] and y_bounds[0] <= landing_y <= y_bounds[1]

    def _nearest_index(self, value: float, lower: float, upper: float, count: int) -> int:
        if count == 1:
            return 0
        grid = np.linspace(lower, upper, count)
        return int(np.argmin(np.abs(grid - value)))

    def _actions_close(self, left: ShotAction, right: ShotAction) -> bool:
        return (
            np.isclose(left.v_x, right.v_x)
            and np.isclose(left.v_y, right.v_y)
            and np.isclose(left.v_z, right.v_z)
            and np.isclose(left.x_rec, right.x_rec)
            and np.isclose(left.y_rec, right.y_rec)
        )
