from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import candidate_intercept_points, validate_and_clip_shot_action, vy_bounds_for_hitter
from badminton1d.state import ShotAction, StageState
from badminton1d.trajectory import ballistic_landing_time
from badminton1d.utils import (
    opponent_side,
    player_position,
    recovery_bounds,
    service_target_bounds_for_receiver,
    side_center_y,
    target_bounds_for_receiver,
)


def _target_bounds_for_stage(state: StageState, receiver: str, config: SimulationConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    if state.stage_index == 0:
        return service_target_bounds_for_receiver(receiver, config)
    return target_bounds_for_receiver(receiver, config)


class HitterPolicy(Protocol):
    def choose_action(self, state: StageState, config: SimulationConfig) -> ShotAction:
        ...


class ReceiverPolicy(Protocol):
    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
    ) -> int | None:
        ...


def _shot_from_landing_target(
    state: StageState,
    *,
    landing_x: float,
    landing_y: float,
    v_z: float,
    x_rec: float,
    y_rec: float,
    config: SimulationConfig,
) -> ShotAction:
    flight_time = ballistic_landing_time(state.z0, v_z, config.action.gravity)
    return ShotAction(
        v_x=(landing_x - state.x0) / flight_time,
        v_y=(landing_y - state.y0) / flight_time,
        v_z=v_z,
        x_rec=x_rec,
        y_rec=y_rec,
    )


@dataclass
class RandomValidHitter:
    seed: int | None = None
    max_attempts: int = 500
    fallback_x_count: int = 11
    fallback_y_count: int = 11
    fallback_vz_count: int = 21
    fallback_rec_count: int = 5
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def choose_action(self, state: StageState, config: SimulationConfig) -> ShotAction:
        if config.action.uses_square_drag:
            return self._choose_action_from_velocity_space(state, config)

        receiver = opponent_side(state.current_hitter)
        (target_x_low, target_x_high), (target_y_low, target_y_high) = _target_bounds_for_stage(state, receiver, config)
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, config)

        for _ in range(self.max_attempts):
            landing_x = float(self.rng.uniform(target_x_low, target_x_high))
            landing_y = float(self.rng.uniform(target_y_low, target_y_high))
            v_z = float(self.rng.uniform(config.action.vz_min, config.action.vz_max))
            x_rec = float(self.rng.uniform(rec_x_low, rec_x_high))
            y_rec = float(self.rng.uniform(rec_y_low, rec_y_high))
            action = _shot_from_landing_target(
                state,
                landing_x=landing_x,
                landing_y=landing_y,
                v_z=v_z,
                x_rec=x_rec,
                y_rec=y_rec,
                config=config,
            )
            try:
                validate_and_clip_shot_action(state, action, config)
                return action
            except ValueError:
                continue

        fallback = self._grid_search_action(state, config)
        if fallback is not None:
            return fallback
        raise RuntimeError("RandomValidHitter could not sample a valid action.")

    def _choose_action_from_velocity_space(self, state: StageState, config: SimulationConfig) -> ShotAction:
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, config)
        vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, config)

        for _ in range(self.max_attempts):
            action = ShotAction(
                v_x=float(self.rng.uniform(config.action.vx_min, config.action.vx_max)),
                v_y=float(self.rng.uniform(vy_low, vy_high)),
                v_z=float(self.rng.uniform(config.action.vz_min, config.action.vz_max)),
                x_rec=float(self.rng.uniform(rec_x_low, rec_x_high)),
                y_rec=float(self.rng.uniform(rec_y_low, rec_y_high)),
            )
            try:
                validate_and_clip_shot_action(state, action, config)
                return action
            except ValueError:
                continue

        fallback = self._grid_search_velocity_action(state, config)
        if fallback is not None:
            return fallback
        raise RuntimeError("RandomValidHitter could not sample a valid drag-square action.")

    def _grid_search_action(self, state: StageState, config: SimulationConfig) -> ShotAction | None:
        receiver = opponent_side(state.current_hitter)
        (target_x_low, target_x_high), (target_y_low, target_y_high) = _target_bounds_for_stage(state, receiver, config)
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, config)
        target_x_grid = np.linspace(target_x_low, target_x_high, self.fallback_x_count)
        target_y_grid = np.linspace(target_y_low, target_y_high, self.fallback_y_count)
        vz_grid = np.linspace(config.action.vz_min, config.action.vz_max, self.fallback_vz_count)
        rec_x_grid = np.linspace(rec_x_low, rec_x_high, self.fallback_rec_count)
        rec_y_grid = np.linspace(rec_y_low, rec_y_high, self.fallback_rec_count)

        preferred_y = target_y_high if receiver == "right" else target_y_low
        target_y_order = np.argsort(np.abs(target_y_grid - preferred_y))
        target_x_order = np.argsort(np.abs(target_x_grid - 0.0))
        preferred_vz = 0.65 * (config.action.vz_max - config.action.vz_min) + config.action.vz_min
        vz_order = np.argsort(np.abs(vz_grid - preferred_vz))
        rec_x_order = np.argsort(np.abs(rec_x_grid - 0.0))
        rec_y_order = np.argsort(np.abs(rec_y_grid - side_center_y(state.current_hitter, config)))

        for rec_y_index in rec_y_order:
            for rec_x_index in rec_x_order:
                for vz_index in vz_order:
                    for target_y_index in target_y_order:
                        for target_x_index in target_x_order:
                            action = _shot_from_landing_target(
                                state,
                                landing_x=float(target_x_grid[int(target_x_index)]),
                                landing_y=float(target_y_grid[int(target_y_index)]),
                                v_z=float(vz_grid[int(vz_index)]),
                                x_rec=float(rec_x_grid[int(rec_x_index)]),
                                y_rec=float(rec_y_grid[int(rec_y_index)]),
                                config=config,
                            )
                            try:
                                return validate_and_clip_shot_action(state, action, config).applied
                            except ValueError:
                                continue
        return None

    def _grid_search_velocity_action(self, state: StageState, config: SimulationConfig) -> ShotAction | None:
        receiver = opponent_side(state.current_hitter)
        receiver_x, _ = player_position(state, receiver)
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, config)
        vy_low, vy_high = vy_bounds_for_hitter(state.current_hitter, config)

        vx_grid = np.linspace(config.action.vx_min, config.action.vx_max, self.fallback_x_count)
        vy_grid = np.linspace(vy_low, vy_high, self.fallback_y_count)
        vz_grid = np.linspace(config.action.vz_min, config.action.vz_max, self.fallback_vz_count)
        rec_x_grid = np.linspace(rec_x_low, rec_x_high, self.fallback_rec_count)
        rec_y_grid = np.linspace(rec_y_low, rec_y_high, self.fallback_rec_count)

        preferred_vx = config.action.vx_min if receiver_x >= 0.0 else config.action.vx_max
        vx_order = np.argsort(np.abs(vx_grid - preferred_vx))
        preferred_vy = vy_high if state.current_hitter == "left" else vy_low
        vy_order = np.argsort(np.abs(vy_grid - preferred_vy))
        preferred_vz = config.action.vz_min + 0.7 * (config.action.vz_max - config.action.vz_min)
        vz_order = np.argsort(np.abs(vz_grid - preferred_vz))
        rec_x_order = np.argsort(np.abs(rec_x_grid - 0.0))
        rec_y_order = np.argsort(np.abs(rec_y_grid - side_center_y(state.current_hitter, config)))

        for rec_y_index in rec_y_order:
            for rec_x_index in rec_x_order:
                for vz_index in vz_order:
                    for vy_index in vy_order:
                        for vx_index in vx_order:
                            action = ShotAction(
                                v_x=float(vx_grid[int(vx_index)]),
                                v_y=float(vy_grid[int(vy_index)]),
                                v_z=float(vz_grid[int(vz_index)]),
                                x_rec=float(rec_x_grid[int(rec_x_index)]),
                                y_rec=float(rec_y_grid[int(rec_y_index)]),
                            )
                            try:
                                return validate_and_clip_shot_action(state, action, config).applied
                            except ValueError:
                                continue
        return None


@dataclass
class SafeHitter:
    preferred_vz_ratio: float = 0.72
    depth_ratio: float = 0.82
    width_bias: float = 0.8
    recovery_x_ratio: float = 0.5
    recovery_y_ratio: float = 0.45

    def choose_action(self, state: StageState, config: SimulationConfig) -> ShotAction:
        receiver = opponent_side(state.current_hitter)
        receiver_x, receiver_y = player_position(state, receiver)
        (target_x_low, target_x_high), (target_y_low, target_y_high) = _target_bounds_for_stage(state, receiver, config)
        (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(state.current_hitter, config)

        if receiver_x >= 0.0:
            landing_x = target_x_low + (1.0 - self.width_bias) * (target_x_high - target_x_low)
        else:
            landing_x = target_x_low + self.width_bias * (target_x_high - target_x_low)

        if receiver == "right":
            landing_y = target_y_low + self.depth_ratio * (target_y_high - target_y_low)
        else:
            landing_y = target_y_high - self.depth_ratio * (target_y_high - target_y_low)

        x_rec = rec_x_low + self.recovery_x_ratio * (rec_x_high - rec_x_low)
        y_rec = rec_y_low + self.recovery_y_ratio * (rec_y_high - rec_y_low)
        v_z = config.action.vz_min + self.preferred_vz_ratio * (config.action.vz_max - config.action.vz_min)
        action = _shot_from_landing_target(
            state,
            landing_x=float(landing_x),
            landing_y=float(landing_y),
            v_z=float(v_z),
            x_rec=float(x_rec),
            y_rec=float(y_rec),
            config=config,
        )
        try:
            return validate_and_clip_shot_action(state, action, config).applied
        except ValueError:
            return RandomValidHitter(seed=0).choose_action(state, config)


@dataclass
class GreedyReceiver:
    mode: str = "earliest"

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
    ) -> int | None:
        if not feasible_indices:
            return None
        if self.mode == "earliest":
            return feasible_indices[0]
        if self.mode == "highest":
            _, _, _, zs = candidate_intercept_points(state, action, config)
            best_index = feasible_indices[0]
            best_height = float(zs[best_index])
            for index in feasible_indices[1:]:
                height = float(zs[index])
                if height > best_height:
                    best_index = index
                    best_height = height
            return best_index
        raise ValueError(f"Unsupported receiver mode: {self.mode}")


@dataclass
class StageAgent:
    name: str
    hitter_policy: HitterPolicy
    receiver_policy: ReceiverPolicy
    reaction_time: float = 0.0

    def choose_shot_action(self, state: StageState, config: SimulationConfig) -> ShotAction:
        return self.hitter_policy.choose_action(state, config)

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
    ) -> int | None:
        return self.receiver_policy.choose_intercept_index(state, action, feasible_indices, config)
