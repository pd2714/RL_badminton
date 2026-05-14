from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from badminton1d.agents import GreedyReceiver, RandomValidHitter
from badminton1d.config import SimulationConfig
from badminton1d.curricula import DefensiveBackcourtCurriculumConfig, DefensiveBackcourtCurriculumSampler
from badminton1d.dynamics import feasible_intercept_indices
from badminton1d.env import Badminton1DEnv
from badminton1d.match import MatchConfig, default_start_positions, reset_for_serve, service_receive_position
from badminton1d.state import Side, StageState
from badminton1d.utils import side_y_bounds, x_bounds


@dataclass(frozen=True)
class ResetSamplingConfig:
    random_start_prob: float = 0.0
    midrally_start_prob: float = 0.0
    opponent_serve_start_prob: float = 0.0
    position_perturb_scale: float = 0.3
    min_hit_height: float = 0.9
    max_hit_height: float = 1.6
    max_seeded_stages: int = 4
    defensive_backcourt_curriculum: DefensiveBackcourtCurriculumConfig | None = None


class ResetSampler:
    def __init__(
        self,
        *,
        sim_config: SimulationConfig,
        match_config: MatchConfig,
        sampling_config: ResetSamplingConfig,
        seed: int | None = None,
    ) -> None:
        self.sim_config = sim_config
        self.match_config = match_config
        self.sampling_config = sampling_config
        self.rng = np.random.default_rng(seed)
        self._random_hitter = RandomValidHitter(seed=None if seed is None else seed + 101)
        self._receiver = GreedyReceiver(mode="earliest")
        self._defensive_curriculum = None
        if self.sampling_config.defensive_backcourt_curriculum is not None:
            self._defensive_curriculum = DefensiveBackcourtCurriculumSampler(
                sim_config=self.sim_config,
                curriculum_config=self.sampling_config.defensive_backcourt_curriculum,
                seed=None if seed is None else seed + 202,
            )

    def sample_initial_state(
        self,
        *,
        default_server: Side,
        forced_server: Side | None = None,
        train_side: Side | None = None,
    ) -> tuple[StageState, bool, Side]:
        if self._defensive_curriculum is not None:
            if train_side is None:
                raise ValueError("train_side is required when a defensive curriculum is enabled.")
            state, sampled_server = self._defensive_curriculum.sample_initial_state(train_side=train_side)
            return state, True, sampled_server

        base_server = forced_server or default_server
        if self.sampling_config.random_start_prob <= 0.0:
            return reset_for_serve(base_server, self.sim_config, self.match_config), False, base_server

        if float(self.rng.random()) >= self.sampling_config.random_start_prob:
            return reset_for_serve(base_server, self.sim_config, self.match_config), False, base_server

        sampled_server: Side = forced_server if forced_server is not None else ("left" if bool(self.rng.integers(0, 2)) else "right")
        if self.sampling_config.midrally_start_prob > 0.0 and float(self.rng.random()) < self.sampling_config.midrally_start_prob:
            state = self._sample_seeded_midrally_state(sampled_server)
            if state is not None:
                return state, True, sampled_server
        return self._sample_randomized_serve_state(sampled_server), True, sampled_server

    def _sample_randomized_serve_state(self, server: Side) -> StageState:
        base_state = reset_for_serve(server, self.sim_config, self.match_config)
        (left_start_x, left_start_y), (right_start_x, right_start_y) = default_start_positions(self.sim_config)
        if server == "left":
            right_start_x, right_start_y = service_receive_position("right", self.sim_config)
        else:
            left_start_x, left_start_y = service_receive_position("left", self.sim_config)
        x_low, x_high = x_bounds(self.sim_config)
        left_y_low, left_y_high = side_y_bounds("left", self.sim_config)
        right_y_low, right_y_high = side_y_bounds("right", self.sim_config)

        left_x = float(
            np.clip(
                left_start_x + self.rng.uniform(-self.sampling_config.position_perturb_scale, self.sampling_config.position_perturb_scale),
                x_low,
                x_high,
            )
        )
        right_x = float(
            np.clip(
                right_start_x + self.rng.uniform(-self.sampling_config.position_perturb_scale, self.sampling_config.position_perturb_scale),
                x_low,
                x_high,
            )
        )
        left_y = float(
            np.clip(
                left_start_y + self.rng.uniform(-self.sampling_config.position_perturb_scale, self.sampling_config.position_perturb_scale),
                left_y_low,
                left_y_high,
            )
        )
        right_y = float(
            np.clip(
                right_start_y + self.rng.uniform(-self.sampling_config.position_perturb_scale, self.sampling_config.position_perturb_scale),
                right_y_low,
                right_y_high,
            )
        )
        if server == "left":
            x0, y0 = left_x, left_y
        else:
            x0, y0 = right_x, right_y
        z0 = float(self.rng.uniform(self.sampling_config.min_hit_height, self.sampling_config.max_hit_height))
        return StageState(
            x_left=left_x,
            y_left=left_y,
            x_right=right_x,
            y_right=right_y,
            current_hitter=server,
            x0=x0,
            y0=y0,
            z0=z0,
            rally_done=False,
            winner=None,
            stage_index=0,
        )

    def _sample_seeded_midrally_state(self, server: Side) -> StageState | None:
        for _ in range(10):
            env = Badminton1DEnv(config=self.sim_config)
            env.reset(self._sample_randomized_serve_state(server))
            rollout_length = int(self.rng.integers(1, max(self.sampling_config.max_seeded_stages, 2)))
            valid_state: StageState | None = None
            for _ in range(rollout_length):
                state = env.state
                if state.rally_done:
                    break
                action = self._random_hitter.choose_action(state, self.sim_config)
                feasible_indices = feasible_intercept_indices(state, action, self.sim_config)
                feasible = self._receiver.choose_intercept_index(
                    state,
                    action,
                    feasible_indices,
                    self.sim_config,
                )
                record = env.step(action, feasible)
                if record.next_state.rally_done:
                    valid_state = None
                    break
                valid_state = record.next_state
            if valid_state is not None:
                return valid_state
        return None
