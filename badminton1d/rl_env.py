from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from badminton1d.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton1d.config import SimulationConfig
from badminton1d.dynamics import feasible_intercept_indices, validate_and_clip_shot_action
from badminton1d.env import Badminton1DEnv
from badminton1d.match import MatchConfig, MatchScore, reset_for_serve
from badminton1d.obs import ObservationConfig, ObservationEncoder
from badminton1d.opponents import DecisionContext, OpponentPolicy, make_opponent
from badminton1d.reset_sampling import ResetSampler, ResetSamplingConfig
from badminton1d.reward_shaping import (
    ActionStreakTracker,
    LoopPenaltyConfig,
    PressureRewardConfig,
    loop_penalty_for_streak,
    pressure_reward_from_record,
)
from badminton1d.state import ShotAction, Side, StageRecord, ValidatedShotAction
from badminton1d.utils import opponent_side, player_position


def _terminal_rewards(winner: Side | None) -> tuple[float, float]:
    if winner == "left":
        return 1.0, -1.0
    if winner == "right":
        return -1.0, 1.0
    return 0.0, 0.0


@dataclass(frozen=True)
class RewardConfig:
    win_reward: float = 1.0
    loss_reward: float = -1.0
    h_clipped_penalty: float = 0.01
    invalid_receiver_penalty: float = 0.0
    defensive_return_reward: float = 0.0
    serve_return_reward: float = 0.0
    max_rally_penalty: float = 0.0
    stage_penalty: float = 0.0
    stall_penalty: float = 0.0
    stall_penalty_start: int = 0
    loop_penalty: LoopPenaltyConfig = field(default_factory=LoopPenaltyConfig)
    pressure_reward: PressureRewardConfig = field(default_factory=PressureRewardConfig)


@dataclass(frozen=True)
class RLEnvConfig:
    train_side: Side = "left"
    opponent_type: str = "safe"
    initial_server: str = "left"
    mirror_train_side: bool = False
    mirror_match_fraction: float = 0.0
    train_reaction_time: float = 0.0
    opponent_reaction_time: float = 0.0
    max_stages_per_rally: int = 30
    serve_z0: float = 1.15
    include_feasible_mask: bool = True
    include_records_in_info: bool = False
    reset_sampling: ResetSamplingConfig = field(default_factory=ResetSamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


class BadmintonRLEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SimulationConfig | None = None,
        rl_config: RLEnvConfig | None = None,
        discrete_action_config: DiscreteActionConfig | None = None,
        opponent: OpponentPolicy | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SimulationConfig()
        self.rl_config = rl_config or RLEnvConfig()
        self.match_config = MatchConfig(
            target_score=1,
            max_stages_per_rally=self.rl_config.max_stages_per_rally,
            serve_z0=self.rl_config.serve_z0,
        )
        self.base_env = Badminton1DEnv(config=self.config)
        self.train_side = self.rl_config.train_side
        self.opponent_side = opponent_side(self.train_side)
        self.action_mapper = DiscreteActionMapper(self.config, discrete_action_config)
        self.observation_encoder = ObservationEncoder(
            self.config,
            ObservationConfig(
                max_score=self.match_config.target_score,
                max_stages_per_rally=self.match_config.max_stages_per_rally,
                include_feasible_mask=self.rl_config.include_feasible_mask,
            ),
        )
        self.action_space = self.action_mapper.action_space
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_encoder.size,),
            dtype=np.float32,
        )

        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.opponent = opponent or make_opponent(self.rl_config.opponent_type, seed=seed)
        self.reset_sampler = ResetSampler(
            sim_config=self.config,
            match_config=self.match_config,
            sampling_config=self.rl_config.reset_sampling,
            seed=seed,
        )

        self.current_server: Side = "left"
        self.score = MatchScore()
        self.role: str = "hitter"
        self.pending_requested_action: ShotAction | None = None
        self.pending_applied_action: ShotAction | None = None
        self.pending_feasible_indices: list[int] = []
        self.records: list[StageRecord] = []
        self.last_episode_info: dict[str, Any] | None = None
        self._episode_hitter_hist = np.zeros(self.action_mapper.hitter_action_count, dtype=np.int64)
        self._episode_intercept_hist = np.zeros(self.config.action.intercept_count, dtype=np.int64)
        self._episode_invalid_action_count = 0
        self._action_streak_tracker = ActionStreakTracker()
        self._episode_loop_penalty_total = 0.0
        self._episode_pressure_reward_total = 0.0
        self._episode_defensive_return_reward_total = 0.0
        self._episode_serve_return_reward_total = 0.0
        self._episode_stage_penalty_total = 0.0
        self._episode_stall_penalty_total = 0.0
        self._episode_random_start = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        self._set_train_side(self._resolve_train_side(options.get("train_side")))
        self.current_server = self._resolve_server(options.get("server"))
        self.score = MatchScore()
        self.records = []
        self.last_episode_info = None
        self._episode_hitter_hist.fill(0)
        self._episode_intercept_hist.fill(0)
        self._episode_invalid_action_count = 0
        self._action_streak_tracker.reset()
        self._episode_loop_penalty_total = 0.0
        self._episode_pressure_reward_total = 0.0
        self._episode_defensive_return_reward_total = 0.0
        self._episode_serve_return_reward_total = 0.0
        self._episode_stage_penalty_total = 0.0
        self._episode_stall_penalty_total = 0.0
        self.pending_requested_action = None
        self.pending_applied_action = None
        self.pending_feasible_indices = []
        forced_server: Side | None = None
        opponent_serve_prob = float(self.rl_config.reset_sampling.opponent_serve_start_prob)
        if opponent_serve_prob > 0.0 and float(self.rng.random()) < opponent_serve_prob:
            forced_server = self.opponent_side

        for _ in range(8):
            initial_state, was_randomized, sampled_server = self.reset_sampler.sample_initial_state(
                default_server=self.current_server,
                forced_server=forced_server,
                train_side=self.train_side,
            )
            self.current_server = sampled_server
            self._episode_random_start = was_randomized
            initial_state = self._apply_reaction_times(initial_state)
            self.base_env.reset(initial_state)
            if hasattr(self.opponent, "on_episode_start"):
                self.opponent.on_episode_start(
                    train_side=self.train_side,
                    opponent_side=self.opponent_side,
                    server_side=self.current_server,
                    config=self.config,
                )

            if self.base_env.state.current_hitter == self.train_side:
                self.role = "hitter"
                break

            if self._prepare_receiver_turn() is None:
                break
        else:
            raise RuntimeError("Failed to initialize a playable receiver turn after repeated reset attempts.")

        observation = self._get_obs()
        return observation, self._base_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.base_env.state.rally_done:
            raise ValueError("Cannot step a finished episode. Call reset().")

        reward = 0.0
        truncated = False

        if self.role == "hitter":
            record, step_reward = self._step_hitter(action)
            reward += step_reward
        elif self.role == "receiver":
            record, step_reward = self._step_receiver(action)
            reward += step_reward
        else:
            raise ValueError(f"Unsupported role: {self.role}")

        self.records.append(record)

        stage_penalty = max(float(self.rl_config.reward.stage_penalty), 0.0)
        if stage_penalty > 0.0:
            reward -= stage_penalty
            self._episode_stage_penalty_total += stage_penalty

        stall_penalty = max(float(self.rl_config.reward.stall_penalty), 0.0)
        stall_start = max(int(self.rl_config.reward.stall_penalty_start), 0)
        if stall_penalty > 0.0 and stall_start > 0 and len(self.records) >= stall_start:
            reward -= stall_penalty
            self._episode_stall_penalty_total += stall_penalty

        if len(self.records) >= self.match_config.max_stages_per_rally and not record.next_state.rally_done:
            truncated = True
            reward -= self.rl_config.reward.max_rally_penalty

        terminated = bool(record.next_state.rally_done)
        if terminated:
            reward += self._terminal_reward(record.next_state.winner)
        elif not truncated:
            if self.base_env.state.current_hitter == self.train_side:
                self.role = "hitter"
                self.pending_requested_action = None
                self.pending_applied_action = None
                self.pending_feasible_indices = []
            else:
                forfeit_record = self._prepare_receiver_turn()
                if forfeit_record is not None:
                    self.records.append(forfeit_record)
                    record = forfeit_record
                    terminated = True
                    reward += self._terminal_reward(record.next_state.winner)

        observation = self._get_obs()
        info = self._base_info()
        info.update(
            {
                "last_record": record,
                "action_role": self.role if not terminated and not truncated else None,
            }
        )

        if terminated or truncated:
            metrics = self._episode_metrics(record, truncated=truncated)
            info["badminton_metrics"] = metrics
            if self.rl_config.include_records_in_info:
                info["episode_records"] = list(self.records)
            self.last_episode_info = info

        return observation, float(reward), terminated, truncated, info

    def current_decision_context(self) -> DecisionContext:
        return DecisionContext(
            state=self.base_env.state,
            role=self.role,
            pending_action=self.pending_applied_action,
            feasible_indices=list(self.pending_feasible_indices),
            receiver_action_count=self.action_mapper.receiver_action_count,
        )

    def _step_hitter(self, action: int) -> tuple[StageRecord, float]:
        decode = self.action_mapper.decode_hitter(action, self.base_env.state)
        self._episode_hitter_hist[decode.flat_index] += 1
        streak = self._action_streak_tracker.observe(decode.flat_index)

        try:
            projected = self.action_mapper.project_hitter_action(self.base_env.state, decode.shot_action)
        except (RuntimeError, ValueError):
            self._episode_invalid_action_count += 1
            record = self._hitter_forfeit_record(self.base_env.state, winner=self.opponent_side, reason="train_no_valid_shot")
            return record, self._terminal_reward(record.next_state.winner)
        if projected.projected:
            self._episode_invalid_action_count += 1
        try:
            validated = validate_and_clip_shot_action(self.base_env.state, projected.shot_action, self.config)
        except ValueError:
            self._episode_invalid_action_count += 1
            record = self._hitter_forfeit_record(self.base_env.state, winner=self.opponent_side, reason="train_no_valid_shot")
            return record, self._terminal_reward(record.next_state.winner)
        feasible = feasible_intercept_indices(self.base_env.state, validated.applied, self.config)
        chosen_index = self.opponent.choose_intercept_index(
            self.base_env.state,
            validated.applied,
            feasible,
            self.config,
            self.current_server,
        )
        record = self.base_env.step(projected.shot_action, chosen_index)

        reward = 0.0
        if projected.projected:
            reward -= self.rl_config.reward.h_clipped_penalty
        if chosen_index is not None and 0 <= chosen_index < self.config.action.intercept_count:
            self._episode_intercept_hist[chosen_index] += 1
        loop_penalty = loop_penalty_for_streak(streak, self.rl_config.reward.loop_penalty)
        reward += loop_penalty
        self._episode_loop_penalty_total += loop_penalty
        pressure_reward = 0.0
        if record.state_before.current_hitter == self.train_side:
            pressure_reward = pressure_reward_from_record(
                record,
                weight=self.rl_config.reward.pressure_reward.weight,
                z_min=self.config.player.z_min,
                z_max=self.config.player.z_max,
            )
            reward += pressure_reward
            self._episode_pressure_reward_total += pressure_reward
        return record, reward

    def _step_receiver(self, action: int) -> tuple[StageRecord, float]:
        intercept_index = self.action_mapper.decode_receiver(action)
        reward = 0.0
        already_invalid = False
        legal_return = False
        if not 0 <= intercept_index < self.config.action.intercept_count:
            self._episode_invalid_action_count += 1
            reward -= self.rl_config.reward.invalid_receiver_penalty
            already_invalid = True
        else:
            self._episode_intercept_hist[intercept_index] += 1

        if self.pending_requested_action is None:
            raise RuntimeError("Receiver step requested without a pending opponent action.")

        record = self.base_env.step(self.pending_requested_action, intercept_index)
        if intercept_index not in record.feasible_indices and not already_invalid:
            self._episode_invalid_action_count += 1
            reward -= self.rl_config.reward.invalid_receiver_penalty
        else:
            legal_return = not already_invalid

        if legal_return and not record.next_state.rally_done:
            defensive_reward = float(self.rl_config.reward.defensive_return_reward)
            if defensive_reward > 0.0:
                reward += defensive_reward
                self._episode_defensive_return_reward_total += defensive_reward
            is_serve_return = (
                record.state_before.stage_index == 0
                and record.state_before.current_hitter == self.opponent_side
                and self.current_server == self.opponent_side
            )
            if is_serve_return:
                serve_return_reward = float(self.rl_config.reward.serve_return_reward)
                if serve_return_reward > 0.0:
                    reward += serve_return_reward
                    self._episode_serve_return_reward_total += serve_return_reward

        self.pending_requested_action = None
        self.pending_applied_action = None
        self.pending_feasible_indices = []
        return record, reward

    def _prepare_receiver_turn(self) -> StageRecord | None:
        state = self.base_env.state
        if state.current_hitter != self.opponent_side:
            raise RuntimeError("Expected opponent to be the current hitter before preparing a receiver turn.")

        try:
            requested = self.opponent.choose_hitter_action(state, self.config, self.current_server)
            validated = validate_and_clip_shot_action(state, requested, self.config)
        except (RuntimeError, ValueError):
            return self._opponent_forfeit_record(state)
        feasible = feasible_intercept_indices(state, validated.applied, self.config)

        self.role = "receiver"
        self.pending_requested_action = requested
        self.pending_applied_action = validated.applied
        self.pending_feasible_indices = feasible
        return None

    def _opponent_forfeit_record(self, state: Any) -> StageRecord:
        return self._hitter_forfeit_record(state, winner=self.train_side, reason="opponent_no_valid_shot")

    def _hitter_forfeit_record(self, state: Any, *, winner: Side, reason: str) -> StageRecord:
        current_hitter_x, current_hitter_y = player_position(state, state.current_hitter)
        placeholder = ShotAction(
            v_x=0.0,
            v_y=0.0,
            v_z=0.0,
            x_rec=current_hitter_x,
            y_rec=current_hitter_y,
        )
        next_state = replace(
            state,
            rally_done=True,
            winner=self.train_side,
            stage_index=state.stage_index + 1,
        )
        self.base_env.state = next_state
        self.role = "hitter"
        self.pending_requested_action = None
        self.pending_applied_action = None
        self.pending_feasible_indices = []
        reward_left, reward_right = _terminal_rewards(next_state.winner)
        return StageRecord(
            stage_index=state.stage_index,
            state_before=state,
            validated_action=ValidatedShotAction(
                requested=placeholder,
                applied=placeholder,
                projected=True,
            ),
            receiver_side=self.train_side,
            candidate_times=np.asarray([], dtype=float),
            feasible_indices=[],
            chosen_index=None,
            chosen_time=None,
            intercept_point=None,
            next_state=next_state,
            reward_left=reward_left,
            reward_right=reward_right,
            terminal_reason=reason,
            notes=[f"{state.current_hitter.capitalize()} had no valid shot and forfeited the rally."],
        )

    def _terminal_reward(self, winner: Side | None) -> float:
        if winner == self.train_side:
            return self.rl_config.reward.win_reward
        return self.rl_config.reward.loss_reward

    def _base_info(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "server": self.current_server,
            "train_side": self.train_side,
            "opponent_label": self.opponent.label() if hasattr(self.opponent, "label") else self.rl_config.opponent_type,
            "feasible_indices": list(self.pending_feasible_indices),
            "randomized_start": self._episode_random_start,
        }

    def _get_obs(self) -> np.ndarray:
        return self.observation_encoder.encode(
            state=self.base_env.state,
            agent_side=self.train_side,
            role=self.role,
            server_side=self.current_server,
            score_left=self.score.left,
            score_right=self.score.right,
            pending_action=self.pending_applied_action,
            feasible_indices=self.pending_feasible_indices,
        )

    def _resolve_server(self, server_value: str | None) -> Side:
        resolved = (server_value or self.rl_config.initial_server).lower()
        if resolved in {"left", "right"}:
            return resolved
        if resolved == "train":
            return self.train_side
        if resolved == "opponent":
            return self.opponent_side
        if resolved == "random":
            return "left" if bool(self.rng.integers(0, 2)) else "right"
        raise ValueError(f"Unsupported initial server: {resolved}")

    def _resolve_train_side(self, train_side_value: str | None) -> Side:
        if train_side_value is not None:
            resolved = train_side_value.lower()
            if resolved not in {"left", "right"}:
                raise ValueError(f"Unsupported train side: {train_side_value}")
            return resolved
        mirror_fraction = float(self.rl_config.mirror_match_fraction)
        if not 0.0 <= mirror_fraction <= 1.0:
            raise ValueError(f"mirror_match_fraction must be in [0, 1], got {mirror_fraction}")
        if self.rl_config.mirror_train_side and mirror_fraction <= 0.0:
            mirror_fraction = 0.5
        if mirror_fraction > 0.0 and float(self.rng.random()) < mirror_fraction:
            return opponent_side(self.rl_config.train_side)
        return self.rl_config.train_side

    def _set_train_side(self, side: Side) -> None:
        self.train_side = side
        self.opponent_side = opponent_side(self.train_side)

    def refresh_opponent_pool(self) -> None:
        if hasattr(self.opponent, "refresh"):
            self.opponent.refresh()

    def _apply_reaction_times(self, state: StageState) -> StageState:
        left_reaction = self.rl_config.train_reaction_time if self.train_side == "left" else self.rl_config.opponent_reaction_time
        right_reaction = self.rl_config.train_reaction_time if self.train_side == "right" else self.rl_config.opponent_reaction_time
        return replace(
            state,
            reaction_time_left=float(left_reaction),
            reaction_time_right=float(right_reaction),
        )

    def _episode_metrics(self, record: StageRecord, *, truncated: bool) -> dict[str, Any]:
        self._action_streak_tracker.finalize()
        rally_won = record.next_state.winner == self.train_side
        total_decisions = int(self._episode_hitter_hist.sum() + self._episode_intercept_hist.sum())
        invalid_rate = self._episode_invalid_action_count / max(float(total_decisions), 1.0)
        metrics = {
            "winner": record.next_state.winner,
            "rally_won": float(rally_won),
            "rally_length": len(self.records),
            "terminal_reason": "max_stages_exceeded" if truncated else record.terminal_reason,
            "invalid_action_count": int(self._episode_invalid_action_count),
            "invalid_action_rate": float(invalid_rate),
            "hitter_action_hist": self._episode_hitter_hist.astype(int).tolist(),
            "intercept_hist": self._episode_intercept_hist.astype(int).tolist(),
            "server": self.current_server,
            "loop_penalty_total": float(self._episode_loop_penalty_total),
            "pressure_reward_total": float(self._episode_pressure_reward_total),
            "defensive_return_reward_total": float(self._episode_defensive_return_reward_total),
            "serve_return_reward_total": float(self._episode_serve_return_reward_total),
            "stage_penalty_total": float(self._episode_stage_penalty_total),
            "stall_penalty_total": float(self._episode_stall_penalty_total),
            "randomized_start": bool(self._episode_random_start),
        }
        metrics.update(self._action_streak_tracker.summary())
        return metrics
