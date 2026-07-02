from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from badminton1d.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton1d.config import SimulationConfig
from badminton1d.dynamics import (
    PreparedShot,
    candidate_intercept_points,
    prepare_shot,
    reaction_miss_probability,
    step_stage,
)
from badminton1d.env import Badminton1DEnv
from badminton1d.match import MatchConfig, MatchScore, reset_for_serve, with_service_court_x_side
from badminton1d.movement import advance_player_toward
from badminton1d.obs import ObservationConfig, ObservationEncoder
from badminton1d.opponents import DecisionContext, HitterActionCandidate, OpponentPolicy, make_opponent
from badminton1d.reset_sampling import ResetSampler, ResetSamplingConfig
from badminton1d.reward_shaping import (
    ActionStreakTracker,
    AttackRewardConfig,
    DefensiveLiftRewardConfig,
    LoopPenaltyConfig,
    NetProximityRewardConfig,
    OpponentTravelRewardConfig,
    PressureRewardConfig,
    ReturnDepthRewardConfig,
    attack_reward_from_record,
    defensive_lift_reward_from_record,
    intercept_flight_ratio_reward_from_record,
    loop_penalty_for_streak,
    net_proximity_reward_from_record,
    opponent_travel_reward_from_record,
    pressure_reward_from_record,
    return_depth_reward_from_record,
)
from badminton1d.shot_generators import TacticRuntimeConfig
from badminton1d.state import ShotAction, Side, StageRecord, ValidatedShotAction
from badminton1d.utils import opponent_side, player_position, player_velocity


def _terminal_rewards(winner: Side | None) -> tuple[float, float]:
    if winner == "left":
        return 1.0, -1.0
    if winner == "right":
        return -1.0, 1.0
    return 0.0, 0.0


def _sparse_histogram_payload(prefix: str, hist: np.ndarray) -> dict[str, Any]:
    indices = np.flatnonzero(hist)
    counts = hist[indices]
    return {
        f"{prefix}_size": int(hist.size),
        f"{prefix}_indices": indices.astype(int).tolist(),
        f"{prefix}_counts": counts.astype(int).tolist(),
    }


RECOVERY_COUNTERFACTUAL_OTHER_SAMPLE_COUNT = 24
COUNTERFACTUAL_OPPONENT_RESPONSE_SAMPLES = 2


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
    opponent_travel_reward: OpponentTravelRewardConfig = field(default_factory=OpponentTravelRewardConfig)
    return_depth_reward: ReturnDepthRewardConfig = field(default_factory=ReturnDepthRewardConfig)
    net_proximity_reward: NetProximityRewardConfig = field(default_factory=NetProximityRewardConfig)
    attack_reward: AttackRewardConfig = field(default_factory=AttackRewardConfig)
    defensive_lift_reward: DefensiveLiftRewardConfig = field(default_factory=DefensiveLiftRewardConfig)
    feasible_pressure_reward_weight: float = 0.0
    no_feasible_intercept_bonus: float = 0.0
    opponent_intercept_continue_penalty: float = 0.0


@dataclass(frozen=True)
class RLEnvConfig:
    train_side: Side = "left"
    opponent_type: str = "safe"
    initial_server: str = "left"
    mirror_train_side: bool = False
    mirror_match_fraction: float = 0.0
    random_service_x: bool = False
    train_reaction_time: float = 0.0
    opponent_reaction_time: float = 0.0
    max_stages_per_rally: int = 30
    serve_z0: float = 1.15
    include_feasible_mask: bool = True
    include_reaction_risk_features: bool = True
    include_records_in_info: bool = False
    policy_type: str = "velocity_oriented"
    tactic_runtime: TacticRuntimeConfig = field(default_factory=TacticRuntimeConfig)
    reset_sampling: ResetSamplingConfig = field(default_factory=ResetSamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    recovery_counterfactual_other_sample_count: int = RECOVERY_COUNTERFACTUAL_OTHER_SAMPLE_COUNT
    counterfactual_opponent_response_samples: int = COUNTERFACTUAL_OPPONENT_RESPONSE_SAMPLES
    recovery_counterfactual_expected_response_target: bool = True
    recovery_full_diagnostics_probability: float = 0.0
    use_shot_cf: bool = False
    shot_cf_coef: float = 0.1
    shot_cf_top_m: int = 20
    shot_cf_num_modes: int = 3
    shot_cf_min_landing_dist: float = 1.0
    shot_cf_depth: int = 1
    shot_cf_include_chosen: bool = True
    shot_cf_skip_low_diversity: bool = True
    shot_cf_min_modes: int = 2
    shot_cf_value_detach: bool = True
    shot_cf_normalize: bool = True
    shot_cf_debug_log: bool = False


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
        self.action_mapper = DiscreteActionMapper(
            self.config,
            discrete_action_config,
            policy_type=self.rl_config.policy_type,
            tactic_runtime_config=self.rl_config.tactic_runtime,
        )
        self.observation_encoder = ObservationEncoder(
            self.config,
            ObservationConfig(
                max_score=self.match_config.target_score,
                max_stages_per_rally=self.match_config.max_stages_per_rally,
                include_feasible_mask=self.rl_config.include_feasible_mask,
                include_reaction_risk_features=self.rl_config.include_reaction_risk_features,
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
        self.pending_prepared_shot: PreparedShot | None = None
        self.records: list[StageRecord] = []
        self.last_episode_info: dict[str, Any] | None = None
        self._episode_hitter_hist = np.zeros(self.action_mapper.hitter_action_count, dtype=np.int64)
        self._episode_intercept_hist = np.zeros(self.config.action.intercept_count, dtype=np.int64)
        self._episode_tactic_zone_hist = np.zeros(len(self.action_mapper.tactic_zone_names), dtype=np.int64)
        self._episode_tactic_angle_hist = np.zeros(len(self.action_mapper.tactic_angle_names), dtype=np.int64)
        self._episode_tactic_power_hist = np.zeros(len(self.action_mapper.tactic_power_names), dtype=np.int64)
        self._episode_tactic_shot_hist = np.zeros(len(self.action_mapper.tactic_shot_names), dtype=np.int64)
        self._episode_tactic_lookup_valid_count = 0
        self._episode_tactic_lookup_fallback_count = 0
        self._episode_invalid_action_count = 0
        self._action_streak_tracker = ActionStreakTracker()
        self._episode_loop_penalty_total = 0.0
        self._episode_pressure_reward_total = 0.0
        self._episode_opponent_travel_reward_total = 0.0
        self._episode_return_depth_reward_total = 0.0
        self._episode_net_proximity_reward_total = 0.0
        self._episode_attack_reward_total = 0.0
        self._episode_defensive_lift_reward_total = 0.0
        self._episode_intercept_flight_ratio_reward_total = 0.0
        self._episode_feasible_pressure_reward_total = 0.0
        self._episode_no_feasible_intercept_bonus_total = 0.0
        self._episode_opponent_intercept_penalty_total = 0.0
        self._episode_defensive_return_reward_total = 0.0
        self._episode_serve_return_reward_total = 0.0
        self._episode_stage_penalty_total = 0.0
        self._episode_stall_penalty_total = 0.0
        self._episode_random_start = False
        self._episode_service_x_side: Side | None = None

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
        self._episode_tactic_zone_hist.fill(0)
        self._episode_tactic_angle_hist.fill(0)
        self._episode_tactic_power_hist.fill(0)
        self._episode_tactic_shot_hist.fill(0)
        self._episode_tactic_lookup_valid_count = 0
        self._episode_tactic_lookup_fallback_count = 0
        self._episode_invalid_action_count = 0
        self._action_streak_tracker.reset()
        self._episode_loop_penalty_total = 0.0
        self._episode_pressure_reward_total = 0.0
        self._episode_opponent_travel_reward_total = 0.0
        self._episode_return_depth_reward_total = 0.0
        self._episode_net_proximity_reward_total = 0.0
        self._episode_attack_reward_total = 0.0
        self._episode_defensive_lift_reward_total = 0.0
        self._episode_intercept_flight_ratio_reward_total = 0.0
        self._episode_feasible_pressure_reward_total = 0.0
        self._episode_no_feasible_intercept_bonus_total = 0.0
        self._episode_opponent_intercept_penalty_total = 0.0
        self._episode_defensive_return_reward_total = 0.0
        self._episode_serve_return_reward_total = 0.0
        self._episode_stage_penalty_total = 0.0
        self._episode_stall_penalty_total = 0.0
        self._episode_service_x_side = None
        self.pending_requested_action = None
        self.pending_applied_action = None
        self.pending_feasible_indices = []
        self.pending_prepared_shot = None
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
            initial_state = self._maybe_randomize_service_x(initial_state)
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
                self.pending_prepared_shot = None
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
        info.update(
            self._recovery_factorized_info(
                record,
                terminated=terminated,
            )
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
        decode = self.action_mapper.decode_hitter_for_agent(action, self.base_env.state, self.train_side)
        self._episode_hitter_hist[decode.flat_index] += 1
        self._record_tactic_decode(decode)
        streak = self._action_streak_tracker.observe(decode.flat_index)

        try:
            projected = self.action_mapper.project_hitter_action(self.base_env.state, decode.shot_action)
        except (RuntimeError, ValueError):
            self._episode_invalid_action_count += 1
            record = self._hitter_forfeit_record(self.base_env.state, winner=self.opponent_side, reason="train_no_valid_shot")
            return record, self._terminal_reward(record.next_state.winner)
        if projected.projected:
            self._episode_invalid_action_count += 1
        prepared = projected.prepared_shot
        validated = prepared.validated_action
        feasible = list(prepared.feasible_indices)
        chosen_index = self.opponent.choose_intercept_index(
            self.base_env.state,
            validated.applied,
            feasible,
            self.config,
            self.current_server,
            prepared_shot=prepared,
        )
        record = self.base_env.step(projected.shot_action, chosen_index, prepared_shot=prepared)

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
                config=self.config,
            )
            reward += pressure_reward
            self._episode_pressure_reward_total += pressure_reward
            opponent_travel_reward = opponent_travel_reward_from_record(
                record,
                weight=self.rl_config.reward.opponent_travel_reward.weight,
                config=self.config,
            )
            reward += opponent_travel_reward
            self._episode_opponent_travel_reward_total += opponent_travel_reward
            return_depth_reward = return_depth_reward_from_record(
                record,
                weight=self.rl_config.reward.return_depth_reward.weight,
                config=self.config,
            )
            reward += return_depth_reward
            self._episode_return_depth_reward_total += return_depth_reward
            net_proximity_reward = net_proximity_reward_from_record(
                record,
                weight=self.rl_config.reward.net_proximity_reward.weight,
                net_y=self.config.court.net_y,
                distance_threshold=self.rl_config.reward.net_proximity_reward.distance_threshold,
            )
            reward += net_proximity_reward
            self._episode_net_proximity_reward_total += net_proximity_reward
            attack_reward = attack_reward_from_record(
                record,
                config=self.config,
                reward_config=self.rl_config.reward.attack_reward,
            )
            reward += attack_reward
            self._episode_attack_reward_total += attack_reward
            defensive_lift_reward = defensive_lift_reward_from_record(
                record,
                config=self.config,
                reward_config=self.rl_config.reward.defensive_lift_reward,
            )
            reward += defensive_lift_reward
            self._episode_defensive_lift_reward_total += defensive_lift_reward
            intercept_flight_ratio_reward = intercept_flight_ratio_reward_from_record(
                record,
                config=self.config,
                reward_config=self.rl_config.reward.defensive_lift_reward,
            )
            reward += intercept_flight_ratio_reward
            self._episode_intercept_flight_ratio_reward_total += intercept_flight_ratio_reward
            if record.terminal_reason != "invalid_serve_target":
                intercept_count = max(int(self.config.action.intercept_count), 1)
                feasible_fraction = len(record.feasible_indices) / float(intercept_count)
                feasible_pressure_reward = (
                    float(self.rl_config.reward.feasible_pressure_reward_weight)
                    * max(0.0, min(1.0, 1.0 - feasible_fraction))
                )
                reward += feasible_pressure_reward
                self._episode_feasible_pressure_reward_total += feasible_pressure_reward
            if record.terminal_reason == "no_feasible_intercept":
                no_feasible_bonus = max(float(self.rl_config.reward.no_feasible_intercept_bonus), 0.0)
                reward += no_feasible_bonus
                self._episode_no_feasible_intercept_bonus_total += no_feasible_bonus
            elif not record.next_state.rally_done:
                intercept_penalty = max(float(self.rl_config.reward.opponent_intercept_continue_penalty), 0.0)
                reward -= intercept_penalty
                self._episode_opponent_intercept_penalty_total += intercept_penalty
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

        record = self.base_env.step(
            self.pending_requested_action,
            intercept_index,
            prepared_shot=self.pending_prepared_shot,
        )
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
        self.pending_prepared_shot = None
        return record, reward

    def _prepare_receiver_turn(self) -> StageRecord | None:
        state = self.base_env.state
        if state.current_hitter != self.opponent_side:
            raise RuntimeError("Expected opponent to be the current hitter before preparing a receiver turn.")

        try:
            requested = self.opponent.choose_hitter_action(state, self.config, self.current_server)
            prepared = self._take_opponent_prepared_hitter_shot(requested) or prepare_shot(state, requested, self.config)
        except (RuntimeError, ValueError):
            return self._opponent_forfeit_record(state)
        validated = prepared.validated_action
        feasible = list(prepared.feasible_indices)

        self.role = "receiver"
        self.pending_requested_action = requested
        self.pending_applied_action = validated.applied
        self.pending_feasible_indices = feasible
        self.pending_prepared_shot = prepared
        return None

    def _take_opponent_prepared_hitter_shot(self, action: ShotAction) -> PreparedShot | None:
        take_prepared = getattr(self.opponent, "take_prepared_hitter_shot", None)
        if not callable(take_prepared):
            return None
        prepared = take_prepared()
        if prepared is None or prepared.validated_action.applied != action:
            return None
        return prepared

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
            winner=winner,
            stage_index=state.stage_index + 1,
        )
        self.base_env.state = next_state
        self.role = "hitter"
        self.pending_requested_action = None
        self.pending_applied_action = None
        self.pending_feasible_indices = []
        self.pending_prepared_shot = None
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
            "score_left": int(self.score.left),
            "score_right": int(self.score.right),
            "opponent_label": self.opponent.label() if hasattr(self.opponent, "label") else self.rl_config.opponent_type,
            "feasible_indices": list(self.pending_feasible_indices),
            "randomized_start": self._episode_random_start,
            "service_x_side": self._episode_service_x_side,
            "policy_type": self.action_mapper.policy_type,
            "include_feasible_mask": bool(self.rl_config.include_feasible_mask),
            "include_reaction_risk_features": bool(self.rl_config.include_reaction_risk_features),
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
            prepared_shot=self.pending_prepared_shot,
        )

    def _recovery_factorized_info(self, record: StageRecord, *, terminated: bool) -> dict[str, Any]:
        """Mark train-hitter transitions whose recovery choice should get credit.

        Non-terminal recovery credit is computed from the real next observation
        returned by step(), after the opponent shot has been sampled and the
        receiver feasible mask is known. If that mask is already empty, the next
        receiver step is forced into a no-feasible-intercept loss, so expose the
        terminal target immediately for stronger recovery credit.
        """
        if record.state_before.current_hitter != self.train_side:
            return {}

        payload: dict[str, Any] = {"recovery_factorized_action": True}
        if self.rl_config.use_shot_cf:
            payload.update(self._shot_counterfactual_info(record, terminated=terminated))
        if terminated:
            payload["recovery_factorized_target"] = float(self._terminal_reward(record.next_state.winner))
        elif self.role == "receiver" and not self.pending_feasible_indices:
            payload["recovery_factorized_target"] = float(self.rl_config.reward.loss_reward)
            payload["recovery_factorized_no_feasible_intercept"] = True
        payload.update(self._recovery_counterfactual_info(record))
        return payload

    def _shot_counterfactual_info(self, record: StageRecord, *, terminated: bool) -> dict[str, Any]:
        if record.state_before.current_hitter != self.train_side:
            return {}
        payload: dict[str, Any] = {
            "shot_cf_action": True,
            "shot_cf_win_reward": float(self.rl_config.reward.win_reward),
            "shot_cf_loss_reward": float(self.rl_config.reward.loss_reward),
            "shot_cf_depth": int(self.rl_config.shot_cf_depth),
        }
        # PPO chooses diverse counterfactual shots from the live policy. The env
        # supplies the sampled opponent response context, so PPO can evaluate an
        # experimental one-response bootstrap without mutating this environment.
        if not terminated and self.role == "receiver" and self.pending_applied_action is not None:
            payload["shot_cf_opponent_response_action"] = self.pending_applied_action
        return payload

    def _recovery_counterfactual_info(self, record: StageRecord) -> dict[str, Any]:
        """Sample alternate recovery bins after the same shot and opponent reply."""
        if (
            record.state_before.current_hitter != self.train_side
            or record.next_state.rally_done
            or record.chosen_time is None
            or self.role != "receiver"
            or self.pending_applied_action is None
        ):
            return {}

        other_sample_count = max(int(self.rl_config.recovery_counterfactual_other_sample_count), 0)
        full_diagnostics_probability = float(self.rl_config.recovery_full_diagnostics_probability)
        if other_sample_count <= 0 and full_diagnostics_probability <= 0.0:
            return {}

        try:
            x_grid, y_grid = self.action_mapper._recovery_grid_for_shot_action(
                record.state_before,
                record.validated_action.applied,
            )
        except (AttributeError, ValueError, RuntimeError):
            return {}

        chosen_index = 0
        chosen_distance = float("inf")
        recovery_points: list[tuple[float, float]] = [
            (float(x_rec), float(y_rec))
            for x_rec in x_grid
            for y_rec in y_grid
        ]
        for flat_index, (x_rec, y_rec) in enumerate(recovery_points):
            distance = float(
                np.hypot(
                    x_rec - record.validated_action.applied.x_rec,
                    y_rec - record.validated_action.applied.y_rec,
                )
            )
            if distance < chosen_distance:
                chosen_distance = distance
                chosen_index = flat_index

        other_indices = [index for index in range(len(recovery_points)) if index != chosen_index]
        if len(other_indices) > other_sample_count:
            other_indices = self.rng.choice(
                np.asarray(other_indices, dtype=int),
                size=other_sample_count,
                replace=False,
            ).astype(int).tolist()
        sampled_indices = [chosen_index, *other_indices]

        sampled_observations, sampled_targets = self._recovery_counterfactual_observations(
            record,
            recovery_points,
            sampled_indices,
        )
        if sampled_observations.size == 0:
            return {}
        sampled_expected_observations = getattr(self, "_last_recovery_expected_observations", None)
        sampled_expected_miss_probabilities = getattr(self, "_last_recovery_expected_miss_probabilities", None)
        sampled_expected_no_miss_targets = getattr(self, "_last_recovery_expected_no_miss_targets", None)
        sampled_response_counts = getattr(self, "_last_recovery_response_counts", None)
        sampled_response_weights = getattr(self, "_last_recovery_response_weights", None)

        payload: dict[str, Any] = {
            "recovery_factorized_counterfactual_observations": sampled_observations,
            "recovery_factorized_counterfactual_targets": sampled_targets,
            "recovery_factorized_counterfactual_chosen_index": 0,
            "recovery_factorized_counterfactual_baseline_indices": list(range(1, len(sampled_indices))),
            "recovery_factorized_counterfactual_sampled_indices": sampled_indices,
            "recovery_factorized_counterfactual_chosen_flat_index": int(chosen_index),
            "recovery_factorized_counterfactual_x_bins": int(len(x_grid)),
            "recovery_factorized_counterfactual_y_bins": int(len(y_grid)),
            "recovery_factorized_counterfactual_sampled_other_count": int(len(sampled_indices) - 1),
            "recovery_factorized_counterfactual_expected_response_target": bool(
                self.rl_config.recovery_counterfactual_expected_response_target
            ),
            "recovery_factorized_counterfactual_loss_reward": float(self.rl_config.reward.loss_reward),
        }
        if isinstance(sampled_response_counts, np.ndarray) and sampled_response_counts.shape[0] == len(sampled_indices):
            payload["recovery_factorized_counterfactual_response_counts"] = sampled_response_counts
        if (
            isinstance(sampled_response_weights, np.ndarray)
            and sampled_response_weights.shape[0] == sampled_observations.shape[0]
        ):
            payload["recovery_factorized_counterfactual_response_weights"] = sampled_response_weights
        if (
            isinstance(sampled_expected_observations, np.ndarray)
            and sampled_expected_observations.shape == sampled_observations.shape
        ):
            payload["recovery_factorized_counterfactual_expected_observations"] = sampled_expected_observations
            payload["recovery_factorized_counterfactual_expected_miss_probabilities"] = (
                sampled_expected_miss_probabilities
            )
            payload["recovery_factorized_counterfactual_expected_no_miss_targets"] = (
                sampled_expected_no_miss_targets
            )
        if full_diagnostics_probability > 0.0 and float(self.rng.random()) < min(full_diagnostics_probability, 1.0):
            full_indices = list(range(len(recovery_points)))
            full_observations, full_targets = self._recovery_counterfactual_observations(
                record,
                recovery_points,
                full_indices,
            )
            full_expected_observations = getattr(self, "_last_recovery_expected_observations", None)
            full_expected_miss_probabilities = getattr(self, "_last_recovery_expected_miss_probabilities", None)
            full_expected_no_miss_targets = getattr(self, "_last_recovery_expected_no_miss_targets", None)
            full_response_counts = getattr(self, "_last_recovery_response_counts", None)
            full_response_weights = getattr(self, "_last_recovery_response_weights", None)
            if full_observations.size:
                payload["recovery_factorized_counterfactual_full_observations"] = full_observations
                payload["recovery_factorized_counterfactual_full_targets"] = full_targets
                if isinstance(full_response_counts, np.ndarray) and full_response_counts.shape[0] == len(full_indices):
                    payload["recovery_factorized_counterfactual_full_response_counts"] = full_response_counts
                if (
                    isinstance(full_response_weights, np.ndarray)
                    and full_response_weights.shape[0] == full_observations.shape[0]
                ):
                    payload["recovery_factorized_counterfactual_full_response_weights"] = full_response_weights
                if (
                    isinstance(full_expected_observations, np.ndarray)
                    and full_expected_observations.shape == full_observations.shape
                ):
                    payload["recovery_factorized_counterfactual_full_expected_observations"] = (
                        full_expected_observations
                    )
                    payload["recovery_factorized_counterfactual_full_expected_miss_probabilities"] = (
                        full_expected_miss_probabilities
                    )
                    payload["recovery_factorized_counterfactual_full_expected_no_miss_targets"] = (
                        full_expected_no_miss_targets
                    )
        return payload

    def _recovery_counterfactual_observations(
        self,
        record: StageRecord,
        recovery_points: list[tuple[float, float]],
        indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        observations: list[np.ndarray] = []
        targets: list[float] = []
        expected_observations: list[np.ndarray] = []
        expected_miss_probabilities: list[float] = []
        expected_no_miss_targets: list[float] = []
        response_counts: list[int] = []
        response_weights: list[float] = []
        use_expected_target = bool(self.rl_config.recovery_counterfactual_expected_response_target)
        self._last_recovery_expected_observations = None
        self._last_recovery_expected_miss_probabilities = None
        self._last_recovery_expected_no_miss_targets = None
        self._last_recovery_response_counts = None
        self._last_recovery_response_weights = None
        for point_index in indices:
            x_rec, y_rec = recovery_points[point_index]
            cf_state = self._state_after_counterfactual_recovery(
                record,
                x_rec=x_rec,
                y_rec=y_rec,
            )
            candidates = self._counterfactual_opponent_response_candidates(cf_state)
            response_counts.append(len(candidates))
            total_weight = float(sum(max(float(candidate.probability), 0.0) for candidate in candidates))
            if total_weight <= 0.0 or not np.isfinite(total_weight):
                candidate_weights = [1.0 / max(len(candidates), 1) for _ in candidates]
            else:
                candidate_weights = [max(float(candidate.probability), 0.0) / total_weight for candidate in candidates]

            for candidate, candidate_weight in zip(candidates, candidate_weights):
                prepared = candidate.prepared_shot or prepare_shot(cf_state, candidate.action, self.config)
                applied_action = prepared.validated_action.applied
                feasible = list(prepared.feasible_indices)
                receiver_observation = self.observation_encoder.encode(
                    state=cf_state,
                    agent_side=self.train_side,
                    role="receiver",
                    server_side=self.current_server,
                    score_left=self.score.left,
                    score_right=self.score.right,
                    pending_action=applied_action,
                    feasible_indices=feasible,
                    prepared_shot=prepared,
                )
                observations.append(receiver_observation)
                targets.append(float(self.rl_config.reward.loss_reward) if not feasible else float("nan"))
                response_weights.append(float(candidate_weight))
                if not use_expected_target:
                    continue

                expected_observation = receiver_observation
                miss_probability = 1.0
                no_miss_target = float("nan")
                if feasible:
                    intercept_index = self._lowest_reaction_risk_intercept_index(
                        cf_state,
                        applied_action,
                        feasible,
                        candidate_times=prepared.candidate_times,
                    )
                    if intercept_index is not None:
                        intercept_time = float(prepared.candidate_times[intercept_index])
                        miss_probability = reaction_miss_probability(intercept_time, self.config)
                        no_miss_record = step_stage(
                            cf_state,
                            applied_action,
                            intercept_index,
                            self.config,
                            enable_reaction_miss=False,
                            prepared_shot=prepared,
                        )
                        if no_miss_record.next_state.rally_done:
                            no_miss_target = float(self._terminal_reward(no_miss_record.next_state.winner))
                        else:
                            expected_observation = self.observation_encoder.encode(
                                state=no_miss_record.next_state,
                                agent_side=self.train_side,
                                role="hitter",
                                server_side=self.current_server,
                                score_left=self.score.left,
                                score_right=self.score.right,
                                pending_action=None,
                                feasible_indices=[],
                            )
                expected_observations.append(expected_observation)
                expected_miss_probabilities.append(float(miss_probability))
                expected_no_miss_targets.append(float(no_miss_target))
        if not observations:
            obs_size = self.observation_encoder.size
            return np.empty((0, obs_size), dtype=np.float32), np.empty((0,), dtype=np.float32)
        observation_array = np.asarray(observations, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.float32)
        if use_expected_target and len(expected_observations) == len(observations):
            self._last_recovery_expected_observations = np.asarray(expected_observations, dtype=np.float32)
            self._last_recovery_expected_miss_probabilities = np.asarray(
                expected_miss_probabilities,
                dtype=np.float32,
            )
            self._last_recovery_expected_no_miss_targets = np.asarray(
                expected_no_miss_targets,
                dtype=np.float32,
            )
        if response_counts:
            self._last_recovery_response_counts = np.asarray(response_counts, dtype=np.int32)
            self._last_recovery_response_weights = np.asarray(response_weights, dtype=np.float32)
        return observation_array, target_array

    def _counterfactual_opponent_response_candidates(self, state: Any) -> list[HitterActionCandidate]:
        assert self.pending_applied_action is not None
        count = max(int(self.rl_config.counterfactual_opponent_response_samples), 1)
        if count > 1:
            choose_candidates = getattr(self.opponent, "choose_likely_hitter_actions", None)
            if callable(choose_candidates):
                try:
                    candidates = choose_candidates(state, self.config, self.current_server, count=count)
                except (RuntimeError, ValueError, TypeError):
                    candidates = []
                valid_candidates = [
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, HitterActionCandidate) and candidate.action is not None
                ]
                if valid_candidates:
                    return valid_candidates[:count]
        return [
            HitterActionCandidate(
                flat_index=None,
                action=self.pending_applied_action,
                probability=1.0,
                prepared_shot=None,
            )
        ]

    def _lowest_reaction_risk_intercept_index(
        self,
        state: Any,
        action: ShotAction,
        feasible: list[int],
        *,
        candidate_times: np.ndarray | None = None,
    ) -> int | None:
        if not feasible:
            return None
        if candidate_times is None:
            candidate_times, _, _, _ = candidate_intercept_points(state, action, self.config)
        best_index = None
        best_key: tuple[float, float] | None = None
        for index in feasible:
            if not 0 <= index < len(candidate_times):
                continue
            intercept_time = float(candidate_times[index])
            key = (reaction_miss_probability(intercept_time, self.config), -intercept_time)
            if best_key is None or key < best_key:
                best_key = key
                best_index = int(index)
        return best_index

    def _state_after_counterfactual_recovery(
        self,
        record: StageRecord,
        *,
        x_rec: float,
        y_rec: float,
    ):
        motion = advance_player_toward(
            player_position(record.state_before, self.train_side),
            player_velocity(record.state_before, self.train_side),
            (float(x_rec), float(y_rec)),
            float(record.chosen_time),
            self.config,
            stop_when_early=True,
        )
        x_pos, y_pos = motion.position
        v_x, v_y = motion.velocity
        if self.train_side == "left":
            return replace(
                record.next_state,
                x_left=float(x_pos),
                y_left=float(y_pos),
                v_x_left=float(v_x),
                v_y_left=float(v_y),
            )
        return replace(
            record.next_state,
            x_right=float(x_pos),
            y_right=float(y_pos),
            v_x_right=float(v_x),
            v_y_right=float(v_y),
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

    def _maybe_randomize_service_x(self, state: StageState) -> StageState:
        if not self.rl_config.random_service_x or state.stage_index != 0 or state.rally_done:
            return state
        server_x_side: Side = "left" if bool(self.rng.integers(0, 2)) else "right"
        self._episode_service_x_side = server_x_side
        return with_service_court_x_side(state, self.config, server_x_side=server_x_side)

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
            "policy_type": self.action_mapper.policy_type,
            "tactic_zone_names": list(self.action_mapper.tactic_zone_names),
            "tactic_angle_names": list(self.action_mapper.tactic_angle_names),
            "tactic_power_names": list(self.action_mapper.tactic_power_names),
            "tactic_shot_names": list(self.action_mapper.tactic_shot_names),
            "tactic_zone_hist": self._episode_tactic_zone_hist.astype(int).tolist(),
            "tactic_angle_hist": self._episode_tactic_angle_hist.astype(int).tolist(),
            "tactic_power_hist": self._episode_tactic_power_hist.astype(int).tolist(),
            "tactic_shot_hist": self._episode_tactic_shot_hist.astype(int).tolist(),
            "tactic_lookup_valid_count": int(self._episode_tactic_lookup_valid_count),
            "tactic_lookup_fallback_count": int(self._episode_tactic_lookup_fallback_count),
            "server": self.current_server,
            "loop_penalty_total": float(self._episode_loop_penalty_total),
            "pressure_reward_total": float(self._episode_pressure_reward_total),
            "opponent_travel_reward_total": float(self._episode_opponent_travel_reward_total),
            "return_depth_reward_total": float(self._episode_return_depth_reward_total),
            "net_proximity_reward_total": float(self._episode_net_proximity_reward_total),
            "attack_reward_total": float(self._episode_attack_reward_total),
            "defensive_lift_reward_total": float(self._episode_defensive_lift_reward_total),
            "intercept_flight_ratio_reward_total": float(self._episode_intercept_flight_ratio_reward_total),
            "feasible_pressure_reward_total": float(self._episode_feasible_pressure_reward_total),
            "no_feasible_intercept_bonus_total": float(self._episode_no_feasible_intercept_bonus_total),
            "opponent_intercept_penalty_total": float(self._episode_opponent_intercept_penalty_total),
            "defensive_return_reward_total": float(self._episode_defensive_return_reward_total),
            "serve_return_reward_total": float(self._episode_serve_return_reward_total),
            "stage_penalty_total": float(self._episode_stage_penalty_total),
            "stall_penalty_total": float(self._episode_stall_penalty_total),
            "randomized_start": bool(self._episode_random_start),
        }
        metrics.update(_sparse_histogram_payload("hitter_action_hist", self._episode_hitter_hist))
        metrics.update(_sparse_histogram_payload("intercept_hist", self._episode_intercept_hist))
        metrics.update(self._action_streak_tracker.summary())
        return metrics

    def _record_tactic_decode(self, decode) -> None:
        if decode.landing_zone_index is not None and 0 <= decode.landing_zone_index < self._episode_tactic_zone_hist.size:
            self._episode_tactic_zone_hist[decode.landing_zone_index] += 1
        if decode.angle_bin_index is not None and 0 <= decode.angle_bin_index < self._episode_tactic_angle_hist.size:
            self._episode_tactic_angle_hist[decode.angle_bin_index] += 1
        if decode.power_bin_index is not None and 0 <= decode.power_bin_index < self._episode_tactic_power_hist.size:
            self._episode_tactic_power_hist[decode.power_bin_index] += 1
        if decode.tactic_shot_name is not None and self._episode_tactic_shot_hist.size:
            try:
                shot_index = self.action_mapper.tactic_shot_names.index(decode.tactic_shot_name)
            except ValueError:
                shot_index = -1
            if shot_index >= 0:
                self._episode_tactic_shot_hist[shot_index] += 1
        if decode.lookup_valid:
            self._episode_tactic_lookup_valid_count += 1
        if decode.lookup_fallback_used:
            self._episode_tactic_lookup_fallback_count += 1
