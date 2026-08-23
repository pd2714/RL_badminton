from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from badminton.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton.config import SimulationConfig
from badminton.dynamics import PreparedShot
from badminton.evaluation import (
    ModelSelector,
    adapt_observation_to_model,
    choose_model_action,
    evaluate_selector,
    rollout_episode,
    summarize_episodes,
)
from badminton.obs import ObservationConfig, ObservationEncoder
from badminton.opponents import DecisionContext, HitterActionCandidate, OpponentPolicy, make_opponent
from badminton.playback import build_rally_trace, rally_trace_to_dict
from badminton.reset_sampling import ResetSamplingConfig
from badminton.rl_env import (
    COUNTERFACTUAL_OPPONENT_RESPONSE_SAMPLES,
    RECOVERY_COUNTERFACTUAL_OTHER_SAMPLE_COUNT,
    BadmintonRLEnv,
    RLEnvConfig,
    RewardConfig,
)
from badminton.shot_generators import TacticRuntimeConfig
from badminton.state import ShotAction, Side, StageState
from badminton.utils import ensure_directory
from badminton.video import TrainingProgressSample, export_rally_video, export_training_progress_video

_STEP_PATTERN = re.compile(r"(\d+)")
_PREFIX_COMPATIBLE_KEYS = {
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.value_net.0.weight",
}


def replace_with_existing_file(source_path: Path, target_path: Path) -> None:
    ensure_directory(target_path.parent)
    tmp_path = target_path.with_name(f".{target_path.name}.tmp")
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass
    try:
        os.link(source_path, tmp_path)
    except OSError:
        shutil.copy2(source_path, tmp_path)
    tmp_path.replace(target_path)


def _model_action_value(action: Any) -> Any:
    arr = np.asarray(action)
    if arr.shape == ():
        return int(arr)
    return arr.astype(np.float32, copy=False)


def _top_hitter_action_indices(
    model: PPO,
    observation: np.ndarray,
    *,
    hitter_action_count: int,
    count: int,
    batch_size: int = 2048,
) -> list[tuple[int, float]]:
    if hitter_action_count <= 0 or count <= 0:
        return []
    observation = adapt_observation_to_model(model, observation)
    actions = np.arange(int(hitter_action_count), dtype=np.int64)
    log_probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, actions.size, batch_size):
            batch_actions = actions[start : start + batch_size]
            batch_obs = np.repeat(observation.reshape(1, -1), batch_actions.size, axis=0).astype(np.float32)
            obs_tensor, _ = model.policy.obs_to_tensor(batch_obs)
            action_tensor = torch.as_tensor(batch_actions, dtype=torch.long, device=model.device)
            _, batch_log_prob, _ = model.policy.evaluate_actions(obs_tensor, action_tensor)
            log_probs.append(batch_log_prob.detach().cpu().numpy().reshape(-1))
    log_prob_array = np.concatenate(log_probs).astype(float, copy=False)
    finite = np.isfinite(log_prob_array)
    if not np.any(finite):
        predicted, _ = model.predict(observation, deterministic=True)
        return [(int(_model_action_value(predicted)), 1.0)]
    finite_indices = np.flatnonzero(finite)
    finite_scores = log_prob_array[finite_indices]
    order = np.argsort(finite_scores)[::-1][: max(int(count), 1)]
    selected = finite_indices[order]
    selected_scores = finite_scores[order]
    weights = np.exp(selected_scores - float(np.max(selected_scores)))
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        probabilities = np.full(weights.shape[0], 1.0 / max(weights.shape[0], 1), dtype=float)
    else:
        probabilities = weights / total
    return [(int(index), float(probability)) for index, probability in zip(selected, probabilities)]


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    match = _STEP_PATTERN.findall(path.stem)
    step = int(match[-1]) if match else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return step, mtime, path.name


def _is_flat_action_head_mismatch(error: RuntimeError) -> bool:
    message = str(error)
    return (
        "Error(s) in loading state_dict for MaskedBadmintonPolicy" in message
        and "action_net.weight" in message
        and "phi_head.weight" in message
    )


def _copy_policy_state_compatibly(model: PPO, source_state: dict[str, Any]) -> None:
    target_state = model.policy.state_dict()
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            continue
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            continue
        if (
            key in _PREFIX_COMPATIBLE_KEYS
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
        ):
            if source_value.shape[1] >= target_value.shape[1]:
                target_state[key] = source_value[:, : target_value.shape[1]]
            elif target_value.shape[1] - source_value.shape[1] == 4 and target_value.shape[1] <= 53:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, :29] = source_value[:, :29]
                expanded[:, 33:] = source_value[:, 29:]
                target_state[key] = expanded
            elif target_value.shape[1] - source_value.shape[1] == 4:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, : source_value.shape[1]] = source_value
                target_state[key] = expanded
            elif target_value.shape[1] - source_value.shape[1] == 8:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, :29] = source_value[:, :29]
                expanded[:, 33:-4] = source_value[:, 29:]
                target_state[key] = expanded
    model.policy.load_state_dict(target_state, strict=True)


def _load_ppo_with_compatible_policy_state(path: Path) -> PPO:
    data, params, pytorch_variables = load_from_zip_file(path, device="auto")
    if data is None or params is None:
        raise RuntimeError(f"No PPO data found in checkpoint: {path}")
    policy_state = params.get("policy")
    if policy_state is None:
        raise RuntimeError(f"Checkpoint has no policy parameters: {path}")

    if "policy_kwargs" in data and "device" in data["policy_kwargs"]:
        del data["policy_kwargs"]["device"]
    for key in {"observation_space", "action_space"}:
        if key in data:
            data[key] = _convert_space(data[key])

    model = PPO(
        policy=data["policy_class"],
        env=data.get("env"),
        device="auto",
        _init_setup_model=False,
    )
    model.__dict__.update(data)
    model._setup_model()
    _copy_policy_state_compatibly(model, policy_state)

    if pytorch_variables is not None:
        for name, variable in pytorch_variables.items():
            if variable is None:
                continue
            recursive_setattr(model, f"{name}.data", variable.data)
    if model.use_sde:
        model.policy.reset_noise()
    return model


def _load_ppo_for_selfplay(path: Path) -> PPO:
    try:
        return PPO.load(path)
    except RuntimeError as error:
        if not _is_flat_action_head_mismatch(error):
            raise
        return _load_ppo_with_compatible_policy_state(path)


@dataclass
class CheckpointPool:
    checkpoint_dir: Path
    pool_size: int
    sampling_mode: str = "uniform"
    recency_power: float = 2.0
    base_checkpoint_path: Path | None = None
    recent_fraction: float = 0.5
    seed: int | None = None
    max_cached_models: int | None = None
    cached_models: dict[Path, PPO] = field(default_factory=dict, init=False)
    checkpoints: list[Path] = field(default_factory=list, init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        normalized = self.sampling_mode.strip().lower()
        if normalized not in {"uniform", "random", "recency", "linear_recency", "newest"}:
            raise ValueError(f"Unsupported checkpoint sampling mode: {self.sampling_mode}")
        self.sampling_mode = normalized
        if self.recency_power <= 0.0:
            raise ValueError("recency_power must be positive.")
        if not 0.0 < self.recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in (0, 1].")
        if self.max_cached_models is not None and self.max_cached_models < 0:
            raise ValueError("max_cached_models must be zero or greater.")
        self.rng = np.random.default_rng(self.seed)
        self.refresh()

    def refresh(self) -> None:
        ensure_directory(self.checkpoint_dir)
        snapshots = sorted(self.checkpoint_dir.glob("*.zip"), key=_checkpoint_sort_key, reverse=True)
        self.checkpoints = []
        seen: set[Path] = set()

        for path in snapshots[: self.pool_size]:
            resolved = path.resolve()
            if resolved in seen:
                continue
            self.checkpoints.append(resolved)
            seen.add(resolved)

        if self.base_checkpoint_path is not None and self.base_checkpoint_path.exists():
            base = self.base_checkpoint_path.resolve()
            if base not in seen:
                self.checkpoints.append(base)
                seen.add(base)

        self.cached_models = {
            path: model
            for path, model in self.cached_models.items()
            if any(path == checkpoint for checkpoint in self.checkpoints)
        }

    def prune(self) -> None:
        snapshots = sorted(self.checkpoint_dir.glob("*.zip"), key=_checkpoint_sort_key, reverse=True)
        for stale_path in snapshots[self.pool_size :]:
            try:
                stale_resolved = stale_path.resolve()
                self.cached_models.pop(stale_resolved, None)
                stale_path.unlink()
            except FileNotFoundError:
                continue
        self.refresh()

    def sample_path(self) -> Path:
        if not self.checkpoints:
            raise RuntimeError("Checkpoint pool is empty.")
        return self._sample_from_paths(self.checkpoints)

    def newest_path(self) -> Path | None:
        if not self.checkpoints:
            return None
        return self.checkpoints[0]

    def oldest_path(self) -> Path | None:
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def recent_checkpoints(self) -> list[Path]:
        if not self.checkpoints:
            return []
        count = max(1, int(np.ceil(len(self.checkpoints) * self.recent_fraction)))
        return list(self.checkpoints[:count])

    def sample_recent_path(self, *, exclude_newest: bool = False) -> Path | None:
        paths = self.recent_checkpoints()
        if exclude_newest:
            paths = paths[1:]
        if not paths:
            return None
        return self._sample_from_paths(paths)

    def older_checkpoints(self) -> list[Path]:
        if not self.checkpoints:
            return []
        count = max(1, int(np.ceil(len(self.checkpoints) * self.recent_fraction)))
        return list(self.checkpoints[count:])

    def sample_weighted_path(self, *, recent_weight: float, older_weight: float) -> tuple[Path, str]:
        recent = self.recent_checkpoints()
        older = self.older_checkpoints()
        buckets: list[tuple[str, list[Path], float]] = []
        if recent:
            buckets.append(("recent", recent, float(max(recent_weight, 0.0))))
        if older:
            buckets.append(("older", older, float(max(older_weight, 0.0))))
        if not buckets:
            raise RuntimeError("Checkpoint pool is empty.")
        total_weight = sum(weight for _, _, weight in buckets)
        if total_weight <= 0.0:
            index = int(self.rng.integers(len(buckets)))
            bucket_name, paths, _ = buckets[index]
        else:
            probs = np.asarray([weight / total_weight for _, _, weight in buckets], dtype=np.float64)
            index = int(self.rng.choice(len(buckets), p=probs))
            bucket_name, paths, _ = buckets[index]
        path = self._sample_from_paths(paths)
        return path, bucket_name

    def _sample_from_paths(self, paths: list[Path]) -> Path:
        if not paths:
            raise RuntimeError("Checkpoint pool is empty.")
        if self.sampling_mode == "newest":
            return paths[0]
        if self.sampling_mode in {"uniform", "random"}:
            return paths[int(self.rng.integers(len(paths)))]
        positions = np.arange(len(paths), 0, -1, dtype=np.float64)
        if self.sampling_mode == "linear_recency":
            weights = positions
        else:
            weights = np.power(positions, float(self.recency_power))
        probs = weights / weights.sum()
        index = int(self.rng.choice(len(paths), p=probs))
        return paths[index]

    def load_model(self, path: Path) -> PPO:
        resolved = path.resolve()
        model = self.cached_models.get(resolved)
        if model is not None:
            self.cached_models.pop(resolved)
            self.cached_models[resolved] = model
            return model

        model = _load_ppo_for_selfplay(resolved)
        if self.max_cached_models != 0:
            self.cached_models[resolved] = model
            if self.max_cached_models is not None:
                while len(self.cached_models) > self.max_cached_models:
                    oldest_path = next(iter(self.cached_models))
                    self.cached_models.pop(oldest_path, None)
        return model


@dataclass
class FrozenCheckpointOpponent(OpponentPolicy):
    pool: CheckpointPool
    sim_config: SimulationConfig
    discrete_action_config: DiscreteActionConfig
    policy_type: str = "velocity_oriented"
    tactic_runtime_config: TacticRuntimeConfig = field(default_factory=TacticRuntimeConfig)
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    deterministic: bool = False
    hitter_deterministic: bool | None = None
    receiver_deterministic: bool | None = None
    action_mapper: DiscreteActionMapper = field(init=False)
    observation_encoder: ObservationEncoder = field(init=False)
    current_checkpoint_path: Path | None = field(default=None, init=False)
    current_model: PPO | None = field(default=None, init=False)
    current_side: Side = field(default="right", init=False)
    _prepared_hitter_shot: PreparedShot | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.action_mapper = DiscreteActionMapper(
            self.sim_config,
            self.discrete_action_config,
            policy_type=self.policy_type,
            tactic_runtime_config=self.tactic_runtime_config,
        )
        self.observation_encoder = ObservationEncoder(self.sim_config, self.observation_config)

    def on_episode_start(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        self.current_side = opponent_side
        self._prepared_hitter_shot = None
        self.current_checkpoint_path = self.pool.sample_path()
        self.current_model = self.pool.load_model(self.current_checkpoint_path)

    def refresh(self) -> None:
        self.pool.refresh()

    def label(self) -> str:
        if self.current_checkpoint_path is None:
            return "checkpoint_pool_unset"
        return self.current_checkpoint_path.stem

    def choose_hitter_action(self, state: StageState, config: SimulationConfig, server_side: Side) -> ShotAction:
        action = self._predict_action(
            state=state,
            role="hitter",
            server_side=server_side,
            pending_action=None,
            feasible_indices=[],
        )
        decoded = self.action_mapper.decode_hitter_for_agent(action, state, self.current_side).shot_action
        projected = self.action_mapper.project_hitter_action(state, decoded)
        self._prepared_hitter_shot = projected.prepared_shot
        return projected.shot_action

    def choose_likely_hitter_actions(
        self,
        state: StageState,
        config: SimulationConfig,
        server_side: Side,
        *,
        count: int,
    ) -> list[HitterActionCandidate]:
        if self.current_model is None:
            raise RuntimeError("Frozen checkpoint opponent has no active model for the current episode.")
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role="hitter",
            server_side=server_side,
            pending_action=None,
            feasible_indices=[],
        )
        candidates: list[HitterActionCandidate] = []
        for flat_index, probability in _top_hitter_action_indices(
            self.current_model,
            observation,
            hitter_action_count=self.action_mapper.hitter_action_count,
            count=count,
        ):
            decoded = self.action_mapper.decode_hitter_for_agent(flat_index, state, self.current_side).shot_action
            projected = self.action_mapper.project_hitter_action(state, decoded)
            candidates.append(
                HitterActionCandidate(
                    flat_index=int(flat_index),
                    action=projected.shot_action,
                    probability=float(probability),
                    prepared_shot=projected.prepared_shot,
                )
            )
        return candidates

    def take_prepared_hitter_shot(self) -> PreparedShot | None:
        prepared = self._prepared_hitter_shot
        self._prepared_hitter_shot = None
        return prepared

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
        server_side: Side,
        *,
        prepared_shot: PreparedShot | None = None,
    ) -> int | None:
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role="receiver",
            server_side=server_side,
            pending_action=action,
            feasible_indices=feasible_indices,
            prepared_shot=prepared_shot,
        )
        if self.current_model is None:
            raise RuntimeError("Frozen checkpoint opponent has no active model for the current episode.")
        return choose_model_action(
            self.current_model,
            observation,
            DecisionContext(
                state=state,
                role="receiver",
                pending_action=action,
                feasible_indices=list(feasible_indices),
                receiver_action_count=self.action_mapper.receiver_action_count,
            ),
            deterministic=self._deterministic_for_role("receiver"),
        )

    def _predict_action(
        self,
        *,
        state: StageState,
        role: str,
        server_side: Side,
        pending_action: ShotAction | None,
        feasible_indices: list[int],
    ) -> int:
        if self.current_model is None:
            raise RuntimeError("Frozen checkpoint opponent has no active model for the current episode.")
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role=role,
            server_side=server_side,
            pending_action=pending_action,
            feasible_indices=feasible_indices,
        )
        observation = adapt_observation_to_model(self.current_model, observation)
        action, _ = self.current_model.predict(observation, deterministic=self._deterministic_for_role(role))
        return _model_action_value(action)

    def _deterministic_for_role(self, role: str) -> bool:
        if role == "hitter" and self.hitter_deterministic is not None:
            return bool(self.hitter_deterministic)
        if role == "receiver" and self.receiver_deterministic is not None:
            return bool(self.receiver_deterministic)
        return bool(self.deterministic)


@dataclass
class FixedCheckpointOpponent(FrozenCheckpointOpponent):
    checkpoint_path: Path | None = None

    def on_episode_start(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        self.current_side = opponent_side
        self._prepared_hitter_shot = None
        if self.checkpoint_path is None:
            raise RuntimeError("FixedCheckpointOpponent requires checkpoint_path.")
        self.current_checkpoint_path = self.checkpoint_path.resolve()
        self.current_model = self.pool.load_model(self.current_checkpoint_path)


@dataclass
class LiveModelOpponent(OpponentPolicy):
    sim_config: SimulationConfig
    discrete_action_config: DiscreteActionConfig
    model: PPO | None = None
    policy_type: str = "velocity_oriented"
    tactic_runtime_config: TacticRuntimeConfig = field(default_factory=TacticRuntimeConfig)
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    deterministic: bool = False
    hitter_deterministic: bool | None = None
    receiver_deterministic: bool | None = None
    label_name: str = "mirror_self"
    action_mapper: DiscreteActionMapper = field(init=False)
    observation_encoder: ObservationEncoder = field(init=False)
    current_side: Side = field(default="right", init=False)
    _prepared_hitter_shot: PreparedShot | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.action_mapper = DiscreteActionMapper(
            self.sim_config,
            self.discrete_action_config,
            policy_type=self.policy_type,
            tactic_runtime_config=self.tactic_runtime_config,
        )
        self.observation_encoder = ObservationEncoder(self.sim_config, self.observation_config)

    def set_model(self, model: PPO) -> None:
        self.model = model

    def on_episode_start(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        self.current_side = opponent_side
        self._prepared_hitter_shot = None

    def refresh(self) -> None:
        return None

    def label(self) -> str:
        return self.label_name

    def choose_hitter_action(self, state: StageState, config: SimulationConfig, server_side: Side) -> ShotAction:
        action = self._predict_action(
            state=state,
            role="hitter",
            server_side=server_side,
            pending_action=None,
            feasible_indices=[],
        )
        decoded = self.action_mapper.decode_hitter_for_agent(action, state, self.current_side).shot_action
        projected = self.action_mapper.project_hitter_action(state, decoded)
        self._prepared_hitter_shot = projected.prepared_shot
        return projected.shot_action

    def choose_likely_hitter_actions(
        self,
        state: StageState,
        config: SimulationConfig,
        server_side: Side,
        *,
        count: int,
    ) -> list[HitterActionCandidate]:
        if self.model is None:
            raise RuntimeError("LiveModelOpponent has no model bound.")
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role="hitter",
            server_side=server_side,
            pending_action=None,
            feasible_indices=[],
        )
        candidates: list[HitterActionCandidate] = []
        for flat_index, probability in _top_hitter_action_indices(
            self.model,
            observation,
            hitter_action_count=self.action_mapper.hitter_action_count,
            count=count,
        ):
            decoded = self.action_mapper.decode_hitter_for_agent(flat_index, state, self.current_side).shot_action
            projected = self.action_mapper.project_hitter_action(state, decoded)
            candidates.append(
                HitterActionCandidate(
                    flat_index=int(flat_index),
                    action=projected.shot_action,
                    probability=float(probability),
                    prepared_shot=projected.prepared_shot,
                )
            )
        return candidates

    def take_prepared_hitter_shot(self) -> PreparedShot | None:
        prepared = self._prepared_hitter_shot
        self._prepared_hitter_shot = None
        return prepared

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
        server_side: Side,
        *,
        prepared_shot: PreparedShot | None = None,
    ) -> int | None:
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role="receiver",
            server_side=server_side,
            pending_action=action,
            feasible_indices=feasible_indices,
            prepared_shot=prepared_shot,
        )
        if self.model is None:
            raise RuntimeError("LiveModelOpponent has no model bound.")
        return choose_model_action(
            self.model,
            observation,
            DecisionContext(
                state=state,
                role="receiver",
                pending_action=action,
                feasible_indices=list(feasible_indices),
                receiver_action_count=self.action_mapper.receiver_action_count,
            ),
            deterministic=self._deterministic_for_role("receiver"),
        )

    def _predict_action(
        self,
        *,
        state: StageState,
        role: str,
        server_side: Side,
        pending_action: ShotAction | None,
        feasible_indices: list[int],
    ) -> int:
        if self.model is None:
            raise RuntimeError("LiveModelOpponent has no model bound.")
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role=role,
            server_side=server_side,
            pending_action=pending_action,
            feasible_indices=feasible_indices,
        )
        observation = adapt_observation_to_model(self.model, observation)
        action, _ = self.model.predict(observation, deterministic=self._deterministic_for_role(role))
        return _model_action_value(action)

    def _deterministic_for_role(self, role: str) -> bool:
        if role == "hitter" and self.hitter_deterministic is not None:
            return bool(self.hitter_deterministic)
        if role == "receiver" and self.receiver_deterministic is not None:
            return bool(self.receiver_deterministic)
        return bool(self.deterministic)


@dataclass
class MixedCheckpointOpponent(OpponentPolicy):
    checkpoint_pool: CheckpointPool
    sim_config: SimulationConfig
    discrete_action_config: DiscreteActionConfig
    historical_anchor_pool: CheckpointPool | None = None
    policy_type: str = "velocity_oriented"
    tactic_runtime_config: TacticRuntimeConfig = field(default_factory=TacticRuntimeConfig)
    heuristic_opponent_prob: float = 0.05
    recent_weight: float = 0.6
    older_weight: float = 0.4
    historical_anchor_weight: float = 0.0
    recent_continuation_weight: float = 0.0
    newest_continuation_weight: float = 0.0
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    deterministic: bool = False
    hitter_deterministic: bool | None = None
    receiver_deterministic: bool | None = None
    action_mapper: DiscreteActionMapper = field(init=False)
    observation_encoder: ObservationEncoder = field(init=False)
    heuristic_opponent: OpponentPolicy = field(default_factory=lambda: make_opponent("safe"))
    current_checkpoint_path: Path | None = field(default=None, init=False)
    current_model: PPO | None = field(default=None, init=False)
    current_side: Side = field(default="right", init=False)
    active_source: str = field(default="checkpoint", init=False)
    active_bucket: str = field(default="recent", init=False)
    _prepared_hitter_shot: PreparedShot | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.action_mapper = DiscreteActionMapper(
            self.sim_config,
            self.discrete_action_config,
            policy_type=self.policy_type,
            tactic_runtime_config=self.tactic_runtime_config,
        )
        self.observation_encoder = ObservationEncoder(self.sim_config, self.observation_config)

    def on_episode_start(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        self.current_side = opponent_side
        self.current_checkpoint_path = None
        self.current_model = None
        self._prepared_hitter_shot = None

        if self._uses_variety_pool():
            self._start_variety_episode(
                train_side=train_side,
                opponent_side=opponent_side,
                server_side=server_side,
                config=config,
            )
            return

        if float(self.checkpoint_pool.rng.random()) < self.heuristic_opponent_prob:
            self.active_source = "heuristic"
            self.active_bucket = "heuristic"
            self.heuristic_opponent.on_episode_start(
                train_side=train_side,
                opponent_side=opponent_side,
                server_side=server_side,
                config=config,
            )
            return
        self.active_source = "checkpoint"
        self.current_checkpoint_path, self.active_bucket = self.checkpoint_pool.sample_weighted_path(
            recent_weight=self.recent_weight,
            older_weight=self.older_weight,
        )
        self.current_model = self.checkpoint_pool.load_model(self.current_checkpoint_path)

    def refresh(self) -> None:
        self.checkpoint_pool.refresh()

    def _uses_variety_pool(self) -> bool:
        return (
            self.historical_anchor_pool is not None
            or self.historical_anchor_weight > 0.0
            or self.recent_continuation_weight > 0.0
            or self.newest_continuation_weight > 0.0
        )

    def _start_variety_episode(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        newest_path = self.checkpoint_pool.newest_path()
        recent_path = self.checkpoint_pool.sample_recent_path(exclude_newest=True)

        buckets: list[tuple[str, float]] = []
        if self.heuristic_opponent_prob > 0.0:
            buckets.append(("heuristic", float(self.heuristic_opponent_prob)))
        if (
            self.historical_anchor_pool is not None
            and self.historical_anchor_pool.checkpoints
            and self.historical_anchor_weight > 0.0
        ):
            buckets.append(("historical", float(self.historical_anchor_weight)))
        if recent_path is not None and self.recent_continuation_weight > 0.0:
            buckets.append(("recent_continuation", float(self.recent_continuation_weight)))
        if newest_path is not None and self.newest_continuation_weight > 0.0:
            buckets.append(("newest_continuation", float(self.newest_continuation_weight)))

        if not buckets:
            raise RuntimeError("Variety opponent pool has no available bucket.")

        weights = np.asarray([max(weight, 0.0) for _, weight in buckets], dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            bucket_name = buckets[int(self.checkpoint_pool.rng.integers(len(buckets)))][0]
        else:
            probabilities = weights / weights.sum()
            bucket_name = buckets[int(self.checkpoint_pool.rng.choice(len(buckets), p=probabilities))][0]

        if bucket_name == "heuristic":
            self.active_source = "heuristic"
            self.active_bucket = "heuristic"
            self.heuristic_opponent.on_episode_start(
                train_side=train_side,
                opponent_side=opponent_side,
                server_side=server_side,
                config=config,
            )
            return

        if bucket_name == "historical":
            if self.historical_anchor_pool is None:
                raise RuntimeError("Historical anchor bucket selected without a historical anchor pool.")
            self.active_source = "checkpoint"
            self.active_bucket = "historical"
            self.current_checkpoint_path = self.historical_anchor_pool.sample_path()
            self.current_model = self.historical_anchor_pool.load_model(self.current_checkpoint_path)
            return

        if bucket_name == "newest_continuation":
            if newest_path is None:
                raise RuntimeError("Newest continuation bucket selected without a checkpoint.")
            self.active_source = "checkpoint"
            self.active_bucket = "newest_continuation"
            self.current_checkpoint_path = newest_path
            self.current_model = self.checkpoint_pool.load_model(self.current_checkpoint_path)
            return

        if recent_path is None:
            raise RuntimeError("Recent continuation bucket selected without a checkpoint.")
        self.active_source = "checkpoint"
        self.active_bucket = "recent_continuation"
        self.current_checkpoint_path = recent_path
        self.current_model = self.checkpoint_pool.load_model(self.current_checkpoint_path)

    def label(self) -> str:
        if self.active_source == "heuristic":
            return "heuristic_safe"
        if self.current_checkpoint_path is None:
            return "checkpoint_pool_unset"
        return f"{self.active_bucket}:{self.current_checkpoint_path.stem}"

    def choose_hitter_action(self, state: StageState, config: SimulationConfig, server_side: Side) -> ShotAction:
        if self.active_source == "heuristic":
            self._prepared_hitter_shot = None
            return self.heuristic_opponent.choose_hitter_action(state, config, server_side)
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role="hitter",
            server_side=server_side,
            pending_action=None,
            feasible_indices=[],
        )
        if self.current_model is None:
            raise RuntimeError("Checkpoint opponent model was not loaded for this episode.")
        observation = adapt_observation_to_model(self.current_model, observation)
        action, _ = self.current_model.predict(
            observation,
            deterministic=self._deterministic_for_role("hitter"),
        )
        decoded = self.action_mapper.decode_hitter_for_agent(_model_action_value(action), state, self.current_side).shot_action
        projected = self.action_mapper.project_hitter_action(state, decoded)
        self._prepared_hitter_shot = projected.prepared_shot
        return projected.shot_action

    def choose_likely_hitter_actions(
        self,
        state: StageState,
        config: SimulationConfig,
        server_side: Side,
        *,
        count: int,
    ) -> list[HitterActionCandidate]:
        if self.active_source == "heuristic":
            return [
                HitterActionCandidate(
                    flat_index=None,
                    action=self.heuristic_opponent.choose_hitter_action(state, config, server_side),
                    probability=1.0,
                    prepared_shot=None,
                )
            ]
        if self.current_model is None:
            raise RuntimeError("Checkpoint opponent model was not loaded for this episode.")
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role="hitter",
            server_side=server_side,
            pending_action=None,
            feasible_indices=[],
        )
        candidates: list[HitterActionCandidate] = []
        for flat_index, probability in _top_hitter_action_indices(
            self.current_model,
            observation,
            hitter_action_count=self.action_mapper.hitter_action_count,
            count=count,
        ):
            decoded = self.action_mapper.decode_hitter_for_agent(flat_index, state, self.current_side).shot_action
            projected = self.action_mapper.project_hitter_action(state, decoded)
            candidates.append(
                HitterActionCandidate(
                    flat_index=int(flat_index),
                    action=projected.shot_action,
                    probability=float(probability),
                    prepared_shot=projected.prepared_shot,
                )
            )
        return candidates

    def take_prepared_hitter_shot(self) -> PreparedShot | None:
        prepared = self._prepared_hitter_shot
        self._prepared_hitter_shot = None
        return prepared

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
        server_side: Side,
        *,
        prepared_shot: PreparedShot | None = None,
    ) -> int | None:
        if self.active_source == "heuristic":
            return self.heuristic_opponent.choose_intercept_index(state, action, feasible_indices, config, server_side)
        observation = self.observation_encoder.encode(
            state=state,
            agent_side=self.current_side,
            role="receiver",
            server_side=server_side,
            pending_action=action,
            feasible_indices=feasible_indices,
            prepared_shot=prepared_shot,
        )
        if self.current_model is None:
            raise RuntimeError("Checkpoint opponent model was not loaded for this episode.")
        return choose_model_action(
            self.current_model,
            observation,
            DecisionContext(
                state=state,
                role="receiver",
                pending_action=action,
                feasible_indices=list(feasible_indices),
                receiver_action_count=self.action_mapper.receiver_action_count,
            ),
            deterministic=self._deterministic_for_role("receiver"),
        )

    def _deterministic_for_role(self, role: str) -> bool:
        if role == "hitter" and self.hitter_deterministic is not None:
            return bool(self.hitter_deterministic)
        if role == "receiver" and self.receiver_deterministic is not None:
            return bool(self.receiver_deterministic)
        return bool(self.deterministic)


class SelfPlayCheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        latest_model_path: Path,
        save_freq: int,
        pool_size: int,
        train_env: Any,
        timestep_offset: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.checkpoint_dir = checkpoint_dir
        self.latest_model_path = latest_model_path
        self.save_freq = save_freq
        self.pool_size = pool_size
        self.train_env = train_env
        self.timestep_offset = int(timestep_offset)
        self.last_save_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save_timestep < self.save_freq:
            return True
        self.last_save_timestep = self.num_timesteps
        effective_timestep = self.timestep_offset + self.num_timesteps
        ensure_directory(self.checkpoint_dir)
        checkpoint_path = self.checkpoint_dir / f"selfplay_step_{effective_timestep}.zip"
        self.model.save(checkpoint_path)
        replace_with_existing_file(checkpoint_path, self.latest_model_path)
        self._prune_snapshots()
        if hasattr(self.train_env, "env_method"):
            self.train_env.env_method("refresh_opponent_pool")
        self.logger.record("selfplay/pool_checkpoint_count", len(list(self.checkpoint_dir.glob("*.zip"))))
        return True

    def _prune_snapshots(self) -> None:
        snapshots = sorted(self.checkpoint_dir.glob("*.zip"), key=_checkpoint_sort_key, reverse=True)
        for stale_path in snapshots[self.pool_size :]:
            try:
                stale_path.unlink()
            except FileNotFoundError:
                continue


class SelfPlayEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_freq: int,
        eval_episodes: int,
        eval_matchups: str,
        eval_seed: int,
        output_dir: Path,
        best_model_path: Path,
        train_side: Side,
        initial_server: str,
        mirror_train_side: bool,
        mirror_match_fraction: float,
        sim_config: SimulationConfig,
        reward_config: RewardConfig,
        reset_sampling_config: ResetSamplingConfig,
        train_reaction_time: float,
        opponent_reaction_time: float,
        max_stages_per_rally: int,
        discrete_action_config: DiscreteActionConfig,
        policy_type: str,
        tactic_runtime_config: TacticRuntimeConfig,
        checkpoint_pool: CheckpointPool,
        random_service_x: bool = False,
        base_checkpoint_path: Path | None = None,
        anchor_eval_interval: int = 0,
        anchor_checkpoint_dir: Path | None = None,
        deterministic: bool = False,
        timestep_offset: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_matchups = eval_matchups
        self.eval_seed = eval_seed
        self.output_dir = output_dir
        self.best_model_path = best_model_path
        self.train_side = train_side
        self.initial_server = initial_server
        self.mirror_train_side = mirror_train_side
        self.mirror_match_fraction = mirror_match_fraction
        self.random_service_x = bool(random_service_x)
        self.sim_config = sim_config
        self.reward_config = reward_config
        self.reset_sampling_config = reset_sampling_config
        self.train_reaction_time = float(train_reaction_time)
        self.opponent_reaction_time = float(opponent_reaction_time)
        self.max_stages_per_rally = int(max_stages_per_rally)
        self.discrete_action_config = discrete_action_config
        self.policy_type = policy_type
        self.tactic_runtime_config = tactic_runtime_config
        self.checkpoint_pool = checkpoint_pool
        self.base_checkpoint_path = None if base_checkpoint_path is None else base_checkpoint_path.resolve()
        self.anchor_eval_interval = int(anchor_eval_interval)
        self.anchor_checkpoint_dir = None if anchor_checkpoint_dir is None else anchor_checkpoint_dir.resolve()
        self.deterministic = bool(deterministic)
        self.timestep_offset = int(timestep_offset)
        self.best_score = float("-inf")
        self.last_eval_timestep = self._initial_last_eval_timestep()
        self.current_anchor_step = self._initial_anchor_step()
        self.current_anchor_path = self._initialize_anchor_checkpoint()

    @staticmethod
    def _initial_last_eval_timestep_for_offset(*, timestep_offset: int, eval_freq: int) -> int:
        if eval_freq <= 0:
            return 0
        remainder = int(timestep_offset) % int(eval_freq)
        if remainder == 0:
            return 0
        return -remainder

    def _initial_last_eval_timestep(self) -> int:
        return self._initial_last_eval_timestep_for_offset(
            timestep_offset=self.timestep_offset,
            eval_freq=self.eval_freq,
        )

    def _initial_anchor_step(self) -> int | None:
        if self.anchor_eval_interval <= 0:
            return None
        if self.timestep_offset < self.anchor_eval_interval:
            return None
        return (self.timestep_offset // self.anchor_eval_interval) * self.anchor_eval_interval

    def _anchor_path_for_step(self, step: int) -> Path:
        if self.anchor_checkpoint_dir is None:
            raise RuntimeError("anchor_checkpoint_dir must be set when anchor evaluation is enabled.")
        return self.anchor_checkpoint_dir / f"anchor_step_{step}.zip"

    def _initialize_anchor_checkpoint(self) -> Path | None:
        if self.current_anchor_step is None:
            return None
        if self.base_checkpoint_path is None or not self.base_checkpoint_path.exists():
            raise FileNotFoundError("Anchor evaluation requires an existing base checkpoint path.")
        anchor_path = self._anchor_path_for_step(self.current_anchor_step)
        ensure_directory(anchor_path.parent)
        if not anchor_path.exists():
            shutil.copy2(self.base_checkpoint_path, anchor_path)
        return anchor_path.resolve()

    def _maybe_roll_anchor(self, *, effective_timestep: int) -> None:
        if self.anchor_eval_interval <= 0 or self.anchor_checkpoint_dir is None:
            return
        if effective_timestep <= 0:
            return
        if effective_timestep % self.anchor_eval_interval != 0:
            return
        next_anchor_step = effective_timestep
        if self.current_anchor_step is None:
            if next_anchor_step <= 0:
                return
            anchor_path = self._anchor_path_for_step(next_anchor_step)
            ensure_directory(anchor_path.parent)
            self.model.save(anchor_path)
            self.current_anchor_step = next_anchor_step
            self.current_anchor_path = anchor_path.resolve()
            return
        if next_anchor_step <= self.current_anchor_step:
            return
        anchor_path = self._anchor_path_for_step(next_anchor_step)
        ensure_directory(anchor_path.parent)
        self.model.save(anchor_path)
        self.current_anchor_step = next_anchor_step
        self.current_anchor_path = anchor_path.resolve()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps

        effective_timestep = self.timestep_offset + self.num_timesteps
        evaluation_report = self._run_evaluation()
        summaries = evaluation_report["summaries"]
        current_matchups = evaluation_report["current_matchup_summaries"]
        heuristic_summary = summaries.get("heuristic")
        newest_summary = current_matchups.get("current_vs_newest_checkpoint")
        mirror_summary = current_matchups.get("current_vs_mirror_self")
        anchor_summary = current_matchups.get("current_vs_anchor_checkpoint")
        if isinstance(heuristic_summary, dict):
            self.logger.record("selfplay_eval/win_rate_vs_heuristic", heuristic_summary["win_rate"])
        if isinstance(newest_summary, dict):
            self.logger.record("selfplay_eval/win_rate_vs_newest_checkpoint", newest_summary["win_rate"])
            for server_key, metric_name in (("left_serve_first", "left"), ("right_serve_first", "right")):
                server_summary = newest_summary.get("server_breakdown", {}).get(server_key)
                if isinstance(server_summary, dict):
                    self.logger.record(
                        f"selfplay_eval/wr_newest_{metric_name}_serve",
                        server_summary["win_rate"],
                    )
            self.logger.record("selfplay_eval/avg_rally_length_vs_newest_checkpoint", newest_summary["avg_rally_length"])
            self.logger.record(
                "selfplay_eval/avg_invalid_action_rate_vs_newest_checkpoint",
                newest_summary["avg_invalid_action_rate"],
            )
        if isinstance(mirror_summary, dict):
            self.logger.record("selfplay_eval/win_rate_vs_mirror_self", mirror_summary["win_rate"])
            self.logger.record("selfplay_eval/avg_rally_length_vs_mirror_self", mirror_summary["avg_rally_length"])
            self.logger.record(
                "selfplay_eval/avg_invalid_action_rate_vs_mirror_self",
                mirror_summary["avg_invalid_action_rate"],
            )
        if isinstance(anchor_summary, dict):
            self.logger.record("selfplay_eval/win_rate_vs_anchor_checkpoint", anchor_summary["win_rate"])
            for server_key, metric_name in (("left_serve_first", "left"), ("right_serve_first", "right")):
                server_summary = anchor_summary.get("server_breakdown", {}).get(server_key)
                if isinstance(server_summary, dict):
                    self.logger.record(
                        f"selfplay_eval/wr_anchor_{metric_name}_serve",
                        server_summary["win_rate"],
                    )
            self.logger.record("selfplay_eval/avg_rally_length_vs_anchor_checkpoint", anchor_summary["avg_rally_length"])
            self.logger.record(
                "selfplay_eval/avg_invalid_action_rate_vs_anchor_checkpoint",
                anchor_summary["avg_invalid_action_rate"],
            )
        self.logger.record(
            "selfplay_eval/narrow_opponent_dependency",
            1.0 if bool(evaluation_report["narrow_opponent_dependency"]) else 0.0,
        )

        score_terms: list[float] = []
        if isinstance(heuristic_summary, dict):
            score_terms.append(float(heuristic_summary["win_rate"]))
        if isinstance(newest_summary, dict):
            score_terms.append(float(newest_summary["win_rate"]))
        if isinstance(mirror_summary, dict):
            score_terms.append(float(mirror_summary["win_rate"]))
        if isinstance(anchor_summary, dict):
            score_terms.append(float(anchor_summary["win_rate"]))
        if not score_terms:
            raise RuntimeError("Self-play evaluation produced no matchup summaries.")
        score = float(np.mean(score_terms))
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.best_model_path)

        ensure_directory(self.output_dir)
        payload = {
            "num_timesteps": effective_timestep,
            "best_score": self.best_score,
            **evaluation_report,
        }
        report_path = self.output_dir / f"selfplay_eval_{effective_timestep}.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._maybe_roll_anchor(effective_timestep=effective_timestep)
        return True

    def _run_evaluation(self) -> dict[str, Any]:
        selector = ModelSelector(model=self.model, deterministic=self.deterministic)
        summaries: dict[str, dict[str, object]] = {}
        current_matchups: dict[str, dict[str, object]] = {}
        effective_timestep = self.timestep_offset + self.num_timesteps
        base_seed = self.eval_seed + effective_timestep

        include_heuristic = self.eval_matchups == "full"
        include_newest = self.eval_matchups in {"newest-only", "newest-and-mirror", "full"}
        include_mirror = self.eval_matchups in {"newest-and-mirror", "full"}
        include_anchor = self.current_anchor_path is not None

        if include_heuristic:
            heuristic_env = BadmintonRLEnv(
                config=self.sim_config,
                rl_config=RLEnvConfig(
                    train_side=self.train_side,
                    initial_server=self.initial_server,
                    mirror_train_side=self.mirror_train_side,
                    mirror_match_fraction=self.mirror_match_fraction,
                    random_service_x=self.random_service_x,
                    train_reaction_time=self.train_reaction_time,
                    opponent_reaction_time=self.opponent_reaction_time,
                    max_stages_per_rally=self.max_stages_per_rally,
                    policy_type=self.policy_type,
                    tactic_runtime=self.tactic_runtime_config,
                    reset_sampling=self.reset_sampling_config,
                    reward=self.reward_config,
                    recovery_counterfactual_other_sample_count=0,
                    recovery_counterfactual_expected_response_target=False,
                ),
                discrete_action_config=self.discrete_action_config,
                opponent=make_opponent("safe", seed=base_seed + 1),
                seed=base_seed + 1,
            )
            heuristic_summary, _ = evaluate_selector(
                "heuristic_safe",
                selector,
                heuristic_env,
                self.eval_episodes,
                base_seed + 1_000,
            )
            summaries["heuristic"] = heuristic_summary

        if include_mirror:
            mirror_opponent = LiveModelOpponent(
                sim_config=self.sim_config,
                discrete_action_config=self.discrete_action_config,
                policy_type=self.policy_type,
                tactic_runtime_config=self.tactic_runtime_config,
                model=self.model,
                deterministic=self.deterministic,
                label_name="mirror_self",
            )
            mirror_env = build_selfplay_env(
                train_side=self.train_side,
                mirror_train_side=self.mirror_train_side,
                mirror_match_fraction=self.mirror_match_fraction,
                initial_server=self.initial_server,
                random_service_x=self.random_service_x,
                sim_config=self.sim_config,
                train_reaction_time=self.train_reaction_time,
                opponent_reaction_time=self.opponent_reaction_time,
                max_stages_per_rally=self.max_stages_per_rally,
                reward_config=self.reward_config,
                reset_sampling_config=self.reset_sampling_config,
                policy_type=self.policy_type,
                tactic_runtime_config=self.tactic_runtime_config,
                seed=base_seed + 11,
                discrete_action_config=self.discrete_action_config,
                opponent=mirror_opponent,
                recovery_counterfactual_other_sample_count=0,
                recovery_counterfactual_expected_response_target=False,
            )
            mirror_summary = self._evaluate_fixed_server_matchup(
                selector=selector,
                matchup_name="current_vs_mirror_self",
                env_builder=lambda server: build_selfplay_env(
                    train_side=self.train_side,
                    mirror_train_side=self.mirror_train_side,
                    mirror_match_fraction=self.mirror_match_fraction,
                    initial_server=server,
                    random_service_x=self.random_service_x,
                    sim_config=self.sim_config,
                    train_reaction_time=self.train_reaction_time,
                    opponent_reaction_time=self.opponent_reaction_time,
                    max_stages_per_rally=self.max_stages_per_rally,
                    reward_config=self.reward_config,
                    reset_sampling_config=self.reset_sampling_config,
                    policy_type=self.policy_type,
                    tactic_runtime_config=self.tactic_runtime_config,
                    seed=base_seed + 11,
                    discrete_action_config=self.discrete_action_config,
                    opponent=mirror_opponent,
                    recovery_counterfactual_other_sample_count=0,
                    recovery_counterfactual_expected_response_target=False,
                ),
                seed_base=base_seed + 11_000,
            )
            current_matchups["current_vs_mirror_self"] = mirror_summary

        if include_newest:
            self.checkpoint_pool.refresh()
            newest = self.checkpoint_pool.newest_path()
            if newest is not None:
                newest_summary = self._evaluate_fixed_server_matchup(
                    selector=selector,
                    matchup_name="current_vs_newest_checkpoint",
                    env_builder=lambda server: build_selfplay_env(
                        train_side=self.train_side,
                        mirror_train_side=self.mirror_train_side,
                        mirror_match_fraction=self.mirror_match_fraction,
                        initial_server=server,
                        random_service_x=self.random_service_x,
                        sim_config=self.sim_config,
                        train_reaction_time=self.train_reaction_time,
                        opponent_reaction_time=self.opponent_reaction_time,
                        max_stages_per_rally=self.max_stages_per_rally,
                        reward_config=self.reward_config,
                        reset_sampling_config=self.reset_sampling_config,
                        policy_type=self.policy_type,
                        tactic_runtime_config=self.tactic_runtime_config,
                        seed=base_seed + 3,
                        discrete_action_config=self.discrete_action_config,
                        opponent=FixedCheckpointOpponent(
                            pool=self.checkpoint_pool,
                            checkpoint_path=newest,
                            sim_config=self.sim_config,
                            discrete_action_config=self.discrete_action_config,
                            policy_type=self.policy_type,
                            tactic_runtime_config=self.tactic_runtime_config,
                            deterministic=self.deterministic,
                        ),
                        recovery_counterfactual_other_sample_count=0,
                        recovery_counterfactual_expected_response_target=False,
                    ),
                    seed_base=base_seed + 3_000,
                )
                current_matchups["current_vs_newest_checkpoint"] = newest_summary

        if include_anchor and self.current_anchor_path is not None:
            anchor_summary = self._evaluate_fixed_server_matchup(
                selector=selector,
                matchup_name="current_vs_anchor_checkpoint",
                env_builder=lambda server: build_selfplay_env(
                    train_side=self.train_side,
                    mirror_train_side=self.mirror_train_side,
                    mirror_match_fraction=self.mirror_match_fraction,
                    initial_server=server,
                    random_service_x=self.random_service_x,
                    sim_config=self.sim_config,
                    train_reaction_time=self.train_reaction_time,
                    opponent_reaction_time=self.opponent_reaction_time,
                    max_stages_per_rally=self.max_stages_per_rally,
                    reward_config=self.reward_config,
                    reset_sampling_config=self.reset_sampling_config,
                    policy_type=self.policy_type,
                    tactic_runtime_config=self.tactic_runtime_config,
                    seed=base_seed + 7,
                    discrete_action_config=self.discrete_action_config,
                    opponent=FixedCheckpointOpponent(
                        pool=self.checkpoint_pool,
                        checkpoint_path=self.current_anchor_path,
                        sim_config=self.sim_config,
                        discrete_action_config=self.discrete_action_config,
                        policy_type=self.policy_type,
                        tactic_runtime_config=self.tactic_runtime_config,
                        deterministic=self.deterministic,
                    ),
                    recovery_counterfactual_other_sample_count=0,
                    recovery_counterfactual_expected_response_target=False,
                ),
                seed_base=base_seed + 7_000,
            )
            anchor_summary["anchor_step"] = self.current_anchor_step
            anchor_summary["anchor_checkpoint_path"] = str(self.current_anchor_path)
            current_matchups["current_vs_anchor_checkpoint"] = anchor_summary

        win_rates = [float(summary["win_rate"]) for summary in summaries.values()]
        win_rates.extend(float(summary["win_rate"]) for summary in current_matchups.values())
        narrow_dependency = len(win_rates) >= 2 and (max(win_rates) - min(win_rates) > 0.25)
        return {
            "eval_matchups": self.eval_matchups,
            "anchor_eval_interval": self.anchor_eval_interval,
            "current_anchor_step": self.current_anchor_step,
            "current_anchor_path": None if self.current_anchor_path is None else str(self.current_anchor_path),
            "summaries": summaries,
            "current_matchup_summaries": current_matchups,
            "narrow_opponent_dependency": narrow_dependency,
        }

    def _evaluate_fixed_server_matchup(
        self,
        *,
        selector: ModelSelector,
        matchup_name: str,
        env_builder: Any,
        seed_base: int,
    ) -> dict[str, Any]:
        server_breakdown: dict[str, dict[str, object]] = {}
        combined_results: list[dict[str, object]] = []
        for index, server in enumerate(("left", "right")):
            env = env_builder(server)
            server_name = f"{server}_serve_first"
            summary, results = evaluate_selector(
                f"{matchup_name}_{server_name}",
                selector,
                env,
                self.eval_episodes,
                seed_base + index * 10_000,
            )
            server_breakdown[server_name] = summary
            combined_results.extend(results)
        combined_summary = summarize_episodes(combined_results)
        combined_summary["name"] = matchup_name
        combined_summary["server_breakdown"] = server_breakdown
        return combined_summary


class SelfPlayProgressVideoCallback(BaseCallback):
    def __init__(
        self,
        *,
        sample_env: BadmintonRLEnv,
        output_dir: Path,
        record_freq: int,
        sample_seed: int,
        fps: int = 18,
        stage_pause: float = 0.15,
        rally_pause: float = 0.9,
        record_initial_sample: bool = True,
        deterministic: bool = False,
        matchup_name: str = "training_progress",
        save_sample_media: bool = False,
        write_combined_frames: bool = False,
        write_combined_gif: bool = False,
        render_dpi: int = 90,
        timestep_offset: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.sample_env = sample_env
        self.output_dir = output_dir
        self.record_freq = record_freq
        self.sample_seed = sample_seed
        self.fps = fps
        self.stage_pause = stage_pause
        self.rally_pause = rally_pause
        self.record_initial_sample = record_initial_sample
        self.deterministic = deterministic
        self.matchup_name = matchup_name
        self.save_sample_media = save_sample_media
        self.write_combined_frames = write_combined_frames
        self.write_combined_gif = write_combined_gif
        self.render_dpi = int(render_dpi)
        self.timestep_offset = int(timestep_offset)
        self.last_record_timestep = -1
        self.samples: list[TrainingProgressSample] = []
        self.sample_reports: list[dict[str, Any]] = []
        self.combined_report: dict[str, Any] | None = None

    def _on_training_start(self) -> None:
        if self.record_initial_sample:
            self._record_sample(force=True)

    def _on_step(self) -> bool:
        if self.record_freq <= 0:
            return True
        if self.num_timesteps - max(self.last_record_timestep, 0) < self.record_freq:
            return True
        self._record_sample(force=False)
        return True

    def _on_training_end(self) -> None:
        if self.last_record_timestep != self.num_timesteps:
            self._record_sample(force=True)
        self._export_combined_progress_video()

    def _record_sample(self, *, force: bool) -> None:
        if self.last_record_timestep == self.num_timesteps and not force:
            return

        ensure_directory(self.output_dir)
        effective_timestep = self.timestep_offset + self.num_timesteps
        if hasattr(self.sample_env, "refresh_opponent_pool"):
            self.sample_env.refresh_opponent_pool()
        sample_opponent = getattr(self.sample_env, "opponent", None)
        if hasattr(sample_opponent, "set_model"):
            sample_opponent.set_model(self.model)
        selector = ModelSelector(model=self.model, deterministic=self.deterministic)
        result = rollout_episode(
            self.sample_env,
            selector,
            seed=self.sample_seed + effective_timestep,
        )
        records = result["records"]
        if not isinstance(records, list) or not records:
            return

        trace = build_rally_trace(records, self.sample_env.config)
        sample = TrainingProgressSample(
            step=effective_timestep,
            trace=trace,
            opponent_label=str(result.get("opponent_label")) if result.get("opponent_label") is not None else None,
            rally_won=bool(result["rally_won"]),
            invalid_action_rate=float(result["invalid_action_rate"]),
        )
        self.samples.append(sample)
        self.last_record_timestep = self.num_timesteps

        sample_dir = self.output_dir / f"step_{effective_timestep:09d}"
        ensure_directory(sample_dir)
        trace_path = sample_dir / "rally_trace.json"
        trace_path.write_text(json.dumps(rally_trace_to_dict(trace), indent=2), encoding="utf-8")
        gif_path: str | None = None
        mp4_path: str | None = None
        if self.save_sample_media:
            export_result = export_rally_video(
                trace,
                self.sample_env.config,
                sample_dir,
                fps=self.fps,
                stage_pause=self.stage_pause,
                dpi=self.render_dpi,
            )
            gif_path = str(export_result.gif_path)
            mp4_path = str(export_result.mp4_path) if export_result.mp4_path is not None else None
        sample_report = {
            "step": effective_timestep,
            "matchup": self.matchup_name,
            "reward": float(result["reward"]),
            "winner": result["winner"],
            "rally_won": bool(result["rally_won"]),
            "rally_length": int(result["rally_length"]),
            "invalid_action_rate": float(result["invalid_action_rate"]),
            "opponent_label": result.get("opponent_label"),
            "server": result.get("server"),
            "sample_dir": str(sample_dir),
            "gif_path": gif_path,
            "mp4_path": mp4_path,
            "trace_path": str(trace_path),
        }
        self.sample_reports.append(sample_report)
        self._write_progress_manifest()

    def _export_combined_progress_video(self) -> None:
        if not self.samples:
            self.combined_report = None
            self._write_progress_manifest()
            return
        combined_result = export_training_progress_video(
            self.samples,
            self.sample_env.config,
            self.output_dir / "combined",
            fps=self.fps,
            stage_pause=self.stage_pause,
            rally_pause=self.rally_pause,
            write_frames=self.write_combined_frames,
            dpi=self.render_dpi,
            write_gif=self.write_combined_gif,
            write_mp4=True,
        )
        self.combined_report = {
            "combined_gif_path": str(combined_result.gif_path),
            "combined_mp4_path": str(combined_result.mp4_path) if combined_result.mp4_path is not None else None,
            "combined_trace_path": str(combined_result.trace_path),
        }
        self._write_progress_manifest()

    def _write_progress_manifest(self) -> None:
        progress_report = {
            "matchup": self.matchup_name,
            "sample_count": len(self.sample_reports),
            "samples": self.sample_reports,
            "combined_gif_path": None,
            "combined_mp4_path": None,
            "combined_trace_path": None,
        }
        if self.combined_report is not None:
            progress_report.update(self.combined_report)
        (self.output_dir / "progress_manifest.json").write_text(
            json.dumps(progress_report, indent=2),
            encoding="utf-8",
        )


def build_selfplay_env(
    *,
    train_side: Side,
    mirror_train_side: bool,
    mirror_match_fraction: float = 0.0,
    initial_server: str,
    random_service_x: bool = False,
    sim_config: SimulationConfig | None = None,
    train_reaction_time: float = 0.0,
    opponent_reaction_time: float = 0.0,
    max_stages_per_rally: int = 30,
    policy_type: str = "velocity_oriented",
    tactic_runtime_config: TacticRuntimeConfig | None = None,
    reward_config: RewardConfig | None = None,
    reset_sampling_config: ResetSamplingConfig | None = None,
    seed: int,
    discrete_action_config: DiscreteActionConfig,
    opponent: OpponentPolicy,
    include_records_in_info: bool = False,
    recovery_counterfactual_other_sample_count: int = RECOVERY_COUNTERFACTUAL_OTHER_SAMPLE_COUNT,
    counterfactual_opponent_response_samples: int = COUNTERFACTUAL_OPPONENT_RESPONSE_SAMPLES,
    recovery_counterfactual_expected_response_target: bool = True,
    recovery_full_diagnostics_probability: float = 0.0,
    use_shot_cf: bool = False,
    shot_cf_coef: float = 0.1,
    shot_cf_top_m: int = 20,
    shot_cf_num_modes: int = 3,
    shot_cf_min_landing_dist: float = 1.0,
    shot_cf_depth: int = 1,
    shot_cf_include_chosen: bool = True,
    shot_cf_skip_low_diversity: bool = True,
    shot_cf_min_modes: int = 2,
    shot_cf_value_detach: bool = True,
    shot_cf_normalize: bool = True,
    shot_cf_debug_log: bool = False,
) -> BadmintonRLEnv:
    return BadmintonRLEnv(
        config=sim_config or SimulationConfig(),
        rl_config=RLEnvConfig(
            train_side=train_side,
            initial_server=initial_server,
            train_reaction_time=train_reaction_time,
            opponent_reaction_time=opponent_reaction_time,
            max_stages_per_rally=max_stages_per_rally,
            include_records_in_info=include_records_in_info,
            mirror_train_side=mirror_train_side,
            mirror_match_fraction=mirror_match_fraction,
            random_service_x=random_service_x,
            policy_type=policy_type,
            tactic_runtime=tactic_runtime_config or TacticRuntimeConfig(),
            reward=reward_config or RewardConfig(),
            reset_sampling=reset_sampling_config or ResetSamplingConfig(),
            recovery_counterfactual_other_sample_count=recovery_counterfactual_other_sample_count,
            counterfactual_opponent_response_samples=counterfactual_opponent_response_samples,
            recovery_counterfactual_expected_response_target=recovery_counterfactual_expected_response_target,
            recovery_full_diagnostics_probability=recovery_full_diagnostics_probability,
            use_shot_cf=use_shot_cf,
            shot_cf_coef=shot_cf_coef,
            shot_cf_top_m=shot_cf_top_m,
            shot_cf_num_modes=shot_cf_num_modes,
            shot_cf_min_landing_dist=shot_cf_min_landing_dist,
            shot_cf_depth=shot_cf_depth,
            shot_cf_include_chosen=shot_cf_include_chosen,
            shot_cf_skip_low_diversity=shot_cf_skip_low_diversity,
            shot_cf_min_modes=shot_cf_min_modes,
            shot_cf_value_detach=shot_cf_value_detach,
            shot_cf_normalize=shot_cf_normalize,
            shot_cf_debug_log=shot_cf_debug_log,
        ),
        discrete_action_config=discrete_action_config,
        opponent=opponent,
        seed=seed,
    )
