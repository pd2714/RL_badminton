from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from badminton1d.dynamics import (
    candidate_intercept_points,
    landing_position,
    prepare_shot,
    reaction_miss_probability,
    step_stage,
)
from badminton1d.obs import ObservationConfig, ObservationEncoder
from badminton1d.policy import ROLE_IS_RECEIVER_INDEX, STAGE_PROGRESS_INDEX, _state_from_observation_row
from badminton1d.shot_cf import ShotCFCandidate, select_diverse_shot_candidates
from badminton1d.state import ShotAction, StageRecord


class RecoveryFactorizedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_log_prob_shot: th.Tensor
    old_log_prob_recovery: th.Tensor
    recovery_advantages: th.Tensor
    recovery_loss_mask: th.Tensor
    recovery_distribution_targets: th.Tensor
    recovery_distribution_mask: th.Tensor
    shot_cf_advantages: th.Tensor
    shot_cf_loss_mask: th.Tensor


class RecoveryFactorizedRolloutBuffer(RolloutBuffer):
    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs)
        self.log_probs_shot = np.zeros(shape, dtype=np.float32)
        self.log_probs_recovery = np.zeros(shape, dtype=np.float32)
        self.recovery_advantages = np.zeros(shape, dtype=np.float32)
        self.recovery_loss_mask = np.zeros(shape, dtype=np.float32)
        self.shot_cf_advantages = np.zeros(shape, dtype=np.float32)
        self.shot_cf_loss_mask = np.zeros(shape, dtype=np.float32)
        distribution_bin_count = getattr(self, "_recovery_distribution_bin_count", None)
        if distribution_bin_count is None:
            self.recovery_distribution_targets = None
            self.recovery_distribution_mask = None
        else:
            distribution_shape = (*shape, int(distribution_bin_count))
            self.recovery_distribution_targets = np.zeros(distribution_shape, dtype=np.float32)
            self.recovery_distribution_mask = np.zeros(distribution_shape, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        *,
        log_prob_shot: th.Tensor | None = None,
        log_prob_recovery: th.Tensor | None = None,
        recovery_advantage: th.Tensor | None = None,
        recovery_loss_mask: th.Tensor | None = None,
        recovery_distribution_target: th.Tensor | None = None,
        recovery_distribution_mask: th.Tensor | None = None,
        shot_cf_advantage: th.Tensor | None = None,
        shot_cf_loss_mask: th.Tensor | None = None,
    ) -> None:
        pos = self.pos
        if log_prob_shot is None:
            log_prob_shot = log_prob
        if log_prob_recovery is None:
            log_prob_recovery = th.zeros_like(log_prob)
        if recovery_advantage is None:
            recovery_advantage = th.zeros_like(log_prob)
        if recovery_loss_mask is None:
            recovery_loss_mask = th.zeros_like(log_prob)
        if shot_cf_advantage is None:
            shot_cf_advantage = th.zeros_like(log_prob)
        if shot_cf_loss_mask is None:
            shot_cf_loss_mask = th.zeros_like(log_prob)

        super().add(obs, action, reward, episode_start, value, log_prob)
        self.log_probs_shot[pos] = log_prob_shot.clone().cpu().numpy().flatten()
        self.log_probs_recovery[pos] = log_prob_recovery.clone().cpu().numpy().flatten()
        self.recovery_advantages[pos] = recovery_advantage.clone().cpu().numpy().flatten()
        self.recovery_loss_mask[pos] = recovery_loss_mask.clone().cpu().numpy().flatten()
        self.shot_cf_advantages[pos] = shot_cf_advantage.clone().cpu().numpy().flatten()
        self.shot_cf_loss_mask[pos] = shot_cf_loss_mask.clone().cpu().numpy().flatten()
        if recovery_distribution_target is not None and recovery_distribution_mask is not None:
            target_np = recovery_distribution_target.detach().cpu().numpy()
            mask_np = recovery_distribution_mask.detach().cpu().numpy()
            if target_np.ndim != 2 or target_np.shape != mask_np.shape or target_np.shape[0] != self.n_envs:
                raise ValueError("Recovery distribution targets and masks must have shape (n_envs, recovery_bins).")
            if self.recovery_distribution_targets is None or self.recovery_distribution_mask is None:
                self._recovery_distribution_bin_count = int(target_np.shape[1])
                distribution_shape = (self.buffer_size, self.n_envs, self._recovery_distribution_bin_count)
                self.recovery_distribution_targets = np.zeros(distribution_shape, dtype=np.float32)
                self.recovery_distribution_mask = np.zeros(distribution_shape, dtype=np.float32)
            if target_np.shape[1] != self.recovery_distribution_targets.shape[2]:
                raise ValueError("Recovery distribution bin count changed within a rollout.")
            self.recovery_distribution_targets[pos] = target_np
            self.recovery_distribution_mask[pos] = mask_np

    def get(self, batch_size: int | None = None) -> Generator[RecoveryFactorizedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            if self.recovery_distribution_targets is None or self.recovery_distribution_mask is None:
                distribution_shape = (self.buffer_size, self.n_envs, 0)
                self.recovery_distribution_targets = np.zeros(distribution_shape, dtype=np.float32)
                self.recovery_distribution_mask = np.zeros(distribution_shape, dtype=np.float32)
            tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "log_probs_shot",
                "log_probs_recovery",
                "recovery_advantages",
                "recovery_loss_mask",
                "recovery_distribution_targets",
                "recovery_distribution_mask",
                "shot_cf_advantages",
                "shot_cf_loss_mask",
            ]
            for tensor in tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> RecoveryFactorizedRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.log_probs_shot[batch_inds].flatten(),
            self.log_probs_recovery[batch_inds].flatten(),
            self.recovery_advantages[batch_inds].flatten(),
            self.recovery_loss_mask[batch_inds].flatten(),
            self.recovery_distribution_targets[batch_inds],
            self.recovery_distribution_mask[batch_inds],
            self.shot_cf_advantages[batch_inds].flatten(),
            self.shot_cf_loss_mask[batch_inds].flatten(),
        )
        return RecoveryFactorizedRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class RecoveryFactorizedPPO(PPO):
    """PPO variant that can train recovery with a recovery-only advantage."""

    def __init__(
        self,
        *args,
        use_recovery_factorized_advantage: bool = False,
        recovery_counterfactual_baseline: str = "average",
        recovery_counterfactual_advantage_coef: float = 0.05,
        recovery_counterfactual_distribution_coef: float = 0.0,
        recovery_counterfactual_distribution_temperature: float = 0.25,
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
        **kwargs,
    ) -> None:
        self.use_recovery_factorized_advantage = bool(use_recovery_factorized_advantage)
        self.use_shot_cf = bool(use_shot_cf)
        if recovery_counterfactual_baseline not in {"average", "best"}:
            raise ValueError(
                "recovery_counterfactual_baseline must be either "
                f"'average' or 'best', got {recovery_counterfactual_baseline!r}"
            )
        self.recovery_counterfactual_baseline = recovery_counterfactual_baseline
        if recovery_counterfactual_advantage_coef < 0.0:
            raise ValueError("recovery_counterfactual_advantage_coef must be zero or greater")
        self.recovery_counterfactual_advantage_coef = float(recovery_counterfactual_advantage_coef)
        if recovery_counterfactual_distribution_coef < 0.0:
            raise ValueError("recovery_counterfactual_distribution_coef must be zero or greater")
        if recovery_counterfactual_distribution_temperature <= 0.0:
            raise ValueError("recovery_counterfactual_distribution_temperature must be positive")
        self.recovery_counterfactual_distribution_coef = float(recovery_counterfactual_distribution_coef)
        self.recovery_counterfactual_distribution_temperature = float(
            recovery_counterfactual_distribution_temperature
        )
        if shot_cf_coef < 0.0:
            raise ValueError("shot_cf_coef must be zero or greater")
        if shot_cf_top_m <= 0:
            raise ValueError("shot_cf_top_m must be positive")
        if shot_cf_num_modes <= 0:
            raise ValueError("shot_cf_num_modes must be positive")
        if shot_cf_min_landing_dist < 0.0:
            raise ValueError("shot_cf_min_landing_dist must be non-negative")
        if shot_cf_depth != 1:
            raise ValueError("Only shot_cf_depth=1 is implemented")
        if shot_cf_min_modes <= 0:
            raise ValueError("shot_cf_min_modes must be positive")
        self.shot_cf_coef = float(shot_cf_coef)
        self.shot_cf_top_m = int(shot_cf_top_m)
        self.shot_cf_num_modes = int(shot_cf_num_modes)
        self.shot_cf_min_landing_dist = float(shot_cf_min_landing_dist)
        self.shot_cf_depth = int(shot_cf_depth)
        self.shot_cf_include_chosen = bool(shot_cf_include_chosen)
        self.shot_cf_skip_low_diversity = bool(shot_cf_skip_low_diversity)
        self.shot_cf_min_modes = int(shot_cf_min_modes)
        self.shot_cf_value_detach = bool(shot_cf_value_detach)
        self.shot_cf_normalize = bool(shot_cf_normalize)
        self.shot_cf_debug_log = bool(shot_cf_debug_log)
        self._last_shot_cf_stats: dict[str, float] = {}
        if (self.use_recovery_factorized_advantage or self.use_shot_cf) and kwargs.get("rollout_buffer_class") is None:
            kwargs["rollout_buffer_class"] = RecoveryFactorizedRolloutBuffer
        super().__init__(*args, **kwargs)

    def _can_use_recovery_factorization(self) -> bool:
        return (
            (self.use_recovery_factorized_advantage or self.use_shot_cf)
            and isinstance(self.rollout_buffer, RecoveryFactorizedRolloutBuffer)
            and getattr(self.policy, "output_mode", None) == "conditional_prob"
            and hasattr(self.policy, "evaluate_recovery_factorized_actions")
        )

    def _normalize_recovery_advantage(self, advantages: th.Tensor, mask: th.Tensor) -> th.Tensor:
        mask_bool = mask > 0.5
        normalized = th.zeros_like(advantages)
        if not th.any(mask_bool):
            return normalized
        active = advantages[mask_bool]
        if self.normalize_advantage and active.numel() > 1:
            active = (active - active.mean()) / (active.std() + 1e-8)
        normalized[mask_bool] = active
        return normalized

    def _recovery_probability_by_bin(
        self,
        obs_tensor: th.Tensor | None,
        actions_tensor: th.Tensor | None,
    ) -> np.ndarray | None:
        if obs_tensor is None or actions_tensor is None:
            return None
        required = (
            "_conditional_decompose_actions",
            "_conditional_component_logits",
        )
        if not all(hasattr(self.policy, name) for name in required):
            return None
        try:
            features = self.policy.extract_features(obs_tensor)
            if self.policy.share_features_extractor:
                latent_pi, _ = self.policy.mlp_extractor(features)
            else:
                pi_features, _ = features
                latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            phi, theta, speed, _ = self.policy._conditional_decompose_actions(actions_tensor)
            _, _, _, recovery_logits, _ = self.policy._conditional_component_logits(
                obs_tensor,
                latent_pi,
                phi=phi,
                theta=theta,
                speed=speed,
            )
            if recovery_logits is None:
                return None
            return th.softmax(recovery_logits, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return None

    def _sampled_recovery_distribution_target(
        self,
        scores: np.ndarray,
        sampled_indices: np.ndarray,
        *,
        bin_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = np.zeros(max(int(bin_count), 0), dtype=np.float32)
        mask = np.zeros_like(target)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        sampled_indices = np.asarray(sampled_indices, dtype=int).reshape(-1)
        if scores.shape[0] != sampled_indices.shape[0] or scores.size < 2 or target.size == 0:
            return target, mask

        valid = np.isfinite(scores) & (0 <= sampled_indices) & (sampled_indices < target.size)
        if np.count_nonzero(valid) < 2:
            return target, mask
        valid_scores = scores[valid]
        valid_indices = sampled_indices[valid]
        if np.unique(valid_indices).size != valid_indices.size:
            return target, mask

        scaled_scores = valid_scores / self.recovery_counterfactual_distribution_temperature
        weights = np.exp(scaled_scores - float(np.max(scaled_scores)))
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return target, mask
        target[valid_indices] = weights / total
        mask[valid_indices] = 1.0
        return target, mask

    def _sampled_recovery_distribution_loss(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        targets: th.Tensor,
        mask: th.Tensor,
    ) -> th.Tensor:
        zero = observations.sum() * 0.0
        if targets.ndim != 2 or mask.shape != targets.shape or targets.shape[1] == 0:
            return zero
        required = (
            "_conditional_decompose_actions",
            "_conditional_component_logits",
        )
        if not all(hasattr(self.policy, name) for name in required):
            return zero

        features = self.policy.extract_features(observations)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        phi, theta, speed, _ = self.policy._conditional_decompose_actions(actions)
        _, _, _, recovery_logits, _ = self.policy._conditional_component_logits(
            observations,
            latent_pi,
            phi=phi,
            theta=theta,
            speed=speed,
        )
        if recovery_logits is None or recovery_logits.shape != targets.shape:
            return zero

        candidate_mask = mask > 0.5
        active_rows = (candidate_mask.sum(dim=1) >= 2) & (targets.sum(dim=1) > 0.0)
        if not th.any(active_rows):
            return zero
        active_mask = candidate_mask[active_rows]
        active_targets = targets[active_rows]
        active_targets = active_targets / active_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
        active_logits = recovery_logits[active_rows].masked_fill(~active_mask, -th.inf)
        log_probabilities = F.log_softmax(active_logits, dim=1)
        selected_log_probabilities = th.where(active_mask, log_probabilities, th.zeros_like(log_probabilities))
        return -(active_targets * selected_log_probabilities).sum(dim=1).mean()

    def _annotate_recovery_diagnostics(
        self,
        info: dict,
        *,
        chosen_training_advantage: float,
        scores: np.ndarray,
        chosen_index: int,
        no_feasible: np.ndarray,
        sampled_indices: list[int] | None,
        recovery_probabilities: np.ndarray | None,
    ) -> None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        no_feasible = np.asarray(no_feasible, dtype=bool).reshape(-1)
        if scores.size == 0 or not 0 <= chosen_index < scores.size:
            return

        x_bins = int(info.get("recovery_factorized_counterfactual_x_bins", 0) or 0)
        y_bins = int(info.get("recovery_factorized_counterfactual_y_bins", 0) or 0)
        full_bin_count = x_bins * y_bins if x_bins > 0 and y_bins > 0 else scores.size
        full_grid = scores.size == full_bin_count and full_bin_count > 0
        chosen_flat_index = int(info.get("recovery_factorized_counterfactual_chosen_flat_index", chosen_index))
        if sampled_indices is not None and 0 <= chosen_index < len(sampled_indices):
            chosen_flat_index = int(sampled_indices[chosen_index])
        if full_grid and not 0 <= chosen_flat_index < full_bin_count:
            chosen_flat_index = chosen_index

        if no_feasible.shape[0] != scores.shape[0]:
            finite_targets = np.zeros(scores.shape[0], dtype=bool)
        else:
            finite_targets = no_feasible

        chosen_score = float(scores[chosen_index])
        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        eps = 1e-8
        rank = int(1 + np.sum(scores > chosen_score + eps))
        a_rec_all_bins = float(chosen_score - mean_score)
        diagnostic: dict[str, object] = {
            "x_bins": x_bins,
            "y_bins": y_bins,
            "bin_count": int(full_bin_count),
            "evaluated_bin_count": int(scores.size),
            "sampled_only": bool(not full_grid),
            "chosen_flat_index": chosen_flat_index,
            "chosen_x_index": int(chosen_flat_index // y_bins) if y_bins > 0 else -1,
            "chosen_y_index": int(chosen_flat_index % y_bins) if y_bins > 0 else -1,
            "chosen_score": chosen_score,
            "mean_score": mean_score,
            "max_score": max_score,
            "chosen_rank": rank,
            "chosen_rank_fraction": float(rank / max(float(scores.size), 1.0)),
            "chosen_above_average": bool(chosen_score > mean_score),
            "chosen_best": bool(chosen_score >= max_score - eps),
            "a_rec": a_rec_all_bins,
            "training_recovery_advantage": float(chosen_training_advantage),
        }
        if sampled_indices is not None:
            diagnostic["sampled_indices"] = [int(index) for index in sampled_indices]
            diagnostic["sampled_scores"] = scores.astype(float).tolist()
            diagnostic["sampled_no_feasible"] = finite_targets.astype(int).tolist()
        if full_grid and x_bins > 0 and y_bins > 0:
            diagnostic["score_grid"] = scores.reshape(x_bins, y_bins).astype(float).tolist()
            diagnostic["no_feasible_grid"] = finite_targets.reshape(x_bins, y_bins).astype(int).tolist()
        if recovery_probabilities is not None:
            if recovery_probabilities.shape[0] == full_bin_count:
                if full_grid and x_bins > 0 and y_bins > 0:
                    diagnostic["policy_probability_grid"] = (
                        recovery_probabilities.reshape(x_bins, y_bins).astype(float).tolist()
                    )
                if 0 <= chosen_flat_index < recovery_probabilities.shape[0]:
                    diagnostic["chosen_probability"] = float(recovery_probabilities[chosen_flat_index])
                if sampled_indices is not None:
                    valid_indices = [
                        int(index)
                        for index in sampled_indices
                        if 0 <= int(index) < recovery_probabilities.shape[0]
                    ]
                    diagnostic["sampled_policy_probabilities"] = (
                        recovery_probabilities[valid_indices].astype(float).tolist()
                    )
            elif recovery_probabilities.shape[0] == scores.shape[0]:
                diagnostic["sampled_policy_probabilities"] = recovery_probabilities.astype(float).tolist()
                diagnostic["chosen_probability"] = float(recovery_probabilities[chosen_index])
        info["recovery_factorized_diagnostics"] = diagnostic

    @staticmethod
    def _counterfactual_value_observations(
        info: dict,
        cf_obs_arr: np.ndarray,
        *,
        prefix: str,
    ) -> np.ndarray:
        expected_obs = info.get(f"{prefix}_expected_observations")
        if (
            bool(info.get("recovery_factorized_counterfactual_expected_response_target", False))
            and expected_obs is not None
        ):
            expected_obs_arr = np.asarray(expected_obs, dtype=np.float32)
            if expected_obs_arr.shape == cf_obs_arr.shape:
                return expected_obs_arr
        return cf_obs_arr

    @staticmethod
    def _apply_counterfactual_targets(
        info: dict,
        values_np: np.ndarray,
        *,
        prefix: str,
        default_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(values_np, dtype=np.float32).reshape(-1).copy()
        if bool(info.get("recovery_factorized_counterfactual_expected_response_target", False)):
            miss_probabilities = np.asarray(
                info.get(
                    f"{prefix}_expected_miss_probabilities",
                    np.full(scores.shape[0], np.nan, dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1)
            no_miss_targets = np.asarray(
                info.get(
                    f"{prefix}_expected_no_miss_targets",
                    np.full(scores.shape[0], np.nan, dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1)
            if miss_probabilities.shape[0] == scores.shape[0]:
                miss_probabilities = np.nan_to_num(miss_probabilities, nan=0.0)
                miss_probabilities = np.clip(miss_probabilities, 0.0, 1.0)
                if no_miss_targets.shape[0] == scores.shape[0]:
                    finite_no_miss_targets = np.isfinite(no_miss_targets)
                    scores[finite_no_miss_targets] = no_miss_targets[finite_no_miss_targets]
                loss_reward = float(info.get("recovery_factorized_counterfactual_loss_reward", -1.0))
                scores = miss_probabilities * loss_reward + (1.0 - miss_probabilities) * scores

        targets = np.asarray(default_targets, dtype=np.float32).reshape(-1)
        if targets.shape[0] == scores.shape[0]:
            finite_targets = np.isfinite(targets)
            scores[finite_targets] = targets[finite_targets]
        else:
            finite_targets = np.zeros(scores.shape[0], dtype=bool)
        return scores, finite_targets

    @staticmethod
    def _aggregate_counterfactual_response_scores(
        info: dict,
        scores: np.ndarray,
        finite_targets: np.ndarray,
        *,
        prefix: str,
        expected_bin_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        finite_targets = np.asarray(finite_targets, dtype=bool).reshape(-1)
        expected_bin_count = int(expected_bin_count)
        counts = np.asarray(info.get(f"{prefix}_response_counts", []), dtype=int).reshape(-1)
        weights = np.asarray(info.get(f"{prefix}_response_weights", []), dtype=np.float32).reshape(-1)
        if (
            expected_bin_count <= 0
            or counts.shape[0] != expected_bin_count
            or int(np.sum(counts)) != scores.shape[0]
            or weights.shape[0] != scores.shape[0]
        ):
            return scores, finite_targets

        aggregated_scores = np.zeros(expected_bin_count, dtype=np.float32)
        aggregated_targets = np.zeros(expected_bin_count, dtype=bool)
        offset = 0
        for bin_index, count in enumerate(counts):
            count = int(count)
            if count <= 0:
                aggregated_scores[bin_index] = 0.0
                continue
            end = offset + count
            group_scores = scores[offset:end]
            group_weights = weights[offset:end].astype(np.float32, copy=False)
            total_weight = float(np.sum(group_weights))
            if total_weight <= 0.0 or not np.isfinite(total_weight):
                group_weights = np.full(count, 1.0 / float(count), dtype=np.float32)
            else:
                group_weights = group_weights / total_weight
            aggregated_scores[bin_index] = float(np.sum(group_scores * group_weights))
            aggregated_targets[bin_index] = bool(np.any(finite_targets[offset:end]))
            offset = end
        return aggregated_scores, aggregated_targets

    def _recovery_advantage_from_transitions(
        self,
        values_before: th.Tensor,
        infos: list[dict],
        next_obs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        current_obs_tensor: th.Tensor | None = None,
        actions_tensor: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        before = values_before.flatten()
        recovery_bin_count = max(int(getattr(self.policy, "_conditional_recovery_count", 0)), 0)
        distribution_target_np = np.zeros((len(infos), recovery_bin_count), dtype=np.float32)
        distribution_mask_np = np.zeros_like(distribution_target_np)
        mask_np = np.asarray(
            [bool(info.get("recovery_factorized_action", False)) for info in infos],
            dtype=np.float32,
        )
        if not np.any(mask_np):
            zeros = th.zeros_like(before)
            return (
                zeros,
                zeros,
                th.as_tensor(distribution_target_np, dtype=before.dtype, device=before.device),
                th.as_tensor(distribution_mask_np, dtype=before.dtype, device=before.device),
            )

        counterfactual_advantage_np = np.full_like(mask_np, np.nan, dtype=np.float32)
        jobs: list[dict[str, object]] = []
        all_cf_obs: list[np.ndarray] = []
        all_full_diagnostic_obs: list[np.ndarray] = []
        with th.no_grad():
            for index, info in enumerate(infos):
                cf_obs = info.get("recovery_factorized_counterfactual_observations")
                if cf_obs is None:
                    continue
                cf_obs_arr = np.asarray(cf_obs, dtype=np.float32)
                if cf_obs_arr.ndim != 2 or cf_obs_arr.shape[0] == 0:
                    continue
                chosen_index = int(info.get("recovery_factorized_counterfactual_chosen_index", -1))
                if not 0 <= chosen_index < cf_obs_arr.shape[0]:
                    continue
                full_cf_obs_arr: np.ndarray | None = None
                full_cf_obs = info.get("recovery_factorized_counterfactual_full_observations")
                if full_cf_obs is not None:
                    candidate = np.asarray(full_cf_obs, dtype=np.float32)
                    if candidate.ndim == 2 and candidate.shape[0] > 0:
                        full_cf_obs_arr = candidate
                cf_value_obs = self._counterfactual_value_observations(
                    info,
                    cf_obs_arr,
                    prefix="recovery_factorized_counterfactual",
                )
                full_cf_value_obs = (
                    None
                    if full_cf_obs_arr is None
                    else self._counterfactual_value_observations(
                        info,
                        full_cf_obs_arr,
                        prefix="recovery_factorized_counterfactual_full",
                    )
                )
                job: dict[str, object] = {
                    "index": index,
                    "info": info,
                    "cf_obs": cf_obs_arr,
                    "cf_value_obs": cf_value_obs,
                    "full_cf_obs": full_cf_obs_arr,
                    "full_cf_value_obs": full_cf_value_obs,
                    "chosen_index": chosen_index,
                }
                jobs.append(job)
                all_cf_obs.append(cf_value_obs)
                if isinstance(full_cf_value_obs, np.ndarray):
                    all_full_diagnostic_obs.append(full_cf_value_obs)

            if all_cf_obs:
                batched_cf_obs = np.concatenate(all_cf_obs, axis=0)
                batched_values = self.policy.predict_values(obs_as_tensor(batched_cf_obs, self.device)).flatten()
                batched_values_np = batched_values.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                batched_values_np = np.asarray([], dtype=np.float32)

            if all_full_diagnostic_obs:
                batched_full_obs = np.concatenate(all_full_diagnostic_obs, axis=0)
                batched_full_values = self.policy.predict_values(obs_as_tensor(batched_full_obs, self.device)).flatten()
                batched_full_values_np = batched_full_values.detach().cpu().numpy().astype(np.float32, copy=False)
                recovery_probability_np = self._recovery_probability_by_bin(current_obs_tensor, actions_tensor)
            else:
                batched_full_values_np = np.asarray([], dtype=np.float32)
                recovery_probability_np = None

            offset = 0
            full_offset = 0
            for job in jobs:
                index = int(job["index"])
                info = job["info"]
                assert isinstance(info, dict)
                cf_obs_arr = job["cf_obs"]
                assert isinstance(cf_obs_arr, np.ndarray)
                chosen_index = int(job["chosen_index"])
                end = offset + cf_obs_arr.shape[0]
                cf_values_np = batched_values_np[offset:end]
                offset = end
                cf_targets_np = np.asarray(
                    info.get(
                        "recovery_factorized_counterfactual_targets",
                        np.full(cf_obs_arr.shape[0], np.nan, dtype=np.float32),
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                cf_values_np, finite_targets = self._apply_counterfactual_targets(
                    info,
                    cf_values_np,
                    prefix="recovery_factorized_counterfactual",
                    default_targets=cf_targets_np,
                )
                sampled_indices_np = np.asarray(
                    info.get("recovery_factorized_counterfactual_sampled_indices", []),
                    dtype=int,
                ).reshape(-1)
                cf_values_np, finite_targets = self._aggregate_counterfactual_response_scores(
                    info,
                    cf_values_np,
                    finite_targets,
                    prefix="recovery_factorized_counterfactual",
                    expected_bin_count=int(sampled_indices_np.shape[0]),
                )
                baseline_indices_np = np.asarray(
                    info.get("recovery_factorized_counterfactual_baseline_indices", []),
                    dtype=int,
                ).reshape(-1)
                baseline_indices_np = baseline_indices_np[
                    (0 <= baseline_indices_np) & (baseline_indices_np < cf_values_np.shape[0])
                ]
                baseline_values_np = (
                    cf_values_np[baseline_indices_np]
                    if baseline_indices_np.size
                    else cf_values_np
                )
                baseline = (
                    float(np.max(baseline_values_np))
                    if self.recovery_counterfactual_baseline == "best"
                    else float(np.mean(baseline_values_np))
                )
                counterfactual_advantage_np[index] = float(cf_values_np[chosen_index] - baseline)
                (
                    distribution_target_np[index],
                    distribution_mask_np[index],
                ) = self._sampled_recovery_distribution_target(
                    cf_values_np,
                    sampled_indices_np,
                    bin_count=recovery_bin_count,
                )

                full_cf_obs_arr = job.get("full_cf_obs")
                if isinstance(full_cf_obs_arr, np.ndarray):
                    full_end = full_offset + full_cf_obs_arr.shape[0]
                    full_values_np = batched_full_values_np[full_offset:full_end]
                    full_offset = full_end
                    full_targets_np = np.asarray(
                        info.get(
                            "recovery_factorized_counterfactual_full_targets",
                            np.full(full_cf_obs_arr.shape[0], np.nan, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    full_values_np, full_finite_targets = self._apply_counterfactual_targets(
                        info,
                        full_values_np,
                        prefix="recovery_factorized_counterfactual_full",
                        default_targets=full_targets_np,
                    )
                    full_bin_count = int(info.get("recovery_factorized_counterfactual_x_bins", 0) or 0) * int(
                        info.get("recovery_factorized_counterfactual_y_bins", 0) or 0
                    )
                    if full_bin_count <= 0:
                        full_bin_count = int(full_cf_obs_arr.shape[0])
                    full_values_np, full_finite_targets = self._aggregate_counterfactual_response_scores(
                        info,
                        full_values_np,
                        full_finite_targets,
                        prefix="recovery_factorized_counterfactual_full",
                        expected_bin_count=full_bin_count,
                    )
                    chosen_flat_index = int(
                        info.get("recovery_factorized_counterfactual_chosen_flat_index", chosen_index)
                    )
                    probabilities = (
                        None
                        if recovery_probability_np is None or index >= recovery_probability_np.shape[0]
                        else recovery_probability_np[index]
                    )
                    self._annotate_recovery_diagnostics(
                        info,
                        chosen_training_advantage=float(counterfactual_advantage_np[index]),
                        scores=full_values_np,
                        chosen_index=chosen_flat_index,
                        no_feasible=full_finite_targets,
                        sampled_indices=None,
                        recovery_probabilities=probabilities,
                    )

        next_obs_arr = np.asarray(next_obs)
        with th.no_grad():
            after_values = self.policy.predict_values(obs_as_tensor(next_obs_arr, self.device)).flatten()

        explicit_target_np = np.asarray(
            [
                float(info["recovery_factorized_target"])
                if "recovery_factorized_target" in info
                else np.nan
                for info in infos
            ],
            dtype=np.float32,
        )
        fallback_done_np = np.asarray(dones, dtype=bool) & ~np.isfinite(explicit_target_np)
        explicit_target_np[fallback_done_np] = np.asarray(rewards, dtype=np.float32)[fallback_done_np]

        explicit_mask = th.as_tensor(
            np.isfinite(explicit_target_np),
            dtype=th.bool,
            device=before.device,
        )
        explicit_target = th.as_tensor(
            np.nan_to_num(explicit_target_np, nan=0.0),
            dtype=before.dtype,
            device=before.device,
        )
        target_values = th.where(explicit_mask, explicit_target, after_values)
        mask = th.as_tensor(mask_np, dtype=before.dtype, device=before.device)
        # PPO credit still anchors recovery to the sampled transition outcome.
        # The counterfactual term adds local recovery-bin preference instead of
        # replacing the episode-return signal.
        recovery_advantage = target_values - before
        counterfactual_mask = th.as_tensor(
            np.isfinite(counterfactual_advantage_np),
            dtype=th.bool,
            device=before.device,
        )
        counterfactual_advantage = th.as_tensor(
            np.nan_to_num(counterfactual_advantage_np, nan=0.0),
            dtype=before.dtype,
            device=before.device,
        )
        recovery_advantage = recovery_advantage + self.recovery_counterfactual_advantage_coef * th.where(
            counterfactual_mask,
            counterfactual_advantage,
            th.zeros_like(counterfactual_advantage),
        )
        recovery_advantage = recovery_advantage * mask
        return (
            recovery_advantage,
            mask,
            th.as_tensor(distribution_target_np, dtype=before.dtype, device=before.device),
            th.as_tensor(distribution_mask_np, dtype=before.dtype, device=before.device),
        )

    def _normalize_shot_cf_advantage(self, advantages: th.Tensor, mask: th.Tensor) -> th.Tensor:
        mask_bool = mask > 0.5
        normalized = th.zeros_like(advantages)
        if not th.any(mask_bool):
            return normalized
        active = advantages[mask_bool]
        if self.shot_cf_normalize and active.numel() > 1:
            active = (active - active.mean()) / (active.std() + 1e-8)
        normalized[mask_bool] = active
        return normalized

    def _shot_cf_candidate_actions_for_row(
        self,
        obs_row: th.Tensor,
        latent_pi_row: th.Tensor,
        chosen_action: int,
        *,
        train_side: str,
        top_m: int,
    ) -> tuple[list[ShotCFCandidate], ShotCFCandidate | None]:
        mapper = getattr(self.policy, "_action_mapper", None)
        if mapper is None or float(obs_row[ROLE_IS_RECEIVER_INDEX]) >= 0.5:
            return [], None

        stage_index = 0 if float(obs_row[STAGE_PROGRESS_INDEX]) <= 1e-6 else 1
        state = _state_from_observation_row(obs_row, mapper.config, stage_index=stage_index)
        legal_full = self.policy._conditional_hitter_legal_mask(obs_row.reshape(1, -1))[0].detach().cpu().numpy()
        legal_shots = legal_full.any(axis=3)
        if not np.any(legal_shots):
            return [], None

        obs_batch = obs_row.reshape(1, -1)
        latent_batch = latent_pi_row.reshape(1, -1)
        phi_logits, _, _, _, _ = self.policy._conditional_component_logits(obs_batch, latent_batch)
        phi_logp = F.log_softmax(phi_logits[0], dim=0)
        shot_rows: list[tuple[float, int, int, int]] = []
        for phi_index in range(int(getattr(self.policy, "_conditional_phi_count", 0))):
            phi_tensor = th.as_tensor([phi_index], dtype=th.long, device=obs_row.device)
            _, theta_logits, _, _, _ = self.policy._conditional_component_logits(
                obs_batch,
                latent_batch,
                phi=phi_tensor,
            )
            if theta_logits is None:
                continue
            theta_logp = F.log_softmax(theta_logits[0], dim=0)
            for theta_index in range(int(getattr(self.policy, "_conditional_theta_count", 0))):
                if not np.any(legal_shots[phi_index, theta_index]):
                    continue
                theta_tensor = th.as_tensor([theta_index], dtype=th.long, device=obs_row.device)
                _, _, speed_logits, _, _ = self.policy._conditional_component_logits(
                    obs_batch,
                    latent_batch,
                    phi=phi_tensor,
                    theta=theta_tensor,
                )
                if speed_logits is None:
                    continue
                speed_logp = F.log_softmax(speed_logits[0], dim=0)
                for speed_index in range(int(getattr(self.policy, "_conditional_speed_count", 0))):
                    if not bool(legal_shots[phi_index, theta_index, speed_index]):
                        continue
                    log_probability = float(
                        phi_logp[phi_index].item()
                        + theta_logp[theta_index].item()
                        + speed_logp[speed_index].item()
                    )
                    shot_rows.append((log_probability, phi_index, theta_index, speed_index))

        if not shot_rows:
            return [], None
        shot_rows.sort(key=lambda row: row[0], reverse=True)
        selected_rows = shot_rows[: max(int(top_m), 1)]
        max_logp = float(selected_rows[0][0])
        weights = np.asarray([np.exp(row[0] - max_logp) for row in selected_rows], dtype=np.float64)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            probabilities = np.full(weights.shape[0], 1.0 / max(weights.shape[0], 1), dtype=np.float64)
        else:
            probabilities = weights / total_weight

        def _candidate_from_indices(
            phi_index: int,
            theta_index: int,
            speed_index: int,
            log_probability: float,
            probability: float,
            *,
            forced_flat_index: int | None = None,
        ) -> ShotCFCandidate | None:
            phi_tensor = th.as_tensor([phi_index], dtype=th.long, device=obs_row.device)
            theta_tensor = th.as_tensor([theta_index], dtype=th.long, device=obs_row.device)
            speed_tensor = th.as_tensor([speed_index], dtype=th.long, device=obs_row.device)
            _, _, _, recovery_logits, _ = self.policy._conditional_component_logits(
                obs_batch,
                latent_batch,
                phi=phi_tensor,
                theta=theta_tensor,
                speed=speed_tensor,
            )
            if recovery_logits is None:
                return None
            recovery_mask = th.as_tensor(
                legal_full[phi_index, theta_index, speed_index],
                dtype=th.bool,
                device=obs_row.device,
            )
            if not th.any(recovery_mask):
                return None
            recovery_logits_row = recovery_logits[0].masked_fill(~recovery_mask, -th.inf)
            recovery_index = int(th.argmax(recovery_logits_row).item())
            flat_index = int(
                forced_flat_index
                if forced_flat_index is not None
                else (
                    ((phi_index * self.policy._conditional_theta_count + theta_index)
                     * self.policy._conditional_speed_count + speed_index)
                    * self.policy._conditional_recovery_count
                    + recovery_index
                )
            )
            try:
                decode = mapper.decode_hitter_for_agent(flat_index, state, train_side)
                landing_x, landing_y = landing_position(state, decode.shot_action, mapper.config)
            except (RuntimeError, ValueError, IndexError, FloatingPointError):
                return None
            return ShotCFCandidate(
                flat_index=flat_index,
                shot_key=(int(phi_index), int(theta_index), int(speed_index)),
                log_probability=float(log_probability),
                probability=float(probability),
                shot_action=decode.shot_action,
                landing_x=float(landing_x),
                landing_y=float(landing_y),
            )

        candidates: list[ShotCFCandidate] = []
        for (log_probability, phi_index, theta_index, speed_index), probability in zip(selected_rows, probabilities):
            candidate = _candidate_from_indices(
                phi_index,
                theta_index,
                speed_index,
                log_probability,
                float(probability),
            )
            if candidate is not None:
                candidates.append(candidate)

        chosen_candidate = None
        try:
            phi, theta, speed, _ = self.policy._conditional_decompose_actions(
                th.as_tensor([chosen_action], dtype=th.long, device=obs_row.device)
            )
            chosen_phi = int(phi[0].item())
            chosen_theta = int(theta[0].item())
            chosen_speed = int(speed[0].item())
            matching_logp = next(
                (
                    float(log_probability)
                    for log_probability, row_phi, row_theta, row_speed in shot_rows
                    if (row_phi, row_theta, row_speed) == (chosen_phi, chosen_theta, chosen_speed)
                ),
                0.0,
            )
            chosen_candidate = _candidate_from_indices(
                chosen_phi,
                chosen_theta,
                chosen_speed,
                matching_logp,
                1.0,
                forced_flat_index=int(chosen_action),
            )
        except (RuntimeError, ValueError, IndexError, FloatingPointError):
            chosen_candidate = None
        return candidates, chosen_candidate

    def _shot_cf_terminal_reward(self, info: dict, winner: str | None) -> float:
        if winner == info.get("train_side", "left"):
            return float(info.get("shot_cf_win_reward", 1.0))
        return float(info.get("shot_cf_loss_reward", -1.0))

    def _shot_cf_lowest_reaction_risk_intercept_index(
        self,
        state,
        action: ShotAction,
        feasible: list[int],
        *,
        candidate_times: np.ndarray | None = None,
    ) -> int | None:
        if not feasible:
            return None
        config = getattr(getattr(self.policy, "_action_mapper", None), "config", None)
        if config is None:
            return int(feasible[0])
        if candidate_times is None:
            candidate_times, _, _, _ = candidate_intercept_points(state, action, config)
        best_index = None
        best_key: tuple[float, float] | None = None
        for index in feasible:
            if not 0 <= index < len(candidate_times):
                continue
            intercept_time = float(candidate_times[index])
            key = (reaction_miss_probability(intercept_time, config), -intercept_time)
            if best_key is None or key < best_key:
                best_key = key
                best_index = int(index)
        return best_index

    def _shot_cf_value_inputs_for_selection(
        self,
        selection,
        info: dict,
    ) -> tuple[list[np.ndarray | None], np.ndarray]:
        mapper = getattr(self.policy, "_action_mapper", None)
        if mapper is None:
            return [], np.asarray([], dtype=np.float32)
        record = info.get("last_record")
        if not isinstance(record, StageRecord):
            return [], np.asarray([], dtype=np.float32)
        config = mapper.config
        encoder = ObservationEncoder(
            config,
            ObservationConfig(
                include_feasible_mask=bool(info.get("include_feasible_mask", True)),
                include_reaction_risk_features=bool(info.get("include_reaction_risk_features", True)),
            ),
        )
        response_action = info.get("shot_cf_opponent_response_action")
        score_targets: list[float] = []
        observations: list[np.ndarray | None] = []
        for candidate in selection.candidates:
            try:
                projected = mapper.project_hitter_action(record.state_before, candidate.shot_action)
                prepared = projected.prepared_shot
                applied_action = prepared.validated_action.applied
                feasible = list(prepared.feasible_indices)
                intercept_index = self._shot_cf_lowest_reaction_risk_intercept_index(
                    record.state_before,
                    applied_action,
                    feasible,
                    candidate_times=prepared.candidate_times,
                )
                first_record = step_stage(
                    record.state_before,
                    projected.shot_action,
                    intercept_index,
                    config,
                    enable_reaction_miss=False,
                    prepared_shot=prepared,
                )
            except (RuntimeError, ValueError, IndexError, FloatingPointError):
                observations.append(None)
                score_targets.append(float(info.get("shot_cf_loss_reward", -1.0)))
                continue

            if first_record.next_state.rally_done:
                observations.append(None)
                score_targets.append(self._shot_cf_terminal_reward(info, first_record.next_state.winner))
                continue

            if not isinstance(response_action, ShotAction):
                observations.append(None)
                score_targets.append(float("nan"))
                continue

            try:
                response_prepared = prepare_shot(first_record.next_state, response_action, config)
            except (RuntimeError, ValueError, FloatingPointError):
                observations.append(None)
                score_targets.append(float(info.get("shot_cf_win_reward", 1.0)) * self.gamma)
                continue
            response_applied = response_prepared.validated_action.applied
            response_feasible = list(response_prepared.feasible_indices)
            if not response_feasible:
                observations.append(None)
                score_targets.append(float(info.get("shot_cf_loss_reward", -1.0)) * self.gamma)
                continue
            observations.append(
                encoder.encode(
                    state=first_record.next_state,
                    agent_side=info.get("train_side", "left"),
                    role="receiver",
                    server_side=info.get("server", "left"),
                    score_left=int(info.get("score_left", 0)),
                    score_right=int(info.get("score_right", 0)),
                    pending_action=response_applied,
                    feasible_indices=response_feasible,
                    prepared_shot=response_prepared,
                )
            )
            score_targets.append(float("nan"))
        return observations, np.asarray(score_targets, dtype=np.float32)

    def _shot_cf_advantage_from_transitions(
        self,
        current_obs_tensor: th.Tensor,
        actions_tensor: th.Tensor,
        infos: list[dict],
    ) -> tuple[th.Tensor, th.Tensor]:
        base = actions_tensor.flatten()
        advantages_np = np.zeros(base.shape[0], dtype=np.float32)
        mask_np = np.zeros(base.shape[0], dtype=np.float32)
        stats: dict[str, list[float]] = {
            "modes": [],
            "landing_distance": [],
            "advantage": [],
            "chosen_q": [],
            "candidate_mean_q": [],
            "chosen_best": [],
        }
        skipped = 0
        if not self.use_shot_cf:
            self._last_shot_cf_stats = {"computed_rate": 0.0, "computed_count": 0.0}
            return (
                th.as_tensor(advantages_np, dtype=current_obs_tensor.dtype, device=current_obs_tensor.device),
                th.as_tensor(mask_np, dtype=current_obs_tensor.dtype, device=current_obs_tensor.device),
            )

        obs_rows_for_value: list[np.ndarray] = []
        jobs: list[dict[str, object]] = []
        with th.no_grad():
            features = self.policy.extract_features(current_obs_tensor)
            if self.policy.share_features_extractor:
                latent_pi, _ = self.policy.mlp_extractor(features)
            else:
                pi_features, _ = features
                latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)

            for index, info in enumerate(infos):
                if index >= base.shape[0] or not bool(info.get("shot_cf_action", False)):
                    continue
                record = info.get("last_record")
                if not isinstance(record, StageRecord):
                    skipped += 1
                    continue
                candidates, chosen_candidate = self._shot_cf_candidate_actions_for_row(
                    current_obs_tensor[index],
                    latent_pi[index],
                    int(base[index].item()),
                    train_side=str(info.get("train_side", "left")),
                    top_m=self.shot_cf_top_m,
                )
                selection = select_diverse_shot_candidates(
                    candidates,
                    chosen_candidate=chosen_candidate,
                    num_modes=self.shot_cf_num_modes,
                    min_landing_dist=self.shot_cf_min_landing_dist,
                    include_chosen=self.shot_cf_include_chosen,
                    skip_low_diversity=self.shot_cf_skip_low_diversity,
                    min_modes=self.shot_cf_min_modes,
                )
                if selection.skipped:
                    skipped += 1
                    continue
                observations, score_targets = self._shot_cf_value_inputs_for_selection(selection, info)
                if len(observations) != len(selection.candidates):
                    skipped += 1
                    continue
                value_offsets: list[int] = []
                for observation in observations:
                    if observation is None:
                        value_offsets.append(-1)
                    else:
                        value_offsets.append(len(obs_rows_for_value))
                        obs_rows_for_value.append(observation)
                jobs.append(
                    {
                        "index": index,
                        "info": info,
                        "selection": selection,
                        "targets": score_targets,
                        "value_offsets": value_offsets,
                    }
                )

            if obs_rows_for_value:
                value_obs = np.asarray(obs_rows_for_value, dtype=np.float32)
                values_np = (
                    self.policy.predict_values(obs_as_tensor(value_obs, self.device))
                    .flatten()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            else:
                values_np = np.asarray([], dtype=np.float32)

        for job in jobs:
            index = int(job["index"])
            selection = job["selection"]
            targets = np.asarray(job["targets"], dtype=np.float32)
            value_offsets = list(job["value_offsets"])
            scores = targets.copy()
            for score_index, value_offset in enumerate(value_offsets):
                if value_offset >= 0:
                    scores[score_index] = float(self.gamma) * float(values_np[value_offset])
            if (
                selection.chosen_index < 0
                or selection.chosen_index >= scores.shape[0]
                or scores.shape[0] < max(int(self.shot_cf_min_modes), 1)
                or not np.all(np.isfinite(scores))
            ):
                skipped += 1
                continue
            chosen_score = float(scores[selection.chosen_index])
            candidate_mean = float(np.mean(scores))
            advantage = float(chosen_score - candidate_mean)
            advantages_np[index] = advantage
            mask_np[index] = 1.0
            stats["modes"].append(float(len(selection.candidates)))
            stats["landing_distance"].append(float(selection.mean_landing_distance))
            stats["advantage"].append(advantage)
            stats["chosen_q"].append(chosen_score)
            stats["candidate_mean_q"].append(candidate_mean)
            stats["chosen_best"].append(float(chosen_score >= float(np.max(scores)) - 1e-8))
            if self.shot_cf_debug_log:
                info = job["info"]
                assert isinstance(info, dict)
                info["shot_cf_debug"] = {
                    "chosen_index": int(selection.chosen_index),
                    "scores": scores.astype(float).tolist(),
                    "candidate_landings": [
                        (float(candidate.landing_x), float(candidate.landing_y))
                        for candidate in selection.candidates
                    ],
                }

        computed_count = int(np.count_nonzero(mask_np > 0.5))
        total_count = int(base.shape[0])
        self._last_shot_cf_stats = {
            "computed_rate": float(computed_count / max(total_count, 1)),
            "computed_count": float(computed_count),
            "skipped_count": float(skipped),
        }
        for key, values in stats.items():
            arr = np.asarray(values, dtype=np.float32)
            if arr.size:
                self._last_shot_cf_stats[f"{key}_mean"] = float(np.mean(arr))
                self._last_shot_cf_stats[f"{key}_std"] = float(np.std(arr))
            else:
                self._last_shot_cf_stats[f"{key}_mean"] = 0.0
                self._last_shot_cf_stats[f"{key}_std"] = 0.0

        # Rollout buffers are NumPy-backed, so even when shot_cf_value_detach is
        # false there is no differentiable simulator path here. Keep the flag for
        # config compatibility while using detached critic targets by default.
        return (
            th.as_tensor(advantages_np, dtype=current_obs_tensor.dtype, device=current_obs_tensor.device),
            th.as_tensor(mask_np, dtype=current_obs_tensor.dtype, device=current_obs_tensor.device),
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if not self._can_use_recovery_factorization():
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        assert isinstance(rollout_buffer, RecoveryFactorizedRolloutBuffer)
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                _, log_probs_shot, log_probs_recovery, _ = self.policy.evaluate_recovery_factorized_actions(
                    obs_tensor,
                    actions,
                )
            action_tensor_for_diagnostics = actions
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            if self.use_recovery_factorized_advantage:
                (
                    recovery_advantage,
                    recovery_loss_mask,
                    recovery_distribution_target,
                    recovery_distribution_mask,
                ) = self._recovery_advantage_from_transitions(
                    values,
                    infos,
                    new_obs,
                    rewards,
                    dones,
                    obs_tensor,
                    action_tensor_for_diagnostics,
                )
            else:
                recovery_advantage = th.zeros_like(log_probs)
                recovery_loss_mask = th.zeros_like(log_probs)
                recovery_bin_count = max(int(getattr(self.policy, "_conditional_recovery_count", 0)), 0)
                recovery_distribution_target = th.zeros(
                    (env.num_envs, recovery_bin_count),
                    dtype=log_probs.dtype,
                    device=log_probs.device,
                )
                recovery_distribution_mask = th.zeros_like(recovery_distribution_target)
            shot_cf_advantage, shot_cf_loss_mask = self._shot_cf_advantage_from_transitions(
                obs_tensor,
                action_tensor_for_diagnostics,
                infos,
            )
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                log_prob_shot=log_probs_shot,
                log_prob_recovery=log_probs_recovery,
                recovery_advantage=recovery_advantage,
                recovery_loss_mask=recovery_loss_mask,
                recovery_distribution_target=recovery_distribution_target,
                recovery_distribution_mask=recovery_distribution_mask,
                shot_cf_advantage=shot_cf_advantage,
                shot_cf_loss_mask=shot_cf_loss_mask,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        if not self._can_use_recovery_factorization():
            super().train()
            return

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        recovery_mask_np = self.rollout_buffer.recovery_loss_mask.flatten()
        recovery_advantage_np = self.rollout_buffer.recovery_advantages.flatten()
        recovery_distribution_mask_np = self.rollout_buffer.recovery_distribution_mask.reshape(
            -1,
            self.rollout_buffer.recovery_distribution_mask.shape[-1],
        )
        recovery_distribution_row_count = int(
            np.count_nonzero(np.sum(recovery_distribution_mask_np, axis=1) >= 2)
        )
        recovery_distribution_row_rate = (
            float(recovery_distribution_row_count / recovery_distribution_mask_np.shape[0])
            if recovery_distribution_mask_np.shape[0]
            else 0.0
        )
        active_recovery_advantages = recovery_advantage_np[recovery_mask_np > 0.5]
        recovery_mask_rate = float(np.mean(recovery_mask_np > 0.5)) if recovery_mask_np.size else 0.0
        recovery_mask_count = int(active_recovery_advantages.size)
        if recovery_mask_count > 0:
            recovery_advantage_mean = float(np.mean(active_recovery_advantages))
            recovery_advantage_std = float(np.std(active_recovery_advantages))
            recovery_advantage_abs_mean = float(np.mean(np.abs(active_recovery_advantages)))
            recovery_advantage_min = float(np.min(active_recovery_advantages))
            recovery_advantage_max = float(np.max(active_recovery_advantages))
        else:
            recovery_advantage_mean = 0.0
            recovery_advantage_std = 0.0
            recovery_advantage_abs_mean = 0.0
            recovery_advantage_min = 0.0
            recovery_advantage_max = 0.0
        shot_cf_mask_np = self.rollout_buffer.shot_cf_loss_mask.flatten()
        shot_cf_advantage_np = self.rollout_buffer.shot_cf_advantages.flatten()
        active_shot_cf_advantages = shot_cf_advantage_np[shot_cf_mask_np > 0.5]
        shot_cf_mask_rate = float(np.mean(shot_cf_mask_np > 0.5)) if shot_cf_mask_np.size else 0.0
        shot_cf_mask_count = int(active_shot_cf_advantages.size)
        if shot_cf_mask_count > 0:
            shot_cf_advantage_mean = float(np.mean(active_shot_cf_advantages))
            shot_cf_advantage_std = float(np.std(active_shot_cf_advantages))
        else:
            shot_cf_advantage_mean = 0.0
            shot_cf_advantage_std = 0.0

        entropy_losses = []
        pg_losses, value_losses = [], []
        pg_shot_losses, pg_recovery_losses = [], []
        recovery_distribution_losses = []
        clip_fractions = []
        recovery_clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob_shot, log_prob_recovery, entropy = self.policy.evaluate_recovery_factorized_actions(
                    rollout_data.observations,
                    actions,
                )
                values = values.flatten()
                log_prob = log_prob_shot + log_prob_recovery

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                shot_cf_advantages = self._normalize_shot_cf_advantage(
                    rollout_data.shot_cf_advantages,
                    rollout_data.shot_cf_loss_mask,
                )
                shot_advantages = advantages + self.shot_cf_coef * shot_cf_advantages
                shot_ratio = th.exp(log_prob_shot - rollout_data.old_log_prob_shot)
                shot_loss_1 = shot_advantages * shot_ratio
                shot_loss_2 = shot_advantages * th.clamp(shot_ratio, 1 - clip_range, 1 + clip_range)
                loss_shot = -th.min(shot_loss_1, shot_loss_2).mean()

                recovery_advantages = self._normalize_recovery_advantage(
                    rollout_data.recovery_advantages,
                    rollout_data.recovery_loss_mask,
                )
                recovery_ratio = th.exp(log_prob_recovery - rollout_data.old_log_prob_recovery)
                recovery_loss_1 = recovery_advantages * recovery_ratio
                recovery_loss_2 = recovery_advantages * th.clamp(recovery_ratio, 1 - clip_range, 1 + clip_range)
                recovery_loss_terms = -th.min(recovery_loss_1, recovery_loss_2)
                recovery_mask = rollout_data.recovery_loss_mask > 0.5
                if th.any(recovery_mask):
                    loss_recovery = recovery_loss_terms[recovery_mask].mean()
                    recovery_clip_fraction = th.mean(
                        (th.abs(recovery_ratio[recovery_mask] - 1) > clip_range).float()
                    ).item()
                else:
                    loss_recovery = th.zeros((), dtype=loss_shot.dtype, device=loss_shot.device)
                    recovery_clip_fraction = 0.0

                if self.recovery_counterfactual_distribution_coef > 0.0:
                    loss_recovery_distribution = self._sampled_recovery_distribution_loss(
                        rollout_data.observations,
                        actions,
                        rollout_data.recovery_distribution_targets,
                        rollout_data.recovery_distribution_mask,
                    )
                else:
                    loss_recovery_distribution = values.sum() * 0.0
                policy_loss = (
                    loss_shot
                    + loss_recovery
                    + self.recovery_counterfactual_distribution_coef * loss_recovery_distribution
                )
                pg_losses.append(policy_loss.item())
                pg_shot_losses.append(loss_shot.item())
                pg_recovery_losses.append(loss_recovery.item())
                recovery_distribution_losses.append(loss_recovery_distribution.item())
                clip_fraction = th.mean((th.abs(shot_ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                recovery_clip_fractions.append(recovery_clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/policy_gradient_loss_shot", np.mean(pg_shot_losses))
        self.logger.record("train/policy_gradient_loss_recovery", np.mean(pg_recovery_losses))
        self.logger.record("train/recovery_cf_dist_loss", np.mean(recovery_distribution_losses))
        self.logger.record("train/recovery_cf_dist_row_rate", recovery_distribution_row_rate)
        self.logger.record("train/recovery_cf_dist_row_count", recovery_distribution_row_count)
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/recovery_clip_fraction", np.mean(recovery_clip_fractions))
        self.logger.record("train/recovery_loss_mask_rate", recovery_mask_rate)
        self.logger.record("train/recovery_loss_mask_count", recovery_mask_count)
        self.logger.record("train/recovery_advantage_mean", recovery_advantage_mean)
        self.logger.record("train/recovery_advantage_std", recovery_advantage_std)
        self.logger.record("train/recovery_advantage_abs_mean", recovery_advantage_abs_mean)
        self.logger.record("train/recovery_advantage_min", recovery_advantage_min)
        self.logger.record("train/recovery_advantage_max", recovery_advantage_max)
        self.logger.record("train/shot_cf_mask_rate", shot_cf_mask_rate)
        self.logger.record("train/shot_cf_mask_count", shot_cf_mask_count)
        self.logger.record("train/shot_cf_advantage_mean", shot_cf_advantage_mean)
        self.logger.record("train/shot_cf_advantage_std", shot_cf_advantage_std)
        for key, value in self._last_shot_cf_stats.items():
            self.logger.record(f"train/shot_cf_{key}", value)
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
