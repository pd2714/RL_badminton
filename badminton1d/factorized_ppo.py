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


class RecoveryFactorizedRolloutBuffer(RolloutBuffer):
    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs)
        self.log_probs_shot = np.zeros(shape, dtype=np.float32)
        self.log_probs_recovery = np.zeros(shape, dtype=np.float32)
        self.recovery_advantages = np.zeros(shape, dtype=np.float32)
        self.recovery_loss_mask = np.zeros(shape, dtype=np.float32)

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

        super().add(obs, action, reward, episode_start, value, log_prob)
        self.log_probs_shot[pos] = log_prob_shot.clone().cpu().numpy().flatten()
        self.log_probs_recovery[pos] = log_prob_recovery.clone().cpu().numpy().flatten()
        self.recovery_advantages[pos] = recovery_advantage.clone().cpu().numpy().flatten()
        self.recovery_loss_mask[pos] = recovery_loss_mask.clone().cpu().numpy().flatten()

    def get(self, batch_size: int | None = None) -> Generator[RecoveryFactorizedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
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
        )
        return RecoveryFactorizedRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class RecoveryFactorizedPPO(PPO):
    """PPO variant that can train recovery with a recovery-only advantage."""

    def __init__(self, *args, use_recovery_factorized_advantage: bool = False, **kwargs) -> None:
        self.use_recovery_factorized_advantage = bool(use_recovery_factorized_advantage)
        if self.use_recovery_factorized_advantage and kwargs.get("rollout_buffer_class") is None:
            kwargs["rollout_buffer_class"] = RecoveryFactorizedRolloutBuffer
        super().__init__(*args, **kwargs)

    def _can_use_recovery_factorization(self) -> bool:
        return (
            self.use_recovery_factorized_advantage
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

    def _recovery_advantage_from_infos(
        self,
        values_before: th.Tensor,
        infos: list[dict],
    ) -> tuple[th.Tensor, th.Tensor]:
        before = values_before.flatten()
        mask_np = np.asarray(
            ["recovery_factorized_after_observation" in info for info in infos],
            dtype=np.float32,
        )
        if not np.any(mask_np):
            zeros = th.zeros_like(before)
            return zeros, zeros

        assert self._last_obs is not None
        last_obs = np.asarray(self._last_obs)
        after_obs = np.asarray(
            [
                info.get("recovery_factorized_after_observation", last_obs[index])
                for index, info in enumerate(infos)
            ],
            dtype=last_obs.dtype,
        )
        with th.no_grad():
            after_values = self.policy.predict_values(obs_as_tensor(after_obs, self.device)).flatten()
        mask = th.as_tensor(mask_np, dtype=before.dtype, device=before.device)
        # A_rec is intentionally just the value improvement from the clean
        # post-recovery state over the decision state before choosing recovery.
        recovery_advantage = (after_values - before) * mask
        return recovery_advantage, mask

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
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

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

            recovery_advantage, recovery_loss_mask = self._recovery_advantage_from_infos(values, infos)
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

        entropy_losses = []
        pg_losses, value_losses = [], []
        pg_shot_losses, pg_recovery_losses = [], []
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

                shot_ratio = th.exp(log_prob_shot - rollout_data.old_log_prob_shot)
                shot_loss_1 = advantages * shot_ratio
                shot_loss_2 = advantages * th.clamp(shot_ratio, 1 - clip_range, 1 + clip_range)
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

                policy_loss = loss_shot + loss_recovery
                pg_losses.append(policy_loss.item())
                pg_shot_losses.append(loss_shot.item())
                pg_recovery_losses.append(loss_recovery.item())
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
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/recovery_clip_fraction", np.mean(recovery_clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
