from __future__ import annotations

from functools import partial

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical, Normal
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from badminton.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton.config import SimulationConfig
from badminton.dynamics import landing_position
from badminton.shot_generators import TacticRuntimeConfig
from badminton.state import StageState

X_LEFT_INDEX = 0
Y_LEFT_INDEX = 1
X_RIGHT_INDEX = 2
Y_RIGHT_INDEX = 3
X0_INDEX = 4
Y0_INDEX = 5
Z0_INDEX = 6
CURRENT_HITTER_LEFT_INDEX = 7
CURRENT_HITTER_RIGHT_INDEX = 8
ROLE_IS_RECEIVER_INDEX = 14
STAGE_PROGRESS_INDEX = 17
V_X_LEFT_INDEX = 29
V_Y_LEFT_INDEX = 30
V_X_RIGHT_INDEX = 31
V_Y_RIGHT_INDEX = 32
FEASIBLE_MASK_START_INDEX = 33
MASKED_LOGIT_VALUE = -1e9
CONDITIONAL_PROB_MODE = "conditional_prob"
CONTINUOUS_ACTION_MODE = "continuous_action"
MIXED_DISCRETE_CONTINOUS_MODE = "mixed_discrete_continous"
VELOCITY_ORIENTED_MODE = "velocity_oriented"
CONDITIONAL_DISCRETE_MODES = {CONDITIONAL_PROB_MODE, VELOCITY_ORIENTED_MODE}
CONTINUOUS_LOG_STD = -3.0
CONTINUOUS_LOG_STD_MIN = CONTINUOUS_LOG_STD
CONTINUOUS_LOG_STD_MAX = CONTINUOUS_LOG_STD
MIXED_RECOVERY_LOG_STD = -2.0
CONDITIONAL_RECOVERY_CONTEXT_DIM = 10
CONDITIONAL_RECOVERY_CONTEXT_CACHE_SIZE = 32_768


def _denormalize_signed(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return float(max(min(value, 1.0), -1.0) * scale)


def _denormalize_unit(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return lower
    clipped = float(max(min(value, 1.0), 0.0))
    return lower + clipped * (upper - lower)


def _state_from_observation_row(
    obs_row: th.Tensor,
    config: SimulationConfig,
    *,
    stage_index: int = 0,
) -> StageState:
    current_hitter = "left" if float(obs_row[CURRENT_HITTER_LEFT_INDEX]) >= 0.5 else "right"
    return StageState(
        x_left=_denormalize_signed(float(obs_row[X_LEFT_INDEX]), config.court.half_width),
        y_left=_denormalize_signed(float(obs_row[Y_LEFT_INDEX]), config.court.half_length),
        x_right=_denormalize_signed(float(obs_row[X_RIGHT_INDEX]), config.court.half_width),
        y_right=_denormalize_signed(float(obs_row[Y_RIGHT_INDEX]), config.court.half_length),
        current_hitter=current_hitter,
        x0=_denormalize_signed(float(obs_row[X0_INDEX]), config.court.half_width),
        y0=_denormalize_signed(float(obs_row[Y0_INDEX]), config.court.half_length),
        z0=_denormalize_unit(float(obs_row[Z0_INDEX]), 0.0, config.render.z_max),
        v_x_left=_denormalize_signed(float(obs_row[V_X_LEFT_INDEX]), config.player.v_max)
        if obs_row.shape[0] > V_X_LEFT_INDEX
        else 0.0,
        v_y_left=_denormalize_signed(float(obs_row[V_Y_LEFT_INDEX]), config.player.v_max)
        if obs_row.shape[0] > V_Y_LEFT_INDEX
        else 0.0,
        v_x_right=_denormalize_signed(float(obs_row[V_X_RIGHT_INDEX]), config.player.v_max)
        if obs_row.shape[0] > V_X_RIGHT_INDEX
        else 0.0,
        v_y_right=_denormalize_signed(float(obs_row[V_Y_RIGHT_INDEX]), config.player.v_max)
        if obs_row.shape[0] > V_Y_RIGHT_INDEX
        else 0.0,
        stage_index=stage_index,
    )


def apply_hitter_action_mask(
    logits: th.Tensor,
    obs: th.Tensor,
    *,
    mapper: DiscreteActionMapper | None,
    mask_mid_rally: bool = False,
) -> th.Tensor:
    if mapper is None or logits.ndim != 2 or obs.ndim != 2:
        return logits

    masked_logits = logits.clone()
    hitter_rows = obs[:, ROLE_IS_RECEIVER_INDEX] < 0.5
    if not mask_mid_rally:
        hitter_rows = hitter_rows & (obs[:, STAGE_PROGRESS_INDEX] <= 1e-6)
    if not th.any(hitter_rows):
        return masked_logits

    hitter_indices = th.nonzero(hitter_rows, as_tuple=False).flatten()
    for row_index in hitter_indices.tolist():
        stage_index = 0 if float(obs[row_index, STAGE_PROGRESS_INDEX]) <= 1e-6 else 1
        state = _state_from_observation_row(obs[row_index], mapper.config, stage_index=stage_index)
        legal_mask = mapper.legal_hitter_mask(state) if mask_mid_rally else mapper.legal_serve_hitter_mask(state)
        if not legal_mask.any():
            continue
        row_logits = masked_logits[row_index]
        if row_logits.shape[0] > mapper.hitter_action_count:
            row_logits[mapper.hitter_action_count:] = MASKED_LOGIT_VALUE
        illegal = th.as_tensor(~legal_mask, dtype=th.bool, device=row_logits.device)
        row_logits[: mapper.hitter_action_count] = row_logits[: mapper.hitter_action_count].masked_fill(illegal, MASKED_LOGIT_VALUE)
    return masked_logits


def apply_receiver_action_mask(
    logits: th.Tensor,
    obs: th.Tensor,
    *,
    receiver_action_count: int,
) -> th.Tensor:
    if logits.ndim != 2 or obs.ndim != 2:
        raise ValueError("Expected batched logits and observations.")
    if receiver_action_count <= 0 or logits.shape[1] <= receiver_action_count:
        return logits

    masked_logits = logits.clone()
    receiver_rows = obs[:, ROLE_IS_RECEIVER_INDEX] > 0.5
    if not th.any(receiver_rows):
        return masked_logits

    receiver_indices = th.nonzero(receiver_rows, as_tuple=False).flatten()
    receiver_logits = masked_logits.index_select(0, receiver_indices)
    receiver_obs = obs.index_select(0, receiver_indices)

    receiver_logits[:, receiver_action_count:] = MASKED_LOGIT_VALUE

    feasible_mask_end = FEASIBLE_MASK_START_INDEX + receiver_action_count
    if receiver_obs.shape[1] >= feasible_mask_end:
        feasible_mask = receiver_obs[:, FEASIBLE_MASK_START_INDEX:feasible_mask_end] > 0.5
        has_feasible = feasible_mask.any(dim=1)
        if th.any(has_feasible):
            legal_slice = receiver_logits[has_feasible, :receiver_action_count]
            legal_slice = legal_slice.masked_fill(~feasible_mask[has_feasible], MASKED_LOGIT_VALUE)
            receiver_logits[has_feasible, :receiver_action_count] = legal_slice

    masked_logits[receiver_indices] = receiver_logits
    return masked_logits


class _ConditionalCategoricalPolicyDistribution:
    def __init__(self, policy: "MaskedBadmintonPolicy", obs: th.Tensor, latent_pi: th.Tensor) -> None:
        self.policy = policy
        self.obs = obs
        self.latent_pi = latent_pi
        self._last_actions: th.Tensor | None = None
        self.distribution = Categorical(logits=policy._conditional_padded_logits(obs, latent_pi))

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        actions = self.policy._conditional_sample_actions(self.obs, self.latent_pi, deterministic=deterministic)
        self._last_actions = actions
        return actions

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.policy._conditional_log_prob(actions, self.obs, self.latent_pi)

    def entropy(self) -> th.Tensor:
        if self._last_actions is None:
            actions = self.get_actions(deterministic=True)
        else:
            actions = self._last_actions
        return self.policy._conditional_entropy(actions, self.obs, self.latent_pi)


class _ContinuousConditionalPolicyDistribution:
    def __init__(self, policy: "MaskedBadmintonPolicy", obs: th.Tensor, latent_pi: th.Tensor) -> None:
        self.policy = policy
        self.obs = obs
        self.latent_pi = latent_pi

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        return self.policy._continuous_sample_actions(self.obs, self.latent_pi, deterministic=deterministic)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob, _ = self.policy._continuous_log_prob_entropy(actions, self.obs, self.latent_pi)
        return log_prob

    def entropy(self) -> th.Tensor:
        _, entropy = self.policy._continuous_log_prob_entropy(
            self.get_actions(deterministic=True),
            self.obs,
            self.latent_pi,
        )
        return entropy


class _MixedDiscreteContinuousPolicyDistribution:
    def __init__(self, policy: "MaskedBadmintonPolicy", obs: th.Tensor, latent_pi: th.Tensor) -> None:
        self.policy = policy
        self.obs = obs
        self.latent_pi = latent_pi
        self._last_actions: th.Tensor | None = None

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        actions = self.policy._mixed_sample_actions(self.obs, self.latent_pi, deterministic=deterministic)
        self._last_actions = actions
        return actions

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob, _ = self.policy._mixed_log_prob_entropy(actions, self.obs, self.latent_pi)
        return log_prob

    def entropy(self) -> th.Tensor:
        actions = self._last_actions
        if actions is None:
            actions = self.get_actions(deterministic=True)
        _, entropy = self.policy._mixed_log_prob_entropy(actions, self.obs, self.latent_pi)
        return entropy


class MaskedBadmintonPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        sim_config: SimulationConfig | None = None,
        discrete_action_config: DiscreteActionConfig | None = None,
        policy_type: str = VELOCITY_ORIENTED_MODE,
        tactic_runtime_config: TacticRuntimeConfig | None = None,
        mask_mid_rally_hitter_actions: bool = True,
        **kwargs,
    ):
        normalized_policy_type = policy_type.strip().lower()
        if sim_config is None and normalized_policy_type in (
            CONDITIONAL_DISCRETE_MODES | {CONTINUOUS_ACTION_MODE, MIXED_DISCRETE_CONTINOUS_MODE}
        ):
            sim_config = SimulationConfig()
        self.sim_config = sim_config
        self.discrete_action_config = discrete_action_config
        self.policy_type = normalized_policy_type
        self.output_mode = CONDITIONAL_PROB_MODE if self.policy_type in CONDITIONAL_DISCRETE_MODES else self.policy_type
        self.mask_mid_rally_hitter_actions = bool(mask_mid_rally_hitter_actions)
        self._conditional_recovery_context_cache: dict[tuple[StageState, int], tuple[float, ...]] = {}
        self._action_mapper = None
        if sim_config is not None:
            self._action_mapper = DiscreteActionMapper(
                sim_config,
                discrete_action_config,
                policy_type=policy_type,
                tactic_runtime_config=tactic_runtime_config,
            )
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        if self.output_mode not in {CONDITIONAL_PROB_MODE, CONTINUOUS_ACTION_MODE, MIXED_DISCRETE_CONTINOUS_MODE}:
            super()._build(lr_schedule)
            return

        if self._action_mapper is None:
            raise ValueError(f"{self.output_mode} requires sim_config in policy_kwargs.")

        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        self._conditional_phi_count = self._action_mapper._impl._effective_phi_bins
        self._conditional_theta_count = self._action_mapper.discrete_config.theta_bins
        self._conditional_speed_count = self._action_mapper.discrete_config.speed_bins
        self._conditional_x_rec_count = self._action_mapper._impl._effective_x_rec_bins
        self._conditional_y_rec_count = self._action_mapper.discrete_config.y_rec_bins
        self._conditional_recovery_count = self._conditional_x_rec_count * self._conditional_y_rec_count
        self._conditional_receiver_count = self._action_mapper.receiver_action_count

        if self.output_mode == CONDITIONAL_PROB_MODE:
            self.phi_head = nn.Linear(latent_dim_pi, self._conditional_phi_count)
            self.theta_head = nn.Linear(latent_dim_pi + self._conditional_phi_count, self._conditional_theta_count)
            self.speed_head = nn.Linear(
                latent_dim_pi + self._conditional_phi_count + self._conditional_theta_count,
                self._conditional_speed_count,
            )
            self.recovery_head = nn.Linear(
                latent_dim_pi
                + self._conditional_phi_count
                + self._conditional_theta_count
                + self._conditional_speed_count
                + CONDITIONAL_RECOVERY_CONTEXT_DIM,
                self._conditional_recovery_count,
            )
            self.receiver_head = nn.Linear(latent_dim_pi, self._conditional_receiver_count)
            self.action_net = nn.ModuleDict(
                {
                    "phi": self.phi_head,
                    "theta": self.theta_head,
                    "speed": self.speed_head,
                    "recovery": self.recovery_head,
                    "receiver": self.receiver_head,
                }
            )
        elif self.output_mode == MIXED_DISCRETE_CONTINOUS_MODE:
            self.phi_head = nn.Linear(latent_dim_pi, self._conditional_phi_count)
            self.theta_head = nn.Linear(latent_dim_pi + self._conditional_phi_count, self._conditional_theta_count)
            self.speed_head = nn.Linear(
                latent_dim_pi + self._conditional_phi_count + self._conditional_theta_count,
                self._conditional_speed_count,
            )
            self.recovery_head = nn.Linear(
                latent_dim_pi
                + self._conditional_phi_count
                + self._conditional_theta_count
                + self._conditional_speed_count
                + CONDITIONAL_RECOVERY_CONTEXT_DIM,
                2,
            )
            self.receiver_head = nn.Linear(latent_dim_pi, self._conditional_receiver_count)
            self.action_net = nn.ModuleDict(
                {
                    "phi": self.phi_head,
                    "theta": self.theta_head,
                    "speed": self.speed_head,
                    "recovery": self.recovery_head,
                    "receiver": self.receiver_head,
                }
            )
        else:
            self.phi_head = nn.Linear(latent_dim_pi, 2)
            self.theta_head = nn.Linear(latent_dim_pi + 1, 2)
            self.speed_head = nn.Linear(latent_dim_pi + 2, 2)
            self.recovery_head = nn.Linear(latent_dim_pi + 3, 4)
            self.receiver_head = nn.Linear(latent_dim_pi, 2)
            self.action_net = nn.ModuleDict(
                {
                    "phi": self.phi_head,
                    "theta": self.theta_head,
                    "speed": self.speed_head,
                    "recovery": self.recovery_head,
                    "receiver": self.receiver_head,
                }
            )

        self.value_net = nn.Linear(latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _receiver_action_count(self, obs: th.Tensor) -> int:
        return max(int(obs.shape[-1]) - FEASIBLE_MASK_START_INDEX, 0)

    def _masked_action_distribution(self, obs: th.Tensor, latent_pi: th.Tensor):
        if self.output_mode == CONDITIONAL_PROB_MODE:
            return _ConditionalCategoricalPolicyDistribution(self, obs, latent_pi)
        if self.output_mode == CONTINUOUS_ACTION_MODE:
            return _ContinuousConditionalPolicyDistribution(self, obs, latent_pi)
        if self.output_mode == MIXED_DISCRETE_CONTINOUS_MODE:
            return _MixedDiscreteContinuousPolicyDistribution(self, obs, latent_pi)
        action_logits = self.action_net(latent_pi)
        if isinstance(self.action_dist, CategoricalDistribution):
            action_logits = apply_hitter_action_mask(
                action_logits,
                obs,
                mapper=self._action_mapper,
                mask_mid_rally=self.mask_mid_rally_hitter_actions,
            )
            action_logits = apply_receiver_action_mask(
                action_logits,
                obs,
                receiver_action_count=self._receiver_action_count(obs),
            )
            return self.action_dist.proba_distribution(action_logits=action_logits)
        return self._get_action_dist_from_latent(latent_pi)

    def get_distribution(self, obs):
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._masked_action_distribution(obs, latent_pi)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._masked_action_distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        if self.output_mode == CONDITIONAL_PROB_MODE:
            log_prob, entropy = self._conditional_log_prob_entropy(actions, obs, latent_pi)
            return values, log_prob, entropy
        distribution = self._masked_action_distribution(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def evaluate_recovery_factorized_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor | None]:
        """Evaluate conditional actions with shot and recovery log-probs split for PPO.

        This keeps the policy factorization unchanged. For hitter rows, the shot
        component is phi/theta/speed and the recovery component is the recovery
        head. Receiver rows have no recovery choice, so their receiver log-prob
        stays in the shot loss and the recovery log-prob is zero.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        if self.output_mode == CONDITIONAL_PROB_MODE:
            logp_shot, logp_recovery, entropy = self._conditional_log_prob_components_entropy(
                actions,
                obs,
                latent_pi,
            )
            return values, logp_shot, logp_recovery, entropy

        distribution = self._masked_action_distribution(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, th.zeros_like(log_prob), entropy

    def _conditional_hitter_legal_mask(self, obs: th.Tensor) -> th.Tensor:
        batch = obs.shape[0]
        shape = (
            batch,
            self._conditional_phi_count,
            self._conditional_theta_count,
            self._conditional_speed_count,
            self._conditional_recovery_count,
        )
        masks = th.ones(shape, dtype=th.bool, device=obs.device)
        if self._action_mapper is None:
            return masks

        hitter_rows = obs[:, ROLE_IS_RECEIVER_INDEX] < 0.5
        if not self.mask_mid_rally_hitter_actions:
            hitter_rows = hitter_rows & (obs[:, STAGE_PROGRESS_INDEX] <= 1e-6)
        for row_index in th.nonzero(hitter_rows, as_tuple=False).flatten().tolist():
            stage_index = 0 if float(obs[row_index, STAGE_PROGRESS_INDEX]) <= 1e-6 else 1
            state = _state_from_observation_row(obs[row_index], self._action_mapper.config, stage_index=stage_index)
            legal = (
                self._action_mapper.legal_hitter_mask(state)
                if self.mask_mid_rally_hitter_actions
                else self._action_mapper.legal_serve_hitter_mask(state)
            )
            if not legal.any():
                continue
            legal_tensor = th.as_tensor(legal, dtype=th.bool, device=obs.device)
            masks[row_index] = legal_tensor.reshape(shape[1:])
        return masks

    def _conditional_receiver_legal_mask(self, obs: th.Tensor) -> th.Tensor:
        mask = th.ones((obs.shape[0], self._conditional_receiver_count), dtype=th.bool, device=obs.device)
        receiver_rows = obs[:, ROLE_IS_RECEIVER_INDEX] > 0.5
        feasible_mask_end = FEASIBLE_MASK_START_INDEX + self._conditional_receiver_count
        if obs.shape[1] < feasible_mask_end:
            return mask
        feasible_mask = obs[:, FEASIBLE_MASK_START_INDEX:feasible_mask_end] > 0.5
        has_feasible = feasible_mask.any(dim=1) & receiver_rows
        mask[has_feasible] = feasible_mask[has_feasible]
        return mask

    def _masked_logits(self, logits: th.Tensor, mask: th.Tensor) -> th.Tensor:
        return logits.masked_fill(~mask, MASKED_LOGIT_VALUE)

    def _should_validate_velocity_angle_row(self, obs_row: th.Tensor) -> bool:
        if self.policy_type != VELOCITY_ORIENTED_MODE or self._action_mapper is None:
            return False
        if float(obs_row[ROLE_IS_RECEIVER_INDEX]) >= 0.5:
            return False
        return True

    def _velocity_angle_has_valid_speed(self, obs_row: th.Tensor, phi_index: int, theta_index: int) -> bool:
        if self._action_mapper is None:
            return True
        stage_index = 0 if float(obs_row[STAGE_PROGRESS_INDEX]) <= 1e-6 else 1
        state = _state_from_observation_row(obs_row, self._action_mapper.config, stage_index=stage_index)
        return self._action_mapper.valid_speed_range(state, int(phi_index), int(theta_index)) is not None

    def _theta_logits_for_phi(self, latent_pi: th.Tensor, phi: th.Tensor) -> th.Tensor:
        phi_one_hot = th.nn.functional.one_hot(phi, self._conditional_phi_count).float()
        return self.theta_head(th.cat((latent_pi, phi_one_hot), dim=1))

    def _best_valid_velocity_angle_for_row(
        self,
        obs_row: th.Tensor,
        latent_pi_row: th.Tensor,
        phi_logits_row: th.Tensor,
    ) -> tuple[int, int] | None:
        best_score: float | None = None
        best_pair: tuple[int, int] | None = None
        for phi_index in range(self._conditional_phi_count):
            phi_tensor = th.as_tensor([phi_index], dtype=th.long, device=latent_pi_row.device)
            theta_logits = self._theta_logits_for_phi(latent_pi_row.reshape(1, -1), phi_tensor)[0]
            for theta_index in range(self._conditional_theta_count):
                if not self._velocity_angle_has_valid_speed(obs_row, phi_index, theta_index):
                    continue
                score = float(phi_logits_row[phi_index] + theta_logits[theta_index])
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (phi_index, theta_index)
        return best_pair

    def _resample_invalid_velocity_angles(
        self,
        obs: th.Tensor,
        latent_pi: th.Tensor,
        phi_logits: th.Tensor,
        phi: th.Tensor,
        theta: th.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[th.Tensor, th.Tensor]:
        if self.policy_type != VELOCITY_ORIENTED_MODE or self._action_mapper is None:
            return phi, theta

        phi = phi.clone()
        theta = theta.clone()
        validate_rows = [
            row_index
            for row_index in range(obs.shape[0])
            if self._should_validate_velocity_angle_row(obs[row_index])
        ]
        if not validate_rows:
            return phi, theta

        max_attempts = max(32, self._conditional_phi_count * self._conditional_theta_count)
        for row_index in validate_rows:
            if self._velocity_angle_has_valid_speed(
                obs[row_index],
                int(phi[row_index].item()),
                int(theta[row_index].item()),
            ):
                continue

            if not deterministic:
                phi_dist = Categorical(logits=phi_logits[row_index])
                for _ in range(max_attempts):
                    candidate_phi = phi_dist.sample().reshape(1)
                    theta_logits = self._theta_logits_for_phi(
                        latent_pi[row_index].reshape(1, -1),
                        candidate_phi,
                    )
                    candidate_theta = Categorical(logits=theta_logits[0]).sample()
                    if self._velocity_angle_has_valid_speed(
                        obs[row_index],
                        int(candidate_phi.item()),
                        int(candidate_theta.item()),
                    ):
                        phi[row_index] = candidate_phi[0]
                        theta[row_index] = candidate_theta
                        break
                else:
                    best_pair = self._best_valid_velocity_angle_for_row(
                        obs[row_index],
                        latent_pi[row_index],
                        phi_logits[row_index],
                    )
                    if best_pair is not None:
                        phi[row_index] = best_pair[0]
                        theta[row_index] = best_pair[1]
                continue

            best_pair = self._best_valid_velocity_angle_for_row(
                obs[row_index],
                latent_pi[row_index],
                phi_logits[row_index],
            )
            if best_pair is not None:
                phi[row_index] = best_pair[0]
                theta[row_index] = best_pair[1]
        return phi, theta

    def _conditional_velocity_legal_mask(self, obs: th.Tensor) -> th.Tensor:
        shape = (
            obs.shape[0],
            self._conditional_phi_count,
            self._conditional_theta_count,
            self._conditional_speed_count,
        )
        mask = th.ones(shape, dtype=th.bool, device=obs.device)
        if self._action_mapper is None:
            return mask

        hitter_rows = obs[:, ROLE_IS_RECEIVER_INDEX] < 0.5
        if not self.mask_mid_rally_hitter_actions:
            hitter_rows = hitter_rows & (obs[:, STAGE_PROGRESS_INDEX] <= 1e-6)
        for row_index in th.nonzero(hitter_rows, as_tuple=False).flatten().tolist():
            stage_index = 0 if float(obs[row_index, STAGE_PROGRESS_INDEX]) <= 1e-6 else 1
            state = _state_from_observation_row(obs[row_index], self._action_mapper.config, stage_index=stage_index)
            row_mask = np.zeros(shape[1:], dtype=bool)
            for phi_index in range(self._conditional_phi_count):
                for theta_index in range(self._conditional_theta_count):
                    valid_range = self._action_mapper.valid_speed_range(state, phi_index, theta_index)
                    if valid_range is not None:
                        row_mask[phi_index, theta_index, :] = True
            if row_mask.any():
                mask[row_index] = th.as_tensor(row_mask, dtype=th.bool, device=obs.device)
        return mask

    def _conditional_component_logits(
        self,
        obs: th.Tensor,
        latent_pi: th.Tensor,
        phi: th.Tensor | None = None,
        theta: th.Tensor | None = None,
        speed: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None, th.Tensor | None, th.Tensor | None, th.Tensor]:
        legal_velocity = None
        if self.policy_type != VELOCITY_ORIENTED_MODE:
            legal_velocity = self._conditional_velocity_legal_mask(obs)
            phi_logits = self._masked_logits(self.phi_head(latent_pi), legal_velocity.any(dim=(2, 3)))
        else:
            phi_logits = self.phi_head(latent_pi)
        theta_logits = speed_logits = recovery_logits = None
        if phi is not None:
            phi_one_hot = th.nn.functional.one_hot(phi, self._conditional_phi_count).float()
            theta_logits = self.theta_head(th.cat((latent_pi, phi_one_hot), dim=1))
            if legal_velocity is not None:
                theta_allowed = legal_velocity[th.arange(obs.shape[0], device=obs.device), phi].any(dim=2)
                theta_logits = self._masked_logits(theta_logits, theta_allowed)
        if phi is not None and theta is not None:
            theta_one_hot = th.nn.functional.one_hot(theta, self._conditional_theta_count).float()
            speed_logits = self.speed_head(th.cat((latent_pi, phi_one_hot, theta_one_hot), dim=1))
            if legal_velocity is not None:
                speed_allowed = legal_velocity[th.arange(obs.shape[0], device=obs.device), phi, theta]
                speed_logits = self._masked_logits(speed_logits, speed_allowed)
        if phi is not None and theta is not None and speed is not None:
            speed_one_hot = th.nn.functional.one_hot(speed, self._conditional_speed_count).float()
            recovery_context = self._conditional_recovery_context(obs, phi=phi, theta=theta, speed=speed, dtype=latent_pi.dtype)
            recovery_logits = self.recovery_head(
                th.cat((latent_pi, phi_one_hot, theta_one_hot, speed_one_hot, recovery_context), dim=1)
            )
        receiver_logits = self._masked_logits(self.receiver_head(latent_pi), self._conditional_receiver_legal_mask(obs))
        return phi_logits, theta_logits, speed_logits, recovery_logits, receiver_logits

    def _conditional_recovery_context(
        self,
        obs: th.Tensor,
        *,
        phi: th.Tensor,
        theta: th.Tensor,
        speed: th.Tensor,
        dtype: th.dtype,
    ) -> th.Tensor:
        if self._action_mapper is None:
            return th.zeros((obs.shape[0], CONDITIONAL_RECOVERY_CONTEXT_DIM), dtype=dtype, device=obs.device)

        config = self._action_mapper.config
        max_velocity = max(
            abs(config.action.vx_min),
            abs(config.action.vx_max),
            abs(config.action.vy_min_forward),
            abs(config.action.vy_max_forward),
            abs(config.action.vz_min),
            abs(config.action.vz_max),
            1e-6,
        )
        features: list[list[float]] = []
        for row_index in range(obs.shape[0]):
            try:
                stage_index = 0 if float(obs[row_index, STAGE_PROGRESS_INDEX]) <= 1e-6 else 1
                state = _state_from_observation_row(obs[row_index], config, stage_index=stage_index)
                flat_action = int(
                    (
                        (
                            (int(phi[row_index].item()) * self._conditional_theta_count + int(theta[row_index].item()))
                            * self._conditional_speed_count
                            + int(speed[row_index].item())
                        )
                        * self._conditional_recovery_count
                    )
                )
                features.append(list(self._cached_conditional_recovery_features(state, flat_action, max_velocity)))
            except (RuntimeError, ValueError, IndexError, FloatingPointError):
                features.append([0.0] * CONDITIONAL_RECOVERY_CONTEXT_DIM)
        return th.as_tensor(features, dtype=dtype, device=obs.device)

    def _cached_conditional_recovery_features(
        self,
        state: StageState,
        flat_action: int,
        max_velocity: float,
    ) -> tuple[float, ...]:
        cache_key = (state, int(flat_action))
        cached = self._conditional_recovery_context_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._action_mapper is None:
            return (0.0,) * CONDITIONAL_RECOVERY_CONTEXT_DIM

        config = self._action_mapper.config
        shot_action = self._action_mapper.decode_hitter(flat_action, state).shot_action
        landing_x, landing_y = landing_position(state, shot_action, config)
        shot_phi = float(np.arctan2(shot_action.v_y, shot_action.v_x))
        horizontal_speed = float(np.hypot(shot_action.v_x, shot_action.v_y))
        shot_theta = float(np.arctan2(shot_action.v_z, horizontal_speed))
        shot_speed = float(np.sqrt(shot_action.v_x**2 + shot_action.v_y**2 + shot_action.v_z**2))
        computed = (
            float(np.sin(shot_phi)),
            float(np.cos(shot_phi)),
            float(np.sin(shot_theta)),
            float(np.cos(shot_theta)),
            float(np.clip(shot_action.v_x / max_velocity, -1.0, 1.0)),
            float(np.clip(shot_action.v_y / max_velocity, -1.0, 1.0)),
            float(np.clip(shot_action.v_z / max_velocity, -1.0, 1.0)),
            float(np.clip(shot_speed / max_velocity, 0.0, 1.0)),
            float(np.clip(landing_x / max(config.court.half_width, 1e-6), -1.0, 1.0)),
            float(np.clip(landing_y / max(config.court.half_length, 1e-6), -1.0, 1.0)),
        )
        if len(self._conditional_recovery_context_cache) >= CONDITIONAL_RECOVERY_CONTEXT_CACHE_SIZE:
            self._conditional_recovery_context_cache.clear()
        self._conditional_recovery_context_cache[cache_key] = computed
        return computed

    def _sample_categorical(self, logits: th.Tensor, deterministic: bool) -> th.Tensor:
        if deterministic:
            return th.argmax(logits, dim=1)
        return Categorical(logits=logits).sample()

    def _conditional_sample_actions(self, obs: th.Tensor, latent_pi: th.Tensor, *, deterministic: bool) -> th.Tensor:
        phi_logits, _, _, _, receiver_logits = self._conditional_component_logits(obs, latent_pi)
        phi = self._sample_categorical(phi_logits, deterministic)
        _, theta_logits, _, _, _ = self._conditional_component_logits(obs, latent_pi, phi=phi)
        assert theta_logits is not None
        theta = self._sample_categorical(theta_logits, deterministic)
        phi, theta = self._resample_invalid_velocity_angles(
            obs,
            latent_pi,
            phi_logits,
            phi,
            theta,
            deterministic=deterministic,
        )
        _, _, speed_logits, _, _ = self._conditional_component_logits(obs, latent_pi, phi=phi, theta=theta)
        assert speed_logits is not None
        speed = self._sample_categorical(speed_logits, deterministic)
        _, _, _, recovery_logits, _ = self._conditional_component_logits(obs, latent_pi, phi=phi, theta=theta, speed=speed)
        assert recovery_logits is not None
        recovery = self._sample_categorical(recovery_logits, deterministic)
        hitter_action = self._conditional_compose_action(phi, theta, speed, recovery)
        receiver_action = self._sample_categorical(receiver_logits, deterministic)
        receiver_rows = obs[:, ROLE_IS_RECEIVER_INDEX] > 0.5
        return th.where(receiver_rows, receiver_action, hitter_action).long()

    def _conditional_compose_action(
        self,
        phi: th.Tensor,
        theta: th.Tensor,
        speed: th.Tensor,
        recovery: th.Tensor,
    ) -> th.Tensor:
        return (((phi * self._conditional_theta_count + theta) * self._conditional_speed_count + speed)
                * self._conditional_recovery_count + recovery)

    def _conditional_decompose_actions(self, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        flat = actions.long().flatten() % max(int(self._action_mapper.hitter_action_count), 1)
        recovery = flat % self._conditional_recovery_count
        rem = flat // self._conditional_recovery_count
        speed = rem % self._conditional_speed_count
        rem = rem // self._conditional_speed_count
        theta = rem % self._conditional_theta_count
        phi = rem // self._conditional_theta_count
        return phi, theta, speed, recovery

    def _conditional_log_prob(self, actions: th.Tensor, obs: th.Tensor, latent_pi: th.Tensor) -> th.Tensor:
        log_prob, _ = self._conditional_log_prob_entropy(actions, obs, latent_pi)
        return log_prob

    def _conditional_log_prob_entropy(
        self,
        actions: th.Tensor,
        obs: th.Tensor,
        latent_pi: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        logp_shot, logp_recovery, entropy = self._conditional_log_prob_components_entropy(actions, obs, latent_pi)
        return logp_shot + logp_recovery, entropy

    def _conditional_log_prob_components_entropy(
        self,
        actions: th.Tensor,
        obs: th.Tensor,
        latent_pi: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions = actions.long().flatten()
        phi, theta, speed, recovery = self._conditional_decompose_actions(actions)
        phi_logits, theta_logits, speed_logits, recovery_logits, receiver_logits = self._conditional_component_logits(
            obs,
            latent_pi,
            phi=phi,
            theta=theta,
            speed=speed,
        )
        assert theta_logits is not None and speed_logits is not None and recovery_logits is not None
        hitter_shot_log_prob = (
            Categorical(logits=phi_logits).log_prob(phi)
            + Categorical(logits=theta_logits).log_prob(theta)
            + Categorical(logits=speed_logits).log_prob(speed)
        )
        hitter_recovery_log_prob = Categorical(logits=recovery_logits).log_prob(recovery)
        receiver_action = actions % self._conditional_receiver_count
        receiver_log_prob = Categorical(logits=receiver_logits).log_prob(receiver_action)
        hitter_entropy = (
            Categorical(logits=phi_logits).entropy()
            + Categorical(logits=theta_logits).entropy()
            + Categorical(logits=speed_logits).entropy()
            + Categorical(logits=recovery_logits).entropy()
        )
        receiver_entropy = Categorical(logits=receiver_logits).entropy()
        receiver_rows = obs[:, ROLE_IS_RECEIVER_INDEX] > 0.5
        zero_recovery_log_prob = th.zeros_like(hitter_recovery_log_prob)
        return (
            th.where(receiver_rows, receiver_log_prob, hitter_shot_log_prob),
            th.where(receiver_rows, zero_recovery_log_prob, hitter_recovery_log_prob),
            th.where(receiver_rows, receiver_entropy, hitter_entropy),
        )

    def _conditional_entropy(self, actions: th.Tensor, obs: th.Tensor, latent_pi: th.Tensor) -> th.Tensor:
        _, entropy = self._conditional_log_prob_entropy(actions, obs, latent_pi)
        return entropy

    def _conditional_padded_logits(self, obs: th.Tensor, latent_pi: th.Tensor) -> th.Tensor:
        receiver_logits = self._masked_logits(self.receiver_head(latent_pi), self._conditional_receiver_legal_mask(obs))
        padded = th.full((obs.shape[0], self.action_space.n), MASKED_LOGIT_VALUE, dtype=receiver_logits.dtype, device=obs.device)
        padded[:, : self._conditional_receiver_count] = receiver_logits
        return padded

    def _mixed_component_logits(
        self,
        obs: th.Tensor,
        latent_pi: th.Tensor,
        phi: th.Tensor | None = None,
        theta: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None, th.Tensor | None, th.Tensor]:
        legal_velocity = self._conditional_hitter_legal_mask(obs).any(dim=4)
        phi_logits = self._masked_logits(self.phi_head(latent_pi), legal_velocity.any(dim=(2, 3)))
        theta_logits = speed_logits = None
        if phi is not None:
            phi_one_hot = th.nn.functional.one_hot(phi, self._conditional_phi_count).float()
            theta_allowed = legal_velocity[th.arange(obs.shape[0], device=obs.device), phi].any(dim=2)
            theta_logits = self._masked_logits(self.theta_head(th.cat((latent_pi, phi_one_hot), dim=1)), theta_allowed)
        if phi is not None and theta is not None:
            theta_one_hot = th.nn.functional.one_hot(theta, self._conditional_theta_count).float()
            speed_allowed = legal_velocity[th.arange(obs.shape[0], device=obs.device), phi, theta]
            speed_logits = self._masked_logits(
                self.speed_head(th.cat((latent_pi, phi_one_hot, theta_one_hot), dim=1)),
                speed_allowed,
            )
        receiver_logits = self._masked_logits(self.receiver_head(latent_pi), self._conditional_receiver_legal_mask(obs))
        return phi_logits, theta_logits, speed_logits, receiver_logits

    def _mixed_recovery_distribution(
        self,
        obs: th.Tensor,
        latent_pi: th.Tensor,
        *,
        phi: th.Tensor,
        theta: th.Tensor,
        speed: th.Tensor,
    ) -> Normal:
        phi_one_hot = th.nn.functional.one_hot(phi, self._conditional_phi_count).float()
        theta_one_hot = th.nn.functional.one_hot(theta, self._conditional_theta_count).float()
        speed_one_hot = th.nn.functional.one_hot(speed, self._conditional_speed_count).float()
        recovery_context = self._conditional_recovery_context(obs, phi=phi, theta=theta, speed=speed, dtype=latent_pi.dtype)
        recovery_mean = th.tanh(
            self.recovery_head(
                th.cat((latent_pi, phi_one_hot, theta_one_hot, speed_one_hot, recovery_context), dim=1)
            )
        )
        recovery_log_std = th.full_like(recovery_mean, MIXED_RECOVERY_LOG_STD)
        return Normal(recovery_mean, recovery_log_std.exp())

    def _index_to_signed(self, index: th.Tensor, count: int) -> th.Tensor:
        if count <= 1:
            return th.zeros_like(index, dtype=th.float32)
        return 2.0 * index.float() / float(count - 1) - 1.0

    def _signed_to_index_tensor(self, value: th.Tensor, count: int) -> th.Tensor:
        if count <= 1:
            return th.zeros(value.shape[0], dtype=th.long, device=value.device)
        unit = 0.5 * (th.clamp(value.reshape(-1), -1.0, 1.0) + 1.0)
        return th.clamp(th.round(unit * float(count - 1)).long(), 0, count - 1)

    def _mixed_sample_actions(self, obs: th.Tensor, latent_pi: th.Tensor, *, deterministic: bool) -> th.Tensor:
        phi_logits, _, _, receiver_logits = self._mixed_component_logits(obs, latent_pi)
        phi = self._sample_categorical(phi_logits, deterministic)
        _, theta_logits, _, _ = self._mixed_component_logits(obs, latent_pi, phi=phi)
        assert theta_logits is not None
        theta = self._sample_categorical(theta_logits, deterministic)
        _, _, speed_logits, _ = self._mixed_component_logits(obs, latent_pi, phi=phi, theta=theta)
        assert speed_logits is not None
        speed = self._sample_categorical(speed_logits, deterministic)
        recovery_dist = self._mixed_recovery_distribution(obs, latent_pi, phi=phi, theta=theta, speed=speed)
        recovery = recovery_dist.mean if deterministic else th.clamp(recovery_dist.rsample(), -1.0, 1.0)

        receiver = self._sample_categorical(receiver_logits, deterministic)
        hitter_action = th.cat(
            (
                self._index_to_signed(phi, self._conditional_phi_count).reshape(-1, 1),
                self._index_to_signed(theta, self._conditional_theta_count).reshape(-1, 1),
                self._index_to_signed(speed, self._conditional_speed_count).reshape(-1, 1),
                recovery,
                th.zeros((obs.shape[0], 1), dtype=latent_pi.dtype, device=obs.device),
            ),
            dim=1,
        )
        receiver_action = th.cat(
            (
                th.zeros((obs.shape[0], 5), dtype=latent_pi.dtype, device=obs.device),
                self._index_to_signed(receiver, self._conditional_receiver_count).reshape(-1, 1),
            ),
            dim=1,
        )
        receiver_rows = (obs[:, ROLE_IS_RECEIVER_INDEX] > 0.5).reshape(-1, 1)
        return th.where(receiver_rows, receiver_action, hitter_action)

    def _mixed_log_prob_entropy(
        self,
        actions: th.Tensor,
        obs: th.Tensor,
        latent_pi: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = th.clamp(actions.reshape((-1, 6)).float(), -1.0, 1.0)
        phi = self._signed_to_index_tensor(actions[:, 0], self._conditional_phi_count)
        theta = self._signed_to_index_tensor(actions[:, 1], self._conditional_theta_count)
        speed = self._signed_to_index_tensor(actions[:, 2], self._conditional_speed_count)
        recovery = actions[:, 3:5]
        receiver_action = self._signed_to_index_tensor(actions[:, 5], self._conditional_receiver_count)

        phi_logits, theta_logits, speed_logits, receiver_logits = self._mixed_component_logits(
            obs,
            latent_pi,
            phi=phi,
            theta=theta,
        )
        assert theta_logits is not None and speed_logits is not None
        recovery_dist = self._mixed_recovery_distribution(obs, latent_pi, phi=phi, theta=theta, speed=speed)
        hitter_log_prob = (
            Categorical(logits=phi_logits).log_prob(phi)
            + Categorical(logits=theta_logits).log_prob(theta)
            + Categorical(logits=speed_logits).log_prob(speed)
            + recovery_dist.log_prob(recovery).sum(dim=1)
        )
        hitter_entropy = (
            Categorical(logits=phi_logits).entropy()
            + Categorical(logits=theta_logits).entropy()
            + Categorical(logits=speed_logits).entropy()
            + recovery_dist.entropy().sum(dim=1)
        )
        receiver_dist = Categorical(logits=receiver_logits)
        receiver_log_prob = receiver_dist.log_prob(receiver_action)
        receiver_entropy = receiver_dist.entropy()
        receiver_rows = obs[:, ROLE_IS_RECEIVER_INDEX] > 0.5
        return th.where(receiver_rows, receiver_log_prob, hitter_log_prob), th.where(
            receiver_rows,
            receiver_entropy,
            hitter_entropy,
        )

    def _continuous_param_pair(self, raw: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean, log_std = raw.chunk(2, dim=1)
        return th.tanh(mean), th.full_like(log_std, CONTINUOUS_LOG_STD)

    def _continuous_prefix(self, *values: th.Tensor) -> list[th.Tensor]:
        return [th.clamp(value.reshape(-1, 1), -1.0, 1.0) for value in values]

    def _continuous_sample_normal(self, mean: th.Tensor, log_std: th.Tensor, deterministic: bool) -> th.Tensor:
        if deterministic:
            return mean
        return th.clamp(Normal(mean, log_std.exp()).rsample(), -1.0, 1.0)

    def _continuous_sample_actions(self, obs: th.Tensor, latent_pi: th.Tensor, *, deterministic: bool) -> th.Tensor:
        phi_mean, phi_log_std = self._continuous_param_pair(self.phi_head(latent_pi))
        phi = self._continuous_sample_normal(phi_mean, phi_log_std, deterministic)
        theta_mean, theta_log_std = self._continuous_param_pair(
            self.theta_head(th.cat((latent_pi, *self._continuous_prefix(phi)), dim=1))
        )
        theta = self._continuous_sample_normal(theta_mean, theta_log_std, deterministic)
        speed_mean, speed_log_std = self._continuous_param_pair(
            self.speed_head(th.cat((latent_pi, *self._continuous_prefix(phi, theta)), dim=1))
        )
        speed = self._continuous_sample_normal(speed_mean, speed_log_std, deterministic)
        recovery_raw = self.recovery_head(th.cat((latent_pi, *self._continuous_prefix(phi, theta, speed)), dim=1))
        recovery_mean = th.tanh(recovery_raw[:, :2])
        recovery_log_std = th.full_like(recovery_raw[:, 2:], CONTINUOUS_LOG_STD)
        recovery = recovery_mean if deterministic else th.clamp(
            Normal(recovery_mean, recovery_log_std.exp()).rsample(),
            -1.0,
            1.0,
        )
        receiver_mean, receiver_log_std = self._continuous_param_pair(self.receiver_head(latent_pi))
        receiver = self._continuous_sample_normal(receiver_mean, receiver_log_std, deterministic)

        hitter_action = th.cat((phi, theta, speed, recovery, th.zeros_like(phi)), dim=1)
        receiver_action = th.cat((th.zeros_like(hitter_action[:, :5]), receiver), dim=1)
        receiver_rows = (obs[:, ROLE_IS_RECEIVER_INDEX] > 0.5).reshape(-1, 1)
        return th.where(receiver_rows, receiver_action, hitter_action)

    def _continuous_log_prob_entropy(
        self,
        actions: th.Tensor,
        obs: th.Tensor,
        latent_pi: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = actions.reshape((-1, 6)).float()
        actions = th.clamp(actions, -1.0, 1.0)
        phi = actions[:, 0:1]
        theta = actions[:, 1:2]
        speed = actions[:, 2:3]
        recovery = actions[:, 3:5]
        receiver = actions[:, 5:6]

        phi_mean, phi_log_std = self._continuous_param_pair(self.phi_head(latent_pi))
        theta_mean, theta_log_std = self._continuous_param_pair(
            self.theta_head(th.cat((latent_pi, *self._continuous_prefix(phi)), dim=1))
        )
        speed_mean, speed_log_std = self._continuous_param_pair(
            self.speed_head(th.cat((latent_pi, *self._continuous_prefix(phi, theta)), dim=1))
        )
        recovery_raw = self.recovery_head(th.cat((latent_pi, *self._continuous_prefix(phi, theta, speed)), dim=1))
        recovery_mean = th.tanh(recovery_raw[:, :2])
        recovery_log_std = th.full_like(recovery_raw[:, 2:], CONTINUOUS_LOG_STD)
        receiver_mean, receiver_log_std = self._continuous_param_pair(self.receiver_head(latent_pi))

        hitter_log_prob = (
            Normal(phi_mean, phi_log_std.exp()).log_prob(phi).sum(dim=1)
            + Normal(theta_mean, theta_log_std.exp()).log_prob(theta).sum(dim=1)
            + Normal(speed_mean, speed_log_std.exp()).log_prob(speed).sum(dim=1)
            + Normal(recovery_mean, recovery_log_std.exp()).log_prob(recovery).sum(dim=1)
        )
        hitter_entropy = (
            Normal(phi_mean, phi_log_std.exp()).entropy().sum(dim=1)
            + Normal(theta_mean, theta_log_std.exp()).entropy().sum(dim=1)
            + Normal(speed_mean, speed_log_std.exp()).entropy().sum(dim=1)
            + Normal(recovery_mean, recovery_log_std.exp()).entropy().sum(dim=1)
        )
        receiver_dist = Normal(receiver_mean, receiver_log_std.exp())
        receiver_log_prob = receiver_dist.log_prob(receiver).sum(dim=1)
        receiver_entropy = receiver_dist.entropy().sum(dim=1)
        receiver_rows = obs[:, ROLE_IS_RECEIVER_INDEX] > 0.5
        return th.where(receiver_rows, receiver_log_prob, hitter_log_prob), th.where(
            receiver_rows,
            receiver_entropy,
            hitter_entropy,
        )
