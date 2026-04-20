from __future__ import annotations

import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from badminton1d.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton1d.config import SimulationConfig
from badminton1d.state import StageState

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
FEASIBLE_MASK_START_INDEX = 29
MASKED_LOGIT_VALUE = -1e9


def _denormalize_signed(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return float(max(min(value, 1.0), -1.0) * scale)


def _denormalize_unit(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return lower
    clipped = float(max(min(value, 1.0), 0.0))
    return lower + clipped * (upper - lower)


def _state_from_observation_row(obs_row: th.Tensor, config: SimulationConfig) -> StageState:
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
        stage_index=0,
    )


def apply_hitter_action_mask(
    logits: th.Tensor,
    obs: th.Tensor,
    *,
    mapper: DiscreteActionMapper | None,
) -> th.Tensor:
    if mapper is None or logits.ndim != 2 or obs.ndim != 2:
        return logits

    masked_logits = logits.clone()
    hitter_rows = (obs[:, ROLE_IS_RECEIVER_INDEX] < 0.5) & (obs[:, STAGE_PROGRESS_INDEX] <= 1e-6)
    if not th.any(hitter_rows):
        return masked_logits

    hitter_indices = th.nonzero(hitter_rows, as_tuple=False).flatten()
    for row_index in hitter_indices.tolist():
        state = _state_from_observation_row(obs[row_index], mapper.config)
        legal_mask = mapper.legal_serve_hitter_mask(state)
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


class MaskedBadmintonPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        sim_config: SimulationConfig | None = None,
        discrete_action_config: DiscreteActionConfig | None = None,
        **kwargs,
    ):
        self.sim_config = sim_config
        self.discrete_action_config = discrete_action_config
        self._action_mapper = None
        if sim_config is not None:
            self._action_mapper = DiscreteActionMapper(sim_config, discrete_action_config)
        super().__init__(*args, **kwargs)

    def _receiver_action_count(self, obs: th.Tensor) -> int:
        return max(int(obs.shape[-1]) - FEASIBLE_MASK_START_INDEX, 0)

    def _masked_action_distribution(self, obs: th.Tensor, latent_pi: th.Tensor):
        action_logits = self.action_net(latent_pi)
        if isinstance(self.action_dist, CategoricalDistribution):
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
        distribution = self._masked_action_distribution(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy
