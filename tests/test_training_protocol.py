from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch as th

from badminton.callbacks import SafeWinRateEvalCallback
from badminton.action_space import DiscreteActionConfig
from badminton.config import SimulationConfig
from badminton.factorized_ppo import RecoveryFactorizedPPO, RecoveryFactorizedRolloutBuffer
from badminton.policy import CONTINUOUS_LOG_STD_MAX, CONTINUOUS_LOG_STD_MIN, MaskedBadmintonPolicy
from badminton.rl_env import BadmintonRLEnv, RLEnvConfig
from badminton.shot_cf import ShotCFCandidate, select_diverse_shot_candidates
from badminton.state import ShotAction
from badminton.selfplay import FrozenCheckpointOpponent, LiveModelOpponent, MixedCheckpointOpponent, SelfPlayProgressVideoCallback
from scripts import train_ppo, train_selfplay


class TrainingProtocolTests(unittest.TestCase):
    def test_train_ppo_defaults_match_latest_protocol(self) -> None:
        with patch("sys.argv", ["train_ppo.py"]):
            args = train_ppo.parse_args()

        env_kwargs = train_ppo.build_env_kwargs(args)
        rl_config = env_kwargs["rl_config"]
        sim_config = env_kwargs["config"]

        self.assertEqual(args.initial_server, "random")
        self.assertEqual(args.policy_type, "velocity_oriented")
        self.assertFalse(args.mask_mid_rally_hitter_actions)
        self.assertTrue(args.use_recovery_factorized_advantage)
        self.assertEqual(args.recovery_counterfactual_baseline, "average")
        self.assertEqual(args.recovery_counterfactual_other_sample_count, 24)
        self.assertAlmostEqual(args.recovery_counterfactual_advantage_coef, 0.05)
        self.assertAlmostEqual(args.recovery_counterfactual_distribution_coef, 0.0)
        self.assertAlmostEqual(args.recovery_counterfactual_distribution_temperature, 0.25)
        self.assertEqual(args.counterfactual_opponent_response_samples, 2)
        self.assertTrue(args.recovery_counterfactual_expected_response_target)
        self.assertFalse(args.use_shot_cf)
        self.assertAlmostEqual(args.shot_cf_coef, 0.1)
        self.assertEqual(args.shot_cf_top_m, 20)
        self.assertEqual(args.shot_cf_num_modes, 3)
        self.assertAlmostEqual(args.shot_cf_min_landing_dist, 1.0)
        self.assertEqual(args.shot_cf_depth, 1)
        self.assertTrue(args.shot_cf_include_chosen)
        self.assertTrue(args.shot_cf_skip_low_diversity)
        self.assertEqual(args.shot_cf_min_modes, 2)
        self.assertTrue(args.shot_cf_value_detach)
        self.assertTrue(args.shot_cf_normalize)
        self.assertFalse(args.shot_cf_debug_log)
        self.assertFalse(rl_config.use_shot_cf)
        self.assertEqual(rl_config.counterfactual_opponent_response_samples, 2)
        self.assertTrue(rl_config.recovery_counterfactual_expected_response_target)
        self.assertTrue(args.random_service_x)
        self.assertEqual(args.n_envs, 8)
        self.assertAlmostEqual(args.ent_coef, 0.002)
        self.assertAlmostEqual(args.ent_coef_final, 0.002)
        self.assertAlmostEqual(args.reaction_time, 0.15)
        self.assertAlmostEqual(args.loop_penalty, 0.03)
        self.assertEqual(args.max_rally_stages, 120)
        self.assertAlmostEqual(args.max_rally_penalty, 1.0)
        self.assertAlmostEqual(args.pressure_reward_weight, 0.0)
        self.assertAlmostEqual(args.player_speed, 5.0)
        self.assertAlmostEqual(args.racket_length, 1.6)
        self.assertAlmostEqual(args.max_hitting_height, 2.6)
        self.assertEqual(args.movement_model, "accelerated")
        self.assertAlmostEqual(args.player_acceleration, 8.0)
        self.assertEqual(args.trajectory_mode, "drag_square")
        self.assertEqual(args.intercept_count, 20)
        self.assertAlmostEqual(args.mirror_match_fraction, 0.25)
        self.assertAlmostEqual(rl_config.reset_sampling.opponent_serve_start_prob, 0.0)
        self.assertAlmostEqual(rl_config.reward.stage_penalty, 0.0)
        self.assertAlmostEqual(rl_config.reward.stall_penalty, 0.0)
        self.assertEqual(rl_config.reward.stall_penalty_start, 24)
        self.assertEqual(sim_config.action.trajectory_mode, "drag_square")
        self.assertEqual(sim_config.action.intercept_count, 20)
        self.assertAlmostEqual(sim_config.action.vy_max_forward, 100.0)
        self.assertAlmostEqual(sim_config.player.v_max, 5.0)
        self.assertAlmostEqual(sim_config.player.r_reach, 1.6)
        self.assertAlmostEqual(sim_config.action.reaction_miss_fast_threshold, 0.1)
        self.assertAlmostEqual(sim_config.action.reaction_miss_fast_probability, 0.8)
        self.assertAlmostEqual(sim_config.action.reaction_miss_secondary_threshold, 0.5)
        self.assertAlmostEqual(sim_config.action.reaction_miss_secondary_probability, 0.0)
        self.assertAlmostEqual(sim_config.action.reaction_miss_zero_threshold, 0.5)
        self.assertAlmostEqual(sim_config.player.z_max, 2.6)
        self.assertEqual(sim_config.player.movement_model, "accelerated")
        self.assertAlmostEqual(sim_config.player.acceleration, 8.0)

    def test_train_selfplay_defaults_match_latest_protocol(self) -> None:
        with patch("sys.argv", ["train_selfplay.py", "--base-checkpoint-path", "dummy.zip"]):
            args = train_selfplay.parse_args()

        self.assertEqual(args.initial_server, "random")
        self.assertEqual(args.policy_type, "velocity_oriented")
        self.assertFalse(args.mask_mid_rally_hitter_actions)
        self.assertTrue(args.use_recovery_factorized_advantage)
        self.assertEqual(args.recovery_counterfactual_baseline, "average")
        self.assertEqual(args.recovery_counterfactual_other_sample_count, 24)
        self.assertAlmostEqual(args.recovery_counterfactual_advantage_coef, 0.05)
        self.assertAlmostEqual(args.recovery_counterfactual_distribution_coef, 0.0)
        self.assertAlmostEqual(args.recovery_counterfactual_distribution_temperature, 0.25)
        self.assertEqual(args.counterfactual_opponent_response_samples, 2)
        self.assertTrue(args.recovery_counterfactual_expected_response_target)
        self.assertFalse(args.use_shot_cf)
        self.assertAlmostEqual(args.shot_cf_coef, 0.1)
        self.assertEqual(args.shot_cf_top_m, 20)
        self.assertEqual(args.shot_cf_num_modes, 3)
        self.assertAlmostEqual(args.shot_cf_min_landing_dist, 1.0)
        self.assertEqual(args.shot_cf_depth, 1)
        self.assertTrue(args.shot_cf_include_chosen)
        self.assertTrue(args.shot_cf_skip_low_diversity)
        self.assertEqual(args.shot_cf_min_modes, 2)
        self.assertTrue(args.shot_cf_value_detach)
        self.assertTrue(args.shot_cf_normalize)
        self.assertFalse(args.shot_cf_debug_log)
        self.assertEqual(args.n_envs, 8)
        self.assertEqual(args.log_interval, 10)
        self.assertEqual(args.save_interval, 2000)
        self.assertEqual(args.eval_freq, 100000)
        self.assertAlmostEqual(args.random_start_prob, 0.0)
        self.assertAlmostEqual(args.midrally_start_prob, 0.0)
        self.assertAlmostEqual(args.reaction_time, 0.15)
        self.assertAlmostEqual(args.loop_penalty, 0.1)
        self.assertEqual(args.max_rally_stages, 120)
        self.assertAlmostEqual(args.max_rally_penalty, 1.0)
        self.assertAlmostEqual(args.stage_penalty, 0.0)
        self.assertAlmostEqual(args.stall_penalty, 0.0)
        self.assertAlmostEqual(args.pressure_reward_weight, 0.0)
        self.assertAlmostEqual(args.attack_reward_weight, 0.0)
        self.assertAlmostEqual(args.attack_min_speed, 18.0)
        self.assertAlmostEqual(args.attack_downward_vz_threshold, 0.0)
        self.assertAlmostEqual(args.feasible_pressure_reward_weight, 0.0)
        self.assertAlmostEqual(args.no_feasible_intercept_bonus, 0.0)
        self.assertAlmostEqual(args.opponent_intercept_continue_penalty, 0.0)
        self.assertAlmostEqual(args.defensive_lift_reward_weight, 0.0)
        self.assertAlmostEqual(args.intercept_flight_ratio_reward_weight, 0.0)
        self.assertAlmostEqual(args.defensive_lift_min_theta_deg, 15.0)
        self.assertAlmostEqual(args.defensive_lift_target_flight_time, 1.4)
        self.assertAlmostEqual(args.defensive_lift_min_depth_ratio, 0.7)
        self.assertAlmostEqual(args.intercept_ratio_min_intended_flight_time, 0.8)
        self.assertAlmostEqual(args.opponent_travel_reward_weight, 0.0)
        self.assertAlmostEqual(args.return_depth_reward_weight, 0.0)
        self.assertAlmostEqual(args.net_proximity_reward_weight, 0.0)
        self.assertAlmostEqual(args.net_proximity_threshold, 0.5)
        self.assertAlmostEqual(args.player_speed, 5.0)
        self.assertAlmostEqual(args.racket_length, 1.6)
        self.assertAlmostEqual(args.max_hitting_height, 2.6)
        self.assertEqual(args.movement_model, "accelerated")
        self.assertAlmostEqual(args.player_acceleration, 8.0)
        self.assertEqual(args.trajectory_mode, "drag_square")
        self.assertAlmostEqual(args.shuttle_speed_max, 100.0)
        self.assertEqual(args.intercept_count, 20)
        self.assertAlmostEqual(args.reaction_miss_fast_threshold, 0.1)
        self.assertAlmostEqual(args.reaction_miss_fast_probability, 0.8)
        self.assertAlmostEqual(args.reaction_miss_secondary_threshold, 0.5)
        self.assertAlmostEqual(args.reaction_miss_secondary_probability, 0.0)
        self.assertAlmostEqual(args.reaction_miss_zero_threshold, 0.5)
        self.assertEqual(args.theta_bins, 8)
        self.assertEqual(args.phi_bins, 11)
        self.assertEqual(args.speed_bins, 5)
        self.assertEqual(args.x_rec_bins, 5)
        self.assertEqual(args.y_rec_bins, 5)
        self.assertIsNone(args.recovery_x_margin)
        self.assertEqual(train_selfplay.build_sim_config(args).action.recovery_x_margin, 0.25)
        self.assertEqual(train_selfplay.build_sim_config(args).action.recovery_net_margin, 0.3)
        self.assertEqual(train_selfplay.build_sim_config(args).action.recovery_back_margin, 0.5)
        self.assertFalse(train_selfplay.build_sim_config(args).action.conditional_recovery_grid)

    def test_continuous_policy_default_log_std_is_fixed(self) -> None:
        self.assertAlmostEqual(CONTINUOUS_LOG_STD_MIN, -3.0)
        self.assertAlmostEqual(CONTINUOUS_LOG_STD_MAX, -3.0)
        raw = th.tensor([[2.0, -8.0], [0.0, 1.0]])

        _, log_std = MaskedBadmintonPolicy._continuous_param_pair(None, raw)

        self.assertTrue(th.all(log_std == -3.0))

    def test_train_selfplay_1d_defaults_use_compatible_action_bins(self) -> None:
        with patch("sys.argv", ["train_selfplay.py", "--base-checkpoint-path", "dummy.zip", "--court-mode", "1d"]):
            args = train_selfplay.parse_args()

        self.assertEqual(args.theta_bins, 15)
        self.assertEqual(args.speed_bins, 11)

    def test_train_selfplay_1d_explicit_action_bins_override_defaults(self) -> None:
        with patch(
            "sys.argv",
            [
                "train_selfplay.py",
                "--base-checkpoint-path",
                "dummy.zip",
                "--court-mode",
                "1d",
                "--theta-bins",
                "9",
                "--speed-bins",
                "8",
            ],
        ):
            args = train_selfplay.parse_args()

        self.assertEqual(args.theta_bins, 9)
        self.assertEqual(args.speed_bins, 8)

    def test_stochastic_defaults_propagate_through_training_helpers(self) -> None:
        self.assertFalse(FrozenCheckpointOpponent.__dataclass_fields__["deterministic"].default)
        self.assertFalse(LiveModelOpponent.__dataclass_fields__["deterministic"].default)
        self.assertFalse(MixedCheckpointOpponent.__dataclass_fields__["deterministic"].default)
        self.assertFalse(SelfPlayProgressVideoCallback.__init__.__kwdefaults__["deterministic"])
        self.assertFalse(SafeWinRateEvalCallback.__init__.__kwdefaults__["deterministic"])

    def test_shot_cf_candidate_selection_keeps_diverse_landings(self) -> None:
        action = ShotAction(v_x=0.0, v_y=1.0, v_z=1.0, x_rec=0.0, y_rec=0.0)
        candidates = [
            ShotCFCandidate(0, (0, 0, 0), 0.0, 0.5, action, 0.0, 0.0),
            ShotCFCandidate(1, (0, 0, 1), -0.1, 0.3, action, 0.2, 0.1),
            ShotCFCandidate(2, (0, 1, 0), -0.2, 0.2, action, 2.0, 0.0),
            ShotCFCandidate(3, (1, 0, 0), -0.3, 0.1, action, 0.0, 2.0),
        ]

        selection = select_diverse_shot_candidates(
            candidates,
            chosen_candidate=candidates[2],
            num_modes=3,
            min_landing_dist=1.0,
            include_chosen=True,
            skip_low_diversity=True,
            min_modes=2,
        )

        self.assertFalse(selection.skipped)
        self.assertEqual(len(selection.candidates), 3)
        for left_index, left in enumerate(selection.candidates):
            for right in selection.candidates[left_index + 1 :]:
                self.assertGreaterEqual(
                    float(np.hypot(left.landing_x - right.landing_x, left.landing_y - right.landing_y)),
                    1.0,
                )
        self.assertEqual(selection.candidates[selection.chosen_index].flat_index, 2)

    def test_shot_cf_candidate_selection_skips_low_diversity(self) -> None:
        action = ShotAction(v_x=0.0, v_y=1.0, v_z=1.0, x_rec=0.0, y_rec=0.0)
        candidates = [
            ShotCFCandidate(0, (0, 0, 0), 0.0, 0.5, action, 0.0, 0.0),
            ShotCFCandidate(1, (0, 0, 1), -0.1, 0.3, action, 0.1, 0.1),
            ShotCFCandidate(2, (0, 1, 0), -0.2, 0.2, action, 0.2, 0.0),
        ]

        selection = select_diverse_shot_candidates(
            candidates,
            chosen_candidate=candidates[0],
            num_modes=3,
            min_landing_dist=1.0,
            include_chosen=True,
            skip_low_diversity=True,
            min_modes=2,
        )

        self.assertTrue(selection.skipped)
        self.assertEqual(selection.skip_reason, "low_diversity")

    def test_shot_cf_candidate_selection_includes_chosen_mode(self) -> None:
        action = ShotAction(v_x=0.0, v_y=1.0, v_z=1.0, x_rec=0.0, y_rec=0.0)
        candidates = [
            ShotCFCandidate(0, (0, 0, 0), 0.0, 0.5, action, 0.0, 0.0),
            ShotCFCandidate(1, (0, 0, 1), -0.1, 0.3, action, 2.0, 0.0),
        ]
        chosen = ShotCFCandidate(5, (2, 0, 0), -2.0, 0.01, action, -2.0, 0.0)

        selection = select_diverse_shot_candidates(
            candidates,
            chosen_candidate=chosen,
            num_modes=2,
            min_landing_dist=1.0,
            include_chosen=True,
            skip_low_diversity=True,
            min_modes=2,
        )

        self.assertFalse(selection.skipped)
        self.assertEqual(selection.candidates[selection.chosen_index].flat_index, 5)

    def test_shot_cf_advantage_targets_are_detached(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(
            config=config,
            rl_config=RLEnvConfig(
                policy_type="velocity_oriented",
                initial_server="left",
                max_stages_per_rally=4,
                use_shot_cf=True,
            ),
            seed=29,
        )
        model = RecoveryFactorizedPPO(
            MaskedBadmintonPolicy,
            env,
            policy_kwargs={
                "sim_config": config,
                "policy_type": "velocity_oriented",
            },
            use_recovery_factorized_advantage=True,
            use_shot_cf=True,
            shot_cf_top_m=6,
            shot_cf_num_modes=2,
            shot_cf_min_modes=1,
            shot_cf_value_detach=True,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-4,
            ent_coef=0.0,
            verbose=0,
            seed=29,
        )

        obs, _ = env.reset(seed=29)
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with th.no_grad():
            actions, _, _ = model.policy(obs_tensor)
        _, _, _, _, info = env.step(int(actions.item()))

        advantage, mask = model._shot_cf_advantage_from_transitions(obs_tensor, actions, [info])

        self.assertFalse(advantage.requires_grad)
        self.assertFalse(mask.requires_grad)

    def test_recovery_factorized_components_sum_to_conditional_logprob(self) -> None:
        config = SimulationConfig()
        discrete_config = DiscreteActionConfig(phi_bins=5, theta_bins=4, speed_bins=3, x_rec_bins=3, y_rec_bins=3)
        env = BadmintonRLEnv(
            config=config,
            rl_config=RLEnvConfig(policy_type="velocity_oriented", initial_server="left", max_stages_per_rally=4),
            discrete_action_config=discrete_config,
            seed=11,
        )
        model = RecoveryFactorizedPPO(
            MaskedBadmintonPolicy,
            env,
            policy_kwargs={
                "sim_config": config,
                "discrete_action_config": discrete_config,
                "policy_type": "velocity_oriented",
            },
            use_recovery_factorized_advantage=True,
            recovery_counterfactual_advantage_coef=1.0,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-4,
            ent_coef=0.0,
            verbose=0,
            seed=11,
        )

        obs, _ = env.reset(seed=11)
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        actions, _, log_prob = model.policy(obs_tensor)
        _, logp_shot, logp_recovery, entropy = model.policy.evaluate_recovery_factorized_actions(obs_tensor, actions)

        self.assertIsInstance(model.rollout_buffer, RecoveryFactorizedRolloutBuffer)
        self.assertTrue(th.allclose(logp_shot + logp_recovery, log_prob, atol=1e-6))
        self.assertIsNotNone(entropy)

        model.learn(total_timesteps=4, progress_bar=False)

    def test_recovery_factorized_advantage_uses_real_next_observation_or_target(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(
            config=config,
            rl_config=RLEnvConfig(policy_type="velocity_oriented", initial_server="left", max_stages_per_rally=4),
            seed=13,
        )
        model = RecoveryFactorizedPPO(
            MaskedBadmintonPolicy,
            env,
            policy_kwargs={
                "sim_config": config,
                "policy_type": "velocity_oriented",
            },
            use_recovery_factorized_advantage=True,
            recovery_counterfactual_advantage_coef=1.0,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-4,
            ent_coef=0.0,
            verbose=0,
            seed=13,
        )

        obs, _ = env.reset(seed=13)
        next_obs = obs.copy()
        next_obs[14] = 1.0
        next_obs[33 : 33 + config.action.intercept_count] = 0.0
        next_obs[33] = 1.0

        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        next_obs_tensor, _ = model.policy.obs_to_tensor(next_obs)
        with th.no_grad():
            before_value = model.policy.predict_values(obs_tensor).flatten()
            next_value = model.policy.predict_values(next_obs_tensor).flatten()

        advantage, mask, _, _ = model._recovery_advantage_from_transitions(
            before_value,
            [{"recovery_factorized_action": True}],
            np.asarray([next_obs], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
        )
        self.assertTrue(th.allclose(mask, th.ones_like(mask)))
        self.assertTrue(th.allclose(advantage, next_value - before_value, atol=1e-6))

        terminal_advantage, terminal_mask, _, _ = model._recovery_advantage_from_transitions(
            before_value,
            [{"recovery_factorized_action": True, "recovery_factorized_target": -1.0}],
            np.asarray([obs], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([True]),
        )
        self.assertTrue(th.allclose(terminal_mask, th.ones_like(terminal_mask)))
        self.assertTrue(th.allclose(terminal_advantage, -th.ones_like(before_value) - before_value, atol=1e-6))

    def test_recovery_factorized_advantage_uses_counterfactual_recovery_grid(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(
            config=config,
            rl_config=RLEnvConfig(policy_type="velocity_oriented", initial_server="left", max_stages_per_rally=4),
            seed=17,
        )
        model = RecoveryFactorizedPPO(
            MaskedBadmintonPolicy,
            env,
            policy_kwargs={
                "sim_config": config,
                "policy_type": "velocity_oriented",
            },
            use_recovery_factorized_advantage=True,
            recovery_counterfactual_advantage_coef=1.0,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-4,
            ent_coef=0.0,
            verbose=0,
            seed=17,
        )

        obs, _ = env.reset(seed=17)
        cf_obs = np.stack([obs.copy(), obs.copy(), obs.copy()]).astype(np.float32)
        cf_obs[1, 0] = np.clip(cf_obs[1, 0] + 0.1, -1.0, 1.0)
        cf_obs[2, 1] = np.clip(cf_obs[2, 1] - 0.1, -1.0, 1.0)

        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with th.no_grad():
            before_value = model.policy.predict_values(obs_tensor).flatten()
            cf_values = model.policy.predict_values(th.as_tensor(cf_obs, device=model.device)).flatten()

        advantage, mask, distribution_target, distribution_mask = model._recovery_advantage_from_transitions(
            before_value,
            [
                {
                    "recovery_factorized_action": True,
                    "recovery_factorized_counterfactual_observations": cf_obs,
                    "recovery_factorized_counterfactual_chosen_index": 1,
                    "recovery_factorized_counterfactual_baseline_indices": [0, 2],
                    "recovery_factorized_counterfactual_sampled_indices": [3, 5, 7],
                }
            ],
            np.asarray([obs], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
        )
        expected = cf_values[1] - cf_values[[0, 2]].mean()
        self.assertTrue(th.allclose(mask, th.ones_like(mask)))
        self.assertTrue(th.allclose(advantage, expected.reshape_as(advantage), atol=1e-6))
        self.assertEqual(int(th.count_nonzero(distribution_mask).item()), 3)
        self.assertAlmostEqual(float(distribution_target.sum().item()), 1.0)
        self.assertGreater(float(distribution_target[0, 5]), 0.0)

        best_model = RecoveryFactorizedPPO(
            MaskedBadmintonPolicy,
            env,
            policy_kwargs={
                "sim_config": config,
                "policy_type": "velocity_oriented",
            },
            use_recovery_factorized_advantage=True,
            recovery_counterfactual_baseline="best",
            recovery_counterfactual_advantage_coef=1.0,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-4,
            ent_coef=0.0,
            verbose=0,
            seed=17,
        )
        best_model.policy.load_state_dict(model.policy.state_dict())
        best_advantage, best_mask, _, _ = best_model._recovery_advantage_from_transitions(
            before_value,
            [
                {
                    "recovery_factorized_action": True,
                    "recovery_factorized_counterfactual_observations": cf_obs,
                    "recovery_factorized_counterfactual_chosen_index": 1,
                    "recovery_factorized_counterfactual_baseline_indices": [0, 2],
                }
            ],
            np.asarray([obs], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
        )
        self.assertTrue(th.allclose(best_mask, th.ones_like(best_mask)))
        self.assertTrue(th.allclose(best_advantage, (cf_values[1] - cf_values[[0, 2]].max()).reshape_as(best_advantage), atol=1e-6))

    def test_recovery_counterfactual_expected_response_target_mixes_miss_risk(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(
            config=config,
            rl_config=RLEnvConfig(policy_type="velocity_oriented", initial_server="left", max_stages_per_rally=4),
            seed=19,
        )
        model = RecoveryFactorizedPPO(
            MaskedBadmintonPolicy,
            env,
            policy_kwargs={
                "sim_config": config,
                "policy_type": "velocity_oriented",
            },
            use_recovery_factorized_advantage=True,
            recovery_counterfactual_advantage_coef=1.0,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-4,
            ent_coef=0.0,
            verbose=0,
            seed=19,
        )

        obs, _ = env.reset(seed=19)
        cf_obs = np.stack([obs.copy(), obs.copy(), obs.copy()]).astype(np.float32)
        expected_obs = cf_obs.copy()
        expected_obs[0, 0] = np.clip(expected_obs[0, 0] + 0.05, -1.0, 1.0)
        expected_obs[1, 1] = np.clip(expected_obs[1, 1] - 0.07, -1.0, 1.0)
        expected_obs[2, 2] = np.clip(expected_obs[2, 2] + 0.09, -1.0, 1.0)

        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with th.no_grad():
            before_value = model.policy.predict_values(obs_tensor).flatten()
            no_miss_values = model.policy.predict_values(th.as_tensor(expected_obs, device=model.device)).flatten()

        loss_reward = -1.0
        miss_probabilities = th.as_tensor([0.0, 0.25, 0.5], dtype=no_miss_values.dtype, device=no_miss_values.device)
        no_miss_scores = no_miss_values.clone()
        no_miss_scores[2] = 0.75
        expected_scores = miss_probabilities * loss_reward + (1.0 - miss_probabilities) * no_miss_scores
        expected = expected_scores[1] - expected_scores[[0, 2]].mean()

        advantage, mask, _, _ = model._recovery_advantage_from_transitions(
            before_value,
            [
                {
                    "recovery_factorized_action": True,
                    "recovery_factorized_counterfactual_observations": cf_obs,
                    "recovery_factorized_counterfactual_expected_response_target": True,
                    "recovery_factorized_counterfactual_expected_observations": expected_obs,
                    "recovery_factorized_counterfactual_expected_miss_probabilities": np.asarray(
                        [0.0, 0.25, 0.5],
                        dtype=np.float32,
                    ),
                    "recovery_factorized_counterfactual_expected_no_miss_targets": np.asarray(
                        [np.nan, np.nan, 0.75],
                        dtype=np.float32,
                    ),
                    "recovery_factorized_counterfactual_loss_reward": loss_reward,
                    "recovery_factorized_counterfactual_chosen_index": 1,
                    "recovery_factorized_counterfactual_baseline_indices": [0, 2],
                }
            ],
            np.asarray([obs], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
        )
        self.assertTrue(th.allclose(mask, th.ones_like(mask)))
        self.assertTrue(th.allclose(advantage, expected.reshape_as(advantage), atol=1e-6))

    def test_sampled_recovery_distribution_loss_raises_better_sampled_bin(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(
            config=config,
            rl_config=RLEnvConfig(policy_type="velocity_oriented", initial_server="left", max_stages_per_rally=4),
            seed=23,
        )
        model = RecoveryFactorizedPPO(
            MaskedBadmintonPolicy,
            env,
            policy_kwargs={
                "sim_config": config,
                "policy_type": "velocity_oriented",
            },
            use_recovery_factorized_advantage=True,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-4,
            ent_coef=0.0,
            verbose=0,
            seed=23,
        )

        obs, _ = env.reset(seed=23)
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        actions = th.zeros(1, dtype=th.long, device=model.device)
        bin_count = int(model.policy._conditional_recovery_count)
        targets = th.zeros((1, bin_count), dtype=obs_tensor.dtype, device=model.device)
        mask = th.zeros_like(targets)
        targets[0, 1] = 1.0
        mask[0, :2] = 1.0

        model.policy.optimizer.zero_grad()
        loss = model._sampled_recovery_distribution_loss(obs_tensor, actions, targets, mask)
        loss.backward()

        gradient = model.policy.recovery_head.bias.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient[0]), 0.0)
        self.assertLess(float(gradient[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
