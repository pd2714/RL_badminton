from __future__ import annotations

import unittest
from unittest.mock import patch

import torch as th

from badminton1d.callbacks import SafeWinRateEvalCallback
from badminton1d.action_space import DiscreteActionConfig
from badminton1d.config import SimulationConfig
from badminton1d.factorized_ppo import RecoveryFactorizedPPO, RecoveryFactorizedRolloutBuffer
from badminton1d.policy import CONTINUOUS_LOG_STD_MAX, CONTINUOUS_LOG_STD_MIN, MaskedBadmintonPolicy
from badminton1d.rl_env import BadmintonRLEnv, RLEnvConfig
from badminton1d.selfplay import FrozenCheckpointOpponent, LiveModelOpponent, MixedCheckpointOpponent, SelfPlayProgressVideoCallback
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
        self.assertFalse(args.use_recovery_factorized_advantage)
        self.assertTrue(args.random_service_x)
        self.assertEqual(args.n_envs, 8)
        self.assertAlmostEqual(args.ent_coef, 0.02)
        self.assertAlmostEqual(args.ent_coef_final, 0.002)
        self.assertAlmostEqual(args.reaction_time, 0.15)
        self.assertAlmostEqual(args.loop_penalty, 0.03)
        self.assertEqual(args.max_rally_stages, 120)
        self.assertAlmostEqual(args.max_rally_penalty, 1.0)
        self.assertAlmostEqual(args.pressure_reward_weight, 0.0)
        self.assertAlmostEqual(args.player_speed, 4.0)
        self.assertAlmostEqual(args.racket_length, 1.3)
        self.assertAlmostEqual(args.max_hitting_height, 2.6)
        self.assertEqual(args.movement_model, "accelerated")
        self.assertAlmostEqual(args.player_acceleration, 6.5)
        self.assertEqual(args.trajectory_mode, "drag_square")
        self.assertEqual(args.intercept_count, 20)
        self.assertAlmostEqual(args.mirror_match_fraction, 0.25)
        self.assertAlmostEqual(rl_config.reset_sampling.opponent_serve_start_prob, 0.0)
        self.assertAlmostEqual(rl_config.reward.stage_penalty, 0.0)
        self.assertAlmostEqual(rl_config.reward.stall_penalty, 0.0)
        self.assertEqual(rl_config.reward.stall_penalty_start, 24)
        self.assertEqual(sim_config.action.trajectory_mode, "drag_square")
        self.assertEqual(sim_config.action.intercept_count, 20)
        self.assertAlmostEqual(sim_config.action.vy_max_forward, 80.0)
        self.assertAlmostEqual(sim_config.player.v_max, 4.0)
        self.assertAlmostEqual(sim_config.player.r_reach, 1.3)
        self.assertAlmostEqual(sim_config.action.reaction_miss_fast_threshold, 0.1)
        self.assertAlmostEqual(sim_config.action.reaction_miss_fast_probability, 0.9)
        self.assertAlmostEqual(sim_config.action.reaction_miss_secondary_threshold, 0.5)
        self.assertAlmostEqual(sim_config.action.reaction_miss_secondary_probability, 0.3)
        self.assertAlmostEqual(sim_config.action.reaction_miss_zero_threshold, 0.7)
        self.assertAlmostEqual(sim_config.player.z_max, 2.6)
        self.assertEqual(sim_config.player.movement_model, "accelerated")
        self.assertAlmostEqual(sim_config.player.acceleration, 6.5)

    def test_train_selfplay_defaults_match_latest_protocol(self) -> None:
        with patch("sys.argv", ["train_selfplay.py", "--base-checkpoint-path", "dummy.zip"]):
            args = train_selfplay.parse_args()

        self.assertEqual(args.initial_server, "random")
        self.assertEqual(args.policy_type, "velocity_oriented")
        self.assertFalse(args.mask_mid_rally_hitter_actions)
        self.assertFalse(args.use_recovery_factorized_advantage)
        self.assertEqual(args.n_envs, 8)
        self.assertEqual(args.save_interval, 2000)
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
        self.assertAlmostEqual(args.player_speed, 4.0)
        self.assertAlmostEqual(args.racket_length, 1.3)
        self.assertAlmostEqual(args.max_hitting_height, 2.6)
        self.assertEqual(args.movement_model, "accelerated")
        self.assertAlmostEqual(args.player_acceleration, 6.5)
        self.assertEqual(args.trajectory_mode, "drag_square")
        self.assertAlmostEqual(args.shuttle_speed_max, 80.0)
        self.assertEqual(args.intercept_count, 20)
        self.assertAlmostEqual(args.reaction_miss_fast_threshold, 0.1)
        self.assertAlmostEqual(args.reaction_miss_fast_probability, 0.9)
        self.assertAlmostEqual(args.reaction_miss_secondary_threshold, 0.5)
        self.assertAlmostEqual(args.reaction_miss_secondary_probability, 0.3)
        self.assertAlmostEqual(args.reaction_miss_zero_threshold, 0.7)
        self.assertEqual(args.theta_bins, 8)
        self.assertEqual(args.phi_bins, 11)
        self.assertEqual(args.speed_bins, 5)
        self.assertEqual(args.x_rec_bins, 5)
        self.assertEqual(args.y_rec_bins, 5)
        self.assertIsNone(args.recovery_x_margin)
        self.assertEqual(train_selfplay.build_sim_config(args).action.recovery_x_margin, 0.25)
        self.assertEqual(train_selfplay.build_sim_config(args).action.recovery_net_margin, 0.3)
        self.assertEqual(train_selfplay.build_sim_config(args).action.recovery_back_margin, 0.5)

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


if __name__ == "__main__":
    unittest.main()
