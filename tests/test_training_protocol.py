from __future__ import annotations

import unittest
from unittest.mock import patch

from badminton1d.callbacks import SafeWinRateEvalCallback
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
        self.assertEqual(args.n_envs, 8)
        self.assertAlmostEqual(args.ent_coef, 0.02)
        self.assertAlmostEqual(args.ent_coef_final, 0.002)
        self.assertAlmostEqual(args.reaction_time, 0.3)
        self.assertAlmostEqual(args.loop_penalty, 0.03)
        self.assertEqual(args.max_rally_stages, 120)
        self.assertAlmostEqual(args.max_rally_penalty, 1.0)
        self.assertAlmostEqual(args.pressure_reward_weight, 0.05)
        self.assertAlmostEqual(args.player_speed, 2.6)
        self.assertEqual(args.trajectory_mode, "drag_square")
        self.assertEqual(args.intercept_count, 50)
        self.assertAlmostEqual(args.mirror_match_fraction, 0.25)
        self.assertAlmostEqual(rl_config.reset_sampling.opponent_serve_start_prob, 0.0)
        self.assertAlmostEqual(rl_config.reward.stage_penalty, 0.0)
        self.assertAlmostEqual(rl_config.reward.stall_penalty, 0.0)
        self.assertEqual(rl_config.reward.stall_penalty_start, 24)
        self.assertEqual(sim_config.action.trajectory_mode, "drag_square")
        self.assertEqual(sim_config.action.intercept_count, 50)
        self.assertAlmostEqual(sim_config.player.v_max, 2.6)

    def test_train_selfplay_defaults_match_latest_protocol(self) -> None:
        with patch("sys.argv", ["train_selfplay.py", "--base-checkpoint-path", "dummy.zip"]):
            args = train_selfplay.parse_args()

        self.assertEqual(args.initial_server, "random")
        self.assertEqual(args.n_envs, 8)
        self.assertEqual(args.save_interval, 2000)
        self.assertAlmostEqual(args.random_start_prob, 0.0)
        self.assertAlmostEqual(args.midrally_start_prob, 0.0)
        self.assertAlmostEqual(args.reaction_time, 0.3)
        self.assertAlmostEqual(args.loop_penalty, 0.03)
        self.assertEqual(args.max_rally_stages, 120)
        self.assertAlmostEqual(args.max_rally_penalty, 1.0)
        self.assertAlmostEqual(args.stage_penalty, 0.0)
        self.assertAlmostEqual(args.stall_penalty, 0.0)
        self.assertAlmostEqual(args.pressure_reward_weight, 0.05)
        self.assertAlmostEqual(args.player_speed, 2.6)
        self.assertEqual(args.trajectory_mode, "drag_square")
        self.assertEqual(args.intercept_count, 50)

    def test_stochastic_defaults_propagate_through_training_helpers(self) -> None:
        self.assertFalse(FrozenCheckpointOpponent.__dataclass_fields__["deterministic"].default)
        self.assertFalse(LiveModelOpponent.__dataclass_fields__["deterministic"].default)
        self.assertFalse(MixedCheckpointOpponent.__dataclass_fields__["deterministic"].default)
        self.assertFalse(SelfPlayProgressVideoCallback.__init__.__kwdefaults__["deterministic"])
        self.assertFalse(SafeWinRateEvalCallback.__init__.__kwdefaults__["deterministic"])


if __name__ == "__main__":
    unittest.main()
