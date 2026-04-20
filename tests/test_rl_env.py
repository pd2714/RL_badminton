from __future__ import annotations

import unittest

from badminton1d.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton1d.config import SimulationConfig
from badminton1d.obs import ObservationEncoder
from badminton1d.rl_env import BadmintonRLEnv, RLEnvConfig


class RLEnvTests(unittest.TestCase):
    def test_observation_size_matches_encoder(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(config=config)
        obs, _ = env.reset(seed=3)
        self.assertEqual(obs.shape[0], ObservationEncoder(config).size)

    def test_reset_with_opponent_server_starts_receiver_turn(self) -> None:
        env = BadmintonRLEnv(rl_config=RLEnvConfig(train_side="left", initial_server="right"))
        _, info = env.reset(seed=2)
        self.assertEqual(info["role"], "receiver")
        self.assertIsNotNone(env.pending_applied_action)

    def test_large_receiver_action_is_wrapped_into_legal_range(self) -> None:
        env = BadmintonRLEnv(rl_config=RLEnvConfig(train_side="left", initial_server="right"))
        env.reset(seed=5)
        _, reward, terminated, truncated, info = env.step(env.action_space.n - 1)
        self.assertFalse(terminated or truncated)
        self.assertIn("last_record", info)
        self.assertIn(info["last_record"].chosen_index, info["last_record"].feasible_indices)
        self.assertGreaterEqual(reward, -1.0)

    def test_mirrored_reset_can_switch_train_side(self) -> None:
        env = BadmintonRLEnv(rl_config=RLEnvConfig(train_side="left", mirror_train_side=True))
        observed_sides = {env.reset(seed=seed)[1]["train_side"] for seed in range(4)}
        self.assertTrue(observed_sides.issubset({"left", "right"}))
        self.assertGreaterEqual(len(observed_sides), 1)

    def test_reset_options_can_override_train_side(self) -> None:
        env = BadmintonRLEnv(rl_config=RLEnvConfig(train_side="left"))
        _, info = env.reset(seed=3, options={"train_side": "right"})
        self.assertEqual(info["train_side"], "right")
        self.assertEqual(env.train_side, "right")

    def test_mirror_match_fraction_one_always_flips_train_side(self) -> None:
        env = BadmintonRLEnv(rl_config=RLEnvConfig(train_side="left", mirror_match_fraction=1.0))
        _, info = env.reset(seed=7)
        self.assertEqual(info["train_side"], "right")

    def test_random_initial_server_samples_both_sides(self) -> None:
        env = BadmintonRLEnv(rl_config=RLEnvConfig(train_side="left", initial_server="random"))
        observed_servers = {env.reset(seed=seed)[1]["server"] for seed in range(12)}
        self.assertEqual(observed_servers, {"left", "right"})

    def test_reaction_time_is_applied_on_reset(self) -> None:
        env = BadmintonRLEnv(
            rl_config=RLEnvConfig(
                train_side="left",
                train_reaction_time=0.5,
                opponent_reaction_time=0.5,
            )
        )
        env.reset(seed=4)
        self.assertAlmostEqual(env.base_env.state.reaction_time_left, 0.5)
        self.assertAlmostEqual(env.base_env.state.reaction_time_right, 0.5)

    def test_action_mapper_round_trip_for_safe_bins(self) -> None:
        config = SimulationConfig()
        mapper = DiscreteActionMapper(
            config,
            DiscreteActionConfig(v_x_bins=5, v_y_bins=5, v_z_bins=4, x_rec_bins=5, y_rec_bins=5),
        )
        env = BadmintonRLEnv(config=config)
        env.reset(seed=1)
        decode = mapper.decode_hitter(17, env.base_env.state)
        encoded = mapper.encode_hitter(decode.shot_action, env.base_env.state)
        self.assertEqual(encoded, decode.flat_index)


if __name__ == "__main__":
    unittest.main()
