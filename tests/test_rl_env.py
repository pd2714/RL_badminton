from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from badminton.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton.config import ActionConfig, SimulationConfig
from badminton.dynamics import feasible_intercept_indices, landing_position, prepare_shot, simulate_trajectory
from badminton.obs import ObservationEncoder
from badminton.rl_env import BadmintonRLEnv, RLEnvConfig
from badminton.state import ShotAction, StageState
from badminton.utils import recovery_bounds, side_y_bounds, x_bounds


class RLEnvTests(unittest.TestCase):
    def test_observation_size_matches_encoder(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(config=config)
        obs, _ = env.reset(seed=3)
        self.assertEqual(obs.shape[0], ObservationEncoder(config).size)
        self.assertEqual(env.action_mapper.policy_type, "velocity_oriented")
        self.assertEqual(env.action_space.n, env.action_mapper.action_count)

    def test_observation_includes_reaction_miss_risk_features(self) -> None:
        config = SimulationConfig()
        encoder = ObservationEncoder(config)
        feature_names = encoder.feature_names()
        mean_miss_index = feature_names.index("mean_feasible_reaction_miss_probability")
        safe_fraction_index = feature_names.index("zero_miss_feasible_fraction")
        state = StageState(
            x_left=0.0,
            y_left=-1.2,
            x_right=0.0,
            y_right=1.2,
            current_hitter="left",
            x0=0.0,
            y0=-1.2,
            z0=1.7,
            stage_index=1,
        )
        fast_action = ShotAction(v_x=0.0, v_y=6.5, v_z=2.0, x_rec=0.0, y_rec=-1.0)
        slow_action = ShotAction(v_x=0.0, v_y=1.5, v_z=4.0, x_rec=0.0, y_rec=-2.5)
        fast_feasible = feasible_intercept_indices(state, fast_action, config)
        slow_feasible = feasible_intercept_indices(state, slow_action, config)

        self.assertTrue(fast_feasible)
        self.assertTrue(slow_feasible)

        fast_obs = encoder.encode(
            state=state,
            agent_side="right",
            role="receiver",
            server_side="left",
            pending_action=fast_action,
            feasible_indices=fast_feasible,
        )
        slow_obs = encoder.encode(
            state=state,
            agent_side="right",
            role="receiver",
            server_side="left",
            pending_action=slow_action,
            feasible_indices=slow_feasible,
        )

        self.assertGreater(float(fast_obs[mean_miss_index]), float(slow_obs[mean_miss_index]))
        self.assertLess(float(fast_obs[safe_fraction_index]), float(slow_obs[safe_fraction_index]))

    def test_receiver_observation_reuses_prepared_drag_simulation(self) -> None:
        config = SimulationConfig(action=ActionConfig(trajectory_mode="drag_square"))
        encoder = ObservationEncoder(config)
        state = StageState(
            x_left=-0.5,
            y_left=-2.5,
            x_right=0.5,
            y_right=2.5,
            current_hitter="left",
            x0=-0.5,
            y0=-2.5,
            z0=1.7,
            stage_index=1,
        )
        action = ShotAction(v_x=0.9, v_y=5.8, v_z=5.0, x_rec=0.0, y_rec=-1.8)
        prepared = prepare_shot(state, action, config)
        kwargs = {
            "state": state,
            "agent_side": "right",
            "role": "receiver",
            "server_side": "left",
            "pending_action": action,
            "feasible_indices": list(prepared.feasible_indices),
        }
        direct = encoder.encode(**kwargs)

        with patch("badminton.dynamics.simulate_trajectory", wraps=simulate_trajectory) as mocked_simulate:
            reused = encoder.encode(**kwargs, prepared_shot=prepared)

        self.assertEqual(mocked_simulate.call_count, 0)
        self.assertTrue(np.allclose(reused, direct))

    def test_continuous_hitter_decode_uses_court_recovery_limits(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(config=config, rl_config=RLEnvConfig(policy_type="continuous_action"))
        env.reset(seed=1, options={"server": "left"})

        decoded = env.action_mapper.decode_hitter(
            np.asarray([0.0, 0.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float32),
            env.base_env.state,
        )

        x_bounds, y_bounds = recovery_bounds(env.base_env.state.current_hitter, config)
        self.assertGreaterEqual(decoded.shot_action.x_rec, x_bounds[0])
        self.assertLessEqual(decoded.shot_action.x_rec, x_bounds[1])
        self.assertGreaterEqual(decoded.shot_action.y_rec, y_bounds[0])
        self.assertLessEqual(decoded.shot_action.y_rec, y_bounds[1])
        self.assertAlmostEqual(decoded.shot_action.x_rec, x_bounds[1])
        self.assertAlmostEqual(decoded.shot_action.y_rec, y_bounds[0])

    def test_continuous_recovery_is_conditioned_on_decoded_shot_landing(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(config=config, rl_config=RLEnvConfig(policy_type="continuous_action"))
        env.reset(seed=1, options={"server": "left"})
        state = env.base_env.state

        short_action = np.asarray([0.0, 0.0, -0.8, 0.0, 0.0, 0.0], dtype=np.float32)
        deep_action = np.asarray([0.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)

        short_decoded = env.action_mapper.decode_hitter(short_action, state).shot_action
        deep_decoded = env.action_mapper.decode_hitter(deep_action, state).shot_action
        short_landing = landing_position(state, short_decoded, config)
        deep_landing = landing_position(state, deep_decoded, config)

        self.assertNotAlmostEqual(short_landing[1], deep_landing[1])
        self.assertNotAlmostEqual(short_decoded.y_rec, deep_decoded.y_rec)

    def test_mixed_discrete_continous_decodes_discrete_shot_and_full_court_recovery(self) -> None:
        config = SimulationConfig()
        env = BadmintonRLEnv(config=config, rl_config=RLEnvConfig(policy_type="mixed_discrete_continous"))
        env.reset(seed=1, options={"server": "left"})

        decoded = env.action_mapper.decode_hitter(
            np.asarray([-1.0, 0.0, 1.0, 1.0, -1.0, 0.0], dtype=np.float32),
            env.base_env.state,
        )

        full_x_bounds = x_bounds(config, margin=0.0)
        full_y_bounds = side_y_bounds(env.base_env.state.current_hitter, config, net_margin=0.0, back_margin=0.0)
        self.assertEqual(decoded.phi_index, 0)
        self.assertEqual(decoded.speed_index, env.action_mapper.discrete_config.speed_bins - 1)
        self.assertAlmostEqual(decoded.shot_action.x_rec, full_x_bounds[1])
        self.assertAlmostEqual(decoded.shot_action.y_rec, full_y_bounds[0])

    def test_reset_with_opponent_server_starts_receiver_turn(self) -> None:
        env = BadmintonRLEnv(
            rl_config=RLEnvConfig(
                train_side="left",
                initial_server="right",
                policy_type="velocity_oriented",
            )
        )
        _, info = env.reset(seed=2)
        self.assertEqual(info["role"], "receiver")
        self.assertIsNotNone(env.pending_applied_action)

    def test_large_receiver_action_is_wrapped_into_legal_range(self) -> None:
        env = BadmintonRLEnv(
            rl_config=RLEnvConfig(
                train_side="left",
                initial_server="right",
                policy_type="velocity_oriented",
            )
        )
        _, reset_info = env.reset(seed=5)
        feasible_index = reset_info["feasible_indices"][-1]
        action = env.action_space.n - 1
        action -= (action % env.action_mapper.receiver_action_count - feasible_index) % env.action_mapper.receiver_action_count
        _, reward, terminated, truncated, info = env.step(action)
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

    def test_random_service_x_samples_both_service_court_halves(self) -> None:
        env = BadmintonRLEnv(rl_config=RLEnvConfig(initial_server="left", random_service_x=True))
        observed_x_sides = {env.reset(seed=seed)[1]["service_x_side"] for seed in range(12)}
        self.assertEqual(observed_x_sides, {"left", "right"})

        env.reset(seed=11)
        state = env.base_env.state
        self.assertGreater(state.x_left, 0.0)
        self.assertAlmostEqual(state.x_left, env.config.court.service_x_offset_from_center_line)
        self.assertLess(state.x_right, 0.0)

        env.reset(seed=12)
        state = env.base_env.state
        self.assertLess(state.x_left, 0.0)
        self.assertAlmostEqual(state.x_left, -env.config.court.service_x_offset_from_center_line)
        self.assertGreater(state.x_right, 0.0)

    def test_observation_is_agent_canonical_for_mirrored_physical_state(self) -> None:
        config = SimulationConfig()
        encoder = ObservationEncoder(config)
        state = StageState(
            x_left=-0.4,
            y_left=-3.2,
            x_right=0.7,
            y_right=2.4,
            current_hitter="left",
            x0=-0.4,
            y0=-3.2,
            z0=1.2,
            stage_index=2,
        )
        mirrored_state = StageState(
            x_left=state.x_right,
            y_left=-state.y_right,
            x_right=state.x_left,
            y_right=-state.y_left,
            current_hitter="right",
            x0=state.x0,
            y0=-state.y0,
            z0=state.z0,
            stage_index=state.stage_index,
        )
        pending_action = ShotAction(v_x=1.0, v_y=8.0, v_z=2.0, x_rec=0.2, y_rec=-3.0)
        mirrored_pending = ShotAction(v_x=1.0, v_y=-8.0, v_z=2.0, x_rec=0.2, y_rec=3.0)

        left_obs = encoder.encode(
            state=state,
            agent_side="left",
            role="receiver",
            server_side="left",
            score_left=2,
            score_right=1,
            pending_action=pending_action,
            feasible_indices=[1, 4, 8],
        )
        right_obs = encoder.encode(
            state=mirrored_state,
            agent_side="right",
            role="receiver",
            server_side="right",
            score_left=1,
            score_right=2,
            pending_action=mirrored_pending,
            feasible_indices=[1, 4, 8],
        )

        self.assertTrue(np.allclose(left_obs, right_obs))

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
            DiscreteActionConfig(phi_bins=5, theta_bins=5, speed_bins=4, x_rec_bins=5, y_rec_bins=5),
        )
        env = BadmintonRLEnv(config=config)
        env.reset(seed=1)
        decode = mapper.decode_hitter(17, env.base_env.state)
        encoded = mapper.encode_hitter(decode.shot_action, env.base_env.state)
        self.assertEqual(encoded, decode.flat_index)

    def test_hitter_forfeit_record_honors_requested_winner(self) -> None:
        env = BadmintonRLEnv(rl_config=RLEnvConfig(train_side="left"))
        env.reset(seed=1, options={"server": "left"})

        record = env._hitter_forfeit_record(env.base_env.state, winner="right", reason="test_forfeit")

        self.assertEqual(record.next_state.winner, "right")
        self.assertEqual(record.reward_left, -1.0)
        self.assertEqual(record.reward_right, 1.0)


if __name__ == "__main__":
    unittest.main()
