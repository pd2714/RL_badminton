from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
from stable_baselines3 import PPO

from badminton.action_space import DiscreteActionConfig
from badminton.config import CourtConfig, SimulationConfig
from badminton.policy import MaskedBadmintonPolicy
from badminton.rl_env import BadmintonRLEnv, RLEnvConfig
from badminton.shot_generators import (
    SHOT_NAME_ORDER,
    TacticAction2D,
    TacticLookup1D,
    TacticLookup2D,
    TacticRuntimeConfig,
)
from badminton.shot_generators.tactic_lookup_common import (
    ANGLE_BIN_COUNT_1D,
    LANDING_ZONE_COUNT_1D,
    POWER_BIN_NAMES_1D,
    angle_bin_centers_deg_1d,
    power_speed_targets_1d,
)
from badminton.state import StageState
from scripts.visualize_tactic_lookup_1d import build_plot as build_plot_1d
from scripts.visualize_tactic_lookup_2d import build_plot as build_plot_2d

matplotlib.use("Agg")


class TacticLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.lookup_root = Path(cls.temp_dir.name)
        cls.config_1d = SimulationConfig(court=CourtConfig(mode="1d"))
        cls.runtime_1d = TacticRuntimeConfig(regenerate_lookup_table=True, lookup_dir=cls.lookup_root)
        cls.lookup_1d = TacticLookup1D(cls.config_1d, cls.runtime_1d)
        cls.lookup_1d.ensure_loaded()

        cls.config_2d = SimulationConfig(court=CourtConfig(mode="2d"))
        cls.runtime_2d = TacticRuntimeConfig(regenerate_lookup_table=False, lookup_dir=cls.lookup_root)
        cls.lookup_2d = TacticLookup2D(cls.config_2d, cls.runtime_2d)
        cls._write_fake_2d_lookup(cls.lookup_2d)
        cls.lookup_2d.ensure_loaded()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    @classmethod
    def _write_fake_2d_lookup(cls, lookup: TacticLookup2D) -> None:
        lookup.lookup_path.parent.mkdir(parents=True, exist_ok=True)
        shape = lookup.table_shape
        velocities = np.zeros(shape + (3,), dtype=np.float32)
        velocities[..., 1] = 8.5
        velocities[..., 2] = 4.0
        valid = np.ones(shape, dtype=bool)
        fallback_used = np.zeros(shape, dtype=bool)
        landing_x = np.zeros(shape, dtype=np.float32)
        landing_y = np.zeros(shape, dtype=np.float32)
        net_crossing_height = np.full(shape, 2.2, dtype=np.float32)
        flight_time = np.full(shape, 0.85, dtype=np.float32)
        score = np.full(shape, 10.0, dtype=np.float32)
        shot_name_index = np.full(shape, SHOT_NAME_ORDER.index("generic"), dtype=np.int16)

        for row, landing_y_center in enumerate(lookup.landing_row_centers):
            for col, landing_x_center in enumerate(lookup.landing_col_centers):
                landing_x[:, :, :, row, col, :, :] = float(landing_x_center)
                landing_y[:, :, :, row, col, :, :] = float(landing_y_center)

        np.savez_compressed(
            lookup.lookup_path,
            velocities=velocities,
            valid=valid,
            fallback_used=fallback_used,
            landing_x=landing_x,
            landing_y=landing_y,
            net_crossing_height=net_crossing_height,
            flight_time=flight_time,
            score=score,
            shot_name_index=shot_name_index,
            contact_x_centers=lookup.contact_x_centers.astype(np.float32),
            contact_y_centers=lookup.contact_y_centers.astype(np.float32),
            contact_height_centers=lookup.contact_height_centers.astype(np.float32),
            landing_row_centers=lookup.landing_row_centers.astype(np.float32),
            landing_col_centers=lookup.landing_col_centers.astype(np.float32),
            zone_names=np.asarray(lookup.zone_names),
            angle_names=np.asarray(lookup.angle_names),
            power_names=np.asarray(lookup.power_names),
            shot_names=np.asarray(SHOT_NAME_ORDER),
            version=np.asarray([1], dtype=np.int16),
        )

    def test_lookup_1d_build_has_expected_shape_and_no_nans(self) -> None:
        self.assertEqual(self.lookup_1d.velocities.shape, self.lookup_1d.table_shape + (2,))
        self.assertEqual(len(self.lookup_1d.zone_names), LANDING_ZONE_COUNT_1D)
        self.assertEqual(len(self.lookup_1d.angle_names), ANGLE_BIN_COUNT_1D)
        self.assertEqual(tuple(self.lookup_1d.power_names), POWER_BIN_NAMES_1D)
        self.assertEqual(len(self.lookup_1d.power_names), 6)
        self.assertAlmostEqual(float(np.ptp(power_speed_targets_1d())), 18.0)
        self.assertAlmostEqual(
            float(self.lookup_1d.contact_y_centers[-1]),
            float(self.config_1d.court.net_y - 0.5),
        )
        angle_centers = angle_bin_centers_deg_1d(
            float(self.lookup_1d.contact_y_centers[-1]),
            float(self.lookup_1d.contact_height_centers[0]),
            self.config_1d,
            bins=ANGLE_BIN_COUNT_1D,
        )
        self.assertAlmostEqual(float(angle_centers[-1]), 80.0)
        self.assertFalse(np.isnan(self.lookup_1d.velocities).any())
        summary = self.lookup_1d.summary()
        self.assertAlmostEqual(summary["fallback_fraction"], 0.0)

        state = StageState(
            x_left=0.0,
            y_left=-2.5,
            x_right=0.0,
            y_right=2.5,
            current_hitter="left",
            x0=0.0,
            y0=-2.5,
            z0=1.5,
            stage_index=1,
        )
        entry = self.lookup_1d.lookup(state, self.lookup_1d.flat_to_action(7))
        self.assertTrue(np.isfinite(entry.velocity[0]))
        self.assertTrue(np.isfinite(entry.velocity[1]))
        self.assertTrue(np.isfinite(entry.flight_time))

    def test_lookup_2d_runtime_lookup_and_roundtrip(self) -> None:
        flat = self.lookup_2d.action_to_flat(TacticAction2D(landing_row=2, landing_col=1, angle_bin=3, power_bin=2))
        decoded = self.lookup_2d.flat_to_action(flat)
        self.assertEqual(decoded, TacticAction2D(landing_row=2, landing_col=1, angle_bin=3, power_bin=2))

        state = StageState(
            x_left=0.0,
            y_left=-2.5,
            x_right=0.0,
            y_right=2.5,
            current_hitter="left",
            x0=0.1,
            y0=-2.0,
            z0=1.6,
            stage_index=1,
        )
        entry = self.lookup_2d.lookup(state, decoded)
        self.assertEqual(len(entry.velocity), 3)
        self.assertTrue(entry.valid)
        self.assertFalse(entry.fallback_used)
        self.assertGreater(entry.net_crossing_height or 0.0, 0.0)

    def test_tactic_policy_1d_tiny_rollout_and_logprob(self) -> None:
        env = BadmintonRLEnv(
            config=self.config_1d,
            rl_config=RLEnvConfig(
                policy_type="tactic_oriented",
                tactic_runtime=TacticRuntimeConfig(
                    regenerate_lookup_table=False,
                    lookup_dir=self.lookup_root,
                ),
            ),
            discrete_action_config=DiscreteActionConfig(),
            seed=3,
        )
        model = PPO(
            MaskedBadmintonPolicy,
            env,
            policy_kwargs={
                "sim_config": self.config_1d,
                "discrete_action_config": DiscreteActionConfig(),
                "policy_type": "tactic_oriented",
                "tactic_runtime_config": TacticRuntimeConfig(
                    regenerate_lookup_table=False,
                    lookup_dir=self.lookup_root,
                ),
            },
            n_steps=8,
            batch_size=4,
            learning_rate=1e-4,
            ent_coef=0.0,
            verbose=0,
            seed=3,
        )
        obs, _ = env.reset(seed=3)
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        actions, _, log_prob = model.policy(obs_tensor)
        _, log_prob_eval, entropy = model.policy.evaluate_actions(obs_tensor, actions)

        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(log_prob_eval.shape, (1,))
        self.assertTrue(np.isfinite(entropy.detach().cpu().numpy()).all())
        if env.role == "hitter":
            decode = env.action_mapper.decode_hitter(int(actions.item()), env.base_env.state)
            self.assertIsNotNone(decode.tactic_action)

        for _ in range(4):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                self.assertIn("badminton_metrics", info)
                break

    def test_visualization_builders_smoke(self) -> None:
        args_1d = type(
            "Args1D",
            (),
            {
                "lookup_table_dir": self.lookup_root,
                "regenerate_lookup_table": False,
                "trajectory_mode": "drag_square",
                "save_path": None,
                "no_show": True,
            },
        )()
        fig_1d, _ = build_plot_1d(args_1d, self.config_1d)
        self.assertIsNotNone(fig_1d)
        import matplotlib.pyplot as plt

        plt.close(fig_1d)

        args_2d = type(
            "Args2D",
            (),
            {
                "lookup_table_dir": self.lookup_root,
                "regenerate_lookup_table": False,
                "trajectory_mode": "drag_square",
                "save_path": None,
                "no_show": True,
            },
        )()
        fig_2d, _ = build_plot_2d(args_2d, self.config_2d)
        self.assertIsNotNone(fig_2d)
        plt.close(fig_2d)


if __name__ == "__main__":
    unittest.main()
