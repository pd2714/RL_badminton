from __future__ import annotations

import unittest
from pathlib import Path

from badminton.config import SimulationConfig
from badminton.curricula import (
    DEFAULT_DEFENSIVE_CURRICULUM_NAME,
    DefensiveBackcourtCurriculumConfig,
    build_training_curriculum,
)
from badminton.match import MatchConfig
from badminton.reset_sampling import ResetSampler, ResetSamplingConfig
from badminton.utils import side_y_bounds


class DefensiveCurriculumTests(unittest.TestCase):
    def test_reset_sampler_starts_with_opponent_high_in_backcourt(self) -> None:
        sim_config = SimulationConfig()
        curriculum_config = DefensiveBackcourtCurriculumConfig()
        sampler = ResetSampler(
            sim_config=sim_config,
            match_config=MatchConfig(),
            sampling_config=ResetSamplingConfig(
                defensive_backcourt_curriculum=curriculum_config,
            ),
            seed=7,
        )

        state, randomized, server = sampler.sample_initial_state(
            default_server="left",
            train_side="left",
        )

        self.assertTrue(randomized)
        self.assertEqual(server, "right")
        self.assertEqual(state.current_hitter, "right")
        self.assertEqual(state.stage_index, curriculum_config.stage_index)
        self.assertAlmostEqual(state.x0, state.x_right)
        self.assertAlmostEqual(state.y0, state.y_right)

        right_y_low, right_y_high = side_y_bounds("right", sim_config)
        self.assertGreaterEqual(state.y_right, right_y_low)
        self.assertLessEqual(state.y_right, right_y_high)
        self.assertGreaterEqual(state.z0, curriculum_config.phases[0].hit_height_range[0])
        self.assertLessEqual(state.z0, curriculum_config.phases[0].hit_height_range[1])

    def test_build_training_curriculum_uses_override_checkpoint(self) -> None:
        override = Path("/tmp/custom_curriculum_opponent.zip")
        curriculum = build_training_curriculum(
            DEFAULT_DEFENSIVE_CURRICULUM_NAME,
            opponent_checkpoint_path=override,
        )

        self.assertEqual(curriculum.name, DEFAULT_DEFENSIVE_CURRICULUM_NAME)
        self.assertEqual(curriculum.opponent_checkpoint_path, override.resolve())
        self.assertFalse(curriculum.opponent_hitter_deterministic)
        self.assertTrue(curriculum.opponent_receiver_deterministic)
        self.assertEqual(curriculum.initial_server, "opponent")


if __name__ == "__main__":
    unittest.main()
