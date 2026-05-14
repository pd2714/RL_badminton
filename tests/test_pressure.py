from __future__ import annotations

import unittest

import numpy as np

from badminton1d.config import PlayerConfig, SimulationConfig
from badminton1d.pressure import ShotPressureWeights, shot_pressure_from_candidates


class ShotPressureTests(unittest.TestCase):
    def test_required_speed_component_increases_with_harder_chase(self) -> None:
        config = SimulationConfig(player=PlayerConfig(v_max=4.0, r_reach=0.5))
        weights = ShotPressureWeights(required_speed=1.0, intercept_scarcity=0.0, low_contact=0.0)
        easy = shot_pressure_from_candidates(
            receiver_side="right",
            receiver_start=(0.0, 3.0),
            receiver_velocity=(0.0, 0.0),
            receiver_reaction_time=0.0,
            candidate_times=np.asarray([1.0]),
            candidate_xs=np.asarray([0.0]),
            candidate_ys=np.asarray([3.5]),
            candidate_zs=np.asarray([1.4]),
            feasible_indices=[0],
            config=config,
            weights=weights,
        )
        hard = shot_pressure_from_candidates(
            receiver_side="right",
            receiver_start=(0.0, 3.0),
            receiver_velocity=(0.0, 0.0),
            receiver_reaction_time=0.0,
            candidate_times=np.asarray([1.0]),
            candidate_xs=np.asarray([0.0]),
            candidate_ys=np.asarray([6.5]),
            candidate_zs=np.asarray([1.4]),
            feasible_indices=[0],
            config=config,
            weights=weights,
        )

        self.assertGreater(hard.required_speed_score, easy.required_speed_score)
        self.assertGreater(hard.pressure, easy.pressure)

    def test_intercept_scarcity_component_increases_when_options_shrink(self) -> None:
        config = SimulationConfig()
        weights = ShotPressureWeights(required_speed=0.0, intercept_scarcity=1.0, low_contact=0.0)
        common_kwargs = {
            "receiver_side": "right",
            "receiver_start": (0.0, 3.0),
            "receiver_velocity": (0.0, 0.0),
            "receiver_reaction_time": 0.0,
            "candidate_times": np.asarray([0.5, 0.7, 0.9, 1.1]),
            "candidate_xs": np.asarray([0.0, 0.0, 0.0, 0.0]),
            "candidate_ys": np.asarray([3.0, 3.3, 3.6, 3.9]),
            "candidate_zs": np.asarray([1.4, 1.5, 1.6, 1.7]),
            "config": config,
            "weights": weights,
        }

        many = shot_pressure_from_candidates(feasible_indices=[0, 1, 2, 3], **common_kwargs)
        few = shot_pressure_from_candidates(feasible_indices=[3], **common_kwargs)

        self.assertLess(many.intercept_scarcity_score, few.intercept_scarcity_score)
        self.assertLess(many.pressure, few.pressure)

    def test_low_contact_component_uses_best_feasible_height(self) -> None:
        config = SimulationConfig()
        weights = ShotPressureWeights(required_speed=0.0, intercept_scarcity=0.0, low_contact=1.0)
        low = shot_pressure_from_candidates(
            receiver_side="right",
            receiver_start=(0.0, 3.0),
            receiver_velocity=(0.0, 0.0),
            receiver_reaction_time=0.0,
            candidate_times=np.asarray([0.5, 0.7]),
            candidate_xs=np.asarray([0.0, 0.0]),
            candidate_ys=np.asarray([3.0, 3.2]),
            candidate_zs=np.asarray([0.4, 0.6]),
            feasible_indices=[0, 1],
            config=config,
            weights=weights,
        )
        high = shot_pressure_from_candidates(
            receiver_side="right",
            receiver_start=(0.0, 3.0),
            receiver_velocity=(0.0, 0.0),
            receiver_reaction_time=0.0,
            candidate_times=np.asarray([0.5, 0.7]),
            candidate_xs=np.asarray([0.0, 0.0]),
            candidate_ys=np.asarray([3.0, 3.2]),
            candidate_zs=np.asarray([0.4, 2.4]),
            feasible_indices=[0, 1],
            config=config,
            weights=weights,
        )

        self.assertGreater(low.low_contact_score, high.low_contact_score)
        self.assertGreater(low.pressure, high.pressure)


if __name__ == "__main__":
    unittest.main()
