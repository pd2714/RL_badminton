from __future__ import annotations

import unittest

import numpy as np

from badminton1d.config import PlayerConfig, SimulationConfig
from badminton1d.dynamics import candidate_intercept_points, feasible_intercept_indices
from badminton1d.state import ShotAction, StageState
from badminton1d.utils import move_toward


class FeasibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()
        self.state = StageState(
            x_left=-0.6,
            y_left=-2.5,
            x_right=0.9,
            y_right=2.5,
            current_hitter="left",
            x0=-0.6,
            y0=-2.5,
            z0=1.7,
            stage_index=1,
        )

    def test_feasible_intercepts_are_found_and_respect_receiver_side(self) -> None:
        action = ShotAction(v_x=1.0, v_y=5.4, v_z=5.0, x_rec=0.0, y_rec=-2.0)
        feasible = feasible_intercept_indices(self.state, action, self.config)
        self.assertTrue(feasible)

        times, xs, ys, zs = candidate_intercept_points(self.state, action, self.config)
        for index in feasible:
            self.assertGreater(ys[index], 0.0)
            self.assertGreaterEqual(zs[index], self.config.player.z_min)
            self.assertLessEqual(zs[index], self.config.player.z_max)
            reach = self.config.player.v_max * float(times[index]) + self.config.player.r_reach
            distance = np.hypot(self.state.x_right - float(xs[index]), self.state.y_right - float(ys[index]))
            self.assertLessEqual(distance, reach + 1e-9)

    def test_larger_racket_length_expands_feasible_intercepts(self) -> None:
        action = ShotAction(v_x=0.0, v_y=5.4, v_z=1.0, x_rec=0.0, y_rec=-2.0)
        short_reach = SimulationConfig(player=PlayerConfig(r_reach=0.2))
        long_reach = SimulationConfig(player=PlayerConfig(r_reach=1.5))

        short_feasible = feasible_intercept_indices(self.state, action, short_reach)
        long_feasible = feasible_intercept_indices(self.state, action, long_reach)

        self.assertGreater(len(long_feasible), len(short_feasible))

    def test_move_toward_stops_at_target_or_speed_limit_in_2d(self) -> None:
        self.assertEqual(move_toward((0.0, -2.5), (0.5, -2.0), 2.0), (0.5, -2.0))
        moved = move_toward((0.0, -2.5), (3.0, -2.5), 1.0)
        self.assertEqual(moved, (1.0, -2.5))


if __name__ == "__main__":
    unittest.main()
