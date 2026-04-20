from __future__ import annotations

import unittest

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import candidate_intercept_points, feasible_intercept_indices, valid_hitter_action
from badminton1d.state import ShotAction, StageState
from badminton1d.trajectory import (
    ballistic_landing_time,
    ballistic_net_crossing,
    ballistic_position,
    simulate_drag_trajectory,
)


class TrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()
        self.state = StageState(
            x_left=0.0,
            y_left=-2.5,
            x_right=0.0,
            y_right=2.5,
            current_hitter="left",
            x0=0.0,
            y0=-2.5,
            z0=1.7,
            stage_index=1,
        )

    def test_ballistic_landing_time_matches_ground_contact(self) -> None:
        t_land = ballistic_landing_time(z0=self.state.z0, v_z=4.2, g=self.config.action.gravity)
        _, _, z_land = ballistic_position(self.state.x0, self.state.y0, self.state.z0, 0.8, 5.5, 4.2, t_land, self.config.action.gravity)
        self.assertAlmostEqual(z_land, 0.0, places=6)

    def test_ballistic_net_crossing_validity(self) -> None:
        action = ShotAction(v_x=0.8, v_y=5.4, v_z=5.0, x_rec=0.0, y_rec=-2.0)
        crossing = ballistic_net_crossing(
            self.state.x0,
            self.state.y0,
            self.state.z0,
            action.v_x,
            action.v_y,
            action.v_z,
            self.config.court.net_y,
            g=self.config.action.gravity,
        )
        self.assertIsNotNone(crossing)
        assert crossing is not None
        self.assertGreater(crossing.t, 0.0)
        self.assertGreaterEqual(crossing.z, self.config.court.net_height + self.config.action.net_clearance_margin)

    def test_valid_hitter_action_requires_opponent_side_landing_and_bounds(self) -> None:
        invalid = ShotAction(v_x=0.0, v_y=1.0, v_z=0.8, x_rec=0.0, y_rec=-2.0)
        valid = ShotAction(v_x=0.8, v_y=5.4, v_z=5.0, x_rec=0.0, y_rec=-2.0)
        self.assertFalse(valid_hitter_action(self.state, invalid, self.config))
        self.assertTrue(valid_hitter_action(self.state, valid, self.config))

    def test_drag_trajectory_detects_landing(self) -> None:
        result = simulate_drag_trajectory(
            self.state.x0,
            self.state.y0,
            self.state.z0,
            0.5,
            8.5,
            4.5,
            g=self.config.action.gravity,
            c=0.6,
            dt=0.01,
            net_y=self.config.court.net_y,
        )
        self.assertGreater(result.landing_time, 0.0)
        self.assertGreater(result.landing_y, 0.0)
        self.assertAlmostEqual(result.samples[-1].z, 0.0, places=6)

    def test_anisotropic_drag_changes_vertical_more_than_horizontal(self) -> None:
        equal_drag = simulate_drag_trajectory(
            self.state.x0,
            self.state.y0,
            self.state.z0,
            0.5,
            5.5,
            4.5,
            g=self.config.action.gravity,
            kh=0.2,
            kv=0.2,
            dt=0.01,
            net_y=self.config.court.net_y,
        )
        stronger_vertical_drag = simulate_drag_trajectory(
            self.state.x0,
            self.state.y0,
            self.state.z0,
            0.5,
            5.5,
            4.5,
            g=self.config.action.gravity,
            kh=0.2,
            kv=0.45,
            dt=0.01,
            net_y=self.config.court.net_y,
        )
        equal_drag_peak = max(point.z for point in equal_drag.samples)
        stronger_vertical_drag_peak = max(point.z for point in stronger_vertical_drag.samples)
        self.assertLess(stronger_vertical_drag_peak, equal_drag_peak)
        self.assertNotAlmostEqual(stronger_vertical_drag.landing_y, equal_drag.landing_y, places=3)

    def test_intercept_feasibility_uses_2d_ground_distance(self) -> None:
        action = ShotAction(v_x=1.2, v_y=5.6, v_z=5.0, x_rec=0.0, y_rec=-2.0)
        times, xs, ys, zs = candidate_intercept_points(self.state, action, self.config)
        feasible = feasible_intercept_indices(self.state, action, self.config)
        self.assertTrue(len(times) > 0)
        self.assertTrue(feasible)
        first = feasible[0]
        self.assertGreater(ys[first], 0.0)
        self.assertGreaterEqual(zs[first], self.config.player.z_min)
        self.assertLessEqual(zs[first], self.config.player.z_max)
        self.assertNotEqual(xs[first], self.state.x_right)


if __name__ == "__main__":
    unittest.main()
