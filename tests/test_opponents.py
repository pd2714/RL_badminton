from __future__ import annotations

import unittest

from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.dynamics import landing_position
from badminton1d.opponents import make_opponent
from badminton1d.state import StageState
from badminton1d.utils import service_target_bounds_for_receiver


class OpponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig(
            court=CourtConfig(mode="2d"),
            player=PlayerConfig(v_max=2.6),
            action=ActionConfig(
                trajectory_mode="drag_square",
                intercept_count=50,
            ),
        )
        self.serve_state = StageState(
            x_left=-self.config.court.half_width / 2.0,
            y_left=-3.5,
            x_right=self.config.court.half_width / 2.0,
            y_right=3.5,
            current_hitter="left",
            x0=-self.config.court.half_width / 2.0,
            y0=-3.5,
            z0=1.15,
            stage_index=0,
        )

    def test_safe_heuristic_is_seeded_but_not_frozen(self) -> None:
        opponent = make_opponent("safe", seed=7)
        actions = [
            opponent.choose_hitter_action(self.serve_state, self.config, "left")
            for _ in range(8)
        ]

        serialized = [(round(a.v_x, 4), round(a.v_y, 4), round(a.v_z, 4)) for a in actions]
        self.assertGreater(len(set(serialized)), 1)

        opponent_again = make_opponent("safe", seed=7)
        actions_again = [
            opponent_again.choose_hitter_action(self.serve_state, self.config, "left")
            for _ in range(8)
        ]
        serialized_again = [(round(a.v_x, 4), round(a.v_y, 4), round(a.v_z, 4)) for a in actions_again]
        self.assertEqual(serialized, serialized_again)

    def test_safe_heuristic_serves_land_in_legal_box(self) -> None:
        left_server = make_opponent("safe", seed=7)
        right_server = make_opponent("safe", seed=11)
        right_serve_state = StageState(
            x_left=-self.config.court.half_width / 2.0,
            y_left=-3.5,
            x_right=self.config.court.half_width / 2.0,
            y_right=3.5,
            current_hitter="right",
            x0=self.config.court.half_width / 2.0,
            y0=3.5,
            z0=1.15,
            stage_index=0,
        )

        for _ in range(64):
            left_action = left_server.choose_hitter_action(self.serve_state, self.config, "left")
            left_landing = landing_position(self.serve_state, left_action, self.config)
            left_x_bounds, left_y_bounds = service_target_bounds_for_receiver("right", self.config)
            self.assertTrue(left_x_bounds[0] <= left_landing[0] <= left_x_bounds[1])
            self.assertTrue(left_y_bounds[0] <= left_landing[1] <= left_y_bounds[1])

            right_action = right_server.choose_hitter_action(right_serve_state, self.config, "right")
            right_landing = landing_position(right_serve_state, right_action, self.config)
            right_x_bounds, right_y_bounds = service_target_bounds_for_receiver("left", self.config)
            self.assertTrue(right_x_bounds[0] <= right_landing[0] <= right_x_bounds[1])
            self.assertTrue(right_y_bounds[0] <= right_landing[1] <= right_y_bounds[1])


if __name__ == "__main__":
    unittest.main()
