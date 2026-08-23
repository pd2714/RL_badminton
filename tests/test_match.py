from __future__ import annotations

import unittest

from badminton.agents import SafeHitter, StageAgent
from badminton.config import SimulationConfig
from badminton.dynamics import feasible_intercept_indices
from badminton.env import Badminton1DEnv, default_initial_state
from badminton.match import MatchConfig, reset_for_serve, run_match, service_receive_position, with_service_court_x_side
from badminton.playback import build_match_trace
from badminton.state import ShotAction
from badminton.trajectory import ballistic_landing_time


class AlwaysMissReceiver:
    def choose_intercept_index(self, state, action, feasible_indices, config):
        return None


class MatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()
        self.left_agent = StageAgent(
            name="LeftSafe",
            hitter_policy=SafeHitter(),
            receiver_policy=AlwaysMissReceiver(),
        )
        self.right_agent = StageAgent(
            name="RightSafe",
            hitter_policy=SafeHitter(),
            receiver_policy=AlwaysMissReceiver(),
        )

    def _shot_to_landing_target(self, state, landing_x: float, landing_y: float) -> ShotAction:
        v_z = 5.0
        flight_time = ballistic_landing_time(state.z0, v_z, self.config.action.gravity)
        return ShotAction(
            v_x=(landing_x - state.x0) / flight_time,
            v_y=(landing_y - state.y0) / flight_time,
            v_z=v_z,
            x_rec=state.x0,
            y_rec=state.y0,
        )

    def test_reset_for_serve_uses_default_service_position_and_serve_height(self) -> None:
        state = reset_for_serve("left", self.config)
        expected_receiver_x, expected_receiver_y = service_receive_position("right", self.config)

        self.assertAlmostEqual(state.x_left, -self.config.court.service_x_offset_from_center_line)
        self.assertAlmostEqual(state.y_left, -self.config.court.default_player_start_distance_from_net)
        self.assertAlmostEqual(state.x_right, expected_receiver_x)
        self.assertAlmostEqual(state.y_right, expected_receiver_y)
        self.assertEqual(state.current_hitter, "left")
        self.assertAlmostEqual(state.x0, state.x_left)
        self.assertAlmostEqual(state.y0, state.y_left)
        self.assertAlmostEqual(state.z0, 1.15)
        self.assertFalse(state.rally_done)

    def test_reset_for_serve_moves_receiver_x_half_meter_toward_center(self) -> None:
        left_server_state = reset_for_serve("left", self.config)
        right_server_state = reset_for_serve("right", self.config)
        offset = self.config.court.service_receive_x_offset_toward_center

        self.assertAlmostEqual(left_server_state.x_right, self.config.court.half_width / 2.0 - offset)
        self.assertAlmostEqual(right_server_state.x_left, -self.config.court.half_width / 2.0 + offset)

    def test_default_initial_state_uses_default_start_positions(self) -> None:
        state = default_initial_state(self.config)

        self.assertAlmostEqual(state.x_left, -self.config.court.half_width / 2.0)
        self.assertAlmostEqual(state.y_left, -self.config.court.default_player_start_distance_from_net)
        self.assertAlmostEqual(state.x_right, self.config.court.half_width / 2.0)
        self.assertAlmostEqual(state.y_right, self.config.court.default_player_start_distance_from_net)
        self.assertAlmostEqual(state.x0, state.x_left)
        self.assertAlmostEqual(state.y0, state.y_left)

    def test_invalid_serve_target_gives_point_to_receiver(self) -> None:
        env = Badminton1DEnv(config=self.config)
        match_config = MatchConfig(left_service_x=-self.config.court.half_width / 2.0, left_service_y=-3.5)
        env.reset(reset_for_serve("left", self.config, match_config))

        action = self._shot_to_landing_target(env.state, landing_x=-0.5, landing_y=3.0)
        record = env.step(action, intercept_index=None)

        self.assertTrue(record.next_state.rally_done)
        self.assertEqual(record.next_state.winner, "right")
        self.assertEqual(record.terminal_reason, "invalid_serve_target")

    def test_cross_court_serve_into_right_service_box_stays_alive(self) -> None:
        env = Badminton1DEnv(config=self.config)
        env.reset(reset_for_serve("left", self.config))

        action = self._shot_to_landing_target(env.state, landing_x=0.8, landing_y=3.0)
        feasible = feasible_intercept_indices(env.state, action, self.config)
        self.assertTrue(feasible)
        record = env.step(action, intercept_index=feasible[0])

        self.assertFalse(record.next_state.rally_done)
        self.assertIsNone(record.terminal_reason)

    def test_randomized_service_x_uses_actual_receiver_service_box(self) -> None:
        env = Badminton1DEnv(config=self.config)
        state = with_service_court_x_side(
            reset_for_serve("left", self.config),
            self.config,
            server_x_side="right",
        )
        env.reset(state)

        self.assertAlmostEqual(env.state.x_left, self.config.court.service_x_offset_from_center_line)
        self.assertAlmostEqual(env.state.x0, self.config.court.service_x_offset_from_center_line)

        action = self._shot_to_landing_target(env.state, landing_x=-0.8, landing_y=3.0)
        feasible = feasible_intercept_indices(env.state, action, self.config)
        self.assertTrue(feasible)
        record = env.step(action, intercept_index=feasible[0])

        self.assertFalse(record.next_state.rally_done)
        self.assertIsNone(record.terminal_reason)

    def test_run_match_keeps_next_server_as_rally_winner(self) -> None:
        match_result = run_match(
            self.left_agent,
            self.right_agent,
            self.config,
            match_config=MatchConfig(target_score=3, max_stages_per_rally=5),
            initial_server="right",
        )

        self.assertEqual(match_result.winner, "right")
        self.assertEqual(match_result.final_score.left, 0)
        self.assertEqual(match_result.final_score.right, 3)
        self.assertEqual([rally.server for rally in match_result.rallies], ["right", "right", "right"])
        self.assertEqual([rally.winner for rally in match_result.rallies], ["right", "right", "right"])

    def test_build_match_trace_carries_score_progression_and_pause(self) -> None:
        match_result = run_match(
            self.left_agent,
            self.right_agent,
            self.config,
            match_config=MatchConfig(target_score=2, max_stages_per_rally=5),
            initial_server="left",
        )

        trace = build_match_trace(match_result, self.config, rally_pause=0.5)

        self.assertEqual(len(trace.rallies), 2)
        self.assertEqual(trace.rallies[0].score_before_left, 0)
        self.assertEqual(trace.rallies[0].score_before_right, 0)
        self.assertEqual(trace.rallies[0].score_after_left, 1)
        self.assertEqual(trace.rallies[0].score_after_right, 0)
        self.assertEqual(trace.rallies[-1].match_winner, "left")
        expected_time = sum(rally.total_playback_time + rally.pause_duration for rally in trace.rallies)
        self.assertAlmostEqual(trace.total_playback_time, expected_time)


if __name__ == "__main__":
    unittest.main()
