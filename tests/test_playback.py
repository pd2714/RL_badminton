from __future__ import annotations

import unittest

from badminton1d.config import PlayerConfig, SimulationConfig
from badminton1d.dynamics import effective_flight_time, feasible_intercept_indices, landing_position, step_stage
from badminton1d.playback import build_rally_trace, interpolate_stage
from badminton1d.state import ShotAction, StageState
from badminton1d.utils import move_toward


class PlaybackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()
        self.slow_receiver_config = SimulationConfig(player=PlayerConfig(v_max=1.0))

    def test_interpolated_stage_reaches_intercept_and_partial_recovery(self) -> None:
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
        chosen = feasible_intercept_indices(state, action, self.config)[0]
        record = step_stage(state, action, chosen, self.config)

        trace = build_rally_trace([record], self.config)
        stage = trace.stages[0]
        snapshot = interpolate_stage(stage, stage.playback_duration)

        self.assertTrue(stage.intercepted)
        self.assertAlmostEqual(stage.playback_duration, float(record.chosen_time))
        self.assertAlmostEqual(snapshot.right_player_position[0], record.next_state.x0)
        self.assertAlmostEqual(snapshot.right_player_position[1], record.next_state.y0)
        self.assertAlmostEqual(snapshot.left_player_position[0], record.next_state.x_left)
        self.assertAlmostEqual(snapshot.left_player_position[1], record.next_state.y_left)
        self.assertAlmostEqual(snapshot.shuttle_position[0], record.next_state.x0, places=4)
        self.assertAlmostEqual(snapshot.shuttle_position[1], record.next_state.y0, places=4)
        self.assertAlmostEqual(snapshot.shuttle_position[2], record.next_state.z0, places=4)

    def test_terminal_stage_runs_until_landing_and_shows_receiver_chase(self) -> None:
        state = StageState(
            x_left=-1.5,
            y_left=-2.5,
            x_right=2.4,
            y_right=5.8,
            current_hitter="left",
            x0=-1.5,
            y0=-2.5,
            z0=1.7,
            stage_index=1,
        )
        action = ShotAction(v_x=1.0, v_y=5.4, v_z=5.0, x_rec=0.0, y_rec=-2.0)
        record = step_stage(state, action, intercept_index=None, config=self.slow_receiver_config)

        trace = build_rally_trace([record], self.slow_receiver_config)
        stage = trace.stages[0]
        snapshot = interpolate_stage(stage, stage.playback_duration)
        total_flight_time = effective_flight_time(state, action, self.slow_receiver_config)
        landing_x, landing_y = landing_position(state, action, self.slow_receiver_config)

        self.assertFalse(stage.intercepted)
        self.assertTrue(stage.terminal)
        self.assertAlmostEqual(stage.playback_duration, total_flight_time)
        self.assertAlmostEqual(snapshot.shuttle_position[0], landing_x, places=4)
        self.assertAlmostEqual(snapshot.shuttle_position[1], landing_y, places=4)
        self.assertAlmostEqual(snapshot.shuttle_position[2], 0.0, places=4)
        receiver_chase = move_toward(
            (state.x_right, state.y_right),
            (landing_x, landing_y),
            self.slow_receiver_config.player.v_max * total_flight_time,
        )
        assert isinstance(receiver_chase, tuple)
        self.assertAlmostEqual(snapshot.right_player_position[0], receiver_chase[0])
        self.assertAlmostEqual(snapshot.right_player_position[1], receiver_chase[1])


if __name__ == "__main__":
    unittest.main()
