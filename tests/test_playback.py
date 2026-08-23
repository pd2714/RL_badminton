from __future__ import annotations

import unittest

from badminton.config import PlayerConfig, SimulationConfig
from badminton.dynamics import effective_flight_time, feasible_intercept_indices, landing_position, step_stage
from badminton.movement import advance_player_toward, intercept_body_target_after_reaction
from badminton.playback import build_rally_trace, interpolate_stage
from badminton.state import ShotAction, StageState


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
        self.assertAlmostEqual(snapshot.right_player_position[0], record.next_state.x_right)
        self.assertAlmostEqual(snapshot.right_player_position[1], record.next_state.y_right)
        self.assertAlmostEqual(snapshot.left_player_position[0], record.next_state.x_left)
        self.assertAlmostEqual(snapshot.left_player_position[1], record.next_state.y_left)
        self.assertAlmostEqual(snapshot.shuttle_position[0], record.next_state.x0, places=4)
        self.assertAlmostEqual(snapshot.shuttle_position[1], record.next_state.y0, places=4)
        self.assertAlmostEqual(snapshot.shuttle_position[2], record.next_state.z0, places=4)

    def test_receiver_waits_for_reaction_time_before_playback_movement(self) -> None:
        state = StageState(
            x_left=-0.5,
            y_left=-2.5,
            x_right=0.5,
            y_right=2.5,
            current_hitter="left",
            x0=-0.5,
            y0=-2.5,
            z0=1.7,
            reaction_time_right=0.3,
            stage_index=1,
        )
        action = ShotAction(v_x=0.9, v_y=5.8, v_z=5.0, x_rec=0.0, y_rec=-1.8)
        chosen = feasible_intercept_indices(state, action, self.config)[0]
        record = step_stage(state, action, chosen, self.config)

        trace = build_rally_trace([record], self.config)
        stage = trace.stages[0]
        self.assertGreater(stage.playback_duration, 0.3)
        self.assertAlmostEqual(stage.receiver_reaction_time, 0.3)

        before_reaction = interpolate_stage(stage, 0.29)
        at_reaction = interpolate_stage(stage, 0.3)
        halfway_time = 0.3 + 0.5 * (stage.playback_duration - 0.3)
        halfway_after_reaction = interpolate_stage(stage, halfway_time)
        end = interpolate_stage(stage, stage.playback_duration)
        assert stage.intercept_point is not None
        receiver_target = intercept_body_target_after_reaction(
            stage.right_start,
            stage.right_start_velocity,
            (stage.intercept_point[0], stage.intercept_point[1]),
            "right",
            self.config,
            target_z=stage.intercept_point[2],
            reaction_time=stage.receiver_reaction_time,
        )
        expected_motion = advance_player_toward(
            stage.right_start,
            stage.right_start_velocity,
            receiver_target,
            halfway_time,
            self.config,
            reaction_time=stage.receiver_reaction_time,
            stop_when_early=True,
        )

        self.assertEqual(before_reaction.right_player_position, stage.right_start)
        self.assertEqual(at_reaction.right_player_position, stage.right_start)
        self.assertAlmostEqual(halfway_after_reaction.right_player_position[0], expected_motion.position[0])
        self.assertAlmostEqual(halfway_after_reaction.right_player_position[1], expected_motion.position[1])
        self.assertEqual(end.right_player_position, stage.right_end)

    def test_receiver_reaction_delay_can_be_disabled_for_smooth_video(self) -> None:
        state = StageState(
            x_left=-0.5,
            y_left=-2.5,
            x_right=0.5,
            y_right=2.5,
            current_hitter="left",
            x0=-0.5,
            y0=-2.5,
            z0=1.7,
            reaction_time_right=0.3,
            stage_index=1,
        )
        action = ShotAction(v_x=0.9, v_y=5.8, v_z=5.0, x_rec=0.0, y_rec=-1.8)
        chosen = feasible_intercept_indices(state, action, self.config)[0]
        record = step_stage(state, action, chosen, self.config)

        trace = build_rally_trace([record], self.config)
        stage = trace.stages[0]
        snapshot = interpolate_stage(stage, 0.15, apply_receiver_reaction_delay=False)
        assert stage.intercept_point is not None
        receiver_target = intercept_body_target_after_reaction(
            stage.right_start,
            stage.right_start_velocity,
            (stage.intercept_point[0], stage.intercept_point[1]),
            "right",
            self.config,
            target_z=stage.intercept_point[2],
            reaction_time=0.0,
        )
        expected_motion = advance_player_toward(
            stage.right_start,
            stage.right_start_velocity,
            receiver_target,
            0.15,
            self.config,
            reaction_time=0.0,
            stop_when_early=True,
        )

        self.assertEqual(snapshot.right_player_position, expected_motion.position)

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
        receiver_chase = advance_player_toward(
            stage.right_start,
            stage.right_start_velocity,
            (landing_x, landing_y),
            total_flight_time,
            self.slow_receiver_config,
            stop_when_early=False,
        )
        self.assertAlmostEqual(snapshot.right_player_position[0], receiver_chase.position[0])
        self.assertAlmostEqual(snapshot.right_player_position[1], receiver_chase.position[1])


if __name__ == "__main__":
    unittest.main()
