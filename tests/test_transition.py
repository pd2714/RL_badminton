from __future__ import annotations

import unittest
from unittest.mock import patch

from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.dynamics import (
    REACTION_MISS_FLIGHT_TIME_THRESHOLD,
    REACTION_MISS_SECONDARY_FLIGHT_TIME_THRESHOLD,
    REACTION_MISS_ZERO_FLIGHT_TIME_THRESHOLD,
    candidate_intercept_points,
    feasible_intercept_indices,
    reaction_miss_probability,
    step_stage,
)
from badminton1d.movement import advance_player_toward, intercept_body_target_after_reaction
from badminton1d.state import ShotAction, StageState
from badminton1d.utils import move_toward


class TransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()
        self.slow_receiver_config = SimulationConfig(player=PlayerConfig(v_max=1.0))

    def test_stage_transition_updates_positions_and_hitter(self) -> None:
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
        feasible = feasible_intercept_indices(state, action, self.config)
        self.assertTrue(feasible)

        chosen = feasible[0]
        record = step_stage(state, action, chosen, self.config)
        times, xs, ys, zs = candidate_intercept_points(state, action, self.config)
        t_int = float(times[chosen])
        x_int = float(xs[chosen])
        y_int = float(ys[chosen])
        z_int = float(zs[chosen])
        hitter_move = move_toward((state.x_left, state.y_left), (action.x_rec, action.y_rec), self.config.player.v_max * t_int)
        assert isinstance(hitter_move, tuple)
        receiver_target = intercept_body_target_after_reaction(
            (state.x_right, state.y_right),
            (state.v_x_right, state.v_y_right),
            (x_int, y_int),
            "right",
            self.config,
            target_z=z_int,
            reaction_time=state.reaction_time_right,
        )
        receiver_move = advance_player_toward(
            (state.x_right, state.y_right),
            (state.v_x_right, state.v_y_right),
            receiver_target,
            t_int,
            self.config,
            reaction_time=state.reaction_time_right,
            stop_when_early=True,
        )

        self.assertFalse(record.next_state.rally_done)
        self.assertEqual(record.next_state.current_hitter, "right")
        self.assertAlmostEqual(record.next_state.x0, x_int)
        self.assertAlmostEqual(record.next_state.y0, y_int)
        self.assertAlmostEqual(record.next_state.z0, z_int)
        self.assertAlmostEqual(record.next_state.x_left, hitter_move[0])
        self.assertAlmostEqual(record.next_state.y_left, hitter_move[1])
        self.assertAlmostEqual(record.next_state.x_right, receiver_move.position[0])
        self.assertAlmostEqual(record.next_state.y_right, receiver_move.position[1])

    def test_terminal_when_no_feasible_intercept_exists(self) -> None:
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

        self.assertTrue(record.next_state.rally_done)
        self.assertEqual(record.next_state.winner, "left")
        self.assertEqual(record.terminal_reason, "no_feasible_intercept")

    def test_candidate_sampling_keeps_narrow_feasible_window_interceptable(self) -> None:
        config = SimulationConfig(action=ActionConfig(intercept_count=3))
        state = StageState(
            x_left=-1.2,
            y_left=-2.5,
            x_right=0.8,
            y_right=2.0,
            current_hitter="left",
            x0=-1.2,
            y0=-2.5,
            z0=1.2,
            stage_index=1,
        )
        action = ShotAction(v_x=1.1, v_y=4.9, v_z=5.75, x_rec=0.0, y_rec=-2.0)

        feasible = feasible_intercept_indices(state, action, config)
        self.assertTrue(feasible)

    def test_receiver_reaction_time_reduces_feasible_intercepts(self) -> None:
        action = ShotAction(v_x=0.9, v_y=5.8, v_z=5.0, x_rec=0.0, y_rec=-1.8)
        base_state = StageState(
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
        delayed_state = StageState(
            x_left=-0.5,
            y_left=-2.5,
            x_right=0.5,
            y_right=2.5,
            current_hitter="left",
            x0=-0.5,
            y0=-2.5,
            z0=1.7,
            reaction_time_right=1.0,
            stage_index=1,
        )

        base_feasible = feasible_intercept_indices(base_state, action, self.config)
        delayed_feasible = feasible_intercept_indices(delayed_state, action, self.config)

        self.assertTrue(base_feasible)
        self.assertLess(len(delayed_feasible), len(base_feasible))

    def test_reaction_miss_probability_uses_default_curve(self) -> None:
        self.assertAlmostEqual(reaction_miss_probability(0.05), 0.9)
        self.assertAlmostEqual(reaction_miss_probability(REACTION_MISS_FLIGHT_TIME_THRESHOLD), 0.9)
        self.assertAlmostEqual(reaction_miss_probability(0.3), 0.6)
        self.assertAlmostEqual(reaction_miss_probability(REACTION_MISS_SECONDARY_FLIGHT_TIME_THRESHOLD), 0.3)
        self.assertAlmostEqual(reaction_miss_probability(0.6), 0.15)
        self.assertAlmostEqual(reaction_miss_probability(REACTION_MISS_ZERO_FLIGHT_TIME_THRESHOLD), 0.0)
        self.assertAlmostEqual(reaction_miss_probability(0.8), 0.0)

    def test_fast_intercept_can_fail_from_reaction_miss(self) -> None:
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
        action = ShotAction(v_x=0.0, v_y=6.5, v_z=2.0, x_rec=0.0, y_rec=-1.0)
        feasible = feasible_intercept_indices(state, action, self.config)

        self.assertTrue(feasible)
        chosen = feasible[0]
        times, _, _, _ = candidate_intercept_points(state, action, self.config)
        self.assertGreater(reaction_miss_probability(float(times[chosen]), self.config), 0.0)

        with patch("badminton1d.dynamics.np.random.random", return_value=0.0):
            record = step_stage(state, action, chosen, self.config)

        self.assertTrue(record.next_state.rally_done)
        self.assertEqual(record.next_state.winner, "left")
        self.assertEqual(record.terminal_reason, "reaction_miss")
        self.assertIsNone(record.intercept_point)
        self.assertIsNone(record.chosen_time)

    def test_late_intercept_has_no_reaction_miss_by_default(self) -> None:
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
        action = ShotAction(v_x=0.0, v_y=1.5, v_z=4.0, x_rec=0.0, y_rec=-2.5)
        feasible = feasible_intercept_indices(state, action, self.config)

        self.assertTrue(feasible)
        chosen = feasible[0]
        times, _, _, _ = candidate_intercept_points(state, action, self.config)
        chosen_time = float(times[chosen])
        self.assertGreaterEqual(chosen_time, REACTION_MISS_ZERO_FLIGHT_TIME_THRESHOLD)
        self.assertAlmostEqual(reaction_miss_probability(chosen_time), 0.0)

        with patch("badminton1d.dynamics.np.random.random", return_value=0.29):
            returned_record = step_stage(state, action, chosen, self.config)
        self.assertNotEqual(returned_record.terminal_reason, "reaction_miss")

    def test_mid_speed_intercept_uses_linear_reaction_miss_curve(self) -> None:
        config = SimulationConfig()
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
        action = ShotAction(v_x=0.0, v_y=3.5, v_z=1.5, x_rec=0.0, y_rec=-2.5)
        feasible = feasible_intercept_indices(state, action, config)

        self.assertTrue(feasible)
        chosen = feasible[0]
        times, _, _, _ = candidate_intercept_points(state, action, config)
        chosen_time = float(times[chosen])
        self.assertGreaterEqual(chosen_time, REACTION_MISS_FLIGHT_TIME_THRESHOLD)
        self.assertLess(chosen_time, REACTION_MISS_SECONDARY_FLIGHT_TIME_THRESHOLD)
        miss_probability = reaction_miss_probability(chosen_time, config)
        self.assertGreater(miss_probability, 0.3)
        self.assertLess(miss_probability, 0.9)

        with patch("badminton1d.dynamics.np.random.random", return_value=miss_probability - 0.01):
            missed_record = step_stage(state, action, chosen, config)
        with patch("badminton1d.dynamics.np.random.random", return_value=miss_probability + 0.01):
            returned_record = step_stage(state, action, chosen, config)

        self.assertEqual(missed_record.terminal_reason, "reaction_miss")
        self.assertNotEqual(returned_record.terminal_reason, "reaction_miss")

    def test_one_dimensional_mode_keeps_x_positions_collapsed(self) -> None:
        config = SimulationConfig(court=CourtConfig(mode="1d"))
        state = StageState(
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
        action = ShotAction(v_x=3.0, v_y=5.8, v_z=5.0, x_rec=1.0, y_rec=-1.8)
        feasible = feasible_intercept_indices(state, action, config)

        self.assertTrue(feasible)
        record = step_stage(state, action, feasible[0], config)
        self.assertAlmostEqual(record.validated_action.applied.v_x, 0.0)
        self.assertAlmostEqual(record.validated_action.applied.x_rec, 0.0)
        self.assertAlmostEqual(record.next_state.x0, 0.0)
        self.assertAlmostEqual(record.next_state.x_left, 0.0)
        self.assertAlmostEqual(record.next_state.x_right, 0.0)


if __name__ == "__main__":
    unittest.main()
