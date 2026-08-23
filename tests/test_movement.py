from __future__ import annotations

import unittest

from badminton.config import PlayerConfig, SimulationConfig
from badminton.movement import (
    advance_player_during_reaction,
    advance_player_toward,
    brake_start_time,
    closest_intercept_body_target,
    earliest_arrival_time,
    earliest_stop_arrival_time,
    usable_racket_reach,
)


class MovementTests(unittest.TestCase):
    def test_constant_velocity_arrival_time_uses_reach_radius(self) -> None:
        config = SimulationConfig(player=PlayerConfig(v_max=3.2, movement_model="constant_velocity"))

        arrival = earliest_arrival_time((0.0, 0.0), (2.0, 0.0), (3.65, 0.0), config)

        self.assertAlmostEqual(arrival, (3.65 - config.player.r_reach) / config.player.v_max)

    def test_accelerated_movement_can_brake_when_arriving_early(self) -> None:
        config = SimulationConfig(player=PlayerConfig(v_max=3.2, acceleration=4.5, movement_model="accelerated"))
        start = (0.0, 0.0)
        velocity = (0.0, 0.0)
        target = (2.4, 0.0)

        arrival = earliest_arrival_time(start, velocity, target, config, reach_radius=0.0)
        stop_arrival = earliest_stop_arrival_time(start, velocity, target, config)
        result = advance_player_toward(start, velocity, target, stop_arrival + 0.25, config)

        self.assertLess(arrival, stop_arrival)
        self.assertIsNotNone(brake_start_time(start, velocity, target, stop_arrival + 0.25, config))
        self.assertTrue(result.arrived)
        self.assertAlmostEqual(result.position[0], target[0])
        self.assertAlmostEqual(result.position[1], target[1])
        self.assertAlmostEqual(result.velocity[0], 0.0)
        self.assertAlmostEqual(result.velocity[1], 0.0)

    def test_accelerated_arrival_accounts_for_current_velocity_direction(self) -> None:
        config = SimulationConfig(player=PlayerConfig(v_max=3.2, acceleration=4.5, movement_model="accelerated"))
        start = (0.0, 0.0)
        target = (3.0, 0.0)

        toward = earliest_arrival_time(start, (2.0, 0.0), target, config, reach_radius=0.0)
        away = earliest_arrival_time(start, (-2.0, 0.0), target, config, reach_radius=0.0)

        self.assertLess(toward, away)

    def test_reaction_time_continues_existing_motion_before_new_seek(self) -> None:
        config = SimulationConfig(player=PlayerConfig(v_max=4.0, acceleration=4.0, movement_model="accelerated"))
        start = (0.0, 0.0)
        velocity = (2.0, 0.0)

        reaction_motion = advance_player_during_reaction(start, velocity, 0.25, config)
        result = advance_player_toward(start, velocity, (10.0, 0.0), 0.25, config, reaction_time=0.5)

        self.assertGreater(reaction_motion.position[0], start[0])
        self.assertGreater(reaction_motion.velocity[0], 0.0)
        self.assertAlmostEqual(result.position[0], reaction_motion.position[0])
        self.assertAlmostEqual(result.velocity[0], reaction_motion.velocity[0])

    def test_reaction_time_feasibility_uses_post_reaction_state(self) -> None:
        config = SimulationConfig(player=PlayerConfig(v_max=4.0, acceleration=4.0, movement_model="accelerated"))
        start = (0.0, 0.0)
        target = (4.0, 0.0)

        arrival_from_braking_motion = earliest_arrival_time(start, (4.0, 0.0), target, config, reach_radius=0.0, reaction_time=0.5)
        arrival_from_still_start = earliest_arrival_time(start, (0.0, 0.0), target, config, reach_radius=0.0, reaction_time=0.5)

        self.assertLess(arrival_from_braking_motion, arrival_from_still_start)

    def test_directional_racket_reach_only_points_toward_net(self) -> None:
        config = SimulationConfig(player=PlayerConfig(r_reach=1.3))

        self.assertAlmostEqual(usable_racket_reach((0.0, -3.0), (0.0, -2.0), "left", config), 1.3)
        self.assertAlmostEqual(usable_racket_reach((0.0, -3.0), (0.0, -4.0), "left", config), 0.0)
        self.assertAlmostEqual(usable_racket_reach((0.0, -3.0), (1.0, -3.0), "left", config), 1.3)
        self.assertAlmostEqual(usable_racket_reach((0.0, 3.0), (0.0, 2.0), "right", config), 1.3)
        self.assertAlmostEqual(usable_racket_reach((0.0, 3.0), (0.0, 4.0), "right", config), 0.0)
        self.assertAlmostEqual(usable_racket_reach((0.0, 3.0), (1.0, 3.0), "right", config), 1.3)

    def test_intercept_body_target_stops_one_racket_length_short(self) -> None:
        config = SimulationConfig(player=PlayerConfig(r_reach=1.3))

        target = closest_intercept_body_target((0.0, 3.0), (0.0, 1.0), "right", config)

        self.assertAlmostEqual(target[0], 0.0)
        self.assertAlmostEqual(target[1], 2.3)

    def test_intercept_body_target_for_behind_point_keeps_target_y_but_not_exact_x(self) -> None:
        config = SimulationConfig(player=PlayerConfig(r_reach=1.3))

        target = closest_intercept_body_target((0.0, 3.0), (1.0, 4.0), "right", config)

        self.assertAlmostEqual(target[0], 0.0)
        self.assertAlmostEqual(target[1], 4.0)

    def test_intercept_body_target_clamps_lateral_reach_when_behind_point_is_far_x(self) -> None:
        config = SimulationConfig(player=PlayerConfig(r_reach=1.3))

        target = closest_intercept_body_target((5.0, 3.0), (1.0, 4.0), "right", config)

        self.assertAlmostEqual(target[0], 2.3)
        self.assertAlmostEqual(target[1], 4.0)

    def test_horizontal_racket_reach_shrinks_as_height_approaches_max_z(self) -> None:
        config = SimulationConfig(player=PlayerConfig(r_reach=1.3, z_max=2.6))

        low_target = closest_intercept_body_target((0.0, 3.0), (0.0, 1.0), "right", config, target_z=1.3)
        high_target = closest_intercept_body_target((0.0, 3.0), (0.0, 1.0), "right", config, target_z=1.95)
        max_target = closest_intercept_body_target((0.0, 3.0), (0.0, 1.0), "right", config, target_z=2.6)

        self.assertAlmostEqual(low_target[1], 2.3)
        self.assertGreater(high_target[1], 2.125)
        self.assertLess(high_target[1], 2.126)
        self.assertAlmostEqual(max_target[0], 0.0)
        self.assertAlmostEqual(max_target[1], 1.0)


if __name__ == "__main__":
    unittest.main()
