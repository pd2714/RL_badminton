from __future__ import annotations

import unittest

from badminton1d.config import SimulationConfig
from badminton1d.shot_generators import name_velocity_shot


class VelocityShotNamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()

    def test_front_landing_is_drop(self) -> None:
        name = name_velocity_shot(
            hitter="left",
            contact_x=0.0,
            contact_y=-5.0,
            landing_x=0.0,
            landing_y=1.0,
            theta_degrees=55.0,
            config=self.config,
        )
        self.assertEqual(name, "drop")

    def test_front_landing_uses_first_three_meters_from_net(self) -> None:
        base = dict(
            hitter="left",
            contact_x=0.0,
            contact_y=-5.0,
            landing_x=0.0,
            theta_degrees=20.0,
            config=self.config,
        )
        self.assertEqual(name_velocity_shot(landing_y=3.0, **base), "drop")
        self.assertEqual(name_velocity_shot(landing_y=3.1, **base), "clear")

    def test_back_or_middle_contact_uses_clear_smash_drive_thresholds(self) -> None:
        base = dict(
            hitter="left",
            contact_x=0.0,
            contact_y=-5.0,
            landing_x=0.0,
            landing_y=5.5,
            config=self.config,
        )
        self.assertEqual(name_velocity_shot(theta_degrees=35.0, **base), "clear")
        self.assertEqual(name_velocity_shot(theta_degrees=-2.0, **base), "smash")
        self.assertEqual(name_velocity_shot(theta_degrees=10.0, **base), "drive")

    def test_front_contact_uses_lift_net_kill_push_thresholds(self) -> None:
        base = dict(
            hitter="left",
            contact_x=0.0,
            contact_y=-0.8,
            landing_x=0.0,
            landing_y=5.5,
            config=self.config,
        )
        self.assertEqual(name_velocity_shot(theta_degrees=50.0, **base), "lift")
        self.assertEqual(name_velocity_shot(theta_degrees=-1.0, **base), "net kill")
        self.assertEqual(name_velocity_shot(theta_degrees=20.0, **base), "push")

    def test_cross_court_modifier_requires_opposite_lateral_halves_and_2_4_meters(self) -> None:
        base = dict(
            hitter="left",
            contact_y=-5.0,
            landing_y=5.5,
            theta_degrees=35.0,
            config=self.config,
        )
        self.assertEqual(
            name_velocity_shot(contact_x=-2.0, landing_x=2.0, **base),
            "cross-court clear",
        )
        self.assertEqual(
            name_velocity_shot(contact_x=-1.3, landing_x=1.3, **base),
            "cross-court clear",
        )
        self.assertEqual(name_velocity_shot(contact_x=-1.0, landing_x=1.0, **base), "clear")
        self.assertEqual(name_velocity_shot(contact_x=1.8, landing_x=2.4, **base), "clear")

    def test_right_hitter_uses_mirrored_longitudinal_depths(self) -> None:
        name = name_velocity_shot(
            hitter="right",
            contact_x=0.0,
            contact_y=5.0,
            landing_x=0.0,
            landing_y=-5.5,
            theta_degrees=-5.0,
            config=self.config,
        )
        self.assertEqual(name, "smash")


if __name__ == "__main__":
    unittest.main()
