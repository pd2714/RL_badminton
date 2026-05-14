from __future__ import annotations

import unittest

import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.pressure import shot_pressure_from_record
from badminton1d.reward_shaping import (
    AttackRewardConfig,
    DefensiveLiftRewardConfig,
    attack_reward_from_record,
    defensive_lift_reward_from_record,
    intercept_flight_ratio_reward_from_record,
    net_proximity_reward_from_record,
    opponent_travel_reward_from_record,
    pressure_reward_from_record,
    return_depth_reward_from_record,
)
from badminton1d.state import ShotAction, StageRecord, StageState, ValidatedShotAction
from badminton1d.trajectory import ballistic_landing_time


def make_record(
    *,
    receiver_side: str = "right",
    receiver_y: float = 3.5,
    intercept_point: tuple[float, float, float] | None = (0.0, 5.0, 1.0),
    landing_y: float = 5.0,
) -> StageRecord:
    state_before = StageState(
        x_left=0.0,
        y_left=-3.5,
        x_right=0.0,
        y_right=receiver_y,
        current_hitter="left",
        x0=0.0,
        y0=-3.5,
        z0=1.0,
    )
    next_state = StageState(
        x_left=0.0,
        y_left=-3.5,
        x_right=0.0,
        y_right=receiver_y,
        current_hitter=receiver_side,
        x0=0.0,
        y0=landing_y,
        z0=1.0,
        stage_index=1,
    )
    shot = ShotAction(v_x=0.0, v_y=6.0, v_z=3.0, x_rec=0.0, y_rec=landing_y)
    return StageRecord(
        stage_index=0,
        state_before=state_before,
        validated_action=ValidatedShotAction(requested=shot, applied=shot, projected=False),
        receiver_side=receiver_side,  # type: ignore[arg-type]
        candidate_times=np.asarray([0.2, 0.4, 0.6], dtype=float),
        feasible_indices=[1, 2, 3],
        chosen_index=2,
        chosen_time=0.4,
        intercept_point=intercept_point,
        next_state=next_state,
        reward_left=0.0,
        reward_right=0.0,
    )


def make_landing_record(
    *,
    config: SimulationConfig,
    landing_y: float,
    v_z: float,
    chosen_time: float = 0.8,
    contact_z: float = 0.8,
    intercept_z: float = 1.0,
) -> StageRecord:
    state_before = StageState(
        x_left=0.0,
        y_left=-3.5,
        x_right=0.0,
        y_right=3.5,
        current_hitter="left",
        x0=0.0,
        y0=-3.5,
        z0=contact_z,
        stage_index=1,
    )
    flight_time = ballistic_landing_time(state_before.z0, v_z, config.action.gravity)
    shot = ShotAction(
        v_x=0.0,
        v_y=(landing_y - state_before.y0) / flight_time,
        v_z=v_z,
        x_rec=0.0,
        y_rec=-3.5,
    )
    next_state = StageState(
        x_left=0.0,
        y_left=-3.5,
        x_right=0.0,
        y_right=landing_y,
        current_hitter="right",
        x0=0.0,
        y0=landing_y,
        z0=intercept_z,
        stage_index=2,
    )
    return StageRecord(
        stage_index=1,
        state_before=state_before,
        validated_action=ValidatedShotAction(requested=shot, applied=shot, projected=False),
        receiver_side="right",
        candidate_times=np.asarray([0.2, chosen_time, flight_time], dtype=float),
        feasible_indices=[1],
        chosen_index=1,
        chosen_time=chosen_time,
        intercept_point=(0.0, landing_y, intercept_z),
        next_state=next_state,
        reward_left=0.0,
        reward_right=0.0,
    )


class RewardShapingTests(unittest.TestCase):
    def test_pressure_reward_uses_shot_pressure_index(self) -> None:
        config = SimulationConfig()
        record = make_record()

        reward = pressure_reward_from_record(record, weight=0.01, config=config)

        self.assertAlmostEqual(reward, 0.01 * shot_pressure_from_record(record, config).pressure)

    def test_opponent_travel_reward_increases_with_receiver_movement(self) -> None:
        config = SimulationConfig()
        record = make_record(intercept_point=(0.0, 5.8, 1.0))
        reward = opponent_travel_reward_from_record(record, weight=0.05, config=config)
        self.assertGreater(reward, 0.0)
        self.assertLessEqual(reward, 0.05)

    def test_return_depth_reward_prefers_backcourt(self) -> None:
        config = SimulationConfig()
        shallow = make_record(landing_y=0.4)
        deep = make_record(landing_y=config.court.half_length - config.court.boundary_margin)
        shallow_reward = return_depth_reward_from_record(shallow, weight=0.05, config=config)
        deep_reward = return_depth_reward_from_record(deep, weight=0.05, config=config)
        self.assertLess(shallow_reward, deep_reward)
        self.assertAlmostEqual(deep_reward, 0.05)

    def test_net_proximity_reward_triggers_inside_threshold(self) -> None:
        near_net = make_record(landing_y=0.3)
        far_from_net = make_record(landing_y=1.2)
        self.assertAlmostEqual(
            net_proximity_reward_from_record(near_net, weight=0.05, net_y=0.0, distance_threshold=0.5),
            0.05,
        )
        self.assertAlmostEqual(
            net_proximity_reward_from_record(far_from_net, weight=0.05, net_y=0.0, distance_threshold=0.5),
            0.0,
        )

    def test_attack_reward_triggers_for_fast_downward_opponent_side_winner(self) -> None:
        config = SimulationConfig()
        state_before = StageState(
            x_left=0.0,
            y_left=-0.5,
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=-0.5,
            z0=2.0,
            stage_index=2,
        )
        next_state = StageState(
            x_left=0.0,
            y_left=-0.5,
            x_right=0.0,
            y_right=3.5,
            current_hitter="right",
            x0=0.0,
            y0=1.8,
            z0=0.0,
            rally_done=True,
            winner="left",
            stage_index=3,
        )
        shot = ShotAction(v_x=0.0, v_y=5.0, v_z=-2.0, x_rec=0.0, y_rec=-3.0)
        record = StageRecord(
            stage_index=2,
            state_before=state_before,
            validated_action=ValidatedShotAction(requested=shot, applied=shot, projected=False),
            receiver_side="right",
            candidate_times=np.asarray([0.2, 0.4], dtype=float),
            feasible_indices=[],
            chosen_index=None,
            chosen_time=None,
            intercept_point=None,
            next_state=next_state,
            reward_left=1.0,
            reward_right=-1.0,
        )

        reward = attack_reward_from_record(
            record,
            config=config,
            reward_config=AttackRewardConfig(weight=0.07, min_speed=5.0),
        )

        self.assertAlmostEqual(reward, 0.07)

    def test_defensive_lift_reward_prefers_high_deep_long_return(self) -> None:
        config = SimulationConfig()
        shallow_drive = make_landing_record(config=config, landing_y=2.4, v_z=1.0, chosen_time=0.35)
        high_lift = make_landing_record(config=config, landing_y=5.8, v_z=8.0, chosen_time=1.1)
        reward_config = DefensiveLiftRewardConfig(weight=0.08)

        shallow_reward = defensive_lift_reward_from_record(
            shallow_drive,
            config=config,
            reward_config=reward_config,
        )
        lift_reward = defensive_lift_reward_from_record(
            high_lift,
            config=config,
            reward_config=reward_config,
        )

        self.assertEqual(shallow_reward, 0.0)
        self.assertGreater(lift_reward, 0.0)
        self.assertLessEqual(lift_reward, reward_config.weight)

    def test_defensive_lift_reward_requires_high_theta(self) -> None:
        config = SimulationConfig()
        deep_but_flat = make_landing_record(config=config, landing_y=5.8, v_z=4.0, chosen_time=0.9)

        reward = defensive_lift_reward_from_record(
            deep_but_flat,
            config=config,
            reward_config=DefensiveLiftRewardConfig(weight=0.08, min_theta_deg=45.0),
        )

        self.assertEqual(reward, 0.0)

    def test_defensive_lift_reward_does_not_gate_by_contact_height(self) -> None:
        config = SimulationConfig()
        low_contact = make_landing_record(config=config, landing_y=5.8, v_z=8.0, contact_z=0.4)
        high_contact = make_landing_record(config=config, landing_y=5.8, v_z=8.0, contact_z=2.0)
        reward_config = DefensiveLiftRewardConfig(weight=0.08, target_flight_time=0.1)

        low_reward = defensive_lift_reward_from_record(
            low_contact,
            config=config,
            reward_config=reward_config,
        )
        high_reward = defensive_lift_reward_from_record(
            high_contact,
            config=config,
            reward_config=reward_config,
        )

        self.assertAlmostEqual(low_reward, high_reward)

    def test_intercept_flight_ratio_reward_prefers_later_intercepts(self) -> None:
        config = SimulationConfig()
        early_intercept = make_landing_record(config=config, landing_y=5.8, v_z=8.0, chosen_time=0.4)
        late_intercept = make_landing_record(config=config, landing_y=5.8, v_z=8.0, chosen_time=1.2)
        reward_config = DefensiveLiftRewardConfig(intercept_flight_ratio_reward_weight=0.04)

        early_reward = intercept_flight_ratio_reward_from_record(
            early_intercept,
            config=config,
            reward_config=reward_config,
        )
        late_reward = intercept_flight_ratio_reward_from_record(
            late_intercept,
            config=config,
            reward_config=reward_config,
        )

        self.assertGreater(late_reward, early_reward)
        self.assertLessEqual(late_reward, reward_config.intercept_flight_ratio_reward_weight)


if __name__ == "__main__":
    unittest.main()
