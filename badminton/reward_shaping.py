from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from badminton.config import SimulationConfig
from badminton.dynamics import effective_flight_time, landing_position
from badminton.pressure import shot_pressure_from_record
from badminton.state import StageRecord
from badminton.utils import player_position, side_x_bounds, side_y_bounds


@dataclass(frozen=True)
class LoopPenaltyConfig:
    penalty: float = 0.0
    window: int = 4


@dataclass(frozen=True)
class PressureRewardConfig:
    weight: float = 0.0


@dataclass(frozen=True)
class OpponentTravelRewardConfig:
    weight: float = 0.0


@dataclass(frozen=True)
class ReturnDepthRewardConfig:
    weight: float = 0.0


@dataclass(frozen=True)
class NetProximityRewardConfig:
    weight: float = 0.0
    distance_threshold: float = 0.5


@dataclass(frozen=True)
class AttackRewardConfig:
    weight: float = 0.0
    min_speed: float = 18.0
    downward_vz_threshold: float = 0.0


@dataclass(frozen=True)
class DefensiveLiftRewardConfig:
    weight: float = 0.0
    intercept_flight_ratio_reward_weight: float = 0.0
    min_stage_index: int = 1
    min_theta_deg: float = 15.0
    target_flight_time: float = 1.4
    min_depth_ratio: float = 0.7
    min_ratio_intended_flight_time: float = 0.6


@dataclass
class ActionStreakTracker:
    current_action: int | None = None
    current_streak: int = 0
    longest_streak: int = 0
    repeated_action_events: int = 0
    streak_lengths: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.current_action = None
        self.current_streak = 0
        self.longest_streak = 0
        self.repeated_action_events = 0
        self.streak_lengths.clear()

    def observe(self, action_bin: int) -> int:
        if self.current_action == action_bin:
            self.current_streak += 1
            self.repeated_action_events += 1
        else:
            if self.current_streak > 0:
                self.streak_lengths.append(self.current_streak)
            self.current_action = action_bin
            self.current_streak = 1
        self.longest_streak = max(self.longest_streak, self.current_streak)
        return self.current_streak

    def finalize(self) -> None:
        if self.current_streak > 0:
            self.streak_lengths.append(self.current_streak)

    def summary(self) -> dict[str, float | int]:
        avg_streak = float(np.mean(self.streak_lengths)) if self.streak_lengths else 0.0
        return {
            "max_repeated_action_streak": int(self.longest_streak),
            "avg_repeated_action_streak": avg_streak,
            "repeated_action_events": int(self.repeated_action_events),
        }


def loop_penalty_for_streak(streak: int, config: LoopPenaltyConfig) -> float:
    if config.penalty <= 0.0 or config.window <= 1:
        return 0.0
    if streak < config.window:
        return 0.0
    return -float(config.penalty)


def pressure_reward_from_record(
    record: StageRecord,
    *,
    weight: float,
    config: SimulationConfig,
) -> float:
    if weight <= 0.0:
        return 0.0

    pressure = shot_pressure_from_record(record, config).pressure
    return float(weight * pressure)


def opponent_travel_reward_from_record(
    record: StageRecord,
    *,
    weight: float,
    config: SimulationConfig,
) -> float:
    if weight <= 0.0 or record.intercept_point is None:
        return 0.0

    receiver_start_x, receiver_start_y = player_position(record.state_before, record.receiver_side)
    intercept_x, intercept_y, _ = record.intercept_point
    travel_distance = float(np.hypot(intercept_x - receiver_start_x, intercept_y - receiver_start_y))
    x_low, x_high = side_x_bounds(record.receiver_side, config)
    y_low, y_high = side_y_bounds(record.receiver_side, config)
    max_travel_distance = float(np.hypot(x_high - x_low, y_high - y_low))
    if max_travel_distance <= 1e-6:
        return 0.0
    travel_score = float(np.clip(travel_distance / max_travel_distance, 0.0, 1.0))
    return float(weight * travel_score)


def return_depth_reward_from_record(
    record: StageRecord,
    *,
    weight: float,
    config: SimulationConfig,
) -> float:
    if weight <= 0.0:
        return 0.0

    landing_y = float(record.validated_action.applied.y_rec)
    if record.receiver_side == "left":
        back_line_y = -config.court.half_length + config.court.boundary_margin
        span = max(config.court.net_y - back_line_y, 1e-6)
        depth_score = (config.court.net_y - landing_y) / span
    else:
        back_line_y = config.court.half_length - config.court.boundary_margin
        span = max(back_line_y - config.court.net_y, 1e-6)
        depth_score = (landing_y - config.court.net_y) / span
    return float(weight * np.clip(depth_score, 0.0, 1.0))


def net_proximity_reward_from_record(
    record: StageRecord,
    *,
    weight: float,
    net_y: float,
    distance_threshold: float,
) -> float:
    if weight <= 0.0 or distance_threshold <= 0.0:
        return 0.0

    landing_y = float(record.validated_action.applied.y_rec)
    return float(weight if abs(landing_y - net_y) < distance_threshold else 0.0)


def attack_reward_from_record(
    record: StageRecord,
    *,
    config: SimulationConfig,
    reward_config: AttackRewardConfig,
) -> float:
    if reward_config.weight <= 0.0:
        return 0.0
    if not record.next_state.rally_done or record.next_state.winner != record.state_before.current_hitter:
        return 0.0

    action = record.validated_action.applied
    speed = float(np.linalg.norm([action.v_x, action.v_y, action.v_z]))
    if speed < reward_config.min_speed:
        return 0.0
    if action.v_z >= reward_config.downward_vz_threshold:
        return 0.0

    _, landing_y = landing_position(record.state_before, action, config)
    landing_y = float(landing_y)
    net_y = float(config.court.net_y)
    if record.receiver_side == "right":
        lands_on_opponent_side = landing_y > net_y
    else:
        lands_on_opponent_side = landing_y < net_y
    return float(reward_config.weight if lands_on_opponent_side else 0.0)


def defensive_lift_reward_from_record(
    record: StageRecord,
    *,
    config: SimulationConfig,
    reward_config: DefensiveLiftRewardConfig,
) -> float:
    if reward_config.weight <= 0.0:
        return 0.0
    if record.state_before.stage_index < reward_config.min_stage_index:
        return 0.0
    if record.next_state.rally_done:
        return 0.0

    depth_score = _landing_depth_score(record, config)
    if depth_score < reward_config.min_depth_ratio:
        return 0.0

    action = record.validated_action.applied
    theta_deg = _shot_theta_deg(action=action)
    if theta_deg <= float(reward_config.min_theta_deg):
        return 0.0

    flight_time = effective_flight_time(record.state_before, action, config)
    flight_score = np.clip(flight_time / max(float(reward_config.target_flight_time), 1e-6), 0.0, 1.0)
    upward_score = np.clip(float(action.v_z) / max(float(config.action.vz_max), 1e-6), 0.0, 1.0)
    lift_score = depth_score * flight_score * upward_score
    return float(reward_config.weight * np.clip(lift_score, 0.0, 1.0))


def intercept_flight_ratio_reward_from_record(
    record: StageRecord,
    *,
    config: SimulationConfig,
    reward_config: DefensiveLiftRewardConfig,
) -> float:
    if reward_config.intercept_flight_ratio_reward_weight <= 0.0:
        return 0.0
    if record.state_before.stage_index < reward_config.min_stage_index:
        return 0.0
    if record.next_state.rally_done or record.chosen_time is None:
        return 0.0

    intended_flight_time = effective_flight_time(record.state_before, record.validated_action.applied, config)
    if intended_flight_time < float(reward_config.min_ratio_intended_flight_time):
        return 0.0

    ratio = float(record.chosen_time) / max(float(intended_flight_time), 1e-6)
    return float(reward_config.intercept_flight_ratio_reward_weight * np.clip(ratio, 0.0, 1.0))


def _shot_theta_deg(*, action) -> float:
    horizontal_speed = float(np.hypot(action.v_x, action.v_y))
    return float(np.rad2deg(np.arctan2(float(action.v_z), max(horizontal_speed, 1e-6))))


def _landing_depth_score(record: StageRecord, config: SimulationConfig) -> float:
    _, landing_y = landing_position(record.state_before, record.validated_action.applied, config)
    landing_y = float(landing_y)
    if record.receiver_side == "left":
        back_line_y = -config.court.half_length + config.court.boundary_margin
        span = max(config.court.net_y - back_line_y, 1e-6)
        return float(np.clip((config.court.net_y - landing_y) / span, 0.0, 1.0))

    back_line_y = config.court.half_length - config.court.boundary_margin
    span = max(back_line_y - config.court.net_y, 1e-6)
    return float(np.clip((landing_y - config.court.net_y) / span, 0.0, 1.0))
