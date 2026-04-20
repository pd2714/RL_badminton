from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from badminton1d.state import StageRecord


@dataclass(frozen=True)
class LoopPenaltyConfig:
    penalty: float = 0.0
    window: int = 4


@dataclass(frozen=True)
class PressureRewardConfig:
    weight: float = 0.0
    late_intercept_weight: float = 0.5
    low_intercept_weight: float = 0.5


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
    z_min: float,
    z_max: float,
) -> float:
    if weight <= 0.0 or record.intercept_point is None or record.chosen_time is None:
        return 0.0

    intercept_z = float(record.intercept_point[2])
    z_span = max(z_max - z_min, 1e-6)
    low_contact_score = 1.0 - np.clip((intercept_z - z_min) / z_span, 0.0, 1.0)
    total_time = float(record.candidate_times[-1]) if len(record.candidate_times) else max(float(record.chosen_time), 1e-6)
    late_contact_score = np.clip(float(record.chosen_time) / max(total_time, 1e-6), 0.0, 1.0)
    pressure_score = 0.5 * low_contact_score + 0.5 * late_contact_score
    return float(weight * pressure_score)
