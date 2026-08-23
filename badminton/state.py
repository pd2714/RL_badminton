from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Side = Literal["left", "right"]


@dataclass(frozen=True)
class ShotAction:
    v_x: float
    v_y: float
    v_z: float
    x_rec: float
    y_rec: float


@dataclass(frozen=True)
class ValidatedShotAction:
    requested: ShotAction
    applied: ShotAction
    projected: bool


@dataclass(frozen=True)
class StageState:
    x_left: float
    y_left: float
    x_right: float
    y_right: float
    current_hitter: Side
    x0: float
    y0: float
    z0: float
    v_x_left: float = 0.0
    v_y_left: float = 0.0
    v_x_right: float = 0.0
    v_y_right: float = 0.0
    reaction_time_left: float = 0.0
    reaction_time_right: float = 0.0
    rally_done: bool = False
    winner: Side | None = None
    stage_index: int = 0


@dataclass
class StageRecord:
    stage_index: int
    state_before: StageState
    validated_action: ValidatedShotAction
    receiver_side: Side
    candidate_times: np.ndarray
    feasible_indices: list[int]
    chosen_index: int | None
    chosen_time: float | None
    intercept_point: tuple[float, float, float] | None
    next_state: StageState
    reward_left: float
    reward_right: float
    intended_intercept_time: float | None = None
    intended_intercept_point: tuple[float, float, float] | None = None
    terminal_reason: str | None = None
    notes: list[str] = field(default_factory=list)
