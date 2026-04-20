from __future__ import annotations

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import step_stage
from badminton1d.state import ShotAction, StageRecord, StageState
from badminton1d.utils import default_player_position


def default_initial_state(config: SimulationConfig | None = None) -> StageState:
    active_config = config or SimulationConfig()
    left_x, left_y = default_player_position("left", active_config)
    right_x, right_y = default_player_position("right", active_config)
    return StageState(
        x_left=left_x,
        y_left=left_y,
        x_right=right_x,
        y_right=right_y,
        current_hitter="left",
        x0=left_x,
        y0=left_y,
        z0=1.7,
        rally_done=False,
        winner=None,
        stage_index=0,
    )


class Badminton1DEnv:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.state = default_initial_state(self.config)

    def reset(self, state: StageState | None = None) -> StageState:
        self.state = state or default_initial_state(self.config)
        return self.state

    def step(self, action: ShotAction, intercept_index: int | None) -> StageRecord:
        record = step_stage(self.state, action, intercept_index, self.config)
        self.state = record.next_state
        return record
