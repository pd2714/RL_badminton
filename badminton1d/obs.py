from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import landing_position
from badminton1d.state import ShotAction, Side, StageState
from badminton1d.utils import canonicalize_shot_action_for_agent, canonicalize_state_for_agent, side_to_agent_canonical


@dataclass(frozen=True)
class ObservationConfig:
    max_score: int = 11
    max_stages_per_rally: int = 30
    include_feasible_mask: bool = True


class ObservationEncoder:
    def __init__(
        self,
        config: SimulationConfig,
        observation_config: ObservationConfig | None = None,
    ) -> None:
        self.config = config
        self.observation_config = observation_config or ObservationConfig()

    @property
    def size(self) -> int:
        base = 18
        pending = 11
        movement = 4
        mask = self.config.action.intercept_count if self.observation_config.include_feasible_mask else 0
        return base + pending + movement + mask

    def feature_names(self) -> list[str]:
        names = [
            "x_agent",
            "y_agent",
            "x_opponent",
            "y_opponent",
            "x0",
            "y0",
            "z0",
            "current_hitter_agent",
            "current_hitter_opponent",
            "server_agent",
            "server_opponent",
            "agent_canonical_left",
            "opponent_canonical_right",
            "role_is_hitter",
            "role_is_receiver",
            "score_left",
            "score_right",
            "stage_progress",
            "pending_action_active",
            "pending_feasible_fraction",
            "pending_v_x",
            "pending_v_y",
            "pending_v_z",
            "pending_x_rec",
            "pending_y_rec",
            "pending_landing_x",
            "pending_landing_y",
            "pending_landing_dx",
            "pending_landing_dy",
            "v_x_agent",
            "v_y_agent",
            "v_x_opponent",
            "v_y_opponent",
        ]
        if self.observation_config.include_feasible_mask:
            names.extend(f"feasible_intercept_{index}" for index in range(self.config.action.intercept_count))
        return names

    def encode(
        self,
        *,
        state: StageState,
        agent_side: Side,
        role: str,
        server_side: Side,
        score_left: int = 0,
        score_right: int = 0,
        pending_action: ShotAction | None = None,
        feasible_indices: list[int] | None = None,
    ) -> np.ndarray:
        feasible_indices = feasible_indices or []
        canonical_state = canonicalize_state_for_agent(state, agent_side)
        canonical_server_side = side_to_agent_canonical(server_side, agent_side)
        canonical_pending_action = (
            None if pending_action is None else canonicalize_shot_action_for_agent(pending_action, agent_side)
        )
        if agent_side == "right":
            score_left, score_right = score_right, score_left

        is_hitter = 1.0 if role == "hitter" else 0.0
        mask = np.zeros(self.config.action.intercept_count, dtype=np.float32)
        if self.observation_config.include_feasible_mask:
            for index in feasible_indices:
                if 0 <= index < len(mask):
                    mask[index] = 1.0

        pending_active = 1.0 if pending_action is not None else 0.0
        landing_x = 0.0
        landing_y = 0.0
        if canonical_pending_action is not None:
            landing_x, landing_y = landing_position(canonical_state, canonical_pending_action, self.config)

        max_velocity = max(
            abs(self.config.action.vx_min),
            abs(self.config.action.vx_max),
            abs(self.config.action.vy_min_forward),
            abs(self.config.action.vy_max_forward),
            self.config.action.vz_max,
        )
        features = [
            self._normalize_x(canonical_state.x_left),
            self._normalize_y(canonical_state.y_left),
            self._normalize_x(canonical_state.x_right),
            self._normalize_y(canonical_state.y_right),
            self._normalize_x(canonical_state.x0),
            self._normalize_y(canonical_state.y0),
            self._normalize_unit(canonical_state.z0, 0.0, self.config.render.z_max),
            1.0 if canonical_state.current_hitter == "left" else 0.0,
            1.0 if canonical_state.current_hitter == "right" else 0.0,
            1.0 if canonical_server_side == "left" else 0.0,
            1.0 if canonical_server_side == "right" else 0.0,
            1.0,
            0.0,
            is_hitter,
            1.0 - is_hitter,
            self._normalize_unit(score_left, 0.0, max(float(self.observation_config.max_score), 1.0)),
            self._normalize_unit(score_right, 0.0, max(float(self.observation_config.max_score), 1.0)),
            self._normalize_unit(
                canonical_state.stage_index,
                0.0,
                max(float(self.observation_config.max_stages_per_rally), 1.0),
            ),
            pending_active,
            float(len(feasible_indices)) / max(float(self.config.action.intercept_count), 1.0),
            self._normalize_signed(canonical_pending_action.v_x, max_velocity) if canonical_pending_action else 0.0,
            self._normalize_signed(canonical_pending_action.v_y, max_velocity) if canonical_pending_action else 0.0,
            self._normalize_signed(canonical_pending_action.v_z, max_velocity) if canonical_pending_action else 0.0,
            self._normalize_x(canonical_pending_action.x_rec) if canonical_pending_action else 0.0,
            self._normalize_y(canonical_pending_action.y_rec) if canonical_pending_action else 0.0,
            self._normalize_x(landing_x) if canonical_pending_action else 0.0,
            self._normalize_y(landing_y) if canonical_pending_action else 0.0,
            self._normalize_signed(landing_x - canonical_state.x0, self.config.court.width) if canonical_pending_action else 0.0,
            self._normalize_signed(landing_y - canonical_state.y0, self.config.court.length) if canonical_pending_action else 0.0,
            self._normalize_signed(canonical_state.v_x_left, self.config.player.v_max),
            self._normalize_signed(canonical_state.v_y_left, self.config.player.v_max),
            self._normalize_signed(canonical_state.v_x_right, self.config.player.v_max),
            self._normalize_signed(canonical_state.v_y_right, self.config.player.v_max),
        ]
        if self.observation_config.include_feasible_mask:
            features.extend(mask.tolist())
        return np.asarray(features, dtype=np.float32)

    def _normalize_x(self, value: float) -> float:
        return self._normalize_signed(value, self.config.court.half_width)

    def _normalize_y(self, value: float) -> float:
        return self._normalize_signed(value, self.config.court.half_length)

    def _normalize_signed(self, value: float, scale: float) -> float:
        if scale <= 0.0:
            return 0.0
        return float(np.clip(value / scale, -1.0, 1.0))

    def _normalize_unit(self, value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
