from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CourtConfig:
    mode: str = "2d"
    length: float = 13.4
    width: float = 5.18
    net_y: float = 0.0
    net_height: float = 1.55
    boundary_margin: float = 0.15
    service_line_distance_from_net: float = 2.0
    default_player_start_distance_from_net: float = 3.5
    default_player_start_x: float = 0.0
    service_x_offset_from_center_line: float = 0.1
    service_receive_x_offset_toward_center: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in {"1d", "2d"}:
            raise ValueError("court mode must be '1d' or '2d'.")
        if self.service_x_offset_from_center_line < 0.0:
            raise ValueError("service_x_offset_from_center_line must be non-negative.")
        if self.service_x_offset_from_center_line > self.half_width:
            raise ValueError("service_x_offset_from_center_line must not exceed court half width.")
        if self.service_receive_x_offset_toward_center < 0.0:
            raise ValueError("service_receive_x_offset_toward_center must be non-negative.")

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def lateral_motion_enabled(self) -> bool:
        return self.mode == "2d"


@dataclass(frozen=True)
class PlayerConfig:
    v_max: float = 4.0
    acceleration: float = 6.5
    deceleration: float | None = None
    movement_model: str = "accelerated"
    r_reach: float = 1.3
    z_min: float = 0.15
    z_max: float = 2.6
    marker_radius: float = 0.22

    def __post_init__(self) -> None:
        if self.v_max <= 0.0:
            raise ValueError("v_max must be positive.")
        if self.acceleration <= 0.0:
            raise ValueError("acceleration must be positive.")
        if self.deceleration is not None and self.deceleration <= 0.0:
            raise ValueError("deceleration must be positive when provided.")
        if self.movement_model not in {"constant_velocity", "accelerated"}:
            raise ValueError("movement_model must be 'constant_velocity' or 'accelerated'.")

    @property
    def effective_deceleration(self) -> float:
        return self.acceleration if self.deceleration is None else self.deceleration


@dataclass(frozen=True)
class ActionConfig:
    gravity: float = 9.81
    trajectory_mode: str = "ballistic"
    drag_coefficient: float = 0.2
    horizontal_drag_coefficient: float | None = 0.2
    vertical_drag_coefficient: float | None = 0.16
    drag_dt: float = 0.01
    vx_min: float = -6.0
    vx_max: float = 6.0
    vy_min_forward: float = 0.1
    vy_max_forward: float = 80.0
    vz_min: float = -20.0
    vz_max: float = 20.0
    net_clearance_margin: float = 0.05
    recovery_x_margin: float = 0.25
    recovery_net_margin: float = 0.3
    recovery_back_margin: float = 0.5
    intercept_count: int = 20
    intercept_time_min: float = 0.02
    intercept_margin_before_landing: float = 0.02
    invalid_receiver_choice_loses: bool = True
    reaction_miss_fast_threshold: float = 0.1
    reaction_miss_fast_probability: float = 0.9
    reaction_miss_secondary_threshold: float = 0.5
    reaction_miss_secondary_probability: float = 0.3
    reaction_miss_zero_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive.")
        if self.drag_coefficient < 0.0:
            raise ValueError("drag_coefficient must be non-negative.")
        if self.horizontal_drag_coefficient is not None and self.horizontal_drag_coefficient < 0.0:
            raise ValueError("horizontal_drag_coefficient must be non-negative.")
        if self.vertical_drag_coefficient is not None and self.vertical_drag_coefficient < 0.0:
            raise ValueError("vertical_drag_coefficient must be non-negative.")
        if self.drag_dt <= 0.0:
            raise ValueError("drag_dt must be positive.")
        if self.trajectory_mode not in {"ballistic", "drag", "drag_square"}:
            raise ValueError("trajectory_mode must be 'ballistic', 'drag', or 'drag_square'.")
        if not self.vx_min < self.vx_max:
            raise ValueError("vx range must be increasing.")
        if not self.vy_min_forward < self.vy_max_forward:
            raise ValueError("vy range must be increasing.")
        if not self.vz_min < self.vz_max:
            raise ValueError("vz range must be increasing.")
        if self.intercept_time_min < 0.0:
            raise ValueError("intercept_time_min must be non-negative.")
        if self.intercept_margin_before_landing < 0.0:
            raise ValueError("intercept_margin_before_landing must be non-negative.")
        if self.reaction_miss_fast_threshold < 0.0:
            raise ValueError("reaction_miss_fast_threshold must be non-negative.")
        if self.reaction_miss_secondary_threshold < 0.0:
            raise ValueError("reaction_miss_secondary_threshold must be non-negative.")
        if self.reaction_miss_secondary_threshold < self.reaction_miss_fast_threshold:
            raise ValueError("reaction_miss_secondary_threshold must be at least reaction_miss_fast_threshold.")
        if self.reaction_miss_zero_threshold < 0.0:
            raise ValueError("reaction_miss_zero_threshold must be non-negative.")
        if self.reaction_miss_zero_threshold < self.reaction_miss_secondary_threshold:
            raise ValueError("reaction_miss_zero_threshold must be at least reaction_miss_secondary_threshold.")
        if not 0.0 <= self.reaction_miss_fast_probability <= 1.0:
            raise ValueError("reaction_miss_fast_probability must be in [0, 1].")
        if not 0.0 <= self.reaction_miss_secondary_probability <= 1.0:
            raise ValueError("reaction_miss_secondary_probability must be in [0, 1].")

    @property
    def effective_horizontal_drag_coefficient(self) -> float:
        if self.horizontal_drag_coefficient is not None:
            return self.horizontal_drag_coefficient
        return self.drag_coefficient

    @property
    def effective_vertical_drag_coefficient(self) -> float:
        if self.vertical_drag_coefficient is not None:
            return self.vertical_drag_coefficient
        return self.drag_coefficient

    @property
    def effective_trajectory_mode(self) -> str:
        if self.trajectory_mode == "drag":
            return "drag_square"
        return self.trajectory_mode

    @property
    def uses_square_drag(self) -> bool:
        return self.effective_trajectory_mode == "drag_square"


@dataclass(frozen=True)
class RenderConfig:
    trajectory_samples: int = 200
    court_padding: float = 0.6
    figure_size: tuple[float, float] = (8.5, 10.0)
    gif_fps: float = 1.2
    shuttle_marker_min: float = 55.0
    shuttle_marker_max: float = 150.0
    z_max: float = 6.0


@dataclass(frozen=True)
class SimulationConfig:
    court: CourtConfig = field(default_factory=CourtConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def candidate_times(self, landing_time: float, *, lower_bound: float | None = None) -> np.ndarray:
        upper = landing_time - self.action.intercept_margin_before_landing
        lower = self.action.intercept_time_min if lower_bound is None else lower_bound
        if upper <= lower:
            return np.asarray([], dtype=float)
        return np.linspace(lower, upper, self.action.intercept_count)
