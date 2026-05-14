from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib
import numpy as np

from badminton1d.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.dynamics import candidate_intercept_points, feasible_intercept_indices, valid_hitter_action
from badminton1d.render import (
    GROUND_MARKER_Z,
    OFFICIAL_DOUBLES_WIDTH,
    PLAYER_FOOT_Z,
    setup_3d_court_axes,
    stage_colors,
)
from badminton1d.shot_generators.shot_naming import name_velocity_shot
from badminton1d.state import ShotAction, Side, StageState
from badminton1d.trajectory import simulate_trajectory
from badminton1d.utils import default_player_position, opponent_side, recovery_bounds, service_court_x, side_y_bounds, x_bounds


DEFAULT_RUN_DIR = Path(
    "outputs/rl/"
    "selfplay_2d_accel65_theta65_speed50_rt015_ps40_reach150_zmax26_feasible003_nointercept002_oppint001_attack002_deflift02_ifratio01_reaction_continuation_20k_equal_sides_20260506"
)
DEFAULT_SPEED_SLIDER_MAX = 80.0
DEFAULT_DISCRETE_CONFIG = DiscreteActionConfig()


def configure_interactive_backend() -> None:
    if "--no-show" in sys.argv:
        return
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        return
    for candidate in ("MacOSX", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


configure_interactive_backend()

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a 2D trajectory with continuous phi/theta/speed/recovery sliders."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Self-play output directory containing selfplay_config.json.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional explicit selfplay_config.json path.")
    parser.add_argument("--hitter", choices=("left", "right"), default="left", help="Current hitter side.")
    parser.add_argument("--stage-index", type=int, default=1, help="Use 0 for serves, 1 or higher for rally shots.")
    parser.add_argument("--x0", type=float, default=None, help="Initial contact x. Defaults to hitter start x.")
    parser.add_argument("--y0", type=float, default=None, help="Initial contact y. Defaults to hitter start y.")
    parser.add_argument("--z0", type=float, default=1.7, help="Initial contact height.")
    parser.add_argument("--opponent-x", type=float, default=None, help="Initial opponent/receiver x. Defaults to receiver start x.")
    parser.add_argument("--opponent-y", type=float, default=None, help="Initial opponent/receiver y. Defaults to receiver start y.")
    parser.add_argument("--phi-init", type=float, default=None, dest="phi_init", help="Initial horizontal angle in degrees.")
    parser.add_argument("--theta-init", type=float, default=35.0, dest="theta_init", help="Initial vertical launch angle in degrees.")
    parser.add_argument("--v-init", type=float, default=8.0, dest="v_init", help="Initial total launch speed.")
    parser.add_argument(
        "--v-slider-max",
        type=float,
        default=DEFAULT_SPEED_SLIDER_MAX,
        dest="v_slider_max",
        help="Maximum total speed shown on the slider.",
    )
    parser.add_argument("--x-rec-init", type=float, default=None, dest="x_rec_init", help="Initial recovery target x.")
    parser.add_argument("--y-rec-init", type=float, default=None, dest="y_rec_init", help="Initial recovery target y.")
    parser.add_argument("--kh-init", type=float, default=None, dest="kh_init", help="Initial horizontal drag coefficient.")
    parser.add_argument("--kv-init", type=float, default=None, dest="kv_init", help="Initial vertical drag coefficient.")
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default=None, help="Override trajectory model.")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional output path for saving the current figure.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening an interactive window.")
    return parser.parse_args()


def _load_run_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find run config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Run config must be a JSON object: {path}")
    return data


def _number(data: dict[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _integer(data: dict[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if value is None:
        return int(default)
    return int(value)


def _build_config(run_data: dict[str, object], args: argparse.Namespace) -> tuple[SimulationConfig, float]:
    default = SimulationConfig()
    trajectory_mode = args.trajectory_mode or str(run_data.get("trajectory_mode", "drag_square"))
    horizontal_drag = _number(
        run_data,
        "horizontal_drag_coefficient",
        default.action.effective_horizontal_drag_coefficient,
    )
    vertical_drag = _number(
        run_data,
        "vertical_drag_coefficient",
        default.action.effective_vertical_drag_coefficient,
    )
    if args.kh_init is not None:
        horizontal_drag = float(args.kh_init)
    if args.kv_init is not None:
        vertical_drag = float(args.kv_init)
    config = SimulationConfig(
        court=CourtConfig(mode=str(run_data.get("court_mode", "2d"))),
        player=PlayerConfig(
            v_max=_number(run_data, "player_speed", default.player.v_max),
            acceleration=_number(run_data, "player_acceleration", default.player.acceleration),
            r_reach=_number(run_data, "racket_length", default.player.r_reach),
            z_max=_number(run_data, "max_hitting_height", default.player.z_max),
            movement_model=str(run_data.get("movement_model", default.player.movement_model)),
        ),
        action=ActionConfig(
            trajectory_mode=trajectory_mode,
            drag_coefficient=horizontal_drag,
            horizontal_drag_coefficient=horizontal_drag,
            vertical_drag_coefficient=vertical_drag,
            vy_min_forward=_number(run_data, "shuttle_speed_min", default.action.vy_min_forward),
            vy_max_forward=max(
                _number(run_data, "shuttle_speed_max", default.action.vy_max_forward),
                float(args.v_slider_max),
            ),
            intercept_count=_integer(run_data, "intercept_count", default.action.intercept_count),
        ),
    )
    reaction_time = _number(run_data, "reaction_time", 0.0)
    return config, reaction_time


def _velocity_from_phi_theta_speed(phi_deg: float, theta_deg: float, speed: float) -> tuple[float, float, float]:
    phi = float(np.deg2rad(phi_deg))
    theta = float(np.deg2rad(theta_deg))
    horizontal_speed = float(speed * np.cos(theta))
    return (
        float(horizontal_speed * np.cos(phi)),
        float(horizontal_speed * np.sin(phi)),
        float(speed * np.sin(theta)),
    )


def _action_space_phi_range_deg(state: StageState, config: SimulationConfig) -> tuple[float, float]:
    mapper = DiscreteActionMapper(config, DEFAULT_DISCRETE_CONFIG, policy_type="velocity_oriented")
    phi_grid = mapper.phi_grid(state)
    phi_values = np.rad2deg(phi_grid)
    return float(np.min(phi_values)), float(np.max(phi_values))


def _set_slider_bounds(slider: Slider, lower: float, upper: float, value: float | None = None) -> float:
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        lower, upper = upper, lower
    slider.valmin = lower
    slider.valmax = upper
    slider.ax.set_xlim(lower, upper)
    next_value = float(slider.val if value is None else value)
    next_value = float(np.clip(next_value, lower, upper))
    previous_eventson = slider.eventson
    slider.eventson = False
    slider.set_val(next_value)
    slider.eventson = previous_eventson
    return next_value


def _net_phi_bounds_deg(x0: float, y0: float, config: SimulationConfig) -> tuple[float, float]:
    left_angle = float(np.degrees(np.arctan2(config.court.net_y - y0, -config.court.half_width - x0)))
    right_angle = float(np.degrees(np.arctan2(config.court.net_y - y0, config.court.half_width - x0)))
    return tuple(sorted((left_angle, right_angle)))


def _net_clearance_theta_deg(x0: float, y0: float, z0: float, phi_deg: float, config: SimulationConfig) -> float:
    del x0
    phi = float(np.deg2rad(phi_deg))
    sin_phi = float(np.sin(phi))
    required_z = config.court.net_height + config.action.net_clearance_margin
    if abs(sin_phi) <= 1e-6:
        return 0.0
    horizontal_distance_to_net = abs((config.court.net_y - y0) / sin_phi)
    return float(np.degrees(np.arctan2(required_z - z0, max(horizontal_distance_to_net, 1e-6))))


def _set_scatter3d(artist, x: float | None, y: float | None, z: float | None) -> None:
    if x is None or y is None or z is None:
        artist._offsets3d = ([], [], [])
        return
    artist._offsets3d = ([x], [y], [z])


def _set_scatter3d_many(artist, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
    artist._offsets3d = (xs.tolist(), ys.tolist(), zs.tolist())


def _use_full_width_3d_view(ax: plt.Axes, config: SimulationConfig) -> None:
    display_half_width = max(config.court.half_width, OFFICIAL_DOUBLES_WIDTH / 2.0)
    pad = config.render.court_padding * 0.06
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()
    x_min = -display_half_width - pad
    x_max = display_half_width + pad
    ax.set_xlim(x_min, x_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))


def _player_body_height(config: SimulationConfig) -> float:
    body_height = min(config.player.z_max - config.player.r_reach, 1.7)
    return max(body_height, config.court.net_height * 0.9)


def _update_player_artist(
    body,
    head,
    position: tuple[float, float],
    *,
    body_height: float,
) -> None:
    x_pos, y_pos = position
    body.set_data_3d(
        [x_pos, x_pos],
        [y_pos, y_pos],
        [PLAYER_FOOT_Z, PLAYER_FOOT_Z + body_height],
    )
    _set_scatter3d(head, x_pos, y_pos, PLAYER_FOOT_Z + body_height)


def _state_from_sliders(
    sliders: dict[str, Slider],
    *,
    config: SimulationConfig,
    hitter: Side,
    reaction_time: float,
) -> StageState:
    stage_index = max(int(sliders["stage"].val), 0)
    if stage_index == 0:
        _, default_service_y = default_player_position(hitter, config)
        x0 = float(service_court_x(hitter, config))
        y0 = float(default_service_y)
    else:
        x0 = float(sliders["x0"].val)
        y0 = float(sliders["y0"].val)
    opponent_x = float(sliders["opponent_x"].val)
    opponent_y = float(sliders["opponent_y"].val)
    if hitter == "left":
        x_left, y_left = x0, y0
        x_right, y_right = opponent_x, opponent_y
    else:
        x_left, y_left = opponent_x, opponent_y
        x_right, y_right = x0, y0
    return StageState(
        x_left=float(x_left),
        y_left=float(y_left),
        x_right=float(x_right),
        y_right=float(y_right),
        current_hitter=hitter,
        x0=x0,
        y0=y0,
        z0=float(sliders["z0"].val),
        v_x_left=0.0,
        v_y_left=0.0,
        v_x_right=0.0,
        v_y_right=0.0,
        reaction_time_left=reaction_time,
        reaction_time_right=reaction_time,
        stage_index=max(int(stage_index), 0),
    )


def _receiver_feasibility_config(config: SimulationConfig) -> SimulationConfig:
    return replace(config, player=replace(config.player, movement_model="accelerated"))


def build_plot(
    args: argparse.Namespace,
    config: SimulationConfig,
    *,
    reaction_time: float,
) -> tuple[plt.Figure, dict[str, Slider]]:
    if not config.court.lateral_motion_enabled:
        raise ValueError("This continuous action-space slider is for 2D trajectories.")

    colors = stage_colors(monochrome=False)
    hitter: Side = args.hitter
    receiver = opponent_side(hitter)
    default_hitter_x, default_hitter_y = default_player_position(hitter, config)
    default_receiver_x, default_receiver_y = default_player_position(receiver, config)

    x_low, x_high = x_bounds(config)
    y_low, y_high = side_y_bounds(hitter, config)
    opponent_y_low, opponent_y_high = side_y_bounds(receiver, config)
    z_low, z_high = config.player.z_min, config.player.z_max
    (rec_x_low, rec_x_high), (rec_y_low, rec_y_high) = recovery_bounds(hitter, config)

    stage_init = max(int(args.stage_index), 0)
    if stage_init == 0:
        x0_init = float(np.clip(service_court_x(hitter, config), x_low, x_high))
        y0_init = float(np.clip(default_hitter_y, y_low, y_high))
    else:
        x0_init = float(np.clip(default_hitter_x if args.x0 is None else args.x0, x_low, x_high))
        y0_init = float(np.clip(default_hitter_y if args.y0 is None else args.y0, y_low, y_high))
    z0_init = float(np.clip(args.z0, z_low, z_high))
    opponent_x_init = float(np.clip(default_receiver_x if args.opponent_x is None else args.opponent_x, x_low, x_high))
    opponent_y_init = float(np.clip(default_receiver_y if args.opponent_y is None else args.opponent_y, opponent_y_low, opponent_y_high))
    initial_state = StageState(
        x_left=x0_init if hitter == "left" else opponent_x_init,
        y_left=y0_init if hitter == "left" else opponent_y_init,
        x_right=opponent_x_init if hitter == "left" else x0_init,
        y_right=opponent_y_init if hitter == "left" else y0_init,
        current_hitter=hitter,
        x0=x0_init,
        y0=y0_init,
        z0=z0_init,
        reaction_time_left=reaction_time,
        reaction_time_right=reaction_time,
        stage_index=stage_init,
    )
    phi_low_init, phi_high_init = _action_space_phi_range_deg(initial_state, config)
    phi_default = 90.0 if hitter == "left" else -90.0
    phi_init = float(args.phi_init if args.phi_init is not None else phi_default)
    phi_init = float(np.clip(phi_init, phi_low_init, phi_high_init))
    theta_init = float(np.clip(args.theta_init, -80.0, 80.0))
    speed_slider_max = float(max(args.v_slider_max, config.action.vy_min_forward + 1.0))
    speed_init = float(np.clip(args.v_init, config.action.vy_min_forward, speed_slider_max))
    kh_init = float(max(config.action.effective_horizontal_drag_coefficient, 0.0))
    kv_init = float(max(config.action.effective_vertical_drag_coefficient, 0.0))
    xrec_init = float(np.clip(0.5 * (rec_x_low + rec_x_high) if args.x_rec_init is None else args.x_rec_init, rec_x_low, rec_x_high))
    yrec_init = float(np.clip(0.5 * (rec_y_low + rec_y_high) if args.y_rec_init is None else args.y_rec_init, rec_y_low, rec_y_high))

    fig = plt.figure(figsize=(12.4, 10.0))
    ax = fig.add_axes([0.02, 0.28, 0.70, 0.68], projection="3d")
    setup_3d_court_axes(ax, config, colors, show_axes=False)
    _use_full_width_3d_view(ax, config)
    ax.set_title("Continuous Trajectory Slider", pad=4.0)

    body_height = _player_body_height(config)
    (left_player_body,) = ax.plot([], [], [], color=colors["left_player"], linewidth=5.0, solid_capstyle="round", zorder=4)
    (right_player_body,) = ax.plot([], [], [], color=colors["right_player"], linewidth=5.0, solid_capstyle="round", zorder=4)
    left_player_head = ax.scatter([], [], [], color=colors["left_player"], s=28, zorder=5)
    right_player_head = ax.scatter([], [], [], color=colors["right_player"], s=28, zorder=5)

    (trajectory_line,) = ax.plot([], [], [], color=colors["trajectory"], linewidth=2.2, alpha=0.8, label="trajectory")
    (aim_line,) = ax.plot([], [], [], color="tab:cyan", linewidth=1.8, alpha=0.9, label="phi aim")
    start_point = ax.scatter([], [], [], color=colors["start"], s=70, label="contact", zorder=5)
    landing_point = ax.scatter([], [], [], color=colors["target"], s=70, label="landing", zorder=5)
    net_point = ax.scatter([], [], [], color=colors["intercept"], s=80, label="net crossing", zorder=5)
    recovery_point = ax.scatter([], [], [], color=colors["recovery"], s=75, alpha=0.85, label="recovery", zorder=5)
    candidate_points = ax.scatter([], [], [], color="0.7", s=22, alpha=0.45, label="intercept samples", zorder=4)
    feasible_points = ax.scatter([], [], [], color="gold", s=42, alpha=0.9, label="feasible intercepts", zorder=5)

    info_text = fig.text(0.75, 0.93, "", ha="left", va="top", fontsize=9.5, linespacing=1.32)
    fig.legend(loc="upper left", bbox_to_anchor=(0.75, 0.41), frameon=False, fontsize=9, borderaxespad=0.0)

    slider_specs = [
        ("stage", 0, max(30, stage_init), stage_init, 1),
        ("x0", x_low, x_high, x0_init, 0.01),
        ("y0", y_low, y_high, y0_init, 0.01),
        ("z0", z_low, z_high, z0_init, 0.01),
        ("opponent_x", x_low, x_high, opponent_x_init, 0.01),
        ("opponent_y", opponent_y_low, opponent_y_high, opponent_y_init, 0.01),
        ("phi_deg", phi_low_init, phi_high_init, phi_init, 0.1),
        ("theta_deg", -80.0, 80.0, theta_init, 0.1),
        ("v", config.action.vy_min_forward, speed_slider_max, speed_init, 0.01),
        ("k_h", 0.0, 1.5, kh_init, 0.001),
        ("k_v", 0.0, 1.5, kv_init, 0.001),
        ("x_rec", rec_x_low, rec_x_high, xrec_init, 0.01),
        ("y_rec", rec_y_low, rec_y_high, yrec_init, 0.01),
    ]
    sliders: dict[str, Slider] = {}
    top = 0.255
    step = 0.021
    for index, (label, lower, upper, init, valstep) in enumerate(slider_specs):
        axis = fig.add_axes([0.16, top - index * step, 0.56, 0.016])
        sliders[label] = Slider(axis, label, lower, upper, valinit=init, valstep=valstep)

    def update(_: float) -> None:
        stage_index = max(int(sliders["stage"].val), 0)
        if stage_index == 0:
            _, default_service_y = default_player_position(hitter, config)
            for slider_name, service_value in (
                ("x0", service_court_x(hitter, config)),
                ("y0", default_service_y),
            ):
                slider = sliders[slider_name]
                previous_eventson = slider.eventson
                slider.eventson = False
                slider.set_val(float(service_value))
                slider.eventson = previous_eventson

        local_config = replace(
            config,
            action=replace(
                config.action,
                drag_coefficient=float(sliders["k_h"].val),
                horizontal_drag_coefficient=float(sliders["k_h"].val),
                vertical_drag_coefficient=float(sliders["k_v"].val),
            ),
        )
        state = _state_from_sliders(
            sliders,
            config=config,
            hitter=hitter,
            reaction_time=reaction_time,
        )
        phi_low, phi_high = _action_space_phi_range_deg(state, local_config)
        phi_deg = _set_slider_bounds(sliders["phi_deg"], phi_low, phi_high)
        theta_deg = float(sliders["theta_deg"].val)
        speed = float(sliders["v"].val)
        vx, vy, vz = _velocity_from_phi_theta_speed(phi_deg, theta_deg, speed)
        action = ShotAction(
            v_x=vx,
            v_y=vy,
            v_z=vz,
            x_rec=float(sliders["x_rec"].val),
            y_rec=float(sliders["y_rec"].val),
        )

        _update_player_artist(
            left_player_body,
            left_player_head,
            (state.x_left, state.y_left),
            body_height=body_height,
        )
        _update_player_artist(
            right_player_body,
            right_player_head,
            (state.x_right, state.y_right),
            body_height=body_height,
        )

        result = simulate_trajectory(state.x0, state.y0, state.z0, action.v_x, action.v_y, action.v_z, local_config)
        xs = np.asarray([point.x for point in result.samples], dtype=float)
        ys = np.asarray([point.y for point in result.samples], dtype=float)
        zs = np.asarray([point.z for point in result.samples], dtype=float)
        crossing = result.net_crossing
        legal = valid_hitter_action(state, action, local_config, result=result)
        feasibility_config = _receiver_feasibility_config(local_config)
        candidate_ts, candidate_xs, candidate_ys, candidate_zs = candidate_intercept_points(state, action, feasibility_config)
        feasible = feasible_intercept_indices(state, action, feasibility_config)

        color = "tab:blue" if legal else "tab:red"
        trajectory_line.set_data_3d(xs, ys, zs)
        trajectory_line.set_color(color)
        phi = float(np.deg2rad(phi_deg))
        aim_length = 3.2
        aim_line.set_data_3d(
            [state.x0, state.x0 + aim_length * np.cos(phi)],
            [state.y0, state.y0 + aim_length * np.sin(phi)],
            [state.z0, state.z0],
        )
        _set_scatter3d(start_point, state.x0, state.y0, state.z0)
        _set_scatter3d(landing_point, result.landing_x, result.landing_y, GROUND_MARKER_Z)
        _set_scatter3d(recovery_point, action.x_rec, action.y_rec, GROUND_MARKER_Z)
        if crossing is None:
            _set_scatter3d(net_point, None, None, None)
        else:
            _set_scatter3d(net_point, crossing.x, crossing.y, crossing.z)
            net_point.set_color(color)
        _set_scatter3d_many(candidate_points, candidate_xs, candidate_ys, candidate_zs)
        if feasible:
            feasible_indices = np.asarray(feasible, dtype=int)
            _set_scatter3d_many(
                feasible_points,
                candidate_xs[feasible_indices],
                candidate_ys[feasible_indices],
                candidate_zs[feasible_indices],
            )
        else:
            _set_scatter3d_many(feasible_points, np.asarray([]), np.asarray([]), np.asarray([]))

        net_phi_low, net_phi_high = _net_phi_bounds_deg(state.x0, state.y0, local_config)
        theta_net = _net_clearance_theta_deg(state.x0, state.y0, state.z0, phi_deg, local_config)
        z_net = crossing.z if crossing is not None else float("nan")
        shot_name = name_velocity_shot(
            hitter=hitter,
            contact_x=state.x0,
            contact_y=state.y0,
            landing_x=result.landing_x,
            landing_y=result.landing_y,
            theta_degrees=theta_deg,
            config=local_config,
        )
        info_text.set_text(
            "\n".join(
                [
                    f"run = {args.run_dir.name}",
                    f"hitter = {hitter}  receiver = {receiver}  stage = {stage_index}",
                    f"mode = {local_config.action.trajectory_mode}  drag = "
                    f"({local_config.action.effective_horizontal_drag_coefficient:.3f}, "
                    f"{local_config.action.effective_vertical_drag_coefficient:.3f})",
                    f"receiver reach = accelerated from rest, v_max {feasibility_config.player.v_max:.2f} m/s, "
                    f"a {feasibility_config.player.acceleration:.2f} m/s^2",
                    f"shot = {shot_name}",
                    f"contact = ({state.x0:.2f}, {state.y0:.2f}, {state.z0:.2f})",
                    f"opponent = ({sliders['opponent_x'].val:.2f}, {sliders['opponent_y'].val:.2f})",
                    f"phi = {phi_deg:.1f} deg  action span [{phi_low:.1f}, {phi_high:.1f}]",
                    f"net span = [{net_phi_low:.1f}, {net_phi_high:.1f}]",
                    f"theta = {theta_deg:.1f} deg  net min {theta_net:.1f}",
                    f"speed = {speed:.2f} / {speed_slider_max:.0f} m/s",
                    f"launch = ({action.v_x:.2f}, {action.v_y:.2f}, {action.v_z:.2f}) m/s",
                    f"recovery = ({action.x_rec:.2f}, {action.y_rec:.2f})",
                    f"landing = ({result.landing_x:.2f}, {result.landing_y:.2f}) in {result.landing_time:.2f}s",
                    f"z(net) = {z_net:.2f} m",
                    f"intercepts = {len(feasible)}/{len(candidate_ts)} feasible",
                    f"status: {'legal' if legal else 'invalid'}",
                ]
            )
        )
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)

    update(0.0)
    return fig, sliders


def main() -> None:
    args = parse_args()
    config_path = args.config_path or args.run_dir / "selfplay_config.json"
    run_data = _load_run_config(config_path)
    config, reaction_time = _build_config(run_data, args)

    fig, sliders = build_plot(args, config, reaction_time=reaction_time)

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=160, bbox_inches="tight")
        print(f"saved figure to {args.save_path}")

    print(
        "continuous trajectory sliders ready: "
        f"x0={sliders['x0'].val:.2f}, "
        f"y0={sliders['y0'].val:.2f}, "
        f"z0={sliders['z0'].val:.2f}, "
        f"stage={int(sliders['stage'].val)}, "
        f"opponent_x={sliders['opponent_x'].val:.2f}, "
        f"opponent_y={sliders['opponent_y'].val:.2f}, "
        f"phi={sliders['phi_deg'].val:.1f}deg, "
        f"theta={sliders['theta_deg'].val:.1f}deg, "
        f"v={sliders['v'].val:.2f}, "
        f"k_h={sliders['k_h'].val:.3f}, "
        f"k_v={sliders['k_v'].val:.3f}, "
        f"x_rec={sliders['x_rec'].val:.2f}, "
        f"y_rec={sliders['y_rec'].val:.2f}"
    )

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
