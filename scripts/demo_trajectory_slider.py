from __future__ import annotations

import argparse
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
from badminton1d.config import SimulationConfig
from badminton1d.dynamics import valid_hitter_action
from badminton1d.render import GROUND_MARKER_Z, PLAYER_FOOT_Z, setup_3d_court_axes, stage_colors
from badminton1d.state import ShotAction, StageState
from badminton1d.trajectory import simulate_trajectory
from badminton1d.utils import default_player_position, recovery_bounds, service_court_x

DEFAULT_DISCRETE_CONFIG = DiscreteActionConfig(speed_bins=5)


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
    parser = argparse.ArgumentParser(description="Visualize 2D phi/theta/speed shuttle trajectories.")
    parser.add_argument("--x0", type=float, default=0.0, help="Shuttle start x position.")
    parser.add_argument("--y0", type=float, default=-2.5, help="Shuttle start y position.")
    parser.add_argument("--z0", type=float, default=1.7, help="Shuttle start height.")
    parser.add_argument("--stage-index", type=int, default=1, help="Use 0 for serve legality, 1 or higher for rally-shot legality.")
    parser.add_argument("--phi-init", type=float, default=90.0, dest="phi_init", help="Initial horizontal angle in degrees.")
    parser.add_argument("--theta-init", type=float, default=35.0, dest="theta_init", help="Initial vertical launch angle in degrees.")
    parser.add_argument("--v-init", type=float, default=8.0, dest="v_init", help="Initial total launch speed.")
    parser.add_argument("--kh-init", type=float, default=0.2, dest="kh_init", help="Initial horizontal drag coefficient.")
    parser.add_argument("--kv-init", type=float, default=0.16, dest="kv_init", help="Initial vertical drag coefficient.")
    parser.add_argument("--x-rec-init", type=float, default=0.0, dest="x_rec_init", help="Initial recovery target x.")
    parser.add_argument("--y-rec-init", type=float, default=-2.0, dest="y_rec_init", help="Initial recovery target y.")
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="drag_square", help="Trajectory model to visualize.")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional output path for saving the current figure.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening an interactive window.")
    return parser.parse_args()


def _velocity_from_phi_theta_speed(phi_deg: float, theta_deg: float, speed: float) -> tuple[float, float, float]:
    phi = float(np.deg2rad(phi_deg))
    theta = float(np.deg2rad(theta_deg))
    vh = float(speed * np.cos(theta))
    return (
        float(vh * np.cos(phi)),
        float(vh * np.sin(phi)),
        float(speed * np.sin(theta)),
    )


def _state_from_values(
    *,
    x0: float,
    y0: float,
    z0: float,
    right_x: float,
    right_y: float,
    stage_index: int,
) -> StageState:
    return StageState(
        x_left=x0,
        y_left=y0,
        x_right=right_x,
        y_right=right_y,
        current_hitter="left",
        x0=x0,
        y0=y0,
        z0=z0,
        stage_index=max(int(stage_index), 0),
    )


def _action_mapper(config: SimulationConfig) -> DiscreteActionMapper:
    return DiscreteActionMapper(config, DEFAULT_DISCRETE_CONFIG, policy_type="velocity_oriented")


def _nearest_value(value: float, candidates: np.ndarray) -> float:
    index = int(np.argmin(np.abs(candidates - float(value))))
    return float(candidates[index])


def _nearest_angle_index(angle_deg: float, candidates_rad: np.ndarray) -> int:
    angle = float(np.deg2rad(angle_deg))
    deltas = np.arctan2(np.sin(candidates_rad - angle), np.cos(candidates_rad - angle))
    return int(np.argmin(np.abs(deltas)))


def _configure_slider_range(slider: Slider, values: np.ndarray) -> None:
    slider.valmin = float(values[0])
    slider.valmax = float(values[-1])
    slider.valstep = values
    slider.ax.set_xlim(slider.valmin, slider.valmax)


def _net_phi_bounds_deg(x0: float, y0: float, config: SimulationConfig) -> tuple[float, float]:
    left_angle = float(np.degrees(np.arctan2(config.court.net_y - y0, -config.court.half_width - x0)))
    right_angle = float(np.degrees(np.arctan2(config.court.net_y - y0, config.court.half_width - x0)))
    return tuple(sorted((left_angle, right_angle)))


def _net_clearance_theta_deg(x0: float, y0: float, z0: float, phi_deg: float, config: SimulationConfig) -> float:
    del x0
    phi = float(np.deg2rad(phi_deg))
    sin_phi = float(np.sin(phi))
    required_z = config.court.net_height + config.action.net_clearance_margin
    if sin_phi <= 1e-6:
        return 0.0
    horizontal_distance_to_net = abs((config.court.net_y - y0) / sin_phi)
    return float(np.degrees(np.arctan2(required_z - z0, max(horizontal_distance_to_net, 1e-6))))


def _set_scatter3d(artist, x: float | None, y: float | None, z: float | None) -> None:
    if x is None or y is None or z is None:
        artist._offsets3d = ([], [], [])
        return
    artist._offsets3d = ([x], [y], [z])


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
    head._offsets3d = ([x_pos], [y_pos], [PLAYER_FOOT_Z + body_height])


def build_plot(args: argparse.Namespace, config: SimulationConfig) -> tuple[plt.Figure, dict[str, Slider]]:
    colors = stage_colors(monochrome=False)
    (left_x, left_y), (right_x, right_y) = default_player_position("left", config), default_player_position("right", config)
    (left_rec_x_bounds, left_rec_y_bounds) = recovery_bounds("left", config)

    stage_init = max(int(args.stage_index), 0)
    if stage_init == 0:
        x0_init = float(np.clip(service_court_x("left", config), -config.court.half_width + 0.05, config.court.half_width - 0.05))
        y0_init = float(np.clip(left_y, -config.court.half_length + 0.05, -0.05))
    else:
        x0_init = float(np.clip(args.x0, -config.court.half_width + 0.05, config.court.half_width - 0.05))
        y0_init = float(np.clip(args.y0, -config.court.half_length + 0.05, -0.05))
    z0_init = float(np.clip(args.z0, 0.0, config.render.z_max))
    phi_init = float(np.clip(args.phi_init, -180.0, 180.0))
    theta_init = float(np.clip(args.theta_init, -80.0, 80.0))
    kh_init = float(max(args.kh_init, 0.0))
    kv_init = float(max(args.kv_init, 0.0))
    xrec_init = float(np.clip(args.x_rec_init, left_rec_x_bounds[0], left_rec_x_bounds[1]))
    yrec_init = float(np.clip(args.y_rec_init, left_rec_y_bounds[0], left_rec_y_bounds[1]))
    initial_state = _state_from_values(
        x0=x0_init,
        y0=y0_init,
        z0=z0_init,
        right_x=right_x,
        right_y=right_y,
        stage_index=args.stage_index,
    )
    initial_mapper = _action_mapper(config)
    initial_phi_grid = initial_mapper.phi_grid(initial_state)
    phi_init_index = _nearest_angle_index(phi_init, initial_phi_grid)
    initial_theta_grid = initial_mapper.theta_grid(initial_state, phi_init_index)
    theta_init_index = _nearest_angle_index(theta_init, initial_theta_grid)
    initial_speed_bins = initial_mapper.speed_grid(initial_state, phi_init_index, theta_init_index)
    speed_init = _nearest_value(float(np.clip(args.v_init, initial_speed_bins[0], initial_speed_bins[-1])), initial_speed_bins)

    fig = plt.figure(figsize=(12.0, 10.0))
    ax = fig.add_axes([0.02, 0.28, 0.70, 0.68], projection="3d")
    setup_3d_court_axes(ax, config, colors, show_axes=False)
    ax.set_title("Trajectory Slider", pad=4.0)
    body_height = _player_body_height(config)
    (left_player_body,) = ax.plot([], [], [], color=colors["left_player"], linewidth=5.0, solid_capstyle="round", zorder=4)
    (right_player_body,) = ax.plot([], [], [], color=colors["right_player"], linewidth=5.0, solid_capstyle="round", zorder=4)
    left_player_head = ax.scatter([], [], [], color=colors["left_player"], s=28, zorder=5)
    right_player_head = ax.scatter([], [], [], color=colors["right_player"], s=28, zorder=5)

    (trajectory_line,) = ax.plot([], [], [], color=colors["trajectory"], linewidth=2.2, alpha=0.75, label="trajectory")
    (net_phi_left_line,) = ax.plot([], [], [], color="white", linewidth=1.2, alpha=0.8, linestyle=":", label="net angle span")
    (net_phi_right_line,) = ax.plot([], [], [], color="white", linewidth=1.2, alpha=0.8, linestyle=":")
    (aim_line,) = ax.plot([], [], [], color="tab:cyan", linewidth=1.8, alpha=0.9, label="phi aim")
    start_point = ax.scatter([], [], [], color=colors["start"], s=70, label="start", zorder=5)
    landing_point = ax.scatter([], [], [], color=colors["target"], s=70, label="landing", zorder=5)
    net_point = ax.scatter([], [], [], color=colors["intercept"], s=80, label="net crossing", zorder=5)
    recovery_point = ax.scatter([], [], [], color=colors["recovery"], s=75, alpha=0.8, label="recovery", zorder=5)

    info_text = fig.text(
        0.75,
        0.92,
        "",
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
    )
    fig.legend(
        loc="upper left",
        bbox_to_anchor=(0.75, 0.44),
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
    )

    slider_specs = [
        ("stage", 0, max(30, stage_init), stage_init, 1),
        ("x0", -config.court.half_width + 0.05, config.court.half_width - 0.05, x0_init, 0.01),
        ("y0", -config.court.half_length + 0.05, -0.05, y0_init, 0.01),
        ("z0", 0.0, config.render.z_max, z0_init, 0.01),
        ("phi_bin", 0, initial_mapper._effective_phi_bins - 1, phi_init_index, 1),
        ("theta_bin", 0, DEFAULT_DISCRETE_CONFIG.theta_bins - 1, theta_init_index, 1),
        ("v", float(initial_speed_bins[0]), float(initial_speed_bins[-1]), speed_init, initial_speed_bins),
        ("k_h", 0.0, 1.5, kh_init, 0.01),
        ("k_v", 0.0, 1.5, kv_init, 0.01),
        ("x_rec", left_rec_x_bounds[0], left_rec_x_bounds[1], xrec_init, 0.01),
        ("y_rec", left_rec_y_bounds[0], left_rec_y_bounds[1], yrec_init, 0.01),
    ]
    sliders: dict[str, Slider] = {}
    top = 0.26
    step = 0.03
    for index, (label, lower, upper, init, valstep) in enumerate(slider_specs):
        axis = fig.add_axes([0.16, top - index * step, 0.56, 0.022])
        sliders[label] = Slider(axis, label, lower, upper, valinit=init, valstep=valstep)

    def update(_: float) -> None:
        stage_index = max(int(sliders["stage"].val), 0)
        if stage_index == 0:
            for slider_name, service_value in (
                ("x0", service_court_x("left", config)),
                ("y0", left_y),
            ):
                slider = sliders[slider_name]
                previous_eventson = slider.eventson
                slider.eventson = False
                slider.set_val(float(service_value))
                slider.eventson = previous_eventson

        state = _state_from_values(
            x0=float(sliders["x0"].val),
            y0=float(sliders["y0"].val),
            z0=float(sliders["z0"].val),
            right_x=right_x,
            right_y=right_y,
            stage_index=stage_index,
        )
        local_config = SimulationConfig(
            court=config.court,
            player=config.player,
            render=config.render,
            action=replace(
                config.action,
                trajectory_mode=args.trajectory_mode,
                drag_coefficient=sliders["k_h"].val,
                horizontal_drag_coefficient=sliders["k_h"].val,
                vertical_drag_coefficient=sliders["k_v"].val,
            ),
        )
        mapper = _action_mapper(local_config)
        phi_index = int(sliders["phi_bin"].val)
        theta_index = int(sliders["theta_bin"].val)
        phi = float(mapper.phi_grid(state)[phi_index])
        theta = float(mapper.theta_grid(state, phi_index)[theta_index])
        phi_deg = float(np.degrees(phi))
        theta_deg = float(np.degrees(theta))
        action_phi_values = np.rad2deg(mapper.phi_grid(state))
        action_phi_low = float(np.min(action_phi_values))
        action_phi_high = float(np.max(action_phi_values))
        speed_bins = mapper.speed_grid(state, phi_index, theta_index)
        speed_min = float(speed_bins[0])
        speed_max = float(speed_bins[-1])
        _configure_slider_range(sliders["v"], speed_bins)
        speed = _nearest_value(float(sliders["v"].val), speed_bins)
        speed_slider = sliders["v"]
        previous_eventson = speed_slider.eventson
        speed_slider.eventson = False
        speed_slider.set_val(speed)
        speed_slider.eventson = previous_eventson

        vx, vy, vz = _velocity_from_phi_theta_speed(phi_deg, theta_deg, speed)
        action = ShotAction(
            v_x=vx,
            v_y=vy,
            v_z=vz,
            x_rec=sliders["x_rec"].val,
            y_rec=sliders["y_rec"].val,
        )
        result = simulate_trajectory(state.x0, state.y0, state.z0, action.v_x, action.v_y, action.v_z, local_config)
        xs = np.asarray([point.x for point in result.samples], dtype=float)
        ys = np.asarray([point.y for point in result.samples], dtype=float)
        zs = np.asarray([point.z for point in result.samples], dtype=float)
        crossing = result.net_crossing
        legal = valid_hitter_action(state, action, local_config, result=result)
        color = "tab:blue" if legal else "tab:red"
        phi_low, phi_high = _net_phi_bounds_deg(state.x0, state.y0, local_config)
        theta_net = _net_clearance_theta_deg(state.x0, state.y0, state.z0, phi_deg, local_config)

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
        trajectory_line.set_data_3d(xs, ys, zs)
        trajectory_line.set_color(color)
        net_phi_left_line.set_data_3d(
            [state.x0, -local_config.court.half_width],
            [state.y0, local_config.court.net_y],
            [GROUND_MARKER_Z, GROUND_MARKER_Z],
        )
        net_phi_right_line.set_data_3d(
            [state.x0, local_config.court.half_width],
            [state.y0, local_config.court.net_y],
            [GROUND_MARKER_Z, GROUND_MARKER_Z],
        )
        aim_length = 3.2
        aim_phi = np.deg2rad(phi_deg)
        aim_line.set_data_3d(
            [state.x0, state.x0 + aim_length * np.cos(aim_phi)],
            [state.y0, state.y0 + aim_length * np.sin(aim_phi)],
            [state.z0, state.z0],
        )
        _set_scatter3d(start_point, state.x0, state.y0, state.z0)
        _set_scatter3d(landing_point, result.landing_x, result.landing_y, GROUND_MARKER_Z)
        _set_scatter3d(recovery_point, action.x_rec, action.y_rec, GROUND_MARKER_Z)
        if crossing is not None:
            _set_scatter3d(net_point, crossing.x, crossing.y, crossing.z)
            net_point.set_color(color)
        else:
            _set_scatter3d(net_point, None, None, None)

        status = "legal" if legal else "invalid"
        z_net = crossing.z if crossing is not None else float("nan")
        info_text.set_text(
            "\n".join(
                [
                    f"mode = {local_config.action.trajectory_mode}",
                    f"stage = {stage_index}",
                    f"start = ({state.x0:.2f}, {state.y0:.2f}, {state.z0:.2f})",
                    f"bins = phi {DEFAULT_DISCRETE_CONFIG.phi_bins}, theta {DEFAULT_DISCRETE_CONFIG.theta_bins}, v {DEFAULT_DISCRETE_CONFIG.speed_bins}",
                    f"indices = phi {phi_index}, theta {theta_index}",
                    f"phi = {phi_deg:.1f} deg  action span [{action_phi_low:.1f}, {action_phi_high:.1f}]",
                    f"net span = [{phi_low:.1f}, {phi_high:.1f}]",
                    f"theta = {theta_deg:.1f} deg  net min {theta_net:.1f}",
                    f"v = {speed:.2f} m/s  range [{speed_min:.2f}, {speed_max:.2f}]",
                    f"launch = ({action.v_x:.2f}, {action.v_y:.2f}, {action.v_z:.2f}) m/s",
                    f"drag = (kh={sliders['k_h'].val:.3f}, kv={sliders['k_v'].val:.3f})",
                    f"recovery = ({action.x_rec:.2f}, {action.y_rec:.2f})",
                    f"landing = ({result.landing_x:.2f}, {result.landing_y:.2f}) in {result.landing_time:.2f}s",
                    f"z(net) = {z_net:.2f} m",
                    f"status: {status}",
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
    config = SimulationConfig(
        action=replace(
            SimulationConfig().action,
            trajectory_mode=args.trajectory_mode,
            drag_coefficient=args.kh_init,
            horizontal_drag_coefficient=args.kh_init,
            vertical_drag_coefficient=args.kv_init,
        )
    )
    fig, sliders = build_plot(args, config)

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=160, bbox_inches="tight")
        print(f"saved figure to {args.save_path}")

    print(
        "trajectory sliders ready: "
        f"x0={sliders['x0'].val:.2f}, "
        f"y0={sliders['y0'].val:.2f}, "
        f"z0={sliders['z0'].val:.2f}, "
        f"stage={int(sliders['stage'].val)}, "
        f"phi_bin={int(sliders['phi_bin'].val)}, "
        f"theta_bin={int(sliders['theta_bin'].val)}, "
        f"v={sliders['v'].val:.2f}, "
        f"k_h={sliders['k_h'].val:.3f}, "
        f"k_v={sliders['k_v'].val:.3f}"
    )

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
