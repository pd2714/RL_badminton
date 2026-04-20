from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import valid_hitter_action
from badminton1d.render import setup_court_axes, stage_colors
from badminton1d.state import ShotAction, StageState
from badminton1d.trajectory import simulate_trajectory
from badminton1d.utils import default_player_position, recovery_bounds


def configure_interactive_backend() -> None:
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
    parser = argparse.ArgumentParser(description="Visualize 2D launch-based shuttle trajectories.")
    parser.add_argument("--x0", type=float, default=0.0, help="Shuttle start x position.")
    parser.add_argument("--y0", type=float, default=-2.5, help="Shuttle start y position.")
    parser.add_argument("--z0", type=float, default=1.7, help="Shuttle start height.")
    parser.add_argument("--vx-init", type=float, default=0.8, dest="vx_init", help="Initial lateral velocity.")
    parser.add_argument("--vy-init", type=float, default=5.5, dest="vy_init", help="Initial down-court velocity.")
    parser.add_argument("--vz-init", type=float, default=5.0, dest="vz_init", help="Initial vertical velocity.")
    parser.add_argument("--kh-init", type=float, default=0.2, dest="kh_init", help="Initial horizontal drag coefficient.")
    parser.add_argument("--kv-init", type=float, default=0.16, dest="kv_init", help="Initial vertical drag coefficient.")
    parser.add_argument("--x-rec-init", type=float, default=0.0, dest="x_rec_init", help="Initial recovery target x.")
    parser.add_argument("--y-rec-init", type=float, default=-2.0, dest="y_rec_init", help="Initial recovery target y.")
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="drag_square", help="Trajectory model to visualize.")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional output path for saving the current figure.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening an interactive window.")
    return parser.parse_args()


def build_plot(args: argparse.Namespace, config: SimulationConfig) -> tuple[plt.Figure, dict[str, Slider]]:
    colors = stage_colors(monochrome=False)
    (left_x, left_y), (right_x, right_y) = default_player_position("left", config), default_player_position("right", config)
    (left_rec_x_bounds, left_rec_y_bounds) = recovery_bounds("left", config)
    vy_low = config.action.vy_min_forward
    vy_high = config.action.vy_max_forward

    x0_init = float(np.clip(args.x0, -config.court.half_width + 0.05, config.court.half_width - 0.05))
    y0_init = float(np.clip(args.y0, -config.court.half_length + 0.05, -0.05))
    z0_init = float(np.clip(args.z0, 0.0, config.render.z_max))
    vx_init = float(np.clip(args.vx_init, config.action.vx_min, config.action.vx_max))
    vy_init = float(np.clip(args.vy_init, vy_low, vy_high))
    vz_init = float(np.clip(args.vz_init, config.action.vz_min, config.action.vz_max))
    kh_init = float(max(args.kh_init, 0.0))
    kv_init = float(max(args.kv_init, 0.0))
    xrec_init = float(np.clip(args.x_rec_init, left_rec_x_bounds[0], left_rec_x_bounds[1]))
    yrec_init = float(np.clip(args.y_rec_init, left_rec_y_bounds[0], left_rec_y_bounds[1]))

    fig, (ax_top, ax_side) = plt.subplots(2, 1, figsize=(9.0, 12.0), gridspec_kw={"height_ratios": [1.15, 0.85]})
    fig.subplots_adjust(left=0.1, right=0.96, top=0.94, bottom=0.29, hspace=0.28)
    setup_court_axes(ax_top, config, colors, show_axes=True)
    ax_top.set_title("Trajectory Slider")
    ax_top.scatter([left_x, right_x], [left_y, right_y], c=[colors["left_player"], colors["right_player"]], s=90, zorder=3)

    ax_side.set_title("Side View")
    ax_side.set_xlabel("y (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.set_xlim(-config.court.half_length - 0.3, config.court.half_length + 0.3)
    ax_side.set_ylim(0.0, config.render.z_max)
    ax_side.grid(alpha=0.25)
    ax_side.axhline(0.0, color="0.25", linewidth=1.2)
    ax_side.axvline(config.court.net_y, color="0.35", linestyle="--", linewidth=1.0)
    ax_side.plot(
        [config.court.net_y, config.court.net_y],
        [0.0, config.court.net_height],
        color="0.2",
        linewidth=3.0,
        solid_capstyle="round",
    )

    (trajectory_line,) = ax_top.plot([], [], color="tab:blue", linewidth=2.6, label="ground path")
    (start_point,) = ax_top.plot([], [], marker="o", color="tab:red", markersize=7, label="start")
    (landing_point,) = ax_top.plot([], [], marker="o", color="tab:orange", markersize=7, label="landing")
    (net_point,) = ax_top.plot([], [], marker="o", color="tab:blue", markersize=7, label="net crossing")
    (recovery_point,) = ax_top.plot([], [], marker="o", color="tab:purple", markersize=7, alpha=0.7, label="recovery")

    (side_line,) = ax_side.plot([], [], color="tab:green", linewidth=2.4, label="y-z path")
    (side_start_point,) = ax_side.plot([], [], marker="o", color="tab:red", markersize=7)
    (side_landing_point,) = ax_side.plot([], [], marker="o", color="tab:orange", markersize=7)
    (side_net_point,) = ax_side.plot([], [], marker="o", color="tab:blue", markersize=7)

    info_text = ax_top.text(
        0.015,
        0.98,
        "",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4.0},
    )
    ax_top.legend(loc="lower right")

    slider_specs = [
        ("x0", -config.court.half_width + 0.05, config.court.half_width - 0.05, x0_init),
        ("y0", -config.court.half_length + 0.05, -0.05, y0_init),
        ("z0", 0.0, config.render.z_max, z0_init),
        ("v_x", config.action.vx_min, config.action.vx_max, vx_init),
        ("v_y", vy_low, vy_high, vy_init),
        ("v_z", config.action.vz_min, config.action.vz_max, vz_init),
        ("k_h", 0.0, 1.5, kh_init),
        ("k_v", 0.0, 1.5, kv_init),
        ("x_rec", left_rec_x_bounds[0], left_rec_x_bounds[1], xrec_init),
        ("y_rec", left_rec_y_bounds[0], left_rec_y_bounds[1], yrec_init),
    ]
    sliders: dict[str, Slider] = {}
    top = 0.26
    step = 0.03
    for index, (label, lower, upper, init) in enumerate(slider_specs):
        axis = fig.add_axes([0.18, top - index * step, 0.68, 0.022])
        sliders[label] = Slider(axis, label, lower, upper, valinit=init, valstep=0.01)

    def update(_: float) -> None:
        state = StageState(
            x_left=sliders["x0"].val,
            y_left=sliders["y0"].val,
            x_right=right_x,
            y_right=right_y,
            current_hitter="left",
            x0=sliders["x0"].val,
            y0=sliders["y0"].val,
            z0=sliders["z0"].val,
            stage_index=0,
        )
        action = ShotAction(
            v_x=sliders["v_x"].val,
            v_y=sliders["v_y"].val,
            v_z=sliders["v_z"].val,
            x_rec=sliders["x_rec"].val,
            y_rec=sliders["y_rec"].val,
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
        result = simulate_trajectory(state.x0, state.y0, state.z0, action.v_x, action.v_y, action.v_z, local_config)
        xs = np.asarray([point.x for point in result.samples], dtype=float)
        ys = np.asarray([point.y for point in result.samples], dtype=float)
        zs = np.asarray([point.z for point in result.samples], dtype=float)
        crossing = result.net_crossing
        legal = valid_hitter_action(state, action, local_config, result=result)
        color = "tab:blue" if legal else "tab:red"

        trajectory_line.set_data(xs, ys)
        trajectory_line.set_color(color)
        side_line.set_data(ys, zs)
        side_line.set_color(color)
        start_point.set_data([state.x0], [state.y0])
        landing_point.set_data([result.landing_x], [result.landing_y])
        recovery_point.set_data([action.x_rec], [action.y_rec])
        side_start_point.set_data([state.y0], [state.z0])
        side_landing_point.set_data([result.landing_y], [0.0])
        if crossing is not None:
            net_point.set_data([crossing.x], [crossing.y])
            net_point.set_color(color)
            side_net_point.set_data([crossing.y], [crossing.z])
            side_net_point.set_color(color)
        else:
            net_point.set_data([], [])
            side_net_point.set_data([], [])

        status = "legal" if legal else "invalid"
        z_net = crossing.z if crossing is not None else float("nan")
        info_text.set_text(
            "\n".join(
                [
                    f"mode = {local_config.action.trajectory_mode}",
                    f"start = ({state.x0:.2f}, {state.y0:.2f}, {state.z0:.2f})",
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
        f"v_x={sliders['v_x'].val:.2f}, "
        f"v_y={sliders['v_y'].val:.2f}, "
        f"v_z={sliders['v_z'].val:.2f}, "
        f"k_h={sliders['k_h'].val:.3f}, "
        f"k_v={sliders['k_v'].val:.3f}"
    )

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
