from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib
import numpy as np

from badminton.config import CourtConfig, SimulationConfig
from badminton.dynamics import valid_hitter_action
from badminton.state import ShotAction, StageState
from badminton.trajectory import simulate_trajectory
from badminton.utils import recovery_bounds


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
    parser = argparse.ArgumentParser(description="Visualize 1D side-view shuttle trajectories with sliders.")
    parser.add_argument("--y0", type=float, default=-2.5, help="Shuttle start y position.")
    parser.add_argument("--z0", type=float, default=1.7, help="Shuttle start height.")
    parser.add_argument("--vy-init", type=float, default=5.5, dest="vy_init", help="Initial down-court velocity.")
    parser.add_argument("--vz-init", type=float, default=5.0, dest="vz_init", help="Initial vertical velocity.")
    parser.add_argument("--kh-init", type=float, default=0.2, dest="kh_init", help="Initial horizontal drag coefficient.")
    parser.add_argument("--kv-init", type=float, default=0.16, dest="kv_init", help="Initial vertical drag coefficient.")
    parser.add_argument("--y-rec-init", type=float, default=-2.0, dest="y_rec_init", help="Initial recovery target y.")
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="drag_square", help="Trajectory model to visualize.")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional output path for saving the current figure.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening an interactive window.")
    return parser.parse_args()


def build_plot(args: argparse.Namespace, config: SimulationConfig) -> tuple[plt.Figure, dict[str, Slider]]:
    (_, left_rec_y_bounds) = recovery_bounds("left", config)
    vy_low = config.action.vy_min_forward
    vy_high = config.action.vy_max_forward

    y0_init = float(np.clip(args.y0, -config.court.half_length + 0.05, -0.05))
    z0_init = float(np.clip(args.z0, 0.0, config.render.z_max))
    vy_init = float(np.clip(args.vy_init, vy_low, vy_high))
    vz_init = float(np.clip(args.vz_init, config.action.vz_min, config.action.vz_max))
    kh_init = float(max(args.kh_init, 0.0))
    kv_init = float(max(args.kv_init, 0.0))
    yrec_init = float(np.clip(args.y_rec_init, left_rec_y_bounds[0], left_rec_y_bounds[1]))

    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.3)
    ax.set_title("1D Trajectory Slider")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_xlim(-config.court.half_length - 0.3, config.court.half_length + 0.3)
    ax.set_ylim(0.0, config.render.z_max)
    ax.grid(alpha=0.25)
    ax.axhline(0.0, color="0.25", linewidth=1.2)
    ax.axvline(config.court.net_y, color="0.35", linestyle="--", linewidth=1.0)
    ax.plot(
        [config.court.net_y, config.court.net_y],
        [0.0, config.court.net_height],
        color="0.2",
        linewidth=3.0,
        solid_capstyle="round",
    )

    (side_line,) = ax.plot([], [], color="tab:green", linewidth=2.5, label="trajectory")
    (side_start_point,) = ax.plot([], [], marker="o", color="tab:red", markersize=7, label="start")
    (side_landing_point,) = ax.plot([], [], marker="o", color="tab:orange", markersize=7, label="landing")
    (side_net_point,) = ax.plot([], [], marker="o", color="tab:blue", markersize=7, label="net crossing")
    (recovery_marker,) = ax.plot([], [], marker="x", color="tab:purple", markersize=8, label="recovery y")
    info_text = ax.text(
        0.015,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4.0},
    )
    ax.legend(loc="lower right")

    slider_specs = [
        ("y0", -config.court.half_length + 0.05, -0.05, y0_init),
        ("z0", 0.0, config.render.z_max, z0_init),
        ("v_y", vy_low, vy_high, vy_init),
        ("v_z", config.action.vz_min, config.action.vz_max, vz_init),
        ("k_h", 0.0, 1.5, kh_init),
        ("k_v", 0.0, 1.5, kv_init),
        ("y_rec", left_rec_y_bounds[0], left_rec_y_bounds[1], yrec_init),
    ]
    sliders: dict[str, Slider] = {}
    top = 0.23
    step = 0.034
    for index, (label, lower, upper, init) in enumerate(slider_specs):
        axis = fig.add_axes([0.18, top - index * step, 0.68, 0.024])
        sliders[label] = Slider(axis, label, lower, upper, valinit=init, valstep=0.01)

    def update(_: float) -> None:
        state = StageState(
            x_left=0.0,
            y_left=sliders["y0"].val,
            x_right=0.0,
            y_right=2.5,
            current_hitter="left",
            x0=0.0,
            y0=sliders["y0"].val,
            z0=sliders["z0"].val,
            stage_index=0,
        )
        action = ShotAction(
            v_x=0.0,
            v_y=sliders["v_y"].val,
            v_z=sliders["v_z"].val,
            x_rec=0.0,
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
        ys = np.asarray([point.y for point in result.samples], dtype=float)
        zs = np.asarray([point.z for point in result.samples], dtype=float)
        crossing = result.net_crossing
        legal = valid_hitter_action(state, action, local_config, result=result)
        color = "tab:green" if legal else "tab:red"
        z_net = crossing.z if crossing is not None else float("nan")

        side_line.set_data(ys, zs)
        side_line.set_color(color)
        side_start_point.set_data([state.y0], [state.z0])
        side_landing_point.set_data([result.landing_y], [0.0])
        recovery_marker.set_data([action.y_rec], [0.0])
        if crossing is not None:
            side_net_point.set_data([crossing.y], [crossing.z])
            side_net_point.set_color(color)
        else:
            side_net_point.set_data([], [])

        info_text.set_text(
            "\n".join(
                [
                    f"mode = {local_config.action.trajectory_mode}",
                    f"start = ({state.y0:.2f}, {state.z0:.2f})",
                    f"launch = ({action.v_y:.2f}, {action.v_z:.2f}) m/s",
                    f"drag = (kh={sliders['k_h'].val:.3f}, kv={sliders['k_v'].val:.3f})",
                    f"landing = ({result.landing_y:.2f}, 0.00) in {result.landing_time:.2f}s",
                    f"recovery y = {action.y_rec:.2f}",
                    f"z(net) = {z_net:.2f} m",
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
    base_config = SimulationConfig(court=CourtConfig(mode="1d"))
    config = SimulationConfig(
        court=base_config.court,
        player=base_config.player,
        render=base_config.render,
        action=replace(
            base_config.action,
            trajectory_mode=args.trajectory_mode,
            drag_coefficient=args.kh_init,
            horizontal_drag_coefficient=args.kh_init,
            vertical_drag_coefficient=args.kv_init,
        ),
    )
    fig, sliders = build_plot(args, config)

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=160, bbox_inches="tight")
        print(f"saved figure to {args.save_path}")

    print(
        "1d trajectory sliders ready: "
        f"y0={sliders['y0'].val:.2f}, "
        f"z0={sliders['z0'].val:.2f}, "
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
