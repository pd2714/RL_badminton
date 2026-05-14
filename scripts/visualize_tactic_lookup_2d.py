from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib
import numpy as np

from badminton1d.config import ActionConfig, CourtConfig, SimulationConfig
from badminton1d.render import setup_court_axes, stage_colors
from badminton1d.shot_generators import TacticAction2D, TacticLookup2D, TacticRuntimeConfig
from badminton1d.state import ShotAction, StageState
from badminton1d.trajectory import simulate_trajectory
from badminton1d.utils import side_y_bounds, x_bounds


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
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect 2D tactic lookup trajectories with sliders.")
    parser.add_argument("--lookup-table-dir", type=Path, default=Path("lookup_tables"))
    parser.add_argument("--regenerate-lookup-table", action="store_true")
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="drag_square")
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def build_plot(args: argparse.Namespace, config: SimulationConfig) -> tuple[plt.Figure, dict[str, Slider]]:
    lookup = TacticLookup2D(
        config,
        TacticRuntimeConfig(
            regenerate_lookup_table=args.regenerate_lookup_table,
            lookup_dir=args.lookup_table_dir,
        ),
    )
    lookup.ensure_loaded()
    colors = stage_colors(monochrome=False)
    x_edges = np.linspace(x_bounds(config)[0], x_bounds(config)[1], 4)
    y_edges = np.linspace(side_y_bounds("right", config)[0], side_y_bounds("right", config)[1], 4)

    fig, (ax_top, ax_side) = plt.subplots(2, 1, figsize=(9.2, 12.2), gridspec_kw={"height_ratios": [1.1, 0.85]})
    fig.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.32, hspace=0.26)
    setup_court_axes(ax_top, config, colors, show_axes=True)
    ax_top.set_title("2D Tactic Lookup")

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

    target_patch = Rectangle(
        (x_edges[0], y_edges[0]),
        x_edges[1] - x_edges[0],
        y_edges[1] - y_edges[0],
        facecolor="gold",
        edgecolor="goldenrod",
        alpha=0.28,
    )
    ax_top.add_patch(target_patch)
    (top_line,) = ax_top.plot([], [], color="tab:blue", linewidth=2.6, label="ground path")
    (start_point,) = ax_top.plot([], [], marker="o", color="tab:red", markersize=7, label="start")
    (landing_point,) = ax_top.plot([], [], marker="o", color="tab:orange", markersize=7, label="landing")
    (target_point,) = ax_top.plot([], [], marker="x", color="goldenrod", markersize=9, label="target")
    (net_point,) = ax_top.plot([], [], marker="o", color="tab:blue", markersize=7, label="net crossing")

    (side_line,) = ax_side.plot([], [], color="tab:green", linewidth=2.4, label="trajectory")
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
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 4.0},
    )
    ax_top.legend(loc="lower right")

    slider_specs = [
        ("contact_x_bin", 0, 4, 2),
        ("contact_y_bin", 0, 4, 2),
        ("contact_height_bin", 0, 4, 2),
        ("landing_row", 0, 2, 1),
        ("landing_col", 0, 2, 1),
        ("angle_bin", 0, 4, 2),
        ("power_bin", 0, 2, 1),
    ]
    sliders: dict[str, Slider] = {}
    top = 0.27
    step = 0.033
    for index, (label, lower, upper, init) in enumerate(slider_specs):
        axis = fig.add_axes([0.18, top - index * step, 0.68, 0.022])
        sliders[label] = Slider(axis, label, lower, upper, valinit=init, valstep=1)

    def update(_: float) -> None:
        contact_x_bin = int(sliders["contact_x_bin"].val)
        contact_y_bin = int(sliders["contact_y_bin"].val)
        contact_height_bin = int(sliders["contact_height_bin"].val)
        landing_row = int(sliders["landing_row"].val)
        landing_col = int(sliders["landing_col"].val)
        angle_bin = int(sliders["angle_bin"].val)
        power_bin = int(sliders["power_bin"].val)

        state = StageState(
            x_left=float(lookup.contact_x_centers[contact_x_bin]),
            y_left=float(lookup.contact_y_centers[contact_y_bin]),
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=float(lookup.contact_x_centers[contact_x_bin]),
            y0=float(lookup.contact_y_centers[contact_y_bin]),
            z0=float(lookup.contact_height_centers[contact_height_bin]),
            stage_index=1,
        )
        action = TacticAction2D(
            landing_row=landing_row,
            landing_col=landing_col,
            angle_bin=angle_bin,
            power_bin=power_bin,
        )
        entry = lookup.lookup(state, action)
        shot = ShotAction(
            v_x=entry.velocity[0],
            v_y=entry.velocity[1],
            v_z=entry.velocity[2],
            x_rec=0.0,
            y_rec=state.y0,
        )
        result = simulate_trajectory(state.x0, state.y0, state.z0, shot.v_x, shot.v_y, shot.v_z, config)
        xs = np.asarray([point.x for point in result.samples], dtype=float)
        ys = np.asarray([point.y for point in result.samples], dtype=float)
        zs = np.asarray([point.z for point in result.samples], dtype=float)
        crossing = result.net_crossing
        color = "tab:blue" if entry.valid else "tab:red"

        target_patch.set_x(x_edges[landing_col])
        target_patch.set_y(y_edges[landing_row])
        target_patch.set_width(x_edges[landing_col + 1] - x_edges[landing_col])
        target_patch.set_height(y_edges[landing_row + 1] - y_edges[landing_row])
        top_line.set_data(xs, ys)
        top_line.set_color(color)
        side_line.set_data(ys, zs)
        side_line.set_color(color)
        start_point.set_data([state.x0], [state.y0])
        landing_point.set_data([result.landing_x], [result.landing_y])
        target_point.set_data([lookup.landing_col_centers[landing_col]], [lookup.landing_row_centers[landing_row]])
        side_start_point.set_data([state.y0], [state.z0])
        side_landing_point.set_data([result.landing_y], [0.0])
        if crossing is not None:
            net_point.set_data([crossing.x], [crossing.y])
            side_net_point.set_data([crossing.y], [crossing.z])
        else:
            net_point.set_data([], [])
            side_net_point.set_data([], [])

        info_text.set_text(
            "\n".join(
                [
                    f"valid={entry.valid} fallback={entry.fallback_used}",
                    f"shot={entry.inferred_shot_name}",
                    f"contact bins=(x:{contact_x_bin}, y:{contact_y_bin}, h:{contact_height_bin})",
                    f"zone={lookup.zone_names[action.landing_zone]} angle={lookup.angle_names[angle_bin]} power={lookup.power_names[power_bin]}",
                    f"velocity=(vx={entry.velocity[0]:.2f}, vy={entry.velocity[1]:.2f}, vz={entry.velocity[2]:.2f})",
                    f"target=({lookup.landing_col_centers[landing_col]:.2f}, {lookup.landing_row_centers[landing_row]:.2f})",
                    f"actual=({result.landing_x:.2f}, {result.landing_y:.2f}) flight={entry.flight_time:.2f}s",
                    f"score={entry.score:.2f} net z={'nan' if entry.net_crossing_height is None else f'{entry.net_crossing_height:.2f}'}",
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
        court=CourtConfig(mode="2d"),
        action=ActionConfig(trajectory_mode=args.trajectory_mode),
    )
    fig, _ = build_plot(args, config)
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=160, bbox_inches="tight")
        print(f"saved figure to {args.save_path}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
