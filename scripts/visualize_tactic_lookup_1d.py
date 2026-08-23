from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib
import numpy as np

from badminton.config import ActionConfig, CourtConfig, SimulationConfig
from badminton.shot_generators import TacticAction1D, TacticLookup1D, TacticRuntimeConfig
from badminton.shot_generators.tactic_lookup_common import ANGLE_BIN_COUNT_1D, angle_bin_centers_deg_1d
from badminton.state import ShotAction, StageState
from badminton.trajectory import simulate_trajectory
from badminton.utils import side_y_bounds


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
    parser = argparse.ArgumentParser(description="Inspect 1D tactic lookup trajectories with sliders.")
    parser.add_argument("--lookup-table-dir", type=Path, default=Path("lookup_tables"))
    parser.add_argument("--regenerate-lookup-table", action="store_true")
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="drag_square")
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def build_plot(args: argparse.Namespace, config: SimulationConfig) -> tuple[plt.Figure, dict[str, Slider]]:
    lookup = TacticLookup1D(
        config,
        TacticRuntimeConfig(
            regenerate_lookup_table=args.regenerate_lookup_table,
            lookup_dir=args.lookup_table_dir,
        ),
    )
    lookup.ensure_loaded()
    zone_edges = np.linspace(
        side_y_bounds("right", config)[0],
        side_y_bounds("right", config)[1],
        len(lookup.zone_names) + 1,
    )

    fig, ax = plt.subplots(figsize=(9.0, 8.8))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.93, bottom=0.32)
    ax.set_title("1D Tactic Lookup")
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

    target_patch = Rectangle((zone_edges[0], 0.0), zone_edges[1] - zone_edges[0], 0.18, color="gold", alpha=0.35)
    ax.add_patch(target_patch)
    (trajectory_line,) = ax.plot([], [], color="tab:green", linewidth=2.5, label="trajectory")
    (start_point,) = ax.plot([], [], marker="o", color="tab:red", markersize=7, label="start")
    (landing_point,) = ax.plot([], [], marker="o", color="tab:orange", markersize=7, label="landing")
    (target_point,) = ax.plot([], [], marker="x", color="goldenrod", markersize=9, label="target")
    (net_point,) = ax.plot([], [], marker="o", color="tab:blue", markersize=7, label="net crossing")
    info_text = ax.text(
        0.015,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 4.0},
    )
    ax.legend(loc="lower right")

    slider_specs = [
        ("contact_y_bin", 0, 4, 2),
        ("contact_height_bin", 0, 4, 2),
        ("landing_zone", 0, len(lookup.zone_names) - 1, len(lookup.zone_names) // 2),
        ("angle_bin", 0, ANGLE_BIN_COUNT_1D - 1, ANGLE_BIN_COUNT_1D // 2),
        ("power_bin", 0, len(lookup.power_names) - 1, len(lookup.power_names) // 2),
    ]
    sliders: dict[str, Slider] = {}
    top = 0.24
    step = 0.04
    for index, (label, lower, upper, init) in enumerate(slider_specs):
        axis = fig.add_axes([0.18, top - index * step, 0.68, 0.025])
        sliders[label] = Slider(axis, label, lower, upper, valinit=init, valstep=1)

    def update(_: float) -> None:
        contact_y_bin = int(sliders["contact_y_bin"].val)
        contact_height_bin = int(sliders["contact_height_bin"].val)
        landing_zone = int(sliders["landing_zone"].val)
        angle_bin = int(sliders["angle_bin"].val)
        power_bin = int(sliders["power_bin"].val)

        state = StageState(
            x_left=0.0,
            y_left=float(lookup.contact_y_centers[contact_y_bin]),
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=float(lookup.contact_y_centers[contact_y_bin]),
            z0=float(lookup.contact_height_centers[contact_height_bin]),
            stage_index=1,
        )
        action = TacticAction1D(
            landing_zone=landing_zone,
            angle_bin=angle_bin,
            power_bin=power_bin,
        )
        entry = lookup.lookup(state, action)
        shot = ShotAction(v_x=0.0, v_y=entry.velocity[0], v_z=entry.velocity[1], x_rec=0.0, y_rec=state.y0)
        result = simulate_trajectory(state.x0, state.y0, state.z0, shot.v_x, shot.v_y, shot.v_z, config)
        ys = np.asarray([point.y for point in result.samples], dtype=float)
        zs = np.asarray([point.z for point in result.samples], dtype=float)
        crossing = result.net_crossing
        target_angle_deg = float(
            angle_bin_centers_deg_1d(
                state.y0,
                state.z0,
                config,
                bins=ANGLE_BIN_COUNT_1D,
            )[angle_bin]
        )
        realized_angle_deg = float(np.degrees(np.arctan2(entry.velocity[1], max(abs(entry.velocity[0]), 1e-6))))

        target_patch.set_x(zone_edges[landing_zone])
        target_patch.set_width(zone_edges[landing_zone + 1] - zone_edges[landing_zone])
        trajectory_line.set_data(ys, zs)
        trajectory_line.set_color("tab:green" if entry.valid else "tab:red")
        start_point.set_data([state.y0], [state.z0])
        landing_point.set_data([result.landing_y], [0.0])
        target_point.set_data([lookup.landing_zone_centers[landing_zone]], [0.0])
        if crossing is not None:
            net_point.set_data([crossing.y], [crossing.z])
        else:
            net_point.set_data([], [])

        info_text.set_text(
            "\n".join(
                [
                    f"valid={entry.valid} fallback={entry.fallback_used}",
                    f"shot={entry.inferred_shot_name}",
                    f"contact bins=(y:{contact_y_bin}, h:{contact_height_bin})",
                    f"zone={lookup.zone_names[landing_zone]} angle={lookup.angle_names[angle_bin]} power={lookup.power_names[power_bin]}",
                    f"target angle={target_angle_deg:.1f} deg realized angle={realized_angle_deg:.1f} deg",
                    f"velocity=(vy={entry.velocity[0]:.2f}, vz={entry.velocity[1]:.2f})",
                    f"target y={lookup.landing_zone_centers[landing_zone]:.2f} actual y={result.landing_y:.2f}",
                    f"flight={entry.flight_time:.2f}s score={entry.score:.2f}",
                    f"net z={'nan' if entry.net_crossing_height is None else f'{entry.net_crossing_height:.2f}'}",
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
        court=CourtConfig(mode="1d"),
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
