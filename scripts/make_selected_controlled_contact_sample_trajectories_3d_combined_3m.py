from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs/rl/selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
PROBE_DIR = RUN_DIR / "anchor_metric_eval/controlled_contact_grid_probe"
TEMPLATE_PATH = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/source_data/make_combined_3d_plots.py"
SAMPLES_CSV = PROBE_DIR / "top3_expectation_evolution_probe_views/top3_expectation_evolution_samples.csv"
OUTPUT_PATH = PROBE_DIR / "selected_controlled_contact_sample_trajectories_3d_combined.png"
TRAJECTORY_LINEWIDTH = 1.0
TRAJECTORY_ALPHA = 0.42

PANEL_IDS = [
    "frontcourt_left_low__opponent_frontcourt_left",
    "frontcourt_right_low__opponent_frontcourt_middle",
    "backcourt_left_high__opponent_midcourt_left",
]


def main() -> None:
    template = _load_template()
    template.ensure_writable_matplotlib_config()

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    from scripts.plot_controlled_contact_sample_trajectories_3d import _scenario_lookup

    matplotlib.use("Agg")

    run_config = template.load_run_config(RUN_DIR)
    config = template.build_sim_config(run_config)
    scenarios = _scenario_lookup(PROBE_DIR, str(run_config.get("train_side", "left")), config)
    panels = [_load_panel(panel_id, scenarios) for panel_id in PANEL_IDS]

    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=6_000_000.0)
    fig = template._new_figure()

    for panel, ax_pos, cbar_pos in zip(panels, template._panel_positions(), template._colorbar_positions()):
        rows = panel["rows"]
        trajectories = [template._trajectory_from_row(row, config) for row in rows]
        steps = np.asarray([int(row["step"]) for row in rows], dtype=float)

        ax = fig.add_axes(ax_pos, projection="3d")
        template._draw_court(ax, config)

        segments = [np.column_stack((xs, ys, zs)) for xs, ys, zs in trajectories]
        collection = Line3DCollection(
            segments,
            colors=[cmap(norm(step)) for step in steps],
            linewidths=TRAJECTORY_LINEWIDTH,
            alpha=TRAJECTORY_ALPHA,
        )
        ax.add_collection3d(collection, autolim=False)
        _draw_left_net_crossing_overlays(ax, trajectories, steps, cmap, norm, config)

        ax.scatter(
            [float(row["landing_x"]) for row in rows],
            [float(row["landing_y"]) for row in rows],
            [template.GROUND_MARKER_Z] * len(rows),
            c=steps,
            cmap=cmap,
            norm=norm,
            s=14,
            alpha=0.62,
            linewidths=0.0,
            depthshade=False,
            zorder=7,
        )

        contact = rows[-1]
        contact_xyz = (
            float(contact["contact_x"]),
            float(contact["contact_y"]),
            float(contact["contact_z"]),
        )
        ax.scatter(
            [contact_xyz[0]],
            [contact_xyz[1]],
            [contact_xyz[2]],
            marker="*",
            s=145,
            color="crimson",
            edgecolors="white",
            linewidths=0.5,
            depthshade=False,
            zorder=9,
        )
        ax.plot(
            [contact_xyz[0], contact_xyz[0]],
            [contact_xyz[1], contact_xyz[1]],
            [template.GROUND_MARKER_Z, contact_xyz[2]],
            color="crimson",
            linestyle=":",
            linewidth=0.85,
            zorder=8,
        )

        marker_xy = template._marker_from_scenario(panel["scenario"])
        if marker_xy is not None:
            ax.scatter(
                [marker_xy[0]],
                [marker_xy[1]],
                [0.06],
                marker="s",
                color="royalblue",
                s=42,
                depthshade=False,
                zorder=8,
            )

        template._set_common_view(ax, config)
        template._add_top_colorbar(
            fig,
            cbar_pos,
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            "checkpoint step (M)",
            [1e6, 3e6, 5e6],
            ["1", "3", "5"],
        )

    handles = [
        Line2D([0], [0], color=cmap(norm(5e6)), lw=1.2, alpha=0.85, label="sample trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap(norm(5e6)),
            markeredgecolor="none",
            markersize=6,
            label="chosen landing",
        ),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="crimson", markeredgecolor="crimson", markersize=10, label="fixed contact"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="royalblue", markeredgecolor="royalblue", markersize=6, label="opponent marker"),
    ]
    template._add_right_legend(fig, handles)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=template.DPI, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    template._crop_top_to_colorbar_title(OUTPUT_PATH)
    print(OUTPUT_PATH)


def _draw_left_net_crossing_overlays(
    ax: Any,
    trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    steps: np.ndarray,
    cmap: Any,
    norm: Any,
    config: Any,
) -> None:
    net_y = float(config.court.net_y)
    net_height = float(config.court.net_height)
    for (xs, ys, zs), step in zip(trajectories, steps):
        for index in _net_crossing_segment_indices(ys, net_y):
            dy = float(ys[index + 1] - ys[index])
            if abs(dy) < 1e-12:
                continue
            t = float((net_y - ys[index]) / dy)
            x_cross = float(xs[index] + t * (xs[index + 1] - xs[index]))
            z_cross = float(zs[index] + t * (zs[index + 1] - zs[index]))
            if x_cross >= 0.0 or z_cross < net_height:
                continue

            start = max(0, index - 1)
            end = min(len(xs), index + 3)
            overlay_xs = np.concatenate((xs[start : index + 1], np.asarray([x_cross]), xs[index + 1 : end]))
            overlay_ys = np.concatenate((ys[start : index + 1], np.asarray([net_y]), ys[index + 1 : end]))
            overlay_zs = np.concatenate((zs[start : index + 1], np.asarray([z_cross]), zs[index + 1 : end]))
            ax.plot(
                overlay_xs,
                overlay_ys,
                overlay_zs,
                color=cmap(norm(step)),
                linewidth=TRAJECTORY_LINEWIDTH,
                alpha=TRAJECTORY_ALPHA,
                zorder=10,
            )


def _net_crossing_segment_indices(ys: np.ndarray, net_y: float) -> list[int]:
    signs = ys - net_y
    return [
        index
        for index in range(len(ys) - 1)
        if (signs[index] <= 0.0 <= signs[index + 1]) or (signs[index + 1] <= 0.0 <= signs[index])
    ]


def _load_panel(panel_id: str, scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    with SAMPLES_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("probe_id") == panel_id and _is_true(row.get("valid")) and int(row.get("rank") or 0) == 1:
                rows.append(row)
    rows = _dedupe_rows_per_checkpoint(rows)
    if not rows:
        raise ValueError(f"No valid rank-1 rows found for {panel_id!r} in {SAMPLES_CSV}")
    if panel_id not in scenarios:
        raise KeyError(f"{panel_id!r} not found in expanded controlled-contact scenarios")
    return {"panel_id": panel_id, "scenario": scenarios[panel_id], "rows": rows}


def _dedupe_rows_per_checkpoint(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_by_checkpoint: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["step"]), str(row.get("checkpoint_path") or ""))
        current = best_by_checkpoint.get(key)
        if current is None or int(row.get("sample_index", 0)) < int(current.get("sample_index", 0)):
            best_by_checkpoint[key] = row
    return sorted(best_by_checkpoint.values(), key=lambda row: (int(row["step"]), int(row.get("sample_index", 0))))


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_template() -> Any:
    spec = importlib.util.spec_from_file_location("paper_combined_3d_template", TEMPLATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load template module from {TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
