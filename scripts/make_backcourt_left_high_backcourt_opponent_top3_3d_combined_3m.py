from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs/rl/selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
PROBE_DIR = RUN_DIR / "anchor_metric_eval/controlled_contact_grid_probe"
TEMPLATE_PATH = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/source_data/make_combined_3d_plots.py"
MANIFEST_PATH = PROBE_DIR / "top_shot_3d_views/top_shot_trajectories_3d_manifest.json"
OUTPUT_PATH = (
    PROBE_DIR
    / "backcourt_left_high/opponent_default_position/"
    / "backcourt_left_high__opponent_backcourt_right_middle_left_top3_shot_trajectories_3d_combined.png"
)
PANEL_IDS = [
    "backcourt_left_high__opponent_backcourt_right",
    "backcourt_left_high__opponent_backcourt_middle",
    "backcourt_left_high__opponent_backcourt_left",
]


def main() -> None:
    template = _load_template()
    template.ensure_writable_matplotlib_config()

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    matplotlib.use("Agg")

    from scripts.plot_controlled_contact_top_shot_trajectories_3d import (
        _expand_scenarios_over_opponent_recovery_grid,
    )

    run_config = template.load_run_config(RUN_DIR)
    config = template.build_sim_config(run_config)
    probe_state = json.loads((PROBE_DIR / "controlled_contact_grid_probe_state.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    scenarios = _expand_scenarios_over_opponent_recovery_grid(
        list(probe_state["scenarios"]),
        "right",
        config,
    )
    scenario_by_id = {str(scenario["probe_id"]): scenario for scenario in scenarios}
    panels = [
        {
            "panel_id": panel_id,
            "scenario": scenario_by_id[panel_id],
            "top_shots": manifest["top_shots"][panel_id],
        }
        for panel_id in PANEL_IDS
    ]

    cmap = LinearSegmentedColormap.from_list(
        "probability_white_to_dark",
        [(1.0, 1.0, 1.0, 0.18), (0.62, 0.66, 0.68, 0.56), (0.02, 0.02, 0.02, 0.98)],
    )
    fig = template._new_figure()

    for panel, ax_pos, cbar_pos in zip(panels, template._panel_positions(), template._colorbar_positions()):
        scenario = panel["scenario"]
        state = template._stage_state_from_dict(scenario["response_state"])
        shots = []
        for row in panel["top_shots"]:
            trajectory = template.simulate_trajectory(
                state.x0,
                state.y0,
                state.z0,
                float(row["v_x"]),
                float(row["v_y"]),
                float(row["v_z"]),
                config,
            )
            shots.append({**row, "trajectory": trajectory})
        max_probability = max(float(row["probability"]) for row in shots)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=max_probability)

        ax = fig.add_axes(ax_pos, projection="3d")
        template._draw_court(ax, config)
        ax.scatter([state.x0], [state.y0], [state.z0], color="crimson", s=42, depthshade=False, zorder=8)
        ax.plot(
            [state.x0, state.x0],
            [state.y0, state.y0],
            [0.0, state.z0],
            color="crimson",
            linestyle=":",
            linewidth=0.85,
            zorder=8,
        )
        marker_xy = template._marker_from_scenario(scenario)
        if marker_xy is not None:
            ax.scatter([marker_xy[0]], [marker_xy[1]], [0.06], marker="s", color="royalblue", s=42, depthshade=False, zorder=8)

        for row in shots:
            xs = [point.x for point in row["trajectory"].samples]
            ys = [point.y for point in row["trajectory"].samples]
            zs = [point.z for point in row["trajectory"].samples]
            color = cmap(norm(float(row["probability"])))
            ax.plot(
                xs,
                ys,
                zs,
                color=color,
                linewidth=1.85 if int(row["rank"]) == 1 else 1.15,
                alpha=max(float(color[-1]), 0.56),
                zorder=7,
            )
            ax.scatter(
                [float(row["landing_x"])],
                [float(row["landing_y"])],
                [template.GROUND_MARKER_Z],
                marker="x",
                color=color,
                s=36,
                depthshade=False,
                zorder=8,
            )
        template._set_common_view(ax, config)
        ticks = _topshot_ticks_within_range(template, max_probability)
        template._add_top_colorbar(
            fig,
            cbar_pos,
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            "p",
            ticks,
            [template._format_probability_tick(value) for value in ticks],
        )

    handles = [
        Line2D([0], [0], color="0.12", lw=1.4, label="top shot trajectory"),
        Line2D([0], [0], marker="x", color="black", linestyle="none", markersize=6, label="shot landing"),
        Line2D([0], [0], marker="o", color="crimson", linestyle=":", lw=0.8, markersize=5, label="fixed contact"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="royalblue", markeredgecolor="royalblue", markersize=6, label="opponent marker"),
    ]
    template._add_right_legend(fig, handles)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=template.DPI, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    template._crop_top_to_colorbar_title(OUTPUT_PATH)
    print(OUTPUT_PATH)


def _load_template() -> Any:
    spec = importlib.util.spec_from_file_location("paper_combined_3d_template", TEMPLATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load template module from {TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _topshot_ticks_within_range(template: Any, max_probability: float) -> list[float]:
    ticks = [float(value) for value in template._topshot_ticks(float(max_probability)) if float(value) <= float(max_probability) + 1e-12]
    if max_probability <= 0.35:
        ticks = [value for value in (0.0, 0.1, 0.2, 0.3) if value <= float(max_probability) + 1e-12]
    if ticks and abs(ticks[-1] - float(max_probability)) < 0.025:
        ticks[-1] = float(max_probability)
    return ticks or [0.0, float(max_probability)]


if __name__ == "__main__":
    main()
