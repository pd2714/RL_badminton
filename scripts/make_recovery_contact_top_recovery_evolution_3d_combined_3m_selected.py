from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs/rl/selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
TEMPLATE_PATH = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/source_data/make_combined_3d_plots.py"
OUTPUT_FILENAME = "recovery_contact_top_recovery_evolution_3d_combined_3m_selected.png"

PANEL_SPECS = [
    (
        "recovery_contact_grid_probe_x0_0_yneg2_k0_latest",
        "frontcourt_left_low",
        "recovery_contact_grid_x0_0_yneg2_k0_probe_state.json",
        "recovery_contact_grid_x0_0_yneg2_k0_probe_bins.csv",
    ),
    (
        "recovery_contact_grid_probe_x0_0_yneg6_k0_latest",
        "frontcourt_right_low",
        "recovery_contact_grid_x0_0_yneg6_k0_probe_state.json",
        "recovery_contact_grid_x0_0_yneg6_k0_probe_bins.csv",
    ),
    (
        "backcourt_left_high_smash_recovery_comparison",
        "cross_positive_x_smash",
        "backcourt_left_high_smash_recovery_comparison_probe_state.json",
        "backcourt_left_high_smash_recovery_comparison_probe_bins.csv",
    ),
]


def main() -> None:
    template = _load_template()
    template.ensure_writable_matplotlib_config()

    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.use("Agg")

    config = template.build_sim_config(template.load_run_config(RUN_DIR))
    cmap = mpl.colormaps["plasma"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=6.0)
    fig = template._new_figure()

    panels = [_load_panel(probe_subdir, probe_id, state_name, bins_name) for probe_subdir, probe_id, state_name, bins_name in PANEL_SPECS]
    for panel, ax_pos, cbar_pos in zip(panels, template._panel_positions(), template._colorbar_positions()):
        ax = fig.add_axes(ax_pos, projection="3d")
        template._draw_court(ax, config)
        template._plot_recovery_panel(ax, config, panel["scenario"], panel["rows"], cmap, norm)
        template._set_common_view(ax, config)
        template._add_top_colorbar(
            fig,
            cbar_pos,
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            "checkpoint step (M)",
            [1, 3, 5],
            ["1", "3", "5"],
        )

    handles = [
        Line2D([0], [0], color="tab:blue", lw=1.35, label="fixed shot trajectory"),
        Line2D([0], [0], marker="o", color="tab:blue", linestyle="none", markersize=5, label="hitter contact"),
        Line2D([0], [0], marker="*", color="crimson", linestyle="none", markersize=10, label="opponent contact"),
        Line2D([0], [0], color="tab:red", linestyle="--", lw=1.0, label="likely opponent response"),
        Line2D([0], [0], marker="x", color="tab:red", linestyle="none", markersize=6, label="response landing"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#9c27b0", markeredgecolor="black", markersize=6, label="second recovery"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#facc15", markeredgecolor="black", markersize=6, label="top recovery"),
    ]
    template._add_right_legend(fig, handles)
    template._save(fig, OUTPUT_FILENAME)
    plt.close("all")
    print(template.FIGURE_DIR / OUTPUT_FILENAME)


def _load_panel(probe_subdir: str, probe_id: str, state_name: str, bins_name: str) -> dict[str, Any]:
    probe_dir = RUN_DIR / "anchor_metric_eval" / probe_subdir
    state = json.loads((probe_dir / state_name).read_text(encoding="utf-8"))
    scenarios = {str(scenario["probe_id"]): scenario for scenario in state["scenarios"]}
    if probe_id not in scenarios:
        raise KeyError(f"{probe_id!r} not found in {probe_dir / state_name}")

    rows = []
    with (probe_dir / bins_name).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("probe_id", "")) == probe_id:
                rows.append(row)
    if not rows:
        raise ValueError(f"No bin rows found for {probe_id!r} in {probe_dir / bins_name}")
    return {"probe_dir": str(probe_dir), "panel_id": probe_id, "scenario": scenarios[probe_id], "rows": rows}


def _load_template() -> Any:
    spec = importlib.util.spec_from_file_location("paper_combined_3d_template", TEMPLATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load template module from {TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
