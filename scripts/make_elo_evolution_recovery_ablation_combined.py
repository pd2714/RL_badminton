from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROSS_RUN_ELO_CSV = (
    REPO_ROOT / "outputs/rl/cross_run_fixed_pool_0p4m_to_3p2m_200r_20260611/elo_ratings.csv"
)
DEFAULT_RECOVERY_ABLATION_ELO_CSV = (
    REPO_ROOT
    / "outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
    / "recovery_ablation_fixed_pool_learned_centered/elo_by_variant.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/elo_evolution_recovery_ablation_combined.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the two-panel Elo evolution/recovery ablation figure.")
    parser.add_argument("--cross-run-elo-csv", type=Path, default=DEFAULT_CROSS_RUN_ELO_CSV)
    parser.add_argument("--recovery-ablation-elo-csv", type=Path, default=DEFAULT_RECOVERY_ABLATION_ELO_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/private/tmp")) / "rl_badminton_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_by_key(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(sorted(grouped.items()))


def _cross_run_label(run_label: str) -> str:
    return {
        "norecoverycfadv": "No CRA",
        "recoverycfdefault": "CRA",
    }.get(run_label, run_label)


def _rounded_ylim(values: list[float]) -> tuple[float, float]:
    low = math.floor(min(values) / 50.0) * 50.0
    high = math.ceil(max(values) / 50.0) * 50.0
    if high <= max(values):
        high += 50.0
    if low >= min(values):
        low -= 50.0
    return low, high


def _plot_cross_run_panel(ax: Any, rows: list[dict[str, str]]) -> list[float]:
    colors = {
        "norecoverycfadv": "#2f7fba",
        "recoverycfdefault": "#d0693a",
    }
    plotted: list[float] = []
    for run_label, run_rows in _rows_by_key(rows, "run_label").items():
        run_rows = sorted(run_rows, key=lambda row: float(row["step_millions"]))
        x = [float(row["step_millions"]) for row in run_rows]
        y = [float(row["elo"]) for row in run_rows]
        plotted.extend(y)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4,
            markersize=5.8,
            color=colors.get(run_label),
            label=_cross_run_label(run_label),
        )

    ax.set_xlabel("Training step (M)")
    ax.set_xlim(0.25, 3.35)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    return plotted


def _plot_recovery_ablation_panel(ax: Any, rows: list[dict[str, str]]) -> list[float]:
    colors = {
        "learned": "#1f77b4",
        "centered": "#ff7f0e",
        "heuristic": "#2ca02c",
    }
    order = ["learned", "centered", "heuristic"]
    grouped = _rows_by_key(rows, "recovery_mode")
    plotted: list[float] = []
    for mode in [mode for mode in order if mode in grouped]:
        mode_rows = sorted(grouped[mode], key=lambda row: int(row["step"]))
        x = [float(row["step_millions"]) for row in mode_rows]
        y = [float(row["elo"]) for row in mode_rows]
        plotted.extend(y)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4,
            markersize=5.8,
            color=colors.get(mode),
            label=mode,
        )

    ax.set_xlabel("Training step (M)")
    ax.set_xlim(-0.25, 6.25)
    ax.set_xticks(range(0, 7))
    return plotted


def plot_figure(cross_run_elo_csv: Path, recovery_ablation_elo_csv: Path, output: Path) -> None:
    configure_matplotlib()

    import matplotlib.pyplot as plt

    cross_run_rows = load_csv_rows(cross_run_elo_csv)
    recovery_ablation_rows = load_csv_rows(recovery_ablation_elo_csv)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 35,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 26,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.25), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.2, top=0.97, wspace=0.03)
    all_elos = []
    all_elos.extend(_plot_recovery_ablation_panel(axes[0], recovery_ablation_rows))
    all_elos.extend(_plot_cross_run_panel(axes[1], cross_run_rows))

    y_low, y_high = _rounded_ylim(all_elos)
    for ax in axes:
        ax.set_ylim(y_low, y_high)
        ax.set_yticks(range(int(y_low), int(y_high) + 1, 100))
        ax.grid(True, color="0.72", linewidth=0.8, alpha=0.35)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.075, 1.0), handlelength=1.25, borderaxespad=0.0)
        ax.set_box_aspect(0.72)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    axes[0].set_ylabel("Elo")
    label_box = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="0.82", linewidth=0.8)
    axes[0].text(0.025, 0.965, "A", transform=axes[0].transAxes, fontsize=39, fontweight="bold", va="top", bbox=label_box)
    axes[1].text(0.025, 0.965, "B", transform=axes[1].transAxes, fontsize=39, fontweight="bold", va="top", bbox=label_box)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_figure(args.cross_run_elo_csv, args.recovery_ablation_elo_csv, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
