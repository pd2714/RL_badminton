from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = (
    REPO_ROOT
    / "outputs/rl/selfplay_2d_recoverycfdefault_resp1_2m_heuristicbase_ent002_speed100_anchor100k_fullrec24_20260603"
    / "anchor_metric_eval_200r"
)
DEFAULT_OUTPUT = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/anchor_metric_winrate_rating_combined.png"
FONT_SCALE = 0.9 / 1.3 * 1.1


def _fs(size: float) -> float:
    return size * FONT_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the anchor win-rate/rating figure with a lag-curve inset.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--matrix-json", type=Path, default=None)
    parser.add_argument("--rating-csv", type=Path, default=None)
    parser.add_argument("--lag-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drop-zero-matrix", action="store_true")
    parser.add_argument("--rating-ylim", type=float, nargs=2, default=None)
    parser.add_argument("--inset-lag-max-million", type=float, default=2.0)
    return parser.parse_args()


def load_win_rate_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure_matplotlib(output_path: Path) -> None:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/private/tmp")) / "rl_badminton_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)


def _available(row: dict[str, str]) -> bool:
    return row.get("available", "True").strip().lower() not in {"0", "false", "no"}


def _plot_rating_panel(
    ax: Any,
    rating_rows: list[dict[str, str]],
    *,
    rating_ylim: list[float] | None,
) -> None:
    if "run_label" not in rating_rows[0]:
        rating_steps_m = np.asarray([float(row["step"]) / 1_000_000.0 for row in rating_rows], dtype=float)
        ratings = np.asarray([float(row["elo"]) for row in rating_rows], dtype=float)
        ax.plot(rating_steps_m, ratings, marker="o", linewidth=3.2, markersize=10.5, color="tab:blue")
        ax.set_ylim(1195, 1595)
        ax.set_yticks([1200, 1300, 1400, 1500])
        return

    old_rows = sorted(
        (row for row in rating_rows if row.get("run_label") == "old" and _available(row)),
        key=lambda row: int(row["step"]),
    )
    new_rows = sorted(
        (row for row in rating_rows if row.get("run_label") == "new" and _available(row)),
        key=lambda row: int(row["step"]),
    )
    if old_rows and new_rows:
        shared_prefix_step = 3_000_000
        new_rows = [row for row in old_rows if int(row["step"]) <= shared_prefix_step] + [
            row for row in new_rows if int(row["step"]) > shared_prefix_step
        ]

    series = [
        (old_rows, "#4c78a8", "pure recency"),
        (new_rows, "#f58518", "pure+linear recency"),
    ]
    plotted_ratings: list[float] = []
    for rows, color, legend_label in series:
        if not rows:
            continue
        x = [int(row["step"]) / 1_000_000.0 for row in rows]
        y = [float(row["elo"]) for row in rows]
        plotted_ratings.extend(y)
        ax.plot(x, y, marker="o", linewidth=3.2, markersize=7.2, color=color, label=legend_label)

    if rating_ylim is not None:
        ax.set_ylim(*rating_ylim)
    elif plotted_ratings:
        low = np.floor((min(plotted_ratings) - 25.0) / 100.0) * 100.0
        high = np.ceil((max(plotted_ratings) + 25.0) / 100.0) * 100.0
        ax.set_ylim(low, high)
    ax.legend(
        frameon=False,
        fontsize=_fs(18),
        loc="lower right",
        bbox_to_anchor=(0.98, 0.08),
        handlelength=2.4,
    )


def plot_figure(
    eval_dir: Path,
    output_path: Path,
    *,
    matrix_json: Path | None,
    rating_csv: Path | None,
    lag_csv: Path | None,
    drop_zero_matrix: bool,
    rating_ylim: list[float] | None,
    inset_lag_max_million: float,
) -> None:
    configure_matplotlib(output_path)
    import matplotlib.pyplot as plt

    win_report = load_win_rate_report(matrix_json or (eval_dir / "win_rate_matrix.json"))
    rating_rows = load_csv_rows(rating_csv or (eval_dir / "fixed_pool_ratings.csv"))
    lag_path = lag_csv or (eval_dir / "temporal_lag_curve.csv")
    lag_rows = load_csv_rows(lag_path) if lag_path.exists() else []

    matrix = np.asarray(win_report["win_rate_matrix"], dtype=float)
    row_steps = np.asarray(win_report["row_steps"], dtype=float)
    col_steps = np.asarray(win_report["col_steps"], dtype=float)
    if drop_zero_matrix:
        row_mask = row_steps != 0
        col_mask = col_steps != 0
        matrix = matrix[np.ix_(row_mask, col_mask)]
        row_steps = row_steps[row_mask]
        col_steps = col_steps[col_mask]
    row_steps_m = row_steps / 1_000_000.0
    col_steps_m = col_steps / 1_000_000.0

    if lag_rows:
        lag_steps_m = np.asarray([float(row["lag_million_steps"]) for row in lag_rows], dtype=float)
        lag_mean = np.asarray([float(row["mean_win_rate"]) for row in lag_rows], dtype=float)
        lag_low = np.asarray([float(row["bootstrap_ci_low"]) for row in lag_rows], dtype=float)
        lag_high = np.asarray([float(row["bootstrap_ci_high"]) for row in lag_rows], dtype=float)
        lag_mask = lag_steps_m <= inset_lag_max_million + 1e-9

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": _fs(33),
            "axes.titlesize": _fs(31),
            "xtick.labelsize": _fs(27),
            "ytick.labelsize": _fs(27),
            "axes.linewidth": 2.0,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.major.size": 9,
            "ytick.major.size": 9,
        }
    )

    fig = plt.figure(figsize=(1941 / 180, 1182 / 180), dpi=180)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.02, 1.0],
        left=0.095,
        right=0.985,
        bottom=0.165,
        top=0.79,
        wspace=0.42,
    )
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_rating = fig.add_subplot(grid[0, 1])

    dx = float(np.median(np.diff(col_steps_m)))
    dy = float(np.median(np.diff(row_steps_m)))
    extent = [
        float(col_steps_m[0] - dx / 2),
        float(col_steps_m[-1] + dx / 2),
        float(row_steps_m[-1] + dy / 2),
        float(row_steps_m[0] - dy / 2),
    ]
    image = ax_matrix.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdBu", aspect="equal", extent=extent)
    ax_matrix.plot([0.2, 6.0], [0.2, 6.0], color="0.55", linewidth=2.0, alpha=0.8)
    ax_matrix.set_xlabel("Opponent (M)")
    ax_matrix.set_ylabel("Checkpoint (M)")
    ax_matrix.set_xticks(np.arange(1, 7, 1))
    ax_matrix.set_yticks(np.arange(1, 7, 1))
    ax_matrix.set_xlim(0.1, 6.1)
    ax_matrix.set_ylim(6.1, 0.1)
    for spine in ax_matrix.spines.values():
        spine.set_linewidth(2.0)

    cax = ax_matrix.inset_axes([0.21, 1.045, 0.58, 0.075])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0.0, 0.5, 1.0])
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Win rate", labelpad=5, fontsize=_fs(31))
    cbar.ax.tick_params(labelsize=_fs(27), width=2.0, length=8)
    cbar.outline.set_linewidth(2.0)

    _plot_rating_panel(ax_rating, rating_rows, rating_ylim=rating_ylim)
    ax_rating.set_xlabel("Training (M)")
    ax_rating.set_ylabel("Elo")
    ax_rating.set_xlim(0.0, 6.2)
    ax_rating.set_xticks(np.arange(0, 7, 1))
    ax_rating.grid(True, color="0.75", linewidth=1.6, alpha=0.35)
    for spine in ax_rating.spines.values():
        spine.set_linewidth(2.0)

    if lag_rows:
        inset = ax_rating.inset_axes([0.54, 0.18, 0.43, 0.36])
        inset.axhline(0.5, color="0.35", linewidth=1.0, linestyle="--")
        inset.errorbar(
            lag_steps_m[lag_mask],
            lag_mean[lag_mask],
            yerr=np.vstack([lag_mean[lag_mask] - lag_low[lag_mask], lag_high[lag_mask] - lag_mean[lag_mask]]),
            marker="o",
            markersize=3.2,
            linewidth=1.4,
            capsize=2,
            color="tab:blue",
            ecolor="tab:blue",
        )
        inset.set_xlim(0.0, inset_lag_max_million)
        inset.set_ylim(0.47, 0.72)
        inset.set_xticks([0, 1, 2])
        inset.set_yticks([0.5, 0.6, 0.7])
        inset.set_xlabel("Lag (M)", fontsize=_fs(11), labelpad=0)
        inset.set_ylabel("Win rate", fontsize=_fs(11), labelpad=1)
        inset.tick_params(axis="both", labelsize=_fs(10), width=1.0, length=3)
        inset.grid(True, color="0.82", linewidth=0.8, alpha=0.55)
        inset.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_linewidth(1.0)

    label_box = dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="0.85", linewidth=1.5)
    ax_matrix.text(-0.12, 1.08, "A", transform=ax_matrix.transAxes, fontsize=_fs(33), fontweight="bold", bbox=label_box)
    ax_rating.text(-0.12, 1.08, "B", transform=ax_rating.transAxes, fontsize=_fs(33), fontweight="bold", bbox=label_box)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_figure(
        args.eval_dir,
        args.output,
        matrix_json=args.matrix_json,
        rating_csv=args.rating_csv,
        lag_csv=args.lag_csv,
        drop_zero_matrix=args.drop_zero_matrix,
        rating_ylim=args.rating_ylim,
        inset_lag_max_million=args.inset_lag_max_million,
    )
    print(args.output)


if __name__ == "__main__":
    main()
