from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "outputs/rl/ginsburg_20260622/same_june11_fixed_pool_eval_200r/"
    "split_vs_pure_cfa_aggregate/pure_split_same_fixed_pool_seed_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rl/ginsburg_20260622/eval/pure_split_0_to_6m_new_pairwise_plots"
)
PREFIX = "pure_split_0_to_6m_old_frozen_pool"
FAMILY_LABELS = {"pure": "Pure CFA", "split": "Split CFA"}
COLORS = {"pure": "#4c78a8", "split": "#f58518"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replot pure-CFA vs split-CFA Elo using the old frozen fixed-pool "
            "diagnostic rows instead of within-run pairwise Elo."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"{PREFIX}_elo_rows.csv"
    summary_path = args.output_dir / f"{PREFIX}_elo_summary.csv"
    mean_path = args.output_dir / f"{PREFIX}_elo_mean_sem.png"
    panel_path = args.output_dir / f"{PREFIX}_elo_seed_panels.png"
    overlay_path = args.output_dir / f"{PREFIX}_elo_overlay.png"

    summary_rows = summarize(rows)
    write_rows(rows_path, rows)
    write_summary(summary_path, summary_rows)
    plot_mean_sem(summary_rows, mean_path)
    plot_seed_panels(rows, panel_path)
    plot_overlay(rows, overlay_path)

    print(f"rows: {rows_path}")
    print(f"summary: {summary_path}")
    print(f"mean_sem: {mean_path}")
    print(f"seed_panels: {panel_path}")
    print(f"overlay: {overlay_path}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            family = str(row["family"])
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "family": family,
                    "family_label": FAMILY_LABELS.get(family, family),
                    "label": row["label"],
                    "step": int(row["step"]),
                    "step_millions": float(row["step_millions"]),
                    "elo": float(row["elo"]),
                    "mean_pool_win_rate": float(row["mean_pool_win_rate"]),
                    "evaluated_pair_count": int(row["evaluated_pair_count"]),
                    "source": row.get("source", str(path)),
                }
            )
    return sorted(rows, key=lambda row: (row["seed"], row["family"], row["step"]))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), int(row["step"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (family, step), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        elos = [float(row["elo"]) for row in group]
        win_rates = [float(row["mean_pool_win_rate"]) for row in group]
        n = len(group)
        sd = stdev(elos) if n > 1 else 0.0
        sem = sd / math.sqrt(n) if n > 1 else 0.0
        summary_rows.append(
            {
                "family": family,
                "family_label": FAMILY_LABELS.get(family, family),
                "step": step,
                "step_millions": step / 1_000_000.0,
                "n": n,
                "mean_elo": mean(elos),
                "sd_elo": sd,
                "sem_elo": sem,
                "mean_pool_win_rate": mean(win_rates),
            }
        )
    return summary_rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "family",
        "family_label",
        "label",
        "step",
        "step_millions",
        "elo",
        "mean_pool_win_rate",
        "evaluated_pair_count",
        "source",
    ]
    write_dicts(path, rows, fieldnames)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "family",
        "family_label",
        "step",
        "step_millions",
        "n",
        "mean_elo",
        "sd_elo",
        "sem_elo",
        "mean_pool_win_rate",
    ]
    write_dicts(path, rows, fieldnames)


def write_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_mean_sem(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for family in ("pure", "split"):
        rows = [row for row in summary_rows if row["family"] == family]
        x = [float(row["step_millions"]) for row in rows]
        y = [float(row["mean_elo"]) for row in rows]
        sem = [float(row["sem_elo"]) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.2, markersize=4.2, color=COLORS[family], label=FAMILY_LABELS[family])
        ax.fill_between(x, [a - b for a, b in zip(y, sem)], [a + b for a, b in zip(y, sem)], color=COLORS[family], alpha=0.16)
    ax.set_title("Old frozen-pool Elo: mean +/- SEM")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_seed_panels(rows: list[dict[str, Any]], output_path: Path) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.ravel())
    for ax, seed in zip(flat_axes, seeds):
        for family in ("pure", "split"):
            family_rows = [row for row in rows if row["seed"] == seed and row["family"] == family]
            x = [float(row["step_millions"]) for row in family_rows]
            y = [float(row["elo"]) for row in family_rows]
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=3.8, color=COLORS[family], label=FAMILY_LABELS[family])
        ax.set_title(f"seed {seed}")
        ax.set_xlim(left=0.0)
        ax.grid(True, alpha=0.28)
    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Checkpoint step (M)")
    for ax in flat_axes[::3]:
        ax.set_ylabel("Elo")
    flat_axes[0].legend(frameon=False)
    fig.suptitle("Old frozen-pool Elo: pure CFA vs split CFA")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overlay(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in rows}):
        for family in ("pure", "split"):
            family_rows = [row for row in rows if row["seed"] == seed and row["family"] == family]
            x = [float(row["step_millions"]) for row in family_rows]
            y = [float(row["elo"]) for row in family_rows]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=3.6,
                alpha=0.62,
                color=COLORS[family],
                label=FAMILY_LABELS[family] if seed == 17 else None,
            )
    ax.set_title("Old frozen-pool Elo")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
