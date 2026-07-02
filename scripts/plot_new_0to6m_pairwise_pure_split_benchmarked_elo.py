from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_new_0to6m_pairwise_pure_split_elo import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_DIR, read_rows


PREFIX = "pure_split_0_to_6m_new_pairwise_benchmarked"
BRANCH_STEP = 3_000_000
COLORS = {"pure": "#d0693a", "split": "#2f7fba"}
FAMILY_LABELS = {"pure": "Pure CFA", "split": "Split CFA"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pure-CFA vs split-CFA 0-6M pairwise Elo after benchmarking "
            "split to the shared 0-3M prefix."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--branch-step", type=int, default=BRANCH_STEP)
    parser.add_argument(
        "--offset-mode",
        choices=("branch", "prefix-mean"),
        default="branch",
        help=(
            "How to shift split post-branch Elo. 'branch' joins split to pure at "
            "branch-step; 'prefix-mean' aligns the mean split/pure prefix offset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows, missing = read_rows(args.input_root, include_step_zero=False)
    if missing:
        print("missing: " + "; ".join(f"seed {seed} {family}" for seed, family in missing))
    if not raw_rows:
        raise SystemExit(f"No 0-6M pure/split Elo rows found under {args.input_root}")

    rows = benchmark_rows(raw_rows, branch_step=args.branch_step, offset_mode=args.offset_mode)
    summary_rows = summarize_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"{PREFIX}_elo_rows.csv"
    summary_path = args.output_dir / f"{PREFIX}_elo_summary.csv"
    mean_path = args.output_dir / f"{PREFIX}_elo_mean_sem.png"
    panel_path = args.output_dir / f"{PREFIX}_elo_seed_panels.png"
    overlay_path = args.output_dir / f"{PREFIX}_elo_overlay.png"

    write_rows(rows_path, rows)
    write_summary(summary_path, summary_rows)
    plot_mean_sem(rows, summary_rows, mean_path, branch_step=args.branch_step)
    plot_seed_panels(rows, panel_path, branch_step=args.branch_step)
    plot_overlay(rows, overlay_path, branch_step=args.branch_step)

    print(f"rows: {rows_path}")
    print(f"summary: {summary_path}")
    print(f"mean_sem: {mean_path}")
    print(f"seed_panels: {panel_path}")
    print(f"overlay: {overlay_path}")


def benchmark_rows(
    raw_rows: list[dict[str, Any]],
    *,
    branch_step: int,
    offset_mode: str,
) -> list[dict[str, Any]]:
    by_seed_family_step: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in raw_rows:
        by_seed_family_step[(int(row["seed"]), str(row["family"]), int(row["step"]))] = row

    rows: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in raw_rows})
    for seed in seeds:
        pure_steps = {
            int(row["step"]): row
            for row in raw_rows
            if int(row["seed"]) == seed and str(row["family"]) == "pure"
        }
        split_steps = {
            int(row["step"]): row
            for row in raw_rows
            if int(row["seed"]) == seed and str(row["family"]) == "split"
        }
        shared_prefix_steps = sorted(
            step
            for step in pure_steps.keys() & split_steps.keys()
            if step <= branch_step
        )
        if not shared_prefix_steps:
            continue
        offset = split_offset(
            pure_steps,
            split_steps,
            shared_prefix_steps,
            branch_step=branch_step,
            offset_mode=offset_mode,
        )

        for family, source_steps in (("pure", pure_steps), ("split", split_steps)):
            for step, row in sorted(source_steps.items()):
                raw_elo = float(row["elo"])
                if family == "pure":
                    benchmark_elo = raw_elo
                    applied_offset = 0.0
                    benchmark_note = "reference"
                elif step <= branch_step and step in pure_steps:
                    benchmark_elo = float(pure_steps[step]["elo"])
                    applied_offset = benchmark_elo - raw_elo
                    benchmark_note = "shared_prefix_replaced_by_pure"
                else:
                    benchmark_elo = raw_elo + offset
                    applied_offset = offset
                    benchmark_note = f"post_branch_{offset_mode}_offset"

                rows.append(
                    {
                        "seed": seed,
                        "family": family,
                        "family_label": FAMILY_LABELS[family],
                        "agent": row["agent"],
                        "step": step,
                        "step_millions": step / 1_000_000.0,
                        "raw_elo": raw_elo,
                        "benchmark_elo": benchmark_elo,
                        "applied_offset": applied_offset,
                        "benchmark_note": benchmark_note,
                        "source_path": row["source_path"],
                    }
                )
    return rows


def split_offset(
    pure_steps: dict[int, dict[str, Any]],
    split_steps: dict[int, dict[str, Any]],
    shared_prefix_steps: list[int],
    *,
    branch_step: int,
    offset_mode: str,
) -> float:
    if offset_mode == "branch" and branch_step in pure_steps and branch_step in split_steps:
        return float(pure_steps[branch_step]["elo"]) - float(split_steps[branch_step]["elo"])
    diffs = [
        float(pure_steps[step]["elo"]) - float(split_steps[step]["elo"])
        for step in shared_prefix_steps
    ]
    return mean(diffs)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), int(row["step"])), []).append(float(row["benchmark_elo"]))

    summary: list[dict[str, Any]] = []
    for (family, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(values)
        sd = stdev(values) if n > 1 else 0.0
        sem = sd / math.sqrt(n) if n > 1 else 0.0
        summary.append(
            {
                "family": family,
                "family_label": FAMILY_LABELS[family],
                "step": step,
                "step_millions": step / 1_000_000.0,
                "n": n,
                "mean_benchmark_elo": mean(values),
                "sd_benchmark_elo": sd,
                "sem_benchmark_elo": sem,
            }
        )
    return summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "family",
        "family_label",
        "agent",
        "step",
        "step_millions",
        "raw_elo",
        "benchmark_elo",
        "applied_offset",
        "benchmark_note",
        "source_path",
    ]
    write_dicts(path, rows, fieldnames)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "family",
        "family_label",
        "step",
        "step_millions",
        "n",
        "mean_benchmark_elo",
        "sd_benchmark_elo",
        "sem_benchmark_elo",
    ]
    write_dicts(path, rows, fieldnames)


def write_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_mean_sem(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    branch_step: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)

    for seed in sorted({int(row["seed"]) for row in rows}):
        for family in ("pure", "split"):
            seed_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            if not seed_rows:
                continue
            ax.plot(
                [float(row["step_millions"]) for row in seed_rows],
                [float(row["benchmark_elo"]) for row in seed_rows],
                linewidth=1.0,
                alpha=0.18,
                color=COLORS[family],
                zorder=1,
            )

    for family in ("pure", "split"):
        family_rows = sorted(
            [row for row in summary_rows if row["family"] == family],
            key=lambda row: int(row["step"]),
        )
        x = [float(row["step_millions"]) for row in family_rows]
        y = [float(row["mean_benchmark_elo"]) for row in family_rows]
        sem = [float(row["sem_benchmark_elo"]) for row in family_rows]
        ax.fill_between(
            x,
            [value - err for value, err in zip(y, sem)],
            [value + err for value, err in zip(y, sem)],
            color=COLORS[family],
            alpha=0.16,
            linewidth=0.0,
            zorder=2,
        )
        label = f"{FAMILY_LABELS[family]} (benchmarked)"
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.2,
            linewidth=2.4,
            color=COLORS[family],
            label=label,
            zorder=3,
        )

    ax.axvline(branch_step / 1_000_000.0, color="#333333", linewidth=1.0, alpha=0.28)
    ax.set_title("Benchmarked 0.2-6M pairwise Elo: shared 0-3M prefix")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Benchmarked Elo")
    ax.set_xlim(0.15, 6.05)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_seed_panels(rows: list[dict[str, Any]], output_path: Path, *, branch_step: int) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.2), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.ravel())

    for ax, seed in zip(flat_axes, seeds):
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in family_rows],
                [float(row["benchmark_elo"]) for row in family_rows],
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.axvline(branch_step / 1_000_000.0, color="#333333", linewidth=0.9, alpha=0.24)
        ax.set_title(f"seed {seed}", fontsize=14)
        ax.set_xlim(0.15, 6.05)
        ax.grid(True, alpha=0.28)

    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Checkpoint step (M)")
    for ax in flat_axes[::3]:
        ax.set_ylabel("Benchmarked Elo")
    flat_axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Benchmarked 0-6M pairwise Elo by seed")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overlay(rows: list[dict[str, Any]], output_path: Path, *, branch_step: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in rows}):
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in family_rows],
                [float(row["benchmark_elo"]) for row in family_rows],
                marker="o",
                markersize=3.4,
                linewidth=1.4,
                alpha=0.58,
                color=COLORS[family],
                label=FAMILY_LABELS[family] if seed == 17 else None,
            )
    ax.axvline(branch_step / 1_000_000.0, color="#333333", linewidth=1.0, alpha=0.28)
    ax.set_title("Benchmarked 0.2-6M pairwise Elo")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Benchmarked Elo")
    ax.set_xlim(0.15, 6.05)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
