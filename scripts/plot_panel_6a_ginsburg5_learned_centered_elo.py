from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/rl/ginsburg_20260622/eval"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures"
DEFAULT_SOURCE_DIR = DEFAULT_OUTPUT_DIR / "source_data"
DEFAULT_SEEDS = [17, 23, 31, 47, 59]
DEFAULT_EVAL_NAME = "panel_6a_learned_centered_recovery_200r"
COLORS = {"learned": "#1f77b4", "centered": "#ff7f0e"}
LABELS = {"learned": "learned", "centered": "centered"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot five-seed Figure 6A learned-vs-centered recovery Elo evolution."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--eval-name", type=str, default=DEFAULT_EVAL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, sources = load_seed_rows(args.input_root, args.eval_name, [int(seed) for seed in args.seeds])
    if not rows:
        raise SystemExit(f"No Elo rows found under {args.input_root}/*/{args.eval_name}")

    summary_rows = summarize_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = args.source_dir / "panel_6a_ginsburg5_learned_centered_elo_rows.csv"
    summary_csv = args.source_dir / "panel_6a_ginsburg5_learned_centered_elo_summary.csv"
    sources_csv = args.source_dir / "panel_6a_ginsburg5_learned_centered_sources.csv"
    mean_sem_png = args.output_dir / "panel_6a_ginsburg5_learned_centered_elo_mean_sem.png"
    seed_panels_png = args.output_dir / "panel_6a_ginsburg5_learned_centered_elo_seed_panels.png"
    overlay_png = args.output_dir / "panel_6a_ginsburg5_learned_centered_elo_overlay.png"

    write_csv(rows_csv, rows, ["seed", "agent", "step", "step_millions", "recovery_mode", "elo", "mean_pool_win_rate"])
    write_csv(
        summary_csv,
        summary_rows,
        ["recovery_mode", "step", "step_millions", "n", "mean_elo", "sd_elo", "sem_elo"],
    )
    write_csv(sources_csv, sources, ["seed", "elo_csv", "row_count"])

    plot_mean_sem(rows, summary_rows, mean_sem_png)
    plot_seed_panels(rows, seed_panels_png)
    plot_overlay(rows, overlay_png)

    print(f"rows: {rows_csv}")
    print(f"summary: {summary_csv}")
    print(f"sources: {sources_csv}")
    print(f"mean_sem: {mean_sem_png}")
    print(f"seed_panels: {seed_panels_png}")
    print(f"overlay: {overlay_png}")


def load_seed_rows(input_root: Path, eval_name: str, seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    missing: list[Path] = []
    for seed in seeds:
        csv_path = input_root / f"seed_{seed}" / eval_name / "elo_by_variant.csv"
        if not csv_path.exists():
            missing.append(csv_path)
            continue
        seed_rows = read_csv(csv_path)
        for row in seed_rows:
            rows.append(
                {
                    "seed": seed,
                    "agent": str(row["agent"]),
                    "step": int(row["step"]),
                    "step_millions": float(row["step_millions"]),
                    "recovery_mode": str(row["recovery_mode"]),
                    "elo": float(row["elo"]),
                    "mean_pool_win_rate": float(row["mean_pool_win_rate"])
                    if row.get("mean_pool_win_rate") not in (None, "")
                    else "",
                }
            )
        sources.append({"seed": seed, "elo_csv": str(csv_path), "row_count": len(seed_rows)})
    if missing:
        print("missing Elo CSVs:")
        for path in missing:
            print(path)
    return sorted(rows, key=lambda row: (int(row["seed"]), str(row["recovery_mode"]), int(row["step"]))), sources


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["recovery_mode"]), int(row["step"])), []).append(float(row["elo"]))
    summary: list[dict[str, Any]] = []
    for (mode, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(values)
        sd = stdev(values) if n > 1 else 0.0
        sem = sd / math.sqrt(n) if n > 1 else 0.0
        summary.append(
            {
                "recovery_mode": mode,
                "step": step,
                "step_millions": step / 1_000_000.0,
                "n": n,
                "mean_elo": mean(values),
                "sd_elo": sd,
                "sem_elo": sem,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_mean_sem(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in rows}):
        for mode in ("learned", "centered"):
            seed_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["recovery_mode"] == mode],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in seed_rows],
                [float(row["elo"]) for row in seed_rows],
                linewidth=1.0,
                alpha=0.18,
                color=COLORS[mode],
                zorder=1,
            )
    for mode in ("learned", "centered"):
        mode_rows = sorted(
            [row for row in summary_rows if row["recovery_mode"] == mode],
            key=lambda row: int(row["step"]),
        )
        x = [float(row["step_millions"]) for row in mode_rows]
        y = [float(row["mean_elo"]) for row in mode_rows]
        sem = [float(row["sem_elo"]) for row in mode_rows]
        ax.fill_between(
            x,
            [value - err for value, err in zip(y, sem)],
            [value + err for value, err in zip(y, sem)],
            color=COLORS[mode],
            alpha=0.16,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(x, y, marker="o", markersize=5.0, linewidth=2.5, color=COLORS[mode], label=LABELS[mode], zorder=3)
    format_axis(ax)
    ax.set_ylabel("Elo")
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_seed_panels(rows: list[dict[str, Any]], output_path: Path) -> None:
    configure_style()
    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.ravel())
    for ax, seed in zip(flat_axes, seeds):
        for mode in ("learned", "centered"):
            mode_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["recovery_mode"] == mode],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in mode_rows],
                [float(row["elo"]) for row in mode_rows],
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        ax.set_title(f"seed {seed}", fontsize=13)
        format_axis(ax)
    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Training step (M)")
    for ax in flat_axes[::3]:
        ax.set_ylabel("Elo")
    flat_axes[0].legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overlay(rows: list[dict[str, Any]], output_path: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    seeds = sorted({int(row["seed"]) for row in rows})
    label_seed = seeds[0] if seeds else None
    for seed in seeds:
        for mode in ("learned", "centered"):
            mode_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["recovery_mode"] == mode],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in mode_rows],
                [float(row["elo"]) for row in mode_rows],
                marker="o",
                markersize=3.4,
                linewidth=1.4,
                alpha=0.62,
                color=COLORS[mode],
                label=LABELS[mode] if seed == label_seed else None,
            )
    format_axis(ax)
    ax.set_ylabel("Elo")
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.0,
        }
    )


def format_axis(ax: Any) -> None:
    ax.set_xlabel("Training step (M)")
    ax.set_xlim(-0.2, 6.2)
    ax.set_xticks(range(0, 7))
    ax.grid(True, color="0.72", linewidth=0.8, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


if __name__ == "__main__":
    main()
