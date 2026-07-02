from __future__ import annotations

import argparse
import csv
import json
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

from badminton1d.elo import PairwiseRecord, calculate_elo


DEFAULT_INPUT_ROOT = Path("outputs/rl/ginsburg_20260622/eval")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "pure_split_0_to_6m_new_pairwise_plots"

FAMILY_SOURCES = {
    "pure": ("cfa_purerecency_0_to_6m_pool_elo_200r", "Pure CFA"),
    "split": ("cfa_splitlinear_0_to_6m_pool_elo_200r", "Split CFA"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pure-CFA vs split-CFA 0-6M Elo from new pairwise pool ratings."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-step-zero",
        action="store_true",
        help="Keep the step-0 base checkpoint in the Elo fit and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, missing = read_rows(args.input_root, include_step_zero=bool(args.include_step_zero))
    if not rows:
        raise SystemExit(f"No 0-6M pure/split Elo rows found under {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pure_split_0_to_6m_new_pairwise_elo_rows.csv"
    summary_path = args.output_dir / "pure_split_0_to_6m_new_pairwise_elo_summary.csv"
    mean_path = args.output_dir / "pure_split_0_to_6m_new_pairwise_elo_mean_sem.png"
    panel_path = args.output_dir / "pure_split_0_to_6m_new_pairwise_elo_seed_panels.png"
    overlay_path = args.output_dir / "pure_split_0_to_6m_new_pairwise_elo_overlay.png"
    summary_rows = summarize_rows(rows)
    write_rows(csv_path, rows)
    write_summary(summary_path, summary_rows)
    plot_mean_sem(rows, summary_rows, mean_path)
    plot_seed_panels(rows, missing, panel_path)
    plot_overlay(rows, overlay_path)

    print(f"rows: {csv_path}")
    print(f"summary: {summary_path}")
    print(f"mean_sem: {mean_path}")
    print(f"seed_panels: {panel_path}")
    print(f"overlay: {overlay_path}")
    for seed in sorted({int(row["seed"]) for row in rows}):
        available = sorted({str(row["family"]) for row in rows if int(row["seed"]) == seed})
        print(f"seed {seed}: available={','.join(available)}")
    if missing:
        print("missing: " + "; ".join(f"seed {seed} {family}" for seed, family in missing))


def read_rows(input_root: Path, *, include_step_zero: bool = False) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    rows: list[dict[str, Any]] = []
    missing: list[tuple[int, str]] = []
    seeds = sorted(
        int(path.name.replace("seed_", ""))
        for path in input_root.glob("seed_*")
        if path.is_dir() and path.name.replace("seed_", "").isdigit()
    )
    for seed in seeds:
        for family, (dir_name, label) in FAMILY_SOURCES.items():
            output_dir = input_root / f"seed_{seed}" / dir_name
            report_path = output_dir / "elo_rating_report.json"
            csv_path = output_dir / "elo_ratings.csv"
            if not report_path.exists() and not csv_path.exists():
                missing.append((seed, label))
                continue
            if report_path.exists():
                rows.extend(
                    rows_from_report(
                        report_path,
                        seed=seed,
                        family=family,
                        family_label=label,
                        include_step_zero=include_step_zero,
                    )
                )
            else:
                rows.extend(
                    rows_from_csv(
                        csv_path,
                        seed=seed,
                        family=family,
                        family_label=label,
                        include_step_zero=include_step_zero,
                    )
                )
    return sorted(rows, key=lambda row: (int(row["seed"]), str(row["family"]), int(row["step"]))), missing


def rows_from_report(
    report_path: Path,
    *,
    seed: int,
    family: str,
    family_label: str,
    include_step_zero: bool,
) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records: list[PairwiseRecord] = []
    for pair in report.get("pair_summaries", []):
        agent_a = str(pair["agent_a"])
        agent_b = str(pair["agent_b"])
        step_a = step_from_agent(agent_a)
        step_b = step_from_agent(agent_b)
        if not include_step_zero and (step_a == 0 or step_b == 0):
            continue
        games = float(pair["episodes"])
        records.append(
            PairwiseRecord(
                agent_a=agent_a,
                agent_b=agent_b,
                agent_a_score=float(pair["agent_a_win_rate"]) * games,
                games=games,
            )
        )

    if not records:
        return []
    ratings = calculate_elo(
        records,
        initial_rating=float(report.get("initial_rating", 1500.0)),
        scale=float(report.get("elo_scale", 400.0)),
        prior_std=float(report.get("prior_std", 400.0)),
    )
    rows: list[dict[str, Any]] = []
    for agent, elo in ratings.items():
        step = step_from_agent(agent)
        rows.append(
            {
                "seed": seed,
                "family": family,
                "family_label": family_label,
                "agent": agent,
                "step": step,
                "step_millions": step / 1_000_000.0,
                "elo": float(elo),
                "source_path": str(report_path),
            }
        )
    return rows


def rows_from_csv(
    csv_path: Path,
    *,
    seed: int,
    family: str,
    family_label: str,
    include_step_zero: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            if not include_step_zero and step == 0:
                continue
            rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "family_label": family_label,
                    "agent": row["agent"],
                    "step": step,
                    "step_millions": step / 1_000_000.0,
                    "elo": float(row["elo"]),
                    "source_path": str(csv_path),
                }
            )
    return rows


def step_from_agent(agent: str) -> int:
    return int(agent.rsplit("step", 1)[1])


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["seed", "family", "family_label", "agent", "step", "step_millions", "elo", "source_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    step_millions: dict[int, float] = {}
    labels: dict[str, str] = {}
    for row in rows:
        family = str(row["family"])
        step = int(row["step"])
        grouped.setdefault((family, step), []).append(float(row["elo"]))
        step_millions[step] = float(row["step_millions"])
        labels[family] = str(row["family_label"])

    summary: list[dict[str, Any]] = []
    for (family, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(values)
        sd = stdev(values) if n > 1 else 0.0
        sem = sd / math.sqrt(n) if n > 1 else 0.0
        summary.append(
            {
                "family": family,
                "family_label": labels[family],
                "step": step,
                "step_millions": step_millions[step],
                "n": n,
                "mean_elo": mean(values),
                "sd_elo": sd,
                "sem_elo": sem,
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["family", "family_label", "step", "step_millions", "n", "mean_elo", "sd_elo", "sem_elo"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot_mean_sem(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    colors = {"pure": "#d0693a", "split": "#2f7fba"}
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
                [float(row["elo"]) for row in seed_rows],
                linewidth=1.0,
                alpha=0.18,
                color=colors[family],
                zorder=1,
            )

    for family in ("pure", "split"):
        family_rows = sorted([row for row in summary_rows if row["family"] == family], key=lambda row: int(row["step"]))
        if not family_rows:
            continue
        x = [float(row["step_millions"]) for row in family_rows]
        y = [float(row["mean_elo"]) for row in family_rows]
        sem = [float(row["sem_elo"]) for row in family_rows]
        ax.fill_between(
            x,
            [value - err for value, err in zip(y, sem)],
            [value + err for value, err in zip(y, sem)],
            color=colors[family],
            alpha=0.16,
            linewidth=0.0,
            zorder=2,
        )
        label = f"{family_rows[0]['family_label']} (n={min(int(row['n']) for row in family_rows)}-{max(int(row['n']) for row in family_rows)})"
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.2,
            linewidth=2.4,
            color=colors[family],
            label=label,
            zorder=3,
        )

    ax.set_title("0.2-6M pairwise Elo: pure CFA vs split CFA")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(0.15, 6.05)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_seed_panels(rows: list[dict[str, Any]], missing: list[tuple[int, str]], output_path: Path) -> None:
    colors = {"pure": "#4c78a8", "split": "#f58518"}
    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.2), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.ravel())
    missing_by_seed: dict[int, set[str]] = {}
    for seed, family_label in missing:
        missing_by_seed.setdefault(seed, set()).add(family_label)

    for ax, seed in zip(flat_axes, seeds):
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            if not family_rows:
                continue
            ax.plot(
                [float(row["step_millions"]) for row in family_rows],
                [float(row["elo"]) for row in family_rows],
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                color=colors[family],
                label=str(family_rows[0]["family_label"]),
            )
        title = f"seed {seed}"
        if seed in missing_by_seed:
            title += " (missing " + ", ".join(sorted(missing_by_seed[seed])) + ")"
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0.15, 6.05)
        ax.grid(True, alpha=0.28)

    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Checkpoint step (M)")
    for ax in flat_axes[::3]:
        ax.set_ylabel("Elo")
    flat_axes[0].legend(frameon=False, loc="best")
    fig.suptitle("New 0-6M pairwise Elo: pure CFA vs split CFA")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overlay(rows: list[dict[str, Any]], output_path: Path) -> None:
    colors = {"pure": "#4c78a8", "split": "#f58518"}
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in rows}):
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            if not family_rows:
                continue
            ax.plot(
                [float(row["step_millions"]) for row in family_rows],
                [float(row["elo"]) for row in family_rows],
                marker="o",
                markersize=3.4,
                linewidth=1.4,
                alpha=0.58,
                color=colors[family],
                label=str(family_rows[0]["family_label"]) if seed == 17 else None,
            )
    ax.set_title("New 0.2-6M pairwise Elo")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(0.15, 6.05)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
