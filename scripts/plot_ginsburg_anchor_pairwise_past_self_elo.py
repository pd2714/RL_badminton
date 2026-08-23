from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.elo import PairwiseRecord, calculate_elo


DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/rl/ginsburg_20260622"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "anchor_pairwise_past_self_elo_5seed"
DEFAULT_SEEDS = [17, 23, 31, 47, 59]
FAMILIES = {
    "pure_cfa": "Pure CFA",
    "split_cfa": "Split CFA",
}
STEP_PATTERN = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate five-seed Elo curves from Ginsburg anchor pair-play matrices, "
            "using each checkpoint's matches against earlier checkpoints only."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--prior-std", type=float, default=400.0)
    parser.add_argument(
        "--include-partial-steps",
        action="store_true",
        help="Include steps that are unavailable for some seeds in summaries and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    all_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for family in FAMILIES:
        for seed in seeds:
            pairwise_dir = args.input_root / f"{family}_seed{seed}" / "anchor_pairwise_200r"
            matrix_path = pairwise_dir / "win_rate_matrix.csv"
            report_path = pairwise_dir / "elo_rating" / "elo_rating_report.json"
            if not matrix_path.exists():
                missing.append(str(matrix_path))
                continue
            games = eval_rallies_per_pair(report_path)
            rows, source_count = calculate_past_self_rows(
                matrix_path=matrix_path,
                family=family,
                seed=seed,
                games=games,
                initial_rating=float(args.initial_rating),
                elo_scale=float(args.elo_scale),
                prior_std=float(args.prior_std),
            )
            all_rows.extend(rows)
            source_rows.append(
                {
                    "family": family,
                    "family_label": FAMILIES[family],
                    "seed": seed,
                    "matrix_csv": str(matrix_path),
                    "elo_report_json": str(report_path) if report_path.exists() else "",
                    "eval_rallies_per_pair": games,
                    "record_count": source_count,
                    "rated_checkpoint_count": len(rows),
                }
            )

    if missing:
        raise SystemExit("Missing pair-play matrix files:\n" + "\n".join(missing))
    if not all_rows:
        raise SystemExit(f"No Elo rows found under {args.input_root}")

    common_steps = find_common_steps(all_rows, families=list(FAMILIES), seeds=seeds)
    plot_rows = all_rows if args.include_partial_steps else [row for row in all_rows if int(row["step"]) in common_steps]
    if not plot_rows:
        raise SystemExit("No common steps available across all requested seeds/families")

    summary_rows = summarize_rows(plot_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = args.output_dir / "pure_split_cfa_anchor_pairwise_past_self_elo_rows.csv"
    summary_csv = args.output_dir / "pure_split_cfa_anchor_pairwise_past_self_elo_summary.csv"
    sources_csv = args.output_dir / "pure_split_cfa_anchor_pairwise_past_self_sources.csv"
    plot_png = args.output_dir / "pure_split_cfa_anchor_pairwise_past_self_elo_5seed.png"
    per_seed_png = args.output_dir / "pure_split_cfa_anchor_pairwise_past_self_elo_per_seed.png"

    write_csv(rows_csv, all_rows, ["family", "family_label", "seed", "agent", "step", "step_millions", "elo", "source_matrix"])
    write_csv(
        summary_csv,
        summary_rows,
        ["family", "family_label", "step", "step_millions", "n", "mean_elo", "sd_elo", "sem_elo"],
    )
    write_csv(
        sources_csv,
        source_rows,
        [
            "family",
            "family_label",
            "seed",
            "matrix_csv",
            "elo_report_json",
            "eval_rallies_per_pair",
            "record_count",
            "rated_checkpoint_count",
        ],
    )
    plot_summary(plot_rows, summary_rows, plot_png)
    plot_per_seed(plot_rows, per_seed_png)

    print(f"rows: {rows_csv}")
    print(f"summary: {summary_csv}")
    print(f"sources: {sources_csv}")
    print(f"plot: {plot_png}")
    print(f"per_seed_plot: {per_seed_png}")
    print(f"common_step_count: {len(common_steps)}")
    print("common_steps: " + ",".join(str(step) for step in common_steps))


def eval_rallies_per_pair(report_path: Path) -> float:
    if not report_path.exists():
        return 200.0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return float(report.get("eval_rallies_per_pair", 200.0))


def calculate_past_self_rows(
    *,
    matrix_path: Path,
    family: str,
    seed: int,
    games: float,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
) -> tuple[list[dict[str, Any]], int]:
    matrix_rows = read_matrix(matrix_path)
    records: list[PairwiseRecord] = []
    for row in matrix_rows:
        agent = str(row["checkpoint"])
        agent_step = int(row["step"])
        for opponent, win_rate in row["win_rates"].items():
            opponent_step = step_from_label(opponent)
            if agent_step <= opponent_step:
                continue
            records.append(
                PairwiseRecord(
                    agent_a=agent,
                    agent_b=opponent,
                    agent_a_score=float(win_rate) * float(games),
                    games=float(games),
                )
            )
    if not records:
        raise ValueError(f"No past-self pair records found in {matrix_path}")

    ratings = calculate_elo(
        records,
        initial_rating=initial_rating,
        scale=elo_scale,
        prior_std=prior_std,
    )
    rows: list[dict[str, Any]] = []
    for agent, elo in ratings.items():
        step = step_from_label(agent)
        rows.append(
            {
                "family": family,
                "family_label": FAMILIES[family],
                "seed": seed,
                "agent": agent,
                "step": step,
                "step_millions": step / 1_000_000.0,
                "elo": float(elo),
                "source_matrix": str(matrix_path),
            }
        )
    return sorted(rows, key=lambda row: int(row["step"])), len(records)


def read_matrix(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "checkpoint" not in reader.fieldnames or "step" not in reader.fieldnames:
            raise ValueError(f"Invalid win-rate matrix header: {path}")
        opponent_labels = [field for field in reader.fieldnames if field not in {"checkpoint", "step"}]
        rows: list[dict[str, Any]] = []
        for raw in reader:
            rows.append(
                {
                    "checkpoint": str(raw["checkpoint"]),
                    "step": int(raw["step"]),
                    "win_rates": {label: float(raw[label]) for label in opponent_labels if raw.get(label, "") != ""},
                }
            )
    return rows


def step_from_label(label: str) -> int:
    match = STEP_PATTERN.search(label)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint step from label: {label}")
    return int(match.group(1))


def find_common_steps(rows: list[dict[str, Any]], *, families: list[str], seeds: list[int]) -> list[int]:
    steps_by_key: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        steps_by_key[(str(row["family"]), int(row["seed"]))].add(int(row["step"]))
    required_keys = [(family, seed) for family in families for seed in seeds]
    common = set.intersection(*(steps_by_key[key] for key in required_keys))
    return sorted(common)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), int(row["step"]))].append(float(row["elo"]))

    summary: list[dict[str, Any]] = []
    for (family, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(values)
        sd = stdev(values) if n > 1 else 0.0
        sem = sd / math.sqrt(n) if n > 1 else 0.0
        summary.append(
            {
                "family": family,
                "family_label": FAMILIES[family],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def configure_plot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.linewidth": 1.0,
        }
    )
    return plt


def plot_summary(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = configure_plot()
    colors = {"pure_cfa": "#d0693a", "split_cfa": "#2f7fba"}
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.94)

    for (family, seed), seed_rows in group_rows(rows).items():
        seed_rows = sorted(seed_rows, key=lambda row: int(row["step"]))
        ax.plot(
            [float(row["step_millions"]) for row in seed_rows],
            [float(row["elo"]) for row in seed_rows],
            color=colors[family],
            linewidth=1.0,
            alpha=0.22,
            zorder=1,
        )

    for family in ("pure_cfa", "split_cfa"):
        family_rows = sorted([row for row in summary_rows if row["family"] == family], key=lambda row: int(row["step"]))
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
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.6,
            linewidth=2.4,
            color=colors[family],
            label=FAMILIES[family],
            zorder=3,
        )

    ax.set_title("Past-self pair-play Elo across 5 seeds", fontsize=18, pad=8)
    ax.set_xlabel("Training step (M)")
    ax.set_ylabel("Elo")
    ax.grid(True, color="0.72", linewidth=0.8, alpha=0.35)
    ax.legend(frameon=False, loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_per_seed(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = configure_plot()
    colors = {"pure_cfa": "#d0693a", "split_cfa": "#2f7fba"}
    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.ravel())

    for ax, seed in zip(flat_axes, seeds):
        for family in ("pure_cfa", "split_cfa"):
            seed_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in seed_rows],
                [float(row["elo"]) for row in seed_rows],
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                color=colors[family],
                label=FAMILIES[family],
            )
        ax.set_title(f"seed {seed}", fontsize=16)
        ax.grid(True, color="0.72", linewidth=0.8, alpha=0.35)

    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Training step (M)")
    for ax in flat_axes[::3]:
        ax.set_ylabel("Elo")
    flat_axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Past-self pair-play Elo by seed", fontsize=18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), int(row["seed"]))].append(row)
    return grouped


if __name__ == "__main__":
    main()
