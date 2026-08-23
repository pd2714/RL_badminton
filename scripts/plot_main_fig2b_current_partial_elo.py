from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.elo import PairwiseRecord, calculate_elo


DEFAULT_INPUT_ROOT = Path("outputs/rl/ginsburg_20260622/same_main_fig2b_fixed_pool_eval_200r")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "split_vs_pure_cfa_current_partial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot current partial main-Fig2B pure-CFA vs split-CFA Elo by seed."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--prior-std", type=float, default=400.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        args.input_root,
        initial_rating=float(args.initial_rating),
        elo_scale=float(args.elo_scale),
        prior_std=float(args.prior_std),
    )
    if not rows:
        raise SystemExit(f"No partial Fig2B rows found under {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "current_partial_pure_split_elo_by_seed.csv"
    panel_path = args.output_dir / "current_partial_pure_split_elo_seed_panels.png"
    overlay_path = args.output_dir / "current_partial_pure_split_elo_overlay.png"
    write_rows(csv_path, rows)
    plot_seed_panels(rows, panel_path)
    plot_overlay(rows, overlay_path)

    print(f"rows: {csv_path}")
    print(f"seed_panels: {panel_path}")
    print(f"overlay: {overlay_path}")
    for seed in sorted({int(row['seed']) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        completed = int(seed_rows[0]["seed_completed_pair_count"])
        expected = int(seed_rows[0]["seed_expected_pair_count"])
        print(f"seed {seed}: {completed}/{expected} pairs ({100.0 * completed / expected:.1f}%)")


def build_rows(
    input_root: Path,
    *,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(input_root.glob("seed_*/split_vs_pure_cfa"), key=seed_sort_key):
        manifest_path = seed_dir / "manifest.json"
        partial_path = seed_dir / "pair_results.jsonl"
        if not manifest_path.exists() or not partial_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base_report = json.loads(Path(manifest["base_report"]).read_text(encoding="utf-8"))
        pair_results = read_jsonl(partial_path)
        seed_completed_pair_count = len(pair_results)
        ratings = estimate_combined_ratings(
            [*base_report["pair_results"], *pair_results],
            initial_rating=initial_rating,
            elo_scale=elo_scale,
            prior_std=prior_std,
        )

        opponent_count = sum(1 for entry in manifest["opponent_pool"] if bool(entry.get("available", True)))
        eval_agent_count = sum(1 for entry in manifest["eval_agents"] if bool(entry.get("available", False)))
        seed_expected_pair_count = eval_agent_count * opponent_count
        counts_by_agent: dict[str, int] = defaultdict(int)
        wins_by_agent: dict[str, float] = defaultdict(float)
        games_by_agent: dict[str, float] = defaultdict(float)
        for pair in pair_results:
            agent = str(pair["agent"])
            counts_by_agent[agent] += 1
            wins_by_agent[agent] += float(pair["agent_wins"])
            games_by_agent[agent] += float(pair["episodes"])

        seed = seed_dir.parts[-2].replace("seed_", "")
        for entry in manifest["eval_agents"]:
            if not bool(entry.get("available", False)):
                continue
            label = str(entry["label"])
            family = family_from_label(label)
            if family is None:
                continue
            evaluated = counts_by_agent[label]
            if evaluated == 0:
                continue
            games = games_by_agent[label]
            rows.append(
                {
                    "seed": int(seed),
                    "family": family,
                    "label": label,
                    "step": int(entry["step"]),
                    "step_millions": int(entry["step"]) / 1_000_000.0,
                    "elo": ratings.get(label),
                    "evaluated_pair_count": evaluated,
                    "expected_pair_count": opponent_count,
                    "pair_completion": evaluated / opponent_count if opponent_count else 0.0,
                    "seed_completed_pair_count": seed_completed_pair_count,
                    "seed_expected_pair_count": seed_expected_pair_count,
                    "seed_pair_completion": seed_completed_pair_count / seed_expected_pair_count
                    if seed_expected_pair_count
                    else 0.0,
                    "mean_pool_win_rate": wins_by_agent[label] / games if games else None,
                }
            )
    return sorted(rows, key=lambda row: (int(row["seed"]), str(row["family"]), int(row["step"])))


def seed_sort_key(path: Path) -> int:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def estimate_combined_ratings(
    pair_results: list[dict[str, Any]],
    *,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
) -> dict[str, float]:
    records = []
    for row in pair_results:
        agent = str(row["agent"])
        opponent = str(row["opponent"])
        if agent == opponent:
            continue
        records.append(
            PairwiseRecord(
                agent_a=agent,
                agent_b=opponent,
                agent_a_score=float(row["agent_wins"]),
                games=float(row["episodes"]),
            )
        )
    if not records:
        return {}
    return calculate_elo(records, initial_rating=initial_rating, scale=elo_scale, prior_std=prior_std)


def family_from_label(label: str) -> str | None:
    if label.startswith("pure_cfa_seed"):
        return "pure"
    if label.startswith("split_cfa_seed"):
        return "split"
    return None


def family_text(family: str) -> str:
    return {"pure": "Pure CFA", "split": "Split CFA"}.get(family, family)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "family",
        "label",
        "step",
        "step_millions",
        "elo",
        "evaluated_pair_count",
        "expected_pair_count",
        "pair_completion",
        "seed_completed_pair_count",
        "seed_expected_pair_count",
        "seed_pair_completion",
        "mean_pool_win_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_seed_panels(rows: list[dict[str, Any]], output_path: Path) -> None:
    colors = {"pure": "#4c78a8", "split": "#f58518"}
    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.2), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.ravel())

    for ax, seed in zip(flat_axes, seeds):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in seed_rows if row["family"] == family and row["elo"] is not None],
                key=lambda row: int(row["step"]),
            )
            if not family_rows:
                continue
            x = [float(row["step_millions"]) for row in family_rows]
            y = [float(row["elo"]) for row in family_rows]
            ax.plot(x, y, color=colors[family], linewidth=2.0, alpha=0.86, label=family_text(family))
            ax.scatter(
                x,
                y,
                s=[22.0 + 48.0 * float(row["pair_completion"]) for row in family_rows],
                color=colors[family],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        completed = int(seed_rows[0]["seed_completed_pair_count"])
        expected = int(seed_rows[0]["seed_expected_pair_count"])
        ax.set_title(f"seed {seed} ({100.0 * completed / expected:.0f}% total pairs)")
        ax.set_xlim(0.0, 6.05)
        ax.grid(True, alpha=0.28)

    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Checkpoint step (M)")
    for ax in flat_axes[::3]:
        ax.set_ylabel("Elo")
    flat_axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Current partial main Fig2B Elo: pure CFA vs split CFA")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overlay(rows: list[dict[str, Any]], output_path: Path) -> None:
    colors = {"pure": "#4c78a8", "split": "#f58518"}
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in rows}):
        for family in ("pure", "split"):
            family_rows = sorted(
                [
                    row
                    for row in rows
                    if int(row["seed"]) == seed and row["family"] == family and row["elo"] is not None
                ],
                key=lambda row: int(row["step"]),
            )
            if not family_rows:
                continue
            ax.plot(
                [float(row["step_millions"]) for row in family_rows],
                [float(row["elo"]) for row in family_rows],
                marker="o",
                markersize=3.8,
                linewidth=1.6,
                alpha=0.62,
                color=colors[family],
                label=family_text(family) if seed == 17 else None,
            )
    ax.set_title("Current partial main Fig2B Elo")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(0.0, 6.05)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
