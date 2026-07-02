from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/eval"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/panel_b_purecfa_vs_nocfa_direct_winrate_5seed.png"
DEFAULT_SEED_ROWS = (
    REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/panel_b_purecfa_vs_nocfa_direct_winrate_5seed_rows.csv"
)
DEFAULT_SUMMARY = (
    REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/panel_b_purecfa_vs_nocfa_direct_winrate_5seed_summary.csv"
)
EXPECTED_SEEDS = [17, 23, 31, 47, 59]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot direct matched-checkpoint Pure-CFA/CRA win rates against no-CFA/no-CRA across seeds."
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-rows-csv", type=Path, default=DEFAULT_SEED_ROWS)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seeds", type=int, nargs="+", default=EXPECTED_SEEDS)
    parser.add_argument("--max-step", type=int, default=2_800_000, help="Largest checkpoint step to include.")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested seed has no pair_win_rates.csv or no matched direct rows.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/private/tmp")) / "rl_badminton_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)


def _wilson_interval(wins: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0.0:
        raise ValueError("Wilson interval requires a positive sample size")
    p_hat = wins / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denom
    half_width = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def load_direct_seed_rows(eval_root: Path, seeds: list[int]) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in seeds:
        csv_path = eval_root / f"seed_{seed}" / "panel_b_shared_pool_elo_200r" / "pair_win_rates.csv"
        if not csv_path.exists():
            missing.append(seed)
            continue

        seed_rows = matched_direct_rows(csv_path, seed)
        if not seed_rows:
            missing.append(seed)
            continue
        rows.extend(seed_rows)
    return rows, missing


def matched_direct_rows(csv_path: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_agent_run = f"purecfa_seed{seed}"
    expected_opponent_run = f"nocfa_seed{seed}"
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["agent_run_label"] != expected_agent_run:
                continue
            if row["opponent_run_label"] != expected_opponent_run:
                continue
            if int(row["agent_step"]) != int(row["opponent_step"]):
                continue

            episodes = int(float(row["episodes"]))
            wins = float(row["agent_wins"])
            win_rate = float(row["agent_win_rate"])
            ci_low, ci_high = _wilson_interval(wins, episodes)
            step = int(row["agent_step"])
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "step_millions": step / 1_000_000.0,
                    "episodes": episodes,
                    "cra_wins": wins,
                    "cra_win_rate": win_rate,
                    "no_cra_win_rate": 1.0 - win_rate,
                    "wilson95_low": ci_low,
                    "wilson95_high": ci_high,
                }
            )
    return sorted(rows, key=lambda row: int(row["step"]))


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)

    summary: list[dict[str, Any]] = []
    for step, step_rows in sorted(by_step.items()):
        rates = [float(row["cra_win_rate"]) for row in step_rows]
        wins = sum(float(row["cra_wins"]) for row in step_rows)
        episodes = sum(int(row["episodes"]) for row in step_rows)
        sd = stdev(rates) if len(rates) > 1 else 0.0
        sem = sd / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
        pooled_rate = wins / episodes
        ci_low, ci_high = _wilson_interval(wins, episodes)
        summary.append(
            {
                "step": step,
                "step_millions": step / 1_000_000.0,
                "n_seeds": len(rates),
                "episodes": episodes,
                "cra_wins": wins,
                "mean_cra_win_rate": mean(rates),
                "sd_cra_win_rate": sd,
                "sem_cra_win_rate": sem,
                "pooled_cra_win_rate": pooled_rate,
                "pooled_wilson95_low": ci_low,
                "pooled_wilson95_high": ci_high,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output: Path) -> None:
    configure_matplotlib()

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
        }
    )

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)

    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.18, top=0.97)

    seed_colors = ["#6b8fb3", "#8c7a5b", "#7b9c73", "#a06f8c", "#6f8f8c"]
    for color, (seed, seed_rows) in zip(seed_colors, sorted(by_seed.items())):
        seed_rows = sorted(seed_rows, key=lambda row: int(row["step"]))
        ax.plot(
            [float(row["step_millions"]) for row in seed_rows],
            [float(row["cra_win_rate"]) for row in seed_rows],
            marker="o",
            linewidth=1.1,
            markersize=3.2,
            color=color,
            alpha=0.46,
            zorder=2,
        )

    summary_rows = sorted(summary_rows, key=lambda row: int(row["step"]))
    x = [float(row["step_millions"]) for row in summary_rows]
    y = [float(row["mean_cra_win_rate"]) for row in summary_rows]
    sem = [float(row["sem_cra_win_rate"]) for row in summary_rows]
    ax.fill_between(
        x,
        [max(0.0, value - err) for value, err in zip(y, sem)],
        [min(1.0, value + err) for value, err in zip(y, sem)],
        color="#d0693a",
        alpha=0.18,
        linewidth=0.0,
        zorder=3,
    )
    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2.2,
        markersize=4.8,
        color="#d0693a",
        zorder=4,
    )

    ax.axhline(0.5, color="0.25", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlabel("Checkpoint (M)")
    ax.set_ylabel("CRA win rate vs no CRA")
    ax.set_xlim(0.25, 2.95)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.1f}" for value in x])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(True, color="0.72", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_box_aspect(0.72)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows, missing = load_direct_seed_rows(args.eval_root, [int(seed) for seed in args.seeds])
    if missing and args.require_all:
        raise SystemExit(f"Missing direct pair-win-rate rows for seed(s): {', '.join(str(seed) for seed in missing)}")
    rows = [row for row in rows if int(row["step"]) <= int(args.max_step)]
    if not rows:
        raise SystemExit(f"No matched direct pair-win-rate rows found under {args.eval_root}")

    summary_rows = summarize_rows(rows)
    write_csv(
        args.seed_rows_csv,
        rows,
        [
            "seed",
            "step",
            "step_millions",
            "episodes",
            "cra_wins",
            "cra_win_rate",
            "no_cra_win_rate",
            "wilson95_low",
            "wilson95_high",
        ],
    )
    write_csv(
        args.summary_csv,
        summary_rows,
        [
            "step",
            "step_millions",
            "n_seeds",
            "episodes",
            "cra_wins",
            "mean_cra_win_rate",
            "sd_cra_win_rate",
            "sem_cra_win_rate",
            "pooled_cra_win_rate",
            "pooled_wilson95_low",
            "pooled_wilson95_high",
        ],
    )
    plot(rows, summary_rows, args.output)

    print(f"output: {args.output}")
    print(f"seed_rows_csv: {args.seed_rows_csv}")
    print(f"summary_csv: {args.summary_csv}")
    print(f"seed_count: {len({int(row['seed']) for row in rows})}")
    if missing:
        print(f"missing_seeds: {','.join(str(seed) for seed in missing)}")


if __name__ == "__main__":
    main()
