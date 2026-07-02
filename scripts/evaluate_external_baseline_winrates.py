from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.evaluation import ModelSelector, evaluate_selector
from badminton1d.eval_evolution import (
    build_discrete_action_config,
    build_sim_config,
    checkpoint_step,
    discover_anchor_checkpoints,
    filter_anchor_checkpoints,
    load_anchor_model,
    load_run_config,
)
from badminton1d.mpl_config import ensure_writable_matplotlib_config
from badminton1d.opponents import make_opponent
from badminton1d.selfplay import build_selfplay_env
from badminton1d.shot_generators import TacticRuntimeConfig
from badminton1d.utils import ensure_directory


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "external_baseline_eval_200r"
DEFAULT_FIGURE = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/external_baseline_win_rate_vs_checkpoint.png"
BASELINE_LABELS = {
    "safe": "scripted clear/recover",
    "greedy": "scripted high-intercept",
    "random": "random valid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final-run anchor checkpoints against external heuristic opponents and plot win-rate curves."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--baselines", nargs="+", choices=("safe", "greedy", "random"), default=["safe", "greedy", "random"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--anchor-step-min", type=int, default=None)
    parser.add_argument("--anchor-step-max", type=int, default=None)
    parser.add_argument("--anchor-step-interval", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def config_value(data: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def binomial_ci(win_rate: float, episodes: int, z: float = 1.96) -> tuple[float, float]:
    if episodes <= 0:
        return float("nan"), float("nan")
    # Wilson interval behaves better near 0/1 than the plain normal approximation.
    n = float(episodes)
    p = float(win_rate)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half_width = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def selected_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints = discover_anchor_checkpoints(args.run_dir)
    checkpoints = filter_anchor_checkpoints(
        checkpoints,
        step_min=args.anchor_step_min,
        step_max=args.anchor_step_max,
        step_interval=args.anchor_step_interval,
    )
    checkpoints = checkpoints[:: max(int(args.anchor_stride), 1)]
    if args.max_anchors is not None:
        checkpoints = checkpoints[: max(int(args.max_anchors), 0)]
    return checkpoints


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    ensure_directory(args.output_dir)

    run_config = load_run_config(args.run_dir)
    sim_config = build_sim_config(run_config)
    discrete_action_config = build_discrete_action_config(run_config)
    checkpoints = selected_checkpoints(args)
    policy_type = str(config_value(run_config, "policy_type", default="velocity_oriented"))
    train_side = str(config_value(run_config, "train_side", default="left"))
    initial_server = str(config_value(run_config, "initial_server", default="random"))
    random_service_x = bool(config_value(run_config, "random_service_x", default=True))
    train_reaction_time = float(config_value(run_config, "reaction_time", default=0.15))
    opponent_reaction_time = float(config_value(run_config, "opponent_reaction_time", default=train_reaction_time))
    max_stages_per_rally = int(config_value(run_config, "max_rally_stages", default=120))
    tactic_runtime = TacticRuntimeConfig(
        regenerate_lookup_table=bool(config_value(run_config, "regenerate_lookup_table", default=False)),
        lookup_dir=Path(str(config_value(run_config, "lookup_table_dir", default="lookup_tables"))),
    )

    partial_path = args.output_dir / "external_baseline_pair_results.jsonl"
    rows = load_jsonl(partial_path)
    completed = {(int(row["step"]), str(row["baseline"])) for row in rows}

    if args.dry_run:
        return build_report(args, checkpoints, rows)

    for anchor_index, checkpoint in enumerate(checkpoints):
        step = checkpoint_step(checkpoint)
        needed = [baseline for baseline in args.baselines if (step, baseline) not in completed]
        if not needed:
            continue
        print(f"loading anchor_step_{step}", flush=True)
        model = load_anchor_model(checkpoint, recovery_choice_diagnostics=True)
        selector = ModelSelector(model=model, deterministic=bool(args.deterministic))
        for baseline_index, baseline in enumerate(args.baselines):
            if (step, baseline) in completed:
                print(f"anchor_step_{step} vs {baseline}: already complete", flush=True)
                continue
            pair_seed = int(args.seed) + anchor_index * 100_000 + baseline_index * 10_000
            env = build_selfplay_env(
                train_side=train_side,  # type: ignore[arg-type]
                mirror_train_side=False,
                mirror_match_fraction=0.0,
                initial_server=initial_server,
                random_service_x=random_service_x,
                sim_config=sim_config,
                train_reaction_time=train_reaction_time,
                opponent_reaction_time=opponent_reaction_time,
                max_stages_per_rally=max_stages_per_rally,
                policy_type=policy_type,
                tactic_runtime_config=tactic_runtime,
                seed=pair_seed,
                discrete_action_config=discrete_action_config,
                opponent=make_opponent(baseline, seed=pair_seed + 17),
                include_records_in_info=False,
                recovery_counterfactual_other_sample_count=0,
                counterfactual_opponent_response_samples=0,
                recovery_counterfactual_expected_response_target=False,
            )
            summary, _ = evaluate_selector(
                f"anchor_step_{step}_vs_{baseline}",
                selector,
                env,
                int(args.episodes),
                pair_seed,
            )
            win_rate = float(summary["win_rate"])
            ci_low, ci_high = binomial_ci(win_rate, int(summary["episodes"]))
            row = {
                "step": step,
                "step_million": step / 1_000_000.0,
                "checkpoint_path": str(checkpoint),
                "baseline": baseline,
                "baseline_label": BASELINE_LABELS.get(baseline, baseline),
                "episodes": int(summary["episodes"]),
                "wins": win_rate * int(summary["episodes"]),
                "win_rate": win_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "avg_reward": float(summary["avg_reward"]),
                "avg_rally_length": float(summary["avg_rally_length"]),
                "avg_invalid_action_rate": float(summary["avg_invalid_action_rate"]),
                "truncation_rate": float(summary["truncation_rate"]),
            }
            rows.append(row)
            completed.add((step, baseline))
            append_jsonl(partial_path, row)
            print(f"anchor_step_{step} vs {baseline}: wr={win_rate:.3f} ({summary['episodes']} rallies)", flush=True)

    return build_report(args, checkpoints, rows)


def build_report(args: argparse.Namespace, checkpoints: list[Path], rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_steps = [checkpoint_step(path) for path in checkpoints]
    rows = sorted(
        (row for row in rows if int(row["step"]) in set(selected_steps) and str(row["baseline"]) in set(args.baselines)),
        key=lambda row: (int(row["step"]), str(row["baseline"])),
    )
    return {
        "definition": "Win rate is Pr(anchor checkpoint model beats the named external opponent).",
        "run_dir": str(args.run_dir),
        "output_dir": str(args.output_dir),
        "figure": str(args.figure),
        "baselines": list(args.baselines),
        "episodes_per_pair": int(args.episodes),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "anchor_steps": selected_steps,
        "completed_pair_count": len(rows),
        "expected_pair_count": len(selected_steps) * len(args.baselines),
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "step",
        "step_million",
        "baseline",
        "baseline_label",
        "episodes",
        "wins",
        "win_rate",
        "ci_low",
        "ci_high",
        "avg_reward",
        "avg_rally_length",
        "avg_invalid_action_rate",
        "truncation_rate",
        "checkpoint_path",
    ]
    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_report(report: dict[str, Any], output_path: Path) -> None:
    ensure_writable_matplotlib_config()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = list(report["rows"])
    if not rows:
        raise ValueError("No completed rows to plot.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "axes.linewidth": 1.4,
        }
    )
    colors = {"safe": "#4c78a8", "greedy": "#f58518", "random": "#54a24b"}
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for baseline in report["baselines"]:
        baseline_rows = [row for row in rows if row["baseline"] == baseline]
        if not baseline_rows:
            continue
        baseline_rows = sorted(baseline_rows, key=lambda row: int(row["step"]))
        x = np.asarray([float(row["step_million"]) for row in baseline_rows], dtype=float)
        y = np.asarray([float(row["win_rate"]) for row in baseline_rows], dtype=float)
        low = np.asarray([float(row["ci_low"]) for row in baseline_rows], dtype=float)
        high = np.asarray([float(row["ci_high"]) for row in baseline_rows], dtype=float)
        color = colors.get(str(baseline), None)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.8,
            linewidth=2.2,
            color=color,
            label=BASELINE_LABELS.get(str(baseline), str(baseline)),
        )
        ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0.0)

    ax.axhline(0.5, color="0.25", linewidth=1.2, linestyle="--", alpha=0.75)
    ax.set_xlabel("Training checkpoint (M steps)")
    ax.set_ylabel("Win rate vs external agent")
    ax.set_ylim(0.45, 1.025)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xlim(left=0.0)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.grid(True, color="0.80", linewidth=0.9, alpha=0.55)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "rl_badminton_mplconfig"))
    args = parse_args()
    report = evaluate(args)
    report_path = args.output_dir / "external_baseline_winrates.json"
    csv_path = args.output_dir / "external_baseline_winrates.csv"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, report["rows"])
    if report["rows"]:
        plot_report(report, args.figure)
    print(f"completed: {report['completed_pair_count']}/{report['expected_pair_count']}")
    print(f"json: {report_path}")
    print(f"csv: {csv_path}")
    if report["rows"]:
        print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
