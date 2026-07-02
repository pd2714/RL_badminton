from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.elo import PairwiseRecord, calculate_elo
from badminton1d.eval_evolution import build_discrete_action_config, build_sim_config
from badminton1d.utils import ensure_directory
from scripts.evaluate_requested_cross_run_fixed_pool_200r import (
    Entry,
    append_partial_result,
    evaluate_pair,
    load_json,
    load_model_cached,
    load_partial_results,
    write_pair_csv,
)


BASE_EVAL_DIR = Path(
    "outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
) / "cross_run_fixed_pool_eval_200r"
DEFAULT_BASE_REPORT = BASE_EVAL_DIR / "fixed_pool_eval_report.json"
DEFAULT_BASE_METRICS = BASE_EVAL_DIR / "mean_win_rate_elo_shared_old_prefix_dense_new_from_cached_matrix.csv"
DEFAULT_SHOT_RUN = Path(
    "outputs/rl/selfplay_2d_shotcfadv_from5m_1m_varietypool70hist15recent10heur5newest_20260619"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SHOT_RUN / "cross_run_fixed_pool_eval_200r_same_pool"
DEFAULT_STEPS = [5_200_000, 5_400_000, 5_600_000, 5_800_000, 6_000_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the June 19 shot-CF checkpoints against the exact fixed pool used "
            "by the June 11 shared-prefix cross-run plot."
        )
    )
    parser.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    parser.add_argument("--base-metrics", type=Path, default=DEFAULT_BASE_METRICS)
    parser.add_argument("--shot-run-dir", type=Path, default=DEFAULT_SHOT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=BASE_EVAL_DIR,
        help="Directory for the overlaid old/new/shot plots.",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--eval-rallies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--prior-std", type=float, default=400.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        base_report_path=args.base_report,
        base_metrics_path=args.base_metrics,
        shot_run_dir=args.shot_run_dir,
        output_dir=args.output_dir,
        comparison_dir=args.comparison_dir,
        steps=[int(step) for step in args.steps],
        eval_rallies=int(args.eval_rallies),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        initial_rating=float(args.initial_rating),
        elo_scale=float(args.elo_scale),
        prior_std=float(args.prior_std),
        dry_run=bool(args.dry_run),
    )
    print(f"output: {args.output_dir}")
    print(f"pairs: {args.output_dir / 'pair_results.csv'}")
    print(f"shot_metrics: {args.output_dir / 'mean_win_rate_elo.csv'}")
    print(f"comparison_csv: {args.comparison_dir / 'mean_win_rate_elo_shared_old_prefix_dense_new_with_shotcfadv.csv'}")
    print(f"comparison_elo_plot: {args.comparison_dir / 'elo_vs_checkpoint_shared_old_prefix_with_shotcfadv.png'}")
    print(
        "counts: "
        f"available={report['available_pair_count']} "
        f"completed={report['completed_pair_count']} "
        f"remaining={report['remaining_pair_count']}"
    )


def evaluate(
    *,
    base_report_path: Path,
    base_metrics_path: Path,
    shot_run_dir: Path,
    output_dir: Path,
    comparison_dir: Path,
    steps: list[int],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
    dry_run: bool,
) -> dict[str, Any]:
    if eval_rallies <= 0:
        raise ValueError("--eval-rallies must be positive")
    ensure_directory(output_dir)
    ensure_directory(comparison_dir)

    base_report = load_json(base_report_path)
    train_config = load_json(shot_run_dir / "selfplay_config.json")
    sim_config = build_sim_config(train_config)
    discrete_action_config = build_discrete_action_config(train_config)

    eval_entries = build_shot_entries(shot_run_dir, steps)
    opponent_entries = build_opponent_entries(base_report)
    available_eval = [entry for entry in eval_entries if entry.available]
    available_opponents = [entry for entry in opponent_entries if entry.available]
    available_pairs = {(agent.label, opponent.label) for agent in available_eval for opponent in available_opponents}

    write_manifest(
        output_dir,
        base_report_path=base_report_path,
        base_metrics_path=base_metrics_path,
        shot_run_dir=shot_run_dir,
        eval_entries=eval_entries,
        opponent_entries=opponent_entries,
        eval_rallies=eval_rallies,
        seed=seed,
        deterministic=deterministic,
    )

    partial_path = output_dir / "pair_results.jsonl"
    pair_results = load_partial_results(partial_path)
    completed = {(str(row["agent"]), str(row["opponent"])) for row in pair_results}

    if not dry_run:
        model_cache: dict[Path, Any] = {}
        for agent_index, agent in enumerate(available_eval):
            assert agent.model_path is not None
            model = load_model_cached(model_cache, agent.model_path)
            for opponent_index, opponent in enumerate(available_opponents):
                pair_key = (agent.label, opponent.label)
                if pair_key in completed:
                    print(f"{agent.label} vs {opponent.label}: already complete", flush=True)
                    continue
                assert opponent.model_path is not None
                pair_seed = seed + agent_index * 1_000_000 + opponent_index * 10_000
                summary = evaluate_pair(
                    agent=agent,
                    opponent=opponent,
                    model=model,
                    train_config=train_config,
                    sim_config=sim_config,
                    discrete_action_config=discrete_action_config,
                    episodes=eval_rallies,
                    seed=pair_seed,
                    deterministic=deterministic,
                )
                pair = {
                    "agent": agent.label,
                    "agent_display_label": agent.display_label,
                    "agent_run_label": agent.run_label,
                    "agent_step": agent.step,
                    "opponent": opponent.label,
                    "opponent_display_label": opponent.display_label,
                    "opponent_run_label": opponent.run_label,
                    "opponent_step": opponent.step,
                    "episodes": int(summary["episodes"]),
                    "agent_wins": float(summary["win_rate"]) * int(summary["episodes"]),
                    "agent_win_rate": float(summary["win_rate"]),
                    "opponent_win_rate": 1.0 - float(summary["win_rate"]),
                    "source": "simulated_shotcfadv_same_fixed_pool_200r",
                    "summary": summary,
                }
                pair_results.append(pair)
                append_partial_result(partial_path, pair)
                completed.add(pair_key)
                print(
                    f"{agent.label} vs {opponent.label}: "
                    f"wr={pair['agent_win_rate']:.3f} ({pair['episodes']} rallies)",
                    flush=True,
                )

    report = build_report(
        base_report=base_report,
        base_report_path=base_report_path,
        base_metrics_path=base_metrics_path,
        shot_run_dir=shot_run_dir,
        output_dir=output_dir,
        comparison_dir=comparison_dir,
        eval_entries=eval_entries,
        opponent_entries=opponent_entries,
        pair_results=pair_results,
        available_pairs=available_pairs,
        eval_rallies=eval_rallies,
        seed=seed,
        deterministic=deterministic,
        initial_rating=initial_rating,
        elo_scale=elo_scale,
        prior_std=prior_std,
    )
    write_outputs(output_dir, comparison_dir, base_metrics_path, report)
    return report


def build_shot_entries(run_dir: Path, steps: list[int]) -> list[Entry]:
    entries = []
    for step in steps:
        model_path = run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip"
        entries.append(
            Entry(
                label=f"shotcfadv_{step}",
                display_label=f"shot CF adv {step / 1_000_000.0:.1f}M",
                run_label="shotcfadv",
                step=step,
                run_dir=run_dir,
                model_path=model_path if model_path.exists() else None,
                available=model_path.exists(),
            )
        )
    return entries


def build_opponent_entries(base_report: dict[str, Any]) -> list[Entry]:
    entries = []
    for row in base_report["opponent_pool"]:
        model_path = resolve_existing_model_path(Path(row["model_path"]))
        entries.append(
            Entry(
                label=str(row["label"]),
                display_label=str(row["display_label"]),
                run_label=str(row["run_label"]),
                step=int(row["step"]),
                run_dir=resolve_existing_dir(Path(row["run_dir"])),
                model_path=model_path if model_path.exists() else None,
                available=model_path.exists(),
            )
        )
    return entries


def resolve_existing_model_path(path: Path) -> Path:
    if path.exists():
        return path
    candidate = Path(str(path).replace("/selfplay_2d_recoverycfdefault_resp1_3m_", "/final_selfplay_2d_recoverycfdefault_resp1_3m_"))
    if candidate.exists():
        return candidate
    return path


def resolve_existing_dir(path: Path) -> Path:
    if path.exists():
        return path
    candidate = Path(str(path).replace("/selfplay_2d_recoverycfdefault_resp1_3m_", "/final_selfplay_2d_recoverycfdefault_resp1_3m_"))
    if candidate.exists():
        return candidate
    return path


def build_report(
    *,
    base_report: dict[str, Any],
    base_report_path: Path,
    base_metrics_path: Path,
    shot_run_dir: Path,
    output_dir: Path,
    comparison_dir: Path,
    eval_entries: list[Entry],
    opponent_entries: list[Entry],
    pair_results: list[dict[str, Any]],
    available_pairs: set[tuple[str, str]],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
) -> dict[str, Any]:
    completed = {(str(row["agent"]), str(row["opponent"])) for row in pair_results}
    ratings = estimate_combined_ratings(
        [*base_report["pair_results"], *pair_results],
        initial_rating=initial_rating,
        elo_scale=elo_scale,
        prior_std=prior_std,
    )
    eval_rows = build_eval_rows(eval_entries, pair_results, ratings)
    return {
        "description": "Shot-CF-adv checkpoints evaluated against the same fixed pool as the June 11 shared-prefix plot.",
        "base_report": str(base_report_path),
        "base_metrics": str(base_metrics_path),
        "shot_run_dir": str(shot_run_dir),
        "output_dir": str(output_dir),
        "comparison_dir": str(comparison_dir),
        "seed": seed,
        "deterministic": deterministic,
        "eval_rallies_per_pair": eval_rallies,
        "initial_rating": initial_rating,
        "elo_scale": elo_scale,
        "prior_std": prior_std,
        "available_pair_count": len(available_pairs),
        "completed_pair_count": len(completed & available_pairs),
        "remaining_pair_count": len(available_pairs - completed),
        "missing_eval_entries": [entry_payload(entry) for entry in eval_entries if not entry.available],
        "missing_opponent_entries": [entry_payload(entry) for entry in opponent_entries if not entry.available],
        "eval_agents": [entry_payload(entry) for entry in eval_entries],
        "opponent_pool": [entry_payload(entry) for entry in opponent_entries],
        "pair_results": pair_results,
        "eval_agent_metrics": eval_rows,
    }


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


def build_eval_rows(
    eval_entries: list[Entry],
    pair_results: list[dict[str, Any]],
    ratings: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for entry in eval_entries:
        values = [float(row["agent_win_rate"]) for row in pair_results if str(row["agent"]) == entry.label]
        rows.append(
            {
                **entry_payload(entry),
                "evaluated_pair_count": len(values),
                "mean_pool_win_rate": None if not values else float(sum(values) / len(values)),
                "elo": ratings.get(entry.label),
                "source": "simulated_shotcfadv_same_fixed_pool_200r",
            }
        )
    return rows


def write_manifest(
    output_dir: Path,
    *,
    base_report_path: Path,
    base_metrics_path: Path,
    shot_run_dir: Path,
    eval_entries: list[Entry],
    opponent_entries: list[Entry],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
) -> None:
    manifest = {
        "base_report": str(base_report_path),
        "base_metrics": str(base_metrics_path),
        "shot_run_dir": str(shot_run_dir),
        "eval_rallies_per_pair": eval_rallies,
        "seed": seed,
        "deterministic": deterministic,
        "eval_agents": [entry_payload(entry) for entry in eval_entries],
        "opponent_pool": [entry_payload(entry) for entry in opponent_entries],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_outputs(output_dir: Path, comparison_dir: Path, base_metrics_path: Path, report: dict[str, Any]) -> None:
    (output_dir / "fixed_pool_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_pair_csv(output_dir / "pair_results.csv", report["pair_results"])
    write_eval_metrics_csv(output_dir / "mean_win_rate_elo.csv", report["eval_agent_metrics"])
    write_comparison_csv(
        comparison_dir / "mean_win_rate_elo_shared_old_prefix_dense_new_with_shotcfadv.csv",
        base_metrics_path,
        report["eval_agent_metrics"],
    )
    rows = read_metrics_csv(comparison_dir / "mean_win_rate_elo_shared_old_prefix_dense_new_with_shotcfadv.csv")
    plot_metric(
        rows,
        metric="elo",
        ylabel="Elo",
        title="Fixed-pool Elo",
        output_path=comparison_dir / "elo_vs_checkpoint_shared_old_prefix_with_shotcfadv.png",
    )
    plot_metric(
        rows,
        metric="mean_pool_win_rate",
        ylabel="Mean win rate vs fixed pool",
        title="Mean fixed-pool win rate",
        output_path=comparison_dir / "mean_win_rate_vs_checkpoint_shared_old_prefix_with_shotcfadv.png",
        y_limits=(0.0, 1.0),
        as_percent=True,
    )


def write_eval_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "display_label",
        "run_label",
        "step",
        "available",
        "model_path",
        "evaluated_pair_count",
        "mean_pool_win_rate",
        "elo",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_comparison_csv(path: Path, base_metrics_path: Path, shot_rows: list[dict[str, Any]]) -> None:
    base_rows = read_metrics_csv(base_metrics_path)
    fieldnames = [
        "label",
        "display_label",
        "run_label",
        "step",
        "available",
        "model_path",
        "evaluated_pair_count",
        "mean_pool_win_rate",
        "elo",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
        for row in shot_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_metric(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
    as_percent: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    styles = {
        "old": ("old", "#4c78a8"),
        "new": ("new (shared old prefix)", "#f58518"),
        "shotcfadv": ("shot CF adv", "#54a24b"),
    }
    for run_label in ["old", "new", "shotcfadv"]:
        run_rows = [
            row
            for row in rows
            if row.get("run_label") == run_label
            and row.get(metric) not in (None, "")
            and row.get("available") in (True, "True", "true", "1", 1)
        ]
        run_rows.sort(key=lambda row: int(row["step"]))
        if not run_rows:
            continue
        label, color = styles[run_label]
        ax.plot(
            [int(row["step"]) / 1_000_000.0 for row in run_rows],
            [float(row[metric]) for row in run_rows],
            marker="o",
            linewidth=2.2,
            markersize=4.7,
            color=color,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if as_percent:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def entry_payload(entry: Entry) -> dict[str, Any]:
    return {
        "label": entry.label,
        "display_label": entry.display_label,
        "run_label": entry.run_label,
        "step": entry.step,
        "run_dir": str(entry.run_dir),
        "model_path": None if entry.model_path is None else str(entry.model_path),
        "available": entry.available,
    }


if __name__ == "__main__":
    main()
