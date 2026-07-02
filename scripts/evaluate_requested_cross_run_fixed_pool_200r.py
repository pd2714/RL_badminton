from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.elo import PairwiseRecord, calculate_elo, ratings_table
from badminton1d.eval_evolution import build_discrete_action_config, build_sim_config
from badminton1d.evaluation import ModelSelector, rollout_episode, summarize_episodes
from badminton1d.selfplay import CheckpointPool, FixedCheckpointOpponent, build_selfplay_env
from badminton1d.utils import ensure_directory


OLD_RUN = Path(
    "outputs/rl/selfplay_2d_recoverycfdefault_resp1_2m_heuristicbase_ent002_speed100_anchor100k_fullrec24_20260603"
)
NEW_RUN = Path(
    "outputs/rl/selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
)
DEFAULT_OUTPUT_DIR = NEW_RUN / "cross_run_fixed_pool_eval_200r"
DEFAULT_OLD_MATRIX = OLD_RUN / "anchor_metric_eval_200r" / "win_rate_matrix.csv"
DEFAULT_ZERO_MODEL = OLD_RUN / "recovery_ablation_fixed_pool" / "source_checkpoints" / "compatible_zero_model.zip"

OLD_OPPONENT_STEPS = list(range(0, 6_000_001, 600_000))
NEW_OPPONENT_STEPS = [3_600_000, 4_200_000, 4_800_000, 5_400_000, 6_000_000]
OLD_EVAL_STEPS = sorted(set(range(0, 6_000_001, 400_000)) | set(range(3_000_000, 5_800_001, 400_000)))
NEW_EVAL_STEPS = [3_200_000, 3_600_000, 4_000_000, 4_400_000, 4_800_000, 5_200_000, 5_600_000, 6_000_000]


@dataclass(frozen=True)
class Entry:
    label: str
    display_label: str
    run_label: str
    step: int
    run_dir: Path
    model_path: Path | None
    available: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the requested old/new checkpoint pool against a fixed opponent pool."
    )
    parser.add_argument("--old-run-dir", type=Path, default=OLD_RUN)
    parser.add_argument("--new-run-dir", type=Path, default=NEW_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-win-rate-matrix", type=Path, default=DEFAULT_OLD_MATRIX)
    parser.add_argument("--zero-model", type=Path, default=DEFAULT_ZERO_MODEL)
    parser.add_argument("--eval-rallies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--prior-std", type=float, default=400.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_requested_pool(
        old_run_dir=args.old_run_dir,
        new_run_dir=args.new_run_dir,
        output_dir=args.output_dir,
        old_win_rate_matrix=args.old_win_rate_matrix,
        zero_model=args.zero_model,
        eval_rallies=int(args.eval_rallies),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        initial_rating=float(args.initial_rating),
        elo_scale=float(args.elo_scale),
        prior_std=float(args.prior_std),
        dry_run=bool(args.dry_run),
    )
    print(f"manifest: {args.output_dir / 'manifest.json'}")
    print(f"report: {args.output_dir / 'fixed_pool_eval_report.json'}")
    print(f"pairs: {args.output_dir / 'pair_results.csv'}")
    print(f"ratings: {args.output_dir / 'mean_win_rate_elo.csv'}")
    print(
        "pair_counts: "
        f"available={report['available_pair_count']} "
        f"cached={report['cached_pair_count']} "
        f"completed={report['completed_pair_count']} "
        f"remaining={report['remaining_pair_count']}"
    )


def evaluate_requested_pool(
    *,
    old_run_dir: Path,
    new_run_dir: Path,
    output_dir: Path,
    old_win_rate_matrix: Path,
    zero_model: Path,
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

    train_config = load_json(old_run_dir / "selfplay_config.json")
    sim_config = build_sim_config(train_config)
    discrete_action_config = build_discrete_action_config(train_config)

    eval_entries = (
        build_entries("old", old_run_dir, OLD_EVAL_STEPS, zero_model=zero_model)
        + build_entries("new", new_run_dir, NEW_EVAL_STEPS, zero_model=None)
    )
    opponent_entries = (
        build_entries("old", old_run_dir, OLD_OPPONENT_STEPS, zero_model=zero_model)
        + build_entries("new", new_run_dir, NEW_OPPONENT_STEPS, zero_model=None)
    )
    available_eval = [entry for entry in eval_entries if entry.available]
    available_opponents = [entry for entry in opponent_entries if entry.available]
    available_pairs = {(agent.label, opponent.label) for agent in available_eval for opponent in available_opponents}

    write_manifest(
        output_dir,
        old_run_dir=old_run_dir,
        new_run_dir=new_run_dir,
        old_win_rate_matrix=old_win_rate_matrix,
        zero_model=zero_model,
        eval_entries=eval_entries,
        opponent_entries=opponent_entries,
        eval_rallies=eval_rallies,
        seed=seed,
        deterministic=deterministic,
    )

    partial_path = output_dir / "pair_results.jsonl"
    pair_results = load_partial_results(partial_path)
    completed = {(str(row["agent"]), str(row["opponent"])) for row in pair_results}
    cached_added = seed_from_old_matrix(
        old_win_rate_matrix,
        partial_path=partial_path,
        pair_results=pair_results,
        completed=completed,
        available_pairs=available_pairs,
        eval_rallies=eval_rallies,
    )

    if dry_run:
        report = build_report(
            output_dir=output_dir,
            old_run_dir=old_run_dir,
            new_run_dir=new_run_dir,
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
            cached_pair_count=cached_added,
        )
        write_outputs(output_dir, report)
        return report

    model_cache: dict[Path, PPO] = {}
    for agent_index, agent in enumerate(available_eval):
        assert agent.model_path is not None
        model = load_model_cached(model_cache, agent.model_path)
        for opponent_index, opponent in enumerate(available_opponents):
            if (agent.label, opponent.label) in completed:
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
                "source": "simulated",
                "summary": summary,
            }
            pair_results.append(pair)
            append_partial_result(partial_path, pair)
            completed.add((agent.label, opponent.label))
            print(
                f"{agent.label} vs {opponent.label}: "
                f"wr={pair['agent_win_rate']:.3f} ({pair['episodes']} rallies)",
                flush=True,
            )

    report = build_report(
        output_dir=output_dir,
        old_run_dir=old_run_dir,
        new_run_dir=new_run_dir,
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
        cached_pair_count=cached_added,
    )
    write_outputs(output_dir, report)
    return report


def build_entries(run_label: str, run_dir: Path, steps: list[int], *, zero_model: Path | None) -> list[Entry]:
    entries: list[Entry] = []
    for step in steps:
        if step == 0 and zero_model is not None:
            model_path = zero_model
        else:
            model_path = run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip"
        available = model_path.exists()
        entries.append(
            Entry(
                label=f"{run_label}_{step}",
                display_label=f"{run_label}_{step / 1_000_000.0:.1f}M",
                run_label=run_label,
                step=int(step),
                run_dir=run_dir,
                model_path=model_path if available else None,
                available=available,
            )
        )
    return entries


def evaluate_pair(
    *,
    agent: Entry,
    opponent: Entry,
    model: PPO,
    train_config: dict[str, Any],
    sim_config: Any,
    discrete_action_config: Any,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> dict[str, object]:
    assert opponent.model_path is not None
    pool = CheckpointPool(
        checkpoint_dir=opponent.model_path.parent,
        pool_size=1,
        sampling_mode="newest",
        seed=seed + 17,
    )
    opponent_policy = FixedCheckpointOpponent(
        pool=pool,
        checkpoint_path=opponent.model_path,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        policy_type=str(config_value(train_config, "policy_type", "velocity_oriented")),
        deterministic=deterministic,
    )
    env = build_selfplay_env(
        train_side=str(config_value(train_config, "train_side", "left")),
        mirror_train_side=False,
        mirror_match_fraction=0.0,
        initial_server=str(config_value(train_config, "initial_server", "random")),
        random_service_x=bool(config_value(train_config, "random_service_x", True)),
        sim_config=sim_config,
        train_reaction_time=float(config_value(train_config, "reaction_time", 0.15)),
        opponent_reaction_time=float(config_value(train_config, "opponent_reaction_time", "reaction_time", 0.15)),
        max_stages_per_rally=int(config_value(train_config, "max_stages_per_rally", "max_rally_stages", 120)),
        policy_type=str(config_value(train_config, "policy_type", "velocity_oriented")),
        seed=seed,
        discrete_action_config=discrete_action_config,
        opponent=opponent_policy,
        include_records_in_info=False,
        recovery_counterfactual_other_sample_count=0,
        recovery_counterfactual_expected_response_target=False,
    )
    selector = ModelSelector(model=model, deterministic=deterministic)
    try:
        results = [rollout_episode(env, selector, seed=seed + episode) for episode in range(episodes)]
        summary = summarize_episodes(results)
    finally:
        env.close()
    summary["agent"] = agent.label
    summary["opponent"] = opponent.label
    return summary


def seed_from_old_matrix(
    matrix_path: Path,
    *,
    partial_path: Path,
    pair_results: list[dict[str, Any]],
    completed: set[tuple[str, str]],
    available_pairs: set[tuple[str, str]],
    eval_rallies: int,
) -> int:
    if not matrix_path.exists():
        return 0
    added = 0
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            agent = f"old_{step}"
            for key, raw_value in row.items():
                if not key.startswith("anchor_") or raw_value in (None, ""):
                    continue
                opponent_step = int(key.removeprefix("anchor_"))
                opponent = f"old_{opponent_step}"
                pair_key = (agent, opponent)
                if pair_key not in available_pairs or pair_key in completed:
                    continue
                win_rate = float(raw_value)
                pair = {
                    "agent": agent,
                    "agent_display_label": f"old_{step / 1_000_000.0:.1f}M",
                    "agent_run_label": "old",
                    "agent_step": step,
                    "opponent": opponent,
                    "opponent_display_label": f"old_{opponent_step / 1_000_000.0:.1f}M",
                    "opponent_run_label": "old",
                    "opponent_step": opponent_step,
                    "episodes": int(eval_rallies),
                    "agent_wins": win_rate * float(eval_rallies),
                    "agent_win_rate": win_rate,
                    "opponent_win_rate": 1.0 - win_rate,
                    "source": "cached_old_anchor_metric_eval_200r",
                }
                pair_results.append(pair)
                append_partial_result(partial_path, pair)
                completed.add(pair_key)
                added += 1
    return added


def build_report(
    *,
    output_dir: Path,
    old_run_dir: Path,
    new_run_dir: Path,
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
    cached_pair_count: int,
) -> dict[str, Any]:
    completed = {(str(row["agent"]), str(row["opponent"])) for row in pair_results}
    ratings = estimate_ratings(
        pair_results,
        initial_rating=initial_rating,
        elo_scale=elo_scale,
        prior_std=prior_std,
    )
    eval_rows = build_eval_rows(eval_entries, pair_results, ratings)
    matrix_report = build_matrix(eval_entries, opponent_entries, pair_results)
    return {
        "old_run_dir": str(old_run_dir),
        "new_run_dir": str(new_run_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "deterministic": deterministic,
        "eval_rallies_per_pair": eval_rallies,
        "initial_rating": initial_rating,
        "elo_scale": elo_scale,
        "prior_std": prior_std,
        "available_pair_count": len(available_pairs),
        "cached_pair_count": sum(1 for row in pair_results if row.get("source") == "cached_old_anchor_metric_eval_200r"),
        "cached_pair_count_added_this_run": cached_pair_count,
        "completed_pair_count": len(completed & available_pairs),
        "remaining_pair_count": len(available_pairs - completed),
        "missing_eval_entries": [entry_payload(entry) for entry in eval_entries if not entry.available],
        "missing_opponent_entries": [entry_payload(entry) for entry in opponent_entries if not entry.available],
        "eval_agents": [entry_payload(entry) for entry in eval_entries],
        "opponent_pool": [entry_payload(entry) for entry in opponent_entries],
        "pair_results": pair_results,
        "elo_standings": ratings_table(ratings) if ratings else [],
        "eval_agent_metrics": eval_rows,
        "win_rate_matrix": matrix_report,
    }


def estimate_ratings(
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
                "mean_pool_win_rate": None if not values else float(sum(values) / len(values)),
                "evaluated_pair_count": len(values),
                "elo": ratings.get(entry.label),
            }
        )
    return rows


def build_matrix(
    eval_entries: list[Entry],
    opponent_entries: list[Entry],
    pair_results: list[dict[str, Any]],
) -> dict[str, Any]:
    values = {(str(row["agent"]), str(row["opponent"])): float(row["agent_win_rate"]) for row in pair_results}
    episodes = {(str(row["agent"]), str(row["opponent"])): int(row["episodes"]) for row in pair_results}
    return {
        "row_labels": [entry.label for entry in eval_entries],
        "row_display_labels": [entry.display_label for entry in eval_entries],
        "row_steps": [entry.step for entry in eval_entries],
        "col_labels": [entry.label for entry in opponent_entries],
        "col_display_labels": [entry.display_label for entry in opponent_entries],
        "col_steps": [entry.step for entry in opponent_entries],
        "win_rate_matrix": [
            [values.get((agent.label, opponent.label)) for opponent in opponent_entries]
            for agent in eval_entries
        ],
        "rally_count_matrix": [
            [episodes.get((agent.label, opponent.label)) for opponent in opponent_entries]
            for agent in eval_entries
        ],
    }


def write_manifest(
    output_dir: Path,
    *,
    old_run_dir: Path,
    new_run_dir: Path,
    old_win_rate_matrix: Path,
    zero_model: Path,
    eval_entries: list[Entry],
    opponent_entries: list[Entry],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
) -> None:
    manifest = {
        "old_run_dir": str(old_run_dir),
        "new_run_dir": str(new_run_dir),
        "old_win_rate_matrix_cache": str(old_win_rate_matrix),
        "zero_model": str(zero_model),
        "eval_rallies_per_pair": eval_rallies,
        "seed": seed,
        "deterministic": deterministic,
        "eval_agents": [entry_payload(entry) for entry in eval_entries],
        "opponent_pool": [entry_payload(entry) for entry in opponent_entries],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "fixed_pool_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_pair_csv(output_dir / "pair_results.csv", report["pair_results"])
    write_eval_metrics_csv(output_dir / "mean_win_rate_elo.csv", report["eval_agent_metrics"])
    write_matrix_csv(output_dir / "win_rate_matrix.csv", report["win_rate_matrix"])
    write_plots(output_dir, report["eval_agent_metrics"])


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [
        row
        for row in rows
        if row.get("available")
        and row.get("mean_pool_win_rate") is not None
        and row.get("elo") is not None
    ]
    plot_two_run_metric(
        plt,
        plot_rows,
        metric="mean_pool_win_rate",
        ylabel="Mean win rate vs fixed pool",
        title="Mean fixed-pool win rate",
        output_path=output_dir / "mean_win_rate_vs_checkpoint_two_lines.png",
        y_limits=(0.0, 1.0),
        as_percent=True,
        shared_old_prefix=False,
    )
    plot_two_run_metric(
        plt,
        plot_rows,
        metric="elo",
        ylabel="Elo",
        title="Fixed-pool Elo",
        output_path=output_dir / "elo_vs_checkpoint_two_lines.png",
        y_limits=None,
        as_percent=False,
        shared_old_prefix=False,
    )
    plot_two_run_metric(
        plt,
        plot_rows,
        metric="mean_pool_win_rate",
        ylabel="Mean win rate vs fixed pool",
        title="Mean fixed-pool win rate",
        output_path=output_dir / "mean_win_rate_vs_checkpoint_two_lines_shared_old_prefix.png",
        y_limits=(0.0, 1.0),
        as_percent=True,
        shared_old_prefix=True,
    )
    plot_two_run_metric(
        plt,
        plot_rows,
        metric="elo",
        ylabel="Elo",
        title="Fixed-pool Elo",
        output_path=output_dir / "elo_vs_checkpoint_two_lines_shared_old_prefix.png",
        y_limits=None,
        as_percent=False,
        shared_old_prefix=True,
    )


def plot_two_run_metric(
    plt: Any,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y_limits: tuple[float, float] | None,
    as_percent: bool,
    shared_old_prefix: bool,
) -> None:
    old_rows = sorted((row for row in rows if row["run_label"] == "old"), key=lambda row: int(row["step"]))
    new_rows = sorted((row for row in rows if row["run_label"] == "new"), key=lambda row: int(row["step"]))
    if shared_old_prefix:
        new_rows = [row for row in old_rows if int(row["step"]) <= 3_000_000] + new_rows
        new_label = "new (shared old prefix)"
    else:
        new_label = "new"

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    draw_metric_line(ax, old_rows, metric=metric, label="old", color="#4c78a8")
    draw_metric_line(ax, new_rows, metric=metric, label=new_label, color="#f58518")
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


def draw_metric_line(ax: Any, rows: list[dict[str, Any]], *, metric: str, label: str, color: str) -> None:
    if not rows:
        return
    x = [int(row["step"]) / 1_000_000.0 for row in rows]
    y = [float(row[metric]) for row in rows]
    ax.plot(x, y, marker="o", linewidth=2.2, markersize=4.6, color=color, label=label)


def write_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "agent",
        "agent_display_label",
        "agent_run_label",
        "agent_step",
        "opponent",
        "opponent_display_label",
        "opponent_run_label",
        "opponent_step",
        "episodes",
        "agent_wins",
        "agent_win_rate",
        "opponent_win_rate",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_matrix_csv(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", "display_label", "step", *report["col_labels"]])
        for label, display, step, values in zip(
            report["row_labels"],
            report["row_display_labels"],
            report["row_steps"],
            report["win_rate_matrix"],
        ):
            writer.writerow([label, display, step, *["" if value is None else value for value in values]])


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_partial_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def append_partial_result(path: Path, pair: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(pair) + "\n")


def load_model_cached(cache: dict[Path, PPO], path: Path) -> PPO:
    resolved = path.resolve()
    model = cache.get(resolved)
    if model is None:
        model = PPO.load(resolved)
        cache[resolved] = model
    return model


def config_value(data: dict[str, Any], *keys: object) -> Any:
    default = keys[-1]
    for key in keys[:-1]:
        if key in data:
            return data[key]
    return default


if __name__ == "__main__":
    main()
