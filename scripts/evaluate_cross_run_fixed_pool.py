from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
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

from badminton.elo import PairwiseRecord, calculate_elo, ratings_table
from badminton.evaluation import ModelSelector, summarize_episodes
from badminton.selfplay import CheckpointPool, FixedCheckpointOpponent, build_selfplay_env
from badminton.state import Side
from badminton.utils import ensure_directory
from scripts.round_robin_selfplay_video import (
    AgentSpec,
    _config_value,
    _load_config,
    _resolve_random_server,
    build_discrete_action_config,
    build_sim_config,
    rollout_rally,
)


@dataclass(frozen=True)
class PoolEntry:
    label: str
    run_label: str
    step: int
    source_run_dir: Path
    source_model_path: Path
    run_dir: Path
    model_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected cross-run checkpoints against a fixed cross-run opponent pool."
    )
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=[400_000, 800_000, 1_200_000, 1_600_000, 2_000_000, 2_400_000, 2_800_000, 3_200_000],
    )
    parser.add_argument(
        "--pool-steps",
        type=int,
        nargs="+",
        default=[800_000, 1_600_000, 2_400_000, 3_200_000],
    )
    parser.add_argument("--eval-rallies", type=int, default=200, help="Total side-balanced rallies per pair.")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--prior-std", type=float, default=400.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_cross_run_fixed_pool(
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
        eval_steps=[int(step) for step in args.eval_steps],
        pool_steps=[int(step) for step in args.pool_steps],
        eval_rallies=int(args.eval_rallies),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        initial_rating=float(args.initial_rating),
        elo_scale=float(args.elo_scale),
        prior_std=float(args.prior_std),
    )
    print(f"manifest: {args.output_dir / 'manifest.json'}")
    print(f"report: {args.output_dir / 'fixed_pool_eval_report.json'}")
    print(f"win_rates: {args.output_dir / 'pair_win_rates.csv'}")
    print(f"ratings: {args.output_dir / 'elo_ratings.csv'}")
    print(f"evaluated_pairs: {len(report['pair_results'])}")


def evaluate_cross_run_fixed_pool(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    eval_steps: list[int],
    pool_steps: list[int],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
) -> dict[str, Any]:
    if eval_rallies <= 0:
        raise ValueError("--eval-rallies must be positive")
    if not run_dirs:
        raise ValueError("At least one run directory is required")

    ensure_directory(output_dir)
    eval_pool_dir = output_dir / "eval_pool"
    opponent_pool_dir = output_dir / "opponent_pool"
    ensure_directory(eval_pool_dir)
    ensure_directory(opponent_pool_dir)

    run_labels = unique_run_labels(run_dirs)
    eval_entries = materialize_pool(
        output_dir=eval_pool_dir,
        run_dirs=run_dirs,
        run_labels=run_labels,
        steps=eval_steps,
    )
    opponent_entries = materialize_pool(
        output_dir=opponent_pool_dir,
        run_dirs=run_dirs,
        run_labels=run_labels,
        steps=pool_steps,
    )
    write_manifest(output_dir, run_dirs, eval_entries, opponent_entries, eval_steps, pool_steps, eval_rallies, seed, deterministic)

    configs = {entry.label: _load_config(entry.run_dir) for entry in eval_entries}
    base_config = _load_config(eval_entries[0].run_dir)
    sim_config = build_sim_config(base_config)
    discrete_action_config = build_discrete_action_config(base_config)
    models = {entry.label: PPO.load(entry.model_path) for entry in eval_entries}

    partial_path = output_dir / "pair_results.jsonl"
    pair_results = load_partial_results(partial_path)
    completed = {(str(row["agent"]), str(row["opponent"])) for row in pair_results}
    eval_agents = [AgentSpec(label=entry.label, run_dir=entry.run_dir, model_path=entry.model_path) for entry in eval_entries]
    opponents = [AgentSpec(label=entry.label, run_dir=entry.run_dir, model_path=entry.model_path) for entry in opponent_entries]
    per_side = side_balanced_counts(eval_rallies)

    for agent_index, agent in enumerate(eval_agents):
        for opponent_index, opponent in enumerate(opponents):
            if agent.label == opponent.label:
                continue
            if (agent.label, opponent.label) in completed:
                print(f"{agent.label} vs {opponent.label}: already complete", flush=True)
                continue
            pair_seed = seed + agent_index * 1_000_000 + opponent_index * 10_000
            left_summary = evaluate_side(
                agent=agent,
                opponent=opponent,
                train_side="left",
                model=models[agent.label],
                train_config=configs[agent.label],
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                episodes=per_side[0],
                seed=pair_seed + 101,
                deterministic=deterministic,
            )
            right_summary = evaluate_side(
                agent=agent,
                opponent=opponent,
                train_side="right",
                model=models[agent.label],
                train_config=configs[agent.label],
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                episodes=per_side[1],
                seed=pair_seed + 202,
                deterministic=deterministic,
            )
            pair = combine_side_summaries(
                agent_entry=entry_by_label(eval_entries, agent.label),
                opponent_entry=entry_by_label(opponent_entries, opponent.label),
                left_summary=left_summary,
                right_summary=right_summary,
            )
            pair_results.append(pair)
            append_partial_result(partial_path, pair)
            completed.add((agent.label, opponent.label))
            print(f"{agent.label} vs {opponent.label}: wr={pair['agent_win_rate']:.3f} ({pair['episodes']} rallies)", flush=True)

    report = build_report(
        output_dir=output_dir,
        run_dirs=run_dirs,
        eval_entries=eval_entries,
        opponent_entries=opponent_entries,
        pair_results=pair_results,
        eval_rallies=eval_rallies,
        per_side=per_side,
        seed=seed,
        deterministic=deterministic,
        initial_rating=initial_rating,
        elo_scale=elo_scale,
        prior_std=prior_std,
    )
    write_outputs(output_dir, report)
    return report


def unique_run_labels(run_dirs: list[Path]) -> dict[Path, str]:
    labels: dict[Path, str] = {}
    seen: set[str] = set()
    for run_dir in run_dirs:
        label = infer_run_label(run_dir)
        base = label
        suffix = 2
        while label in seen:
            label = f"{base}_{suffix}"
            suffix += 1
        labels[run_dir] = label
        seen.add(label)
    return labels


def infer_run_label(run_dir: Path) -> str:
    name = run_dir.name
    preferred = (
        "norecoverycfadv",
        "recoverycfdefault",
        "norecovery",
        "recovery",
    )
    for token in preferred:
        if token in name:
            return token
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return cleaned[:40] or "run"


def materialize_pool(
    *,
    output_dir: Path,
    run_dirs: list[Path],
    run_labels: dict[Path, str],
    steps: list[int],
) -> list[PoolEntry]:
    entries: list[PoolEntry] = []
    for run_dir in run_dirs:
        config_path = run_dir / "selfplay_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing run config: {config_path}")
        run_label = run_labels[run_dir]
        for step in steps:
            source_model = run_dir / "anchor_checkpoints" / f"anchor_step_{int(step)}.zip"
            if not source_model.exists():
                raise FileNotFoundError(f"Missing checkpoint: {source_model}")
            label = f"{run_label}_{step_label(step)}"
            agent_dir = output_dir / label
            ensure_directory(agent_dir)
            model_path = agent_dir / "final_model.zip"
            link_or_copy(source_model, model_path)
            replace_symlink(agent_dir / "selfplay_config.json", config_path)
            entries.append(
                PoolEntry(
                    label=label,
                    run_label=run_label,
                    step=int(step),
                    source_run_dir=run_dir,
                    source_model_path=source_model,
                    run_dir=agent_dir,
                    model_path=model_path,
                )
            )
    return entries


def step_label(step: int) -> str:
    return f"{step / 1_000_000.0:.1f}m".replace(".", "p")


def link_or_copy(source_path: Path, target_path: Path) -> None:
    ensure_directory(target_path.parent)
    if target_path.exists():
        try:
            if target_path.samefile(source_path):
                return
        except OSError:
            pass
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
    relative_target = Path(os.path.relpath(target, start=path.parent))
    path.symlink_to(relative_target)


def write_manifest(
    output_dir: Path,
    run_dirs: list[Path],
    eval_entries: list[PoolEntry],
    opponent_entries: list[PoolEntry],
    eval_steps: list[int],
    pool_steps: list[int],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
) -> None:
    manifest = {
        "run_dirs": [str(path) for path in run_dirs],
        "eval_steps": eval_steps,
        "pool_steps": pool_steps,
        "eval_rallies_per_pair": eval_rallies,
        "seed": seed,
        "deterministic": deterministic,
        "eval_agents": [entry_payload(entry) for entry in eval_entries],
        "opponent_pool": [entry_payload(entry) for entry in opponent_entries],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def entry_payload(entry: PoolEntry) -> dict[str, Any]:
    return {
        "label": entry.label,
        "run_label": entry.run_label,
        "step": entry.step,
        "source_run_dir": str(entry.source_run_dir),
        "source_model_path": str(entry.source_model_path),
        "run_dir": str(entry.run_dir),
        "model_path": str(entry.model_path),
    }


def side_balanced_counts(total: int) -> tuple[int, int]:
    left = max(int(total) // 2, 1)
    right = max(int(total) - left, 1)
    return left, right


def evaluate_side(
    *,
    agent: AgentSpec,
    opponent: AgentSpec,
    train_side: Side,
    model: PPO,
    train_config: dict[str, Any],
    sim_config: Any,
    discrete_action_config: Any,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> dict[str, object]:
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
        policy_type=str(_config_value(train_config, "policy_type", "velocity_oriented")),
        deterministic=deterministic,
    )
    env = build_selfplay_env(
        train_side=train_side,
        mirror_train_side=False,
        mirror_match_fraction=0.0,
        initial_server="random",
        random_service_x=bool(_config_value(train_config, "random_service_x", True)),
        sim_config=sim_config,
        train_reaction_time=float(_config_value(train_config, "reaction_time", 0.15)),
        opponent_reaction_time=float(_config_value(train_config, "opponent_reaction_time", 0.15, "reaction_time")),
        max_stages_per_rally=int(_config_value(train_config, "max_stages_per_rally", 120, "max_rally_stages")),
        policy_type=str(_config_value(train_config, "policy_type", "velocity_oriented")),
        seed=seed,
        discrete_action_config=discrete_action_config,
        opponent=opponent_policy,
        include_records_in_info=False,
        recovery_counterfactual_other_sample_count=0,
        recovery_counterfactual_expected_response_target=False,
    )
    selector = ModelSelector(model=model, deterministic=deterministic)
    rng = np.random.default_rng(seed)
    results = [
        rollout_rally(env, selector, seed=seed + episode, server=_resolve_random_server(rng))
        for episode in range(max(int(episodes), 1))
    ]
    summary = summarize_episodes(results)
    summary["name"] = f"{agent.label}_as_{train_side}_vs_{opponent.label}"
    summary["agent"] = agent.label
    summary["opponent"] = opponent.label
    summary["train_side"] = train_side
    env.close()
    return summary


def entry_by_label(entries: list[PoolEntry], label: str) -> PoolEntry:
    for entry in entries:
        if entry.label == label:
            return entry
    raise KeyError(label)


def combine_side_summaries(
    *,
    agent_entry: PoolEntry,
    opponent_entry: PoolEntry,
    left_summary: dict[str, object],
    right_summary: dict[str, object],
) -> dict[str, Any]:
    total = int(left_summary["episodes"]) + int(right_summary["episodes"])
    wins = (
        float(left_summary["win_rate"]) * int(left_summary["episodes"])
        + float(right_summary["win_rate"]) * int(right_summary["episodes"])
    )
    win_rate = wins / max(total, 1)
    return {
        "agent": agent_entry.label,
        "agent_run_label": agent_entry.run_label,
        "agent_step": agent_entry.step,
        "opponent": opponent_entry.label,
        "opponent_run_label": opponent_entry.run_label,
        "opponent_step": opponent_entry.step,
        "episodes": total,
        "agent_wins": wins,
        "agent_win_rate": win_rate,
        "opponent_win_rate": 1.0 - win_rate,
        "agent_as_left": left_summary,
        "agent_as_right": right_summary,
    }


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


def build_report(
    *,
    output_dir: Path,
    run_dirs: list[Path],
    eval_entries: list[PoolEntry],
    opponent_entries: list[PoolEntry],
    pair_results: list[dict[str, Any]],
    eval_rallies: int,
    per_side: tuple[int, int],
    seed: int,
    deterministic: bool,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
) -> dict[str, Any]:
    records = [
        PairwiseRecord(
            agent_a=str(row["agent"]),
            agent_b=str(row["opponent"]),
            agent_a_score=float(row["agent_wins"]),
            games=float(row["episodes"]),
        )
        for row in pair_results
    ]
    ratings = calculate_elo(records, initial_rating=initial_rating, scale=elo_scale, prior_std=prior_std)
    opponent_labels = {entry.label for entry in opponent_entries}
    elo_rows = []
    for standing in ratings_table(ratings):
        label = str(standing["agent"])
        entry = entry_by_label(eval_entries, label)
        elo_rows.append(
            {
                **standing,
                "run_label": entry.run_label,
                "step": entry.step,
                "step_millions": entry.step / 1_000_000.0,
                "in_opponent_pool": label in opponent_labels,
                "mean_pool_win_rate": mean_pool_win_rate(pair_results, label),
            }
        )
    return {
        "run_dirs": [str(path) for path in run_dirs],
        "output_dir": str(output_dir),
        "seed": seed,
        "deterministic": deterministic,
        "eval_rallies_per_pair": eval_rallies,
        "eval_rallies_per_side": list(per_side),
        "initial_rating": initial_rating,
        "elo_scale": elo_scale,
        "prior_std": prior_std,
        "eval_agents": [entry_payload(entry) for entry in eval_entries],
        "opponent_pool": [entry_payload(entry) for entry in opponent_entries],
        "pair_results": pair_results,
        "elo_ratings": elo_rows,
    }


def mean_pool_win_rate(pair_results: list[dict[str, Any]], label: str) -> float | None:
    values = [float(row["agent_win_rate"]) for row in pair_results if str(row["agent"]) == label]
    if not values:
        return None
    return float(sum(values) / len(values))


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "fixed_pool_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_pair_csv(output_dir / "pair_win_rates.csv", report["pair_results"])
    write_elo_csv(output_dir / "elo_ratings.csv", report["elo_ratings"])


def write_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "agent",
        "agent_run_label",
        "agent_step",
        "opponent",
        "opponent_run_label",
        "opponent_step",
        "episodes",
        "agent_wins",
        "agent_win_rate",
        "opponent_win_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_elo_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "agent",
        "run_label",
        "step",
        "step_millions",
        "in_opponent_pool",
        "elo",
        "mean_pool_win_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    main()
