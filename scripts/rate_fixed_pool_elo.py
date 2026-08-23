from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.elo import PairwiseRecord, calculate_elo, ratings_table
from badminton.utils import ensure_directory
from scripts.round_robin_selfplay_video import (
    AgentSpec,
    _load_config,
    build_discrete_action_config,
    build_sim_config,
    combine_pair_summary,
    evaluate_ordered_matchup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate a fixed checkpoint pool with side-balanced Elo.")
    parser.add_argument("pool_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default="final_model.zip")
    parser.add_argument("--eval-rallies-per-side", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--prior-std", type=float, default=400.0)
    return parser.parse_args()


def discover_pool_agents(pool_dir: Path, checkpoint_name: str) -> list[AgentSpec]:
    manifest_path = pool_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("agents", [])
        if not isinstance(entries, list):
            raise ValueError(f"Invalid agents list in {manifest_path}")
        run_dirs = [pool_dir / str(entry["run_dir"]) for entry in entries]
    else:
        run_dirs = sorted((path for path in pool_dir.iterdir() if path.is_dir()), key=_pool_sort_key)

    agents: list[AgentSpec] = []
    seen: set[str] = set()
    for run_dir in run_dirs:
        label = run_dir.name
        model_path = run_dir / checkpoint_name
        config_path = run_dir / "selfplay_config.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing selfplay config: {config_path}")
        if label in seen:
            raise ValueError(f"Duplicate agent label: {label}")
        agents.append(AgentSpec(label=label, run_dir=run_dir, model_path=model_path))
        seen.add(label)

    if len(agents) < 2:
        raise ValueError("At least two pool agents are required.")
    return agents


def _pool_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    if match is None:
        return (0, path.name)
    return (int(match.group(1)), path.name)


def rate_fixed_pool(
    *,
    pool_dir: Path,
    output_dir: Path,
    checkpoint_name: str = "final_model.zip",
    eval_rallies_per_side: int = 100,
    seed: int = 20260511,
    deterministic: bool = False,
    initial_rating: float = 1500.0,
    elo_scale: float = 400.0,
    prior_std: float = 400.0,
) -> dict[str, Any]:
    if eval_rallies_per_side <= 0:
        raise ValueError("eval_rallies_per_side must be positive")

    ensure_directory(output_dir)
    agents = discover_pool_agents(pool_dir, checkpoint_name)
    configs = {agent.label: _load_config(agent.run_dir) for agent in agents}
    base_config = configs[agents[0].label]
    sim_config = build_sim_config(base_config)
    discrete_action_config = build_discrete_action_config(base_config)
    models = {agent.label: PPO.load(agent.model_path) for agent in agents}

    pair_summaries: list[dict[str, object]] = []
    elo_records: list[PairwiseRecord] = []

    for pair_index, (i, j) in enumerate((i, j) for i in range(len(agents)) for j in range(i + 1, len(agents))):
        agent_a = agents[i]
        agent_b = agents[j]
        seed_base = seed + pair_index * 100_000
        a_left_summary, _ = evaluate_ordered_matchup(
            agent=agent_a,
            opponent=agent_b,
            train_side="left",
            model=models[agent_a.label],
            train_config=configs[agent_a.label],
            sim_config=sim_config,
            discrete_action_config=discrete_action_config,
            episodes=eval_rallies_per_side,
            seed=seed_base + 1_000,
            deterministic=deterministic,
        )
        a_right_summary, _ = evaluate_ordered_matchup(
            agent=agent_a,
            opponent=agent_b,
            train_side="right",
            model=models[agent_a.label],
            train_config=configs[agent_a.label],
            sim_config=sim_config,
            discrete_action_config=discrete_action_config,
            episodes=eval_rallies_per_side,
            seed=seed_base + 2_000,
            deterministic=deterministic,
        )
        pair_summary = combine_pair_summary(
            agent_a=agent_a,
            agent_b=agent_b,
            a_left_summary=a_left_summary,
            a_right_summary=a_right_summary,
        )
        pair_summaries.append(pair_summary)

        games = float(pair_summary["episodes"])
        elo_records.append(
            PairwiseRecord(
                agent_a=agent_a.label,
                agent_b=agent_b.label,
                agent_a_score=float(pair_summary["agent_a_win_rate"]) * games,
                games=games,
            )
        )
        print(
            f"{pair_summary['pair']}: {agent_a.label} wr={float(pair_summary['agent_a_win_rate']):.3f}, "
            f"{agent_b.label} wr={float(pair_summary['agent_b_win_rate']):.3f}",
            flush=True,
        )

    ratings = calculate_elo(
        elo_records,
        initial_rating=initial_rating,
        scale=elo_scale,
        prior_std=prior_std,
    )
    standings = ratings_table(ratings)
    report = {
        "pool_dir": str(pool_dir),
        "checkpoint_name": checkpoint_name,
        "seed": seed,
        "deterministic": deterministic,
        "eval_rallies_per_side": eval_rallies_per_side,
        "eval_rallies_per_pair": eval_rallies_per_side * 2,
        "initial_rating": initial_rating,
        "elo_scale": elo_scale,
        "prior_std": prior_std,
        "agents": [
            {"label": agent.label, "run_dir": str(agent.run_dir), "model_path": str(agent.model_path)}
            for agent in agents
        ],
        "standings": standings,
        "pair_summaries": pair_summaries,
    }
    write_report(output_dir, report)
    return report


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    report_path = output_dir / "elo_rating_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output_dir / "elo_ratings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "agent", "step", "elo"])
        writer.writeheader()
        for item in report["standings"]:
            row = dict(item)
            match = re.search(r"(\d+)$", str(row["agent"]))
            row["step"] = "" if match is None else int(match.group(1))
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.pool_dir / "elo_rating")
    report = rate_fixed_pool(
        pool_dir=args.pool_dir,
        output_dir=output_dir,
        checkpoint_name=args.checkpoint_name,
        eval_rallies_per_side=args.eval_rallies_per_side,
        seed=args.seed,
        deterministic=args.deterministic,
        initial_rating=args.initial_rating,
        elo_scale=args.elo_scale,
        prior_std=args.prior_std,
    )
    print(f"report: {output_dir / 'elo_rating_report.json'}")
    print(f"csv: {output_dir / 'elo_ratings.csv'}")
    for item in report["standings"]:
        print(f"{item['rank']:>2}. {item['agent']}: elo={float(item['elo']):.1f}")


if __name__ == "__main__":
    main()
