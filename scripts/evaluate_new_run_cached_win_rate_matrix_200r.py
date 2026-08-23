from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.elo import PairwiseRecord, calculate_elo, ratings_table
from badminton.eval_evolution import build_discrete_action_config, build_sim_config
from badminton.evaluation import ModelSelector, rollout_episode, summarize_episodes
from badminton.mpl_config import ensure_writable_matplotlib_config
from badminton.selfplay import CheckpointPool, FixedCheckpointOpponent, build_selfplay_env
from badminton.utils import ensure_directory


DEFAULT_NEW_RUN = Path(
    "outputs/rl/selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
)
DEFAULT_OLD_MATRIX = Path(
    "outputs/rl/selfplay_2d_recoverycfdefault_resp1_2m_heuristicbase_ent002_speed100_anchor100k_fullrec24_20260603"
) / "anchor_metric_eval_200r" / "win_rate_matrix.csv"
DEFAULT_CROSS_RESULTS = DEFAULT_NEW_RUN / "cross_run_fixed_pool_eval_200r" / "pair_results.csv"


@dataclass(frozen=True)
class Agent:
    label: str
    step: int
    model_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a new-run checkpoint win-rate matrix, reusing old shared-prefix and "
            "cross-run 200-rally caches and simulating only missing pairs."
        )
    )
    parser.add_argument("--new-run-dir", type=Path, default=DEFAULT_NEW_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--old-win-rate-matrix", type=Path, default=DEFAULT_OLD_MATRIX)
    parser.add_argument("--cross-pair-results", type=Path, default=DEFAULT_CROSS_RESULTS)
    parser.add_argument("--step-min", type=int, default=0)
    parser.add_argument("--step-max", type=int, default=5_600_000)
    parser.add_argument("--step-interval", type=int, default=200_000)
    parser.add_argument("--shared-prefix-max", type=int, default=3_000_000)
    parser.add_argument("--eval-rallies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-every", type=int, default=10, help="Refresh CSV/JSON/PNG after this many new simulated pairs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.new_run_dir / "anchor_metric_eval" / "cached_win_rate_matrix_200r")
    report = evaluate(
        new_run_dir=args.new_run_dir,
        output_dir=output_dir,
        old_win_rate_matrix=args.old_win_rate_matrix,
        cross_pair_results=args.cross_pair_results,
        step_min=int(args.step_min),
        step_max=int(args.step_max),
        step_interval=int(args.step_interval),
        shared_prefix_max=int(args.shared_prefix_max),
        eval_rallies=int(args.eval_rallies),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        dry_run=bool(args.dry_run),
        write_every=int(args.write_every),
    )
    print(f"output: {output_dir}")
    print(f"pairs: {output_dir / 'pair_results.csv'}")
    print(f"matrix: {output_dir / 'win_rate_matrix.csv'}")
    print(f"matrix_plot: {output_dir / 'win_rate_matrix.png'}")
    print(
        "counts: "
        f"available={report['available_pair_count']} "
        f"cached={report['cached_pair_count']} "
        f"simulated={report['simulated_pair_count']} "
        f"remaining={report['remaining_pair_count']}"
    )


def evaluate(
    *,
    new_run_dir: Path,
    output_dir: Path,
    old_win_rate_matrix: Path,
    cross_pair_results: Path,
    step_min: int,
    step_max: int,
    step_interval: int,
    shared_prefix_max: int,
    eval_rallies: int,
    seed: int,
    deterministic: bool,
    dry_run: bool,
    write_every: int,
) -> dict[str, Any]:
    if eval_rallies <= 0:
        raise ValueError("--eval-rallies must be positive")
    if step_interval <= 0:
        raise ValueError("--step-interval must be positive")
    ensure_directory(output_dir)

    agents = discover_new_agents(new_run_dir, step_min=step_min, step_max=step_max, step_interval=step_interval)
    available_pairs = {(agent.label, opponent.label) for agent in agents for opponent in agents}
    partial_path = output_dir / "pair_results.jsonl"
    pair_results = load_partial_results(partial_path)
    completed = {(str(row["agent"]), str(row["opponent"])) for row in pair_results}

    cached_old = seed_from_old_matrix(
        old_win_rate_matrix,
        partial_path=partial_path,
        pair_results=pair_results,
        completed=completed,
        available_pairs=available_pairs,
        eval_rallies=eval_rallies,
        shared_prefix_max=shared_prefix_max,
    )
    cached_cross = seed_from_cross_results(
        cross_pair_results,
        partial_path=partial_path,
        pair_results=pair_results,
        completed=completed,
        available_pairs=available_pairs,
        eval_rallies=eval_rallies,
        shared_prefix_max=shared_prefix_max,
    )
    cached_self = seed_identity_pairs(
        agents,
        partial_path=partial_path,
        pair_results=pair_results,
        completed=completed,
        eval_rallies=eval_rallies,
    )
    cached_added = {"old_matrix": cached_old, "cross_run": cached_cross, "identity": cached_self}

    if not dry_run:
        report = build_report(
            new_run_dir=new_run_dir,
            output_dir=output_dir,
            old_win_rate_matrix=old_win_rate_matrix,
            cross_pair_results=cross_pair_results,
            agents=agents,
            pair_results=pair_results,
            available_pairs=available_pairs,
            eval_rallies=eval_rallies,
            seed=seed,
            deterministic=deterministic,
            cached_added=cached_added,
        )
        write_outputs(output_dir, report)

        run_config = load_json(new_run_dir / "selfplay_config.json")
        sim_config = build_sim_config(run_config)
        discrete_action_config = build_discrete_action_config(run_config)
        model_cache: dict[Path, PPO] = {}
        simulated_since_write = 0
        for agent_index, agent in enumerate(agents):
            model = load_model_cached(model_cache, agent.model_path)
            for opponent_index, opponent in enumerate(agents):
                pair_key = (agent.label, opponent.label)
                if pair_key in completed:
                    continue
                pair_seed = seed + agent_index * 1_000_000 + opponent_index * 10_000
                summary = evaluate_pair(
                    agent=agent,
                    opponent=opponent,
                    model=model,
                    train_config=run_config,
                    sim_config=sim_config,
                    discrete_action_config=discrete_action_config,
                    episodes=eval_rallies,
                    seed=pair_seed,
                    deterministic=deterministic,
                )
                pair = pair_payload(
                    agent,
                    opponent,
                    episodes=int(summary["episodes"]),
                    win_rate=float(summary["win_rate"]),
                    source="simulated_missing_200r",
                    summary=summary,
                )
                pair_results.append(pair)
                append_partial_result(partial_path, pair)
                completed.add(pair_key)
                simulated_since_write += 1
                print(
                    f"{agent.label} vs {opponent.label}: "
                    f"wr={pair['agent_win_rate']:.3f} ({pair['episodes']} rallies)",
                    flush=True,
                )
                if write_every > 0 and simulated_since_write >= write_every:
                    report = build_report(
                        new_run_dir=new_run_dir,
                        output_dir=output_dir,
                        old_win_rate_matrix=old_win_rate_matrix,
                        cross_pair_results=cross_pair_results,
                        agents=agents,
                        pair_results=pair_results,
                        available_pairs=available_pairs,
                        eval_rallies=eval_rallies,
                        seed=seed,
                        deterministic=deterministic,
                        cached_added=cached_added,
                    )
                    write_outputs(output_dir, report)
                    simulated_since_write = 0

    report = build_report(
        new_run_dir=new_run_dir,
        output_dir=output_dir,
        old_win_rate_matrix=old_win_rate_matrix,
        cross_pair_results=cross_pair_results,
        agents=agents,
        pair_results=pair_results,
        available_pairs=available_pairs,
        eval_rallies=eval_rallies,
        seed=seed,
        deterministic=deterministic,
        cached_added=cached_added,
    )
    write_outputs(output_dir, report)
    return report


def discover_new_agents(new_run_dir: Path, *, step_min: int, step_max: int, step_interval: int) -> list[Agent]:
    agents: list[Agent] = []
    for step in range(step_min, step_max + 1, step_interval):
        model_path = new_run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip"
        if model_path.exists():
            agents.append(Agent(label=f"anchor_{step}", step=step, model_path=model_path))
    if not agents:
        raise FileNotFoundError(f"No checkpoints selected from {new_run_dir}")
    return agents


def seed_from_old_matrix(
    matrix_path: Path,
    *,
    partial_path: Path,
    pair_results: list[dict[str, Any]],
    completed: set[tuple[str, str]],
    available_pairs: set[tuple[str, str]],
    eval_rallies: int,
    shared_prefix_max: int,
) -> int:
    if not matrix_path.exists():
        return 0
    added = 0
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            if step > shared_prefix_max:
                continue
            agent = f"anchor_{step}"
            for key, raw_value in row.items():
                if not key.startswith("anchor_") or raw_value in (None, ""):
                    continue
                opponent_step = int(key.removeprefix("anchor_"))
                if opponent_step > shared_prefix_max:
                    continue
                opponent = f"anchor_{opponent_step}"
                pair_key = (agent, opponent)
                if pair_key not in available_pairs or pair_key in completed:
                    continue
                pair = pair_payload(
                    Agent(agent, step, Path()),
                    Agent(opponent, opponent_step, Path()),
                    episodes=eval_rallies,
                    win_rate=float(raw_value),
                    source="cached_old_anchor_metric_eval_200r_shared_prefix",
                )
                pair_results.append(pair)
                append_partial_result(partial_path, pair)
                completed.add(pair_key)
                added += 1
    return added


def seed_from_cross_results(
    path: Path,
    *,
    partial_path: Path,
    pair_results: list[dict[str, Any]],
    completed: set[tuple[str, str]],
    available_pairs: set[tuple[str, str]],
    eval_rallies: int,
    shared_prefix_max: int,
) -> int:
    if not path.exists():
        return 0
    added = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            agent_step = int(row["agent_step"])
            opponent_step = int(row["opponent_step"])
            agent = mapped_label(str(row["agent_run_label"]), agent_step, shared_prefix_max)
            opponent = mapped_label(str(row["opponent_run_label"]), opponent_step, shared_prefix_max)
            if agent is None or opponent is None:
                continue
            pair_key = (agent, opponent)
            if pair_key not in available_pairs or pair_key in completed:
                continue
            episodes = int(float(row.get("episodes") or eval_rallies))
            pair = pair_payload(
                Agent(agent, agent_step, Path()),
                Agent(opponent, opponent_step, Path()),
                episodes=episodes,
                win_rate=float(row["agent_win_rate"]),
                source=f"cached_cross_run_fixed_pool_eval_200r:{row.get('source', '')}",
            )
            pair_results.append(pair)
            append_partial_result(partial_path, pair)
            completed.add(pair_key)
            added += 1
    return added


def seed_identity_pairs(
    agents: list[Agent],
    *,
    partial_path: Path,
    pair_results: list[dict[str, Any]],
    completed: set[tuple[str, str]],
    eval_rallies: int,
) -> int:
    added = 0
    for agent in agents:
        pair_key = (agent.label, agent.label)
        if pair_key in completed:
            continue
        pair = pair_payload(
            agent,
            agent,
            episodes=eval_rallies,
            win_rate=0.5,
            source="identity_same_checkpoint_0p5",
        )
        pair_results.append(pair)
        append_partial_result(partial_path, pair)
        completed.add(pair_key)
        added += 1
    return added


def mapped_label(run_label: str, step: int, shared_prefix_max: int) -> str | None:
    if run_label == "new":
        return f"anchor_{step}"
    if run_label == "old" and step <= shared_prefix_max:
        return f"anchor_{step}"
    return None


def evaluate_pair(
    *,
    agent: Agent,
    opponent: Agent,
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


def build_report(
    *,
    new_run_dir: Path,
    output_dir: Path,
    old_win_rate_matrix: Path,
    cross_pair_results: Path,
    agents: list[Agent],
    pair_results: list[dict[str, Any]],
    available_pairs: set[tuple[str, str]],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
    cached_added: dict[str, int],
) -> dict[str, Any]:
    completed = {(str(row["agent"]), str(row["opponent"])) for row in pair_results}
    ratings = estimate_ratings(pair_results)
    matrix_report = build_matrix(agents, pair_results)
    return {
        "description": (
            "New-run win-rate matrix with cached old shared-prefix and cross-run 200-rally "
            "results reused; only missing pairs are simulated."
        ),
        "new_run_dir": str(new_run_dir),
        "output_dir": str(output_dir),
        "old_win_rate_matrix": str(old_win_rate_matrix),
        "cross_pair_results": str(cross_pair_results),
        "old_0_to_3m_same_as_new_0_to_3m": True,
        "eval_rallies_per_pair": eval_rallies,
        "seed": seed,
        "deterministic": deterministic,
        "available_pair_count": len(available_pairs),
        "completed_pair_count": len(completed & available_pairs),
        "remaining_pair_count": len(available_pairs - completed),
        "cached_added_this_run": cached_added,
        "cached_pair_count": sum(1 for row in pair_results if str(row.get("source", "")).startswith("cached_")),
        "identity_pair_count": sum(1 for row in pair_results if row.get("source") == "identity_same_checkpoint_0p5"),
        "simulated_pair_count": sum(1 for row in pair_results if row.get("source") == "simulated_missing_200r"),
        "agents": [agent_payload(agent) for agent in agents],
        "pair_results": pair_results,
        "elo_standings": ratings_table(ratings) if ratings else [],
        "win_rate_matrix": matrix_report,
    }


def estimate_ratings(pair_results: list[dict[str, Any]]) -> dict[str, float]:
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
    return calculate_elo(records, initial_rating=1500.0, scale=400.0, prior_std=400.0)


def build_matrix(agents: list[Agent], pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    values = {(str(row["agent"]), str(row["opponent"])): float(row["agent_win_rate"]) for row in pair_results}
    episodes = {(str(row["agent"]), str(row["opponent"])): int(row["episodes"]) for row in pair_results}
    return {
        "definition": "P_ij = Pr(new-run checkpoint i beats fixed-pool opponent j) estimated by rally win rate.",
        "row_labels": [agent.label for agent in agents],
        "row_steps": [agent.step for agent in agents],
        "col_labels": [agent.label for agent in agents],
        "col_steps": [agent.step for agent in agents],
        "win_rate_matrix": [
            [values.get((agent.label, opponent.label)) for opponent in agents]
            for agent in agents
        ],
        "rally_count_matrix": [
            [episodes.get((agent.label, opponent.label)) for opponent in agents]
            for agent in agents
        ],
    }


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    ensure_directory(output_dir)
    (output_dir / "fixed_pool_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_pair_csv(output_dir / "pair_results.csv", report["pair_results"])
    write_matrix_csv(output_dir / "win_rate_matrix.csv", report["win_rate_matrix"])
    write_matrix_json(output_dir / "win_rate_matrix.json", report["win_rate_matrix"])
    write_elo_csv(output_dir / "elo_standings.csv", report["elo_standings"])
    plot_win_rate_matrix(report["win_rate_matrix"], output_dir / "win_rate_matrix.png")
    plot_elo(report["elo_standings"], output_dir / "fixed_pool_rating_evolution.png")


def write_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "agent",
        "agent_step",
        "opponent",
        "opponent_step",
        "episodes",
        "agent_wins",
        "agent_win_rate",
        "opponent_win_rate",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_matrix_csv(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "step", *report["col_labels"]])
        for label, step, values in zip(report["row_labels"], report["row_steps"], report["win_rate_matrix"]):
            writer.writerow([label, step, *["" if value is None else value for value in values]])


def write_matrix_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_elo_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["agent", "elo"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_win_rate_matrix(report: dict[str, Any], output_path: Path) -> None:
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    matrix = np.asarray(report["win_rate_matrix"], dtype=float)
    steps = np.asarray(report["row_steps"], dtype=float) / 1_000_000.0
    fig, ax = plt.subplots(figsize=(12.5, 10.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=0.0, vmax=1.0, origin="upper", aspect="auto")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="row checkpoint win rate")
    tick_indices = np.arange(len(steps))
    tick_labels = [f"{step:.1f}" for step in steps]
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Opponent checkpoint step (M)")
    ax.set_ylabel("Agent checkpoint step (M)")
    ax.set_title("New-run cached 200-rally win-rate matrix")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_elo(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    parsed = sorted((_step_from_label(str(row["agent"])), float(row["elo"])) for row in rows)
    steps = np.asarray([step for step, _ in parsed], dtype=float) / 1_000_000.0
    elos = np.asarray([elo for _, elo in parsed], dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.plot(steps, elos, marker="o", linewidth=2.0)
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Elo")
    ax.set_title("New-run cached fixed-pool Elo")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def pair_payload(
    agent: Agent,
    opponent: Agent,
    *,
    episodes: int,
    win_rate: float,
    source: str,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "agent": agent.label,
        "agent_step": int(agent.step),
        "opponent": opponent.label,
        "opponent_step": int(opponent.step),
        "episodes": int(episodes),
        "agent_wins": float(win_rate) * float(episodes),
        "agent_win_rate": float(win_rate),
        "opponent_win_rate": 1.0 - float(win_rate),
        "source": source,
        "summary": summary,
    }


def agent_payload(agent: Agent) -> dict[str, Any]:
    return {"label": agent.label, "step": agent.step, "model_path": str(agent.model_path)}


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
    if path not in cache:
        cache[path] = PPO.load(path, device="cpu")
    return cache[path]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def config_value(data: dict[str, Any], *keys: str | Any) -> Any:
    default = keys[-1]
    for key in keys[:-1]:
        if isinstance(key, str) and key in data and data[key] is not None:
            return data[key]
    return default


def _step_from_label(label: str) -> int:
    return int(label.rsplit("_", 1)[1])


if __name__ == "__main__":
    main()
