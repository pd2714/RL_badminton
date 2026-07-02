from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionMapper
from badminton1d.config import SimulationConfig
from badminton1d.elo import PairwiseRecord, calculate_elo, ratings_table
from badminton1d.evaluation import ModelSelector, choose_model_action, summarize_episodes
from badminton1d.opponents import DecisionContext, make_opponent
from badminton1d.policy import MaskedBadmintonPolicy
from badminton1d.rl_env import BadmintonRLEnv, RLEnvConfig, RewardConfig
from badminton1d.selfplay import CheckpointPool, FixedCheckpointOpponent, build_selfplay_env
from badminton1d.shot_generators import TacticRuntimeConfig
from badminton1d.state import ShotAction, Side, StageState
from badminton1d.utils import canonicalize_state_for_agent, ensure_directory, recovery_bounds
from scripts.round_robin_selfplay_video import (
    AgentSpec,
    _config_value,
    _load_config,
    _resolve_random_server,
    build_discrete_action_config,
    build_sim_config,
    rollout_rally,
)
from scripts.train_selfplay import load_base_parameters_compatibly, load_base_policy_state_compatibly

RECOVERY_MODES = ("learned", "centered", "heuristic")


@dataclass(frozen=True)
class RecoveryAgentSpec:
    label: str
    step: int
    recovery_mode: str
    model_path: Path
    source_model_path: Path


@dataclass(frozen=True)
class PoolOpponentSpec:
    label: str
    kind: str
    step: int | None = None
    recovery_mode: str | None = None
    model_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recovery ablations against a fixed opponent pool."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--zero-checkpoint", type=Path, default=None)
    parser.add_argument("--recovery-modes", choices=RECOVERY_MODES, nargs="+", default=list(RECOVERY_MODES))
    parser.add_argument(
        "--no-heuristic-safe-opponent",
        action="store_true",
        help="Do not include the heuristic_safe baseline in the fixed opponent pool.",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000])
    parser.add_argument("--pool-steps", type=int, nargs="+", default=[2_000_000, 4_000_000, 6_000_000])
    parser.add_argument("--eval-rallies", type=int, default=200, help="Total side-balanced rallies per candidate/opponent matchup.")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--prior-std", type=float, default=400.0)
    parser.add_argument("--skip-eval", action="store_true", help="Only build the virtual checkpoint entries and manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or (run_dir / "recovery_ablation_fixed_pool")
    report = run_recovery_ablation(
        run_dir=run_dir,
        output_dir=output_dir,
        steps=list(args.steps),
        pool_steps=list(args.pool_steps),
        recovery_modes=list(args.recovery_modes),
        include_heuristic_safe_opponent=not bool(args.no_heuristic_safe_opponent),
        zero_checkpoint=args.zero_checkpoint,
        eval_rallies=args.eval_rallies,
        seed=args.seed,
        deterministic=args.deterministic,
        initial_rating=args.initial_rating,
        elo_scale=args.elo_scale,
        prior_std=args.prior_std,
        skip_eval=args.skip_eval,
    )
    print(f"manifest: {output_dir / 'manifest.json'}")
    if not args.skip_eval:
        print(f"report: {output_dir / 'recovery_ablation_report.json'}")
        print(f"ratings: {output_dir / 'elo_by_variant.csv'}")
        print(f"plot: {output_dir / 'elo_vs_checkpoint.png'}")
        print(f"plot: {output_dir / 'mean_pool_win_rate_vs_checkpoint.png'}")
        for row in report["elo_by_variant"]:
            print(f"{row['recovery_mode']} step={int(row['step'])}: elo={float(row['elo']):.1f}")


def run_recovery_ablation(
    *,
    run_dir: Path,
    output_dir: Path,
    steps: list[int],
    pool_steps: list[int],
    recovery_modes: list[str],
    include_heuristic_safe_opponent: bool,
    zero_checkpoint: Path | None,
    eval_rallies: int,
    seed: int,
    deterministic: bool,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
    skip_eval: bool,
) -> dict[str, Any]:
    if eval_rallies <= 0:
        raise ValueError("--eval-rallies must be positive")
    recovery_modes = validate_recovery_modes(recovery_modes)
    ensure_directory(output_dir)

    config = _load_config(run_dir)
    sim_config = build_sim_config(config)
    discrete_config = build_discrete_action_config(config)
    requested_steps = [int(step) for step in steps]
    if 0 in requested_steps:
        zero_checkpoint = zero_checkpoint or infer_zero_checkpoint(run_dir, config)
        zero_checkpoint = ensure_compatible_zero_checkpoint(
            zero_checkpoint=zero_checkpoint,
            output_dir=output_dir,
            config=config,
            sim_config=sim_config,
            discrete_config=discrete_config,
        )
    else:
        zero_checkpoint = None

    source_checkpoints = resolve_step_checkpoints(run_dir, requested_steps, zero_checkpoint)
    agents = materialize_virtual_agents(
        output_dir=output_dir,
        run_dir=run_dir,
        config=config,
        source_checkpoints=source_checkpoints,
        recovery_modes=recovery_modes,
    )
    opponents = build_fixed_pool(
        agents,
        pool_steps,
        recovery_modes=recovery_modes,
        include_heuristic_safe_opponent=include_heuristic_safe_opponent,
    )
    write_manifest(output_dir, run_dir, agents, opponents, source_checkpoints)

    if skip_eval:
        return {"agents": [agent.__dict__ for agent in agents], "opponents": [opponent.__dict__ for opponent in opponents]}

    report = evaluate_agents_against_pool(
        run_dir=run_dir,
        output_dir=output_dir,
        config=config,
        sim_config=sim_config,
        discrete_config=discrete_config,
        agents=agents,
        opponents=opponents,
        eval_rallies=eval_rallies,
        seed=seed,
        deterministic=deterministic,
        initial_rating=initial_rating,
        elo_scale=elo_scale,
        prior_std=prior_std,
    )
    write_outputs(output_dir, report)
    plot_elo_curves(output_dir / "elo_vs_checkpoint.png", report["elo_by_variant"])
    plot_mean_pool_win_rate_curves(output_dir / "mean_pool_win_rate_vs_checkpoint.png", report["elo_by_variant"])
    return report


def infer_zero_checkpoint(run_dir: Path, config: dict[str, Any]) -> Path:
    for script_path in sorted(run_dir.glob("launch*.sh")):
        try:
            tokens = shlex.split(script_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if "--resume-step-offset" in tokens:
            continue
        if "--base-checkpoint-path" not in tokens:
            continue
        index = tokens.index("--base-checkpoint-path")
        if index + 1 < len(tokens):
            candidate = Path(tokens[index + 1])
            return candidate if candidate.is_absolute() else REPO_ROOT / candidate

    requested = config.get("requested_base_checkpoint_path") or config.get("base_checkpoint_path")
    if requested is None:
        raise FileNotFoundError("Could not infer a 0-step checkpoint; pass --zero-checkpoint.")
    candidate = Path(str(requested))
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def validate_recovery_modes(recovery_modes: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for mode in recovery_modes:
        mode = str(mode)
        if mode not in RECOVERY_MODES:
            raise ValueError(f"Unsupported recovery mode: {mode}")
        if mode not in selected:
            selected.append(mode)
    if not selected:
        raise ValueError("At least one recovery mode is required")
    return selected


def resolve_step_checkpoints(run_dir: Path, steps: Iterable[int], zero_checkpoint: Path | None) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    anchor_dir = run_dir / "anchor_checkpoints"
    requested_steps = [int(step) for step in steps]
    final_step = max(requested_steps) if requested_steps else None
    for step in requested_steps:
        if int(step) == 0:
            if zero_checkpoint is None:
                raise ValueError("A zero checkpoint is required when step 0 is requested.")
            path = zero_checkpoint
        else:
            path = anchor_dir / f"anchor_step_{int(step)}.zip"
            if not path.exists() and step == final_step:
                latest_path = run_dir / "latest_model.zip"
                if latest_path.exists():
                    path = latest_path
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for step {step}: {path}")
        checkpoints[int(step)] = path.resolve()
    return checkpoints


def ensure_compatible_zero_checkpoint(
    *,
    zero_checkpoint: Path,
    output_dir: Path,
    config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_config: Any,
) -> Path:
    try:
        PPO.load(zero_checkpoint)
        return zero_checkpoint.resolve()
    except RuntimeError as error:
        message = str(error)
        if "Missing key(s)" not in message and "Unexpected key(s)" not in message and "size mismatch" not in message:
            raise

    compatible_path = output_dir / "source_checkpoints" / "compatible_zero_model.zip"
    if compatible_path.exists():
        return compatible_path.resolve()
    ensure_directory(compatible_path.parent)

    env = BadmintonRLEnv(
        config=sim_config,
        rl_config=RLEnvConfig(
            train_side=str(_config_value(config, "train_side", "left")),
            initial_server=str(_config_value(config, "requested_initial_server", "random", "initial_server")),
            random_service_x=bool(_config_value(config, "random_service_x", True)),
            train_reaction_time=float(_config_value(config, "reaction_time", 0.15)),
            opponent_reaction_time=float(_config_value(config, "opponent_reaction_time", 0.15, "reaction_time")),
            max_stages_per_rally=int(_config_value(config, "max_stages_per_rally", 100, "max_rally_stages")),
            policy_type=str(_config_value(config, "policy_type", "velocity_oriented")),
            tactic_runtime=TacticRuntimeConfig(),
            reward=RewardConfig(),
            recovery_counterfactual_other_sample_count=0,
            counterfactual_opponent_response_samples=1,
            recovery_counterfactual_expected_response_target=False,
            recovery_full_diagnostics_probability=0.0,
        ),
        discrete_action_config=discrete_config,
        opponent=make_opponent("safe", seed=int(_config_value(config, "seed", 17)) + 700_000),
        seed=int(_config_value(config, "seed", 17)) + 700_001,
    )
    model = PPO(
        MaskedBadmintonPolicy,
        env,
        policy_kwargs={
            "sim_config": sim_config,
            "discrete_action_config": discrete_config,
            "policy_type": str(_config_value(config, "policy_type", "velocity_oriented")),
            "tactic_runtime_config": TacticRuntimeConfig(),
            "mask_mid_rally_hitter_actions": bool(_config_value(config, "mask_mid_rally_hitter_actions", True)),
        },
        learning_rate=float(_config_value(config, "learning_rate", 3e-4)),
        gamma=float(_config_value(config, "gamma", 0.99)),
        batch_size=8,
        n_steps=8,
        ent_coef=float(_config_value(config, "ent_coef", 0.0)),
        verbose=0,
        seed=int(_config_value(config, "seed", 17)),
    )
    try:
        base_model = PPO.load(zero_checkpoint)
        load_base_parameters_compatibly(model, base_model)
    except RuntimeError as error:
        _, params, _ = load_from_zip_file(zero_checkpoint, device=model.device)
        policy_state = params.get("policy")
        if policy_state is None:
            raise RuntimeError(f"Checkpoint has no policy parameters: {zero_checkpoint}") from error
        load_base_policy_state_compatibly(model, policy_state, source_label=str(zero_checkpoint))
    model.save(compatible_path)
    env.close()
    print(f"created compatible 0M checkpoint: {compatible_path}", flush=True)
    return compatible_path.resolve()


def materialize_virtual_agents(
    *,
    output_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    source_checkpoints: dict[int, Path],
    recovery_modes: list[str],
) -> list[RecoveryAgentSpec]:
    agents: list[RecoveryAgentSpec] = []
    agents_dir = output_dir / "agents"
    for step in sorted(source_checkpoints):
        for mode in recovery_modes:
            label = f"{mode}_{format_step_label(step)}"
            agent_dir = agents_dir / label
            ensure_directory(agent_dir)
            model_path = agent_dir / "model.zip"
            link_or_copy(source_checkpoints[step], model_path)
            (agent_dir / "selfplay_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (agent_dir / "ablation_agent.json").write_text(
                json.dumps(
                    {
                        "label": label,
                        "step": step,
                        "recovery_mode": mode,
                        "source_model_path": str(source_checkpoints[step]),
                        "model_path": str(model_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            agents.append(
                RecoveryAgentSpec(
                    label=label,
                    step=step,
                    recovery_mode=mode,
                    model_path=model_path,
                    source_model_path=source_checkpoints[step],
                )
            )
    return agents


def format_step_label(step: int) -> str:
    step = int(step)
    if step % 1_000_000 == 0:
        return f"{step // 1_000_000}m"
    text = f"{step / 1_000_000.0:.3f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', 'p')}m"


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


def build_fixed_pool(
    agents: list[RecoveryAgentSpec],
    pool_steps: list[int],
    *,
    recovery_modes: list[str],
    include_heuristic_safe_opponent: bool,
) -> list[PoolOpponentSpec]:
    by_key = {(agent.recovery_mode, agent.step): agent for agent in agents}
    opponents = []
    if include_heuristic_safe_opponent:
        opponents.append(PoolOpponentSpec(label="heuristic_safe", kind="heuristic"))
    for step in pool_steps:
        for mode in recovery_modes:
            agent = by_key.get((mode, int(step)))
            if agent is None:
                raise ValueError(f"No ablation agent for pool member {mode} at step {step}")
            opponents.append(
                PoolOpponentSpec(
                    label=agent.label,
                    kind="checkpoint",
                    step=agent.step,
                    recovery_mode=agent.recovery_mode,
                    model_path=agent.model_path,
                )
            )
    return opponents


def write_manifest(
    output_dir: Path,
    run_dir: Path,
    agents: list[RecoveryAgentSpec],
    opponents: list[PoolOpponentSpec],
    source_checkpoints: dict[int, Path],
) -> None:
    manifest = {
        "run_dir": str(run_dir),
        "source_checkpoints": {str(step): str(path) for step, path in sorted(source_checkpoints.items())},
        "agents": [serializable_dataclass(agent) for agent in agents],
        "opponent_pool": [serializable_dataclass(opponent) for opponent in opponents],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def serializable_dataclass(obj: Any) -> dict[str, Any]:
    payload = dict(obj.__dict__)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def evaluate_agents_against_pool(
    *,
    run_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_config: Any,
    agents: list[RecoveryAgentSpec],
    opponents: list[PoolOpponentSpec],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
    initial_rating: float,
    elo_scale: float,
    prior_std: float,
) -> dict[str, Any]:
    model_cache: dict[Path, PPO] = {}
    partial_path = output_dir / "pair_results.jsonl"
    all_partial_pair_results = load_partial_pair_results(partial_path)
    pair_results, ignored_partial_count = current_pair_results(
        all_partial_pair_results,
        agents=agents,
        opponents=opponents,
        eval_rallies=eval_rallies,
    )
    if ignored_partial_count:
        print(
            f"Ignoring {ignored_partial_count} stale/incompatible partial pair results from {partial_path}",
            flush=True,
        )
    completed_pairs = {(str(row["agent"]), str(row["opponent"])) for row in pair_results}
    elo_records: list[PairwiseRecord] = []
    for row in pair_results:
        games = float(row["episodes"])
        elo_records.append(
            PairwiseRecord(
                agent_a=str(row["agent"]),
                agent_b=str(row["opponent"]),
                agent_a_score=float(row["agent_win_rate"]) * games,
                games=games,
            )
        )
    per_side = side_balanced_counts(eval_rallies)

    for agent_index, agent in enumerate(agents):
        model = load_model_cached(model_cache, agent.model_path)
        for opponent_index, opponent in enumerate(opponents):
            if agent.label == opponent.label:
                continue
            if (agent.label, opponent.label) in completed_pairs:
                print(f"{agent.label} vs {opponent.label}: already complete", flush=True)
                continue
            pair_seed = seed + agent_index * 1_000_000 + opponent_index * 10_000
            left_summary = evaluate_candidate_side(
                agent=agent,
                opponent=opponent,
                train_side="left",
                model=model,
                train_config=config,
                sim_config=sim_config,
                discrete_config=discrete_config,
                episodes=per_side[0],
                seed=pair_seed + 101,
                deterministic=deterministic,
            )
            right_summary = evaluate_candidate_side(
                agent=agent,
                opponent=opponent,
                train_side="right",
                model=model,
                train_config=config,
                sim_config=sim_config,
                discrete_config=discrete_config,
                episodes=per_side[1],
                seed=pair_seed + 202,
                deterministic=deterministic,
            )
            total = int(left_summary["episodes"]) + int(right_summary["episodes"])
            wins = float(left_summary["win_rate"]) * int(left_summary["episodes"])
            wins += float(right_summary["win_rate"]) * int(right_summary["episodes"])
            win_rate = wins / max(total, 1)
            pair = {
                "agent": agent.label,
                "opponent": opponent.label,
                "episodes": total,
                "agent_win_rate": win_rate,
                "opponent_win_rate": 1.0 - win_rate,
                "agent_step": agent.step,
                "agent_recovery_mode": agent.recovery_mode,
                "agent_as_left": left_summary,
                "agent_as_right": right_summary,
            }
            pair_results.append(pair)
            append_partial_pair_result(partial_path, pair)
            completed_pairs.add((agent.label, opponent.label))
            elo_records.append(
                PairwiseRecord(
                    agent_a=agent.label,
                    agent_b=opponent.label,
                    agent_a_score=win_rate * total,
                    games=float(total),
                )
            )
            print(f"{agent.label} vs {opponent.label}: wr={win_rate:.3f} ({total} rallies)", flush=True)

    ratings = calculate_elo(
        elo_records,
        initial_rating=initial_rating,
        scale=elo_scale,
        prior_std=prior_std,
    )
    elo_by_variant = [
        {
            "agent": agent.label,
            "step": agent.step,
            "step_millions": agent.step / 1_000_000.0,
            "recovery_mode": agent.recovery_mode,
            "elo": float(ratings[agent.label]),
            "mean_pool_win_rate": mean_pool_win_rate(pair_results, agent.label),
        }
        for agent in agents
    ]
    elo_by_variant.sort(key=lambda item: (str(item["recovery_mode"]), int(item["step"])))
    return {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "deterministic": deterministic,
        "eval_rallies_per_matchup": eval_rallies,
        "eval_rallies_per_side": list(per_side),
        "initial_rating": initial_rating,
        "elo_scale": elo_scale,
        "prior_std": prior_std,
        "agents": [serializable_dataclass(agent) for agent in agents],
        "opponent_pool": [serializable_dataclass(opponent) for opponent in opponents],
        "standings": ratings_table(ratings),
        "elo_by_variant": elo_by_variant,
        "pair_results": pair_results,
    }


def load_partial_pair_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def current_pair_results(
    rows: list[dict[str, Any]],
    *,
    agents: list[RecoveryAgentSpec],
    opponents: list[PoolOpponentSpec],
    eval_rallies: int,
) -> tuple[list[dict[str, Any]], int]:
    agent_by_label = {agent.label: agent for agent in agents}
    opponent_labels = {opponent.label for opponent in opponents}
    expected_pairs = {
        (agent.label, opponent.label)
        for agent in agents
        for opponent in opponents
        if agent.label != opponent.label
    }
    latest_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    ignored = 0
    for row in rows:
        key = (str(row.get("agent", "")), str(row.get("opponent", "")))
        agent = agent_by_label.get(key[0])
        if key not in expected_pairs or key[1] not in opponent_labels or agent is None:
            ignored += 1
            continue
        if int(row.get("episodes", -1)) != int(eval_rallies):
            ignored += 1
            continue
        if int(row.get("agent_step", agent.step)) != int(agent.step):
            ignored += 1
            continue
        if str(row.get("agent_recovery_mode", agent.recovery_mode)) != agent.recovery_mode:
            ignored += 1
            continue
        if key in latest_by_pair:
            ignored += 1
        latest_by_pair[key] = row
    current_rows = [latest_by_pair[key] for key in sorted(latest_by_pair)]
    return current_rows, ignored


def append_partial_pair_result(path: Path, pair: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(pair) + "\n")


def side_balanced_counts(total: int) -> tuple[int, int]:
    left = int(total) // 2
    right = int(total) - left
    return max(left, 1), max(right, 1)


def load_model_cached(cache: dict[Path, PPO], model_path: Path) -> PPO:
    resolved = model_path.resolve()
    model = cache.get(resolved)
    if model is None:
        model = PPO.load(resolved)
        cache[resolved] = model
    return model


def evaluate_candidate_side(
    *,
    agent: RecoveryAgentSpec,
    opponent: PoolOpponentSpec,
    train_side: Side,
    model: PPO,
    train_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_config: Any,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    env = build_ablation_env(
        train_side=train_side,
        train_config=train_config,
        sim_config=sim_config,
        discrete_config=discrete_config,
        opponent=opponent,
        seed=seed,
        deterministic=deterministic,
    )
    selector = RecoveryOverrideModelSelector(
        model=model,
        recovery_mode=agent.recovery_mode,
        action_mapper=env.action_mapper,
        sim_config=sim_config,
        agent_side=train_side,
        deterministic=deterministic,
    )
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


def build_ablation_env(
    *,
    train_side: Side,
    train_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_config: Any,
    opponent: PoolOpponentSpec,
    seed: int,
    deterministic: bool,
):
    if opponent.kind == "heuristic":
        opponent_policy = make_opponent("safe", seed=seed + 17)
    elif opponent.kind == "checkpoint":
        if opponent.model_path is None or opponent.recovery_mode is None:
            raise ValueError(f"Invalid checkpoint opponent spec: {opponent}")
        opponent_policy = RecoveryOverrideCheckpointOpponent(
            pool=CheckpointPool(
                checkpoint_dir=opponent.model_path.parent,
                pool_size=1,
                sampling_mode="newest",
                seed=seed + 17,
            ),
            checkpoint_path=opponent.model_path,
            sim_config=sim_config,
            discrete_action_config=discrete_config,
            policy_type=str(_config_value(train_config, "policy_type", "velocity_oriented")),
            tactic_runtime_config=TacticRuntimeConfig(),
            deterministic=deterministic,
            recovery_mode=opponent.recovery_mode,
        )
    else:
        raise ValueError(f"Unsupported opponent kind: {opponent.kind}")

    return build_selfplay_env(
        train_side=train_side,
        mirror_train_side=False,
        mirror_match_fraction=0.0,
        initial_server="random",
        random_service_x=bool(_config_value(train_config, "random_service_x", True)),
        sim_config=sim_config,
        train_reaction_time=float(_config_value(train_config, "reaction_time", 0.15)),
        opponent_reaction_time=float(_config_value(train_config, "opponent_reaction_time", 0.15)),
        max_stages_per_rally=int(_config_value(train_config, "max_stages_per_rally", 100, "max_rally_stages")),
        policy_type=str(_config_value(train_config, "policy_type", "velocity_oriented")),
        seed=seed,
        discrete_action_config=discrete_config,
        opponent=opponent_policy,
        include_records_in_info=False,
        recovery_counterfactual_other_sample_count=0,
        recovery_counterfactual_expected_response_target=False,
    )


@dataclass
class RecoveryOverrideModelSelector:
    model: PPO
    recovery_mode: str
    action_mapper: DiscreteActionMapper
    sim_config: SimulationConfig
    agent_side: Side
    deterministic: bool = False

    def choose_action(self, observation: np.ndarray, context: DecisionContext) -> object:
        if context.role != "hitter" or self.recovery_mode == "learned":
            return ModelSelector(self.model, deterministic=self.deterministic).choose_action(observation, context)
        action = choose_model_action(self.model, observation, context, deterministic=self.deterministic)
        if not isinstance(action, (int, np.integer)):
            return action
        return override_recovery_index(
            int(action),
            recovery_mode=self.recovery_mode,
            action_mapper=self.action_mapper,
            state=context.state,
            sim_config=self.sim_config,
            agent_side=self.agent_side,
        )


@dataclass
class RecoveryOverrideCheckpointOpponent(FixedCheckpointOpponent):
    recovery_mode: str = "learned"

    def choose_hitter_action(self, state: StageState, config: SimulationConfig, server_side: Side) -> ShotAction:
        action = self._predict_action(
            state=state,
            role="hitter",
            server_side=server_side,
            pending_action=None,
            feasible_indices=[],
        )
        if self.recovery_mode != "learned":
            action = override_recovery_index(
                int(action),
                recovery_mode=self.recovery_mode,
                action_mapper=self.action_mapper,
                state=state,
                sim_config=self.sim_config,
                agent_side=self.current_side,
            )
        decoded = self.action_mapper.decode_hitter_for_agent(int(action), state, self.current_side).shot_action
        projected = self.action_mapper.project_hitter_action(state, decoded)
        self._prepared_hitter_shot = projected.prepared_shot
        return projected.shot_action


def override_recovery_index(
    action: int,
    *,
    recovery_mode: str,
    action_mapper: DiscreteActionMapper,
    state: StageState,
    sim_config: SimulationConfig,
    agent_side: Side,
) -> int:
    if recovery_mode not in {"centered", "heuristic"}:
        return int(action)
    rec_count = max(int(action_mapper._impl._effective_x_rec_bins) * int(action_mapper.discrete_config.y_rec_bins), 1)
    shot_prefix = int(action) // rec_count
    canonical_state = canonicalize_state_for_agent(state, agent_side)
    probe_action = int(shot_prefix * rec_count)
    decoded = action_mapper.decode_hitter(probe_action, canonical_state).shot_action
    x_grid, y_grid = action_mapper._impl._recovery_grid_for_shot_action(canonical_state, decoded)
    target_x, target_y = recovery_target(recovery_mode, canonical_state.current_hitter, sim_config)
    rec_index = nearest_recovery_index(x_grid, y_grid, target_x, target_y)
    return int(shot_prefix * rec_count + rec_index)


def recovery_target(recovery_mode: str, side: Side, config: SimulationConfig) -> tuple[float, float]:
    (x_low, x_high), (y_low, y_high) = recovery_bounds(side, config)
    if recovery_mode == "heuristic":
        return (
            x_low + 0.5 * (x_high - x_low),
            y_low + 0.45 * (y_high - y_low),
        )
    return (
        x_low + 0.5 * (x_high - x_low),
        y_low + 0.5 * (y_high - y_low),
    )


def nearest_recovery_index(x_grid: np.ndarray, y_grid: np.ndarray, target_x: float, target_y: float) -> int:
    best_index = 0
    best_distance = float("inf")
    flat_index = 0
    for x_value in x_grid:
        for y_value in y_grid:
            distance = float(np.hypot(float(x_value) - target_x, float(y_value) - target_y))
            if distance < best_distance:
                best_distance = distance
                best_index = flat_index
            flat_index += 1
    return best_index


def mean_pool_win_rate(pair_results: list[dict[str, Any]], agent_label: str) -> float | None:
    rates = [float(row["agent_win_rate"]) for row in pair_results if row["agent"] == agent_label]
    if not rates:
        return None
    return float(np.mean(rates))


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "recovery_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output_dir / "pair_results.csv", report["pair_results"])
    write_csv(output_dir / "elo_by_variant.csv", report["elo_by_variant"])
    write_csv(output_dir / "elo_ratings.csv", report["standings"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_elo_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    colors = {"learned": "#1f77b4", "centered": "#ff7f0e", "heuristic": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for mode in modes_in_rows(rows):
        mode_rows = sorted((row for row in rows if row["recovery_mode"] == mode), key=lambda row: int(row["step"]))
        xs = [float(row["step_millions"]) for row in mode_rows]
        ys = [float(row["elo"]) for row in mode_rows]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=mode, color=colors.get(mode))
    ax.set_xlabel("Checkpoint (M steps)")
    ax.set_ylabel("Elo vs fixed pool")
    ax.set_title("Recovery ablation with fixed shot policy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mean_pool_win_rate_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    colors = {"learned": "#1f77b4", "centered": "#ff7f0e", "heuristic": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for mode in modes_in_rows(rows):
        mode_rows = sorted((row for row in rows if row["recovery_mode"] == mode), key=lambda row: int(row["step"]))
        plotted_rows = [row for row in mode_rows if row.get("mean_pool_win_rate") not in (None, "")]
        xs = [float(row["step_millions"]) for row in plotted_rows]
        ys = [float(row["mean_pool_win_rate"]) for row in plotted_rows]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=mode, color=colors.get(mode))
    ax.set_xlabel("Checkpoint (M steps)")
    ax.set_ylabel("Mean win rate vs fixed pool")
    ax.set_title("Recovery ablation with fixed shot policy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def modes_in_rows(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row.get("recovery_mode")) for row in rows}
    return [mode for mode in RECOVERY_MODES if mode in present]


if __name__ == "__main__":
    main()
