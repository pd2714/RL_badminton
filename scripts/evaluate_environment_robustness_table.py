from __future__ import annotations

import argparse
import csv
import json
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

from badminton.config import SimulationConfig
from badminton.eval_evolution import build_discrete_action_config, build_sim_config
from badminton.evaluation import ModelSelector, rollout_episode, summarize_episodes
from badminton.selfplay import CheckpointPool, FixedCheckpointOpponent, build_selfplay_env
from badminton.utils import ensure_directory
from scripts.evaluate_recovery_ablation_fixed_pool import (
    RecoveryOverrideCheckpointOpponent,
    RecoveryOverrideModelSelector,
)


DEFAULT_RUN_DIR = Path(
    "outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
)
DEFAULT_NO_CRA_RUN_DIR = Path(
    "outputs/rl/selfplay_2d_norecoverycfadv_2m_heuristicbase_ent002_speed100_anchor100k_eval100k_20260610"
)
RECOVERY_MODES = {"learned", "centered", "heuristic"}


@dataclass(frozen=True)
class RobustnessVariant:
    label: str
    change: str
    overrides: dict[str, float]


@dataclass(frozen=True)
class PairSpec:
    comparison: str
    agent_label: str
    opponent_label: str
    agent_model_path: Path
    opponent_model_path: Path
    agent_recovery_mode: str = "learned"
    opponent_recovery_mode: str = "learned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation-time environment robustness table for the paper run."
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-cra-run-dir", type=Path, default=DEFAULT_NO_CRA_RUN_DIR)
    parser.add_argument("--late-step", type=int, default=6_000_000)
    parser.add_argument("--early-step", type=int, default=3_000_000)
    parser.add_argument(
        "--recovery-step",
        type=int,
        default=6_000_000,
        help="Checkpoint used for learned-vs-centered recovery.",
    )
    parser.add_argument(
        "--cra-step",
        type=int,
        default=3_400_000,
        help="CRA checkpoint used for the CRA-vs-no-CRA comparison.",
    )
    parser.add_argument(
        "--no-cra-step",
        type=int,
        default=3_400_000,
        help="No-CRA checkpoint used for the CRA-vs-no-CRA comparison.",
    )
    parser.add_argument("--eval-rallies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Win-rate threshold for writing yes/no conclusion cells.",
    )
    parser.add_argument(
        "--skip-cra",
        action="store_true",
        help="Do not run the CRA-vs-no-CRA column.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest/table skeleton without simulating rallies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.run_dir / "environment_robustness_eval")
    report = evaluate_robustness(
        run_dir=args.run_dir,
        output_dir=output_dir,
        no_cra_run_dir=args.no_cra_run_dir,
        late_step=int(args.late_step),
        early_step=int(args.early_step),
        recovery_step=int(args.recovery_step),
        cra_step=int(args.cra_step),
        no_cra_step=int(args.no_cra_step),
        eval_rallies=int(args.eval_rallies),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        decision_threshold=float(args.decision_threshold),
        skip_cra=bool(args.skip_cra),
        dry_run=bool(args.dry_run),
    )
    print(f"output: {output_dir}")
    print(f"manifest: {output_dir / 'manifest.json'}")
    print(f"table_csv: {output_dir / 'robustness_table.csv'}")
    print(f"table_md: {output_dir / 'robustness_table.md'}")
    print(
        "counts: "
        f"variants={len(report['variants'])} "
        f"pairs={len(report['pair_results'])} "
        f"dry_run={report['dry_run']}"
    )


def evaluate_robustness(
    *,
    run_dir: Path,
    output_dir: Path,
    no_cra_run_dir: Path,
    late_step: int,
    early_step: int,
    recovery_step: int,
    cra_step: int,
    no_cra_step: int,
    eval_rallies: int,
    seed: int,
    deterministic: bool,
    decision_threshold: float,
    skip_cra: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if eval_rallies <= 0:
        raise ValueError("--eval-rallies must be positive")
    ensure_directory(output_dir)

    train_config = load_json(run_dir / "selfplay_config.json")
    base_sim_config = build_sim_config(train_config)
    discrete_config = build_discrete_action_config(train_config)
    variants = build_variants(train_config)

    pair_specs = build_pair_specs(
        run_dir=run_dir,
        no_cra_run_dir=no_cra_run_dir,
        late_step=late_step,
        early_step=early_step,
        recovery_step=recovery_step,
        cra_step=cra_step,
        no_cra_step=no_cra_step,
        skip_cra=skip_cra,
    )
    manifest = {
        "run_dir": str(run_dir),
        "no_cra_run_dir": str(no_cra_run_dir),
        "eval_rallies_per_comparison": int(eval_rallies),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "decision_threshold": float(decision_threshold),
        "base_environment": environment_payload(train_config, base_sim_config),
        "variants": [variant_payload(variant) for variant in variants],
        "pair_specs": [pair_spec_payload(pair) for pair in pair_specs],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    partial_path = output_dir / "pair_results.jsonl"
    pair_results = load_partial_results(partial_path)
    completed = {
        (str(row["variant"]), str(row["comparison"]))
        for row in pair_results
    }

    if not dry_run:
        model_cache: dict[Path, PPO] = {}
        for variant_index, variant in enumerate(variants):
            variant_config = apply_variant(train_config, variant)
            sim_config = build_sim_config(variant_config)
            for pair_index, pair in enumerate(pair_specs):
                key = (variant.label, pair.comparison)
                if key in completed:
                    print(f"{variant.label} {pair.comparison}: already complete", flush=True)
                    continue
                pair_seed = seed + variant_index * 1_000_000 + pair_index * 10_000
                model = load_model_cached(model_cache, pair.agent_model_path)
                summary = evaluate_side_balanced_pair(
                    pair=pair,
                    model=model,
                    train_config=variant_config,
                    sim_config=sim_config,
                    discrete_config=discrete_config,
                    episodes=eval_rallies,
                    seed=pair_seed,
                    deterministic=deterministic,
                )
                result = {
                    "variant": variant.label,
                    "change": variant.change,
                    "comparison": pair.comparison,
                    "agent_label": pair.agent_label,
                    "opponent_label": pair.opponent_label,
                    "agent_model_path": str(pair.agent_model_path),
                    "opponent_model_path": str(pair.opponent_model_path),
                    "agent_recovery_mode": pair.agent_recovery_mode,
                    "opponent_recovery_mode": pair.opponent_recovery_mode,
                    "episodes": int(summary["episodes"]),
                    "agent_wins": float(summary["agent_wins"]),
                    "agent_win_rate": float(summary["win_rate"]),
                    "opponent_win_rate": 1.0 - float(summary["win_rate"]),
                    "source": "evaluation_time_environment_perturbation",
                    "summary": summary,
                }
                pair_results.append(result)
                append_partial_result(partial_path, result)
                completed.add(key)
                print(
                    f"{variant.label} {pair.comparison}: "
                    f"wr={result['agent_win_rate']:.3f} ({result['episodes']} rallies)",
                    flush=True,
                )

    table_rows = build_table_rows(
        variants=variants,
        pair_results=pair_results,
        decision_threshold=decision_threshold,
        skip_cra=skip_cra,
    )
    report = {
        **manifest,
        "output_dir": str(output_dir),
        "dry_run": bool(dry_run),
        "pair_results": pair_results,
        "table_rows": table_rows,
    }
    write_outputs(output_dir, report, skip_cra=skip_cra)
    return report


def build_variants(config: dict[str, Any]) -> list[RobustnessVariant]:
    kh = float(config_value(config, "horizontal_drag_coefficient", 0.20))
    kv = float(config_value(config, "vertical_drag_coefficient", 0.16))
    speed = float(config_value(config, "player_speed", 5.0))
    reaction = float(config_value(config, "reaction_time", 0.15))
    return [
        RobustnessVariant("Default", "original settings", {}),
        RobustnessVariant(
            "Lower drag",
            "kh, kv -20%",
            {
                "horizontal_drag_coefficient": kh * 0.8,
                "vertical_drag_coefficient": kv * 0.8,
            },
        ),
        RobustnessVariant(
            "Higher drag",
            "kh, kv +20%",
            {
                "horizontal_drag_coefficient": kh * 1.2,
                "vertical_drag_coefficient": kv * 1.2,
            },
        ),
        RobustnessVariant("Slower player", "speed -15%", {"player_speed": speed * 0.85}),
        RobustnessVariant("Faster player", "speed +15%", {"player_speed": speed * 1.15}),
        RobustnessVariant(
            "Longer reaction",
            "reaction time +50ms",
            {
                "reaction_time": reaction + 0.05,
                "opponent_reaction_time": float(config_value(config, "opponent_reaction_time", reaction)) + 0.05,
            },
        ),
        RobustnessVariant("No fast-reaction miss", "miss prob = 0", {"reaction_miss_fast_probability": 0.0}),
    ]


def apply_variant(config: dict[str, Any], variant: RobustnessVariant) -> dict[str, Any]:
    updated = dict(config)
    for key, value in variant.overrides.items():
        updated[key] = float(value)
    return updated


def build_pair_specs(
    *,
    run_dir: Path,
    no_cra_run_dir: Path,
    late_step: int,
    early_step: int,
    recovery_step: int,
    cra_step: int,
    no_cra_step: int,
    skip_cra: bool,
) -> list[PairSpec]:
    specs = [
        PairSpec(
            comparison="late_vs_early",
            agent_label=f"late_{late_step}",
            opponent_label=f"early_{early_step}",
            agent_model_path=resolve_checkpoint(run_dir, late_step),
            opponent_model_path=resolve_checkpoint(run_dir, early_step),
        ),
        PairSpec(
            comparison="learned_vs_centered_recovery",
            agent_label=f"learned_{recovery_step}",
            opponent_label=f"centered_{recovery_step}",
            agent_model_path=resolve_checkpoint(run_dir, recovery_step),
            opponent_model_path=resolve_checkpoint(run_dir, recovery_step),
            agent_recovery_mode="learned",
            opponent_recovery_mode="centered",
        ),
    ]
    if not skip_cra:
        specs.append(
            PairSpec(
                comparison="cra_vs_no_cra",
                agent_label=f"cra_{cra_step}",
                opponent_label=f"no_cra_{no_cra_step}",
                agent_model_path=resolve_checkpoint(run_dir, cra_step),
                opponent_model_path=resolve_checkpoint(no_cra_run_dir, no_cra_step),
            )
        )
    return specs


def resolve_checkpoint(run_dir: Path, step: int) -> Path:
    candidates = [
        run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip",
        run_dir / "historical_anchors" / f"anchor_step_{step}.zip",
    ]
    if int(step) < 0:
        candidates = []
    elif int(step) == 0:
        candidates.extend(
            [
                run_dir / "compatible_base_model.zip",
                run_dir / "best_model.zip",
            ]
        )
    candidates.extend([run_dir / "latest_model.zip", run_dir / "final_model.zip"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find checkpoint for step {step} in {run_dir}")


def evaluate_side_balanced_pair(
    *,
    pair: PairSpec,
    model: PPO,
    train_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_config: Any,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    left_count, right_count = side_balanced_counts(episodes)
    left_summary = evaluate_candidate_side(
        pair=pair,
        train_side="left",
        model=model,
        train_config=train_config,
        sim_config=sim_config,
        discrete_config=discrete_config,
        episodes=left_count,
        seed=seed + 101,
        deterministic=deterministic,
    )
    right_summary = evaluate_candidate_side(
        pair=pair,
        train_side="right",
        model=model,
        train_config=train_config,
        sim_config=sim_config,
        discrete_config=discrete_config,
        episodes=right_count,
        seed=seed + 202,
        deterministic=deterministic,
    )
    total = int(left_summary["episodes"]) + int(right_summary["episodes"])
    wins = float(left_summary["win_rate"]) * int(left_summary["episodes"])
    wins += float(right_summary["win_rate"]) * int(right_summary["episodes"])
    win_rate = wins / max(total, 1)
    return {
        "comparison": pair.comparison,
        "agent": pair.agent_label,
        "opponent": pair.opponent_label,
        "episodes": total,
        "agent_wins": wins,
        "win_rate": win_rate,
        "agent_as_left": left_summary,
        "agent_as_right": right_summary,
    }


def evaluate_candidate_side(
    *,
    pair: PairSpec,
    train_side: str,
    model: PPO,
    train_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_config: Any,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    env = build_pair_env(
        pair=pair,
        train_side=train_side,
        train_config=train_config,
        sim_config=sim_config,
        discrete_config=discrete_config,
        seed=seed,
        deterministic=deterministic,
    )
    if pair.agent_recovery_mode == "learned":
        selector = ModelSelector(model=model, deterministic=deterministic)
    else:
        selector = RecoveryOverrideModelSelector(
            model=model,
            recovery_mode=pair.agent_recovery_mode,
            action_mapper=env.action_mapper,
            sim_config=sim_config,
            agent_side=train_side,
            deterministic=deterministic,
        )
    try:
        results = [rollout_episode(env, selector, seed=seed + episode) for episode in range(max(int(episodes), 1))]
        summary = summarize_episodes(results)
    finally:
        env.close()
    summary["agent"] = pair.agent_label
    summary["opponent"] = pair.opponent_label
    summary["train_side"] = train_side
    return summary


def build_pair_env(
    *,
    pair: PairSpec,
    train_side: str,
    train_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_config: Any,
    seed: int,
    deterministic: bool,
):
    opponent_policy_cls = FixedCheckpointOpponent
    kwargs: dict[str, Any] = {}
    if pair.opponent_recovery_mode != "learned":
        if pair.opponent_recovery_mode not in RECOVERY_MODES:
            raise ValueError(f"Unsupported opponent recovery mode: {pair.opponent_recovery_mode}")
        opponent_policy_cls = RecoveryOverrideCheckpointOpponent
        kwargs["recovery_mode"] = pair.opponent_recovery_mode

    opponent = opponent_policy_cls(
        pool=CheckpointPool(
            checkpoint_dir=pair.opponent_model_path.parent,
            pool_size=1,
            sampling_mode="newest",
            seed=seed + 17,
        ),
        checkpoint_path=pair.opponent_model_path,
        sim_config=sim_config,
        discrete_action_config=discrete_config,
        policy_type=str(config_value(train_config, "policy_type", "velocity_oriented")),
        deterministic=deterministic,
        **kwargs,
    )
    reaction_time = float(config_value(train_config, "reaction_time", 0.15))
    return build_selfplay_env(
        train_side=train_side,
        mirror_train_side=False,
        mirror_match_fraction=0.0,
        initial_server=str(config_value(train_config, "initial_server", "random")),
        random_service_x=bool(config_value(train_config, "random_service_x", True)),
        sim_config=sim_config,
        train_reaction_time=reaction_time,
        opponent_reaction_time=float(config_value(train_config, "opponent_reaction_time", reaction_time)),
        max_stages_per_rally=int(config_value(train_config, "max_stages_per_rally", "max_rally_stages", 120)),
        policy_type=str(config_value(train_config, "policy_type", "velocity_oriented")),
        seed=seed,
        discrete_action_config=discrete_config,
        opponent=opponent,
        include_records_in_info=False,
        recovery_counterfactual_other_sample_count=0,
        recovery_counterfactual_expected_response_target=False,
    )


def side_balanced_counts(total: int) -> tuple[int, int]:
    left = int(total) // 2
    right = int(total) - left
    return max(left, 1), max(right, 1)


def build_table_rows(
    *,
    variants: list[RobustnessVariant],
    pair_results: list[dict[str, Any]],
    decision_threshold: float,
    skip_cra: bool,
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["variant"]), str(row["comparison"])): row
        for row in pair_results
    }
    rows = []
    for variant in variants:
        row = {
            "Variant": variant.label,
            "Change": variant.change,
            "Does late policy still beat early?": conclusion(
                by_key.get((variant.label, "late_vs_early")),
                threshold=decision_threshold,
            ),
            "late_vs_early_win_rate": metric(by_key.get((variant.label, "late_vs_early"))),
            "Does learned recovery beat centered recovery?": conclusion(
                by_key.get((variant.label, "learned_vs_centered_recovery")),
                threshold=decision_threshold,
            ),
            "learned_vs_centered_win_rate": metric(by_key.get((variant.label, "learned_vs_centered_recovery"))),
        }
        if not skip_cra:
            row["Does CRA still help?"] = conclusion(
                by_key.get((variant.label, "cra_vs_no_cra")),
                threshold=decision_threshold,
            )
            row["cra_vs_no_cra_win_rate"] = metric(by_key.get((variant.label, "cra_vs_no_cra")))
        rows.append(row)
    return rows


def conclusion(row: dict[str, Any] | None, *, threshold: float) -> str:
    if row is None:
        return "pending"
    win_rate = float(row["agent_win_rate"])
    if win_rate > threshold:
        return "yes"
    if win_rate < threshold:
        return "no"
    return "tie"


def metric(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    return float(row["agent_win_rate"])


def write_outputs(output_dir: Path, report: dict[str, Any], *, skip_cra: bool) -> None:
    (output_dir / "robustness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output_dir / "pair_results.csv", report["pair_results"])
    write_csv(output_dir / "robustness_table.csv", report["table_rows"])
    write_markdown_table(output_dir / "robustness_table.md", report["table_rows"], skip_cra=skip_cra)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], *, skip_cra: bool) -> None:
    columns = [
        "Variant",
        "Change",
        "Does late policy still beat early?",
        "Does learned recovery beat centered recovery?",
    ]
    if not skip_cra:
        columns.append("Does CRA still help?")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def append_partial_result(path: Path, row: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def load_model_cached(cache: dict[Path, PPO], path: Path) -> PPO:
    resolved = path.resolve()
    model = cache.get(resolved)
    if model is None:
        model = PPO.load(resolved)
        cache[resolved] = model
    return model


def environment_payload(config: dict[str, Any], sim_config: SimulationConfig) -> dict[str, Any]:
    reaction_time = float(config_value(config, "reaction_time", 0.15))
    return {
        "player_speed": float(sim_config.player.v_max),
        "player_acceleration": float(sim_config.player.acceleration),
        "racket_length": float(sim_config.player.r_reach),
        "reaction_time": reaction_time,
        "opponent_reaction_time": float(config_value(config, "opponent_reaction_time", reaction_time)),
        "horizontal_drag_coefficient": float(sim_config.action.effective_horizontal_drag_coefficient),
        "vertical_drag_coefficient": float(sim_config.action.effective_vertical_drag_coefficient),
        "reaction_miss_fast_threshold": float(sim_config.action.reaction_miss_fast_threshold),
        "reaction_miss_fast_probability": float(sim_config.action.reaction_miss_fast_probability),
    }


def variant_payload(variant: RobustnessVariant) -> dict[str, Any]:
    return {
        "label": variant.label,
        "change": variant.change,
        "overrides": dict(variant.overrides),
    }


def pair_spec_payload(pair: PairSpec) -> dict[str, Any]:
    return {
        "comparison": pair.comparison,
        "agent_label": pair.agent_label,
        "opponent_label": pair.opponent_label,
        "agent_model_path": str(pair.agent_model_path),
        "opponent_model_path": str(pair.opponent_model_path),
        "agent_recovery_mode": pair.agent_recovery_mode,
        "opponent_recovery_mode": pair.opponent_recovery_mode,
    }


def config_value(data: dict[str, Any], *keys: object) -> Any:
    default = keys[-1]
    for key in keys[:-1]:
        if key in data:
            return data[key]
    return default


if __name__ == "__main__":
    main()
