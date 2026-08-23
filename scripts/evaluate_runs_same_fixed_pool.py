from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.elo import PairwiseRecord, calculate_elo
from badminton.eval_evolution import build_discrete_action_config, build_sim_config
from badminton.utils import ensure_directory
from scripts.evaluate_requested_cross_run_fixed_pool_200r import (
    Entry,
    append_partial_result,
    evaluate_pair,
    load_json,
    load_model_cached,
    load_partial_results,
    write_matrix_csv,
    write_pair_csv,
)


DEFAULT_BASE_REPORT = (
    Path("outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611")
    / "cross_run_fixed_pool_eval_200r"
    / "fixed_pool_eval_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate arbitrary run checkpoints against an existing fixed-pool report."
    )
    parser.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_DIR"),
        required=True,
        help="Run label and run directory. May be repeated.",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument(
        "--zero-model",
        type=Path,
        default=None,
        help="Model to use for explicit step 0 entries when runs do not have anchor_step_0.zip.",
    )
    parser.add_argument("--step-min", type=int, default=3_200_000)
    parser.add_argument("--step-interval", type=int, default=400_000)
    parser.add_argument("--auto-step-mode", choices=["union", "intersection"], default="union")
    parser.add_argument(
        "--freeze-steps-file",
        type=Path,
        default=None,
        help="Read/write the auto-discovered step list so resumptions keep the same grid.",
    )
    parser.add_argument("--eval-rallies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--prior-std", type=float, default=400.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [(str(label), Path(run_dir)) for label, run_dir in args.run]
    report = evaluate(
        base_report_path=args.base_report,
        output_dir=args.output_dir,
        runs=runs,
        steps=None if args.steps is None else [int(step) for step in args.steps],
        zero_model=args.zero_model,
        step_min=int(args.step_min),
        step_interval=int(args.step_interval),
        auto_step_mode=str(args.auto_step_mode),
        freeze_steps_file=args.freeze_steps_file,
        eval_rallies=int(args.eval_rallies),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        initial_rating=float(args.initial_rating),
        elo_scale=float(args.elo_scale),
        prior_std=float(args.prior_std),
        dry_run=bool(args.dry_run),
    )
    print(f"output: {args.output_dir}")
    print(f"report: {args.output_dir / 'fixed_pool_eval_report.json'}")
    print(f"pairs: {args.output_dir / 'pair_results.csv'}")
    print(f"ratings: {args.output_dir / 'mean_win_rate_elo.csv'}")
    print(
        "counts: "
        f"available={report['available_pair_count']} "
        f"completed={report['completed_pair_count']} "
        f"remaining={report['remaining_pair_count']}"
    )


def evaluate(
    *,
    base_report_path: Path,
    output_dir: Path,
    runs: list[tuple[str, Path]],
    steps: list[int] | None,
    zero_model: Path | None,
    step_min: int,
    step_interval: int,
    auto_step_mode: str,
    freeze_steps_file: Path | None,
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
    if not runs:
        raise ValueError("At least one --run is required")
    ensure_directory(output_dir)

    base_report = load_json(base_report_path)
    selected_steps = resolve_steps(
        runs=runs,
        explicit_steps=steps,
        step_min=step_min,
        step_interval=step_interval,
        auto_step_mode=auto_step_mode,
        freeze_steps_file=freeze_steps_file,
    )
    eval_entries = build_run_entries(runs, selected_steps, zero_model=zero_model)
    opponent_entries = build_opponent_entries(base_report)
    available_eval = [entry for entry in eval_entries if entry.available]
    available_opponents = [entry for entry in opponent_entries if entry.available]
    available_pairs = {(agent.label, opponent.label) for agent in available_eval for opponent in available_opponents}

    write_manifest(
        output_dir,
        base_report_path=base_report_path,
        runs=runs,
        selected_steps=selected_steps,
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

    if not dry_run:
        model_cache: dict[Path, PPO] = {}
        config_cache: dict[Path, dict[str, Any]] = {}
        sim_config_cache: dict[Path, Any] = {}
        discrete_config_cache: dict[Path, Any] = {}
        for agent_index, agent in enumerate(available_eval):
            assert agent.model_path is not None
            model = load_model_cached(model_cache, agent.model_path)
            train_config = load_config_cached(config_cache, agent.run_dir)
            sim_config, discrete_action_config = build_env_configs_cached(
                sim_config_cache, discrete_config_cache, agent.run_dir, train_config
            )
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
                    "source": "simulated_same_fixed_pool_200r",
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
        output_dir=output_dir,
        runs=runs,
        selected_steps=selected_steps,
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
    write_outputs(output_dir, report)
    return report


def resolve_steps(
    *,
    runs: list[tuple[str, Path]],
    explicit_steps: list[int] | None,
    step_min: int,
    step_interval: int,
    auto_step_mode: str,
    freeze_steps_file: Path | None,
) -> list[int]:
    if explicit_steps is not None:
        steps = sorted({int(step) for step in explicit_steps})
    elif freeze_steps_file is not None and freeze_steps_file.exists():
        steps = [int(step) for step in json.loads(freeze_steps_file.read_text(encoding="utf-8"))["steps"]]
    else:
        step_sets = [set(discover_anchor_steps(run_dir)) for _, run_dir in runs]
        if auto_step_mode == "intersection":
            available = set.intersection(*step_sets) if step_sets else set()
        elif auto_step_mode == "union":
            available = set.union(*step_sets) if step_sets else set()
        else:
            raise ValueError(f"Unknown auto_step_mode: {auto_step_mode}")
        steps = [
            step
            for step in sorted(available)
            if step >= step_min and (step - step_min) % step_interval == 0
        ]
    if not steps:
        raise ValueError("No checkpoint steps selected")
    if freeze_steps_file is not None and not freeze_steps_file.exists():
        ensure_directory(freeze_steps_file.parent)
        freeze_steps_file.write_text(json.dumps({"steps": steps}, indent=2), encoding="utf-8")
    return steps


def discover_anchor_steps(run_dir: Path) -> list[int]:
    anchor_dir = run_dir / "anchor_checkpoints"
    steps = []
    for path in anchor_dir.glob("anchor_step_*.zip"):
        try:
            steps.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return sorted(set(steps))


def build_run_entries(
    runs: list[tuple[str, Path]],
    steps: list[int],
    *,
    zero_model: Path | None,
) -> list[Entry]:
    entries = []
    for run_label, run_dir in runs:
        for step in steps:
            if step == 0 and zero_model is not None:
                model_path = resolve_existing_path(zero_model)
            else:
                model_path = run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip"
            available = model_path.exists()
            entries.append(
                Entry(
                    label=f"{run_label}_{step}",
                    display_label=f"{run_label} {step / 1_000_000.0:.1f}M",
                    run_label=run_label,
                    step=int(step),
                    run_dir=run_dir,
                    model_path=model_path if available else None,
                    available=available,
                )
            )
    return entries


def build_opponent_entries(base_report: dict[str, Any]) -> list[Entry]:
    if "opponent_pool" in base_report:
        return build_opponent_entries_from_pool(base_report["opponent_pool"])
    if "agents" in base_report:
        return build_opponent_entries_from_agents(base_report["agents"])
    raise KeyError("base report must contain either 'opponent_pool' or 'agents'")


def build_opponent_entries_from_pool(opponent_pool: list[dict[str, Any]]) -> list[Entry]:
    entries = []
    for row in opponent_pool:
        model_path = resolve_existing_path(Path(row["model_path"]))
        run_dir = resolve_existing_path(Path(row["run_dir"]))
        entries.append(
            Entry(
                label=str(row["label"]),
                display_label=str(row["display_label"]),
                run_label=str(row["run_label"]),
                step=int(row["step"]),
                run_dir=run_dir,
                model_path=model_path if model_path.exists() else None,
                available=model_path.exists(),
            )
        )
    return entries


def build_opponent_entries_from_agents(agents: list[dict[str, Any]]) -> list[Entry]:
    entries = []
    for row in agents:
        model_path = resolve_existing_path(Path(row["model_path"]))
        run_dir = infer_run_dir_from_model_path(model_path)
        step = int(row["step"])
        label = str(row["label"])
        entries.append(
            Entry(
                label=label,
                display_label=str(row.get("display_label", f"{step / 1_000_000.0:.1f}M")),
                run_label=str(row.get("run_label", "anchor")),
                step=step,
                run_dir=run_dir,
                model_path=model_path if model_path.exists() else None,
                available=model_path.exists(),
            )
        )
    return entries


def infer_run_dir_from_model_path(model_path: Path) -> Path:
    if model_path.parent.name == "anchor_checkpoints":
        return model_path.parent.parent
    return model_path.parent


def resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    replacements = [
        (
            "/selfplay_2d_recoverycfdefault_resp1_3m_",
            "/final_selfplay_2d_recoverycfdefault_resp1_3m_",
        ),
    ]
    text = str(path)
    for old, new in replacements:
        candidate = Path(text.replace(old, new))
        if candidate.exists():
            return candidate
    return path


def load_config_cached(cache: dict[Path, dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    if run_dir not in cache:
        cache[run_dir] = load_json(run_dir / "selfplay_config.json")
    return cache[run_dir]


def build_env_configs_cached(
    sim_cache: dict[Path, Any],
    discrete_cache: dict[Path, Any],
    run_dir: Path,
    train_config: dict[str, Any],
) -> tuple[Any, Any]:
    if run_dir not in sim_cache:
        sim_cache[run_dir] = build_sim_config(train_config)
    if run_dir not in discrete_cache:
        discrete_cache[run_dir] = build_discrete_action_config(train_config)
    return sim_cache[run_dir], discrete_cache[run_dir]


def write_manifest(
    output_dir: Path,
    *,
    base_report_path: Path,
    runs: list[tuple[str, Path]],
    selected_steps: list[int],
    zero_model: Path | None,
    eval_entries: list[Entry],
    opponent_entries: list[Entry],
    eval_rallies: int,
    seed: int,
    deterministic: bool,
) -> None:
    manifest = {
        "base_report": str(base_report_path),
        "zero_model": None if zero_model is None else str(zero_model),
        "runs": [{"label": label, "run_dir": str(run_dir)} for label, run_dir in runs],
        "selected_steps": selected_steps,
        "eval_rallies_per_pair": eval_rallies,
        "seed": seed,
        "deterministic": deterministic,
        "eval_agents": [entry_payload(entry) for entry in eval_entries],
        "opponent_pool": [entry_payload(entry) for entry in opponent_entries],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_report(
    *,
    base_report: dict[str, Any],
    base_report_path: Path,
    output_dir: Path,
    runs: list[tuple[str, Path]],
    selected_steps: list[int],
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
    matrix_report = build_matrix(eval_entries, opponent_entries, pair_results)
    return {
        "description": "Run checkpoints evaluated against the fixed pool from base_report.",
        "base_report": str(base_report_path),
        "output_dir": str(output_dir),
        "runs": [{"label": label, "run_dir": str(run_dir)} for label, run_dir in runs],
        "selected_steps": selected_steps,
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
        "eval_agent_metrics": build_eval_rows(eval_entries, pair_results, ratings),
        "win_rate_matrix": matrix_report,
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


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "fixed_pool_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_pair_csv(output_dir / "pair_results.csv", report["pair_results"])
    write_eval_metrics_csv(output_dir / "mean_win_rate_elo.csv", report["eval_agent_metrics"])
    write_matrix_csv(output_dir / "win_rate_matrix.csv", report["win_rate_matrix"])


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
