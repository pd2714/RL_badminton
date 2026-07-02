from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.evaluation import ModelSelector
from badminton1d.playback import match_trace_to_dict
from badminton1d.utils import ensure_directory
from scripts.export_checkpoint_matchup_videos import checkpoint_agent
from scripts.round_robin_selfplay_video import (
    _load_config,
    _next_server,
    _rally_trace_from_result,
    _resolve_random_server,
    build_checkpoint_env,
    build_discrete_action_config,
    build_match_trace,
    build_sim_config,
    rollout_rally,
)


DEFAULT_MATCHUPS = ((3_000_000, 6_000_000), (4_000_000, 6_000_000), (5_000_000, 6_000_000), (6_000_000, 6_000_000))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trace-only checkpoint matchups for human sanity panels A/B.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-score", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--rally-pause", type=float, default=0.2)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--matchup",
        nargs=2,
        type=int,
        action="append",
        metavar=("LEFT_STEP", "RIGHT_STEP"),
        help="Checkpoint steps to export. May be repeated. Defaults to 3M/4M/5M/6M vs 6M.",
    )
    return parser.parse_args()


def export_trace_matchup(
    *,
    run_dir: Path,
    left_step: int,
    right_step: int,
    output_dir: Path,
    target_score: int,
    seed: int,
    rally_pause: float,
    deterministic: bool,
) -> dict[str, Any]:
    ensure_directory(output_dir)
    config = _load_config(run_dir)
    sim_config = build_sim_config(config)
    discrete_action_config = build_discrete_action_config(config)
    left_agent = checkpoint_agent(run_dir, left_step)
    right_agent = checkpoint_agent(run_dir, right_step)
    left_model = PPO.load(left_agent.model_path)
    env = build_checkpoint_env(
        train_side="left",
        train_config=config,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        opponent=right_agent,
        seed=seed,
        deterministic=deterministic,
        include_records_in_info=True,
    )
    selector = ModelSelector(model=left_model, deterministic=deterministic)
    rng = np.random.default_rng(seed)
    current_server = _resolve_random_server(rng)
    score_left = 0
    score_right = 0
    traces = []
    summaries: list[dict[str, Any]] = []
    for rally_number in range(1, 10_000):
        result = rollout_rally(env, selector, seed=seed + rally_number, server=current_server)
        winner = result["winner"]
        next_score_left = score_left + (1 if winner == "left" else 0)
        next_score_right = score_right + (1 if winner == "right" else 0)
        traces.append(
            _rally_trace_from_result(
                result,
                sim_config,
                rally_number=rally_number,
                score_before_left=score_left,
                score_before_right=score_right,
                score_after_left=next_score_left,
                score_after_right=next_score_right,
                rally_pause=rally_pause,
            )
        )
        summaries.append(
            {
                "rally_number": rally_number,
                "server": current_server,
                "winner": winner,
                "score_after_left": next_score_left,
                "score_after_right": next_score_right,
                "rally_length": result["rally_length"],
                "invalid_action_rate": result["invalid_action_rate"],
            }
        )
        score_left = next_score_left
        score_right = next_score_right
        if max(score_left, score_right) >= target_score:
            break
        current_server = _next_server(
            current_server=current_server,
            winner=winner,
            random_server_each_rally=True,
            rng=rng,
        )

    match_trace = build_match_trace(traces, score_left, score_right)
    trace_path = output_dir / "match_trace.json"
    trace_path.write_text(json.dumps(match_trace_to_dict(match_trace), indent=2), encoding="utf-8")
    summary = {
        "left_agent": left_agent.label,
        "right_agent": right_agent.label,
        "left_step": left_step,
        "right_step": right_step,
        "target_score": target_score,
        "score_left": score_left,
        "score_right": score_right,
        "winner": left_agent.label if score_left > score_right else right_agent.label,
        "random_server_each_rally": True,
        "deterministic": deterministic,
        "trace_path": str(trace_path),
        "rallies": summaries,
    }
    (output_dir / "match_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.target_score <= 0:
        raise ValueError("--target-score must be positive")
    if args.rally_pause < 0.0:
        raise ValueError("--rally-pause must be non-negative")
    matchups = tuple((int(left), int(right)) for left, right in args.matchup) if args.matchup else DEFAULT_MATCHUPS
    output_dir = args.output_dir or (args.run_dir / "videos" / "human_sanity_match_traces")
    ensure_directory(output_dir)
    summaries = []
    for index, (left_step, right_step) in enumerate(matchups):
        matchup_dir = output_dir / f"step{left_step // 1000}k_vs_step{right_step // 1000}k"
        summaries.append(
            export_trace_matchup(
                run_dir=args.run_dir,
                left_step=left_step,
                right_step=right_step,
                output_dir=matchup_dir,
                target_score=args.target_score,
                seed=args.seed + index * 100_000,
                rally_pause=args.rally_pause,
                deterministic=args.deterministic,
            )
        )
    manifest = {
        "run_dir": str(args.run_dir),
        "output_dir": str(output_dir),
        "target_score": args.target_score,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "matchups": [{"left_step": left, "right_step": right} for left, right in matchups],
        "matches": summaries,
    }
    manifest_path = output_dir / "human_sanity_match_traces_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    for summary in summaries:
        print(f"{summary['left_agent']} vs {summary['right_agent']}: {summary['trace_path']}")


if __name__ == "__main__":
    main()
