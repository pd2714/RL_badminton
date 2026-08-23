from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.utils import ensure_directory
from scripts.round_robin_selfplay_video import (
    AgentSpec,
    _load_config,
    build_discrete_action_config,
    build_sim_config,
    export_pair_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export videos for specific self-play anchor checkpoint matchups.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-score", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--rally-pause", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def checkpoint_agent(run_dir: Path, step: int) -> AgentSpec:
    model_path = run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    return AgentSpec(label=f"step{step // 1000}k", run_dir=run_dir, model_path=model_path)


def export_matchup(
    *,
    run_dir: Path,
    left_step: int,
    right_step: int,
    output_dir: Path,
    target_score: int,
    fps: int,
    rally_pause: float,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    config = _load_config(run_dir)
    sim_config = build_sim_config(config)
    discrete_action_config = build_discrete_action_config(config)
    left_agent = checkpoint_agent(run_dir, left_step)
    right_agent = checkpoint_agent(run_dir, right_step)
    left_model = PPO.load(left_agent.model_path)
    matchup_dir = output_dir / f"{left_agent.label}_vs_{right_agent.label}"
    return export_pair_video(
        left_agent=left_agent,
        right_agent=right_agent,
        left_model=left_model,
        left_config=config,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        target_score=target_score,
        output_dir=matchup_dir,
        seed=seed,
        fps=fps,
        rally_pause=rally_pause,
        deterministic=deterministic,
    )


def main() -> None:
    args = parse_args()
    if args.target_score <= 0:
        raise ValueError("--target-score must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.rally_pause < 0.0:
        raise ValueError("--rally-pause must be non-negative")

    output_dir = args.output_dir or (args.run_dir / "videos" / "checkpoint_matchups")
    ensure_directory(output_dir)
    matchups = [(200_000, 3_000_000), (6_000_000, 3_000_000)]
    summaries = [
        export_matchup(
            run_dir=args.run_dir,
            left_step=left_step,
            right_step=right_step,
            output_dir=output_dir,
            target_score=args.target_score,
            fps=args.fps,
            rally_pause=args.rally_pause,
            seed=args.seed + index * 100_000,
            deterministic=args.deterministic,
        )
        for index, (left_step, right_step) in enumerate(matchups)
    ]
    manifest = {
        "run_dir": str(args.run_dir),
        "output_dir": str(output_dir),
        "target_score": args.target_score,
        "fps": args.fps,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "videos": summaries,
    }
    manifest_path = output_dir / "checkpoint_matchup_videos_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    for summary in summaries:
        print(f"{summary['left_agent']} vs {summary['right_agent']}: {summary['mp4_path']}")


if __name__ == "__main__":
    main()
