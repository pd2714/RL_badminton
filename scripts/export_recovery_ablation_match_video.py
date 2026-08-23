from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.state import Side
from badminton.utils import ensure_directory
from badminton.video import export_match_video
from scripts.evaluate_recovery_ablation_fixed_pool import (
    PoolOpponentSpec,
    RecoveryAgentSpec,
    RecoveryOverrideModelSelector,
    build_ablation_env,
)
from scripts.round_robin_selfplay_video import (
    _load_config,
    _next_server,
    _rally_trace_from_result,
    _resolve_random_server,
    build_discrete_action_config,
    build_match_trace,
    build_sim_config,
    ensure_mp4,
    rollout_rally,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a match video between recovery-ablation variants.")
    parser.add_argument("fixed_pool_dir", type=Path)
    parser.add_argument("--left", default="learned_6m")
    parser.add_argument("--right", default="centered_6m")
    parser.add_argument("--target-score", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--rally-pause", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_manifest(fixed_pool_dir: Path) -> dict[str, Any]:
    manifest_path = fixed_pool_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _agent_from_manifest(fixed_pool_dir: Path, manifest: dict[str, Any], label: str) -> RecoveryAgentSpec:
    for row in manifest.get("agents", []):
        if row.get("label") == label:
            return RecoveryAgentSpec(
                label=str(row["label"]),
                step=int(row["step"]),
                recovery_mode=str(row["recovery_mode"]),
                model_path=_resolve_path(str(row["model_path"])),
                source_model_path=_resolve_path(str(row["source_model_path"])),
            )
    fallback_dir = fixed_pool_dir / "agents" / label
    metadata_path = fallback_dir / "ablation_agent.json"
    if not metadata_path.exists():
        match = re.fullmatch(r"(learned|centered|heuristic)_(\d+)m", label)
        if match is None:
            raise ValueError(f"Unknown recovery-ablation agent: {label}")
        mode = match.group(1)
        step = int(match.group(2)) * 1_000_000
        run_dir = _resolve_path(str(manifest["run_dir"]))
        model_path = run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing inferred checkpoint for {label}: {model_path}")
        return RecoveryAgentSpec(
            label=label,
            step=step,
            recovery_mode=mode,
            model_path=model_path,
            source_model_path=model_path,
        )
    row = json.loads(metadata_path.read_text(encoding="utf-8"))
    return RecoveryAgentSpec(
        label=str(row["label"]),
        step=int(row["step"]),
        recovery_mode=str(row["recovery_mode"]),
        model_path=_resolve_path(str(row["model_path"])),
        source_model_path=_resolve_path(str(row["source_model_path"])),
    )


def _config_dir_for_agent(fixed_pool_dir: Path, manifest: dict[str, Any], label: str) -> Path:
    agent_dir = fixed_pool_dir / "agents" / label
    if (agent_dir / "selfplay_config.json").exists():
        return agent_dir
    run_dir = _resolve_path(str(manifest["run_dir"]))
    if not (run_dir / "selfplay_config.json").exists():
        raise FileNotFoundError(f"Missing selfplay config for {label}: {run_dir / 'selfplay_config.json'}")
    return run_dir


def export_recovery_ablation_match(
    *,
    fixed_pool_dir: Path,
    left_label: str,
    right_label: str,
    target_score: int,
    output_dir: Path,
    seed: int,
    fps: int,
    rally_pause: float,
    deterministic: bool,
) -> dict[str, Any]:
    manifest = _load_manifest(fixed_pool_dir)
    left_agent = _agent_from_manifest(fixed_pool_dir, manifest, left_label)
    right_agent = _agent_from_manifest(fixed_pool_dir, manifest, right_label)
    if not left_agent.model_path.exists():
        raise FileNotFoundError(f"Missing left model: {left_agent.model_path}")
    if not right_agent.model_path.exists():
        raise FileNotFoundError(f"Missing right model: {right_agent.model_path}")

    config = _load_config(_config_dir_for_agent(fixed_pool_dir, manifest, left_label))
    sim_config = build_sim_config(config)
    discrete_config = build_discrete_action_config(config)
    left_model = PPO.load(left_agent.model_path)
    right_opponent = PoolOpponentSpec(
        label=right_agent.label,
        kind="checkpoint",
        step=right_agent.step,
        recovery_mode=right_agent.recovery_mode,
        model_path=right_agent.model_path,
    )
    env = build_ablation_env(
        train_side="left",
        train_config=config,
        sim_config=sim_config,
        discrete_config=discrete_config,
        opponent=right_opponent,
        seed=seed,
        deterministic=deterministic,
    )
    selector = RecoveryOverrideModelSelector(
        model=left_model,
        recovery_mode=left_agent.recovery_mode,
        action_mapper=env.action_mapper,
        sim_config=sim_config,
        agent_side="left",
        deterministic=deterministic,
    )

    rng = np.random.default_rng(seed)
    current_server: Side = _resolve_random_server(rng)
    score_left = 0
    score_right = 0
    traces = []
    rallies: list[dict[str, Any]] = []
    try:
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
            rallies.append(
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
    finally:
        env.close()

    ensure_directory(output_dir)
    match_trace = build_match_trace(traces, score_left, score_right)
    video_result = export_match_video(match_trace, sim_config, output_dir, fps=fps, write_mp4=False)
    mp4_path = ensure_mp4(video_result, fps)
    summary = {
        "fixed_pool_dir": str(fixed_pool_dir),
        "left_agent": left_agent.label,
        "left_recovery_mode": left_agent.recovery_mode,
        "right_agent": right_agent.label,
        "right_recovery_mode": right_agent.recovery_mode,
        "target_score": target_score,
        "score_left": score_left,
        "score_right": score_right,
        "winner": left_agent.label if score_left > score_right else right_agent.label,
        "seed": seed,
        "deterministic": deterministic,
        "random_server_each_rally": True,
        "fps": fps,
        "rally_pause": rally_pause,
        "gif_path": str(video_result.gif_path),
        "mp4_path": None if mp4_path is None else str(mp4_path),
        "trace_path": str(video_result.trace_path),
        "rallies": rallies,
    }
    (output_dir / "match_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.target_score <= 0:
        raise ValueError("--target-score must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.rally_pause < 0.0:
        raise ValueError("--rally-pause must be non-negative")
    fixed_pool_dir = args.fixed_pool_dir
    output_dir = args.output_dir or fixed_pool_dir / "videos" / f"{args.left}__vs__{args.right}_to{args.target_score}"
    summary = export_recovery_ablation_match(
        fixed_pool_dir=fixed_pool_dir,
        left_label=args.left,
        right_label=args.right,
        target_score=args.target_score,
        output_dir=output_dir,
        seed=args.seed,
        fps=args.fps,
        rally_pause=args.rally_pause,
        deterministic=args.deterministic,
    )
    print(f"score: {summary['score_left']}-{summary['score_right']}")
    print(f"winner: {summary['winner']}")
    print(f"mp4: {summary['mp4_path']}")
    print(f"gif: {summary['gif_path']}")
    print(f"trace: {summary['trace_path']}")
    print(f"summary: {output_dir / 'match_summary.json'}")


if __name__ == "__main__":
    main()
