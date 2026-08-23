from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.evaluation import ModelSelector
from badminton.playback import MatchTrace, RallyTrace, build_rally_trace
from badminton.state import Side
from badminton.utils import ensure_directory, side_y_bounds
from badminton.video import _shot_type_text, export_match_video
from scripts.round_robin_selfplay_video import (
    AgentSpec,
    _load_config,
    build_checkpoint_env,
    build_discrete_action_config,
    build_sim_config,
    ensure_mp4,
    rollout_rally,
)


EventPredicate = Callable[[dict[str, Any]], bool]


EVENTS: dict[str, EventPredicate] = {
    "opponent_lift_to_it": lambda stage: stage["hitter"] == "right" and bool(stage["is_lift_like"]),
    "it_lift_to_opponent": lambda stage: stage["hitter"] == "left" and bool(stage["is_lift_like"]),
    "opponent_smash_to_it": lambda stage: stage["hitter"] == "right" and bool(stage["is_smash_like"]),
    "it_smash_to_opponent": lambda stage: stage["hitter"] == "left" and bool(stage["is_smash_like"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find short checkpoint-vs-checkpoint rallies containing named shot events and export videos."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--anchor-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--rally-pause", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--min-stages", type=int, default=4)
    parser.add_argument("--max-stages", type=int, default=6)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def checkpoint_agent(run_dir: Path, step: int) -> AgentSpec:
    model_path = run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    return AgentSpec(label=f"step{step // 1000}k", run_dir=run_dir, model_path=model_path)


def _depth_ratio_from_net(side: Side, y: float, config: Any) -> float:
    low, high = side_y_bounds(side, config)
    if side == "left":
        return float((config.court.net_y - float(y)) / max(config.court.net_y - float(low), 1e-6))
    return float((float(y) - config.court.net_y) / max(float(high) - config.court.net_y, 1e-6))


def shot_sequence(trace: RallyTrace, config: Any) -> list[dict[str, Any]]:
    sequence = []
    for stage in trace.stages:
        shot_label = _shot_type_text(stage, config).removeprefix("shot ")
        vx, vy, vz = stage.shuttle_velocity
        horizontal_speed = float(np.hypot(vx, vy))
        speed = float(np.linalg.norm(stage.shuttle_velocity))
        theta_degrees = float(np.degrees(np.arctan2(vz, horizontal_speed)))
        landing_depth_ratio = _depth_ratio_from_net(stage.receiver_side, stage.shuttle_landing[1], config)
        is_lift_like = ("lift" in shot_label) or (
            theta_degrees >= 15.0 and landing_depth_ratio >= 0.60
        )
        is_smash_like = ("smash" in shot_label) or (
            theta_degrees <= -6.0 and speed >= 8.0
        )
        sequence.append(
            {
                "stage_index": int(stage.stage_index),
                "hitter": stage.hitter_side,
                "receiver": stage.receiver_side,
                "shot_type": shot_label,
                "theta_degrees": theta_degrees,
                "speed": speed,
                "landing_depth_ratio": landing_depth_ratio,
                "is_lift_like": is_lift_like,
                "is_smash_like": is_smash_like,
                "contact": [float(v) for v in stage.shuttle_start],
                "landing": [float(v) for v in stage.shuttle_landing],
            }
        )
    return sequence


def find_event_rallies(
    *,
    left_agent: AgentSpec,
    right_agent: AgentSpec,
    config: dict[str, Any],
    seed: int,
    max_candidates: int,
    min_stages: int,
    max_stages: int,
    deterministic: bool,
) -> tuple[dict[str, dict[str, Any]], Any]:
    sim_config = build_sim_config(config)
    discrete_action_config = build_discrete_action_config(config)
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
    remaining = set(EVENTS)
    selected: dict[str, dict[str, Any]] = {}
    length_counts: dict[int, int] = {}
    event_candidate_counts = {name: 0 for name in EVENTS}

    for candidate_index in range(max_candidates):
        if candidate_index and candidate_index % 500 == 0:
            print(
                f"{left_agent.label} vs {right_agent.label}: searched {candidate_index}, "
                f"remaining={sorted(remaining)}, lengths={dict(sorted(length_counts.items()))}, "
                f"event_hits={event_candidate_counts}",
                flush=True,
            )
        server: Side = "left" if bool(rng.integers(0, 2)) else "right"
        result = rollout_rally(env, selector, seed=seed + candidate_index, server=server)
        records = result["records"]
        assert isinstance(records, list)
        trace = build_rally_trace(records, sim_config)
        stage_count = len(trace.stages)
        length_counts[stage_count] = length_counts.get(stage_count, 0) + 1
        if not min_stages <= stage_count <= max_stages:
            continue

        sequence = shot_sequence(trace, sim_config)
        for event_name, predicate in EVENTS.items():
            if any(predicate(stage) for stage in sequence):
                event_candidate_counts[event_name] += 1
        for event_name in list(remaining):
            predicate = EVENTS[event_name]
            if any(predicate(stage) for stage in sequence):
                selected[event_name] = {
                    "candidate_index": candidate_index,
                    "seed": seed + candidate_index,
                    "server": server,
                    "result": {
                        key: value
                        for key, value in result.items()
                        if key not in {"records", "config", "metrics"}
                    },
                    "trace": trace,
                    "sequence": sequence,
                }
                remaining.remove(event_name)
                break
        if not remaining:
            return selected, sim_config

    missing = ", ".join(sorted(remaining))
    raise RuntimeError(
        f"Could not find all events for {left_agent.label} vs {right_agent.label} "
        f"within {max_candidates} candidate rallies; missing: {missing}"
    )


def with_score_metadata(
    trace: RallyTrace,
    *,
    rally_number: int,
    score_before_left: int,
    score_before_right: int,
    rally_pause: float,
) -> RallyTrace:
    winner = trace.winner
    score_after_left = score_before_left + (1 if winner == "left" else 0)
    score_after_right = score_before_right + (1 if winner == "right" else 0)
    return RallyTrace(
        stages=trace.stages,
        rally_done=trace.rally_done,
        winner=trace.winner,
        total_playback_time=trace.total_playback_time,
        rally_number=rally_number,
        server=None,
        score_before_left=score_before_left,
        score_before_right=score_before_right,
        score_after_left=score_after_left,
        score_after_right=score_after_right,
        pause_duration=rally_pause,
        match_winner=None,
    )


def export_trace_video(
    *,
    traces: list[RallyTrace],
    sim_config: Any,
    output_dir: Path,
    fps: int,
) -> dict[str, str | None]:
    ensure_directory(output_dir)
    score_left = traces[-1].score_after_left if traces else 0
    score_right = traces[-1].score_after_right if traces else 0
    winner: Side | None
    if score_left == score_right:
        winner = None
    else:
        winner = "left" if score_left > score_right else "right"
    if traces and winner is not None:
        last = traces[-1]
        traces[-1] = RallyTrace(**{**asdict(last), "match_winner": winner})
    match_trace = MatchTrace(
        rallies=traces,
        target_score=max(score_left, score_right, 1),
        score_left=score_left,
        score_right=score_right,
        winner=winner,
        total_playback_time=sum(trace.total_playback_time + trace.pause_duration for trace in traces),
    )
    export_result = export_match_video(match_trace, sim_config, output_dir, fps=fps, write_mp4=False)
    mp4_path = ensure_mp4(export_result, fps)
    return {
        "gif_path": str(export_result.gif_path),
        "mp4_path": None if mp4_path is None else str(mp4_path),
        "trace_path": str(export_result.trace_path),
    }


def export_matchup(
    *,
    run_dir: Path,
    left_step: int,
    right_step: int,
    output_dir: Path,
    seed: int,
    max_candidates: int,
    min_stages: int,
    max_stages: int,
    fps: int,
    rally_pause: float,
    deterministic: bool,
) -> dict[str, Any]:
    left_agent = checkpoint_agent(run_dir, left_step)
    right_agent = checkpoint_agent(run_dir, right_step)
    config = _load_config(run_dir)
    selected, sim_config = find_event_rallies(
        left_agent=left_agent,
        right_agent=right_agent,
        config=config,
        seed=seed,
        max_candidates=max_candidates,
        min_stages=min_stages,
        max_stages=max_stages,
        deterministic=deterministic,
    )
    matchup_dir = output_dir / f"{left_agent.label}_vs_{right_agent.label}"
    ensure_directory(matchup_dir)

    score_left = 0
    score_right = 0
    combined_traces: list[RallyTrace] = []
    event_summaries: dict[str, Any] = {}
    for rally_number, event_name in enumerate(EVENTS, start=1):
        payload = selected[event_name]
        trace = payload["trace"]
        assert isinstance(trace, RallyTrace)
        trace_with_score = with_score_metadata(
            trace,
            rally_number=rally_number,
            score_before_left=score_left,
            score_before_right=score_right,
            rally_pause=rally_pause,
        )
        score_left = trace_with_score.score_after_left
        score_right = trace_with_score.score_after_right
        combined_traces.append(trace_with_score)
        individual_paths = export_trace_video(
            traces=[trace_with_score],
            sim_config=sim_config,
            output_dir=matchup_dir / event_name,
            fps=fps,
        )
        event_summaries[event_name] = {
            "candidate_index": payload["candidate_index"],
            "seed": payload["seed"],
            "server": payload["server"],
            "stage_count": len(trace.stages),
            "winner": trace.winner,
            "sequence": payload["sequence"],
            "video": individual_paths,
        }

    combined_paths = export_trace_video(
        traces=combined_traces,
        sim_config=sim_config,
        output_dir=matchup_dir / "combined_selected_rallies",
        fps=fps,
    )
    summary = {
        "matchup": f"{left_agent.label}_vs_{right_agent.label}",
        "left_agent": left_agent.label,
        "right_agent": right_agent.label,
        "left_model_path": str(left_agent.model_path),
        "right_model_path": str(right_agent.model_path),
        "seed": seed,
        "deterministic": deterministic,
        "stage_range": [min_stages, max_stages],
        "combined_video": combined_paths,
        "events": event_summaries,
    }
    (matchup_dir / "selected_rallies_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.rally_pause < 0.0:
        raise ValueError("--rally-pause must be non-negative")
    if args.min_stages <= 0 or args.max_stages < args.min_stages:
        raise ValueError("--min-stages and --max-stages must describe a positive range")

    output_dir = args.output_dir or (args.run_dir / "videos" / "selected_checkpoint_rallies")
    ensure_directory(output_dir)
    matchups = [(200_000, 3_000_000), (6_000_000, 3_000_000)]
    summaries = []
    for index, (left_step, right_step) in enumerate(matchups):
        summaries.append(
            export_matchup(
                run_dir=args.run_dir,
                left_step=left_step,
                right_step=right_step,
                output_dir=output_dir,
                seed=args.seed + index * 100_000,
                max_candidates=args.max_candidates,
                min_stages=args.min_stages,
                max_stages=args.max_stages,
                fps=args.fps,
                rally_pause=args.rally_pause,
                deterministic=args.deterministic,
            )
        )

    manifest = {
        "run_dir": str(args.run_dir),
        "output_dir": str(output_dir),
        "matchups": summaries,
    }
    manifest_path = output_dir / "selected_checkpoint_rallies_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    for summary in summaries:
        print(summary["matchup"], summary["combined_video"]["mp4_path"])


if __name__ == "__main__":
    main()
