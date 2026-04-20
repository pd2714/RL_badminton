from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.agents import GreedyReceiver, RandomValidHitter, StageAgent
from badminton1d.config import SimulationConfig
from badminton1d.match import MatchConfig, MatchResult, run_match
from badminton1d.playback import build_match_trace
from badminton1d.render import ScoreboardOverlay, render_stage
from badminton1d.state import StageRecord
from badminton1d.video import export_match_video


def build_demo_agents(seed: int) -> tuple[StageAgent, StageAgent]:
    left_agent = StageAgent(
        name="LeftRandom",
        hitter_policy=RandomValidHitter(seed=seed + 1),
        receiver_policy=GreedyReceiver(mode="earliest"),
    )
    right_agent = StageAgent(
        name="RightRandom",
        hitter_policy=RandomValidHitter(seed=seed + 2),
        receiver_policy=GreedyReceiver(mode="highest"),
    )
    return left_agent, right_agent


def _score_overlay(match_result: MatchResult, rally_index: int, record: StageRecord) -> ScoreboardOverlay:
    rally_result = match_result.rallies[rally_index]
    is_terminal_stage = record.next_state.rally_done
    score_left = rally_result.score_after.left if is_terminal_stage else rally_result.score_before.left
    score_right = rally_result.score_after.right if is_terminal_stage else rally_result.score_before.right
    match_winner = match_result.winner if (
        is_terminal_stage
        and rally_index == len(match_result.rallies) - 1
    ) else None
    return ScoreboardOverlay(
        score_left=score_left,
        score_right=score_right,
        current_server=rally_result.server,
        rally_number=rally_result.rally_number,
        stage_number=record.stage_index + 1,
        hitter_side=record.state_before.current_hitter,
        point_winner=rally_result.winner if is_terminal_stage else None,
        match_winner=match_winner,
    )


def write_stage_images(
    match_result: MatchResult,
    config: SimulationConfig,
    output_dir: Path,
) -> list[Path]:
    stage_dir = output_dir / "stages"
    stage_paths: list[Path] = []
    for rally_index, rally_result in enumerate(match_result.rallies):
        for record in rally_result.records:
            stage_path = stage_dir / f"rally_{rally_result.rally_number:03d}_stage_{record.stage_index + 1:03d}.png"
            render_stage(
                record,
                config,
                stage_path,
                overlay=_score_overlay(match_result, rally_index, record),
            )
            stage_paths.append(stage_path)
    return stage_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a full badminton match and render match-level video.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_match_video"))
    parser.add_argument("--target-score", type=int, default=11)
    parser.add_argument("--initial-server", choices=("left", "right"), default="left")
    parser.add_argument("--max-stages-per-rally", type=int, default=120)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rally-pause", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_score <= 0:
        raise ValueError("--target-score must be positive")
    if args.max_stages_per_rally <= 0:
        raise ValueError("--max-stages-per-rally must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.rally_pause < 0.0:
        raise ValueError("--rally-pause must be zero or greater")

    config = SimulationConfig()
    match_config = MatchConfig(
        target_score=args.target_score,
        max_stages_per_rally=args.max_stages_per_rally,
    )
    left_agent, right_agent = build_demo_agents(seed=args.seed)
    match_result = run_match(
        left_agent,
        right_agent,
        config,
        match_config=match_config,
        initial_server=args.initial_server,
    )
    stage_paths = write_stage_images(match_result, config, args.output_dir)
    match_trace = build_match_trace(match_result, config, rally_pause=args.rally_pause)
    export_result = export_match_video(
        match_trace,
        config,
        args.output_dir,
        fps=args.fps,
    )

    print(f"target score: {match_result.target_score}")
    print(f"initial server: {args.initial_server}")
    print(f"rallies: {len(match_result.rallies)}")
    print(f"stages: {sum(len(rally.records) for rally in match_result.rallies)}")
    print(f"score: {match_result.final_score.left}-{match_result.final_score.right}")
    print(f"winner: {match_result.winner}")
    print(f"stage images: {len(stage_paths)}")
    print(f"frames: {len(export_result.frame_paths)}")
    print(f"gif: {export_result.gif_path}")
    if export_result.mp4_path is not None:
        print(f"mp4: {export_result.mp4_path}")
    else:
        print("mp4: unavailable")
    print(f"trace: {export_result.trace_path}")


if __name__ == "__main__":
    main()
