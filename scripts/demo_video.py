from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.agents import GreedyReceiver, RandomValidHitter, SafeHitter, StageAgent
from badminton1d.config import SimulationConfig
from badminton1d.dynamics import feasible_intercept_indices, validate_and_clip_shot_action
from badminton1d.env import Badminton1DEnv, default_initial_state
from badminton1d.playback import build_rally_trace
from badminton1d.video import export_rally_video


def build_demo_agents() -> tuple[StageAgent, StageAgent]:
    left_agent = StageAgent(
        name="LeftSafe",
        hitter_policy=SafeHitter(),
        receiver_policy=GreedyReceiver(mode="earliest"),
    )
    right_agent = StageAgent(
        name="RightRandom",
        hitter_policy=RandomValidHitter(seed=7),
        receiver_policy=GreedyReceiver(mode="highest"),
    )
    return left_agent, right_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a continuous 2D badminton rally video.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_video"))
    parser.add_argument("--max-stages", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stage-pause", type=float, default=0.15)
    args = parser.parse_args()

    config = SimulationConfig()
    env = Badminton1DEnv(config=config)
    left_agent, right_agent = build_demo_agents()
    env.reset(
        replace(
            default_initial_state(config),
            reaction_time_left=float(left_agent.reaction_time),
            reaction_time_right=float(right_agent.reaction_time),
        )
    )

    records = []
    for _ in range(args.max_stages):
        state = env.state
        if state.rally_done:
            break

        hitter_agent = left_agent if state.current_hitter == "left" else right_agent
        receiver_agent = right_agent if state.current_hitter == "left" else left_agent

        proposed_action = hitter_agent.choose_shot_action(state, config)
        validated = validate_and_clip_shot_action(state, proposed_action, config)
        feasible = feasible_intercept_indices(state, validated.applied, config)
        chosen_index = receiver_agent.choose_intercept_index(state, validated.applied, feasible, config)
        records.append(env.step(proposed_action, chosen_index))

    trace = build_rally_trace(records, config)
    result = export_rally_video(
        trace,
        config,
        args.output_dir,
        fps=args.fps,
        stage_pause=args.stage_pause,
    )

    print(f"frames: {len(result.frame_paths)}")
    print(f"gif: {result.gif_path}")
    if result.mp4_path is not None:
        print(f"mp4: {result.mp4_path}")
    else:
        print("mp4: unavailable")
    print(f"trace: {result.trace_path}")
    if trace.rally_done:
        print(f"winner: {trace.winner}")
    else:
        print(f"stopped after {args.max_stages} stages without a terminal rally result")


if __name__ == "__main__":
    main()
