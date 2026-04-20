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
from badminton1d.render import render_stage, save_gif


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


def log_stage(record) -> None:
    action = record.validated_action.applied
    feasible = ",".join(str(idx) for idx in record.feasible_indices) if record.feasible_indices else "none"
    chosen = "None" if record.chosen_index is None else str(record.chosen_index)
    time_text = "None" if record.chosen_time is None else f"{record.chosen_time:.2f}s"
    print(
        f"stage={record.stage_index:02d} "
        f"hitter={record.state_before.current_hitter} "
        f"vx={action.v_x:.2f} vy={action.v_y:.2f} vz={action.v_z:.2f} "
        f"rec=({action.x_rec:.2f}, {action.y_rec:.2f}) feasible=[{feasible}] chosen={chosen} t={time_text}"
    )
    if record.notes:
        for note in record.notes:
            print(f"  note: {note}")
    if record.next_state.rally_done:
        print(f"  terminal: {record.terminal_reason}, winner={record.next_state.winner}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a heuristic 2D badminton rally demo.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_rally"))
    parser.add_argument("--max-stages", type=int, default=20)
    parser.add_argument("--no-gif", action="store_true", help="Skip creating outputs/demo_rally/rally.gif")
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

    print(f"left agent:  {left_agent.name}")
    print(f"right agent: {right_agent.name}")

    image_paths: list[Path] = []
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

        record = env.step(proposed_action, chosen_index)
        stage_path = args.output_dir / f"stage_{record.stage_index:03d}.png"
        render_stage(record, config, stage_path)
        image_paths.append(stage_path)
        log_stage(record)

    if image_paths and not args.no_gif:
        gif_path = args.output_dir / "rally.gif"
        save_gif(image_paths, gif_path, config)
        print(f"gif saved to {gif_path}")

    if env.state.rally_done:
        print(f"winner: {env.state.winner}")
    else:
        print(f"stopped after {args.max_stages} stages without a terminal rally result")


if __name__ == "__main__":
    main()
