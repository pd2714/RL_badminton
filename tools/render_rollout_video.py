from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.agents import GreedyReceiver, RandomValidHitter, StageAgent
from badminton1d.config import SimulationConfig
from badminton1d.dynamics import feasible_intercept_indices, validate_and_clip_shot_action
from badminton1d.env import Badminton1DEnv
from badminton1d.render import render_stage_image
from badminton1d.state import Side, StageRecord, StageState
from badminton1d.utils import ensure_directory, recovery_bounds

DEFAULT_OUTPUT_DIR = Path("outputs/rollout_videos")
UINT32_MAX_EXCLUSIVE = 2**32


@dataclass
class RolloutResult:
    seed: int
    initial_state: StageState
    records: list[StageRecord]
    frames: list[np.ndarray]
    final_state: StageState


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return parsed


def build_random_agents(seed: int) -> tuple[StageAgent, StageAgent]:
    seed_rng = np.random.default_rng(seed)
    left_seed = int(seed_rng.integers(UINT32_MAX_EXCLUSIVE))
    right_seed = int(seed_rng.integers(UINT32_MAX_EXCLUSIVE))
    left_agent = StageAgent(
        name="LeftRandom",
        hitter_policy=RandomValidHitter(seed=left_seed),
        receiver_policy=GreedyReceiver(mode="earliest"),
    )
    right_agent = StageAgent(
        name="RightRandom",
        hitter_policy=RandomValidHitter(seed=right_seed),
        receiver_policy=GreedyReceiver(mode="highest"),
    )
    return left_agent, right_agent


def random_initial_state(rng: np.random.Generator, config: SimulationConfig) -> StageState:
    x_left_low, x_left_high = recovery_bounds("left", config)
    x_right_low, x_right_high = recovery_bounds("right", config)
    x_left = float(rng.uniform(x_left_low, x_left_high))
    x_right = float(rng.uniform(x_right_low, x_right_high))
    current_hitter: Side = "left" if rng.random() < 0.5 else "right"
    x0 = x_left if current_hitter == "left" else x_right
    z0_low = max(config.player.z_min + 0.1, 0.8)
    z0_high = min(config.player.z_max - 0.1, 2.4)
    z0 = float(rng.uniform(z0_low, z0_high))
    return StageState(
        x_left=x_left,
        x_right=x_right,
        current_hitter=current_hitter,
        x0=x0,
        z0=z0,
        rally_done=False,
        winner=None,
        stage_index=0,
    )


def scaled_figure_size(config: SimulationConfig, figure_scale: float) -> tuple[float, float]:
    base_width, base_height = config.render.figure_size
    return base_width * figure_scale, base_height * figure_scale


def simulate_rollout(
    rollout_seed: int,
    max_stages: int,
    config: SimulationConfig,
    *,
    figure_scale: float,
    dpi: int,
) -> RolloutResult:
    rng = np.random.default_rng(rollout_seed)
    initial_state = random_initial_state(rng, config)
    left_agent, right_agent = build_random_agents(int(rng.integers(UINT32_MAX_EXCLUSIVE)))
    initial_state = replace(
        initial_state,
        reaction_time_left=float(left_agent.reaction_time),
        reaction_time_right=float(right_agent.reaction_time),
    )

    env = Badminton1DEnv(config=config)
    env.reset(initial_state)

    records: list[StageRecord] = []
    frames: list[np.ndarray] = []
    panel_size = scaled_figure_size(config, figure_scale)

    for _ in range(max_stages):
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
        records.append(record)
        frames.append(
            render_stage_image(
                record,
                config,
                figure_size=panel_size,
                dpi=dpi,
                annotate=False,
                show_player_labels=False,
                monochrome=True,
            )
        )

    return RolloutResult(
        seed=rollout_seed,
        initial_state=initial_state,
        records=records,
        frames=frames,
        final_state=env.state,
    )


def grid_shape(count: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return rows, cols


def tile_rollout_frames(rollouts: list[RolloutResult], padding: int) -> list[np.ndarray]:
    if not rollouts:
        return []
    if len(rollouts) == 1:
        return rollouts[0].frames

    rows, cols = grid_shape(len(rollouts))
    frame_height, frame_width, channels = rollouts[0].frames[0].shape
    canvas_height = rows * frame_height + padding * max(rows - 1, 0)
    canvas_width = cols * frame_width + padding * max(cols - 1, 0)
    frame_count = max(len(rollout.frames) for rollout in rollouts)

    tiled_frames: list[np.ndarray] = []
    for frame_index in range(frame_count):
        tiled = np.full((canvas_height, canvas_width, channels), 255, dtype=np.uint8)
        for rollout_index, rollout in enumerate(rollouts):
            source_frame = rollout.frames[min(frame_index, len(rollout.frames) - 1)]
            row = rollout_index // cols
            col = rollout_index % cols
            top = row * (frame_height + padding)
            left = col * (frame_width + padding)
            tiled[top : top + frame_height, left : left + frame_width] = source_frame
        tiled_frames.append(tiled)
    return tiled_frames


def default_output_path(seed: int, num_rollouts: int) -> Path:
    stem = f"seed_{seed:04d}"
    if num_rollouts == 1:
        return DEFAULT_OUTPUT_DIR / f"rally_{stem}.mp4"
    return DEFAULT_OUTPUT_DIR / f"rally_grid_{num_rollouts}_{stem}.mp4"


def save_animation(frames: list[np.ndarray], output_path: Path, fps: float) -> Path:
    if not frames:
        raise ValueError("cannot save an animation with zero frames")

    ensure_directory(output_path.parent)
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        imageio.mimsave(output_path, frames, duration=1.0 / fps)
        return output_path

    target_path = output_path if suffix == ".mp4" else output_path.with_suffix(".mp4")
    try:
        imageio.mimsave(target_path, frames, fps=fps, macro_block_size=1)
        return target_path
    except Exception:
        fallback_path = output_path.with_suffix(".gif")
        imageio.mimsave(fallback_path, frames, duration=1.0 / fps)
        return fallback_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render random badminton rally rollouts to mp4 or gif.")
    parser.add_argument("--out", type=Path, default=None, help="Output file path. Defaults to outputs/rollout_videos/*.mp4")
    parser.add_argument(
        "--horizon",
        "--max-stages",
        dest="max_stages",
        type=_positive_int,
        default=20,
        help="Maximum number of stages to simulate per rollout.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master random seed.")
    parser.add_argument(
        "--num-rollouts",
        type=_positive_int,
        default=1,
        help="Number of random rollouts to tile into one video.",
    )
    parser.add_argument("--fps", type=_positive_float, default=2.0, help="Animation frames per second.")
    parser.add_argument(
        "--figure-scale",
        type=_positive_float,
        default=0.65,
        help="Scale factor for each rendered rollout panel.",
    )
    parser.add_argument("--dpi", type=_positive_int, default=120, help="Rendering dpi for each rollout panel.")
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Pixel padding between rollout panels in tiled mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.padding < 0:
        raise ValueError("--padding must be zero or greater")

    config = SimulationConfig()
    master_rng = np.random.default_rng(args.seed)
    rollout_seeds = [int(master_rng.integers(UINT32_MAX_EXCLUSIVE)) for _ in range(args.num_rollouts)]

    rollouts = [
        simulate_rollout(
            rollout_seed,
            args.max_stages,
            config,
            figure_scale=args.figure_scale,
            dpi=args.dpi,
        )
        for rollout_seed in rollout_seeds
    ]
    frames = tile_rollout_frames(rollouts, args.padding)

    output_path = args.out or default_output_path(args.seed, args.num_rollouts)
    saved_path = save_animation(frames, output_path, args.fps)
    print(saved_path.resolve())


if __name__ == "__main__":
    main()
