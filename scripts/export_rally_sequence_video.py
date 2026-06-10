from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionConfig
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.opponents import make_opponent
from badminton1d.playback import MatchTrace, RallyTrace, build_rally_trace
from badminton1d.rl_env import BadmintonRLEnv
from badminton1d.evaluation import choose_model_action
from badminton1d.selfplay import CheckpointPool, FixedCheckpointOpponent, LiveModelOpponent, build_selfplay_env
from badminton1d.shot_generators import TacticRuntimeConfig
from badminton1d.state import Side
from badminton1d.utils import ensure_directory
from badminton1d.video import VideoExportResult, export_match_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a continuous video for a sequence of PPO rallies.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-side", choices=("left", "right"), default="left")
    parser.add_argument("--opponent", choices=("safe", "random", "greedy", "mirror-self", "newest-checkpoint"), default="safe")
    parser.add_argument("--checkpoint-pool-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-pool-base", type=Path, default=None)
    parser.add_argument("--checkpoint-pool-size", type=int, default=6)
    parser.add_argument("--checkpoint-sampling-mode", choices=("uniform", "random"), default="uniform")
    parser.add_argument("--initial-server", choices=("left", "right", "train", "opponent", "random"), default="random")
    parser.add_argument("--random-server-each-rally", action="store_true")
    parser.add_argument("--fixed-server-each-rally", action="store_true")
    parser.add_argument("--start-rally", type=int, default=0)
    parser.add_argument("--end-rally", type=int, default=100)
    parser.add_argument("--target-score", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--rally-pause", type=float, default=0.2)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mirror-sides", action="store_true")
    parser.add_argument("--mirror-match-fraction", type=float, default=0.0)
    parser.add_argument("--random-service-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reaction-time", type=float, default=0.15)
    parser.add_argument("--court-mode", choices=("1d", "2d"), default="2d")
    parser.add_argument(
        "--policy-type",
        choices=("conditional_prob", "continuous_action", "velocity_oriented", "tactic_oriented", "mixed_discrete_continous"),
        default="velocity_oriented",
    )
    parser.add_argument("--regenerate-lookup-table", action="store_true")
    parser.add_argument("--lookup-table-dir", type=Path, default=Path("lookup_tables"))
    parser.add_argument("--player-speed", type=float, default=SimulationConfig().player.v_max)
    parser.add_argument("--racket-length", type=float, default=SimulationConfig().player.r_reach)
    parser.add_argument("--max-hitting-height", type=float, default=SimulationConfig().player.z_max)
    parser.add_argument("--movement-model", choices=("constant_velocity", "accelerated"), default=SimulationConfig().player.movement_model)
    parser.add_argument("--player-acceleration", type=float, default=SimulationConfig().player.acceleration)
    parser.add_argument("--player-deceleration", type=float, default=None)
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="ballistic")
    parser.add_argument("--drag-coefficient", type=float, default=0.2)
    parser.add_argument("--horizontal-drag-coefficient", type=float, default=0.2)
    parser.add_argument("--vertical-drag-coefficient", type=float, default=0.16)
    parser.add_argument("--shuttle-speed-min", type=float, default=0.1)
    parser.add_argument("--shuttle-speed-max", type=float, default=SimulationConfig().action.vy_max_forward)
    parser.add_argument("--intercept-count", type=int, default=20)
    parser.add_argument("--reaction-miss-fast-threshold", type=float, default=SimulationConfig().action.reaction_miss_fast_threshold)
    parser.add_argument("--reaction-miss-fast-probability", type=float, default=SimulationConfig().action.reaction_miss_fast_probability)
    parser.add_argument("--reaction-miss-secondary-threshold", type=float, default=SimulationConfig().action.reaction_miss_secondary_threshold)
    parser.add_argument("--reaction-miss-secondary-probability", type=float, default=SimulationConfig().action.reaction_miss_secondary_probability)
    parser.add_argument("--reaction-miss-zero-threshold", type=float, default=SimulationConfig().action.reaction_miss_zero_threshold)
    parser.add_argument("--phi-bins", type=int, default=DiscreteActionConfig().phi_bins)
    parser.add_argument("--theta-bins", type=int, default=DiscreteActionConfig().theta_bins)
    parser.add_argument("--speed-bins", type=int, default=DiscreteActionConfig().speed_bins)
    parser.add_argument("--vx-bins", dest="phi_bins", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--vy-bins", dest="theta_bins", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--vz-bins", dest="speed_bins", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--x-rec-bins", type=int, default=DiscreteActionConfig().x_rec_bins)
    parser.add_argument("--y-rec-bins", type=int, default=DiscreteActionConfig().y_rec_bins)
    return parser.parse_args()


def _resolve_initial_server(value: str, train_side: Side, rng: np.random.Generator) -> Side:
    if value in {"left", "right"}:
        return value
    if value == "train":
        return train_side
    if value == "opponent":
        return "right" if train_side == "left" else "left"
    if value == "random":
        return "left" if bool(rng.integers(0, 2)) else "right"
    raise ValueError(f"Unsupported initial server: {value}")


def _resolve_next_server(
    *,
    winner: object,
    current_server: Side,
    train_side: Side,
    random_server_each_rally: bool,
    rng: np.random.Generator,
    fixed_server_each_rally: bool = False,
) -> Side:
    if fixed_server_each_rally:
        return current_server
    if random_server_each_rally:
        return _resolve_initial_server("random", train_side, rng)
    if winner == "left":
        return "left"
    if winner == "right":
        return "right"
    return current_server


def _build_sim_config(args: argparse.Namespace) -> SimulationConfig:
    horizontal_drag = args.horizontal_drag_coefficient
    vertical_drag = args.vertical_drag_coefficient
    return SimulationConfig(
        court=CourtConfig(mode=args.court_mode),
        player=PlayerConfig(
            v_max=args.player_speed,
            r_reach=args.racket_length,
            z_max=args.max_hitting_height,
            acceleration=args.player_acceleration,
            deceleration=args.player_deceleration,
            movement_model=args.movement_model,
        ),
        action=ActionConfig(
            trajectory_mode=args.trajectory_mode,
            drag_coefficient=args.drag_coefficient,
            horizontal_drag_coefficient=horizontal_drag,
            vertical_drag_coefficient=vertical_drag,
            vy_min_forward=args.shuttle_speed_min,
            vy_max_forward=args.shuttle_speed_max,
            intercept_count=args.intercept_count,
            reaction_miss_fast_threshold=args.reaction_miss_fast_threshold,
            reaction_miss_fast_probability=args.reaction_miss_fast_probability,
            reaction_miss_secondary_threshold=args.reaction_miss_secondary_threshold,
            reaction_miss_secondary_probability=args.reaction_miss_secondary_probability,
            reaction_miss_zero_threshold=args.reaction_miss_zero_threshold,
        )
    )


def _build_env(args: argparse.Namespace, *, model: PPO) -> BadmintonRLEnv:
    sim_config = _build_sim_config(args)
    tactic_runtime_config = TacticRuntimeConfig(
        regenerate_lookup_table=args.regenerate_lookup_table,
        lookup_dir=args.lookup_table_dir,
    )
    discrete_action_config = DiscreteActionConfig(
        phi_bins=args.phi_bins,
        theta_bins=args.theta_bins,
        speed_bins=args.speed_bins,
        x_rec_bins=args.x_rec_bins,
        y_rec_bins=args.y_rec_bins,
    )
    if args.opponent == "mirror-self":
        opponent = LiveModelOpponent(
            sim_config=sim_config,
            discrete_action_config=discrete_action_config,
            model=model,
            policy_type=args.policy_type,
            tactic_runtime_config=tactic_runtime_config,
            deterministic=args.deterministic,
            label_name="mirror_self",
        )
    else:
        if args.opponent == "newest-checkpoint":
            checkpoint_pool = CheckpointPool(
                checkpoint_dir=args.checkpoint_pool_dir or (args.output_dir / "checkpoint_pool"),
                base_checkpoint_path=args.checkpoint_pool_base,
                pool_size=args.checkpoint_pool_size,
                sampling_mode="newest",
                seed=args.seed + 1,
            )
            newest_checkpoint = checkpoint_pool.newest_path()
            if newest_checkpoint is None:
                raise RuntimeError("No checkpoint available for --opponent newest-checkpoint.")
            opponent = FixedCheckpointOpponent(
                pool=checkpoint_pool,
                checkpoint_path=newest_checkpoint,
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                policy_type=args.policy_type,
                tactic_runtime_config=tactic_runtime_config,
                deterministic=args.deterministic,
            )
        else:
            opponent = make_opponent(args.opponent, seed=args.seed + 1)
    return build_selfplay_env(
        train_side=args.train_side,
        mirror_train_side=args.mirror_sides,
        mirror_match_fraction=args.mirror_match_fraction,
        initial_server=args.initial_server,
        random_service_x=args.random_service_x,
        sim_config=sim_config,
        train_reaction_time=args.reaction_time,
        opponent_reaction_time=args.reaction_time,
        policy_type=args.policy_type,
        tactic_runtime_config=tactic_runtime_config,
        seed=args.seed,
        discrete_action_config=discrete_action_config,
        opponent=opponent,
        include_records_in_info=True,
        recovery_counterfactual_other_sample_count=0,
        recovery_counterfactual_expected_response_target=False,
    )


def rollout_rally(
    env: BadmintonRLEnv,
    model: PPO,
    *,
    server: Side,
    seed: int,
    deterministic: bool,
) -> tuple[dict[str, object], list[object]]:
    observation, info = env.reset(seed=seed, options={"server": server})
    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action = choose_model_action(
            model,
            observation,
            env.current_decision_context(),
            deterministic=deterministic,
        )
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

    metrics = info["badminton_metrics"]
    return {
        "reward": total_reward,
        "winner": metrics["winner"],
        "rally_won": metrics["rally_won"],
        "rally_length": metrics["rally_length"],
        "invalid_action_rate": metrics["invalid_action_rate"],
        "terminal_reason": metrics["terminal_reason"],
        "server": info["server"],
        "service_x_side": info.get("service_x_side"),
        "opponent_label": info.get("opponent_label"),
        "truncated": truncated,
    }, list(env.records)


def build_match_trace_from_rallies(
    traces: list[RallyTrace],
    *,
    final_score_left: int,
    final_score_right: int,
) -> MatchTrace:
    total_playback_time = sum(trace.total_playback_time + trace.pause_duration for trace in traces)
    winner: Side = "left" if final_score_left >= final_score_right else "right"
    return MatchTrace(
        rallies=traces,
        target_score=max(final_score_left, final_score_right),
        score_left=final_score_left,
        score_right=final_score_right,
        winner=winner,
        total_playback_time=total_playback_time,
    )


def ensure_mp4(video_result: VideoExportResult, fps: int) -> Path | None:
    if video_result.mp4_path is not None and video_result.mp4_path.exists():
        return video_result.mp4_path

    frame_pattern = video_result.frame_paths[0].parent / "frame_%05d.png"
    output_path = video_result.trace_path.parent / "match.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return output_path if output_path.exists() else None


def main() -> None:
    args = parse_args()
    if args.start_rally < 0:
        raise ValueError("--start-rally must be zero or greater")
    if args.target_score is not None and args.target_score <= 0:
        raise ValueError("--target-score must be positive when provided")
    if args.target_score is None and args.end_rally < args.start_rally:
        raise ValueError("--end-rally must be greater than or equal to --start-rally")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.rally_pause < 0.0:
        raise ValueError("--rally-pause must be zero or greater")

    ensure_directory(args.output_dir)
    config = _build_sim_config(args)
    model = PPO.load(args.model_path)
    env = _build_env(args, model=model)

    rng = np.random.default_rng(args.seed)
    initial_server_mode = "random" if args.random_server_each_rally else args.initial_server
    initial_server = _resolve_initial_server(initial_server_mode, args.train_side, rng)
    current_server = initial_server
    score_left = 0
    score_right = 0
    selected_traces: list[RallyTrace] = []
    rally_summaries: list[dict[str, object]] = []

    max_rallies = args.end_rally + 1 if args.target_score is None else max(args.end_rally + 1, 10_000)
    for rally_index in range(max_rallies):
        result, records = rollout_rally(
            env,
            model,
            server=current_server,
            seed=args.seed + rally_index,
            deterministic=args.deterministic,
        )
        trace = build_rally_trace(records, config)
        winner = result["winner"]
        if winner == "left":
            next_score_left = score_left + 1
            next_score_right = score_right
        elif winner == "right":
            next_score_left = score_left
            next_score_right = score_right + 1
        else:
            next_score_left = score_left
            next_score_right = score_right
        next_server = _resolve_next_server(
            winner=winner,
            current_server=current_server,
            train_side=args.train_side,
            random_server_each_rally=args.random_server_each_rally,
            fixed_server_each_rally=args.fixed_server_each_rally,
            rng=rng,
        )

        if args.start_rally <= rally_index <= args.end_rally:
            selected_traces.append(
                RallyTrace(
                    stages=trace.stages,
                    rally_done=trace.rally_done,
                    winner=trace.winner,
                    total_playback_time=trace.total_playback_time,
                    rally_number=rally_index,
                    server=current_server,
                    score_before_left=score_left,
                    score_before_right=score_right,
                    score_after_left=next_score_left,
                    score_after_right=next_score_right,
                    pause_duration=args.rally_pause,
                    match_winner=None,
                )
            )

        rally_summaries.append(
            {
                "rally_index": rally_index,
                "server": current_server,
                "winner": winner,
                "score_before_left": score_left,
                "score_before_right": score_right,
                "score_after_left": next_score_left,
                "score_after_right": next_score_right,
                "rally_length": result["rally_length"],
                "invalid_action_rate": result["invalid_action_rate"],
                "terminal_reason": result["terminal_reason"],
                "service_x_side": result["service_x_side"],
                "illegal_shot": result["terminal_reason"] == "invalid_serve_target",
                "opponent_label": result["opponent_label"],
                "truncated": result["truncated"],
            }
        )

        score_left = next_score_left
        score_right = next_score_right
        current_server = next_server

        if args.target_score is not None and max(score_left, score_right) >= args.target_score:
            args.end_rally = rally_index
            break

    if args.target_score is not None and max(score_left, score_right) < args.target_score:
        raise RuntimeError(
            f"Target score {args.target_score} was not reached within {max_rallies} rallies."
        )

    if selected_traces:
        final_winner: Side = "left" if score_left >= score_right else "right"
        last_trace = selected_traces[-1]
        selected_traces[-1] = RallyTrace(
            stages=last_trace.stages,
            rally_done=last_trace.rally_done,
            winner=last_trace.winner,
            total_playback_time=last_trace.total_playback_time,
            rally_number=last_trace.rally_number,
            server=last_trace.server,
            score_before_left=last_trace.score_before_left,
            score_before_right=last_trace.score_before_right,
            score_after_left=last_trace.score_after_left,
            score_after_right=last_trace.score_after_right,
            pause_duration=last_trace.pause_duration,
            match_winner=final_winner,
        )

    match_trace = build_match_trace_from_rallies(
        selected_traces,
        final_score_left=score_left,
        final_score_right=score_right,
    )
    export_result = export_match_video(
        match_trace,
        config,
        args.output_dir,
        fps=args.fps,
        write_mp4=False,
    )
    mp4_path = ensure_mp4(export_result, args.fps)

    payload = {
        "model_path": str(args.model_path),
        "train_side": args.train_side,
        "opponent": args.opponent,
        "initial_server": args.initial_server,
        "resolved_initial_server": initial_server,
        "random_server_each_rally": args.random_server_each_rally,
        "fixed_server_each_rally": args.fixed_server_each_rally,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "policy_type": args.policy_type,
        "start_rally": args.start_rally,
        "end_rally": args.end_rally,
        "target_score": args.target_score,
        "selected_rally_count": len(selected_traces),
        "mirror_match_fraction": args.mirror_match_fraction,
        "random_service_x": args.random_service_x,
        "reaction_time": args.reaction_time,
        "player_speed": args.player_speed,
        "racket_length": args.racket_length,
        "max_hitting_height": args.max_hitting_height,
        "score_left": score_left,
        "score_right": score_right,
        "reaction_miss_fast_threshold": args.reaction_miss_fast_threshold,
        "reaction_miss_fast_probability": args.reaction_miss_fast_probability,
        "reaction_miss_secondary_threshold": args.reaction_miss_secondary_threshold,
        "reaction_miss_secondary_probability": args.reaction_miss_secondary_probability,
        "reaction_miss_zero_threshold": args.reaction_miss_zero_threshold,
        "gif_path": str(export_result.gif_path),
        "mp4_path": None if mp4_path is None else str(mp4_path),
        "trace_path": str(export_result.trace_path),
        "rallies": rally_summaries,
    }
    (args.output_dir / "rally_sequence_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"model: {args.model_path}")
    print(f"selected rallies: {args.start_rally}-{args.end_rally}")
    print(f"score after {args.end_rally + 1} rallies: {score_left}-{score_right}")
    print(f"gif: {export_result.gif_path}")
    print(f"mp4: {mp4_path if mp4_path is not None else 'unavailable'}")
    print(f"summary: {args.output_dir / 'rally_sequence_summary.json'}")


if __name__ == "__main__":
    main()
