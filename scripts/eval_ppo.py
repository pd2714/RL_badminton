from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionConfig
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.eval_diversity import DiversityEvalConfig, run_diversity_evaluation
from badminton1d.evaluation import BaselineSelector, ModelSelector, evaluate_selector
from badminton1d.match import MatchResult, MatchScore, RallyResult
from badminton1d.obs import ObservationEncoder
from badminton1d.opponents import make_baseline_policy
from badminton1d.playback import build_match_trace, build_rally_trace
from badminton1d.reset_sampling import ResetSamplingConfig
from badminton1d.rl_env import BadmintonRLEnv, RLEnvConfig, RewardConfig
from badminton1d.shot_generators import TacticRuntimeConfig
from badminton1d.selfplay import CheckpointPool, FixedCheckpointOpponent, LiveModelOpponent, build_selfplay_env
from badminton1d.utils import ensure_directory
from badminton1d.video import export_match_video, export_rally_video

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO badminton checkpoint and export example videos.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl/eval"))
    parser.add_argument("--train-side", choices=("left", "right"), default="left")
    parser.add_argument("--opponent", choices=("safe", "random", "greedy"), default="safe")
    parser.add_argument("--initial-server", choices=("left", "right", "train", "opponent", "random"), default="random")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--save-rally-videos", type=int, default=3)
    parser.add_argument("--save-match-video", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rally-pause", type=float, default=0.6)
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
    parser.add_argument("--checkpoint-pool-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-pool-base", type=Path, default=None)
    parser.add_argument("--checkpoint-pool-size", type=int, default=6)
    parser.add_argument("--checkpoint-sampling-mode", choices=("uniform", "random", "newest"), default="newest")
    parser.add_argument("--diversity-eval", action="store_true")
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


def build_sim_config(args: argparse.Namespace) -> SimulationConfig:
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


def build_env(args: argparse.Namespace, *, include_records_in_info: bool = True) -> BadmintonRLEnv:
    tactic_runtime = TacticRuntimeConfig(
        regenerate_lookup_table=args.regenerate_lookup_table,
        lookup_dir=args.lookup_table_dir,
    )
    return BadmintonRLEnv(
        config=build_sim_config(args),
        rl_config=RLEnvConfig(
            train_side=args.train_side,
            opponent_type=args.opponent,
            initial_server=args.initial_server,
            random_service_x=args.random_service_x,
            mirror_train_side=args.mirror_sides,
            mirror_match_fraction=args.mirror_match_fraction,
            train_reaction_time=args.reaction_time,
            opponent_reaction_time=args.reaction_time,
            include_records_in_info=include_records_in_info,
            policy_type=args.policy_type,
            tactic_runtime=tactic_runtime,
        ),
        discrete_action_config=DiscreteActionConfig(
            phi_bins=args.phi_bins,
            theta_bins=args.theta_bins,
            speed_bins=args.speed_bins,
            x_rec_bins=args.x_rec_bins,
            y_rec_bins=args.y_rec_bins,
        ),
        seed=args.seed,
    )


def save_videos(
    results: list[dict[str, object]],
    config: SimulationConfig,
    output_dir: Path,
    *,
    fps: int,
    rally_pause: float,
    save_rally_videos: int,
    save_match_video: bool,
) -> dict[str, str]:
    ensure_directory(output_dir)
    saved_paths: dict[str, str] = {}

    for index, result in enumerate(results[:save_rally_videos]):
        records = result["records"]
        assert isinstance(records, list)
        if not records:
            continue
        rally_dir = output_dir / f"rally_{index:03d}"
        rally_trace = build_rally_trace(records, config)
        export_result = export_rally_video(rally_trace, config, rally_dir, fps=fps)
        saved_paths[f"rally_{index:03d}"] = str(export_result.gif_path)

    if save_match_video and results:
        best_result = max(results, key=lambda item: (float(item["rally_won"]), -float(item["invalid_action_rate"])))
        records = best_result["records"]
        assert isinstance(records, list)
        if records:
            winner = records[-1].next_state.winner
            if winner is None:
                raise RuntimeError("Cannot build match video without a rally winner.")
            score_before = MatchScore()
            score_after = score_before.award_point(winner)
            rally_result = RallyResult(
                rally_number=1,
                server=str(best_result["server"]),
                score_before=score_before,
                score_after=score_after,
                initial_state=records[0].state_before,
                records=records,
                winner=winner,
                final_state=records[-1].next_state,
            )
            match_result = MatchResult(
                rallies=[rally_result],
                target_score=1,
                initial_server=str(best_result["server"]),
                final_score=score_after,
                winner=winner,
            )
            match_trace = build_match_trace(match_result, config, rally_pause=rally_pause)
            match_dir = output_dir / "match_1pt"
            export_result = export_match_video(match_trace, config, match_dir, fps=fps)
            saved_paths["match_1pt"] = str(export_result.gif_path)
    return saved_paths


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    ensure_directory(args.output_dir)
    config = build_sim_config(args)
    env = build_env(args, include_records_in_info=True)
    model = PPO.load(args.model_path)

    model_selector = ModelSelector(model=model, deterministic=args.deterministic)
    baseline_random = BaselineSelector(make_baseline_policy("random", env.action_mapper, seed=args.seed + 1))
    baseline_safe = BaselineSelector(make_baseline_policy("safe", env.action_mapper, seed=args.seed + 2))
    baseline_greedy = BaselineSelector(make_baseline_policy("greedy", env.action_mapper, seed=args.seed + 3))

    summaries: list[dict[str, object]] = []

    model_summary, model_results = evaluate_selector("ppo_model", model_selector, env, args.episodes, args.seed)
    summaries.append(model_summary)
    random_summary, _ = evaluate_selector("random_baseline", baseline_random, env, args.episodes, args.seed + 1_000)
    summaries.append(random_summary)
    safe_summary, _ = evaluate_selector("safe_baseline", baseline_safe, env, args.episodes, args.seed + 2_000)
    summaries.append(safe_summary)
    greedy_summary, _ = evaluate_selector("greedy_baseline", baseline_greedy, env, args.episodes, args.seed + 3_000)
    summaries.append(greedy_summary)

    mirror_self_env = build_selfplay_env(
        train_side=args.train_side,
        mirror_train_side=args.mirror_sides,
        mirror_match_fraction=args.mirror_match_fraction,
        initial_server=args.initial_server,
        random_service_x=args.random_service_x,
        sim_config=config,
        train_reaction_time=args.reaction_time,
        opponent_reaction_time=args.reaction_time,
        policy_type=args.policy_type,
        tactic_runtime_config=TacticRuntimeConfig(
            regenerate_lookup_table=args.regenerate_lookup_table,
            lookup_dir=args.lookup_table_dir,
        ),
        seed=args.seed + 3_500,
        discrete_action_config=DiscreteActionConfig(
            phi_bins=args.phi_bins,
            theta_bins=args.theta_bins,
            speed_bins=args.speed_bins,
            x_rec_bins=args.x_rec_bins,
            y_rec_bins=args.y_rec_bins,
        ),
        opponent=LiveModelOpponent(
            sim_config=config,
            discrete_action_config=DiscreteActionConfig(
                phi_bins=args.phi_bins,
                theta_bins=args.theta_bins,
                speed_bins=args.speed_bins,
                x_rec_bins=args.x_rec_bins,
                y_rec_bins=args.y_rec_bins,
            ),
            policy_type=args.policy_type,
            tactic_runtime_config=TacticRuntimeConfig(
                regenerate_lookup_table=args.regenerate_lookup_table,
                lookup_dir=args.lookup_table_dir,
            ),
            model=model,
            deterministic=args.deterministic,
            label_name="mirror_self",
        ),
        include_records_in_info=False,
    )
    mirror_self_summary, _ = evaluate_selector("current_vs_mirror_self", model_selector, mirror_self_env, args.episodes, args.seed + 3_500)
    summaries.append(mirror_self_summary)

    checkpoint_pool = None
    if args.checkpoint_pool_dir is not None or args.checkpoint_pool_base is not None:
        discrete_action_config = DiscreteActionConfig(
            phi_bins=args.phi_bins,
            theta_bins=args.theta_bins,
            speed_bins=args.speed_bins,
            x_rec_bins=args.x_rec_bins,
            y_rec_bins=args.y_rec_bins,
        )
        checkpoint_pool = CheckpointPool(
            checkpoint_dir=args.checkpoint_pool_dir or (args.output_dir / "checkpoint_pool"),
            base_checkpoint_path=args.checkpoint_pool_base,
            pool_size=args.checkpoint_pool_size,
            sampling_mode=args.checkpoint_sampling_mode,
            seed=args.seed + 4_001,
        )
        newest_checkpoint = checkpoint_pool.newest_path()
        if newest_checkpoint is not None:
            checkpoint_env = build_selfplay_env(
                train_side=args.train_side,
                mirror_train_side=args.mirror_sides,
                mirror_match_fraction=args.mirror_match_fraction,
                initial_server=args.initial_server,
                random_service_x=args.random_service_x,
                sim_config=config,
                policy_type=args.policy_type,
                tactic_runtime_config=TacticRuntimeConfig(
                    regenerate_lookup_table=args.regenerate_lookup_table,
                    lookup_dir=args.lookup_table_dir,
                ),
                seed=args.seed + 4_000,
                discrete_action_config=discrete_action_config,
                opponent=FixedCheckpointOpponent(
                    pool=checkpoint_pool,
                    checkpoint_path=newest_checkpoint,
                    sim_config=config,
                    discrete_action_config=discrete_action_config,
                    policy_type=args.policy_type,
                    tactic_runtime_config=TacticRuntimeConfig(
                        regenerate_lookup_table=args.regenerate_lookup_table,
                        lookup_dir=args.lookup_table_dir,
                    ),
                ),
                include_records_in_info=False,
            )
            checkpoint_summary, _ = evaluate_selector("current_vs_newest_checkpoint", model_selector, checkpoint_env, args.episodes, args.seed + 4_000)
            summaries.append(checkpoint_summary)

    diversity_report: dict[str, object] | None = None
    if args.diversity_eval and checkpoint_pool is not None:
        diversity_report = run_diversity_evaluation(
            model=model,
            eval_config=DiversityEvalConfig(
                train_side=args.train_side,
                initial_server=args.initial_server,
                mirror_train_side=args.mirror_sides,
                mirror_match_fraction=args.mirror_match_fraction,
                sim_config=config,
                reward_config=RewardConfig(),
                reset_sampling_config=ResetSamplingConfig(),
                train_reaction_time=args.reaction_time,
                opponent_reaction_time=args.reaction_time,
                episodes=args.episodes,
                seed=args.seed + 8_000,
                deterministic=args.deterministic,
            ),
            discrete_action_config=DiscreteActionConfig(
                phi_bins=args.phi_bins,
                theta_bins=args.theta_bins,
                speed_bins=args.speed_bins,
                x_rec_bins=args.x_rec_bins,
                y_rec_bins=args.y_rec_bins,
            ),
            checkpoint_pool=checkpoint_pool,
        )

    saved_paths = save_videos(
        model_results,
        config,
        args.output_dir / "videos",
        fps=args.fps,
        rally_pause=args.rally_pause,
        save_rally_videos=args.save_rally_videos,
        save_match_video=args.save_match_video,
    )

    payload = {
        "model_path": str(args.model_path),
        "train_side": args.train_side,
        "opponent": args.opponent,
        "initial_server": args.initial_server,
        "mirror_sides": args.mirror_sides,
        "mirror_match_fraction": args.mirror_match_fraction,
        "random_service_x": args.random_service_x,
        "episodes": args.episodes,
        "reaction_time": args.reaction_time,
        "court_mode": args.court_mode,
        "player_speed": args.player_speed,
        "racket_length": args.racket_length,
        "max_hitting_height": args.max_hitting_height,
        "movement_model": args.movement_model,
        "player_acceleration": args.player_acceleration,
        "player_deceleration": args.player_deceleration,
        "trajectory_mode": args.trajectory_mode,
        "drag_coefficient": args.drag_coefficient,
        "horizontal_drag_coefficient": args.horizontal_drag_coefficient,
        "vertical_drag_coefficient": args.vertical_drag_coefficient,
        "shuttle_speed_min": args.shuttle_speed_min,
        "shuttle_speed_max": args.shuttle_speed_max,
        "intercept_count": args.intercept_count,
        "reaction_miss_fast_threshold": args.reaction_miss_fast_threshold,
        "reaction_miss_fast_probability": args.reaction_miss_fast_probability,
        "reaction_miss_secondary_threshold": args.reaction_miss_secondary_threshold,
        "reaction_miss_secondary_probability": args.reaction_miss_secondary_probability,
        "reaction_miss_zero_threshold": args.reaction_miss_zero_threshold,
        "summaries": summaries,
        "diversity_report": diversity_report,
        "saved_paths": saved_paths,
        "observation_features": ObservationEncoder(config).feature_names(),
    }
    report_path = args.output_dir / "evaluation_summary.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for summary in summaries:
        print(
            f"{summary['name']}: "
            f"win_rate={summary['win_rate']:.3f} "
            f"avg_reward={summary['avg_reward']:.3f} "
            f"avg_rally_length={summary['avg_rally_length']:.3f} "
            f"avg_invalid_action_rate={summary['avg_invalid_action_rate']:.3f}"
        )
    print(f"report: {report_path}")
    if diversity_report is not None:
        print(f"diversity narrowness flag: {diversity_report['narrow_opponent_dependency']}")
    for label, path in saved_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
