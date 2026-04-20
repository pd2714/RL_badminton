from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionConfig
from badminton1d.callbacks import EntropyScheduleCallback, RallyDiagnosticsCallback
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.curricula import (
    available_training_curricula,
    build_training_curriculum,
)
from badminton1d.policy import MaskedBadmintonPolicy
from badminton1d.reset_sampling import ResetSamplingConfig
from badminton1d.reward_shaping import LoopPenaltyConfig, PressureRewardConfig
from badminton1d.rl_env import RLEnvConfig, RewardConfig
from badminton1d.selfplay import (
    CheckpointPool,
    FixedCheckpointOpponent,
    FrozenCheckpointOpponent,
    LiveModelOpponent,
    MixedCheckpointOpponent,
    SelfPlayCheckpointCallback,
    SelfPlayEvalCallback,
    SelfPlayProgressVideoCallback,
    build_selfplay_env,
)
from badminton1d.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue PPO training with self-play or a named curriculum opponent."
    )
    parser.add_argument("--base-checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl/selfplay_run"))
    parser.add_argument("--resume-step-offset", type=int, default=0)
    parser.add_argument("--train-side", choices=("left", "right"), default="left")
    parser.add_argument("--mirror-sides", action="store_true")
    parser.add_argument("--mirror-match-fraction", type=float, default=0.25)
    parser.add_argument("--initial-server", choices=("left", "right", "train", "opponent", "random"), default="random")
    parser.add_argument("--selfplay-total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--ent-coef-final", type=float, default=0.002)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--opponent-sampling-mode", choices=("uniform", "random", "recency", "newest"), default="recency")
    parser.add_argument("--checkpoint-recency-power", type=float, default=3.0)
    parser.add_argument("--recent-opponent-weight", type=float, default=0.9)
    parser.add_argument("--older-opponent-weight", type=float, default=0.1)
    parser.add_argument("--heuristic-opponent-prob", type=float, default=0.05)
    parser.add_argument(
        "--curriculum",
        choices=("none", *available_training_curricula()),
        default="none",
    )
    parser.add_argument("--curriculum-opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--random-start-prob", type=float, default=0.0)
    parser.add_argument("--midrally-start-prob", type=float, default=0.0)
    parser.add_argument("--opponent-serve-start-prob", type=float, default=0.0)
    parser.add_argument("--reaction-time", type=float, default=0.3)
    parser.add_argument("--court-mode", choices=("1d", "2d"), default="2d")
    parser.add_argument("--loop-penalty", type=float, default=0.03)
    parser.add_argument("--loop-window", type=int, default=4)
    parser.add_argument("--defensive-return-reward", type=float, default=0.0)
    parser.add_argument("--serve-return-reward", type=float, default=0.0)
    parser.add_argument("--max-rally-stages", type=int, default=120)
    parser.add_argument("--max-rally-penalty", type=float, default=1.0)
    parser.add_argument("--stage-penalty", type=float, default=0.0)
    parser.add_argument("--stall-penalty", type=float, default=0.0)
    parser.add_argument("--stall-penalty-start", type=int, default=24)
    parser.add_argument("--pressure-reward-weight", type=float, default=0.05)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.add_argument("--anchor-eval-interval", type=int, default=0)
    parser.add_argument(
        "--eval-matchups",
        choices=("newest-only", "newest-and-mirror", "full"),
        default="newest-only",
    )
    parser.add_argument("--progress-video-freq", type=int, default=0)
    parser.add_argument("--progress-video-fps", type=int, default=18)
    parser.add_argument("--progress-video-dpi", type=int, default=90)
    parser.add_argument("--progress-video-stage-pause", type=float, default=0.15)
    parser.add_argument("--progress-video-rally-pause", type=float, default=0.9)
    parser.add_argument("--progress-video-save-sample-media", action="store_true")
    parser.add_argument("--progress-video-write-combined-frames", action="store_true")
    parser.add_argument("--progress-video-write-combined-gif", action="store_true")
    parser.add_argument("--progress-video-deterministic", action="store_true")
    parser.add_argument(
        "--progress-video-matchups",
        choices=("newest-only", "newest-and-mirror"),
        default="newest-and-mirror",
    )
    parser.add_argument("--no-progress-video-initial-sample", action="store_true")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("outputs/rl/tensorboard"))
    parser.add_argument("--player-speed", type=float, default=2.6)
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="drag_square")
    parser.add_argument("--drag-coefficient", type=float, default=0.2)
    parser.add_argument("--horizontal-drag-coefficient", type=float, default=0.2)
    parser.add_argument("--vertical-drag-coefficient", type=float, default=0.16)
    parser.add_argument("--shuttle-speed-min", type=float, default=0.1)
    parser.add_argument("--shuttle-speed-max", type=float, default=100.0)
    parser.add_argument("--intercept-count", type=int, default=50)
    parser.add_argument("--vx-bins", type=int, default=7)
    parser.add_argument("--vy-bins", type=int, default=11)
    parser.add_argument("--vz-bins", type=int, default=7)
    parser.add_argument("--x-rec-bins", type=int, default=3)
    parser.add_argument("--y-rec-bins", type=int, default=5)
    return parser.parse_args()


def build_sim_config(args: argparse.Namespace) -> SimulationConfig:
    horizontal_drag = args.horizontal_drag_coefficient
    vertical_drag = args.vertical_drag_coefficient
    return SimulationConfig(
        court=CourtConfig(mode=args.court_mode),
        player=PlayerConfig(v_max=args.player_speed),
        action=ActionConfig(
            trajectory_mode=args.trajectory_mode,
            drag_coefficient=args.drag_coefficient,
            horizontal_drag_coefficient=horizontal_drag,
            vertical_drag_coefficient=vertical_drag,
            vy_min_forward=args.shuttle_speed_min,
            vy_max_forward=args.shuttle_speed_max,
            intercept_count=args.intercept_count,
        )
    )


def build_fixed_checkpoint_opponent(
    *,
    checkpoint_path: Path,
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    seed: int,
    deterministic: bool = False,
    hitter_deterministic: bool | None = None,
    receiver_deterministic: bool | None = None,
) -> FixedCheckpointOpponent:
    return FixedCheckpointOpponent(
        pool=CheckpointPool(
            checkpoint_dir=checkpoint_path.parent,
            base_checkpoint_path=checkpoint_path,
            pool_size=1,
            sampling_mode="newest",
            seed=seed,
        ),
        checkpoint_path=checkpoint_path,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        deterministic=deterministic,
        hitter_deterministic=hitter_deterministic,
        receiver_deterministic=receiver_deterministic,
    )


def main() -> None:
    args = parse_args()
    if not args.base_checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint_path}")
    if args.selfplay_total_timesteps <= 0:
        raise ValueError("--selfplay-total-timesteps must be positive")
    if args.resume_step_offset < 0:
        raise ValueError("--resume-step-offset must be zero or greater")
    if args.anchor_eval_interval < 0:
        raise ValueError("--anchor-eval-interval must be zero or greater")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if not 0.0 <= args.mirror_match_fraction <= 1.0:
        raise ValueError("--mirror-match-fraction must be in [0, 1]")
    if args.progress_video_fps <= 0:
        raise ValueError("--progress-video-fps must be positive")
    if args.progress_video_stage_pause < 0.0:
        raise ValueError("--progress-video-stage-pause must be zero or greater")
    if args.progress_video_rally_pause < 0.0:
        raise ValueError("--progress-video-rally-pause must be zero or greater")
    if args.stage_penalty < 0.0:
        raise ValueError("--stage-penalty must be zero or greater")
    if args.stall_penalty < 0.0:
        raise ValueError("--stall-penalty must be zero or greater")
    if args.stall_penalty_start < 0:
        raise ValueError("--stall-penalty-start must be zero or greater")

    run_dir = args.output_dir
    checkpoint_dir = run_dir / "checkpoint_pool"
    eval_dir = run_dir / "selfplay_eval"
    anchor_checkpoint_dir = run_dir / "anchor_checkpoints"
    progress_video_dir = run_dir / "training_progress"
    latest_model_path = run_dir / "latest_model.zip"
    best_model_path = run_dir / "best_model.zip"
    final_model_path = run_dir / "final_model.zip"
    ensure_directory(run_dir)
    ensure_directory(checkpoint_dir)
    ensure_directory(eval_dir)
    ensure_directory(anchor_checkpoint_dir)
    ensure_directory(progress_video_dir)
    ensure_directory(args.tensorboard_log)

    discrete_action_config = DiscreteActionConfig(
        v_x_bins=args.vx_bins,
        v_y_bins=args.vy_bins,
        v_z_bins=args.vz_bins,
        x_rec_bins=args.x_rec_bins,
        y_rec_bins=args.y_rec_bins,
    )
    sim_config = build_sim_config(args)
    train_pool = CheckpointPool(
        checkpoint_dir=checkpoint_dir,
        base_checkpoint_path=args.base_checkpoint_path,
        pool_size=args.pool_size,
        sampling_mode=args.opponent_sampling_mode,
        recency_power=args.checkpoint_recency_power,
        recent_fraction=0.5,
        seed=args.seed + 1,
    )
    reward_config = RewardConfig(
        defensive_return_reward=args.defensive_return_reward,
        serve_return_reward=args.serve_return_reward,
        max_rally_penalty=args.max_rally_penalty,
        stage_penalty=args.stage_penalty,
        stall_penalty=args.stall_penalty,
        stall_penalty_start=args.stall_penalty_start,
        loop_penalty=LoopPenaltyConfig(penalty=args.loop_penalty, window=args.loop_window),
        pressure_reward=PressureRewardConfig(weight=args.pressure_reward_weight),
    )
    curriculum = None if args.curriculum == "none" else build_training_curriculum(
        args.curriculum,
        opponent_checkpoint_path=args.curriculum_opponent_checkpoint,
    )
    effective_initial_server = args.initial_server
    progress_primary_label = "current_vs_newest_checkpoint"
    progress_primary_dir = progress_video_dir / "current_vs_newest_checkpoint"
    eval_pool = train_pool
    if curriculum is None:
        reset_sampling_config = ResetSamplingConfig(
            random_start_prob=args.random_start_prob,
            midrally_start_prob=args.midrally_start_prob,
            opponent_serve_start_prob=args.opponent_serve_start_prob,
        )
    else:
        if not curriculum.opponent_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Curriculum opponent checkpoint not found: {curriculum.opponent_checkpoint_path}"
            )
        effective_initial_server = curriculum.initial_server
        progress_primary_label = "current_vs_curriculum_opponent"
        progress_primary_dir = progress_video_dir / "current_vs_curriculum_opponent"
        reset_sampling_config = ResetSamplingConfig(
            random_start_prob=1.0,
            midrally_start_prob=0.0,
            opponent_serve_start_prob=1.0,
            defensive_backcourt_curriculum=curriculum.sampler_config,
        )
        eval_pool = CheckpointPool(
            checkpoint_dir=run_dir / "_curriculum_eval_pool",
            base_checkpoint_path=curriculum.opponent_checkpoint_path,
            pool_size=1,
            sampling_mode="newest",
            seed=args.seed + 9,
        )

    def make_training_opponent(include_records: bool = False):
        if curriculum is not None:
            return build_fixed_checkpoint_opponent(
                checkpoint_path=curriculum.opponent_checkpoint_path,
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                seed=args.seed + 2 + (1 if include_records else 0),
                deterministic=False,
                hitter_deterministic=curriculum.opponent_hitter_deterministic,
                receiver_deterministic=curriculum.opponent_receiver_deterministic,
            )
        return MixedCheckpointOpponent(
            checkpoint_pool=CheckpointPool(
                checkpoint_dir=checkpoint_dir,
                base_checkpoint_path=args.base_checkpoint_path,
                pool_size=args.pool_size,
                sampling_mode=args.opponent_sampling_mode,
                recency_power=args.checkpoint_recency_power,
                recent_fraction=0.5,
                seed=args.seed + 2 + (1 if include_records else 0),
            ),
            sim_config=sim_config,
            discrete_action_config=discrete_action_config,
            heuristic_opponent_prob=args.heuristic_opponent_prob,
            recent_weight=args.recent_opponent_weight,
            older_weight=args.older_opponent_weight,
            deterministic=False,
        )

    def make_selfplay_env(include_records: bool = False):
        return build_selfplay_env(
            train_side=args.train_side,
            mirror_train_side=args.mirror_sides,
            mirror_match_fraction=args.mirror_match_fraction,
            initial_server=effective_initial_server,
            sim_config=sim_config,
            train_reaction_time=args.reaction_time,
            opponent_reaction_time=args.reaction_time,
            max_stages_per_rally=args.max_rally_stages,
            reward_config=reward_config,
            reset_sampling_config=reset_sampling_config,
            seed=args.seed,
            discrete_action_config=discrete_action_config,
            opponent=make_training_opponent(include_records=include_records),
            include_records_in_info=include_records,
        )

    def monitored_factory() -> Monitor:
        return Monitor(make_selfplay_env(include_records=False))

    vec_env_cls = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    train_env = make_vec_env(
        monitored_factory,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
    )

    progress_newest_env = build_selfplay_env(
        train_side=args.train_side,
        mirror_train_side=args.mirror_sides,
        mirror_match_fraction=args.mirror_match_fraction,
        initial_server=effective_initial_server,
        sim_config=sim_config,
        train_reaction_time=args.reaction_time,
        opponent_reaction_time=args.reaction_time,
        max_stages_per_rally=args.max_rally_stages,
        reward_config=reward_config,
        reset_sampling_config=reset_sampling_config,
        seed=args.seed + 50_000,
        discrete_action_config=discrete_action_config,
        opponent=(
            build_fixed_checkpoint_opponent(
                checkpoint_path=curriculum.opponent_checkpoint_path,
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                seed=args.seed + 50_001,
                deterministic=args.progress_video_deterministic,
                hitter_deterministic=curriculum.opponent_hitter_deterministic,
                receiver_deterministic=curriculum.opponent_receiver_deterministic,
            )
            if curriculum is not None
            else FrozenCheckpointOpponent(
                pool=CheckpointPool(
                    checkpoint_dir=checkpoint_dir,
                    base_checkpoint_path=args.base_checkpoint_path,
                    pool_size=args.pool_size,
                    sampling_mode="newest",
                    recency_power=args.checkpoint_recency_power,
                    recent_fraction=0.5,
                    seed=args.seed + 50_001,
                ),
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                deterministic=args.progress_video_deterministic,
            )
        ),
        include_records_in_info=True,
    )
    progress_mirror_env = build_selfplay_env(
        train_side=args.train_side,
        mirror_train_side=args.mirror_sides,
        mirror_match_fraction=args.mirror_match_fraction,
        initial_server=effective_initial_server,
        sim_config=sim_config,
        train_reaction_time=args.reaction_time,
        opponent_reaction_time=args.reaction_time,
        max_stages_per_rally=args.max_rally_stages,
        reward_config=reward_config,
        reset_sampling_config=reset_sampling_config,
        seed=args.seed + 60_000,
        discrete_action_config=discrete_action_config,
        opponent=LiveModelOpponent(
            sim_config=sim_config,
            discrete_action_config=discrete_action_config,
            deterministic=args.progress_video_deterministic,
            label_name="mirror_self",
        ),
        include_records_in_info=True,
    )

    model = PPO(
        MaskedBadmintonPolicy,
        train_env,
        policy_kwargs={
            "sim_config": sim_config,
            "discrete_action_config": discrete_action_config,
        },
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
    )
    base_model = PPO.load(args.base_checkpoint_path)
    model.set_parameters(base_model.get_parameters(), exact_match=False)

    diagnostics_callback = RallyDiagnosticsCallback(run_dir)
    callbacks_list = [diagnostics_callback]
    if args.ent_coef_final is not None and args.ent_coef_final != args.ent_coef:
        callbacks_list.append(
            EntropyScheduleCallback(
                ent_coef_initial=args.ent_coef,
                ent_coef_final=args.ent_coef_final,
                total_timesteps=args.selfplay_total_timesteps,
            )
        )
    checkpoint_callback = SelfPlayCheckpointCallback(
        checkpoint_dir=checkpoint_dir,
        latest_model_path=latest_model_path,
        save_freq=args.save_interval,
        pool_size=args.pool_size,
        train_env=train_env,
        timestep_offset=args.resume_step_offset,
    )
    eval_callback = SelfPlayEvalCallback(
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_matchups=args.eval_matchups,
        eval_seed=args.seed + 30_000,
        output_dir=eval_dir,
        best_model_path=best_model_path,
        train_side=args.train_side,
        initial_server=effective_initial_server,
        mirror_train_side=args.mirror_sides,
        mirror_match_fraction=args.mirror_match_fraction,
        sim_config=sim_config,
        reward_config=reward_config,
        reset_sampling_config=reset_sampling_config,
        train_reaction_time=args.reaction_time,
        opponent_reaction_time=args.reaction_time,
        max_stages_per_rally=args.max_rally_stages,
        discrete_action_config=discrete_action_config,
        checkpoint_pool=eval_pool,
        base_checkpoint_path=args.base_checkpoint_path,
        anchor_eval_interval=args.anchor_eval_interval,
        anchor_checkpoint_dir=anchor_checkpoint_dir,
        deterministic=args.eval_deterministic,
        timestep_offset=args.resume_step_offset,
    )
    callbacks_list.extend([checkpoint_callback, eval_callback])
    if args.progress_video_freq > 0:
        callbacks_list.append(
            SelfPlayProgressVideoCallback(
                sample_env=progress_newest_env,
                output_dir=progress_primary_dir,
                record_freq=args.progress_video_freq,
                sample_seed=args.seed + 40_000,
                fps=args.progress_video_fps,
                stage_pause=args.progress_video_stage_pause,
                rally_pause=args.progress_video_rally_pause,
                record_initial_sample=not args.no_progress_video_initial_sample,
                matchup_name=progress_primary_label,
                save_sample_media=args.progress_video_save_sample_media,
                write_combined_frames=args.progress_video_write_combined_frames,
                write_combined_gif=args.progress_video_write_combined_gif,
                render_dpi=args.progress_video_dpi,
                deterministic=args.progress_video_deterministic,
                timestep_offset=args.resume_step_offset,
            )
        )
        if args.progress_video_matchups == "newest-and-mirror":
            callbacks_list.append(
                SelfPlayProgressVideoCallback(
                    sample_env=progress_mirror_env,
                    output_dir=progress_video_dir / "current_vs_mirror_self",
                    record_freq=args.progress_video_freq,
                    sample_seed=args.seed + 41_000,
                    fps=args.progress_video_fps,
                    stage_pause=args.progress_video_stage_pause,
                    rally_pause=args.progress_video_rally_pause,
                    record_initial_sample=not args.no_progress_video_initial_sample,
                    matchup_name="current_vs_mirror_self",
                    save_sample_media=args.progress_video_save_sample_media,
                    write_combined_frames=args.progress_video_write_combined_frames,
                    write_combined_gif=args.progress_video_write_combined_gif,
                    render_dpi=args.progress_video_dpi,
                    deterministic=args.progress_video_deterministic,
                    timestep_offset=args.resume_step_offset,
                )
            )
    callbacks = CallbackList(callbacks_list)

    model.learn(total_timesteps=args.selfplay_total_timesteps, callback=callbacks, progress_bar=False)
    model.save(final_model_path)
    model.save(latest_model_path)

    train_pool.refresh()
    summary = {
        "base_checkpoint_path": str(args.base_checkpoint_path),
        "resume_step_offset": args.resume_step_offset,
        "train_side": args.train_side,
        "mirror_sides": args.mirror_sides,
        "mirror_match_fraction": args.mirror_match_fraction,
        "initial_server": effective_initial_server,
        "requested_initial_server": args.initial_server,
        "selfplay_total_timesteps": args.selfplay_total_timesteps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "ent_coef": args.ent_coef,
        "ent_coef_final": args.ent_coef_final,
        "save_interval": args.save_interval,
        "pool_size": args.pool_size,
        "opponent_sampling_mode": args.opponent_sampling_mode,
        "checkpoint_recency_power": args.checkpoint_recency_power,
        "heuristic_opponent_prob": args.heuristic_opponent_prob,
        "curriculum": None if curriculum is None else curriculum.name,
        "curriculum_description": None if curriculum is None else curriculum.description,
        "curriculum_opponent_checkpoint": None if curriculum is None else str(curriculum.opponent_checkpoint_path),
        "curriculum_hitter_deterministic": None if curriculum is None else curriculum.opponent_hitter_deterministic,
        "curriculum_receiver_deterministic": None if curriculum is None else curriculum.opponent_receiver_deterministic,
        "recent_opponent_weight": args.recent_opponent_weight,
        "older_opponent_weight": args.older_opponent_weight,
        "random_start_prob": reset_sampling_config.random_start_prob,
        "midrally_start_prob": reset_sampling_config.midrally_start_prob,
        "opponent_serve_start_prob": reset_sampling_config.opponent_serve_start_prob,
        "defensive_backcourt_curriculum": (
            None
            if reset_sampling_config.defensive_backcourt_curriculum is None
            else {
                "name": reset_sampling_config.defensive_backcourt_curriculum.name,
                "stage_index": reset_sampling_config.defensive_backcourt_curriculum.stage_index,
                "phases": [
                    {
                        "name": phase.name,
                        "start_episode": phase.start_episode,
                        "attacker_depth_range": list(phase.attacker_depth_range),
                        "attacker_lateral_span": phase.attacker_lateral_span,
                        "defender_depth_range": list(phase.defender_depth_range),
                        "defender_lateral_span": phase.defender_lateral_span,
                        "hit_height_range": list(phase.hit_height_range),
                    }
                    for phase in reset_sampling_config.defensive_backcourt_curriculum.phases
                ],
            }
        ),
        "reaction_time": args.reaction_time,
        "court_mode": args.court_mode,
        "loop_penalty": args.loop_penalty,
        "loop_window": args.loop_window,
        "defensive_return_reward": args.defensive_return_reward,
        "serve_return_reward": args.serve_return_reward,
        "max_rally_stages": args.max_rally_stages,
        "max_rally_penalty": args.max_rally_penalty,
        "stage_penalty": args.stage_penalty,
        "stall_penalty": args.stall_penalty,
        "stall_penalty_start": args.stall_penalty_start,
        "pressure_reward_weight": args.pressure_reward_weight,
        "eval_episodes": args.eval_episodes,
        "eval_deterministic": args.eval_deterministic,
        "anchor_eval_interval": args.anchor_eval_interval,
        "anchor_checkpoint_dir": str(anchor_checkpoint_dir),
        "eval_matchups": args.eval_matchups,
        "final_model_path": str(final_model_path),
        "latest_model_path": str(latest_model_path),
        "best_model_path": str(best_model_path),
        "checkpoint_pool_dir": str(checkpoint_dir),
        "checkpoint_pool_entries": [str(path) for path in train_pool.checkpoints],
        "tensorboard_log": str(args.tensorboard_log),
        "progress_video_dir": str(progress_video_dir),
        "progress_video_freq": args.progress_video_freq,
        "progress_video_fps": args.progress_video_fps,
        "progress_video_dpi": args.progress_video_dpi,
        "progress_video_stage_pause": args.progress_video_stage_pause,
        "progress_video_rally_pause": args.progress_video_rally_pause,
        "progress_video_initial_sample": not args.no_progress_video_initial_sample,
        "progress_video_deterministic": args.progress_video_deterministic,
        "progress_video_matchups": args.progress_video_matchups,
        "progress_video_save_sample_media": args.progress_video_save_sample_media,
        "progress_video_write_combined_frames": args.progress_video_write_combined_frames,
        "progress_video_write_combined_gif": args.progress_video_write_combined_gif,
        "player_speed": args.player_speed,
        "trajectory_mode": args.trajectory_mode,
        "drag_coefficient": args.drag_coefficient,
        "horizontal_drag_coefficient": args.horizontal_drag_coefficient,
        "vertical_drag_coefficient": args.vertical_drag_coefficient,
        "shuttle_speed_min": args.shuttle_speed_min,
        "shuttle_speed_max": args.shuttle_speed_max,
        "intercept_count": args.intercept_count,
    }
    (run_dir / "selfplay_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"run dir: {run_dir}")
    print(f"final model: {final_model_path}")
    print(f"latest model: {latest_model_path}")
    print(f"best model: {best_model_path}")
    print(f"checkpoint pool: {checkpoint_dir}")
    if curriculum is not None:
        print(f"curriculum: {curriculum.name}")
        print(f"curriculum opponent: {curriculum.opponent_checkpoint_path}")
    if args.progress_video_freq > 0:
        print(f"training progression video: {progress_video_dir / 'combined' / 'training_progress.mp4'}")


if __name__ == "__main__":
    main()
