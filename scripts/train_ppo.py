from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionConfig
from badminton1d.callbacks import EntropyScheduleCallback, RallyDiagnosticsCallback, SafeWinRateEvalCallback
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.factorized_ppo import RecoveryFactorizedPPO
from badminton1d.policy import CONTINUOUS_LOG_STD_MAX, CONTINUOUS_LOG_STD_MIN, MaskedBadmintonPolicy
from badminton1d.reset_sampling import ResetSamplingConfig
from badminton1d.reward_shaping import AttackRewardConfig, LoopPenaltyConfig, PressureRewardConfig
from badminton1d.rl_env import (
    COUNTERFACTUAL_OPPONENT_RESPONSE_SAMPLES,
    RECOVERY_COUNTERFACTUAL_OTHER_SAMPLE_COUNT,
    BadmintonRLEnv,
    RLEnvConfig,
    RewardConfig,
)
from badminton1d.shot_generators import TacticRuntimeConfig
from badminton1d.utils import ensure_directory


def _flag_was_provided(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


PREFIX_COMPATIBLE_KEYS = {
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.value_net.0.weight",
}


def load_base_parameters_compatibly(model: PPO, base_model: PPO) -> None:
    try:
        model.set_parameters(base_model.get_parameters(), exact_match=False)
        return
    except RuntimeError as error:
        if "size mismatch" not in str(error):
            raise

    target_state = model.policy.state_dict()
    source_state = base_model.policy.state_dict()
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            continue
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            continue
        if (
            key in PREFIX_COMPATIBLE_KEYS
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
        ):
            if source_value.shape[1] >= target_value.shape[1]:
                target_state[key] = source_value[:, : target_value.shape[1]]
            elif target_value.shape[1] - source_value.shape[1] == 4 and target_value.shape[1] <= 53:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, :29] = source_value[:, :29]
                expanded[:, 33:] = source_value[:, 29:]
                target_state[key] = expanded
            elif target_value.shape[1] - source_value.shape[1] == 4:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, : source_value.shape[1]] = source_value
                target_state[key] = expanded
            elif target_value.shape[1] - source_value.shape[1] == 8:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, :29] = source_value[:, :29]
                expanded[:, 33:-4] = source_value[:, 29:]
                target_state[key] = expanded
    model.policy.load_state_dict(target_state, strict=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on the 2D badminton rally environment.")
    parser.add_argument("--base-checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl/ppo_run"))
    parser.add_argument("--train-side", choices=("left", "right"), default="left")
    parser.add_argument("--opponent", choices=("safe", "random", "greedy"), default="safe")
    parser.add_argument("--mirror-sides", action="store_true")
    parser.add_argument("--mirror-match-fraction", type=float, default=0.25)
    parser.add_argument("--initial-server", choices=("left", "right", "train", "opponent", "random"), default="random")
    parser.add_argument("--random-service-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="subproc")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--checkpoint-freq", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tensorboard-log", type=Path, default=Path("outputs/rl/tensorboard"))
    parser.add_argument("--ent-coef", type=float, default=0.002)
    parser.add_argument("--ent-coef-final", type=float, default=0.002)
    parser.add_argument("--random-start-prob", type=float, default=0.0)
    parser.add_argument("--midrally-start-prob", type=float, default=0.0)
    parser.add_argument("--opponent-serve-start-prob", type=float, default=0.0)
    parser.add_argument("--reaction-time", type=float, default=0.15)
    parser.add_argument("--court-mode", choices=("1d", "2d"), default="2d")
    parser.add_argument(
        "--policy-type",
        choices=("conditional_prob", "continuous_action", "velocity_oriented", "tactic_oriented", "mixed_discrete_continous"),
        default="velocity_oriented",
    )
    parser.add_argument("--regenerate-lookup-table", action="store_true")
    parser.add_argument("--lookup-table-dir", type=Path, default=Path("lookup_tables"))
    parser.add_argument("--loop-penalty", type=float, default=0.03)
    parser.add_argument("--loop-window", type=int, default=4)
    parser.add_argument("--defensive-return-reward", type=float, default=0.0)
    parser.add_argument("--serve-return-reward", type=float, default=0.0)
    parser.add_argument("--max-rally-stages", type=int, default=120)
    parser.add_argument("--max-rally-penalty", type=float, default=1.0)
    parser.add_argument("--stage-penalty", type=float, default=0.0)
    parser.add_argument("--stall-penalty", type=float, default=0.0)
    parser.add_argument("--stall-penalty-start", type=int, default=24)
    parser.add_argument("--pressure-reward-weight", type=float, default=0.0)
    parser.add_argument("--attack-reward-weight", type=float, default=0.0)
    parser.add_argument("--attack-min-speed", type=float, default=18.0)
    parser.add_argument("--attack-downward-vz-threshold", type=float, default=0.0)
    parser.add_argument("--mask-mid-rally-hitter-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-recovery-factorized-advantage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recovery-counterfactual-baseline", choices=("average", "best"), default="average")
    parser.add_argument(
        "--recovery-counterfactual-other-sample-count",
        type=int,
        default=RECOVERY_COUNTERFACTUAL_OTHER_SAMPLE_COUNT,
    )
    parser.add_argument("--recovery-counterfactual-advantage-coef", type=float, default=0.05)
    parser.add_argument("--recovery-counterfactual-distribution-coef", type=float, default=0.0)
    parser.add_argument("--recovery-counterfactual-distribution-temperature", type=float, default=0.25)
    parser.add_argument(
        "--counterfactual-opponent-response-samples",
        type=int,
        default=COUNTERFACTUAL_OPPONENT_RESPONSE_SAMPLES,
    )
    parser.add_argument(
        "--recovery-counterfactual-expected-response-target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--recovery-full-diagnostics-probability", type=float, default=0.0)
    parser.add_argument("--eval-deterministic", action="store_true")
    parser.add_argument("--player-speed", type=float, default=SimulationConfig().player.v_max)
    parser.add_argument("--racket-length", type=float, default=SimulationConfig().player.r_reach)
    parser.add_argument("--max-hitting-height", type=float, default=SimulationConfig().player.z_max)
    parser.add_argument("--movement-model", choices=("constant_velocity", "accelerated"), default=SimulationConfig().player.movement_model)
    parser.add_argument("--player-acceleration", type=float, default=SimulationConfig().player.acceleration)
    parser.add_argument("--player-deceleration", type=float, default=None)
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="drag_square")
    parser.add_argument("--drag-coefficient", type=float, default=0.2)
    parser.add_argument("--horizontal-drag-coefficient", type=float, default=0.2)
    parser.add_argument("--vertical-drag-coefficient", type=float, default=0.16)
    parser.add_argument("--shuttle-speed-min", type=float, default=0.1)
    parser.add_argument("--shuttle-speed-max", type=float, default=SimulationConfig().action.vy_max_forward)
    parser.add_argument("--conditional-recovery-grid", action=argparse.BooleanOptionalAction, default=False)
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
    args = parser.parse_args()
    argv = sys.argv[1:]
    if args.court_mode == "1d":
        if not _flag_was_provided(argv, "--theta-bins") and not _flag_was_provided(argv, "--vy-bins"):
            args.theta_bins = 15
        if not _flag_was_provided(argv, "--speed-bins") and not _flag_was_provided(argv, "--vz-bins"):
            args.speed_bins = 11
        if not _flag_was_provided(argv, "--y-rec-bins"):
            args.y_rec_bins = 5
    return args


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
            conditional_recovery_grid=args.conditional_recovery_grid,
            intercept_count=args.intercept_count,
            reaction_miss_fast_threshold=args.reaction_miss_fast_threshold,
            reaction_miss_fast_probability=args.reaction_miss_fast_probability,
            reaction_miss_secondary_threshold=args.reaction_miss_secondary_threshold,
            reaction_miss_secondary_probability=args.reaction_miss_secondary_probability,
            reaction_miss_zero_threshold=args.reaction_miss_zero_threshold,
        )
    )


def build_env_kwargs(args: argparse.Namespace) -> dict[str, object]:
    tactic_runtime = TacticRuntimeConfig(
        regenerate_lookup_table=args.regenerate_lookup_table,
        lookup_dir=args.lookup_table_dir,
    )
    return {
        "config": build_sim_config(args),
        "rl_config": RLEnvConfig(
            train_side=args.train_side,
            opponent_type=args.opponent,
            initial_server=args.initial_server,
            random_service_x=args.random_service_x,
            mirror_train_side=args.mirror_sides,
            mirror_match_fraction=args.mirror_match_fraction,
            train_reaction_time=args.reaction_time,
            opponent_reaction_time=args.reaction_time,
            max_stages_per_rally=args.max_rally_stages,
            policy_type=args.policy_type,
            tactic_runtime=tactic_runtime,
            reset_sampling=ResetSamplingConfig(
                random_start_prob=args.random_start_prob,
                midrally_start_prob=args.midrally_start_prob,
                opponent_serve_start_prob=args.opponent_serve_start_prob,
            ),
            reward=RewardConfig(
                defensive_return_reward=args.defensive_return_reward,
                serve_return_reward=args.serve_return_reward,
                max_rally_penalty=args.max_rally_penalty,
                stage_penalty=args.stage_penalty,
                stall_penalty=args.stall_penalty,
                stall_penalty_start=args.stall_penalty_start,
                loop_penalty=LoopPenaltyConfig(penalty=args.loop_penalty, window=args.loop_window),
                pressure_reward=PressureRewardConfig(weight=args.pressure_reward_weight),
                attack_reward=AttackRewardConfig(
                    weight=args.attack_reward_weight,
                    min_speed=args.attack_min_speed,
                    downward_vz_threshold=args.attack_downward_vz_threshold,
                ),
            ),
            recovery_counterfactual_other_sample_count=args.recovery_counterfactual_other_sample_count,
            counterfactual_opponent_response_samples=args.counterfactual_opponent_response_samples,
            recovery_counterfactual_expected_response_target=args.recovery_counterfactual_expected_response_target,
            recovery_full_diagnostics_probability=args.recovery_full_diagnostics_probability,
        ),
        "discrete_action_config": DiscreteActionConfig(
            phi_bins=args.phi_bins,
            theta_bins=args.theta_bins,
            speed_bins=args.speed_bins,
            x_rec_bins=args.x_rec_bins,
            y_rec_bins=args.y_rec_bins,
        ),
    }


def make_monitored_env(args: argparse.Namespace, output_dir: Path, include_records: bool = False):
    env_kwargs = build_env_kwargs(args)
    rl_config = env_kwargs["rl_config"]
    assert isinstance(rl_config, RLEnvConfig)
    env_kwargs["rl_config"] = RLEnvConfig(
        train_side=rl_config.train_side,
        opponent_type=rl_config.opponent_type,
        initial_server=rl_config.initial_server,
        random_service_x=rl_config.random_service_x,
        mirror_train_side=rl_config.mirror_train_side,
        mirror_match_fraction=rl_config.mirror_match_fraction,
        train_reaction_time=rl_config.train_reaction_time,
        opponent_reaction_time=rl_config.opponent_reaction_time,
        max_stages_per_rally=rl_config.max_stages_per_rally,
        serve_z0=rl_config.serve_z0,
        include_feasible_mask=rl_config.include_feasible_mask,
        include_reaction_risk_features=rl_config.include_reaction_risk_features,
        include_records_in_info=include_records,
        policy_type=rl_config.policy_type,
        tactic_runtime=rl_config.tactic_runtime,
        reset_sampling=rl_config.reset_sampling,
        reward=rl_config.reward,
        recovery_counterfactual_other_sample_count=rl_config.recovery_counterfactual_other_sample_count,
        counterfactual_opponent_response_samples=rl_config.counterfactual_opponent_response_samples,
        recovery_counterfactual_expected_response_target=(
            rl_config.recovery_counterfactual_expected_response_target
        ),
        recovery_full_diagnostics_probability=rl_config.recovery_full_diagnostics_probability,
    )

    def _factory() -> Monitor:
        return Monitor(BadmintonRLEnv(**env_kwargs))

    return _factory


def main() -> None:
    args = parse_args()
    if args.base_checkpoint_path is not None and not args.base_checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint_path}")
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if not 0.0 <= args.mirror_match_fraction <= 1.0:
        raise ValueError("--mirror-match-fraction must be in [0, 1]")
    if args.recovery_counterfactual_other_sample_count < 0:
        raise ValueError("--recovery-counterfactual-other-sample-count must be zero or greater")
    if args.recovery_counterfactual_advantage_coef < 0.0:
        raise ValueError("--recovery-counterfactual-advantage-coef must be zero or greater")
    if args.recovery_counterfactual_distribution_coef < 0.0:
        raise ValueError("--recovery-counterfactual-distribution-coef must be zero or greater")
    if args.recovery_counterfactual_distribution_temperature <= 0.0:
        raise ValueError("--recovery-counterfactual-distribution-temperature must be positive")
    if args.counterfactual_opponent_response_samples < 1:
        raise ValueError("--counterfactual-opponent-response-samples must be positive")
    if not 0.0 <= args.recovery_full_diagnostics_probability <= 1.0:
        raise ValueError("--recovery-full-diagnostics-probability must be in [0, 1]")

    run_dir = args.output_dir
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    ensure_directory(run_dir)
    ensure_directory(checkpoint_dir)
    ensure_directory(best_dir)
    ensure_directory(args.tensorboard_log)

    vec_env_cls = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    train_env = make_vec_env(
        make_monitored_env(args, run_dir),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
    )
    train_env_kwargs = build_env_kwargs(args)
    policy_kwargs = {
        "sim_config": train_env_kwargs["config"],
        "discrete_action_config": train_env_kwargs["discrete_action_config"],
        "policy_type": args.policy_type,
        "mask_mid_rally_hitter_actions": args.mask_mid_rally_hitter_actions,
        "tactic_runtime_config": TacticRuntimeConfig(
            regenerate_lookup_table=args.regenerate_lookup_table,
            lookup_dir=args.lookup_table_dir,
        ),
    }
    safe_eval_args = argparse.Namespace(**vars(args))
    safe_eval_args.opponent = "safe"
    safe_eval_kwargs = build_env_kwargs(safe_eval_args)
    safe_eval_rl_config = safe_eval_kwargs["rl_config"]
    assert isinstance(safe_eval_rl_config, RLEnvConfig)
    safe_eval_kwargs["rl_config"] = RLEnvConfig(
        train_side=safe_eval_rl_config.train_side,
        opponent_type="safe",
        initial_server=safe_eval_rl_config.initial_server,
        random_service_x=safe_eval_rl_config.random_service_x,
        mirror_train_side=safe_eval_rl_config.mirror_train_side,
        mirror_match_fraction=safe_eval_rl_config.mirror_match_fraction,
        train_reaction_time=safe_eval_rl_config.train_reaction_time,
        opponent_reaction_time=safe_eval_rl_config.opponent_reaction_time,
        max_stages_per_rally=safe_eval_rl_config.max_stages_per_rally,
        serve_z0=safe_eval_rl_config.serve_z0,
        include_feasible_mask=safe_eval_rl_config.include_feasible_mask,
        include_reaction_risk_features=safe_eval_rl_config.include_reaction_risk_features,
        include_records_in_info=False,
        policy_type=safe_eval_rl_config.policy_type,
        tactic_runtime=safe_eval_rl_config.tactic_runtime,
        reset_sampling=safe_eval_rl_config.reset_sampling,
        reward=safe_eval_rl_config.reward,
        recovery_counterfactual_other_sample_count=0,
        counterfactual_opponent_response_samples=safe_eval_rl_config.counterfactual_opponent_response_samples,
        recovery_counterfactual_expected_response_target=False,
        recovery_full_diagnostics_probability=0.0,
    )
    safe_eval_env = BadmintonRLEnv(**safe_eval_kwargs, seed=args.seed + 10_000)

    ppo_class = RecoveryFactorizedPPO if args.use_recovery_factorized_advantage else PPO
    ppo_extra_kwargs = (
        {
            "use_recovery_factorized_advantage": True,
            "recovery_counterfactual_baseline": args.recovery_counterfactual_baseline,
            "recovery_counterfactual_advantage_coef": args.recovery_counterfactual_advantage_coef,
            "recovery_counterfactual_distribution_coef": args.recovery_counterfactual_distribution_coef,
            "recovery_counterfactual_distribution_temperature": args.recovery_counterfactual_distribution_temperature,
        }
        if args.use_recovery_factorized_advantage
        else {}
    )
    if args.base_checkpoint_path is None:
        model = ppo_class(
            MaskedBadmintonPolicy,
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=str(args.tensorboard_log),
            seed=args.seed,
            **ppo_extra_kwargs,
        )
    else:
        model = ppo_class(
            MaskedBadmintonPolicy,
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=str(args.tensorboard_log),
            seed=args.seed,
            **ppo_extra_kwargs,
        )
        base_model = PPO.load(
            args.base_checkpoint_path,
        )
        load_base_parameters_compatibly(model, base_model)

    diagnostics_callback = RallyDiagnosticsCallback(run_dir)
    callbacks_list = [diagnostics_callback]
    if args.ent_coef_final is not None and not np.isclose(args.ent_coef_final, args.ent_coef):
        callbacks_list.append(
            EntropyScheduleCallback(
                ent_coef_initial=args.ent_coef,
                ent_coef_final=args.ent_coef_final,
                total_timesteps=args.total_timesteps,
            )
        )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_badminton",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = SafeWinRateEvalCallback(
        eval_env=safe_eval_env,
        best_model_save_path=best_dir,
        log_path=run_dir / "eval_logs",
        eval_freq=max(args.eval_freq, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=args.eval_deterministic,
        eval_seed=args.seed + 30_000,
    )
    callbacks_list.extend([checkpoint_callback, eval_callback])
    callbacks = CallbackList(callbacks_list)

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=False)
    final_model_path = run_dir / "final_model.zip"
    model.save(final_model_path)

    summary = {
        "base_checkpoint_path": str(args.base_checkpoint_path) if args.base_checkpoint_path is not None else None,
        "train_side": args.train_side,
        "opponent": args.opponent,
        "mirror_sides": args.mirror_sides,
        "mirror_match_fraction": args.mirror_match_fraction,
        "initial_server": args.initial_server,
        "random_service_x": args.random_service_x,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env,
        "seed": args.seed,
        "ent_coef": args.ent_coef,
        "ent_coef_final": args.ent_coef_final,
        "random_start_prob": args.random_start_prob,
        "midrally_start_prob": args.midrally_start_prob,
        "opponent_serve_start_prob": args.opponent_serve_start_prob,
        "reaction_time": args.reaction_time,
        "court_mode": args.court_mode,
        "policy_type": args.policy_type,
        "continuous_log_std_min": CONTINUOUS_LOG_STD_MIN,
        "continuous_log_std_max": CONTINUOUS_LOG_STD_MAX,
        "regenerate_lookup_table": args.regenerate_lookup_table,
        "lookup_table_dir": str(args.lookup_table_dir),
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
        "attack_reward_weight": args.attack_reward_weight,
        "attack_min_speed": args.attack_min_speed,
        "attack_downward_vz_threshold": args.attack_downward_vz_threshold,
        "mask_mid_rally_hitter_actions": args.mask_mid_rally_hitter_actions,
        "use_recovery_factorized_advantage": args.use_recovery_factorized_advantage,
        "recovery_counterfactual_baseline": args.recovery_counterfactual_baseline,
        "recovery_counterfactual_other_sample_count": args.recovery_counterfactual_other_sample_count,
        "recovery_counterfactual_advantage_coef": args.recovery_counterfactual_advantage_coef,
        "recovery_counterfactual_distribution_coef": args.recovery_counterfactual_distribution_coef,
        "recovery_counterfactual_distribution_temperature": args.recovery_counterfactual_distribution_temperature,
        "counterfactual_opponent_response_samples": args.counterfactual_opponent_response_samples,
        "recovery_counterfactual_expected_response_target": args.recovery_counterfactual_expected_response_target,
        "recovery_full_diagnostics_probability": args.recovery_full_diagnostics_probability,
        "eval_deterministic": args.eval_deterministic,
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
        "conditional_recovery_grid": args.conditional_recovery_grid,
        "intercept_count": args.intercept_count,
        "reaction_miss_fast_threshold": args.reaction_miss_fast_threshold,
        "reaction_miss_fast_probability": args.reaction_miss_fast_probability,
        "reaction_miss_secondary_threshold": args.reaction_miss_secondary_threshold,
        "reaction_miss_secondary_probability": args.reaction_miss_secondary_probability,
        "reaction_miss_zero_threshold": args.reaction_miss_zero_threshold,
        "include_reaction_risk_features": True,
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_dir / "best_model.zip"),
        "tensorboard_log": str(args.tensorboard_log),
    }
    (run_dir / "training_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"run dir: {run_dir}")
    print(f"final model: {final_model_path}")
    print(f"best model: {best_dir / 'best_model.zip'}")
    print(f"tensorboard: {args.tensorboard_log}")


if __name__ == "__main__":
    main()
