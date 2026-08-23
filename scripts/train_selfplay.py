from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.action_space import DiscreteActionConfig
from badminton.callbacks import EntropyScheduleCallback, RallyDiagnosticsCallback
from badminton.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton.curricula import (
    available_training_curricula,
    build_training_curriculum,
)
from badminton.factorized_ppo import RecoveryFactorizedPPO
from badminton.policy import CONTINUOUS_LOG_STD_MAX, CONTINUOUS_LOG_STD_MIN, MIXED_RECOVERY_LOG_STD, MaskedBadmintonPolicy
from badminton.reset_sampling import ResetSamplingConfig
from badminton.reward_shaping import (
    AttackRewardConfig,
    DefensiveLiftRewardConfig,
    LoopPenaltyConfig,
    NetProximityRewardConfig,
    OpponentTravelRewardConfig,
    PressureRewardConfig,
    ReturnDepthRewardConfig,
)
from badminton.opponents import make_opponent
from badminton.rl_env import (
    COUNTERFACTUAL_OPPONENT_RESPONSE_SAMPLES,
    RECOVERY_COUNTERFACTUAL_OTHER_SAMPLE_COUNT,
    BadmintonRLEnv,
    RLEnvConfig,
    RewardConfig,
)
from badminton.shot_generators import TacticRuntimeConfig
from badminton.selfplay import (
    CheckpointPool,
    FixedCheckpointOpponent,
    FrozenCheckpointOpponent,
    LiveModelOpponent,
    MixedCheckpointOpponent,
    SelfPlayCheckpointCallback,
    SelfPlayEvalCallback,
    SelfPlayProgressVideoCallback,
    build_selfplay_env,
    replace_with_existing_file,
)
from badminton.utils import ensure_directory


PREFIX_COMPATIBLE_KEYS = {
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.value_net.0.weight",
}


def _flag_was_provided(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def load_base_policy_state_compatibly(model: PPO, source_state: dict[str, object], *, source_label: str) -> None:
    target_state = model.policy.state_dict()
    copied = 0
    prefix_copied = 0
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            continue
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            copied += 1
            continue
        if (
            key in PREFIX_COMPATIBLE_KEYS
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
        ):
            if source_value.shape[1] >= target_value.shape[1]:
                target_state[key] = source_value[:, : target_value.shape[1]]
                prefix_copied += 1
            elif target_value.shape[1] - source_value.shape[1] == 4 and target_value.shape[1] <= 53:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, :29] = source_value[:, :29]
                expanded[:, 33:] = source_value[:, 29:]
                target_state[key] = expanded
                prefix_copied += 1
            elif target_value.shape[1] - source_value.shape[1] == 4:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, : source_value.shape[1]] = source_value
                target_state[key] = expanded
                prefix_copied += 1
            elif target_value.shape[1] - source_value.shape[1] == 8:
                expanded = target_value.clone()
                expanded.zero_()
                expanded[:, :29] = source_value[:, :29]
                expanded[:, 33:-4] = source_value[:, 29:]
                target_state[key] = expanded
                prefix_copied += 1

    model.policy.load_state_dict(target_state, strict=True)
    print(
        f"Loaded compatible base policy parameters from {source_label} "
        f"({copied} exact tensors, {prefix_copied} input-prefix tensors)."
    )


def load_base_parameters_compatibly(model: PPO, base_model: PPO) -> None:
    try:
        model.set_parameters(base_model.get_parameters(), exact_match=False)
        return
    except RuntimeError as error:
        if "size mismatch" not in str(error):
            raise

    load_base_policy_state_compatibly(model, base_model.policy.state_dict(), source_label="loaded PPO model")


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
    parser.add_argument("--random-service-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selfplay-total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.002)
    parser.add_argument("--ent-coef-final", type=float, default=0.002)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="subproc")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument(
        "--opponent-sampling-mode",
        choices=("uniform", "random", "recency", "newest", "variety"),
        default="recency",
    )
    parser.add_argument("--checkpoint-recency-power", type=float, default=3.0)
    parser.add_argument("--recent-opponent-weight", type=float, default=0.9)
    parser.add_argument("--older-opponent-weight", type=float, default=0.1)
    parser.add_argument("--heuristic-opponent-prob", type=float, default=0.05)
    parser.add_argument("--historical-anchor-dir", type=Path, default=None)
    parser.add_argument("--historical-anchor-pool-size", type=int, default=1000)
    parser.add_argument("--historical-anchor-weight", type=float, default=0.70)
    parser.add_argument("--recent-continuation-weight", type=float, default=0.15)
    parser.add_argument("--newest-continuation-weight", type=float, default=0.05)
    parser.add_argument(
        "--curriculum",
        choices=("none", *available_training_curricula()),
        default="none",
    )
    parser.add_argument("--curriculum-opponent-checkpoint", type=Path, default=None)
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
    parser.add_argument("--loop-penalty", type=float, default=0.1)
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
    parser.add_argument("--feasible-pressure-reward-weight", type=float, default=0.0)
    parser.add_argument("--no-feasible-intercept-bonus", type=float, default=0.0)
    parser.add_argument("--opponent-intercept-continue-penalty", type=float, default=0.0)
    parser.add_argument("--defensive-lift-reward-weight", type=float, default=0.0)
    parser.add_argument("--intercept-flight-ratio-reward-weight", type=float, default=0.0)
    parser.add_argument("--defensive-lift-min-theta-deg", type=float, default=15.0)
    parser.add_argument("--defensive-lift-target-flight-time", type=float, default=1.4)
    parser.add_argument("--defensive-lift-min-depth-ratio", type=float, default=0.7)
    parser.add_argument("--intercept-ratio-min-intended-flight-time", type=float, default=0.8)
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
    parser.add_argument("--use-shot-cf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shot-cf-coef", type=float, default=0.1)
    parser.add_argument("--shot-cf-top-m", type=int, default=20)
    parser.add_argument("--shot-cf-num-modes", type=int, default=3)
    parser.add_argument("--shot-cf-min-landing-dist", type=float, default=1.0)
    parser.add_argument("--shot-cf-depth", type=int, default=1)
    parser.add_argument("--shot-cf-include-chosen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shot-cf-skip-low-diversity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shot-cf-min-modes", type=int, default=2)
    parser.add_argument("--shot-cf-value-detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shot-cf-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shot-cf-debug-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--opponent-travel-reward-weight", type=float, default=0.0)
    parser.add_argument("--return-depth-reward-weight", type=float, default=0.0)
    parser.add_argument("--net-proximity-reward-weight", type=float, default=0.0)
    parser.add_argument("--net-proximity-threshold", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=100000)
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
    parser.add_argument("--recovery-x-margin", type=float, default=None)
    parser.add_argument("--recovery-net-margin", type=float, default=None)
    parser.add_argument("--recovery-back-margin", type=float, default=None)
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
    action_defaults = SimulationConfig().action
    mixed_recovery = args.policy_type == "mixed_discrete_continous"
    recovery_x_margin = 0.0 if mixed_recovery and args.recovery_x_margin is None else args.recovery_x_margin
    recovery_net_margin = 0.0 if mixed_recovery and args.recovery_net_margin is None else args.recovery_net_margin
    recovery_back_margin = 0.0 if mixed_recovery and args.recovery_back_margin is None else args.recovery_back_margin
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
            recovery_x_margin=(
                action_defaults.recovery_x_margin if recovery_x_margin is None else recovery_x_margin
            ),
            recovery_net_margin=(
                action_defaults.recovery_net_margin if recovery_net_margin is None else recovery_net_margin
            ),
            recovery_back_margin=(
                action_defaults.recovery_back_margin if recovery_back_margin is None else recovery_back_margin
            ),
            conditional_recovery_grid=args.conditional_recovery_grid,
            intercept_count=args.intercept_count,
            reaction_miss_fast_threshold=args.reaction_miss_fast_threshold,
            reaction_miss_fast_probability=args.reaction_miss_fast_probability,
            reaction_miss_secondary_threshold=args.reaction_miss_secondary_threshold,
            reaction_miss_secondary_probability=args.reaction_miss_secondary_probability,
            reaction_miss_zero_threshold=args.reaction_miss_zero_threshold,
        )
    )


def build_fixed_checkpoint_opponent(
    *,
    checkpoint_path: Path,
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    policy_type: str,
    tactic_runtime_config: TacticRuntimeConfig,
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
        policy_type=policy_type,
        tactic_runtime_config=tactic_runtime_config,
        deterministic=deterministic,
        hitter_deterministic=hitter_deterministic,
        receiver_deterministic=receiver_deterministic,
    )


def create_compatible_base_checkpoint(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    tactic_runtime_config: TacticRuntimeConfig,
    reward_config: RewardConfig,
    reset_sampling_config: ResetSamplingConfig,
) -> Path:
    compatible_path = run_dir / "compatible_base_model.zip"
    env = BadmintonRLEnv(
        config=sim_config,
        rl_config=RLEnvConfig(
            train_side=args.train_side,
            initial_server=args.initial_server,
            random_service_x=args.random_service_x,
            train_reaction_time=args.reaction_time,
            opponent_reaction_time=args.reaction_time,
            max_stages_per_rally=args.max_rally_stages,
            policy_type=args.policy_type,
            tactic_runtime=tactic_runtime_config,
            reward=reward_config,
            reset_sampling=reset_sampling_config,
            recovery_counterfactual_other_sample_count=args.recovery_counterfactual_other_sample_count,
            counterfactual_opponent_response_samples=args.counterfactual_opponent_response_samples,
            recovery_counterfactual_expected_response_target=args.recovery_counterfactual_expected_response_target,
            recovery_full_diagnostics_probability=args.recovery_full_diagnostics_probability,
            use_shot_cf=args.use_shot_cf,
            shot_cf_coef=args.shot_cf_coef,
            shot_cf_top_m=args.shot_cf_top_m,
            shot_cf_num_modes=args.shot_cf_num_modes,
            shot_cf_min_landing_dist=args.shot_cf_min_landing_dist,
            shot_cf_depth=args.shot_cf_depth,
            shot_cf_include_chosen=args.shot_cf_include_chosen,
            shot_cf_skip_low_diversity=args.shot_cf_skip_low_diversity,
            shot_cf_min_modes=args.shot_cf_min_modes,
            shot_cf_value_detach=args.shot_cf_value_detach,
            shot_cf_normalize=args.shot_cf_normalize,
            shot_cf_debug_log=args.shot_cf_debug_log,
        ),
        discrete_action_config=discrete_action_config,
        opponent=make_opponent("safe", seed=args.seed + 700_000),
        seed=args.seed + 700_001,
    )
    model = PPO(
        MaskedBadmintonPolicy,
        env,
        policy_kwargs={
            "sim_config": sim_config,
            "discrete_action_config": discrete_action_config,
            "policy_type": args.policy_type,
            "tactic_runtime_config": tactic_runtime_config,
            "mask_mid_rally_hitter_actions": args.mask_mid_rally_hitter_actions,
        },
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=8,
        n_steps=8,
        ent_coef=args.ent_coef,
        verbose=0,
        seed=args.seed,
    )
    try:
        base_model = PPO.load(args.base_checkpoint_path)
        load_base_parameters_compatibly(model, base_model)
    except RuntimeError as error:
        print(f"Falling back to raw checkpoint parameter copy after PPO.load failed: {error}")
        _, params, _ = load_from_zip_file(args.base_checkpoint_path, device=model.device)
        policy_state = params.get("policy")
        if policy_state is None:
            raise RuntimeError(f"Checkpoint has no policy parameters: {args.base_checkpoint_path}") from error
        load_base_policy_state_compatibly(model, policy_state, source_label=str(args.base_checkpoint_path))
    model.save(compatible_path)
    env.close()
    return compatible_path


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
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive")
    if args.opponent_sampling_mode == "variety" and args.historical_anchor_dir is None:
        raise ValueError("--opponent-sampling-mode variety requires --historical-anchor-dir")
    if args.historical_anchor_dir is not None and not args.historical_anchor_dir.exists():
        raise FileNotFoundError(f"Historical anchor dir not found: {args.historical_anchor_dir}")
    if args.historical_anchor_pool_size <= 0:
        raise ValueError("--historical-anchor-pool-size must be positive")
    if args.heuristic_opponent_prob < 0.0:
        raise ValueError("--heuristic-opponent-prob must be zero or greater")
    if args.historical_anchor_weight < 0.0:
        raise ValueError("--historical-anchor-weight must be zero or greater")
    if args.recent_continuation_weight < 0.0:
        raise ValueError("--recent-continuation-weight must be zero or greater")
    if args.newest_continuation_weight < 0.0:
        raise ValueError("--newest-continuation-weight must be zero or greater")
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
    if args.defensive_lift_target_flight_time <= 0.0:
        raise ValueError("--defensive-lift-target-flight-time must be positive")
    if not -90.0 <= args.defensive_lift_min_theta_deg <= 90.0:
        raise ValueError("--defensive-lift-min-theta-deg must be in [-90, 90]")
    if not 0.0 <= args.defensive_lift_min_depth_ratio <= 1.0:
        raise ValueError("--defensive-lift-min-depth-ratio must be in [0, 1]")
    if args.intercept_ratio_min_intended_flight_time <= 0.0:
        raise ValueError("--intercept-ratio-min-intended-flight-time must be positive")
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
    if args.shot_cf_coef < 0.0:
        raise ValueError("--shot-cf-coef must be zero or greater")
    if args.shot_cf_top_m <= 0:
        raise ValueError("--shot-cf-top-m must be positive")
    if args.shot_cf_num_modes <= 0:
        raise ValueError("--shot-cf-num-modes must be positive")
    if args.shot_cf_min_landing_dist < 0.0:
        raise ValueError("--shot-cf-min-landing-dist must be non-negative")
    if args.shot_cf_depth != 1:
        raise ValueError("Only --shot-cf-depth 1 is implemented")
    if args.shot_cf_min_modes <= 0:
        raise ValueError("--shot-cf-min-modes must be positive")

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
        phi_bins=args.phi_bins,
        theta_bins=args.theta_bins,
        speed_bins=args.speed_bins,
        x_rec_bins=args.x_rec_bins,
        y_rec_bins=args.y_rec_bins,
    )
    sim_config = build_sim_config(args)
    tactic_runtime_config = TacticRuntimeConfig(
        regenerate_lookup_table=args.regenerate_lookup_table,
        lookup_dir=args.lookup_table_dir,
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
        attack_reward=AttackRewardConfig(
            weight=args.attack_reward_weight,
            min_speed=args.attack_min_speed,
            downward_vz_threshold=args.attack_downward_vz_threshold,
        ),
        opponent_travel_reward=OpponentTravelRewardConfig(weight=args.opponent_travel_reward_weight),
        return_depth_reward=ReturnDepthRewardConfig(weight=args.return_depth_reward_weight),
        net_proximity_reward=NetProximityRewardConfig(
            weight=args.net_proximity_reward_weight,
            distance_threshold=args.net_proximity_threshold,
        ),
        defensive_lift_reward=DefensiveLiftRewardConfig(
            weight=args.defensive_lift_reward_weight,
            intercept_flight_ratio_reward_weight=args.intercept_flight_ratio_reward_weight,
            min_theta_deg=args.defensive_lift_min_theta_deg,
            target_flight_time=args.defensive_lift_target_flight_time,
            min_depth_ratio=args.defensive_lift_min_depth_ratio,
            min_ratio_intended_flight_time=args.intercept_ratio_min_intended_flight_time,
        ),
        feasible_pressure_reward_weight=args.feasible_pressure_reward_weight,
        no_feasible_intercept_bonus=args.no_feasible_intercept_bonus,
        opponent_intercept_continue_penalty=args.opponent_intercept_continue_penalty,
    )
    curriculum = None if args.curriculum == "none" else build_training_curriculum(
        args.curriculum,
        opponent_checkpoint_path=args.curriculum_opponent_checkpoint,
    )
    effective_initial_server = args.initial_server
    progress_primary_label = "current_vs_newest_checkpoint"
    progress_primary_dir = progress_video_dir / "current_vs_newest_checkpoint"
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

    effective_base_checkpoint_path = create_compatible_base_checkpoint(
        args=args,
        run_dir=run_dir,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        tactic_runtime_config=tactic_runtime_config,
        reward_config=reward_config,
        reset_sampling_config=reset_sampling_config,
    )
    checkpoint_sampling_mode = "recency" if args.opponent_sampling_mode == "variety" else args.opponent_sampling_mode
    train_pool = CheckpointPool(
        checkpoint_dir=checkpoint_dir,
        base_checkpoint_path=effective_base_checkpoint_path,
        pool_size=args.pool_size,
        sampling_mode=checkpoint_sampling_mode,
        recency_power=args.checkpoint_recency_power,
        recent_fraction=0.5,
        seed=args.seed + 1,
    )
    eval_pool = train_pool
    if curriculum is not None:
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
                policy_type=args.policy_type,
                tactic_runtime_config=tactic_runtime_config,
                seed=args.seed + 2 + (1 if include_records else 0),
                deterministic=False,
                hitter_deterministic=curriculum.opponent_hitter_deterministic,
                receiver_deterministic=curriculum.opponent_receiver_deterministic,
            )
        return MixedCheckpointOpponent(
            checkpoint_pool=CheckpointPool(
                checkpoint_dir=checkpoint_dir,
                base_checkpoint_path=effective_base_checkpoint_path,
                pool_size=args.pool_size,
                sampling_mode=checkpoint_sampling_mode,
                recency_power=args.checkpoint_recency_power,
                recent_fraction=0.5,
                seed=args.seed + 2 + (1 if include_records else 0),
            ),
            historical_anchor_pool=(
                CheckpointPool(
                    checkpoint_dir=args.historical_anchor_dir,
                    pool_size=args.historical_anchor_pool_size,
                    sampling_mode="linear_recency",
                    seed=args.seed + 102 + (1 if include_records else 0),
                    max_cached_models=1,
                )
                if args.opponent_sampling_mode == "variety" and args.historical_anchor_dir is not None
                else None
            ),
            sim_config=sim_config,
            discrete_action_config=discrete_action_config,
            policy_type=args.policy_type,
            tactic_runtime_config=tactic_runtime_config,
            heuristic_opponent_prob=args.heuristic_opponent_prob,
            recent_weight=args.recent_opponent_weight,
            older_weight=args.older_opponent_weight,
            historical_anchor_weight=(
                args.historical_anchor_weight if args.opponent_sampling_mode == "variety" else 0.0
            ),
            recent_continuation_weight=(
                args.recent_continuation_weight if args.opponent_sampling_mode == "variety" else 0.0
            ),
            newest_continuation_weight=(
                args.newest_continuation_weight if args.opponent_sampling_mode == "variety" else 0.0
            ),
            deterministic=False,
        )

    def make_selfplay_env(include_records: bool = False):
        return build_selfplay_env(
            train_side=args.train_side,
            mirror_train_side=args.mirror_sides,
            mirror_match_fraction=args.mirror_match_fraction,
            initial_server=effective_initial_server,
            random_service_x=args.random_service_x,
            sim_config=sim_config,
            train_reaction_time=args.reaction_time,
            opponent_reaction_time=args.reaction_time,
            max_stages_per_rally=args.max_rally_stages,
            policy_type=args.policy_type,
            tactic_runtime_config=tactic_runtime_config,
            reward_config=reward_config,
            reset_sampling_config=reset_sampling_config,
            seed=args.seed,
            discrete_action_config=discrete_action_config,
            opponent=make_training_opponent(include_records=include_records),
            include_records_in_info=include_records,
            recovery_counterfactual_other_sample_count=args.recovery_counterfactual_other_sample_count,
            counterfactual_opponent_response_samples=args.counterfactual_opponent_response_samples,
            recovery_counterfactual_expected_response_target=args.recovery_counterfactual_expected_response_target,
            recovery_full_diagnostics_probability=args.recovery_full_diagnostics_probability,
            use_shot_cf=args.use_shot_cf,
            shot_cf_coef=args.shot_cf_coef,
            shot_cf_top_m=args.shot_cf_top_m,
            shot_cf_num_modes=args.shot_cf_num_modes,
            shot_cf_min_landing_dist=args.shot_cf_min_landing_dist,
            shot_cf_depth=args.shot_cf_depth,
            shot_cf_include_chosen=args.shot_cf_include_chosen,
            shot_cf_skip_low_diversity=args.shot_cf_skip_low_diversity,
            shot_cf_min_modes=args.shot_cf_min_modes,
            shot_cf_value_detach=args.shot_cf_value_detach,
            shot_cf_normalize=args.shot_cf_normalize,
            shot_cf_debug_log=args.shot_cf_debug_log,
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

    ppo_class = RecoveryFactorizedPPO if args.use_recovery_factorized_advantage or args.use_shot_cf else PPO
    ppo_extra_kwargs = (
        {
            "use_recovery_factorized_advantage": args.use_recovery_factorized_advantage,
            "recovery_counterfactual_baseline": args.recovery_counterfactual_baseline,
            "recovery_counterfactual_advantage_coef": args.recovery_counterfactual_advantage_coef,
            "recovery_counterfactual_distribution_coef": args.recovery_counterfactual_distribution_coef,
            "recovery_counterfactual_distribution_temperature": args.recovery_counterfactual_distribution_temperature,
            "use_shot_cf": args.use_shot_cf,
            "shot_cf_coef": args.shot_cf_coef,
            "shot_cf_top_m": args.shot_cf_top_m,
            "shot_cf_num_modes": args.shot_cf_num_modes,
            "shot_cf_min_landing_dist": args.shot_cf_min_landing_dist,
            "shot_cf_depth": args.shot_cf_depth,
            "shot_cf_include_chosen": args.shot_cf_include_chosen,
            "shot_cf_skip_low_diversity": args.shot_cf_skip_low_diversity,
            "shot_cf_min_modes": args.shot_cf_min_modes,
            "shot_cf_value_detach": args.shot_cf_value_detach,
            "shot_cf_normalize": args.shot_cf_normalize,
            "shot_cf_debug_log": args.shot_cf_debug_log,
        }
        if args.use_recovery_factorized_advantage or args.use_shot_cf
        else {}
    )
    model = ppo_class(
        MaskedBadmintonPolicy,
        train_env,
        policy_kwargs={
            "sim_config": sim_config,
            "discrete_action_config": discrete_action_config,
            "policy_type": args.policy_type,
            "tactic_runtime_config": tactic_runtime_config,
            "mask_mid_rally_hitter_actions": args.mask_mid_rally_hitter_actions,
        },
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        **ppo_extra_kwargs,
    )
    base_model = PPO.load(effective_base_checkpoint_path)
    load_base_parameters_compatibly(model, base_model)

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
        random_service_x=args.random_service_x,
        sim_config=sim_config,
        reward_config=reward_config,
        reset_sampling_config=reset_sampling_config,
        train_reaction_time=args.reaction_time,
        opponent_reaction_time=args.reaction_time,
        max_stages_per_rally=args.max_rally_stages,
        discrete_action_config=discrete_action_config,
        policy_type=args.policy_type,
        tactic_runtime_config=tactic_runtime_config,
        checkpoint_pool=eval_pool,
        base_checkpoint_path=effective_base_checkpoint_path,
        anchor_eval_interval=args.anchor_eval_interval,
        anchor_checkpoint_dir=anchor_checkpoint_dir,
        deterministic=args.eval_deterministic,
        timestep_offset=args.resume_step_offset,
    )
    callbacks_list.extend([checkpoint_callback, eval_callback])
    if args.progress_video_freq > 0:
        progress_newest_env = build_selfplay_env(
            train_side=args.train_side,
            mirror_train_side=args.mirror_sides,
            mirror_match_fraction=args.mirror_match_fraction,
            initial_server=effective_initial_server,
            random_service_x=args.random_service_x,
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
                    policy_type=args.policy_type,
                    tactic_runtime_config=tactic_runtime_config,
                    seed=args.seed + 50_001,
                    deterministic=args.progress_video_deterministic,
                    hitter_deterministic=curriculum.opponent_hitter_deterministic,
                    receiver_deterministic=curriculum.opponent_receiver_deterministic,
                )
                if curriculum is not None
                else FrozenCheckpointOpponent(
                    pool=CheckpointPool(
                        checkpoint_dir=checkpoint_dir,
                        base_checkpoint_path=effective_base_checkpoint_path,
                        pool_size=args.pool_size,
                        sampling_mode="newest",
                        recency_power=args.checkpoint_recency_power,
                        recent_fraction=0.5,
                        seed=args.seed + 50_001,
                    ),
                    sim_config=sim_config,
                    discrete_action_config=discrete_action_config,
                    policy_type=args.policy_type,
                    tactic_runtime_config=tactic_runtime_config,
                    deterministic=args.progress_video_deterministic,
                )
            ),
            include_records_in_info=True,
            recovery_counterfactual_other_sample_count=0,
            counterfactual_opponent_response_samples=args.counterfactual_opponent_response_samples,
            recovery_counterfactual_expected_response_target=False,
            recovery_full_diagnostics_probability=0.0,
        )
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
            progress_mirror_env = build_selfplay_env(
                train_side=args.train_side,
                mirror_train_side=args.mirror_sides,
                mirror_match_fraction=args.mirror_match_fraction,
                initial_server=effective_initial_server,
                random_service_x=args.random_service_x,
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
                    policy_type=args.policy_type,
                    tactic_runtime_config=tactic_runtime_config,
                    deterministic=args.progress_video_deterministic,
                    label_name="mirror_self",
                ),
                include_records_in_info=True,
                recovery_counterfactual_other_sample_count=0,
                counterfactual_opponent_response_samples=args.counterfactual_opponent_response_samples,
                recovery_counterfactual_expected_response_target=False,
                recovery_full_diagnostics_probability=0.0,
            )
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

    model.learn(
        total_timesteps=args.selfplay_total_timesteps,
        callback=callbacks,
        log_interval=args.log_interval,
        progress_bar=False,
    )
    model.save(final_model_path)
    replace_with_existing_file(final_model_path, latest_model_path)

    train_pool.refresh()
    summary = {
        "base_checkpoint_path": str(effective_base_checkpoint_path),
        "requested_base_checkpoint_path": str(args.base_checkpoint_path),
        "resume_step_offset": args.resume_step_offset,
        "train_side": args.train_side,
        "mirror_sides": args.mirror_sides,
        "mirror_match_fraction": args.mirror_match_fraction,
        "initial_server": effective_initial_server,
        "requested_initial_server": args.initial_server,
        "random_service_x": args.random_service_x,
        "selfplay_total_timesteps": args.selfplay_total_timesteps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env,
        "seed": args.seed,
        "log_interval": args.log_interval,
        "ent_coef": args.ent_coef,
        "ent_coef_final": args.ent_coef_final,
        "save_interval": args.save_interval,
        "pool_size": args.pool_size,
        "opponent_sampling_mode": args.opponent_sampling_mode,
        "checkpoint_recency_power": args.checkpoint_recency_power,
        "heuristic_opponent_prob": args.heuristic_opponent_prob,
        "historical_anchor_dir": None if args.historical_anchor_dir is None else str(args.historical_anchor_dir),
        "historical_anchor_pool_size": args.historical_anchor_pool_size,
        "historical_anchor_weight": args.historical_anchor_weight,
        "recent_continuation_weight": args.recent_continuation_weight,
        "newest_continuation_weight": args.newest_continuation_weight,
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
        "policy_type": args.policy_type,
        "continuous_log_std_min": CONTINUOUS_LOG_STD_MIN,
        "continuous_log_std_max": CONTINUOUS_LOG_STD_MAX,
        "mixed_recovery_log_std": MIXED_RECOVERY_LOG_STD,
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
        "feasible_pressure_reward_weight": args.feasible_pressure_reward_weight,
        "no_feasible_intercept_bonus": args.no_feasible_intercept_bonus,
        "opponent_intercept_continue_penalty": args.opponent_intercept_continue_penalty,
        "defensive_lift_reward_weight": args.defensive_lift_reward_weight,
        "intercept_flight_ratio_reward_weight": args.intercept_flight_ratio_reward_weight,
        "defensive_lift_min_theta_deg": args.defensive_lift_min_theta_deg,
        "defensive_lift_target_flight_time": args.defensive_lift_target_flight_time,
        "defensive_lift_min_depth_ratio": args.defensive_lift_min_depth_ratio,
        "intercept_ratio_min_intended_flight_time": args.intercept_ratio_min_intended_flight_time,
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
        "use_shot_cf": args.use_shot_cf,
        "shot_cf_coef": args.shot_cf_coef,
        "shot_cf_top_m": args.shot_cf_top_m,
        "shot_cf_num_modes": args.shot_cf_num_modes,
        "shot_cf_min_landing_dist": args.shot_cf_min_landing_dist,
        "shot_cf_depth": args.shot_cf_depth,
        "shot_cf_include_chosen": args.shot_cf_include_chosen,
        "shot_cf_skip_low_diversity": args.shot_cf_skip_low_diversity,
        "shot_cf_min_modes": args.shot_cf_min_modes,
        "shot_cf_value_detach": args.shot_cf_value_detach,
        "shot_cf_normalize": args.shot_cf_normalize,
        "shot_cf_debug_log": args.shot_cf_debug_log,
        "opponent_travel_reward_weight": args.opponent_travel_reward_weight,
        "return_depth_reward_weight": args.return_depth_reward_weight,
        "net_proximity_reward_weight": args.net_proximity_reward_weight,
        "net_proximity_threshold": args.net_proximity_threshold,
        "eval_episodes": args.eval_episodes,
        "eval_freq": args.eval_freq,
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
        "recovery_x_margin": sim_config.action.recovery_x_margin,
        "recovery_net_margin": sim_config.action.recovery_net_margin,
        "recovery_back_margin": sim_config.action.recovery_back_margin,
        "conditional_recovery_grid": sim_config.action.conditional_recovery_grid,
        "intercept_count": args.intercept_count,
        "reaction_miss_fast_threshold": args.reaction_miss_fast_threshold,
        "reaction_miss_fast_probability": args.reaction_miss_fast_probability,
        "reaction_miss_secondary_threshold": args.reaction_miss_secondary_threshold,
        "reaction_miss_secondary_probability": args.reaction_miss_secondary_probability,
        "reaction_miss_zero_threshold": args.reaction_miss_zero_threshold,
        "include_reaction_risk_features": True,
        "phi_bins": args.phi_bins,
        "theta_bins": args.theta_bins,
        "speed_bins": args.speed_bins,
        "x_rec_bins": args.x_rec_bins,
        "y_rec_bins": args.y_rec_bins,
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
