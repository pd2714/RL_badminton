from __future__ import annotations

from dataclasses import dataclass

from stable_baselines3 import PPO

from badminton.action_space import DiscreteActionConfig
from badminton.config import SimulationConfig
from badminton.evaluation import ModelSelector, evaluate_selector
from badminton.opponents import make_opponent
from badminton.reset_sampling import ResetSamplingConfig
from badminton.rl_env import BadmintonRLEnv, RLEnvConfig, RewardConfig
from badminton.selfplay import CheckpointPool, FixedCheckpointOpponent, LiveModelOpponent


@dataclass(frozen=True)
class DiversityEvalConfig:
    train_side: str
    initial_server: str
    mirror_train_side: bool
    mirror_match_fraction: float
    sim_config: SimulationConfig
    reward_config: RewardConfig
    reset_sampling_config: ResetSamplingConfig
    train_reaction_time: float
    opponent_reaction_time: float
    episodes: int
    seed: int
    deterministic: bool = True


def run_diversity_evaluation(
    *,
    model: PPO,
    eval_config: DiversityEvalConfig,
    discrete_action_config: DiscreteActionConfig,
    checkpoint_pool: CheckpointPool | None = None,
) -> dict[str, object]:
    selector = ModelSelector(model=model, deterministic=eval_config.deterministic)
    summaries: dict[str, object] = {}
    current_matchup_summaries: dict[str, object] = {}

    heuristic_env = BadmintonRLEnv(
        config=eval_config.sim_config,
        rl_config=RLEnvConfig(
            train_side=eval_config.train_side,
            initial_server=eval_config.initial_server,
            mirror_train_side=eval_config.mirror_train_side,
            mirror_match_fraction=eval_config.mirror_match_fraction,
            train_reaction_time=eval_config.train_reaction_time,
            opponent_reaction_time=eval_config.opponent_reaction_time,
            reset_sampling=eval_config.reset_sampling_config,
            reward=eval_config.reward_config,
            recovery_counterfactual_other_sample_count=0,
            recovery_counterfactual_expected_response_target=False,
        ),
        discrete_action_config=discrete_action_config,
        opponent=make_opponent("safe", seed=eval_config.seed + 1),
        seed=eval_config.seed + 1,
    )
    heuristic_summary, _ = evaluate_selector("heuristic_safe", selector, heuristic_env, eval_config.episodes, eval_config.seed)
    summaries["heuristic"] = heuristic_summary

    mirror_env = BadmintonRLEnv(
        config=eval_config.sim_config,
        rl_config=RLEnvConfig(
            train_side=eval_config.train_side,
            initial_server=eval_config.initial_server,
            mirror_train_side=eval_config.mirror_train_side,
            mirror_match_fraction=eval_config.mirror_match_fraction,
            train_reaction_time=eval_config.train_reaction_time,
            opponent_reaction_time=eval_config.opponent_reaction_time,
            reset_sampling=eval_config.reset_sampling_config,
            reward=eval_config.reward_config,
            recovery_counterfactual_other_sample_count=0,
            recovery_counterfactual_expected_response_target=False,
        ),
        discrete_action_config=discrete_action_config,
        opponent=LiveModelOpponent(
            sim_config=eval_config.sim_config,
            discrete_action_config=discrete_action_config,
            model=model,
            deterministic=eval_config.deterministic,
            label_name="mirror_self",
        ),
        seed=eval_config.seed + 11,
    )
    mirror_summary, _ = evaluate_selector("current_vs_mirror_self", selector, mirror_env, eval_config.episodes, eval_config.seed + 11_000)
    summaries["mirror_self"] = mirror_summary
    current_matchup_summaries["current_vs_mirror_self"] = mirror_summary

    if checkpoint_pool is not None:
        checkpoint_pool.refresh()
        newest = checkpoint_pool.newest_path()
        if newest is not None:
            newest_env = BadmintonRLEnv(
                config=eval_config.sim_config,
                rl_config=RLEnvConfig(
                    train_side=eval_config.train_side,
                    initial_server=eval_config.initial_server,
                    mirror_train_side=eval_config.mirror_train_side,
                    mirror_match_fraction=eval_config.mirror_match_fraction,
                    train_reaction_time=eval_config.train_reaction_time,
                    opponent_reaction_time=eval_config.opponent_reaction_time,
                    reset_sampling=eval_config.reset_sampling_config,
                    reward=eval_config.reward_config,
                    recovery_counterfactual_other_sample_count=0,
                    recovery_counterfactual_expected_response_target=False,
                ),
                discrete_action_config=discrete_action_config,
                opponent=FixedCheckpointOpponent(
                    pool=checkpoint_pool,
                    checkpoint_path=newest,
                    sim_config=eval_config.sim_config,
                    discrete_action_config=discrete_action_config,
                ),
                seed=eval_config.seed + 3,
            )
            newest_summary, _ = evaluate_selector("current_vs_newest_checkpoint", selector, newest_env, eval_config.episodes, eval_config.seed + 3_000)
            summaries["newest_checkpoint"] = newest_summary
            current_matchup_summaries["current_vs_newest_checkpoint"] = newest_summary

    win_rates = [float(summary["win_rate"]) for summary in summaries.values() if isinstance(summary, dict) and "win_rate" in summary]
    narrow = False
    if win_rates:
        narrow = max(win_rates) - min(win_rates) > 0.25

    return {
        "summaries": summaries,
        "current_matchup_summaries": current_matchup_summaries,
        "narrow_opponent_dependency": narrow,
    }
