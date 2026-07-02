from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from badminton1d.action_space import DiscreteActionConfig
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.evaluation import ModelSelector, adapt_observation_to_model, rollout_episode, summarize_episodes
from badminton1d.factorized_ppo import RecoveryFactorizedPPO
from badminton1d.mpl_config import ensure_writable_matplotlib_config
from badminton1d.playback import MatchTrace, RallyTrace, build_rally_trace
from badminton1d.pressure import ShotPressureWeights, shot_pressure_from_stage_trace
from badminton1d.render import setup_court_axes, stage_colors
from badminton1d.selfplay import CheckpointPool, FixedCheckpointOpponent, LiveModelOpponent, build_selfplay_env
from badminton1d.shot_generators import name_velocity_shot
from badminton1d.state import Side
from badminton1d.utils import ensure_directory, side_y_bounds


_STEP_PATTERN = re.compile(r"(\d+)")
_REPO_ROOT = Path(__file__).resolve().parents[1]

SHOT_TYPE_ORDER = (
    "clear",
    "smash",
    "drive",
    "drop",
    "lift",
    "push",
    "net kill",
    "cross-court clear",
    "cross-court smash",
    "cross-court drive",
    "cross-court drop",
    "cross-court lift",
    "cross-court push",
    "cross-court net kill",
)
LANDING_ZONE_NAMES = tuple(
    f"{depth}_{lane}"
    for depth in ("front", "middle", "back")
    for lane in ("left", "center", "right")
)


@dataclass(frozen=True)
class AnchorEvaluationConfig:
    episodes: int = 24
    seed: int = 20260511
    deterministic: bool = False
    max_anchors: int | None = None
    anchor_stride: int = 1
    anchor_step_min: int | None = None
    anchor_step_max: int | None = None
    anchor_step_interval: int | None = None
    rating_pool_dir: Path | None = None
    pool_checkpoint_name: str = "final_model.zip"
    recovery_choice_diagnostics: bool = True
    speed_bins: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 90.0)
    height_bins: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


@dataclass(frozen=True)
class PoolAgentSpec:
    label: str
    step: int | None
    run_dir: Path
    model_path: Path


@dataclass(frozen=True)
class EvaluatedMatch:
    trace: MatchTrace
    recovery_diagnostics: list[dict[str, Any]]
    episode_summary: dict[str, Any]


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "selfplay_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selfplay config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_sim_config(data: dict[str, Any]) -> SimulationConfig:
    defaults = SimulationConfig()
    conditional_recovery_default = True if "conditional_recovery_grid" not in data else defaults.action.conditional_recovery_grid
    return SimulationConfig(
        court=CourtConfig(mode=str(_config_value(data, "court_mode", defaults.court.mode))),
        player=PlayerConfig(
            v_max=float(_config_value(data, "player_speed", defaults.player.v_max)),
            r_reach=float(_config_value(data, "racket_length", defaults.player.r_reach)),
            z_max=float(_config_value(data, "max_hitting_height", defaults.player.z_max)),
            acceleration=float(_config_value(data, "player_acceleration", defaults.player.acceleration)),
            deceleration=_config_value(data, "player_deceleration", defaults.player.deceleration),
            movement_model=str(_config_value(data, "movement_model", defaults.player.movement_model)),
        ),
        action=ActionConfig(
            trajectory_mode=str(_config_value(data, "trajectory_mode", defaults.action.trajectory_mode)),
            drag_coefficient=float(_config_value(data, "drag_coefficient", defaults.action.drag_coefficient)),
            horizontal_drag_coefficient=float(
                _config_value(data, "horizontal_drag_coefficient", defaults.action.horizontal_drag_coefficient)
            ),
            vertical_drag_coefficient=float(
                _config_value(data, "vertical_drag_coefficient", defaults.action.vertical_drag_coefficient)
            ),
            vy_min_forward=float(_config_value(data, "shuttle_speed_min", defaults.action.vy_min_forward)),
            vy_max_forward=float(_config_value(data, "shuttle_speed_max", defaults.action.vy_max_forward)),
            intercept_count=int(_config_value(data, "intercept_count", defaults.action.intercept_count)),
            reaction_miss_fast_threshold=float(
                _config_value(data, "reaction_miss_fast_threshold", defaults.action.reaction_miss_fast_threshold)
            ),
            reaction_miss_fast_probability=float(
                _config_value(data, "reaction_miss_fast_probability", defaults.action.reaction_miss_fast_probability)
            ),
            reaction_miss_secondary_threshold=float(
                _config_value(
                    data,
                    "reaction_miss_secondary_threshold",
                    defaults.action.reaction_miss_secondary_threshold,
                )
            ),
            reaction_miss_secondary_probability=float(
                _config_value(
                    data,
                    "reaction_miss_secondary_probability",
                    defaults.action.reaction_miss_secondary_probability,
                )
            ),
            reaction_miss_zero_threshold=float(
                _config_value(data, "reaction_miss_zero_threshold", defaults.action.reaction_miss_zero_threshold)
            ),
            conditional_recovery_grid=bool(
                _config_value(data, "conditional_recovery_grid", conditional_recovery_default)
            ),
        ),
    )


def build_discrete_action_config(data: dict[str, Any]) -> DiscreteActionConfig:
    defaults = DiscreteActionConfig()
    return DiscreteActionConfig(
        phi_bins=int(_config_value(data, "phi_bins", "vx_bins", defaults.phi_bins)),
        theta_bins=int(_config_value(data, "theta_bins", "vy_bins", defaults.theta_bins)),
        speed_bins=int(_config_value(data, "speed_bins", "vz_bins", defaults.speed_bins)),
        x_rec_bins=int(_config_value(data, "x_rec_bins", defaults.x_rec_bins)),
        y_rec_bins=int(_config_value(data, "y_rec_bins", defaults.y_rec_bins)),
    )


def discover_anchor_checkpoints(run_dir: Path) -> list[Path]:
    config = load_run_config(run_dir)
    configured = Path(str(config.get("anchor_checkpoint_dir") or "anchor_checkpoints"))
    candidates = [configured]
    if not configured.is_absolute():
        candidates.extend([run_dir / configured, _REPO_ROOT / configured, run_dir / "anchor_checkpoints"])
    checkpoint_dir = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    checkpoints = sorted(checkpoint_dir.glob("anchor_step_*.zip"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No anchor_step_*.zip checkpoints found in {checkpoint_dir}")
    return checkpoints


def discover_rating_pool_agents(pool_dir: Path, checkpoint_name: str = "final_model.zip") -> list[PoolAgentSpec]:
    manifest_path = pool_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("agents", [])
        if not isinstance(entries, list):
            raise ValueError(f"Invalid agents list in {manifest_path}")
        raw_entries = [
            {
                "run_dir": pool_dir / str(entry["run_dir"]),
                "label": str(entry.get("label") or Path(str(entry["run_dir"])).name),
                "step": entry.get("step"),
            }
            for entry in entries
        ]
    else:
        raw_entries = [
            {"run_dir": path, "label": path.name, "step": _step_from_label(path.name)}
            for path in sorted((path for path in pool_dir.iterdir() if path.is_dir()), key=lambda path: (_step_from_label(path.name) or -1, path.name))
        ]

    agents: list[PoolAgentSpec] = []
    seen: set[str] = set()
    for entry in raw_entries:
        run_dir = Path(entry["run_dir"])
        label = str(entry["label"])
        model_path = run_dir / checkpoint_name
        if label in seen:
            raise ValueError(f"Duplicate rating-pool label: {label}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing rating-pool checkpoint: {model_path}")
        if not (run_dir / "selfplay_config.json").exists():
            raise FileNotFoundError(f"Missing rating-pool config: {run_dir / 'selfplay_config.json'}")
        raw_step = entry.get("step")
        step = int(raw_step) if raw_step is not None else _step_from_label(label)
        agents.append(PoolAgentSpec(label=label, step=step, run_dir=run_dir, model_path=model_path))
        seen.add(label)

    if not agents:
        raise FileNotFoundError(f"No rating-pool agents found in {pool_dir}")
    return agents


def checkpoint_step(path: Path) -> int:
    matches = _STEP_PATTERN.findall(path.stem)
    return int(matches[-1]) if matches else -1


def _step_from_label(label: str) -> int | None:
    matches = _STEP_PATTERN.findall(label)
    return int(matches[-1]) if matches else None


def filter_anchor_checkpoints(
    checkpoints: list[Path],
    *,
    step_min: int | None,
    step_max: int | None,
    step_interval: int | None,
) -> list[Path]:
    filtered: list[Path] = []
    interval_base = step_min if step_min is not None else 0
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        if step_interval is not None and (step - interval_base) % step_interval != 0:
            continue
        filtered.append(checkpoint)
    return filtered


def load_anchor_model(checkpoint: Path, *, recovery_choice_diagnostics: bool) -> PPO:
    if recovery_choice_diagnostics:
        try:
            return RecoveryFactorizedPPO.load(checkpoint)
        except Exception:
            pass
    return PPO.load(checkpoint)


def model_supports_recovery_choice_diagnostics(model: PPO) -> bool:
    return (
        hasattr(model, "_recovery_advantage_from_transitions")
        and hasattr(model.policy, "predict_values")
        and getattr(model.policy, "output_mode", None) == "conditional_prob"
        and not isinstance(model.action_space, spaces.Box)
    )


def rollout_episode_with_recovery_diagnostics(
    env: Any,
    model: PPO,
    *,
    deterministic: bool,
    seed: int | None = None,
) -> dict[str, object]:
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    selector = ModelSelector(model=model, deterministic=deterministic)
    recovery_diagnostics: list[dict[str, Any]] = []
    use_recovery_diagnostics = model_supports_recovery_choice_diagnostics(model)

    while not terminated and not truncated:
        current_observation = observation
        action = selector.choose_action(current_observation, env.current_decision_context())
        obs_tensor = None
        action_tensor = None
        values_before = None
        if use_recovery_diagnostics:
            adapted_observation = adapt_observation_to_model(model, current_observation)
            obs_tensor, _ = model.policy.obs_to_tensor(adapted_observation)
            with torch.no_grad():
                values_before = model.policy.predict_values(obs_tensor).flatten()
            action_tensor = torch.as_tensor([int(action)], dtype=torch.long, device=model.device)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if (
            use_recovery_diagnostics
            and values_before is not None
            and obs_tensor is not None
            and action_tensor is not None
        ):
            adapted_next_observation = adapt_observation_to_model(model, observation)
            try:
                model._recovery_advantage_from_transitions(  # type: ignore[attr-defined]
                    values_before,
                    [info],
                    np.asarray([adapted_next_observation], dtype=np.float32),
                    np.asarray([reward], dtype=np.float32),
                    np.asarray([terminated or truncated], dtype=bool),
                    obs_tensor,
                    action_tensor,
                )
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
            diagnostic = info.get("recovery_factorized_diagnostics")
            if isinstance(diagnostic, dict):
                recovery_diagnostics.append(diagnostic)

    metrics = info["badminton_metrics"]
    return {
        "reward": total_reward,
        "winner": metrics["winner"],
        "rally_won": metrics["rally_won"],
        "rally_length": metrics["rally_length"],
        "invalid_action_rate": metrics["invalid_action_rate"],
        "metrics": metrics,
        "records": list(env.records),
        "config": env.config,
        "server": info["server"],
        "train_side": info["train_side"],
        "opponent_label": info.get("opponent_label"),
        "truncated": truncated,
        "recovery_diagnostics": recovery_diagnostics,
    }


def rollout_anchor_match_trace(
    *,
    model: PPO,
    run_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    episodes: int,
    seed: int,
    deterministic: bool,
    recovery_choice_diagnostics: bool,
) -> EvaluatedMatch:
    policy_type = str(_config_value(run_config, "policy_type", "velocity_oriented"))
    opponent = LiveModelOpponent(
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        model=model,
        policy_type=policy_type,
        deterministic=deterministic,
        label_name="mirror_self",
    )
    env = build_selfplay_env(
        train_side=str(_config_value(run_config, "train_side", "left")),  # type: ignore[arg-type]
        mirror_train_side=bool(_config_value(run_config, "mirror_sides", False)),
        mirror_match_fraction=float(_config_value(run_config, "mirror_match_fraction", 0.0)),
        initial_server=str(_config_value(run_config, "initial_server", "random")),
        random_service_x=bool(_config_value(run_config, "random_service_x", True)),
        sim_config=sim_config,
        train_reaction_time=float(_config_value(run_config, "reaction_time", 0.15)),
        opponent_reaction_time=float(_config_value(run_config, "opponent_reaction_time", 0.15)),
        max_stages_per_rally=int(_config_value(run_config, "max_rally_stages", 120)),
        policy_type=policy_type,
        seed=seed,
        discrete_action_config=discrete_action_config,
        opponent=opponent,
        include_records_in_info=True,
        recovery_counterfactual_other_sample_count=24 if recovery_choice_diagnostics else 0,
        recovery_counterfactual_expected_response_target=bool(recovery_choice_diagnostics),
        recovery_full_diagnostics_probability=1.0 if recovery_choice_diagnostics else 0.0,
    )
    rallies: list[RallyTrace] = []
    recovery_diagnostics: list[dict[str, Any]] = []
    results: list[dict[str, object]] = []
    score_left = 0
    score_right = 0
    for episode in range(episodes):
        if recovery_choice_diagnostics:
            result = rollout_episode_with_recovery_diagnostics(
                env,
                model,
                deterministic=deterministic,
                seed=seed + episode,
            )
        else:
            selector = ModelSelector(model=model, deterministic=deterministic)
            result = rollout_episode(env, selector, seed=seed + episode)
        if result["winner"] == "left":
            score_left += 1
        elif result["winner"] == "right":
            score_right += 1
        recovery_diagnostics.extend(result.get("recovery_diagnostics", []))  # type: ignore[arg-type]
        records = result["records"]
        assert isinstance(records, list)
        trace = build_rally_trace(records, sim_config)
        results.append(result)
        rallies.append(
            RallyTrace(
                stages=trace.stages,
                rally_done=trace.rally_done,
                winner=trace.winner,
                total_playback_time=trace.total_playback_time,
                rally_number=episode + 1,
                server=str(result["server"]),  # type: ignore[arg-type]
                score_before_left=score_left - (1 if result["winner"] == "left" else 0),
                score_before_right=score_right - (1 if result["winner"] == "right" else 0),
                score_after_left=score_left,
                score_after_right=score_right,
            )
        )
    winner: Side | None = None
    if score_left != score_right:
        winner = "left" if score_left > score_right else "right"
    return EvaluatedMatch(
        trace=MatchTrace(
            rallies=rallies,
            target_score=max(score_left, score_right),
            score_left=score_left,
            score_right=score_right,
            winner=winner,
            total_playback_time=sum(rally.total_playback_time for rally in rallies),
        ),
        recovery_diagnostics=recovery_diagnostics,
        episode_summary=summarize_episodes(results),
    )


def rollout_fixed_pool_match_trace(
    *,
    model: PPO,
    opponent: PoolAgentSpec,
    run_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    episodes: int,
    seed: int,
    deterministic: bool,
    recovery_choice_diagnostics: bool,
) -> EvaluatedMatch:
    policy_type = str(_config_value(run_config, "policy_type", "velocity_oriented"))
    pool = CheckpointPool(
        checkpoint_dir=opponent.model_path.parent,
        pool_size=1,
        sampling_mode="newest",
        seed=seed + 17,
    )
    opponent_policy = FixedCheckpointOpponent(
        pool=pool,
        checkpoint_path=opponent.model_path,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        policy_type=policy_type,
        deterministic=deterministic,
    )
    env = build_selfplay_env(
        train_side=str(_config_value(run_config, "train_side", "left")),  # type: ignore[arg-type]
        mirror_train_side=False,
        mirror_match_fraction=0.0,
        initial_server=str(_config_value(run_config, "initial_server", "random")),
        random_service_x=bool(_config_value(run_config, "random_service_x", True)),
        sim_config=sim_config,
        train_reaction_time=float(_config_value(run_config, "reaction_time", 0.15)),
        opponent_reaction_time=float(_config_value(run_config, "opponent_reaction_time", 0.15)),
        max_stages_per_rally=int(_config_value(run_config, "max_rally_stages", 120)),
        policy_type=policy_type,
        seed=seed,
        discrete_action_config=discrete_action_config,
        opponent=opponent_policy,
        include_records_in_info=True,
        recovery_counterfactual_other_sample_count=24 if recovery_choice_diagnostics else 0,
        recovery_counterfactual_expected_response_target=bool(recovery_choice_diagnostics),
        recovery_full_diagnostics_probability=1.0 if recovery_choice_diagnostics else 0.0,
    )
    rallies: list[RallyTrace] = []
    recovery_diagnostics: list[dict[str, Any]] = []
    results: list[dict[str, object]] = []
    score_left = 0
    score_right = 0
    for episode in range(episodes):
        if recovery_choice_diagnostics:
            result = rollout_episode_with_recovery_diagnostics(
                env,
                model,
                deterministic=deterministic,
                seed=seed + episode,
            )
        else:
            selector = ModelSelector(model=model, deterministic=deterministic)
            result = rollout_episode(env, selector, seed=seed + episode)
        if result["winner"] == "left":
            score_left += 1
        elif result["winner"] == "right":
            score_right += 1
        recovery_diagnostics.extend(result.get("recovery_diagnostics", []))  # type: ignore[arg-type]
        records = result["records"]
        assert isinstance(records, list)
        trace = build_rally_trace(records, sim_config)
        results.append(result)
        rallies.append(
            RallyTrace(
                stages=trace.stages,
                rally_done=trace.rally_done,
                winner=trace.winner,
                total_playback_time=trace.total_playback_time,
                rally_number=episode + 1,
                server=str(result["server"]),  # type: ignore[arg-type]
                score_before_left=score_left - (1 if result["winner"] == "left" else 0),
                score_before_right=score_right - (1 if result["winner"] == "right" else 0),
                score_after_left=score_left,
                score_after_right=score_right,
            )
        )
    winner: Side | None = None
    if score_left != score_right:
        winner = "left" if score_left > score_right else "right"
    return EvaluatedMatch(
        trace=MatchTrace(
            rallies=rallies,
            target_score=max(score_left, score_right),
            score_left=score_left,
            score_right=score_right,
            winner=winner,
            total_playback_time=sum(rally.total_playback_time for rally in rallies),
        ),
        recovery_diagnostics=recovery_diagnostics,
        episode_summary=summarize_episodes(results),
    )


def combine_match_traces(traces: list[MatchTrace]) -> MatchTrace:
    rallies: list[RallyTrace] = []
    score_left = 0
    score_right = 0
    for trace in traces:
        for rally in trace.rallies:
            rallies.append(
                RallyTrace(
                    stages=rally.stages,
                    rally_done=rally.rally_done,
                    winner=rally.winner,
                    total_playback_time=rally.total_playback_time,
                    rally_number=len(rallies) + 1,
                    server=rally.server,
                    score_before_left=score_left,
                    score_before_right=score_right,
                    score_after_left=score_left + (1 if rally.winner == "left" else 0),
                    score_after_right=score_right + (1 if rally.winner == "right" else 0),
                    pause_duration=rally.pause_duration,
                    match_winner=rally.match_winner,
                )
            )
            if rally.winner == "left":
                score_left += 1
            elif rally.winner == "right":
                score_right += 1
    winner: Side | None = None
    if score_left != score_right:
        winner = "left" if score_left > score_right else "right"
    return MatchTrace(
        rallies=rallies,
        target_score=max(score_left, score_right),
        score_left=score_left,
        score_right=score_right,
        winner=winner,
        total_playback_time=sum(rally.total_playback_time for rally in rallies),
    )


def summarize_recovery_choice_diagnostics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "recovery_counterfactual_count": 0,
            "recovery_grid_samples": [],
        }

    rank = [float(sample.get("chosen_rank", 0.0)) for sample in samples]
    rank_fraction = [float(sample.get("chosen_rank_fraction", 0.0)) for sample in samples]
    above_average = [1.0 if sample.get("chosen_above_average") else 0.0 for sample in samples]
    best = [1.0 if sample.get("chosen_best") else 0.0 for sample in samples]
    a_rec = [float(sample.get("a_rec", 0.0)) for sample in samples]
    training_advantage = [float(sample.get("training_recovery_advantage", 0.0)) for sample in samples]
    chosen_probability = [
        float(sample["chosen_probability"])
        for sample in samples
        if sample.get("chosen_probability") is not None
    ]

    no_feasible_counts: np.ndarray | None = None
    bin_counts: np.ndarray | None = None
    x_count = 0
    y_count = 0
    grid_samples: list[dict[str, Any]] = []
    for sample in samples:
        no_feasible_grid = np.asarray(sample.get("no_feasible_grid", []), dtype=np.float64)
        if no_feasible_grid.ndim == 2 and no_feasible_grid.size > 0:
            flat = no_feasible_grid.reshape(-1)
            if no_feasible_counts is None or no_feasible_counts.shape != flat.shape:
                no_feasible_counts = np.zeros_like(flat)
                bin_counts = np.zeros_like(flat)
            no_feasible_counts += flat
            assert bin_counts is not None
            bin_counts += 1.0
            x_count = int(no_feasible_grid.shape[0])
            y_count = int(no_feasible_grid.shape[1])
        if len(grid_samples) < 24 and isinstance(sample.get("score_grid"), list):
            grid_samples.append(
                {
                    "chosen_flat_index": int(sample.get("chosen_flat_index", -1)),
                    "chosen_x_index": int(sample.get("chosen_x_index", -1)),
                    "chosen_y_index": int(sample.get("chosen_y_index", -1)),
                    "chosen_rank": int(sample.get("chosen_rank", 0)),
                    "chosen_above_average": bool(sample.get("chosen_above_average", False)),
                    "chosen_best": bool(sample.get("chosen_best", False)),
                    "a_rec": float(sample.get("a_rec", 0.0)),
                    "score_grid": sample.get("score_grid", []),
                    "policy_probability_grid": sample.get("policy_probability_grid", []),
                    "no_feasible_grid": sample.get("no_feasible_grid", []),
                }
            )

    payload: dict[str, Any] = {
        "recovery_counterfactual_count": len(samples),
        "recovery_chosen_mean_rank": _mean(rank),
        "recovery_chosen_mean_rank_fraction": _mean(rank_fraction),
        "recovery_chosen_above_average_fraction": _mean(above_average),
        "recovery_chosen_best_fraction": _mean(best),
        "recovery_a_rec_mean": _mean(a_rec),
        "recovery_a_rec_std": _std(a_rec),
        "recovery_a_rec_min": _min(a_rec),
        "recovery_a_rec_max": _max(a_rec),
        "recovery_training_advantage_mean": _mean(training_advantage),
        "recovery_training_advantage_std": _std(training_advantage),
        "recovery_training_advantage_min": _min(training_advantage),
        "recovery_training_advantage_max": _max(training_advantage),
        "recovery_chosen_probability_mean": _mean(chosen_probability),
        "recovery_grid_samples": grid_samples,
    }
    if no_feasible_counts is not None and bin_counts is not None and x_count > 0 and y_count > 0:
        rates = np.divide(no_feasible_counts, np.maximum(bin_counts, 1.0))
        payload["recovery_no_feasible_count_by_bin"] = no_feasible_counts.astype(float).tolist()
        payload["recovery_no_feasible_rate_by_bin"] = rates.astype(float).tolist()
        payload["recovery_no_feasible_rate_grid"] = rates.reshape(x_count, y_count).astype(float).tolist()
    return payload


def summarize_match_trace_metrics(
    trace: MatchTrace,
    config: SimulationConfig,
    *,
    speed_bins: tuple[float, ...],
    height_bins: tuple[float, ...],
    pressure_weights: ShotPressureWeights | None = None,
    include_spatial_samples: bool = True,
) -> dict[str, Any]:
    shot_types: list[str] = []
    landing_zones: list[str] = []
    speeds: list[float] = []
    contact_heights: list[float] = []
    pressures: list[float] = []
    pressure_required_speed: list[float] = []
    pressure_scarcity: list[float] = []
    pressure_low_contact: list[float] = []
    pressure_reaction_miss: list[float] = []
    recovery_x: list[float] = []
    recovery_y: list[float] = []
    next_hit_x: list[float] = []
    next_hit_y: list[float] = []
    recovery_landing_x: list[float] = []
    recovery_landing_y: list[float] = []
    intended_landing_x: list[float] = []
    intended_landing_y: list[float] = []

    for rally in trace.rallies:
        for stage_index, stage in enumerate(rally.stages):
            velocity = np.asarray(stage.shuttle_velocity, dtype=float)
            speed = float(np.linalg.norm(velocity))
            theta = math.degrees(math.atan2(float(stage.shuttle_velocity[2]), float(np.hypot(*stage.shuttle_velocity[:2]))))
            shot_types.append(
                name_velocity_shot(
                    hitter=stage.hitter_side,
                    contact_x=float(stage.shuttle_start[0]),
                    contact_y=float(stage.shuttle_start[1]),
                    landing_x=float(stage.shuttle_landing[0]),
                    landing_y=float(stage.shuttle_landing[1]),
                    theta_degrees=theta,
                    config=config,
                )
            )
            landing_zones.append(landing_zone_name(stage.receiver_side, stage.shuttle_landing, config))
            speeds.append(speed)
            contact_heights.append(float(stage.shuttle_start[2]))
            pressure = shot_pressure_from_stage_trace(stage, config, weights=pressure_weights)
            pressures.append(float(pressure.pressure))
            pressure_required_speed.append(float(pressure.required_speed_score))
            pressure_scarcity.append(float(pressure.intercept_scarcity_score))
            pressure_low_contact.append(float(pressure.low_contact_score))
            pressure_reaction_miss.append(float(pressure.reaction_miss_score))
            recovery_landing_x.append(float(stage.recovery_target[0]))
            recovery_landing_y.append(float(stage.recovery_target[1]))
            intended_landing_x.append(float(stage.shuttle_landing[0]))
            intended_landing_y.append(float(stage.shuttle_landing[1]))

            if stage_index + 1 < len(rally.stages):
                next_stage = rally.stages[stage_index + 1]
                recovery_x.append(float(stage.recovery_target[0]))
                recovery_y.append(float(stage.recovery_target[1]))
                next_hit_x.append(float(next_stage.shuttle_start[0]))
                next_hit_y.append(float(next_stage.shuttle_start[1]))

    shot_counter = Counter(shot_types)
    landing_counter = Counter(landing_zones)
    ordered_shot_names = _ordered_names(SHOT_TYPE_ORDER, shot_counter)
    ordered_zone_names = _ordered_names(LANDING_ZONE_NAMES, landing_counter)
    speed_hist = np.histogram(np.asarray(speeds, dtype=float), bins=np.asarray(speed_bins, dtype=float))[0]
    height_hist = np.histogram(np.asarray(contact_heights, dtype=float), bins=np.asarray(height_bins, dtype=float))[0]
    next_hit_distances = np.hypot(
        np.asarray(recovery_x) - np.asarray(next_hit_x),
        np.asarray(recovery_y) - np.asarray(next_hit_y),
    )
    landing_distances = np.hypot(
        np.asarray(recovery_landing_x) - np.asarray(intended_landing_x),
        np.asarray(recovery_landing_y) - np.asarray(intended_landing_y),
    )

    metrics = {
        "rally_count": len(trace.rallies),
        "shot_count": len(speeds),
        "shot_type_names": ordered_shot_names,
        "shot_type_counts": {name: int(shot_counter.get(name, 0)) for name in ordered_shot_names},
        "shot_type_frequency": _frequency_dict(ordered_shot_names, shot_counter),
        "shot_type_entropy": entropy_bits([shot_counter.get(name, 0) for name in ordered_shot_names]),
        "shot_type_entropy_normalized": normalized_entropy([shot_counter.get(name, 0) for name in ordered_shot_names]),
        "landing_zone_names": ordered_zone_names,
        "landing_zone_counts": {name: int(landing_counter.get(name, 0)) for name in ordered_zone_names},
        "landing_zone_frequency": _frequency_dict(ordered_zone_names, landing_counter),
        "landing_zone_entropy": entropy_bits([landing_counter.get(name, 0) for name in ordered_zone_names]),
        "landing_zone_entropy_normalized": normalized_entropy([landing_counter.get(name, 0) for name in ordered_zone_names]),
        "shot_speed_mean": _mean(speeds),
        "shot_speed_std": _std(speeds),
        "shot_speed_hist": speed_hist.astype(int).tolist(),
        "shot_speed_bins": list(speed_bins),
        "shot_height_mean": _mean(contact_heights),
        "shot_height_std": _std(contact_heights),
        "shot_height_hist": height_hist.astype(int).tolist(),
        "shot_height_bins": list(height_bins),
        "pressure_mean": _mean(pressures),
        "pressure_max": _max(pressures),
        "pressure_required_speed_score_mean": _mean(pressure_required_speed),
        "pressure_intercept_scarcity_score_mean": _mean(pressure_scarcity),
        "pressure_low_contact_score_mean": _mean(pressure_low_contact),
        "pressure_reaction_miss_score_mean": _mean(pressure_reaction_miss),
        "recovery_next_hit_pair_count": len(recovery_x),
        "recovery_next_hit_x_corr": pearson_or_none(recovery_x, next_hit_x),
        "recovery_next_hit_y_corr": pearson_or_none(recovery_y, next_hit_y),
        "recovery_next_hit_distance_mean": None if next_hit_distances.size == 0 else float(np.mean(next_hit_distances)),
        "recovery_intended_landing_pair_count": len(intended_landing_x),
        "recovery_intended_landing_x_corr": pearson_or_none(recovery_landing_x, intended_landing_x),
        "recovery_intended_landing_y_corr": pearson_or_none(recovery_landing_y, intended_landing_y),
        "recovery_intended_landing_distance_mean": None if landing_distances.size == 0 else float(np.mean(landing_distances)),
    }
    if include_spatial_samples:
        metrics["landing_samples_xy"] = [
            [float(x), float(y)] for x, y in zip(intended_landing_x, intended_landing_y)
        ]
        metrics["recovery_samples_xy"] = [
            [float(x), float(y)] for x, y in zip(recovery_landing_x, recovery_landing_y)
        ]
    return metrics


def summarize_tactical_match_metrics(
    trace: MatchTrace,
    config: SimulationConfig,
    *,
    pressure_weights: ShotPressureWeights | None = None,
) -> dict[str, Any]:
    """Summarize state-conditioned tactical signals from a match trace.

    Shot value is an empirical delta in rally win probability. The value model is
    estimated from this trace by tactical state bucket, so it is descriptive
    rather than a learned critic/Q estimate.
    """
    samples: list[dict[str, Any]] = []
    bucket_outcomes: dict[str, list[float]] = {}

    for rally in trace.rallies:
        stage_infos = _tactical_stage_infos(rally, config, pressure_weights=pressure_weights)
        for info in stage_infos:
            outcome = 1.0 if rally.winner == info["hitter_side"] else 0.0
            bucket = str(info["state_bucket"])
            bucket_outcomes.setdefault(bucket, []).append(outcome)

    all_outcomes = [value for values in bucket_outcomes.values() for value in values]
    global_value = _mean(all_outcomes)
    bucket_values = {
        bucket: float(np.mean(np.asarray(values, dtype=float)))
        for bucket, values in bucket_outcomes.items()
        if values
    }

    for rally in trace.rallies:
        stage_infos = _tactical_stage_infos(rally, config, pressure_weights=pressure_weights)
        for index, info in enumerate(stage_infos):
            hitter = str(info["hitter_side"])
            before_value = bucket_values.get(str(info["state_bucket"]), global_value)
            after_value: float | None = None
            if info["terminal"]:
                after_value = 1.0 if rally.winner == hitter else 0.0
            elif index + 1 < len(stage_infos):
                next_bucket = str(stage_infos[index + 1]["state_bucket"])
                next_hitter_value = bucket_values.get(next_bucket, global_value)
                if next_hitter_value is not None:
                    after_value = 1.0 - float(next_hitter_value)

            shot_value_delta = None
            if before_value is not None and after_value is not None:
                shot_value_delta = float(after_value - float(before_value))

            next_info = stage_infos[index + 1] if index + 1 < len(stage_infos) else None
            next_own_info = stage_infos[index + 2] if index + 2 < len(stage_infos) else None
            forced_weak_return = None
            recovery_success = None
            opponent_can_smash = False
            if next_info is not None and str(next_info["hitter_side"]) != hitter:
                forced_weak_return = bool(
                    float(next_info["pressure"]) <= 0.35
                    or float(next_info["contact_height"]) <= 1.05
                )
                opponent_can_smash = bool(
                    "smash" in str(next_info["shot_type"])
                    or float(next_info["contact_height"]) >= 1.55
                )
                if bool(next_info["terminal"]):
                    recovery_success = rally.winner == hitter
                elif next_own_info is not None and str(next_own_info["hitter_side"]) == hitter:
                    recovery_success = True
                else:
                    recovery_success = False

            defense_to_attack = None
            attack_retention = None
            if next_own_info is not None and str(next_own_info["hitter_side"]) == hitter:
                if info["state_bucket"] == "defensive_state":
                    defense_to_attack = next_own_info["state_bucket"] == "attacking_state"
                if info["state_bucket"] == "attacking_state":
                    attack_retention = next_own_info["state_bucket"] == "attacking_state"

            samples.append(
                {
                    **info,
                    "shot_value_delta": shot_value_delta,
                    "forced_weak_return": forced_weak_return,
                    "recovery_success": recovery_success,
                    "opponent_can_smash": opponent_can_smash,
                    "defense_to_attack": defense_to_attack,
                    "attack_retention": attack_retention,
                }
            )

    by_context = {
        "all": _summarize_tactical_samples(samples),
        "attacking_state": _summarize_tactical_samples(
            [sample for sample in samples if sample["state_bucket"] == "attacking_state"]
        ),
        "neutral_state": _summarize_tactical_samples(
            [sample for sample in samples if sample["state_bucket"] == "neutral_state"]
        ),
        "defensive_state": _summarize_tactical_samples(
            [sample for sample in samples if sample["state_bucket"] == "defensive_state"]
        ),
        "opponent_can_smash": _summarize_tactical_samples(
            [sample for sample in samples if sample["opponent_can_smash"]]
        ),
    }
    return {
        "definition": (
            "Empirical state-conditioned tactical metrics. Shot value delta uses "
            "trace-level bucketed rally win probability as a proxy for V(s)."
        ),
        "state_bucket_values": bucket_values,
        "by_context": by_context,
        **_flatten_tactical_metrics(by_context),
    }


def evaluate_anchor_folder(
    run_dir: Path,
    output_dir: Path,
    eval_config: AnchorEvaluationConfig,
) -> dict[str, Any]:
    ensure_directory(output_dir)
    run_config = load_run_config(run_dir)
    sim_config = build_sim_config(run_config)
    discrete_action_config = build_discrete_action_config(run_config)
    checkpoints = discover_anchor_checkpoints(run_dir)
    checkpoints = filter_anchor_checkpoints(
        checkpoints,
        step_min=eval_config.anchor_step_min,
        step_max=eval_config.anchor_step_max,
        step_interval=eval_config.anchor_step_interval,
    )
    checkpoints = checkpoints[:: max(int(eval_config.anchor_stride), 1)]
    if eval_config.max_anchors is not None:
        checkpoints = checkpoints[: max(int(eval_config.max_anchors), 0)]
    pool_agents = (
        discover_rating_pool_agents(eval_config.rating_pool_dir, eval_config.pool_checkpoint_name)
        if eval_config.rating_pool_dir is not None
        else None
    )

    rows: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints):
        print(f"evaluating anchor_step_{checkpoint_step(checkpoint)}", flush=True)
        model = load_anchor_model(checkpoint, recovery_choice_diagnostics=eval_config.recovery_choice_diagnostics)
        pool_matchups: list[dict[str, Any]] = []
        recovery_diagnostics: list[dict[str, Any]] = []
        if pool_agents is None:
            evaluated = rollout_anchor_match_trace(
                model=model,
                run_config=run_config,
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                episodes=eval_config.episodes,
                seed=eval_config.seed + index * 10_000,
                deterministic=eval_config.deterministic,
                recovery_choice_diagnostics=eval_config.recovery_choice_diagnostics,
            )
            trace = evaluated.trace
            recovery_diagnostics = evaluated.recovery_diagnostics
        else:
            traces = []
            for pool_index, opponent in enumerate(pool_agents):
                print(f"  vs {opponent.label} ({eval_config.episodes} rallies)", flush=True)
                matchup = rollout_fixed_pool_match_trace(
                    model=model,
                    opponent=opponent,
                    run_config=run_config,
                    sim_config=sim_config,
                    discrete_action_config=discrete_action_config,
                    episodes=eval_config.episodes,
                    seed=eval_config.seed + index * 100_000 + pool_index * 1_000,
                    deterministic=eval_config.deterministic,
                    recovery_choice_diagnostics=eval_config.recovery_choice_diagnostics,
                )
                traces.append(matchup.trace)
                recovery_diagnostics.extend(matchup.recovery_diagnostics)
                pool_matchups.append(
                    {
                        "opponent": opponent.label,
                        "opponent_step": opponent.step,
                        "opponent_model_path": str(opponent.model_path),
                        **match_score_metrics(
                            matchup.trace,
                            train_side=str(_config_value(run_config, "train_side", "left")),  # type: ignore[arg-type]
                        ),
                        **summarize_match_trace_metrics(
                            matchup.trace,
                            sim_config,
                            speed_bins=eval_config.speed_bins,
                            height_bins=eval_config.height_bins,
                            include_spatial_samples=False,
                        ),
                        **summarize_recovery_choice_diagnostics(matchup.recovery_diagnostics),
                    }
                )
            trace = combine_match_traces(traces)
        metrics = summarize_match_trace_metrics(
            trace,
            sim_config,
            speed_bins=eval_config.speed_bins,
            height_bins=eval_config.height_bins,
        )
        tactical_metrics = summarize_tactical_match_metrics(trace, sim_config)
        rows.append(
            {
                "step": checkpoint_step(checkpoint),
                "checkpoint_path": str(checkpoint),
                "pool_matchups": pool_matchups,
                **summarize_recovery_choice_diagnostics(recovery_diagnostics),
                **metrics,
                "tactical_metrics": tactical_metrics,
                **_flatten_row_tactical_metrics(tactical_metrics),
            }
        )

    report = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "episodes_per_anchor": eval_config.episodes,
        "seed": eval_config.seed,
        "deterministic": eval_config.deterministic,
        "anchor_stride": eval_config.anchor_stride,
        "anchor_step_min": eval_config.anchor_step_min,
        "anchor_step_max": eval_config.anchor_step_max,
        "anchor_step_interval": eval_config.anchor_step_interval,
        "anchor_count": len(rows),
        "rating_pool_dir": None if eval_config.rating_pool_dir is None else str(eval_config.rating_pool_dir),
        "pool_checkpoint_name": eval_config.pool_checkpoint_name,
        "pool_agent_count": 0 if pool_agents is None else len(pool_agents),
        "recovery_choice_diagnostics": eval_config.recovery_choice_diagnostics,
        "pool_agents": []
        if pool_agents is None
        else [
            {
                "label": agent.label,
                "step": agent.step,
                "run_dir": str(agent.run_dir),
                "model_path": str(agent.model_path),
            }
            for agent in pool_agents
        ],
        "speed_bins": list(eval_config.speed_bins),
        "height_bins": list(eval_config.height_bins),
        "rows": rows,
    }
    report_path = output_dir / "anchor_metric_evolution.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_metric_csv(output_dir / "anchor_metric_evolution.csv", rows)
    write_tactical_metric_csv(output_dir / "tactical_metrics_by_context.csv", rows)
    plot_metric_evolution(report, output_dir, sim_config)
    plot_paths = dict(report.get("plots", {}))
    plot_paths.update(write_fixed_pool_rating_outputs(report, output_dir))
    report["plots"] = plot_paths
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def match_score_metrics(trace: MatchTrace, *, train_side: Side) -> dict[str, Any]:
    if train_side == "right":
        agent_score = int(trace.score_right)
        opponent_score = int(trace.score_left)
    else:
        agent_score = int(trace.score_left)
        opponent_score = int(trace.score_right)
    rally_count = agent_score + opponent_score
    return {
        "agent_side": train_side,
        "score_agent": agent_score,
        "score_opponent": opponent_score,
        "agent_win_rate": None if rally_count <= 0 else float(agent_score / rally_count),
        "opponent_win_rate": None if rally_count <= 0 else float(opponent_score / rally_count),
        "match_winner": trace.winner,
    }


def write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    scalar_keys = [
        "step",
        "rally_count",
        "shot_count",
        "shot_type_entropy",
        "shot_type_entropy_normalized",
        "landing_zone_entropy",
        "landing_zone_entropy_normalized",
        "shot_speed_mean",
        "shot_speed_std",
        "shot_height_mean",
        "shot_height_std",
        "pressure_mean",
        "pressure_max",
        "pressure_required_speed_score_mean",
        "pressure_intercept_scarcity_score_mean",
        "pressure_low_contact_score_mean",
        "pressure_reaction_miss_score_mean",
        "recovery_counterfactual_count",
        "recovery_chosen_mean_rank",
        "recovery_chosen_mean_rank_fraction",
        "recovery_chosen_above_average_fraction",
        "recovery_chosen_best_fraction",
        "recovery_a_rec_mean",
        "recovery_a_rec_std",
        "recovery_a_rec_min",
        "recovery_a_rec_max",
        "recovery_training_advantage_mean",
        "recovery_training_advantage_std",
        "recovery_training_advantage_min",
        "recovery_training_advantage_max",
        "recovery_chosen_probability_mean",
        "recovery_next_hit_pair_count",
        "recovery_next_hit_x_corr",
        "recovery_next_hit_y_corr",
        "recovery_next_hit_distance_mean",
        "recovery_intended_landing_pair_count",
        "recovery_intended_landing_x_corr",
        "recovery_intended_landing_y_corr",
        "recovery_intended_landing_distance_mean",
        "tactical_shot_value_delta_mean",
        "tactical_forced_weak_return_rate",
        "tactical_recovery_success_rate",
        "tactical_opponent_displacement_mean",
        "tactical_time_pressure_mean",
        "tactical_unforced_error_rate",
        "tactical_defense_to_attack_transition_rate",
        "tactical_attack_retention_rate",
        "tactical_attacking_state_shot_value_delta_mean",
        "tactical_defensive_state_shot_value_delta_mean",
        "tactical_attacking_state_forced_weak_return_rate",
        "tactical_defensive_state_recovery_success_rate",
        "tactical_opponent_can_smash_recovery_success_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def plot_metric_evolution(
    report: dict[str, Any],
    output_dir: Path,
    config: SimulationConfig | None = None,
) -> dict[str, str]:
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    rows = sorted(report["rows"], key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in rows]
    plot_paths: dict[str, str] = {}

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), sharex=True)
    scalar_specs = [
        ("shot_type_entropy", "Shot-type entropy (bits)"),
        ("landing_zone_entropy", "Landing-zone entropy (bits)"),
        ("shot_speed_mean", "Average shot speed (m/s)"),
        ("shot_height_mean", "Average contact height (m)"),
        ("pressure_mean", "Average pressure"),
        ("recovery_intended_landing_distance_mean", "Recovery to landing distance (m)"),
    ]
    for ax, (key, label) in zip(axes.ravel(), scalar_specs):
        ax.plot(steps, [_nan_for_none(row.get(key)) for row in rows], marker="o", linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Training step")
    axes[-1, 1].set_xlabel("Training step")
    fig.suptitle("Anchor metric evolution")
    fig.tight_layout()
    path = output_dir / "metric_evolution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths["metric_evolution"] = str(path)

    plot_paths["shot_type_frequency"] = _plot_stacked_frequency(
        rows,
        steps,
        counter_key="shot_type_frequency",
        title="Shot-type frequency evolution",
        ylabel="Shot frequency",
        output_path=output_dir / "shot_type_frequency.png",
        plt=plt,
    )
    plot_paths["landing_zone_frequency"] = _plot_stacked_frequency(
        rows,
        steps,
        counter_key="landing_zone_frequency",
        title="Landing-zone distribution evolution",
        ylabel="Landing frequency",
        output_path=output_dir / "landing_zone_distribution.png",
        plt=plt,
    )
    plot_paths["pressure_components"] = _plot_lines(
        rows,
        steps,
        keys=[
            ("pressure_required_speed_score_mean", "required speed"),
            ("pressure_intercept_scarcity_score_mean", "intercept scarcity"),
            ("pressure_low_contact_score_mean", "low contact"),
            ("pressure_reaction_miss_score_mean", "reaction miss"),
        ],
        title="Pressure component evolution",
        ylabel="Component score",
        output_path=output_dir / "pressure_components.png",
        plt=plt,
    )
    plot_paths["recovery_intended_correlation"] = _plot_lines(
        rows,
        steps,
        keys=[
            ("recovery_intended_landing_x_corr", "x corr"),
            ("recovery_intended_landing_y_corr", "y corr"),
        ],
        title="Recovery vs intended landing correlation",
        ylabel="Pearson correlation",
        output_path=output_dir / "recovery_intended_correlation.png",
        plt=plt,
    )
    recovery_choice_path = _plot_recovery_choice_evolution(
        rows,
        steps,
        output_path=output_dir / "recovery_choice_evolution.png",
        plt=plt,
        title_suffix=" vs rating pool" if report.get("rating_pool_dir") else "",
    )
    if recovery_choice_path is not None:
        plot_paths["recovery_choice_evolution"] = recovery_choice_path
    plot_paths["speed_height_distributions"] = _plot_histogram_snapshots(rows, output_dir / "speed_height_distributions.png", plt)
    heatmap_path = _plot_landing_recovery_heatmap(
        rows,
        output_dir=output_dir,
        config=config or SimulationConfig(),
        plt=plt,
    )
    if heatmap_path is not None:
        plot_paths["landing_recovery_heatmap"] = heatmap_path
    plot_paths["tactical_metric_evolution"] = _plot_tactical_metric_evolution(
        rows,
        steps,
        output_path=output_dir / "tactical_metric_evolution.png",
        plt=plt,
    )
    report["plots"] = plot_paths
    (output_dir / "anchor_metric_evolution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return plot_paths


def _plot_recovery_choice_evolution(
    rows: list[dict[str, Any]],
    steps: list[int],
    *,
    output_path: Path,
    plt: Any,
    title_suffix: str = "",
) -> str | None:
    if not any(int(row.get("recovery_counterfactual_count", 0) or 0) > 0 for row in rows):
        return None

    rank_fraction = [_nan_for_none(row.get("recovery_chosen_mean_rank_fraction")) for row in rows]
    above_average = [_nan_for_none(row.get("recovery_chosen_above_average_fraction")) for row in rows]
    best = [_nan_for_none(row.get("recovery_chosen_best_fraction")) for row in rows]
    a_rec = [_nan_for_none(row.get("recovery_a_rec_mean")) for row in rows]
    training_advantage = [_nan_for_none(row.get("recovery_training_advantage_mean")) for row in rows]
    counts = [int(row.get("recovery_counterfactual_count", 0) or 0) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    ax_rank, ax_fraction, ax_advantage, ax_grid = axes.flat

    ax_rank.plot(steps, rank_fraction, marker="o", color="tab:blue", linewidth=2.0)
    ax_rank.set_title("Chosen recovery rank fraction")
    ax_rank.set_ylabel("Rank fraction (lower is better)")
    ax_rank.set_ylim(0.0, 1.0)
    ax_rank.grid(True, alpha=0.3)

    ax_fraction.plot(steps, above_average, marker="o", color="tab:green", linewidth=2.0, label="above average")
    ax_fraction.plot(steps, best, marker="o", color="tab:purple", linewidth=2.0, label="best bin")
    ax_fraction.set_title("Choice quality rates")
    ax_fraction.set_ylabel("Fraction")
    ax_fraction.set_ylim(0.0, 1.0)
    ax_fraction.grid(True, alpha=0.3)
    ax_fraction.legend()

    ax_advantage.plot(steps, a_rec, marker="o", color="tab:orange", linewidth=2.0, label="a_rec")
    ax_advantage.plot(
        steps,
        training_advantage,
        marker="o",
        color="tab:red",
        linewidth=1.8,
        linestyle="--",
        label="training advantage",
    )
    ax_advantage.axhline(0.0, color="0.2", linewidth=1.0, alpha=0.6)
    ax_advantage.set_title("Recovery advantage")
    ax_advantage.set_xlabel("Anchor training step")
    ax_advantage.set_ylabel("Advantage")
    ax_advantage.grid(True, alpha=0.3)
    ax_advantage.legend()

    rows_with_grid = [
        row
        for row in rows
        if int(row.get("recovery_counterfactual_count", 0) or 0) > 0
        and isinstance(row.get("recovery_no_feasible_rate_grid"), list)
    ]
    if rows_with_grid:
        grid = np.asarray(rows_with_grid[-1]["recovery_no_feasible_rate_grid"], dtype=float)
        if grid.ndim == 2 and grid.size:
            image = ax_grid.imshow(grid.T, origin="lower", vmin=0.0, vmax=1.0, cmap="magma")
            ax_grid.set_title(f"Final no-feasible rate ({counts[-1]} samples)")
            ax_grid.set_xlabel("x recovery bin")
            ax_grid.set_ylabel("y recovery bin")
            ax_grid.set_xticks(range(grid.shape[0]))
            ax_grid.set_yticks(range(grid.shape[1]))
            fig.colorbar(image, ax=ax_grid, fraction=0.046, pad=0.04)
        else:
            ax_grid.axis("off")
    else:
        ax_grid.axis("off")
        ax_grid.text(0.5, 0.5, "No recovery bin grid recorded", ha="center", va="center")

    fig.suptitle(f"Anchor recovery choice evolution{title_suffix}")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def _plot_landing_recovery_heatmap(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    config: SimulationConfig,
    plt: Any,
) -> str | None:
    rows_with_samples = [
        row
        for row in sorted(rows, key=lambda item: int(item["step"]))
        if row.get("landing_samples_xy") and row.get("recovery_samples_xy")
    ]
    if not rows_with_samples:
        return None

    row = rows_with_samples[-1]
    landing = np.asarray(row["landing_samples_xy"], dtype=float)
    recovery = np.asarray(row["recovery_samples_xy"], dtype=float)
    if landing.ndim != 2 or landing.shape[1] != 2 or recovery.ndim != 2 or recovery.shape[1] != 2:
        return None

    step = int(row["step"])
    output_path = output_dir / f"landing_recovery_heatmap_step_{step}.png"
    colors = stage_colors(monochrome=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.0), sharex=True, sharey=True)
    specs = [
        (axes[0], landing, "Landing zone"),
        (axes[1], recovery, "Recovery position"),
    ]
    for ax, points, title in specs:
        setup_court_axes(ax, config, colors, show_axes=True)
        _draw_point_heatmap(ax, points, config=config, plt=plt)
        ax.set_title(f"{title} heatmap")
    fig.suptitle(f"Landing and recovery snapshot at step {step}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _draw_point_heatmap(
    ax: Any,
    points: np.ndarray,
    *,
    config: SimulationConfig,
    plt: Any,
) -> None:
    if points.size == 0:
        return
    if config.court.lateral_motion_enabled:
        x_min, x_max = -config.court.half_width, config.court.half_width
    else:
        lane_half_width = max(config.player.marker_radius * 2.2, 0.55)
        x_min = config.court.default_player_start_x - lane_half_width
        x_max = config.court.default_player_start_x + lane_half_width
    y_min, y_max = -config.court.half_length, config.court.half_length
    x_edges = np.linspace(x_min, x_max, 41)
    y_edges = np.linspace(y_min, y_max, 61)
    heatmap, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_edges, y_edges])
    total = float(np.sum(heatmap))
    if total <= 0.0:
        return
    heatmap = heatmap / total
    image = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="magma",
        alpha=0.72,
        zorder=3,
        aspect="auto",
    )
    ax.scatter(points[:, 0], points[:, 1], s=4, color="white", alpha=0.22, linewidths=0, zorder=4)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="shot fraction")


def _tactical_stage_infos(
    rally: RallyTrace,
    config: SimulationConfig,
    *,
    pressure_weights: ShotPressureWeights | None,
) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    pressures = [
        shot_pressure_from_stage_trace(stage, config, weights=pressure_weights)
        for stage in rally.stages
    ]
    for index, stage in enumerate(rally.stages):
        incoming_pressure = None if index == 0 else float(pressures[index - 1].pressure)
        pressure = pressures[index]
        velocity = np.asarray(stage.shuttle_velocity, dtype=float)
        speed = float(np.linalg.norm(velocity))
        theta = math.degrees(math.atan2(float(stage.shuttle_velocity[2]), float(np.hypot(*stage.shuttle_velocity[:2]))))
        shot_type = name_velocity_shot(
            hitter=stage.hitter_side,
            contact_x=float(stage.shuttle_start[0]),
            contact_y=float(stage.shuttle_start[1]),
            landing_x=float(stage.shuttle_landing[0]),
            landing_y=float(stage.shuttle_landing[1]),
            theta_degrees=theta,
            config=config,
        )
        contact_height = float(stage.shuttle_start[2])
        state_bucket = _tactical_state_bucket(
            contact_height=contact_height,
            incoming_pressure=incoming_pressure,
            hitter_y=float(stage.hitter_start[1]),
            hitter_side=stage.hitter_side,
            config=config,
        )
        infos.append(
            {
                "stage_index": int(stage.stage_index),
                "hitter_side": stage.hitter_side,
                "state_bucket": state_bucket,
                "shot_type": shot_type,
                "shot_speed": speed,
                "contact_height": contact_height,
                "pressure": float(pressure.pressure),
                "time_pressure": float(pressure.required_speed_ratio),
                "intercept_scarcity": float(pressure.intercept_scarcity_score),
                "opponent_displacement": float(
                    np.hypot(
                        float(stage.receiver_end[0]) - float(stage.receiver_start[0]),
                        float(stage.receiver_end[1]) - float(stage.receiver_start[1]),
                    )
                ),
                "terminal": bool(stage.terminal),
                "winner": stage.winner,
                "terminal_reason": stage.terminal_reason,
                "unforced_error": bool(
                    stage.terminal
                    and stage.winner != stage.hitter_side
                    and stage.terminal_reason in {
                        "invalid_serve_target",
                        "train_no_valid_shot",
                        "opponent_no_valid_shot",
                    }
                ),
            }
        )
    return infos


def _tactical_state_bucket(
    *,
    contact_height: float,
    incoming_pressure: float | None,
    hitter_y: float,
    hitter_side: Side,
    config: SimulationConfig,
) -> str:
    pressure = 0.0 if incoming_pressure is None else float(incoming_pressure)
    depth_from_net = (config.court.net_y - hitter_y) if hitter_side == "left" else (hitter_y - config.court.net_y)
    y_low, y_high = side_y_bounds(hitter_side, config)
    side_depth = abs(float(y_high) - float(y_low))
    depth_ratio = float(np.clip(depth_from_net / max(side_depth, 1e-9), 0.0, 1.0))
    if contact_height >= 1.45 and pressure <= 0.45 and depth_ratio <= 0.78:
        return "attacking_state"
    if contact_height <= 1.05 or pressure >= 0.65 or depth_ratio >= 0.82:
        return "defensive_state"
    return "neutral_state"


def _summarize_tactical_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "shot_value_delta_mean": _mean_optional(sample.get("shot_value_delta") for sample in samples),
        "shot_value_delta_std": _std_optional(sample.get("shot_value_delta") for sample in samples),
        "forced_weak_return_rate": _rate_optional(sample.get("forced_weak_return") for sample in samples),
        "recovery_success_rate": _rate_optional(sample.get("recovery_success") for sample in samples),
        "opponent_displacement_mean": _mean_optional(sample.get("opponent_displacement") for sample in samples),
        "time_pressure_mean": _mean_optional(sample.get("time_pressure") for sample in samples),
        "shot_speed_mean": _mean_optional(sample.get("shot_speed") for sample in samples),
        "contact_height_mean": _mean_optional(sample.get("contact_height") for sample in samples),
        "unforced_error_rate": _rate_optional(sample.get("unforced_error") for sample in samples),
        "defense_to_attack_transition_rate": _rate_optional(sample.get("defense_to_attack") for sample in samples),
        "attack_retention_rate": _rate_optional(sample.get("attack_retention") for sample in samples),
    }


def _flatten_tactical_metrics(by_context: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for context, metrics in by_context.items():
        prefix = "tactical" if context == "all" else f"tactical_{context}"
        for key, value in metrics.items():
            if key == "sample_count":
                flattened[f"{prefix}_{key}"] = value
            elif value is None or isinstance(value, (int, float)):
                flattened[f"{prefix}_{key}"] = value
    return flattened


def _flatten_row_tactical_metrics(tactical_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in tactical_metrics.items()
        if key.startswith("tactical_") and (value is None or isinstance(value, (int, float)))
    }


def write_tactical_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    contexts = ["all", "attacking_state", "neutral_state", "defensive_state", "opponent_can_smash"]
    metric_keys = [
        "sample_count",
        "shot_value_delta_mean",
        "forced_weak_return_rate",
        "recovery_success_rate",
        "opponent_displacement_mean",
        "time_pressure_mean",
        "shot_speed_mean",
        "contact_height_mean",
        "unforced_error_rate",
        "defense_to_attack_transition_rate",
        "attack_retention_rate",
    ]
    fieldnames = ["step", "context", *metric_keys]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            tactical = row.get("tactical_metrics", {})
            by_context = tactical.get("by_context", {}) if isinstance(tactical, dict) else {}
            for context in contexts:
                metrics = by_context.get(context, {}) if isinstance(by_context, dict) else {}
                writer.writerow(
                    {
                        "step": row.get("step"),
                        "context": context,
                        **{key: metrics.get(key) for key in metric_keys},
                    }
                )


def write_fixed_pool_rating_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    rows = sorted(report.get("rows", []), key=lambda row: int(row["step"]))
    if not any(row.get("pool_matchups") for row in rows):
        return {}
    if not all(
        "score_agent" in matchup
        for row in rows
        for matchup in row.get("pool_matchups", [])
        if isinstance(matchup, dict)
    ):
        return {}

    rating_report = estimate_fixed_pool_ratings(rows)
    matrix_report = build_win_rate_matrix_report(rows, report.get("pool_agents", []))
    rating_json = output_dir / "fixed_pool_rating_report.json"
    matrix_json = output_dir / "win_rate_matrix.json"
    rating_json.write_text(json.dumps(rating_report, indent=2), encoding="utf-8")
    matrix_json.write_text(json.dumps(matrix_report, indent=2), encoding="utf-8")
    write_fixed_pool_rating_csv(output_dir / "fixed_pool_ratings.csv", rating_report)
    write_win_rate_matrix_csv(output_dir / "win_rate_matrix.csv", matrix_report)

    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    rating_plot = plot_fixed_pool_ratings(
        rating_report,
        output_path=output_dir / "fixed_pool_rating_evolution.png",
        plt=plt,
    )
    matrix_plot = plot_win_rate_matrix(
        matrix_report,
        output_path=output_dir / "win_rate_matrix.png",
        plt=plt,
    )
    return {
        "fixed_pool_rating_evolution": rating_plot,
        "win_rate_matrix": matrix_plot,
    }


def estimate_fixed_pool_ratings(
    rows: list[dict[str, Any]],
    *,
    initial_rating: float = 1500.0,
    elo_scale: float = 400.0,
    prior_std: float = 400.0,
) -> dict[str, Any]:
    records_by_pair: dict[tuple[str, str], dict[str, float | str]] = {}
    agent_steps: dict[str, int | None] = {}
    ordered_record_count = 0
    for row in rows:
        agent = f"anchor_{int(row['step'])}"
        agent_steps[agent] = int(row["step"])
        for matchup in row.get("pool_matchups", []):
            if not isinstance(matchup, dict):
                continue
            opponent = str(matchup["opponent"])
            agent_steps.setdefault(opponent, _step_from_label(opponent))
            games = float(matchup.get("rally_count") or 0.0)
            if games <= 0.0 or agent == opponent:
                continue
            ordered_record_count += 1
            agent_a, agent_b = sorted((agent, opponent))
            key = (agent_a, agent_b)
            record = records_by_pair.setdefault(
                key,
                {
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "agent_a_score": 0.0,
                    "games": 0.0,
                },
            )
            score_agent = float(matchup["score_agent"])
            record["agent_a_score"] = float(record["agent_a_score"]) + (
                score_agent if agent == agent_a else games - score_agent
            )
            record["games"] = float(record["games"]) + games
    records = list(records_by_pair.values())
    ratings, standard_errors = _fit_elo_with_standard_errors(
        records,
        initial_rating=initial_rating,
        scale=elo_scale,
        prior_std=prior_std,
    )
    ordered = sorted(
        ratings,
        key=lambda label: (
            agent_steps.get(label) is None,
            agent_steps.get(label) if agent_steps.get(label) is not None else label,
        ),
    )
    rating_rows = [
        {
            "agent": label,
            "step": agent_steps.get(label),
            "elo": float(ratings[label]),
            "elo_se": float(standard_errors[label]),
            "elo_ci_low": float(ratings[label] - 1.96 * standard_errors[label]),
            "elo_ci_high": float(ratings[label] + 1.96 * standard_errors[label]),
        }
        for label in ordered
    ]
    return {
        "definition": (
            "Bradley-Terry/Elo fit from anchor-vs-fixed-pool rally outcomes; reciprocal cross-play "
            "records are weight-summed by unordered pair, and intervals use the fitted Hessian."
        ),
        "initial_rating": initial_rating,
        "elo_scale": elo_scale,
        "prior_std": prior_std,
        "ordered_record_count": ordered_record_count,
        "record_count": len(records),
        "ratings": rating_rows,
    }


def build_win_rate_matrix_report(rows: list[dict[str, Any]], pool_agents: list[dict[str, Any]]) -> dict[str, Any]:
    row_labels = [f"anchor_{int(row['step'])}" for row in rows]
    row_steps = [int(row["step"]) for row in rows]
    if pool_agents:
        col_labels = [str(agent["label"]) for agent in pool_agents]
        col_steps = [agent.get("step") for agent in pool_agents]
    else:
        col_labels = sorted({str(matchup["opponent"]) for row in rows for matchup in row.get("pool_matchups", [])})
        col_steps = [_step_from_label(label) for label in col_labels]

    matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)
    counts = np.zeros((len(row_labels), len(col_labels)), dtype=int)
    col_index = {label: index for index, label in enumerate(col_labels)}
    for row_index, row in enumerate(rows):
        for matchup in row.get("pool_matchups", []):
            if not isinstance(matchup, dict):
                continue
            opponent = str(matchup["opponent"])
            if opponent not in col_index:
                continue
            j = col_index[opponent]
            matrix[row_index, j] = _nan_for_none(matchup.get("agent_win_rate"))
            counts[row_index, j] = int(matchup.get("rally_count", 0) or 0)
    return {
        "definition": "P_ij = Pr(anchor checkpoint i beats fixed-pool opponent j) estimated by rally win rate.",
        "row_labels": row_labels,
        "row_steps": row_steps,
        "col_labels": col_labels,
        "col_steps": col_steps,
        "win_rate_matrix": matrix.tolist(),
        "rally_count_matrix": counts.astype(int).tolist(),
    }


def write_fixed_pool_rating_csv(path: Path, report: dict[str, Any]) -> None:
    import csv

    fieldnames = ["agent", "step", "elo", "elo_se", "elo_ci_low", "elo_ci_high"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["ratings"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_win_rate_matrix_csv(path: Path, report: dict[str, Any]) -> None:
    import csv

    col_labels = list(report["col_labels"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "step", *col_labels])
        for label, step, values in zip(report["row_labels"], report["row_steps"], report["win_rate_matrix"]):
            writer.writerow([label, step, *values])


def plot_fixed_pool_ratings(report: dict[str, Any], *, output_path: Path, plt: Any) -> str:
    rating_rows = [row for row in report["ratings"] if row.get("step") is not None]
    steps = np.asarray([int(row["step"]) for row in rating_rows], dtype=float)
    ratings = np.asarray([float(row["elo"]) for row in rating_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(steps, ratings, marker="o", linewidth=2.0, color="tab:blue")
    ax.set_title("Checkpoint fixed-pool rating")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Elo rating")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def plot_win_rate_matrix(report: dict[str, Any], *, output_path: Path, plt: Any) -> str:
    matrix = np.asarray(report["win_rate_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdBu", aspect="auto")
    ax.set_title("Win-rate matrix: P(row checkpoint beats column opponent)")
    ax.set_xlabel("Fixed-pool opponent")
    ax.set_ylabel("Evaluated checkpoint")
    col_steps = [str(step) if step is not None else str(label) for label, step in zip(report["col_labels"], report["col_steps"])]
    row_steps = [str(step) for step in report["row_steps"]]
    ax.set_xticks(range(len(col_steps)))
    ax.set_xticklabels(col_steps, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_steps)))
    ax.set_yticklabels(row_steps, fontsize=8)
    diagonal_end = min(len(row_steps), len(col_steps)) - 0.5
    ax.plot([-0.5, diagonal_end], [-0.5, diagonal_end], color="0.2", linewidth=1.0, alpha=0.35)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Win rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_tactical_metric_evolution(
    rows: list[dict[str, Any]],
    steps: list[int],
    *,
    output_path: Path,
    plt: Any,
) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), sharex=True, constrained_layout=True)
    specs = [
        (
            axes[0, 0],
            [
                ("tactical_shot_value_delta_mean", "all"),
                ("tactical_attacking_state_shot_value_delta_mean", "attacking"),
                ("tactical_defensive_state_shot_value_delta_mean", "defensive"),
            ],
            "Shot value delta",
            "Delta win probability",
        ),
        (
            axes[0, 1],
            [
                ("tactical_forced_weak_return_rate", "all"),
                ("tactical_attacking_state_forced_weak_return_rate", "attacking"),
            ],
            "Forced weak return",
            "Rate",
        ),
        (
            axes[1, 0],
            [
                ("tactical_recovery_success_rate", "all"),
                ("tactical_defensive_state_recovery_success_rate", "defensive"),
                ("tactical_opponent_can_smash_recovery_success_rate", "opponent can smash"),
            ],
            "Recovery success",
            "Rate",
        ),
        (
            axes[1, 1],
            [
                ("tactical_opponent_displacement_mean", "opponent displacement"),
                ("tactical_time_pressure_mean", "time pressure"),
            ],
            "Pressure created",
            "Mean",
        ),
        (
            axes[2, 0],
            [("tactical_unforced_error_rate", "unforced errors")],
            "Unforced error rate",
            "Rate",
        ),
        (
            axes[2, 1],
            [
                ("tactical_defense_to_attack_transition_rate", "defense to attack"),
                ("tactical_attack_retention_rate", "attack retention"),
            ],
            "Attack/defense transitions",
            "Rate",
        ),
    ]
    for ax, series, title, ylabel in specs:
        for key, label in series:
            ax.plot(steps, [_nan_for_none(row.get(key)) for row in rows], marker="o", linewidth=1.7, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1, 0].set_xlabel("Training step")
    axes[-1, 1].set_xlabel("Training step")
    fig.suptitle("State-conditioned tactical metric evolution")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def _fit_elo_with_standard_errors(
    records: list[dict[str, Any]],
    *,
    initial_rating: float,
    scale: float,
    prior_std: float,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
) -> tuple[dict[str, float], dict[str, float]]:
    if not records:
        raise ValueError("records must not be empty")
    agents = sorted({str(record["agent_a"]) for record in records} | {str(record["agent_b"]) for record in records})
    index = {agent: i for i, agent in enumerate(agents)}
    ratings = np.full(len(agents), float(initial_rating), dtype=float)
    beta = math.log(10.0) / float(scale)
    prior_precision = 1.0 / (float(prior_std) ** 2)

    neg_hessian = np.eye(len(agents), dtype=float) * prior_precision
    for _ in range(max(int(max_iterations), 1)):
        gradient = np.zeros_like(ratings)
        neg_hessian = np.eye(len(agents), dtype=float) * prior_precision
        for record in records:
            i = index[str(record["agent_a"])]
            j = index[str(record["agent_b"])]
            games = float(record["games"])
            score_a = float(record["agent_a_score"])
            probability_a = 1.0 / (1.0 + math.exp(-beta * float(ratings[i] - ratings[j])))
            residual = score_a - games * probability_a
            gradient[i] += beta * residual
            gradient[j] -= beta * residual
            curvature = beta * beta * games * probability_a * (1.0 - probability_a)
            neg_hessian[i, i] += curvature
            neg_hessian[j, j] += curvature
            neg_hessian[i, j] -= curvature
            neg_hessian[j, i] -= curvature
        gradient -= (ratings - initial_rating) * prior_precision
        step = np.linalg.solve(neg_hessian, gradient)
        ratings += step
        if float(np.max(np.abs(step))) < tolerance:
            break

    covariance = np.linalg.inv(neg_hessian)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return (
        {agent: float(rating) for agent, rating in zip(agents, ratings.tolist())},
        {agent: float(se) for agent, se in zip(agents, standard_errors.tolist())},
    )


def landing_zone_name(receiver_side: Side, landing_xy: tuple[float, float], config: SimulationConfig) -> str:
    x = float(landing_xy[0])
    y = float(landing_xy[1])
    x_edges = np.linspace(-config.court.half_width, config.court.half_width, 4)
    lane_index = int(np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, 2))
    y_low, y_high = side_y_bounds(receiver_side, config)
    depth_from_net = (config.court.net_y - y) if receiver_side == "left" else (y - config.court.net_y)
    side_depth = abs(float(y_high) - float(y_low))
    depth_ratio = float(np.clip(depth_from_net / max(side_depth, 1e-9), 0.0, 0.999999))
    depth_index = int(np.clip(math.floor(depth_ratio * 3.0), 0, 2))
    return LANDING_ZONE_NAMES[depth_index * 3 + lane_index]


def entropy_bits(counts: list[int] | tuple[int, ...]) -> float:
    values = np.asarray(counts, dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probs = values[values > 0.0] / total
    return float(-np.sum(probs * np.log2(probs)))


def normalized_entropy(counts: list[int] | tuple[int, ...]) -> float:
    nonzero_or_possible = len(counts)
    if nonzero_or_possible <= 1:
        return 0.0
    return float(entropy_bits(counts) / math.log2(nonzero_or_possible))


def pearson_or_none(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _plot_stacked_frequency(
    rows: list[dict[str, Any]],
    steps: list[int],
    *,
    counter_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    plt: Any,
) -> str:
    names = _top_frequency_names(rows, counter_key, limit=10)
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    y_values = [[float(row.get(counter_key, {}).get(name, 0.0)) for row in rows] for name in names]
    if y_values:
        ax.stackplot(steps, y_values, labels=names, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    if names:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _plot_lines(
    rows: list[dict[str, Any]],
    steps: list[int],
    *,
    keys: list[tuple[str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
    plt: Any,
) -> str:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for key, label in keys:
        ax.plot(steps, [_nan_for_none(row.get(key)) for row in rows], marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _plot_histogram_snapshots(rows: list[dict[str, Any]], output_path: Path, plt: Any) -> str:
    if not rows:
        return str(output_path)
    picks = [0, len(rows) // 2, len(rows) - 1]
    picks = sorted(set(picks))
    fig, axes = plt.subplots(2, len(picks), figsize=(5.0 * len(picks), 7.0), squeeze=False)
    for col, index in enumerate(picks):
        row = rows[index]
        speed_bins = np.asarray(row["shot_speed_bins"], dtype=float)
        speed_hist = np.asarray(row["shot_speed_hist"], dtype=float)
        height_bins = np.asarray(row["shot_height_bins"], dtype=float)
        height_hist = np.asarray(row["shot_height_hist"], dtype=float)
        axes[0, col].bar(speed_bins[:-1], _hist_freq(speed_hist), width=np.diff(speed_bins), align="edge", alpha=0.8)
        axes[0, col].set_title(f"Step {int(row['step'])}")
        axes[0, col].set_ylabel("Speed frequency")
        axes[0, col].set_xlabel("m/s")
        axes[0, col].grid(True, alpha=0.25)
        axes[1, col].bar(height_bins[:-1], _hist_freq(height_hist), width=np.diff(height_bins), align="edge", alpha=0.8)
        axes[1, col].set_ylabel("Height frequency")
        axes[1, col].set_xlabel("m")
        axes[1, col].grid(True, alpha=0.25)
    fig.suptitle("Shot speed and contact-height distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _hist_freq(hist: np.ndarray) -> np.ndarray:
    total = float(np.sum(hist))
    if total <= 0.0:
        return hist
    return hist / total


def _top_frequency_names(rows: list[dict[str, Any]], key: str, *, limit: int) -> list[str]:
    totals: Counter[str] = Counter()
    for row in rows:
        for name, value in row.get(key, {}).items():
            totals[name] += float(value)
    return [name for name, _ in totals.most_common(limit)]


def _ordered_names(preferred: tuple[str, ...], counter: Counter[str]) -> list[str]:
    names = [name for name in preferred if counter.get(name, 0) > 0]
    names.extend(sorted(name for name in counter if name not in set(preferred)))
    return names


def _frequency_dict(names: list[str], counter: Counter[str]) -> dict[str, float]:
    total = max(float(sum(counter.values())), 1.0)
    return {name: float(counter.get(name, 0) / total) for name in names}


def _config_value(data: dict[str, Any], key: str, *aliases_and_default: Any) -> Any:
    if not aliases_and_default:
        raise TypeError("_config_value requires a default value")
    *aliases, default = aliases_and_default
    for candidate in (key, *aliases):
        value = data.get(str(candidate))
        if value is not None:
            return value
    return default


def _nan_for_none(value: object) -> float:
    return float("nan") if value is None else float(value)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _std(values: list[float]) -> float | None:
    return None if not values else float(np.std(np.asarray(values, dtype=float)))


def _min(values: list[float]) -> float | None:
    return None if not values else float(np.min(np.asarray(values, dtype=float)))


def _max(values: list[float]) -> float | None:
    return None if not values else float(np.max(np.asarray(values, dtype=float)))


def _numeric_values(values: Any) -> list[float]:
    numbers: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def _mean_optional(values: Any) -> float | None:
    numbers = _numeric_values(values)
    return _mean(numbers)


def _std_optional(values: Any) -> float | None:
    numbers = _numeric_values(values)
    return _std(numbers)


def _rate_optional(values: Any) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(np.mean(np.asarray([1.0 if bool(value) else 0.0 for value in valid], dtype=float)))
