from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from badminton1d.action_space import DiscreteActionConfig
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.evaluation import ModelSelector, rollout_episode
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
    speed_bins: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 90.0)
    height_bins: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


@dataclass(frozen=True)
class PoolAgentSpec:
    label: str
    step: int | None
    run_dir: Path
    model_path: Path


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "selfplay_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selfplay config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_sim_config(data: dict[str, Any]) -> SimulationConfig:
    defaults = SimulationConfig()
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


def rollout_anchor_match_trace(
    *,
    model: PPO,
    run_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> MatchTrace:
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
    )
    selector = ModelSelector(model=model, deterministic=deterministic)
    rallies: list[RallyTrace] = []
    score_left = 0
    score_right = 0
    for episode in range(episodes):
        result = rollout_episode(env, selector, seed=seed + episode)
        if result["winner"] == "left":
            score_left += 1
        elif result["winner"] == "right":
            score_right += 1
        records = result["records"]
        assert isinstance(records, list)
        trace = build_rally_trace(records, sim_config)
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
    return MatchTrace(
        rallies=rallies,
        target_score=max(score_left, score_right),
        score_left=score_left,
        score_right=score_right,
        winner=winner,
        total_playback_time=sum(rally.total_playback_time for rally in rallies),
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
) -> MatchTrace:
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
    )
    selector = ModelSelector(model=model, deterministic=deterministic)
    rallies: list[RallyTrace] = []
    score_left = 0
    score_right = 0
    for episode in range(episodes):
        result = rollout_episode(env, selector, seed=seed + episode)
        if result["winner"] == "left":
            score_left += 1
        elif result["winner"] == "right":
            score_right += 1
        records = result["records"]
        assert isinstance(records, list)
        trace = build_rally_trace(records, sim_config)
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
    return MatchTrace(
        rallies=rallies,
        target_score=max(score_left, score_right),
        score_left=score_left,
        score_right=score_right,
        winner=winner,
        total_playback_time=sum(rally.total_playback_time for rally in rallies),
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
        model = PPO.load(checkpoint)
        pool_matchups: list[dict[str, Any]] = []
        if pool_agents is None:
            trace = rollout_anchor_match_trace(
                model=model,
                run_config=run_config,
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                episodes=eval_config.episodes,
                seed=eval_config.seed + index * 10_000,
                deterministic=eval_config.deterministic,
            )
        else:
            traces = []
            for pool_index, opponent in enumerate(pool_agents):
                print(f"  vs {opponent.label} ({eval_config.episodes} rallies)", flush=True)
                matchup_trace = rollout_fixed_pool_match_trace(
                    model=model,
                    opponent=opponent,
                    run_config=run_config,
                    sim_config=sim_config,
                    discrete_action_config=discrete_action_config,
                    episodes=eval_config.episodes,
                    seed=eval_config.seed + index * 100_000 + pool_index * 1_000,
                    deterministic=eval_config.deterministic,
                )
                traces.append(matchup_trace)
                pool_matchups.append(
                    {
                        "opponent": opponent.label,
                        "opponent_step": opponent.step,
                        "opponent_model_path": str(opponent.model_path),
                        **summarize_match_trace_metrics(
                            matchup_trace,
                            sim_config,
                            speed_bins=eval_config.speed_bins,
                            height_bins=eval_config.height_bins,
                            include_spatial_samples=False,
                        ),
                    }
                )
            trace = combine_match_traces(traces)
        metrics = summarize_match_trace_metrics(
            trace,
            sim_config,
            speed_bins=eval_config.speed_bins,
            height_bins=eval_config.height_bins,
        )
        rows.append(
            {
                "step": checkpoint_step(checkpoint),
                "checkpoint_path": str(checkpoint),
                "pool_matchups": pool_matchups,
                **metrics,
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
    plot_metric_evolution(report, output_dir, sim_config)
    return report


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
        "recovery_next_hit_pair_count",
        "recovery_next_hit_x_corr",
        "recovery_next_hit_y_corr",
        "recovery_next_hit_distance_mean",
        "recovery_intended_landing_pair_count",
        "recovery_intended_landing_x_corr",
        "recovery_intended_landing_y_corr",
        "recovery_intended_landing_distance_mean",
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
    plot_paths["speed_height_distributions"] = _plot_histogram_snapshots(rows, output_dir / "speed_height_distributions.png", plt)
    heatmap_path = _plot_landing_recovery_heatmap(
        rows,
        output_dir=output_dir,
        config=config or SimulationConfig(),
        plt=plt,
    )
    if heatmap_path is not None:
        plot_paths["landing_recovery_heatmap"] = heatmap_path
    report["plots"] = plot_paths
    (output_dir / "anchor_metric_evolution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return plot_paths


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


def _config_value(data: dict[str, Any], key: str, default: Any) -> Any:
    value = data.get(key)
    return default if value is None else value


def _nan_for_none(value: object) -> float:
    return float("nan") if value is None else float(value)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _std(values: list[float]) -> float | None:
    return None if not values else float(np.std(np.asarray(values, dtype=float)))


def _max(values: list[float]) -> float | None:
    return None if not values else float(np.max(np.asarray(values, dtype=float)))
