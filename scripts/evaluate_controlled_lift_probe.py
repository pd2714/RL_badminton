from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.action_space import DiscreteActionMapper
from badminton.dynamics import (
    candidate_intercept_points,
    feasible_intercept_indices,
    landing_position,
    validate_and_clip_shot_action,
)
from badminton.env import Badminton1DEnv
from badminton.eval_evolution import (
    LANDING_ZONE_NAMES,
    SHOT_TYPE_ORDER,
    build_discrete_action_config,
    build_sim_config,
    checkpoint_step,
    discover_anchor_checkpoints,
    filter_anchor_checkpoints,
    landing_zone_name,
    load_anchor_model,
    load_run_config,
)
from badminton.evaluation import adapt_observation_to_model
from badminton.mpl_config import ensure_writable_matplotlib_config
from badminton.obs import ObservationConfig, ObservationEncoder
from badminton.pressure import shot_pressure_from_record
from badminton.shot_generators import name_velocity_shot
from badminton.state import ShotAction, Side, StageState
from badminton.utils import (
    default_player_position,
    ensure_directory,
    opponent_side,
    recovery_bounds,
    side_center_y,
    side_y_bounds,
    x_bounds,
)


@dataclass(frozen=True)
class ProbeScenario:
    probe_id: str
    title: str
    response_state: StageState
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare anchor policies from controlled hitter probe states."
    )
    parser.add_argument("run_dir", type=Path, help="Self-play run directory containing selfplay_config.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/anchor_metric_eval/<probe-name>_probe.",
    )
    parser.add_argument(
        "--probe-preset",
        choices=("controlled-lift", "contact-grid"),
        default="controlled-lift",
        help="controlled-lift keeps the original incoming-lift probe; contact-grid probes direct contact states.",
    )
    parser.add_argument(
        "--probe-name",
        type=str,
        default=None,
        help="Output file prefix. Defaults to controlled_lift or controlled_contact_grid.",
    )
    parser.add_argument("--samples", type=int, default=256, help="Policy action samples per anchor.")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--deterministic", action="store_true", help="Use one deterministic action per anchor.")
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--anchor-step-min", type=int, default=None)
    parser.add_argument("--anchor-step-max", type=int, default=None)
    parser.add_argument("--anchor-step-interval", type=int, default=None)
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional fixed checkpoint used only for critic/value estimates. "
            "Actions are still sampled from each probed anchor checkpoint."
        ),
    )
    parser.add_argument("--contact-index", type=int, default=16, help="Fixed intercept index for receiving the lift.")
    parser.add_argument("--lift-start-x", type=float, default=0.0)
    parser.add_argument("--lift-start-y", type=float, default=1.2)
    parser.add_argument("--lift-start-z", type=float, default=0.8)
    parser.add_argument("--lift-vx", type=float, default=0.0)
    parser.add_argument("--lift-vy", type=float, default=-11.0)
    parser.add_argument("--lift-vz", type=float, default=14.75)
    parser.add_argument(
        "--contact-grid-stage-index",
        type=int,
        default=5,
        help="Stage index assigned to direct contact-grid probe states.",
    )
    parser.add_argument(
        "--opponent-recovery-grid-3x3",
        action="store_true",
        help=(
            "For contact-grid probes, expand each hitter contact state over a 3x3 opponent-position "
            "grid on the opponent court."
        ),
    )
    parser.add_argument(
        "--opponent-grid-side",
        choices=("left", "right"),
        default=None,
        help="Court side for the opponent-position grid. Defaults to the opponent of train_side.",
    )
    parser.add_argument(
        "--train-side",
        choices=("left", "right"),
        default=None,
        help="Defaults to train_side in selfplay_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.anchor_stride <= 0:
        raise ValueError("--anchor-stride must be positive")
    if args.anchor_step_interval is not None and args.anchor_step_interval <= 0:
        raise ValueError("--anchor-step-interval must be positive")

    if args.probe_name:
        probe_name = args.probe_name
    elif args.probe_preset == "controlled-lift":
        probe_name = "controlled_lift"
    elif args.opponent_recovery_grid_3x3:
        probe_name = "controlled_contact_grid_opponent_grid3x3"
    else:
        probe_name = "controlled_contact_grid"
    output_dir = args.output_dir or (args.run_dir / "anchor_metric_eval" / f"{probe_name}_probe")
    ensure_directory(output_dir)

    run_config = load_run_config(args.run_dir)
    sim_config = build_sim_config(run_config)
    discrete_config = build_discrete_action_config(run_config)
    policy_type = str(run_config.get("policy_type", "velocity_oriented"))
    train_side: Side = args.train_side or str(run_config.get("train_side", "left"))  # type: ignore[assignment]
    reaction_time = float(run_config.get("reaction_time", 0.0) or 0.0)

    mapper = DiscreteActionMapper(sim_config, discrete_config, policy_type=policy_type)
    encoder = ObservationEncoder(
        sim_config,
        ObservationConfig(
            max_score=1,
            max_stages_per_rally=int(run_config.get("max_rally_stages", 120) or 120),
            include_feasible_mask=bool(run_config.get("include_feasible_mask", True)),
            include_reaction_risk_features=bool(run_config.get("include_reaction_risk_features", True)),
        ),
    )
    value_model = None
    value_checkpoint_path = args.value_checkpoint
    if value_checkpoint_path is not None:
        value_checkpoint_path = value_checkpoint_path.expanduser()
        print(f"fixed value critic: {value_checkpoint_path}", flush=True)
        value_model = load_anchor_model(value_checkpoint_path, recovery_choice_diagnostics=False)

    scenarios = _build_probe_scenarios(
        args=args,
        run_config=run_config,
        train_side=train_side,
        sim_config=sim_config,
    )
    probe_metadata = {
        "preset": args.probe_preset,
        "probe_name": probe_name,
        "value_checkpoint_path": None if value_checkpoint_path is None else str(value_checkpoint_path),
        "value_checkpoint_step": None if value_checkpoint_path is None else checkpoint_step(value_checkpoint_path),
        "scenario_count": len(scenarios),
        "scenarios": [scenario.metadata for scenario in scenarios],
    }
    (output_dir / f"{probe_name}_probe_state.json").write_text(
        json.dumps(probe_metadata, indent=2),
        encoding="utf-8",
    )

    checkpoints = discover_anchor_checkpoints(args.run_dir)
    checkpoints = filter_anchor_checkpoints(
        checkpoints,
        step_min=args.anchor_step_min,
        step_max=args.anchor_step_max,
        step_interval=args.anchor_step_interval,
    )
    checkpoints = checkpoints[:: args.anchor_stride]

    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        step = checkpoint_step(checkpoint)
        print(f"probing anchor_step_{step}", flush=True)
        model = load_anchor_model(checkpoint, recovery_choice_diagnostics=False)
        critic_model = value_model or model
        for scenario_index, scenario in enumerate(scenarios):
            anchor_before_value = _model_win_probability(
                model,
                encoder,
                scenario.response_state,
                agent_side=train_side,
                role="hitter",
                server_side=train_side,
            )
            before_value = _model_win_probability(
                critic_model,
                encoder,
                scenario.response_state,
                agent_side=train_side,
                role="hitter",
                server_side=train_side,
            )
            print(f"  {scenario.probe_id}", flush=True)
            rows = _sample_anchor_actions(
                model=model,
                value_model=critic_model,
                mapper=mapper,
                encoder=encoder,
                response_state=scenario.response_state,
                train_side=train_side,
                server_side=train_side,
                config=sim_config,
                step=step,
                checkpoint_path=checkpoint,
                samples=1 if args.deterministic else int(args.samples),
                deterministic=bool(args.deterministic),
                seed=int(args.seed) + checkpoint_index * 100_000 + scenario_index * 10_000_000,
                before_win_probability=before_value,
                anchor_before_win_probability=anchor_before_value,
                value_checkpoint_path=value_checkpoint_path,
                scenario=scenario,
            )
            sample_rows.extend(rows)
            summary_rows.append(
                _summarize_anchor_rows(
                    step,
                    checkpoint,
                    rows,
                    before_value,
                    anchor_before_value,
                    scenario=scenario,
                )
            )

    _write_csv(output_dir / f"{probe_name}_probe_samples.csv", sample_rows)
    _write_csv(output_dir / f"{probe_name}_probe_summary.csv", summary_rows)
    report = {
        "run_dir": str(args.run_dir),
        "output_dir": str(output_dir),
        "sample_count_per_anchor": 1 if args.deterministic else int(args.samples),
        "deterministic": bool(args.deterministic),
        "seed": int(args.seed),
        "policy_type": policy_type,
        "train_side": train_side,
        "value_checkpoint_path": None if value_checkpoint_path is None else str(value_checkpoint_path),
        "value_checkpoint_step": None if value_checkpoint_path is None else checkpoint_step(value_checkpoint_path),
        "probe": probe_metadata,
        "rows": summary_rows,
    }
    (output_dir / f"{probe_name}_probe_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    plot_paths = _write_probe_plots(output_dir, summary_rows, sample_rows, sim_config, scenarios, probe_name)
    report["plots"] = plot_paths
    (output_dir / f"{probe_name}_probe_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    summary_path = output_dir / f"{probe_name}_probe_summary.csv"
    print(f"summary: {summary_path}")
    for name, path in plot_paths.items():
        print(f"{name}: {path}")


def _build_probe_scenarios(
    *,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    train_side: Side,
    sim_config: Any,
) -> list[ProbeScenario]:
    reaction_time = float(run_config.get("reaction_time", 0.0) or 0.0)
    if args.probe_preset == "controlled-lift":
        lift = _build_fixed_lift(args, sim_config, train_side=train_side, reaction_time=reaction_time)
        response_state, lift_record = _resolve_fixed_lift(
            lift["state_before"],
            lift["action"],
            contact_index=int(args.contact_index),
            config=sim_config,
        )
        if response_state.current_hitter != train_side:
            raise RuntimeError(
                f"Expected fixed lift receiver to become {train_side!r}, got {response_state.current_hitter!r}."
            )
        metadata = _controlled_lift_metadata(
            args=args,
            run_config=run_config,
            train_side=train_side,
            sim_config=sim_config,
            lift_state=lift["state_before"],
            lift_action=lift["action"],
            lift_record=lift_record,
            response_state=response_state,
        )
        return [
            ProbeScenario(
                probe_id="controlled_lift",
                title="Controlled fixed-lift response probe",
                response_state=response_state,
                metadata=metadata,
            )
        ]
    if args.probe_preset == "contact-grid":
        scenarios = _build_contact_grid_scenarios(
            train_side=train_side,
            config=sim_config,
            reaction_time=reaction_time,
            stage_index=int(args.contact_grid_stage_index),
        )
        if args.opponent_recovery_grid_3x3:
            opponent_grid_side: Side = args.opponent_grid_side or opponent_side(train_side)  # type: ignore[assignment]
            scenarios = _expand_scenarios_over_opponent_recovery_grid(
                scenarios,
                opponent_grid_side=opponent_grid_side,
                config=sim_config,
            )
        return scenarios
    raise ValueError(f"Unsupported probe preset: {args.probe_preset}")


def _build_contact_grid_scenarios(
    *,
    train_side: Side,
    config: Any,
    reaction_time: float,
    stage_index: int,
) -> list[ProbeScenario]:
    x_positions = _contact_x_positions(config)
    y_positions = _contact_y_positions(train_side, config)
    z_positions = (("high", 2.5), ("mid", 1.5), ("low", 0.5))
    scenarios: list[ProbeScenario] = []
    for y_label, y in y_positions:
        for x_label, x in x_positions:
            for z_label, z in z_positions:
                state = _direct_contact_state(
                    train_side=train_side,
                    config=config,
                    contact_x=x,
                    contact_y=y,
                    contact_z=z,
                    reaction_time=reaction_time,
                    stage_index=stage_index,
                )
                probe_id = f"{y_label}_{x_label}_{z_label}"
                metadata = {
                    "probe_id": probe_id,
                    "preset": "contact-grid",
                    "description": "Direct hitter-state probe at a fixed agent contact point.",
                    "train_side": train_side,
                    "x_region": x_label,
                    "y_region": y_label,
                    "z_level": z_label,
                    "contact_point": {"x": float(x), "y": float(y), "z": float(z)},
                    "response_state": asdict(state),
                }
                scenarios.append(
                    ProbeScenario(
                        probe_id=probe_id,
                        title=f"{y_label} / {x_label} / {z_label} contact probe",
                        response_state=state,
                        metadata=metadata,
                    )
                )
    return scenarios


def _contact_x_positions(config: Any) -> tuple[tuple[str, float], ...]:
    low, high = x_bounds(config)
    span = high - low
    return (
        ("left", float(low + span / 6.0)),
        ("middle", float(0.5 * (low + high))),
        ("right", float(high - span / 6.0)),
    )


def _contact_y_positions(side: Side, config: Any) -> tuple[tuple[str, float], ...]:
    low, high = side_y_bounds(side, config)
    if side == "left":
        back = low + (high - low) / 6.0
        mid = 0.5 * (low + high)
        front = high - (high - low) / 6.0
    else:
        front = low + (high - low) / 6.0
        mid = 0.5 * (low + high)
        back = high - (high - low) / 6.0
    return (
        ("backcourt", float(back)),
        ("midcourt", float(mid)),
        ("frontcourt", float(front)),
    )


def _expand_scenarios_over_opponent_recovery_grid(
    scenarios: list[ProbeScenario],
    *,
    opponent_grid_side: Side,
    config: Any,
) -> list[ProbeScenario]:
    x_cells, y_cells = _opponent_recovery_grid_cells(opponent_grid_side, config)
    expanded: list[ProbeScenario] = []
    for scenario in scenarios:
        base_probe_id = scenario.probe_id
        for y_label, y in y_cells:
            for x_label, x in x_cells:
                cell_id = f"opponent_{y_label}_{x_label}"
                probe_id = f"{base_probe_id}__{cell_id}"
                if opponent_grid_side == "left":
                    response_state = replace(scenario.response_state, x_left=float(x), y_left=float(y))
                else:
                    response_state = replace(scenario.response_state, x_right=float(x), y_right=float(y))
                metadata = copy.deepcopy(scenario.metadata)
                metadata.update(
                    {
                        "probe_id": probe_id,
                        "contact_probe_id": base_probe_id,
                        "opponent_grid_side": opponent_grid_side,
                        "opponent_cell_id": cell_id,
                        "opponent_cell": {"x_region": x_label, "y_region": y_label},
                        "opponent_position": {"x": float(x), "y": float(y)},
                        "response_state": asdict(response_state),
                    }
                )
                expanded.append(
                    ProbeScenario(
                        probe_id=probe_id,
                        title=f"{scenario.title} | {cell_id.replace('_', ' ')}",
                        response_state=response_state,
                        metadata=metadata,
                    )
                )
    return expanded


def _opponent_recovery_grid_cells(
    side: Side,
    config: Any,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    (x_low, x_high), (y_low, y_high) = recovery_bounds(side, config)
    x_grid = _axis_grid(float(x_low), float(x_high), 5, lateral_motion_enabled=bool(config.court.lateral_motion_enabled))
    y_grid = _axis_grid(float(y_low), float(y_high), 5, lateral_motion_enabled=True)
    selected = (0, 2, 4)
    x_labels = ("left", "middle", "right")
    y_labels = ("backcourt", "midcourt", "frontcourt") if side == "left" else ("frontcourt", "midcourt", "backcourt")
    x_cells = tuple((label, float(x_grid[index])) for label, index in zip(x_labels, selected))
    y_cells = tuple((label, float(y_grid[index])) for label, index in zip(y_labels, selected))
    return x_cells, y_cells


def _axis_grid(
    low: float,
    high: float,
    bins: int,
    *,
    lateral_motion_enabled: bool,
) -> np.ndarray:
    if not lateral_motion_enabled:
        return np.full(bins, 0.5 * (low + high), dtype=float)
    return np.linspace(low, high, bins, dtype=float)


def _direct_contact_state(
    *,
    train_side: Side,
    config: Any,
    contact_x: float,
    contact_y: float,
    contact_z: float,
    reaction_time: float,
    stage_index: int,
) -> StageState:
    left_x, left_y = default_player_position("left", config)
    right_x, right_y = default_player_position("right", config)
    if train_side == "left":
        left_x, left_y = float(contact_x), float(contact_y)
    else:
        right_x, right_y = float(contact_x), float(contact_y)
    return StageState(
        x_left=float(left_x),
        y_left=float(left_y),
        x_right=float(right_x),
        y_right=float(right_y),
        current_hitter=train_side,
        x0=float(contact_x),
        y0=float(contact_y),
        z0=float(contact_z),
        reaction_time_left=reaction_time,
        reaction_time_right=reaction_time,
        rally_done=False,
        winner=None,
        stage_index=stage_index,
    )


def _build_fixed_lift(
    args: argparse.Namespace,
    config: Any,
    *,
    train_side: Side,
    reaction_time: float,
) -> dict[str, Any]:
    opponent = opponent_side(train_side)
    if train_side != "left":
        raise ValueError("The default lift probe currently assumes train_side='left'. Pass left-side runs only.")
    state = StageState(
        x_left=0.0,
        y_left=side_center_y("left", config),
        x_right=float(args.lift_start_x),
        y_right=float(args.lift_start_y),
        current_hitter=opponent,
        x0=float(args.lift_start_x),
        y0=float(args.lift_start_y),
        z0=float(args.lift_start_z),
        reaction_time_left=reaction_time,
        reaction_time_right=reaction_time,
        rally_done=False,
        winner=None,
        stage_index=5,
    )
    action = ShotAction(
        v_x=float(args.lift_vx),
        v_y=float(args.lift_vy),
        v_z=float(args.lift_vz),
        x_rec=0.0,
        y_rec=side_center_y(opponent, config),
    )
    validate_and_clip_shot_action(state, action, config)
    return {"state_before": state, "action": action}


def _resolve_fixed_lift(
    state: StageState,
    action: ShotAction,
    *,
    contact_index: int,
    config: Any,
) -> tuple[StageState, Any]:
    validated = validate_and_clip_shot_action(state, action, config)
    feasible = feasible_intercept_indices(state, validated.applied, config)
    if contact_index not in feasible:
        raise ValueError(f"--contact-index {contact_index} is not feasible for the fixed lift; feasible={feasible}")
    env = Badminton1DEnv(config=config)
    env.reset(state)
    record = env.step(action, contact_index)
    if record.next_state.rally_done:
        raise RuntimeError("The fixed lift ended the rally; choose a feasible contact index.")
    return record.next_state, record


def _sample_anchor_actions(
    *,
    model: Any,
    value_model: Any,
    mapper: DiscreteActionMapper,
    encoder: ObservationEncoder,
    response_state: StageState,
    train_side: Side,
    server_side: Side,
    config: Any,
    step: int,
    checkpoint_path: Path,
    samples: int,
    deterministic: bool,
    seed: int,
    before_win_probability: float,
    anchor_before_win_probability: float,
    value_checkpoint_path: Path | None,
    scenario: ProbeScenario,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending_after_state_indices: list[int] = []
    pending_after_states: list[StageState] = []
    observation = encoder.encode(
        state=response_state,
        agent_side=train_side,
        role="hitter",
        server_side=server_side,
    )
    adapted_observation = adapt_observation_to_model(model, observation)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if samples == 1:
        raw_actions, _ = model.predict(adapted_observation, deterministic=deterministic)
        raw_action_list = [raw_actions]
    else:
        batched_observation = np.repeat(adapted_observation[None, ...], samples, axis=0)
        raw_actions, _ = model.predict(batched_observation, deterministic=deterministic)
        raw_action_list = list(np.asarray(raw_actions))
    for sample_index, raw_action in enumerate(raw_action_list):
        decoded = mapper.decode_hitter_for_agent(raw_action, response_state, train_side)
        try:
            projected = mapper.project_hitter_action(response_state, decoded.shot_action)
            validated = validate_and_clip_shot_action(response_state, projected.shot_action, config)
        except (RuntimeError, ValueError) as error:
            rows.append(
                {
                    **_scenario_row_fields(scenario),
                    "step": step,
                    "checkpoint_path": str(checkpoint_path),
                    "value_checkpoint_path": None if value_checkpoint_path is None else str(value_checkpoint_path),
                    "sample_index": sample_index,
                    "valid": False,
                    "anchor_before_win_probability": float(anchor_before_win_probability),
                    "error": str(error),
                }
            )
            continue

        action = validated.applied
        landing_x, landing_y = landing_position(response_state, action, config)
        speed = float(np.linalg.norm([action.v_x, action.v_y, action.v_z]))
        theta_degrees = math.degrees(math.atan2(float(action.v_z), float(np.hypot(action.v_x, action.v_y))))
        shot_type = name_velocity_shot(
            hitter=train_side,
            contact_x=float(response_state.x0),
            contact_y=float(response_state.y0),
            landing_x=float(landing_x),
            landing_y=float(landing_y),
            theta_degrees=theta_degrees,
            config=config,
        )
        zone = landing_zone_name(opponent_side(train_side), (float(landing_x), float(landing_y)), config)
        record = _response_record(response_state, action, config, rng=rng)
        pressure = shot_pressure_from_record(record, config)
        pending_after_state_indices.append(len(rows))
        pending_after_states.append(record.next_state)
        rows.append(
            {
                **_scenario_row_fields(scenario),
                "step": step,
                "checkpoint_path": str(checkpoint_path),
                "value_checkpoint_path": None if value_checkpoint_path is None else str(value_checkpoint_path),
                "sample_index": sample_index,
                "valid": True,
                "projected": bool(projected.projected or validated.projected),
                "shot_speed": speed,
                "pressure": float(pressure.pressure),
                "pressure_required_speed_score": float(pressure.required_speed_score),
                "pressure_intercept_scarcity_score": float(pressure.intercept_scarcity_score),
                "pressure_low_contact_score": float(pressure.low_contact_score),
                "pressure_reaction_miss_score": float(pressure.reaction_miss_score),
                "chosen_reaction_miss_probability": pressure.chosen_reaction_miss_probability,
                "shot_value": None,
                "before_win_probability": float(before_win_probability),
                "after_win_probability": None,
                "anchor_shot_value": None,
                "anchor_before_win_probability": float(anchor_before_win_probability),
                "anchor_after_win_probability": None,
                "shot_type": shot_type,
                "landing_zone": zone,
                "landing_x": float(landing_x),
                "landing_y": float(landing_y),
                "v_x": float(action.v_x),
                "v_y": float(action.v_y),
                "v_z": float(action.v_z),
                "recovery_x": float(action.x_rec),
                "recovery_y": float(action.y_rec),
                "receiver_feasible_count": int(len(record.feasible_indices)),
                "receiver_chosen_index": None if record.chosen_index is None else int(record.chosen_index),
                "terminal_reason": record.terminal_reason,
            }
        )
    after_probabilities = _after_shot_win_probabilities(
        value_model,
        encoder,
        pending_after_states,
        train_side=train_side,
        server_side=server_side,
    )
    anchor_after_probabilities = _after_shot_win_probabilities(
        model,
        encoder,
        pending_after_states,
        train_side=train_side,
        server_side=server_side,
    )
    for row_index, after_win_probability in zip(pending_after_state_indices, after_probabilities):
        rows[row_index]["after_win_probability"] = float(after_win_probability)
        rows[row_index]["shot_value"] = float(after_win_probability - before_win_probability)
    for row_index, anchor_after_win_probability in zip(pending_after_state_indices, anchor_after_probabilities):
        rows[row_index]["anchor_after_win_probability"] = float(anchor_after_win_probability)
        rows[row_index]["anchor_shot_value"] = float(anchor_after_win_probability - anchor_before_win_probability)
    return rows


def _response_record(state: StageState, action: ShotAction, config: Any, *, rng: np.random.Generator) -> Any:
    feasible = feasible_intercept_indices(state, action, config)
    chosen_index = _highest_contact_index(state, action, feasible, config, rng=rng)
    env = Badminton1DEnv(config=config)
    env.reset(state)
    return env.step(action, chosen_index)


def _highest_contact_index(
    state: StageState,
    action: ShotAction,
    feasible: list[int],
    config: Any,
    *,
    rng: np.random.Generator,
) -> int | None:
    if not feasible:
        return None
    _, _, _, zs = candidate_intercept_points(state, action, config)
    max_height = max(float(zs[index]) for index in feasible if 0 <= index < len(zs))
    candidates = [index for index in feasible if 0 <= index < len(zs) and abs(float(zs[index]) - max_height) <= 1e-9]
    return int(rng.choice(candidates))


def _model_win_probability(
    model: Any,
    encoder: ObservationEncoder,
    state: StageState,
    *,
    agent_side: Side,
    role: str,
    server_side: Side,
) -> float:
    observation = encoder.encode(
        state=state,
        agent_side=agent_side,
        role=role,
        server_side=server_side,
    )
    observation = adapt_observation_to_model(model, observation)
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        value = float(model.policy.predict_values(obs_tensor).detach().cpu().numpy().reshape(-1)[0])
    return float(np.clip(0.5 * (value + 1.0), 0.0, 1.0))


def _after_shot_win_probability(
    model: Any,
    encoder: ObservationEncoder,
    state: StageState,
    *,
    train_side: Side,
    server_side: Side,
) -> float:
    if state.rally_done:
        return 1.0 if state.winner == train_side else 0.0
    active_side = state.current_hitter
    active_win_probability = _model_win_probability(
        model,
        encoder,
        state,
        agent_side=active_side,
        role="hitter",
        server_side=server_side,
    )
    if active_side == train_side:
        return active_win_probability
    return float(1.0 - active_win_probability)


def _after_shot_win_probabilities(
    model: Any,
    encoder: ObservationEncoder,
    states: list[StageState],
    *,
    train_side: Side,
    server_side: Side,
) -> list[float]:
    if not states:
        return []
    probabilities = np.full(len(states), np.nan, dtype=float)
    nonterminal_by_side: dict[Side, list[int]] = {}
    for index, state in enumerate(states):
        if state.rally_done:
            probabilities[index] = 1.0 if state.winner == train_side else 0.0
            continue
        nonterminal_by_side.setdefault(state.current_hitter, []).append(index)

    for active_side, indices in nonterminal_by_side.items():
        observations = np.stack(
            [
                encoder.encode(
                    state=states[index],
                    agent_side=active_side,
                    role="hitter",
                    server_side=server_side,
                )
                for index in indices
            ],
            axis=0,
        )
        observations = adapt_observation_to_model(model, observations)
        obs_tensor, _ = model.policy.obs_to_tensor(observations)
        with torch.no_grad():
            values = model.policy.predict_values(obs_tensor).detach().cpu().numpy().reshape(-1)
        active_probabilities = np.clip(0.5 * (values + 1.0), 0.0, 1.0)
        if active_side != train_side:
            active_probabilities = 1.0 - active_probabilities
        for index, probability in zip(indices, active_probabilities):
            probabilities[index] = float(probability)
    return [float(value) for value in probabilities]


def _summarize_anchor_rows(
    step: int,
    checkpoint: Path,
    rows: list[dict[str, Any]],
    before_win_probability: float,
    anchor_before_win_probability: float,
    *,
    scenario: ProbeScenario,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("valid")]
    shot_types = Counter(str(row["shot_type"]) for row in valid_rows)
    landing_zones = Counter(str(row["landing_zone"]) for row in valid_rows)
    summary: dict[str, Any] = {
        **_scenario_row_fields(scenario),
        "step": step,
        "checkpoint_path": str(checkpoint),
        "value_checkpoint_path": next(
            (row.get("value_checkpoint_path") for row in rows if row.get("value_checkpoint_path") is not None),
            None,
        ),
        "sample_count": len(rows),
        "valid_sample_count": len(valid_rows),
        "invalid_sample_count": len(rows) - len(valid_rows),
        "before_win_probability": before_win_probability,
        "anchor_before_win_probability": anchor_before_win_probability,
    }
    for key in (
        "shot_speed",
        "pressure",
        "pressure_required_speed_score",
        "pressure_intercept_scarcity_score",
        "pressure_low_contact_score",
        "pressure_reaction_miss_score",
        "chosen_reaction_miss_probability",
        "shot_value",
        "after_win_probability",
        "anchor_shot_value",
        "anchor_after_win_probability",
    ):
        values = [float(row[key]) for row in valid_rows if row.get(key) is not None]
        summary[f"{key}_mean"] = _mean(values)
        summary[f"{key}_std"] = _std(values)
    total = max(float(len(valid_rows)), 1.0)
    for name in SHOT_TYPE_ORDER:
        summary[f"shot_type_freq_{_field_name(name)}"] = float(shot_types.get(name, 0) / total)
    for name in LANDING_ZONE_NAMES:
        summary[f"landing_zone_freq_{name}"] = float(landing_zones.get(name, 0) / total)
    return summary


def _scenario_row_fields(scenario: ProbeScenario) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "probe_id": scenario.probe_id,
        "probe_title": scenario.title,
    }
    for key in ("preset", "x_region", "y_region", "z_level", "contact_probe_id", "opponent_grid_side", "opponent_cell_id"):
        if key in scenario.metadata:
            fields[key] = scenario.metadata[key]
    contact = scenario.metadata.get("contact_point")
    if isinstance(contact, dict):
        fields["contact_x"] = float(contact["x"])
        fields["contact_y"] = float(contact["y"])
        fields["contact_z"] = float(contact["z"])
    opponent = scenario.metadata.get("opponent_position")
    if isinstance(opponent, dict):
        fields["opponent_x"] = float(opponent["x"])
        fields["opponent_y"] = float(opponent["y"])
    return fields


def _controlled_lift_metadata(
    *,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    train_side: Side,
    sim_config: Any,
    lift_state: StageState,
    lift_action: ShotAction,
    lift_record: Any,
    response_state: StageState,
) -> dict[str, Any]:
    landing_x, landing_y = landing_position(lift_state, lift_action, sim_config)
    return {
        "probe_id": "controlled_lift",
        "preset": "controlled-lift",
        "description": "Fixed front-court low lift to the train agent back court, then one fixed intercept state.",
        "train_side": train_side,
        "reaction_time": float(run_config.get("reaction_time", 0.0) or 0.0),
        "contact_index": int(args.contact_index),
        "lift_state_before": asdict(lift_state),
        "lift_action": asdict(lift_action),
        "lift_landing": {"x": float(landing_x), "y": float(landing_y)},
        "lift_feasible_indices": list(map(int, lift_record.feasible_indices)),
        "lift_chosen_time": lift_record.chosen_time,
        "lift_intercept_point": None
        if lift_record.intercept_point is None
        else [float(value) for value in lift_record.intercept_point],
        "response_state": asdict(response_state),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_probe_plots(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    config: Any,
    scenarios: list[ProbeScenario],
    probe_name: str,
) -> dict[str, str]:
    if len(scenarios) == 1:
        scenario = scenarios[0]
        return _write_plots(
            output_dir,
            summary_rows,
            sample_rows,
            config,
            scenario.metadata,
            plot_prefix=probe_name,
            plot_title=scenario.title,
        )

    paths: dict[str, str] = {}
    for scenario in scenarios:
        scenario_dir = output_dir / scenario.probe_id
        ensure_directory(scenario_dir)
        scenario_summary_rows = [row for row in summary_rows if row.get("probe_id") == scenario.probe_id]
        scenario_sample_rows = [row for row in sample_rows if row.get("probe_id") == scenario.probe_id]
        scenario_paths = _write_plots(
            scenario_dir,
            scenario_summary_rows,
            scenario_sample_rows,
            config,
            scenario.metadata,
            plot_prefix=scenario.probe_id,
            plot_title=scenario.title,
        )
        for name, path in scenario_paths.items():
            paths[f"{scenario.probe_id}/{name}"] = path
    summary_path = output_dir / f"{probe_name}_latest_probe_grid.png"
    _write_latest_probe_grid(summary_path, summary_rows, scenarios, probe_name)
    paths["latest_probe_grid"] = str(summary_path)
    return paths


def _write_plots(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    config: Any,
    probe_metadata: dict[str, Any],
    *,
    plot_prefix: str,
    plot_title: str,
) -> dict[str, str]:
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    steps = np.asarray([int(row["step"]) for row in summary_rows], dtype=float)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 14.5), constrained_layout=True)
    _plot_mean_with_std(axes[0, 0], steps, summary_rows, "shot_speed", "Average chosen shot speed", "m/s")
    _plot_mean_with_std(axes[0, 1], steps, summary_rows, "pressure", "Miss-aware pressure created", "index", ylim=(0.0, 1.0))
    _plot_critic_value_delta(axes[1, 0], steps, summary_rows)
    _plot_anchor_critic_value_delta(axes[1, 1], steps, summary_rows)
    _plot_stacked_summary(
        axes[2, 0],
        steps,
        summary_rows,
        [name for name in SHOT_TYPE_ORDER if any(row.get(f"shot_type_freq_{_field_name(name)}", 0.0) for row in summary_rows)],
        "shot_type_freq_",
        "Shot type frequency",
    )
    _plot_stacked_summary(
        axes[2, 1],
        steps,
        summary_rows,
        [name for name in LANDING_ZONE_NAMES if any(row.get(f"landing_zone_freq_{name}", 0.0) for row in summary_rows)],
        "landing_zone_freq_",
        "Landing zone distribution",
    )
    _plot_landing_scatter(axes[3, 0], sample_rows, config, probe_metadata)
    axes[3, 1].axis("off")
    fig.suptitle(plot_title)
    path = output_dir / f"{plot_prefix}_probe.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["probe"] = str(path)

    path = output_dir / f"{plot_prefix}_shot_type_frequency.png"
    _write_stacked_plot(path, steps, summary_rows, SHOT_TYPE_ORDER, "shot_type_freq_", "Shot type frequency", plt)
    paths["shot_type_frequency"] = str(path)

    path = output_dir / f"{plot_prefix}_landing_zone_distribution.png"
    _write_stacked_plot(path, steps, summary_rows, LANDING_ZONE_NAMES, "landing_zone_freq_", "Landing zone distribution", plt)
    paths["landing_zone_distribution"] = str(path)
    return paths


def _write_latest_probe_grid(
    path: Path,
    summary_rows: list[dict[str, Any]],
    scenarios: list[ProbeScenario],
    probe_name: str,
) -> None:
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    contact_scenarios = [scenario for scenario in scenarios if scenario.metadata.get("preset") == "contact-grid"]
    if not contact_scenarios:
        return
    latest_by_probe: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        probe_id = str(row.get("probe_id", ""))
        current = latest_by_probe.get(probe_id)
        if current is None or int(row.get("step", -1)) > int(current.get("step", -1)):
            latest_by_probe[probe_id] = row

    x_labels = ("left", "middle", "right")
    y_labels = ("backcourt", "midcourt", "frontcourt")
    z_labels = ("high", "mid", "low")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), constrained_layout=True, sharey=True)
    arrays: list[np.ndarray] = []
    for z_label in z_labels:
        values = np.full((len(y_labels), len(x_labels)), np.nan, dtype=float)
        for scenario in contact_scenarios:
            meta = scenario.metadata
            if meta.get("z_level") != z_label:
                continue
            row = latest_by_probe.get(scenario.probe_id)
            if row is None:
                continue
            y_index = y_labels.index(str(meta["y_region"]))
            x_index = x_labels.index(str(meta["x_region"]))
            values[y_index, x_index] = _nan(row.get("shot_value_mean"))
        arrays.append(values)

    finite_chunks = [values[np.isfinite(values)] for values in arrays if np.isfinite(values).any()]
    finite_values = np.concatenate(finite_chunks) if finite_chunks else np.asarray([], dtype=float)
    if finite_values.size:
        vmax = float(np.nanmax(np.abs(finite_values)))
        vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0

    image = None
    for ax, z_label, values in zip(axes, z_labels, arrays):
        image = ax.imshow(values, cmap="coolwarm", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(f"{z_label} contact")
        ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
        ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
        for y_index in range(values.shape[0]):
            for x_index in range(values.shape[1]):
                value = values[y_index, x_index]
                text = "n/a" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(x_index, y_index, text, ha="center", va="center", fontsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="latest shot value mean")
    fig.suptitle(f"{probe_name}: latest contact-grid shot value")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_critic_value_delta(ax: Any, steps: np.ndarray, rows: list[dict[str, Any]]) -> None:
    before = np.asarray([_nan(row.get("before_win_probability")) for row in rows], dtype=float)
    after = np.asarray([_nan(row.get("after_win_probability_mean")) for row in rows], dtype=float)
    after_std = np.asarray([_nan(row.get("after_win_probability_std")) for row in rows], dtype=float)
    delta = np.asarray([_nan(row.get("shot_value_mean")) for row in rows], dtype=float)
    delta_std = np.asarray([_nan(row.get("shot_value_std")) for row in rows], dtype=float)

    ax.plot(steps, before, marker="o", linewidth=1.8, color="C0", label="before")
    ax.plot(steps, after, marker="o", linewidth=1.8, color="C2", label="after")
    if np.isfinite(after_std).any():
        ax.fill_between(steps, after - after_std, after + after_std, color="C2", alpha=0.12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("win-prob proxy")
    ax.grid(True, alpha=0.25)

    delta_ax = ax.twinx()
    delta_ax.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.45)
    delta_ax.plot(steps, delta, marker="o", linewidth=1.8, color="C3", label="delta")
    if np.isfinite(delta_std).any():
        delta_ax.fill_between(steps, delta - delta_std, delta + delta_std, color="C3", alpha=0.12)
    finite_delta = delta[np.isfinite(delta)]
    finite_std = delta_std[np.isfinite(delta_std)]
    if finite_delta.size:
        if finite_std.size == finite_delta.size:
            extent_values = np.concatenate([finite_delta - finite_std, finite_delta + finite_std])
        else:
            extent_values = finite_delta
        max_abs = max(float(np.nanmax(np.abs(extent_values))), 0.05)
        delta_ax.set_ylim(-1.1 * max_abs, 1.1 * max_abs)
    delta_ax.set_ylabel("delta")

    handles, labels = ax.get_legend_handles_labels()
    delta_handles, delta_labels = delta_ax.get_legend_handles_labels()
    ax.legend(handles + delta_handles, labels + delta_labels, loc="best", fontsize=7)
    ax.set_title("Fixed critic before/after and shot delta")


def _plot_anchor_critic_value_delta(ax: Any, steps: np.ndarray, rows: list[dict[str, Any]]) -> None:
    smooth_window = 5
    if not rows or "anchor_before_win_probability" not in rows[0]:
        ax.set_title("Own critic before/after and shot delta")
        ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    before = np.asarray([_nan(row.get("anchor_before_win_probability")) for row in rows], dtype=float)
    after = np.asarray([_nan(row.get("anchor_after_win_probability_mean")) for row in rows], dtype=float)
    delta = np.asarray([_nan(row.get("anchor_shot_value_mean")) for row in rows], dtype=float)
    before_smooth = _centered_rolling_nanmean(before, smooth_window)
    after_smooth = _centered_rolling_nanmean(after, smooth_window)
    delta_smooth = _centered_rolling_nanmean(delta, smooth_window)

    ax.plot(steps, before, linewidth=0.9, color="C0", alpha=0.25)
    ax.plot(steps, after, linewidth=0.9, color="C2", alpha=0.25)
    ax.plot(steps, before_smooth, marker="o", linewidth=1.8, color="C0", label="before avg")
    ax.plot(steps, after_smooth, marker="o", linewidth=1.8, color="C2", label="after avg")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("win-prob proxy")
    ax.grid(True, alpha=0.25)

    delta_ax = ax.twinx()
    delta_ax.axhline(0.0, color="0.45", linewidth=1.0, alpha=0.45)
    delta_ax.plot(steps, delta, linewidth=0.9, color="C3", alpha=0.25)
    delta_ax.plot(steps, delta_smooth, marker="o", linewidth=1.8, color="C3", label="delta avg")
    finite_delta = delta[np.isfinite(delta)]
    if finite_delta.size:
        max_abs = max(float(np.nanmax(np.abs(finite_delta))), 0.05)
        delta_ax.set_ylim(-1.1 * max_abs, 1.1 * max_abs)
    delta_ax.set_ylabel("delta")

    handles, labels = ax.get_legend_handles_labels()
    delta_handles, delta_labels = delta_ax.get_legend_handles_labels()
    ax.legend(handles + delta_handles, labels + delta_labels, loc="best", fontsize=7)
    ax.set_title(f"Own critic before/after and shot delta ({smooth_window}-anchor avg)")


def _centered_rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    half_window = window // 2
    smoothed = np.full_like(values, np.nan, dtype=float)
    for index in range(values.size):
        start = max(0, index - half_window)
        stop = min(values.size, index + half_window + 1)
        chunk = values[start:stop]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            smoothed[index] = float(np.mean(finite))
    return smoothed


def _plot_mean_with_std(
    ax: Any,
    steps: np.ndarray,
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    ylabel: str,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    mean = np.asarray([_nan(row.get(f"{key}_mean")) for row in rows], dtype=float)
    std = np.asarray([_nan(row.get(f"{key}_std")) for row in rows], dtype=float)
    ax.plot(steps, mean, marker="o", linewidth=1.8)
    if np.isfinite(std).any():
        ax.fill_between(steps, mean - std, mean + std, alpha=0.18)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)


def _plot_stacked_summary(
    ax: Any,
    steps: np.ndarray,
    rows: list[dict[str, Any]],
    names: list[str] | tuple[str, ...],
    prefix: str,
    title: str,
) -> None:
    if not names:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no samples", transform=ax.transAxes, ha="center", va="center")
        return
    values = np.asarray(
        [[float(row.get(f"{prefix}{_field_name(name) if prefix == 'shot_type_freq_' else name}", 0.0) or 0.0) for row in rows] for name in names],
        dtype=float,
    )
    ax.stackplot(steps, values, labels=names, alpha=0.82)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)


def _plot_landing_scatter(ax: Any, rows: list[dict[str, Any]], config: Any, probe_metadata: dict[str, Any]) -> None:
    valid_rows = [row for row in rows if row.get("valid")]
    if valid_rows:
        steps = np.asarray([int(row["step"]) for row in valid_rows], dtype=float)
        scatter = ax.scatter(
            [float(row["landing_x"]) for row in valid_rows],
            [float(row["landing_y"]) for row in valid_rows],
            c=steps,
            cmap="viridis",
            s=12,
            alpha=0.32,
            linewidths=0,
        )
        ax.figure.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02, label="step")
    lift = probe_metadata.get("lift_landing")
    contact = probe_metadata.get("lift_intercept_point")
    if isinstance(lift, dict):
        ax.scatter([float(lift["x"])], [float(lift["y"])], marker="x", s=80, color="black", label="fixed lift landing")
    if contact is not None:
        ax.scatter([float(contact[0])], [float(contact[1])], marker="*", s=120, color="tab:red", label="fixed contact")
    direct_contact = probe_metadata.get("contact_point")
    if isinstance(direct_contact, dict):
        ax.scatter(
            [float(direct_contact["x"])],
            [float(direct_contact["y"])],
            marker="*",
            s=120,
            color="tab:red",
            label="fixed contact",
        )
    ax.axhline(config.court.net_y, color="black", linewidth=1.0)
    ax.set_xlim(-config.court.half_width, config.court.half_width)
    ax.set_ylim(-config.court.half_length, config.court.half_length)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Chosen landing samples")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


def _write_stacked_plot(
    path: Path,
    steps: np.ndarray,
    rows: list[dict[str, Any]],
    names: list[str] | tuple[str, ...],
    prefix: str,
    title: str,
    plt: Any,
) -> None:
    active_names = [
        name
        for name in names
        if any(row.get(f"{prefix}{_field_name(name) if prefix == 'shot_type_freq_' else name}", 0.0) for row in rows)
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    _plot_stacked_summary(ax, steps, rows, active_names, prefix, title)
    ax.set_xlabel("Training step")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _field_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(np.std(np.asarray(values, dtype=float)))


def _nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


if __name__ == "__main__":
    main()
