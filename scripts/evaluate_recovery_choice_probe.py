from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionMapper
from badminton1d.dynamics import (
    candidate_intercept_points,
    feasible_intercept_indices,
    landing_position,
    reaction_miss_probability,
    validate_and_clip_shot_action,
)
from badminton1d.env import Badminton1DEnv
from badminton1d.eval_evolution import (
    build_discrete_action_config,
    build_sim_config,
    checkpoint_step,
    discover_anchor_checkpoints,
    filter_anchor_checkpoints,
    load_anchor_model,
    load_run_config,
)
from badminton1d.evaluation import adapt_observation_to_model, choose_model_action
from badminton1d.mpl_config import ensure_writable_matplotlib_config
from badminton1d.opponents import DecisionContext
from badminton1d.obs import ObservationConfig, ObservationEncoder
from badminton1d.render import setup_court_axes, stage_colors
from badminton1d.state import ShotAction, Side, StageState
from badminton1d.utils import (
    default_player_position,
    ensure_directory,
    opponent_side,
    recovery_bounds,
    side_y_bounds,
    x_bounds,
)


@dataclass(frozen=True)
class RecoveryScenario:
    probe_id: str
    title: str
    state_before: StageState
    fixed_action: ShotAction
    intercept_index: int
    intercept_point: tuple[float, float, float]
    target_point: tuple[float, float, float]
    shot_component_action: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RecoveryBin:
    flat_index: int
    x_index: int
    y_index: int
    x_rec: float
    y_rec: float
    score: float
    critic_score: float
    policy_probability: float
    rank: int
    rank_fraction: float
    score_tie_count: int
    opponent_response_count: int
    opponent_action_flat_index: int | None
    opponent_v_x: float | None
    opponent_v_y: float | None
    opponent_v_z: float | None
    opponent_recovery_x: float | None
    opponent_recovery_y: float | None
    opponent_landing_x: float | None
    opponent_landing_y: float | None
    response_intercept_index: int | None
    response_intercept_x: float | None
    response_intercept_y: float | None
    response_intercept_z: float | None
    response_flight_time: float | None
    response_miss_probability: float | None
    response_no_miss_score: float | None
    response_receiver_feasible_count: int | None
    response_terminal_reason: str | None
    response_error: str | None
    opponent_responses: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ResponseProbeResult:
    score: float
    opponent_action_flat_index: int | None
    opponent_action: ShotAction | None
    opponent_landing: tuple[float, float] | None
    response_intercept_index: int | None
    response_intercept_point: tuple[float, float, float] | None
    response_flight_time: float | None
    response_miss_probability: float | None
    response_no_miss_score: float | None
    response_receiver_feasible_count: int | None
    response_terminal_reason: str | None
    response_error: str | None = None


@dataclass(frozen=True)
class OpponentShotPlan:
    action_flat_index: int | None
    action: ShotAction | None
    landing: tuple[float, float] | None
    probability: float = 1.0
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe recovery choices after fixed shots to controlled opponent contact positions."
    )
    parser.add_argument("run_dir", type=Path, help="Self-play run directory containing selfplay_config.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/anchor_metric_eval/recovery_contact_grid_probe.",
    )
    parser.add_argument("--probe-name", type=str, default="recovery_contact_grid")
    parser.add_argument("--samples", type=int, default=256, help="Recovery samples per anchor/scenario.")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--deterministic", action="store_true", help="Use the top-probability recovery bin once.")
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--anchor-step-min", type=int, default=None)
    parser.add_argument("--anchor-step-max", type=int, default=None)
    parser.add_argument("--anchor-step-interval", type=int, default=None)
    parser.add_argument("--train-side", choices=("left", "right"), default=None)
    parser.add_argument("--hitter-contact-x", type=float, default=None)
    parser.add_argument("--hitter-contact-y", type=float, default=None)
    parser.add_argument("--hitter-contact-z", type=float, default=1.5)
    parser.add_argument("--stage-index", type=int, default=5)
    parser.add_argument(
        "--target-distance-z-weight",
        type=float,
        default=1.0,
        help="Relative weight for contact-height error when searching fixed shots.",
    )
    parser.add_argument(
        "--counterfactual-opponent-response-samples",
        type=int,
        default=2,
        help="Score each recovery bin against the top-K likely opponent replies.",
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
    if args.counterfactual_opponent_response_samples <= 0:
        raise ValueError("--counterfactual-opponent-response-samples must be positive")

    output_dir = args.output_dir or (args.run_dir / "anchor_metric_eval" / f"{args.probe_name}_probe")
    ensure_directory(output_dir)

    run_config = load_run_config(args.run_dir)
    sim_config = build_sim_config(run_config)
    discrete_config = build_discrete_action_config(run_config)
    policy_type = str(run_config.get("policy_type", "velocity_oriented"))
    train_side: Side = args.train_side or str(run_config.get("train_side", "left"))  # type: ignore[assignment]

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
    reaction_time = float(run_config.get("reaction_time", 0.0) or 0.0)

    scenarios = _build_recovery_grid_scenarios(
        args=args,
        mapper=mapper,
        config=sim_config,
        train_side=train_side,
        reaction_time=reaction_time,
    )
    probe_metadata = {
        "probe_name": args.probe_name,
        "description": (
            "Fixed-shot recovery probe. Each scenario searches the discrete shot space for a train-agent "
            "shot whose opponent feasible contact is closest to the requested x/y/z target, then samples "
            "the policy recovery head for that fixed shot."
        ),
        "run_dir": str(args.run_dir),
        "train_side": train_side,
        "policy_type": policy_type,
        "sample_count_per_anchor": 1 if args.deterministic else int(args.samples),
        "deterministic": bool(args.deterministic),
        "counterfactual_opponent_response_samples": int(args.counterfactual_opponent_response_samples),
        "seed": int(args.seed),
        "scenarios": [scenario.metadata for scenario in scenarios],
    }
    (output_dir / f"{args.probe_name}_probe_state.json").write_text(
        json.dumps(probe_metadata, indent=2),
        encoding="utf-8",
    )

    checkpoints = filter_anchor_checkpoints(
        discover_anchor_checkpoints(args.run_dir),
        step_min=args.anchor_step_min,
        step_max=args.anchor_step_max,
        step_interval=args.anchor_step_interval,
    )[:: args.anchor_stride]

    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        step = checkpoint_step(checkpoint)
        print(f"probing anchor_step_{step}", flush=True)
        model = load_anchor_model(checkpoint, recovery_choice_diagnostics=False)
        for scenario_index, scenario in enumerate(scenarios):
            print(f"  {scenario.probe_id}", flush=True)
            bins = _score_recovery_bins(
                model=model,
                encoder=encoder,
                mapper=mapper,
                scenario=scenario,
                train_side=train_side,
                server_side=train_side,
                config=sim_config,
                opponent_response_samples=int(args.counterfactual_opponent_response_samples),
            )
            rng = np.random.default_rng(int(args.seed) + checkpoint_index * 100_000 + scenario_index * 10_000_000)
            rows = _sample_recovery_bins(
                scenario=scenario,
                step=step,
                checkpoint=checkpoint,
                bins=bins,
                samples=1 if args.deterministic else int(args.samples),
                deterministic=bool(args.deterministic),
                rng=rng,
            )
            sample_rows.extend(rows)
            summary_rows.append(_summarize_recovery_samples(scenario, step, checkpoint, bins, rows))
            bin_rows.extend(_recovery_bin_rows(scenario, step, checkpoint, bins))

    _write_csv(output_dir / f"{args.probe_name}_probe_samples.csv", sample_rows)
    _write_csv(output_dir / f"{args.probe_name}_probe_summary.csv", summary_rows)
    _write_csv(output_dir / f"{args.probe_name}_probe_bins.csv", bin_rows)
    report = {
        **probe_metadata,
        "output_dir": str(output_dir),
        "rows": summary_rows,
    }
    (output_dir / f"{args.probe_name}_probe_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    plot_paths = _write_recovery_probe_plots(output_dir, args.probe_name, scenarios, summary_rows, sample_rows, bin_rows, sim_config)
    report["plots"] = plot_paths
    (output_dir / f"{args.probe_name}_probe_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"summary: {output_dir / f'{args.probe_name}_probe_summary.csv'}")
    for name, path in plot_paths.items():
        print(f"{name}: {path}")


def _build_recovery_grid_scenarios(
    *,
    args: argparse.Namespace,
    mapper: DiscreteActionMapper,
    config: Any,
    train_side: Side,
    reaction_time: float,
) -> list[RecoveryScenario]:
    hitter_x, hitter_y = default_player_position(train_side, config)
    if args.hitter_contact_x is not None:
        hitter_x = float(args.hitter_contact_x)
    if args.hitter_contact_y is not None:
        hitter_y = float(args.hitter_contact_y)
    target_y_positions = _target_y_positions(opponent_side(train_side), config)
    target_x_positions = _target_x_positions(config)
    target_z_positions = (("high", 2.5), ("mid", 1.5), ("low", 0.5))
    scenarios: list[RecoveryScenario] = []
    for y_label, target_y in target_y_positions:
        for x_label, target_x in target_x_positions:
            for z_label, target_z in target_z_positions:
                state = _fixed_hitter_state(
                    train_side=train_side,
                    config=config,
                    hitter_x=float(hitter_x),
                    hitter_y=float(hitter_y),
                    hitter_z=float(args.hitter_contact_z),
                    opponent_x=float(target_x),
                    opponent_y=float(target_y),
                    reaction_time=reaction_time,
                    stage_index=int(args.stage_index),
                )
                shot = _find_fixed_shot_to_target(
                    mapper=mapper,
                    state=state,
                    train_side=train_side,
                    config=config,
                    target=(float(target_x), float(target_y), float(target_z)),
                    z_weight=float(args.target_distance_z_weight),
                )
                probe_id = f"{y_label}_{x_label}_{z_label}"
                scenario = RecoveryScenario(
                    probe_id=probe_id,
                    title=f"Recovery after shot to {y_label} / {x_label} / {z_label}",
                    state_before=state,
                    fixed_action=shot["action"],
                    intercept_index=int(shot["intercept_index"]),
                    intercept_point=tuple(float(value) for value in shot["intercept_point"]),
                    target_point=(float(target_x), float(target_y), float(target_z)),
                    shot_component_action=int(shot["shot_component_action"]),
                    metadata={
                        "probe_id": probe_id,
                        "target_y_region": y_label,
                        "target_x_region": x_label,
                        "target_z_level": z_label,
                        "target_point": {"x": float(target_x), "y": float(target_y), "z": float(target_z)},
                        "actual_intercept_point": [float(value) for value in shot["intercept_point"]],
                        "target_distance": float(shot["target_distance"]),
                        "intercept_index": int(shot["intercept_index"]),
                        "fixed_action": asdict(shot["action"]),
                        "shot_component_action": int(shot["shot_component_action"]),
                        "state_before": asdict(state),
                    },
                )
                scenarios.append(scenario)
    return scenarios


def _target_x_positions(config: Any) -> tuple[tuple[str, float], ...]:
    low, high = x_bounds(config)
    span = high - low
    return (
        ("left", float(low + span / 6.0)),
        ("middle", float(0.5 * (low + high))),
        ("right", float(high - span / 6.0)),
    )


def _target_y_positions(side: Side, config: Any) -> tuple[tuple[str, float], ...]:
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


def _fixed_hitter_state(
    *,
    train_side: Side,
    config: Any,
    hitter_x: float,
    hitter_y: float,
    hitter_z: float,
    opponent_x: float,
    opponent_y: float,
    reaction_time: float,
    stage_index: int,
) -> StageState:
    if train_side == "left":
        left_x, left_y = hitter_x, hitter_y
        right_x, right_y = opponent_x, opponent_y
    else:
        left_x, left_y = opponent_x, opponent_y
        right_x, right_y = hitter_x, hitter_y
    return StageState(
        x_left=float(left_x),
        y_left=float(left_y),
        x_right=float(right_x),
        y_right=float(right_y),
        current_hitter=train_side,
        x0=float(hitter_x),
        y0=float(hitter_y),
        z0=float(hitter_z),
        reaction_time_left=reaction_time,
        reaction_time_right=reaction_time,
        rally_done=False,
        winner=None,
        stage_index=stage_index,
    )


def _find_fixed_shot_to_target(
    *,
    mapper: DiscreteActionMapper,
    state: StageState,
    train_side: Side,
    config: Any,
    target: tuple[float, float, float],
    z_weight: float,
) -> dict[str, Any]:
    x_rec_count = int(mapper._impl._effective_x_rec_bins)
    y_rec_count = int(mapper.discrete_config.y_rec_bins)
    center_x = x_rec_count // 2
    center_y = y_rec_count // 2
    best: dict[str, Any] | None = None
    for phi_index in range(int(mapper._impl._effective_phi_bins)):
        for theta_index in range(int(mapper.discrete_config.theta_bins)):
            if mapper.valid_speed_range(state, phi_index, theta_index) is None:
                continue
            for speed_index in range(int(mapper.discrete_config.speed_bins)):
                flat_action = (
                    ((((phi_index * mapper.discrete_config.theta_bins + theta_index) * mapper.discrete_config.speed_bins + speed_index)
                      * x_rec_count + center_x)
                     * y_rec_count)
                    + center_y
                )
                try:
                    decoded = mapper.decode_hitter_for_agent(flat_action, state, train_side)
                    validated = validate_and_clip_shot_action(state, decoded.shot_action, config)
                    feasible = feasible_intercept_indices(state, validated.applied, config)
                except (RuntimeError, ValueError):
                    continue
                if not feasible:
                    continue
                _, xs, ys, zs = candidate_intercept_points(state, validated.applied, config)
                for intercept_index in feasible:
                    if not 0 <= intercept_index < len(xs):
                        continue
                    intercept = (float(xs[intercept_index]), float(ys[intercept_index]), float(zs[intercept_index]))
                    distance = math.sqrt(
                        (intercept[0] - target[0]) ** 2
                        + (intercept[1] - target[1]) ** 2
                        + float(z_weight) * (intercept[2] - target[2]) ** 2
                    )
                    if best is None or distance < float(best["target_distance"]):
                        component_action = flat_action - (flat_action % (x_rec_count * y_rec_count))
                        best = {
                            "action": validated.applied,
                            "intercept_index": int(intercept_index),
                            "intercept_point": intercept,
                            "target_distance": float(distance),
                            "shot_component_action": int(component_action),
                        }
    if best is None:
        raise RuntimeError(f"No feasible fixed shot found for target={target}")
    return best


def _score_recovery_bins(
    *,
    model: Any,
    encoder: ObservationEncoder,
    mapper: DiscreteActionMapper,
    scenario: RecoveryScenario,
    train_side: Side,
    server_side: Side,
    config: Any,
    opponent_response_samples: int,
) -> list[RecoveryBin]:
    probabilities = _policy_recovery_probabilities(
        model=model,
        encoder=encoder,
        scenario=scenario,
        train_side=train_side,
        server_side=server_side,
    )
    x_grid, y_grid = mapper._recovery_grid_for_shot_action(scenario.state_before, scenario.fixed_action)
    first_step_records: list[tuple[int, int, float, float, float, StageState]] = []
    for x_index, x_rec in enumerate(x_grid):
        for y_index, y_rec in enumerate(y_grid):
            action = replace(scenario.fixed_action, x_rec=float(x_rec), y_rec=float(y_rec))
            env = Badminton1DEnv(config=config)
            env.reset(scenario.state_before)
            record = env.step(action, scenario.intercept_index)
            critic_score = _after_shot_win_probability(
                model,
                encoder,
                record.next_state,
                train_side=train_side,
                server_side=server_side,
            )
            first_step_records.append((int(x_index), int(y_index), float(x_rec), float(y_rec), float(critic_score), record.next_state))

    scores: list[float] = []
    records: list[tuple[int, int, float, float, float, ResponseProbeResult, tuple[dict[str, Any], ...]]] = []
    for x_index, y_index, x_rec, y_rec, critic_score, next_state in first_step_records:
        opponent_plans = _choose_likely_opponent_shots(
            model=model,
            encoder=encoder,
            mapper=mapper,
            state_after_fixed_shot=next_state,
            train_side=train_side,
            server_side=server_side,
            config=config,
            count=opponent_response_samples,
        )
        if not opponent_plans:
            opponent_plans = [
                OpponentShotPlan(action_flat_index=None, action=None, landing=None, probability=1.0, error="no_plan")
            ]
        responses = [
            _score_fixed_opponent_response(
                model=model,
                encoder=encoder,
                mapper=mapper,
                state_after_fixed_shot=next_state,
                opponent_plan=opponent_plan,
                train_side=train_side,
                server_side=server_side,
                config=config,
            )
            for opponent_plan in opponent_plans
        ]
        weights = np.asarray([max(float(plan.probability), 0.0) for plan in opponent_plans], dtype=float)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            weights = np.full(len(responses), 1.0 / max(len(responses), 1), dtype=float)
        else:
            weights = weights / total_weight
        score = float(sum(float(response.score) * float(weight) for response, weight in zip(responses, weights)))
        representative_index = int(np.argmax(weights))
        response = responses[representative_index]
        response_payloads = tuple(
            _response_payload(
                response_result,
                probability=float(weight),
                rank=response_index + 1,
            )
            for response_index, (response_result, weight) in enumerate(zip(responses, weights))
        )
        scores.append(score)
        records.append(
            (
                int(x_index),
                int(y_index),
                float(x_rec),
                float(y_rec),
                float(critic_score),
                response,
                response_payloads,
            )
        )
    score_array = np.asarray(scores, dtype=float)
    ranks, tie_counts = _score_ranks(score_array)
    if probabilities.shape[0] != score_array.shape[0]:
        probabilities = np.full(score_array.shape[0], 1.0 / max(score_array.shape[0], 1), dtype=float)
    bins: list[RecoveryBin] = []
    for flat_index, ((x_index, y_index, x_rec, y_rec, critic_score, response, response_payloads), score) in enumerate(zip(records, score_array)):
        opponent_action = response.opponent_action
        response_intercept = response.response_intercept_point
        bins.append(
            RecoveryBin(
                flat_index=int(flat_index),
                x_index=x_index,
                y_index=y_index,
                x_rec=x_rec,
                y_rec=y_rec,
                score=float(score),
                critic_score=float(critic_score),
                policy_probability=float(probabilities[flat_index]),
                rank=int(ranks[flat_index]),
                rank_fraction=float(ranks[flat_index] / max(score_array.shape[0], 1)),
                score_tie_count=int(tie_counts[flat_index]),
                opponent_response_count=int(len(response_payloads)),
                opponent_action_flat_index=response.opponent_action_flat_index,
                opponent_v_x=None if opponent_action is None else float(opponent_action.v_x),
                opponent_v_y=None if opponent_action is None else float(opponent_action.v_y),
                opponent_v_z=None if opponent_action is None else float(opponent_action.v_z),
                opponent_recovery_x=None if opponent_action is None else float(opponent_action.x_rec),
                opponent_recovery_y=None if opponent_action is None else float(opponent_action.y_rec),
                opponent_landing_x=None if response.opponent_landing is None else float(response.opponent_landing[0]),
                opponent_landing_y=None if response.opponent_landing is None else float(response.opponent_landing[1]),
                response_intercept_index=response.response_intercept_index,
                response_intercept_x=None if response_intercept is None else float(response_intercept[0]),
                response_intercept_y=None if response_intercept is None else float(response_intercept[1]),
                response_intercept_z=None if response_intercept is None else float(response_intercept[2]),
                response_flight_time=response.response_flight_time,
                response_miss_probability=response.response_miss_probability,
                response_no_miss_score=response.response_no_miss_score,
                response_receiver_feasible_count=response.response_receiver_feasible_count,
                response_terminal_reason=response.response_terminal_reason,
                response_error=response.response_error,
                opponent_responses=response_payloads,
            )
        )
    return bins


def _score_ranks(scores: np.ndarray, *, atol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Return tie-aware competition ranks and tie counts for higher-is-better scores."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    ranks = np.ones(scores.shape[0], dtype=int)
    tie_counts = np.ones(scores.shape[0], dtype=int)
    finite = np.isfinite(scores)
    for index, score in enumerate(scores):
        if not finite[index]:
            ranks[index] = int(scores.shape[0])
            tie_counts[index] = int(np.count_nonzero(~finite))
            continue
        greater = finite & (scores > score + atol)
        tied = finite & np.isclose(scores, score, rtol=0.0, atol=atol)
        ranks[index] = int(np.count_nonzero(greater) + 1)
        tie_counts[index] = int(np.count_nonzero(tied))
    return ranks, tie_counts


def _choose_likely_opponent_shots(
    *,
    model: Any,
    encoder: ObservationEncoder,
    mapper: DiscreteActionMapper,
    state_after_fixed_shot: StageState,
    train_side: Side,
    server_side: Side,
    config: Any,
    count: int,
) -> list[OpponentShotPlan]:
    del train_side
    if state_after_fixed_shot.rally_done:
        return [OpponentShotPlan(action_flat_index=None, action=None, landing=None, probability=1.0, error="fixed_shot_terminal")]

    opponent = state_after_fixed_shot.current_hitter

    try:
        observation = encoder.encode(
            state=state_after_fixed_shot,
            agent_side=opponent,
            role="hitter",
            server_side=server_side,
        )
        action_probabilities = _top_hitter_action_probabilities(
            model,
            observation,
            hitter_action_count=mapper.hitter_action_count,
            count=max(int(count) * 8, int(count) + 16),
        )
    except (RuntimeError, ValueError) as error:
        return [OpponentShotPlan(action_flat_index=None, action=None, landing=None, probability=1.0, error=str(error))]

    plans = _valid_opponent_shot_plans(
        mapper=mapper,
        state_after_fixed_shot=state_after_fixed_shot,
        opponent=opponent,
        config=config,
        action_probabilities=action_probabilities,
        count=count,
    )
    if len(plans) < int(count) and len(action_probabilities) < int(mapper.hitter_action_count):
        tried = {int(flat_action) for flat_action, _ in action_probabilities}
        fallback_probabilities = _top_hitter_action_probabilities(
            model,
            observation,
            hitter_action_count=mapper.hitter_action_count,
            count=int(mapper.hitter_action_count),
        )
        fallback_plans = _valid_opponent_shot_plans(
            mapper=mapper,
            state_after_fixed_shot=state_after_fixed_shot,
            opponent=opponent,
            config=config,
            action_probabilities=fallback_probabilities,
            count=max(int(count) - len(plans), 0),
            skip_flat_actions=tried,
            require_receiver_feasible=True,
        )
        if not fallback_plans:
            fallback_plans = _valid_opponent_shot_plans(
                mapper=mapper,
                state_after_fixed_shot=state_after_fixed_shot,
                opponent=opponent,
                config=config,
                action_probabilities=fallback_probabilities,
                count=max(int(count) - len(plans), 0),
                skip_flat_actions=tried,
            )
        plans.extend(fallback_plans)

    total_probability = float(sum(max(float(plan.probability), 0.0) for plan in plans))
    if plans and total_probability > 0.0 and np.isfinite(total_probability):
        plans = [
            replace(plan, probability=max(float(plan.probability), 0.0) / total_probability)
            for plan in plans
        ]
    return plans


def _valid_opponent_shot_plans(
    *,
    mapper: DiscreteActionMapper,
    state_after_fixed_shot: StageState,
    opponent: Side,
    config: Any,
    action_probabilities: list[tuple[int, float]],
    count: int,
    skip_flat_actions: set[int] | None = None,
    require_receiver_feasible: bool = False,
) -> list[OpponentShotPlan]:
    plans: list[OpponentShotPlan] = []
    skip_flat_actions = skip_flat_actions or set()
    for flat_action, probability in action_probabilities:
        if int(flat_action) in skip_flat_actions:
            continue
        try:
            decoded = mapper.decode_hitter_for_agent(flat_action, state_after_fixed_shot, opponent)
            projected = mapper.project_hitter_action(state_after_fixed_shot, decoded.shot_action)
            validated = validate_and_clip_shot_action(state_after_fixed_shot, projected.shot_action, config)
        except (RuntimeError, ValueError):
            continue
        opponent_action = validated.applied
        if require_receiver_feasible and not feasible_intercept_indices(state_after_fixed_shot, opponent_action, config):
            continue
        plans.append(
            OpponentShotPlan(
                action_flat_index=int(decoded.flat_index),
                action=opponent_action,
                landing=landing_position(state_after_fixed_shot, opponent_action, config),
                probability=float(probability),
            )
        )
        if len(plans) >= int(count):
            break
    return plans


def _response_payload(
    response: ResponseProbeResult,
    *,
    probability: float,
    rank: int,
) -> dict[str, Any]:
    action = response.opponent_action
    landing = response.opponent_landing
    intercept = response.response_intercept_point
    return {
        "rank": int(rank),
        "probability": float(probability),
        "score": float(response.score),
        "opponent_action_flat_index": response.opponent_action_flat_index,
        "opponent_v_x": None if action is None else float(action.v_x),
        "opponent_v_y": None if action is None else float(action.v_y),
        "opponent_v_z": None if action is None else float(action.v_z),
        "opponent_recovery_x": None if action is None else float(action.x_rec),
        "opponent_recovery_y": None if action is None else float(action.y_rec),
        "opponent_landing_x": None if landing is None else float(landing[0]),
        "opponent_landing_y": None if landing is None else float(landing[1]),
        "response_intercept_index": response.response_intercept_index,
        "response_intercept_x": None if intercept is None else float(intercept[0]),
        "response_intercept_y": None if intercept is None else float(intercept[1]),
        "response_intercept_z": None if intercept is None else float(intercept[2]),
        "response_flight_time": response.response_flight_time,
        "response_miss_probability": response.response_miss_probability,
        "response_no_miss_score": response.response_no_miss_score,
        "response_receiver_feasible_count": response.response_receiver_feasible_count,
        "response_terminal_reason": response.response_terminal_reason,
        "response_error": response.response_error,
    }


def _top_hitter_action_probabilities(
    model: Any,
    observation: np.ndarray,
    *,
    hitter_action_count: int,
    count: int,
    batch_size: int = 2048,
) -> list[tuple[int, float]]:
    observation = adapt_observation_to_model(model, observation)
    actions = np.arange(max(int(hitter_action_count), 0), dtype=np.int64)
    if actions.size == 0:
        return []
    log_probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, actions.size, batch_size):
            batch_actions = actions[start : start + batch_size]
            batch_obs = np.repeat(observation.reshape(1, -1), batch_actions.size, axis=0).astype(np.float32)
            obs_tensor, _ = model.policy.obs_to_tensor(batch_obs)
            action_tensor = torch.as_tensor(batch_actions, dtype=torch.long, device=model.device)
            _, batch_log_prob, _ = model.policy.evaluate_actions(obs_tensor, action_tensor)
            log_probs.append(batch_log_prob.detach().cpu().numpy().reshape(-1))
    log_prob_array = np.concatenate(log_probs).astype(float, copy=False)
    finite = np.isfinite(log_prob_array)
    if not np.any(finite):
        raw_action, _ = model.predict(observation, deterministic=True)
        return [(int(raw_action), 1.0)]
    finite_indices = np.flatnonzero(finite)
    finite_scores = log_prob_array[finite_indices]
    order = np.argsort(finite_scores)[::-1][: max(int(count), 1)]
    selected = finite_indices[order]
    selected_scores = finite_scores[order]
    weights = np.exp(selected_scores - float(np.max(selected_scores)))
    total = float(np.sum(weights))
    probabilities = weights / total if total > 0.0 and np.isfinite(total) else np.full(weights.shape[0], 1.0 / max(weights.shape[0], 1))
    return [(int(index), float(probability)) for index, probability in zip(selected, probabilities)]


def _score_fixed_opponent_response(
    *,
    model: Any,
    encoder: ObservationEncoder,
    mapper: DiscreteActionMapper,
    state_after_fixed_shot: StageState,
    opponent_plan: OpponentShotPlan,
    train_side: Side,
    server_side: Side,
    config: Any,
) -> ResponseProbeResult:
    if state_after_fixed_shot.rally_done:
        return ResponseProbeResult(
            score=1.0 if state_after_fixed_shot.winner == train_side else 0.0,
            opponent_action_flat_index=opponent_plan.action_flat_index,
            opponent_action=opponent_plan.action,
            opponent_landing=opponent_plan.landing,
            response_intercept_index=None,
            response_intercept_point=None,
            response_flight_time=None,
            response_miss_probability=None,
            response_no_miss_score=None,
            response_receiver_feasible_count=None,
            response_terminal_reason="fixed_shot_terminal",
        )

    opponent_action = opponent_plan.action
    if opponent_action is None:
        return ResponseProbeResult(
            score=0.0,
            opponent_action_flat_index=opponent_plan.action_flat_index,
            opponent_action=None,
            opponent_landing=opponent_plan.landing,
            response_intercept_index=None,
            response_intercept_point=None,
            response_flight_time=None,
            response_miss_probability=None,
            response_no_miss_score=None,
            response_receiver_feasible_count=None,
            response_terminal_reason="opponent_no_valid_shot",
            response_error=opponent_plan.error,
        )

    feasible = feasible_intercept_indices(state_after_fixed_shot, opponent_action, config)
    response_index = _choose_receiver_intercept_index(
        model=model,
        encoder=encoder,
        mapper=mapper,
        state=state_after_fixed_shot,
        action=opponent_action,
        feasible=feasible,
        train_side=train_side,
        server_side=server_side,
    )
    response_flight_time = None
    response_miss_probability = 0.0
    if response_index is not None:
        candidate_times, _, _, _ = candidate_intercept_points(state_after_fixed_shot, opponent_action, config)
        if 0 <= int(response_index) < len(candidate_times):
            response_flight_time = float(candidate_times[int(response_index)])
            response_miss_probability = reaction_miss_probability(response_flight_time, config)

    env = Badminton1DEnv(config=config)
    env.reset(state_after_fixed_shot)
    with patch("badminton1d.dynamics.np.random.random", return_value=1.0):
        response_record = env.step(opponent_action, response_index)
    no_miss_score = _after_shot_win_probability(
        model,
        encoder,
        response_record.next_state,
        train_side=train_side,
        server_side=server_side,
    )
    response_score = (1.0 - float(response_miss_probability)) * float(no_miss_score)
    response_point = response_record.intercept_point or response_record.intended_intercept_point
    return ResponseProbeResult(
        score=float(response_score),
        opponent_action_flat_index=opponent_plan.action_flat_index,
        opponent_action=opponent_action,
        opponent_landing=opponent_plan.landing,
        response_intercept_index=None if response_record.chosen_index is None else int(response_record.chosen_index),
        response_intercept_point=response_point,
        response_flight_time=response_flight_time,
        response_miss_probability=float(response_miss_probability),
        response_no_miss_score=float(no_miss_score),
        response_receiver_feasible_count=int(len(feasible)),
        response_terminal_reason=response_record.terminal_reason,
    )


def _choose_receiver_intercept_index(
    *,
    model: Any,
    encoder: ObservationEncoder,
    mapper: DiscreteActionMapper,
    state: StageState,
    action: ShotAction,
    feasible: list[int],
    train_side: Side,
    server_side: Side,
) -> int | None:
    if not feasible:
        return None
    observation = encoder.encode(
        state=state,
        agent_side=train_side,
        role="receiver",
        server_side=server_side,
        pending_action=action,
        feasible_indices=feasible,
    )
    return int(
        choose_model_action(
            model,
            observation,
            DecisionContext(
                state=state,
                role="receiver",
                pending_action=action,
                feasible_indices=list(feasible),
                receiver_action_count=mapper.receiver_action_count,
            ),
            deterministic=True,
        )
    )


def _policy_recovery_probabilities(
    *,
    model: Any,
    encoder: ObservationEncoder,
    scenario: RecoveryScenario,
    train_side: Side,
    server_side: Side,
) -> np.ndarray:
    observation = encoder.encode(
        state=scenario.state_before,
        agent_side=train_side,
        role="hitter",
        server_side=server_side,
    )
    observation = adapt_observation_to_model(model, observation)
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    action_tensor = torch.as_tensor([int(scenario.shot_component_action)], dtype=torch.long, device=model.device)
    with torch.no_grad():
        features = model.policy.extract_features(obs_tensor)
        if model.policy.share_features_extractor:
            latent_pi, _ = model.policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
        phi, theta, speed, _ = model.policy._conditional_decompose_actions(action_tensor)
        _, _, _, recovery_logits, _ = model.policy._conditional_component_logits(
            obs_tensor,
            latent_pi,
            phi=phi,
            theta=theta,
            speed=speed,
        )
        if recovery_logits is None:
            raise RuntimeError("Model does not expose conditional recovery logits.")
        probabilities = torch.softmax(recovery_logits[0], dim=0).detach().cpu().numpy().astype(float, copy=False)
    total = float(np.sum(probabilities))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(probabilities.shape[0], 1.0 / max(probabilities.shape[0], 1), dtype=float)
    return probabilities / total


def _sample_recovery_bins(
    *,
    scenario: RecoveryScenario,
    step: int,
    checkpoint: Path,
    bins: list[RecoveryBin],
    samples: int,
    deterministic: bool,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    probabilities = np.asarray([recovery_bin.policy_probability for recovery_bin in bins], dtype=float)
    probabilities = probabilities / max(float(np.sum(probabilities)), 1e-12)
    if deterministic:
        sampled_indices = [int(np.argmax(probabilities))]
    else:
        sampled_indices = rng.choice(np.arange(len(bins), dtype=int), size=samples, replace=True, p=probabilities).astype(int).tolist()
    rows: list[dict[str, Any]] = []
    for sample_index, bin_index in enumerate(sampled_indices):
        recovery_bin = bins[int(bin_index)]
        rows.append(
            {
                **_scenario_fields(scenario),
                "step": int(step),
                "checkpoint_path": str(checkpoint),
                "sample_index": int(sample_index),
                "recovery_flat_index": recovery_bin.flat_index,
                "recovery_x_index": recovery_bin.x_index,
                "recovery_y_index": recovery_bin.y_index,
                "recovery_x": recovery_bin.x_rec,
                "recovery_y": recovery_bin.y_rec,
                "policy_probability": recovery_bin.policy_probability,
                "score": recovery_bin.score,
                "critic_score": recovery_bin.critic_score,
                "score_rank": recovery_bin.rank,
                "score_rank_fraction": recovery_bin.rank_fraction,
                "score_tie_count": recovery_bin.score_tie_count,
                "chosen_best": bool(recovery_bin.rank == 1),
            }
        )
    return rows


def _summarize_recovery_samples(
    scenario: RecoveryScenario,
    step: int,
    checkpoint: Path,
    bins: list[RecoveryBin],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best_score = max(item.score for item in bins)
    best_bins = [item for item in bins if np.isclose(item.score, best_score, rtol=0.0, atol=1e-9)]
    best_bin = max(best_bins, key=lambda item: item.policy_probability)
    top_prob_bin = max(bins, key=lambda item: item.policy_probability)
    counts = Counter(int(row["recovery_flat_index"]) for row in rows)
    sample_count = max(len(rows), 1)
    return {
        **_scenario_fields(scenario),
        "step": int(step),
        "checkpoint_path": str(checkpoint),
        "sample_count": len(rows),
        "recovery_x_mean": _mean([float(row["recovery_x"]) for row in rows]),
        "recovery_x_std": _std([float(row["recovery_x"]) for row in rows]),
        "recovery_y_mean": _mean([float(row["recovery_y"]) for row in rows]),
        "recovery_y_std": _std([float(row["recovery_y"]) for row in rows]),
        "sampled_score_mean": _mean([float(row["score"]) for row in rows]),
        "sampled_score_std": _std([float(row["score"]) for row in rows]),
        "sampled_rank_mean": _mean([float(row["score_rank"]) for row in rows]),
        "sampled_rank_fraction_mean": _mean([float(row["score_rank_fraction"]) for row in rows]),
        "sampled_best_frequency": float(sum(1 for row in rows if row["chosen_best"]) / sample_count),
        "sampled_top_probability_bin_frequency": float(counts.get(top_prob_bin.flat_index, 0) / sample_count),
        "policy_expected_score": float(sum(item.policy_probability * item.score for item in bins)),
        "policy_expected_critic_score": float(sum(item.policy_probability * item.critic_score for item in bins)),
        "policy_expected_rank_fraction": float(sum(item.policy_probability * item.rank_fraction for item in bins)),
        "policy_entropy": _entropy([item.policy_probability for item in bins]),
        "top_probability_bin": int(top_prob_bin.flat_index),
        "top_probability": float(top_prob_bin.policy_probability),
        "top_probability_x": float(top_prob_bin.x_rec),
        "top_probability_y": float(top_prob_bin.y_rec),
        "top_probability_score": float(top_prob_bin.score),
        "top_probability_critic_score": float(top_prob_bin.critic_score),
        "top_probability_response_miss_probability": top_prob_bin.response_miss_probability,
        "top_probability_score_rank": int(top_prob_bin.rank),
        "top_probability_score_rank_fraction": float(top_prob_bin.rank_fraction),
        "top_probability_score_tie_count": int(top_prob_bin.score_tie_count),
        "top_probability_is_best_score": bool(top_prob_bin.rank == 1),
        "best_score_bin": int(best_bin.flat_index),
        "best_score": float(best_bin.score),
        "best_score_x": float(best_bin.x_rec),
        "best_score_y": float(best_bin.y_rec),
        "best_score_policy_probability": float(best_bin.policy_probability),
        "best_score_policy_probability_mass": float(sum(item.policy_probability for item in best_bins)),
        "best_score_bin_count": int(len(best_bins)),
        "best_score_critic_score": float(best_bin.critic_score),
        "best_score_response_miss_probability": best_bin.response_miss_probability,
    }


def _recovery_bin_rows(
    scenario: RecoveryScenario,
    step: int,
    checkpoint: Path,
    bins: list[RecoveryBin],
) -> list[dict[str, Any]]:
    return [
        {
            **_scenario_fields(scenario),
            "step": int(step),
            "checkpoint_path": str(checkpoint),
            "recovery_flat_index": item.flat_index,
            "recovery_x_index": item.x_index,
            "recovery_y_index": item.y_index,
            "recovery_x": item.x_rec,
            "recovery_y": item.y_rec,
            "policy_probability": item.policy_probability,
            "score": item.score,
            "critic_score": item.critic_score,
            "score_rank": item.rank,
            "score_rank_fraction": item.rank_fraction,
            "score_tie_count": item.score_tie_count,
            "opponent_response_count": item.opponent_response_count,
            "opponent_action_flat_index": item.opponent_action_flat_index,
            "opponent_v_x": item.opponent_v_x,
            "opponent_v_y": item.opponent_v_y,
            "opponent_v_z": item.opponent_v_z,
            "opponent_recovery_x": item.opponent_recovery_x,
            "opponent_recovery_y": item.opponent_recovery_y,
            "opponent_landing_x": item.opponent_landing_x,
            "opponent_landing_y": item.opponent_landing_y,
            "response_intercept_index": item.response_intercept_index,
            "response_intercept_x": item.response_intercept_x,
            "response_intercept_y": item.response_intercept_y,
            "response_intercept_z": item.response_intercept_z,
            "response_flight_time": item.response_flight_time,
            "response_miss_probability": item.response_miss_probability,
            "response_no_miss_score": item.response_no_miss_score,
            "response_receiver_feasible_count": item.response_receiver_feasible_count,
            "response_terminal_reason": item.response_terminal_reason,
            "response_error": item.response_error,
            "opponent_responses_json": json.dumps(item.opponent_responses, separators=(",", ":"), sort_keys=True),
        }
        for item in bins
    ]


def _scenario_fields(scenario: RecoveryScenario) -> dict[str, Any]:
    target_x, target_y, target_z = scenario.target_point
    actual_x, actual_y, actual_z = scenario.intercept_point
    return {
        "probe_id": scenario.probe_id,
        "probe_title": scenario.title,
        "target_x_region": scenario.metadata["target_x_region"],
        "target_y_region": scenario.metadata["target_y_region"],
        "target_z_level": scenario.metadata["target_z_level"],
        "target_x": float(target_x),
        "target_y": float(target_y),
        "target_z": float(target_z),
        "actual_intercept_x": float(actual_x),
        "actual_intercept_y": float(actual_y),
        "actual_intercept_z": float(actual_z),
        "target_distance": float(scenario.metadata["target_distance"]),
    }


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


def _write_recovery_probe_plots(
    output_dir: Path,
    probe_name: str,
    scenarios: list[RecoveryScenario],
    summary_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    config: Any,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for scenario in scenarios:
        scenario_dir = output_dir / scenario.probe_id
        ensure_directory(scenario_dir)
        scenario_summary = [row for row in summary_rows if row.get("probe_id") == scenario.probe_id]
        scenario_samples = [row for row in sample_rows if row.get("probe_id") == scenario.probe_id]
        scenario_bins = [row for row in bin_rows if row.get("probe_id") == scenario.probe_id]
        path = scenario_dir / f"{scenario.probe_id}_recovery_probe.png"
        _write_scenario_plot(path, scenario, scenario_summary, scenario_samples, scenario_bins, config)
        paths[f"{scenario.probe_id}/recovery_probe"] = str(path)
    path = output_dir / f"{probe_name}_latest_recovery_grid.png"
    _write_latest_recovery_grid(path, scenarios, summary_rows)
    paths["latest_recovery_grid"] = str(path)
    return paths


def _write_scenario_plot(
    path: Path,
    scenario: RecoveryScenario,
    summary_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    config: Any,
) -> None:
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    steps = np.asarray([int(row["step"]) for row in summary_rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    _plot_mean_with_std(axes[0, 0], steps, summary_rows, "recovery_x", "Chosen recovery x", "m")
    _plot_mean_with_std(axes[0, 1], steps, summary_rows, "recovery_y", "Chosen recovery y", "m")
    _plot_line(axes[1, 0], steps, summary_rows, "policy_entropy", "Recovery policy entropy", "nats")
    _plot_latest_recovery_distribution(axes[1, 1], scenario, sample_rows, bin_rows, config)
    fig.suptitle(scenario.title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_mean_with_std(ax: Any, steps: np.ndarray, rows: list[dict[str, Any]], key: str, title: str, ylabel: str) -> None:
    mean = np.asarray([_nan(row.get(f"{key}_mean")) for row in rows], dtype=float)
    std = np.asarray([_nan(row.get(f"{key}_std")) for row in rows], dtype=float)
    ax.plot(steps, mean, marker="o", linewidth=1.8)
    if np.isfinite(std).any():
        ax.fill_between(steps, mean - std, mean + std, alpha=0.18)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _plot_line(ax: Any, steps: np.ndarray, rows: list[dict[str, Any]], key: str, title: str, ylabel: str) -> None:
    values = np.asarray([_nan(row.get(key)) for row in rows], dtype=float)
    ax.plot(steps, values, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _plot_latest_recovery_distribution(
    ax: Any,
    scenario: RecoveryScenario,
    sample_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    config: Any,
) -> None:
    if not bin_rows:
        ax.text(0.5, 0.5, "no recovery bins", transform=ax.transAxes, ha="center", va="center")
        return
    setup_court_axes(ax, config, stage_colors(monochrome=True), show_axes=True)
    latest_step = max(int(row["step"]) for row in bin_rows)
    latest_bins = [row for row in bin_rows if int(row["step"]) == latest_step]
    probabilities = np.asarray([float(row["policy_probability"]) for row in latest_bins], dtype=float)
    scatter = ax.scatter(
        [float(row["recovery_x"]) for row in latest_bins],
        [float(row["recovery_y"]) for row in latest_bins],
        c=probabilities,
        cmap="viridis",
        s=260 * np.maximum(probabilities, 0.02),
        alpha=0.85,
        edgecolors="black",
        linewidths=0.35,
        zorder=5,
        label="policy bins",
    )
    latest_samples = [row for row in sample_rows if int(row["step"]) == latest_step]
    if latest_samples:
        ax.scatter(
            [float(row["recovery_x"]) for row in latest_samples],
            [float(row["recovery_y"]) for row in latest_samples],
            color="white",
            edgecolors="tab:blue",
            s=18,
            alpha=0.45,
            linewidths=0.5,
            zorder=6,
            label="samples",
    )
    best_score = max(float(row["score"]) for row in latest_bins)
    best_score_bins = [
        row
        for row in latest_bins
        if np.isclose(float(row["score"]), best_score, rtol=0.0, atol=1e-9)
    ]
    best_xs = [float(row["recovery_x"]) for row in best_score_bins]
    best_ys = [float(row["recovery_y"]) for row in best_score_bins]
    ax.scatter(
        best_xs,
        best_ys,
        marker="X",
        s=95,
        color="tab:cyan",
        edgecolors="black",
        linewidths=0.6,
        zorder=8,
        label="best score bins",
    )
    if best_score_bins:
        label_row = max(best_score_bins, key=lambda row: float(row["policy_probability"]))
        ax.annotate(
            f"best score\n{best_score:.2f} ({len(best_score_bins)} bins)",
            xy=(float(label_row["recovery_x"]), float(label_row["recovery_y"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            color="black",
            zorder=9,
        )
    target = scenario.intercept_point
    ax.scatter([target[0]], [target[1]], marker="*", s=120, color="tab:red", zorder=7, label="opponent contact")
    step_label = f"{latest_step / 1_000_000:g}M" if latest_step >= 1_000_000 else str(latest_step)
    ax.set_title(f"Latest recovery, step {step_label}")
    ax.figure.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02, label="policy probability")
    ax.legend(fontsize=7)


def _write_latest_recovery_grid(
    path: Path,
    scenarios: list[RecoveryScenario],
    summary_rows: list[dict[str, Any]],
) -> None:
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

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
        for scenario in scenarios:
            meta = scenario.metadata
            if meta.get("target_z_level") != z_label:
                continue
            row = latest_by_probe.get(scenario.probe_id)
            if row is None:
                continue
            y_index = y_labels.index(str(meta["target_y_region"]))
            x_index = x_labels.index(str(meta["target_x_region"]))
            values[y_index, x_index] = _nan(row.get("policy_expected_rank_fraction"))
        arrays.append(values)

    image = None
    for ax, z_label, values in zip(axes, z_labels, arrays):
        image = ax.imshow(values, cmap="magma_r", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_title(f"{z_label} contact")
        ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
        ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
        for y_index in range(values.shape[0]):
            for x_index in range(values.shape[1]):
                value = values[y_index, x_index]
                text = "n/a" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(x_index, y_index, text, ha="center", va="center", fontsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="expected rank fraction")
    fig.suptitle("Latest recovery choice quality by opponent contact target")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(np.std(np.asarray(values, dtype=float)))


def _entropy(probabilities: list[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


if __name__ == "__main__":
    main()
