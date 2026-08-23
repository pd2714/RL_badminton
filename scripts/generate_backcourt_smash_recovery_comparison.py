from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

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
from badminton.eval_evolution import (
    build_discrete_action_config,
    build_sim_config,
    checkpoint_step,
    discover_anchor_checkpoints,
    filter_anchor_checkpoints,
    load_anchor_model,
    load_run_config,
)
from badminton.mpl_config import ensure_writable_matplotlib_config
from badminton.obs import ObservationConfig, ObservationEncoder
from badminton.utils import default_player_position, ensure_directory, opponent_side
from scripts.evaluate_recovery_choice_probe import (
    RecoveryScenario,
    _fixed_hitter_state,
    _recovery_bin_rows,
    _sample_recovery_bins,
    _score_recovery_bins,
    _summarize_recovery_samples,
    _target_x_positions,
    _target_y_positions,
    _write_csv,
)
from scripts.plot_recovery_top_choice_evolution_3d import (
    _load_bins_by_probe,
    _plot_scenario,
    _render_probe_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a focused recovery-choice evolution probe for a high backcourt-left "
            "smash, comparing straight negative-x and cross-court positive-x targets."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Self-play run directory containing selfplay_config.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/anchor_metric_eval/backcourt_left_high_smash_recovery_comparison.",
    )
    parser.add_argument("--probe-name", type=str, default="backcourt_left_high_smash_recovery_comparison")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--anchor-step-min", type=int, default=0)
    parser.add_argument("--anchor-step-max", type=int, default=6_000_000)
    parser.add_argument("--anchor-step-interval", type=int, default=1_000_000)
    parser.add_argument("--counterfactual-opponent-response-samples", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_writable_matplotlib_config()

    output_dir = args.output_dir or (
        args.run_dir / "anchor_metric_eval" / "backcourt_left_high_smash_recovery_comparison"
    )
    ensure_directory(output_dir)

    run_config = load_run_config(args.run_dir)
    sim_config = build_sim_config(run_config)
    discrete_config = build_discrete_action_config(run_config)
    policy_type = str(run_config.get("policy_type", "velocity_oriented"))
    train_side = str(run_config.get("train_side", "left"))
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

    scenarios = _build_smash_scenarios(
        mapper=mapper,
        config=sim_config,
        train_side=train_side,
        reaction_time=reaction_time,
    )
    probe_metadata = {
        "probe_name": args.probe_name,
        "description": (
            "Two-scenario recovery probe for a train-agent smash from backcourt_left_high. "
            "The opponent starts at the middle recovery position; only the smash target x "
            "changes between straight negative x and cross-court positive x."
        ),
        "run_dir": str(args.run_dir),
        "train_side": train_side,
        "policy_type": policy_type,
        "sample_count_per_anchor": int(args.samples),
        "deterministic": False,
        "counterfactual_opponent_response_samples": int(args.counterfactual_opponent_response_samples),
        "seed": int(args.seed),
        "scenarios": [scenario.metadata for scenario in scenarios],
    }
    state_path = output_dir / f"{args.probe_name}_probe_state.json"
    state_path.write_text(json.dumps(probe_metadata, indent=2), encoding="utf-8")

    checkpoints = filter_anchor_checkpoints(
        discover_anchor_checkpoints(args.run_dir),
        step_min=int(args.anchor_step_min),
        step_max=int(args.anchor_step_max),
        step_interval=int(args.anchor_step_interval),
    )

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
                samples=int(args.samples),
                deterministic=False,
                rng=rng,
            )
            sample_rows.extend(rows)
            summary_rows.append(_summarize_recovery_samples(scenario, step, checkpoint, bins, rows))
            bin_rows.extend(_recovery_bin_rows(scenario, step, checkpoint, bins))

    _write_csv(output_dir / f"{args.probe_name}_probe_samples.csv", sample_rows)
    _write_csv(output_dir / f"{args.probe_name}_probe_summary.csv", summary_rows)
    bins_path = output_dir / f"{args.probe_name}_probe_bins.csv"
    _write_csv(bins_path, bin_rows)
    report = {**probe_metadata, "output_dir": str(output_dir), "rows": summary_rows}
    (output_dir / f"{args.probe_name}_probe_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    _render_probe_dir(output_dir, run_dir=args.run_dir, output_subdir="top_recovery_evolution_3d_views", dpi=int(args.dpi))
    comparison_path = output_dir / "backcourt_left_high_smash_recovery_evolution_3d_comparison.png"
    _write_comparison_plot(
        comparison_path,
        scenarios=[scenario.metadata for scenario in scenarios],
        bins_path=bins_path,
        config=sim_config,
        dpi=int(args.dpi),
    )

    print(f"state: {state_path}")
    print(f"bins: {bins_path}")
    print(f"comparison: {comparison_path}")


def _build_smash_scenarios(
    *,
    mapper: DiscreteActionMapper,
    config: Any,
    train_side: str,
    reaction_time: float,
) -> list[RecoveryScenario]:
    x_lookup = dict(_target_x_positions(config))
    hitter_y_lookup = dict(_target_y_positions(train_side, config))
    receiver_y_lookup = dict(_target_y_positions(opponent_side(train_side), config))
    _, opponent_y = default_player_position(opponent_side(train_side), config)
    opponent_x = 0.0

    hitter_x = float(x_lookup["left"])
    hitter_y = float(hitter_y_lookup["backcourt"])
    hitter_z = 2.5
    target_y = float(opponent_y)
    target_z = 1.1

    specs = (
        ("straight_negative_x_smash", "negative x straight smash", "left", float(x_lookup["left"])),
        ("cross_positive_x_smash", "positive x cross-court smash", "right", float(x_lookup["right"])),
    )
    scenarios: list[RecoveryScenario] = []
    for probe_id, title_suffix, x_region, target_x in specs:
        state = _fixed_hitter_state(
            train_side=train_side,
            config=config,
            hitter_x=hitter_x,
            hitter_y=hitter_y,
            hitter_z=hitter_z,
            opponent_x=float(opponent_x),
            opponent_y=float(opponent_y),
            reaction_time=float(reaction_time),
            stage_index=5,
        )
        shot = _find_fixed_smash_to_target(
            mapper=mapper,
            state=state,
            train_side=train_side,
            config=config,
            target=(target_x, target_y, target_z),
        )
        speed = float(math.sqrt(shot["action"].v_x ** 2 + shot["action"].v_y ** 2 + shot["action"].v_z ** 2))
        horizontal_speed = float(math.hypot(shot["action"].v_x, shot["action"].v_y))
        theta_degrees = float(math.degrees(math.atan2(shot["action"].v_z, horizontal_speed)))
        metadata = {
            "probe_id": probe_id,
            "target_y_region": "middle_recovery",
            "target_x_region": x_region,
            "target_z_level": "smash_contact",
            "target_point": {"x": target_x, "y": target_y, "z": target_z},
            "actual_intercept_point": [float(value) for value in shot["intercept_point"]],
            "target_distance": float(shot["target_distance"]),
            "intercept_index": int(shot["intercept_index"]),
            "fixed_action": asdict(shot["action"]),
            "shot_component_action": int(shot["shot_component_action"]),
            "state_before": asdict(state),
            "hitter_contact_region": "backcourt_left_high",
            "opponent_start_region": "middle_recovery",
            "opponent_start": {"x": float(opponent_x), "y": float(opponent_y)},
            "smash_constraints": {"min_speed": 35.0, "max_theta_degrees": -5.0},
            "shot_speed": speed,
            "shot_theta_degrees": theta_degrees,
            "shot_landing_point": {"x": float(shot["landing_point"][0]), "y": float(shot["landing_point"][1])},
        }
        scenarios.append(
            RecoveryScenario(
                probe_id=probe_id,
                title=f"Recovery after {title_suffix} from backcourt_left_high",
                state_before=state,
                fixed_action=shot["action"],
                intercept_index=int(shot["intercept_index"]),
                intercept_point=tuple(float(value) for value in shot["intercept_point"]),
                target_point=(target_x, target_y, target_z),
                shot_component_action=int(shot["shot_component_action"]),
                metadata=metadata,
            )
        )
    return scenarios


def _find_fixed_smash_to_target(
    *,
    mapper: DiscreteActionMapper,
    state: Any,
    train_side: str,
    config: Any,
    target: tuple[float, float, float],
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

                action = validated.applied
                horizontal_speed = float(math.hypot(action.v_x, action.v_y))
                speed = float(math.sqrt(horizontal_speed * horizontal_speed + action.v_z * action.v_z))
                theta_degrees = float(math.degrees(math.atan2(action.v_z, horizontal_speed)))
                if speed < 35.0 or theta_degrees > -5.0:
                    continue

                _, xs, ys, zs = candidate_intercept_points(state, action, config)
                landing = landing_position(state, action, config)
                for intercept_index in feasible:
                    if not 0 <= int(intercept_index) < len(xs):
                        continue
                    intercept = (
                        float(xs[int(intercept_index)]),
                        float(ys[int(intercept_index)]),
                        float(zs[int(intercept_index)]),
                    )
                    distance = math.sqrt(
                        2.0 * (intercept[0] - target[0]) ** 2
                        + (intercept[1] - target[1]) ** 2
                        + 0.5 * (intercept[2] - target[2]) ** 2
                    )
                    score = (
                        distance,
                        -speed,
                        abs(theta_degrees),
                        abs(float(landing[0]) - target[0]),
                    )
                    if best is None or score < best["sort_score"]:
                        component_action = flat_action - (flat_action % (x_rec_count * y_rec_count))
                        best = {
                            "action": action,
                            "intercept_index": int(intercept_index),
                            "intercept_point": intercept,
                            "target_distance": float(distance),
                            "shot_component_action": int(component_action),
                            "landing_point": (float(landing[0]), float(landing[1])),
                            "sort_score": score,
                        }
    if best is None:
        raise RuntimeError(f"No feasible downward high-speed fixed smash found for target={target}")
    return best


def _write_comparison_plot(
    path: Path,
    *,
    scenarios: list[dict[str, Any]],
    bins_path: Path,
    config: Any,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    bins = _load_bins_by_probe(bins_path)
    fig = plt.figure(figsize=(16.4, 7.4), constrained_layout=True)
    mappable = None
    titles = {
        "straight_negative_x_smash": "Straight smash to negative x",
        "cross_positive_x_smash": "Cross-court smash to positive x",
    }
    for index, scenario in enumerate(scenarios, start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        probe_id = str(scenario["probe_id"])
        mappable = _plot_scenario(ax, scenario, bins.get(probe_id, []), config, compact=False)
        opponent_start = scenario.get("opponent_start")
        if isinstance(opponent_start, dict):
            start_x = float(opponent_start["x"])
            start_y = float(opponent_start["y"])
            ax.scatter(
                [start_x],
                [start_y],
                [0.08],
                marker="^",
                color="tab:orange",
                edgecolors="black",
                s=72,
                depthshade=False,
                zorder=22,
            )
            ax.text(start_x, start_y, 0.36, "opponent start", color="tab:orange", fontsize=8, ha="center")
        ax.set_title(titles.get(probe_id, probe_id.replace("_", " ")), fontsize=12)
    if mappable is not None:
        colorbar = fig.colorbar(
            mappable,
            ax=fig.axes,
            location="top",
            fraction=0.045,
            pad=0.02,
            shrink=0.56,
            label="checkpoint step (M)",
        )
        colorbar.set_ticks(np.arange(7, dtype=float))
    fig.suptitle(
        "Recovery-choice evolution after high backcourt-left smash",
        fontsize=15,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
