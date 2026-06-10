from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionMapper
from badminton1d.eval_evolution import (
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
from badminton1d.obs import ObservationConfig, ObservationEncoder
from badminton1d.pressure import shot_pressure_from_record
from badminton1d.state import Side
from badminton1d.utils import ensure_directory, opponent_side

from scripts.evaluate_controlled_lift_probe import (
    ProbeScenario,
    _after_shot_win_probabilities,
    _build_contact_grid_scenarios,
    _plot_anchor_critic_value_delta,
    _plot_critic_value_delta,
    _plot_landing_scatter,
    _plot_mean_with_std,
    _plot_stacked_summary,
    _field_name,
    _model_win_probability,
    _response_record,
    _write_stacked_plot,
    _write_csv,
)
from scripts.plot_controlled_contact_top_shot_trajectories_3d import (
    _expand_scenarios_over_opponent_recovery_grid,
    _stage_state_from_dict,
    _top_shots_for_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render evolution-style opponent-position probe plots using weighted "
            "expectations over each checkpoint's top-3 shots."
        )
    )
    parser.add_argument("probe_dir", type=Path, help="controlled_contact_grid_probe directory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Defaults to probe summary metadata.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--anchor-step-min", type=int, default=None)
    parser.add_argument("--anchor-step-max", type=int, default=None)
    parser.add_argument("--anchor-step-interval", type=int, default=None)
    parser.add_argument("--contact-state", action="append", default=None)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--write-cache-only",
        action="store_true",
        help="Compute CSV/JSON cache but skip PNG rendering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    probe_dir = args.probe_dir
    probe_summary = _load_probe_summary(probe_dir)
    run_dir = args.run_dir or Path(str(probe_summary.get("run_dir")))
    run_config = load_run_config(run_dir)
    config = build_sim_config(run_config)
    discrete_config = build_discrete_action_config(run_config)
    policy_type = str(run_config.get("policy_type", "velocity_oriented"))
    train_side: Side = str(run_config.get("train_side", probe_summary.get("train_side", "left")))  # type: ignore[assignment]
    receiver_side = opponent_side(train_side)

    mapper = DiscreteActionMapper(config, discrete_config, policy_type=policy_type)
    encoder = ObservationEncoder(
        config,
        ObservationConfig(
            max_score=1,
            max_stages_per_rally=int(run_config.get("max_rally_stages", 120) or 120),
            include_feasible_mask=bool(run_config.get("include_feasible_mask", True)),
            include_reaction_risk_features=bool(run_config.get("include_reaction_risk_features", True)),
        ),
    )

    value_checkpoint = _value_checkpoint_from_summary(probe_summary)
    value_model = None if value_checkpoint is None else load_anchor_model(value_checkpoint, recovery_choice_diagnostics=False)
    scenarios = _expanded_scenarios(probe_dir, probe_summary, train_side, config)
    if args.contact_state:
        selected = set(args.contact_state)
        scenarios = [scenario for scenario in scenarios if str(scenario.metadata["contact_probe_id"]) in selected]
        missing = selected.difference(str(scenario.metadata["contact_probe_id"]) for scenario in scenarios)
        if missing:
            raise ValueError(f"Unknown contact-state probe_id(s): {', '.join(sorted(missing))}")

    checkpoints = _matching_checkpoints(run_dir, probe_dir, args)
    cache_dir = probe_dir / "top3_expectation_evolution_probe_views"
    ensure_directory(cache_dir)

    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        print(f"top3 expectation anchor_step_{step}", flush=True)
        model = load_anchor_model(checkpoint, recovery_choice_diagnostics=False)
        critic_model = value_model or model
        for scenario in scenarios:
            rows, summary = _top3_expected_rows_for_scenario(
                model=model,
                critic_model=critic_model,
                mapper=mapper,
                encoder=encoder,
                scenario=scenario,
                train_side=train_side,
                receiver_side=receiver_side,
                config=config,
                checkpoint=checkpoint,
                value_checkpoint=value_checkpoint,
                top_k=int(args.top_k),
            )
            sample_rows.extend(rows)
            summary_rows.append(summary)

    _write_csv(cache_dir / "top3_expectation_evolution_samples.csv", sample_rows)
    _write_csv(cache_dir / "top3_expectation_evolution_summary.csv", summary_rows)
    (cache_dir / "top3_expectation_evolution_summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "probe_dir": str(probe_dir),
                "train_side": train_side,
                "top_k": int(args.top_k),
                "value_checkpoint_path": None if value_checkpoint is None else str(value_checkpoint),
                "value_checkpoint_step": None if value_checkpoint is None else checkpoint_step(value_checkpoint),
                "scenario_count": len(scenarios),
                "checkpoint_count": len(checkpoints),
                "rows": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.write_cache_only:
        _write_evolution_plots(probe_dir, scenarios, summary_rows, sample_rows, config)


def _load_probe_summary(probe_dir: Path) -> dict[str, Any]:
    path = probe_dir / "controlled_contact_grid_probe_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing probe summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _value_checkpoint_from_summary(summary: dict[str, Any]) -> Path | None:
    value = summary.get("value_checkpoint_path")
    if value:
        return Path(str(value))
    probe_value = summary.get("probe", {}).get("value_checkpoint_path")
    if probe_value:
        return Path(str(probe_value))
    return None


def _expanded_scenarios(
    probe_dir: Path,
    probe_summary: dict[str, Any],
    train_side: Side,
    config: Any,
) -> list[ProbeScenario]:
    state_path = probe_dir / "controlled_contact_grid_probe_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        scenario_dicts = _expand_scenarios_over_opponent_recovery_grid(
            list(state["scenarios"]),
            opponent_side(train_side),
            config,
        )
        return [
            ProbeScenario(
                probe_id=str(item["probe_id"]),
                title=str(item["probe_id"]).replace("__", " / ").replace("_", " "),
                response_state=_stage_state_from_dict(item["response_state"]),
                metadata={
                    **item,
                    "preset": "contact-grid",
                    "probe_id": str(item["probe_id"]),
                    "contact_probe_id": str(item["contact_probe_id"]),
                    "opponent_cell_id": str(item["opponent_cell_id"]),
                },
            )
            for item in scenario_dicts
        ]

    reaction_time = float(probe_summary.get("probe", {}).get("reaction_time", 0.0) or 0.0)
    stage_index = 5
    for item in probe_summary.get("probe", {}).get("scenarios", []):
        response_state = item.get("response_state", {})
        if "stage_index" in response_state:
            stage_index = int(response_state["stage_index"])
            break
    base_scenarios = _build_contact_grid_scenarios(
        train_side=train_side,
        config=config,
        reaction_time=reaction_time,
        stage_index=stage_index,
    )
    scenario_dicts = [
        {
            **scenario.metadata,
            "probe_id": scenario.probe_id,
            "response_state": scenario.metadata["response_state"],
        }
        for scenario in base_scenarios
    ]
    expanded = _expand_scenarios_over_opponent_recovery_grid(scenario_dicts, opponent_side(train_side), config)
    return [
        ProbeScenario(
            probe_id=str(item["probe_id"]),
            title=str(item["probe_id"]).replace("__", " / ").replace("_", " "),
            response_state=_stage_state_from_dict(item["response_state"]),
            metadata=item,
        )
        for item in expanded
    ]


def _matching_checkpoints(run_dir: Path, probe_dir: Path, args: argparse.Namespace) -> list[Path]:
    checkpoints = discover_anchor_checkpoints(run_dir)
    checkpoints = filter_anchor_checkpoints(
        checkpoints,
        step_min=args.anchor_step_min,
        step_max=args.anchor_step_max,
        step_interval=args.anchor_step_interval,
    )
    if args.anchor_step_min is not None or args.anchor_step_max is not None or args.anchor_step_interval is not None:
        return checkpoints

    summary_csv = probe_dir / "controlled_contact_grid_probe_summary.csv"
    if not summary_csv.exists():
        return checkpoints
    with summary_csv.open(encoding="utf-8") as handle:
        wanted_steps = {int(row["step"]) for row in csv.DictReader(handle) if row.get("step")}
    return [checkpoint for checkpoint in checkpoints if checkpoint_step(checkpoint) in wanted_steps]


def _top3_expected_rows_for_scenario(
    *,
    model: Any,
    critic_model: Any,
    mapper: DiscreteActionMapper,
    encoder: ObservationEncoder,
    scenario: ProbeScenario,
    train_side: Side,
    receiver_side: Side,
    config: Any,
    checkpoint: Path,
    value_checkpoint: Path | None,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    step = checkpoint_step(checkpoint)
    before_value = _model_win_probability(
        critic_model,
        encoder,
        scenario.response_state,
        agent_side=train_side,
        role="hitter",
        server_side=train_side,
    )
    anchor_before_value = _model_win_probability(
        model,
        encoder,
        scenario.response_state,
        agent_side=train_side,
        role="hitter",
        server_side=train_side,
    )
    top_shots = _top_shots_for_state(
        model=model,
        mapper=mapper,
        encoder=encoder,
        state=scenario.response_state,
        agent_side=train_side,
        server_side=train_side,
        config=config,
        top_k=top_k,
    )
    probabilities = np.asarray([float(row["probability"]) for row in top_shots], dtype=float)
    top_mass = float(np.sum(probabilities))
    weights = probabilities / top_mass if top_mass > 0.0 else np.full(len(top_shots), 1.0 / max(len(top_shots), 1))

    rng = np.random.default_rng(20260604 + step + _stable_index(str(scenario.probe_id)))
    rows: list[dict[str, Any]] = []
    after_states = []
    pending_indices = []
    for index, (weight, shot) in enumerate(zip(weights, top_shots)):
        decoded = mapper.decode_hitter_for_agent(int(shot["action"]), scenario.response_state, train_side)
        projected = mapper.project_hitter_action(scenario.response_state, decoded.shot_action)
        action = projected.shot_action
        record = _response_record(scenario.response_state, action, config, rng=rng)
        pressure = shot_pressure_from_record(record, config)
        zone = landing_zone_name(receiver_side, (float(shot["landing_x"]), float(shot["landing_y"])), config)
        pending_indices.append(len(rows))
        after_states.append(record.next_state)
        rows.append(
            {
                **_scenario_fields(scenario),
                "step": step,
                "checkpoint_path": str(checkpoint),
                "value_checkpoint_path": None if value_checkpoint is None else str(value_checkpoint),
                "sample_index": index,
                "rank": int(shot["rank"]),
                "valid": True,
                "top3_weight": float(weight),
                "top3_mass": top_mass,
                "projected": bool(shot["projected"]),
                "shot_speed": float(np.linalg.norm([float(shot["v_x"]), float(shot["v_y"]), float(shot["v_z"])])),
                "pressure": float(pressure.pressure),
                "pressure_required_speed_score": float(pressure.required_speed_score),
                "pressure_intercept_scarcity_score": float(pressure.intercept_scarcity_score),
                "pressure_low_contact_score": float(pressure.low_contact_score),
                "pressure_reaction_miss_score": float(pressure.reaction_miss_score),
                "shot_value": None,
                "before_win_probability": float(before_value),
                "after_win_probability": None,
                "anchor_shot_value": None,
                "anchor_before_win_probability": float(anchor_before_value),
                "anchor_after_win_probability": None,
                "shot_type": str(shot["shot_type"]),
                "landing_zone": zone,
                "landing_x": float(shot["landing_x"]),
                "landing_y": float(shot["landing_y"]),
                "v_x": float(shot["v_x"]),
                "v_y": float(shot["v_y"]),
                "v_z": float(shot["v_z"]),
                "terminal_reason": record.terminal_reason,
            }
        )

    after_values = _after_shot_win_probabilities(
        critic_model,
        encoder,
        after_states,
        train_side=train_side,
        server_side=train_side,
    )
    anchor_after_values = _after_shot_win_probabilities(
        model,
        encoder,
        after_states,
        train_side=train_side,
        server_side=train_side,
    )
    for row_index, value in zip(pending_indices, after_values):
        rows[row_index]["after_win_probability"] = float(value)
        rows[row_index]["shot_value"] = float(value - before_value)
    for row_index, value in zip(pending_indices, anchor_after_values):
        rows[row_index]["anchor_after_win_probability"] = float(value)
        rows[row_index]["anchor_shot_value"] = float(value - anchor_before_value)

    summary = _weighted_summary(step, checkpoint, rows, before_value, anchor_before_value, scenario)
    return rows, summary


def _weighted_summary(
    step: int,
    checkpoint: Path,
    rows: list[dict[str, Any]],
    before_value: float,
    anchor_before_value: float,
    scenario: ProbeScenario,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("valid")]
    weights = np.asarray([float(row.get("top3_weight", 0.0)) for row in valid_rows], dtype=float)
    if weights.size and float(np.sum(weights)) > 0.0:
        weights = weights / float(np.sum(weights))
    summary: dict[str, Any] = {
        **_scenario_fields(scenario),
        "step": step,
        "checkpoint_path": str(checkpoint),
        "value_checkpoint_path": next((row.get("value_checkpoint_path") for row in rows), None),
        "sample_count": len(rows),
        "valid_sample_count": len(valid_rows),
        "invalid_sample_count": len(rows) - len(valid_rows),
        "top3_mass": float(next((row.get("top3_mass") for row in valid_rows), 0.0) or 0.0),
        "before_win_probability": before_value,
        "anchor_before_win_probability": anchor_before_value,
    }
    for key in (
        "shot_speed",
        "pressure",
        "pressure_reaction_miss_score",
        "shot_value",
        "after_win_probability",
        "anchor_shot_value",
        "anchor_after_win_probability",
    ):
        values = np.asarray([float(row[key]) for row in valid_rows if row.get(key) is not None], dtype=float)
        if values.size == weights.size and values.size:
            mean = float(np.dot(weights, values))
            std = float(math.sqrt(max(float(np.dot(weights, (values - mean) ** 2)), 0.0)))
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None

    shot_type_weights: dict[str, float] = defaultdict(float)
    landing_zone_weights: dict[str, float] = defaultdict(float)
    for weight, row in zip(weights, valid_rows):
        shot_type_weights[str(row["shot_type"])] += float(weight)
        landing_zone_weights[str(row["landing_zone"])] += float(weight)
    for name in SHOT_TYPE_ORDER:
        summary[f"shot_type_freq_{_field_name(name)}"] = float(shot_type_weights.get(name, 0.0))
    for name in LANDING_ZONE_NAMES:
        summary[f"landing_zone_freq_{name}"] = float(landing_zone_weights.get(name, 0.0))
    return summary


def _write_evolution_plots(
    probe_dir: Path,
    scenarios: list[ProbeScenario],
    summary_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    config: Any,
) -> None:
    for scenario in scenarios:
        contact_id = str(scenario.metadata["contact_probe_id"])
        opponent_id = str(scenario.metadata["opponent_cell_id"])
        scenario_dir = probe_dir / contact_id / opponent_id
        ensure_directory(scenario_dir)
        scenario_summary_rows = [row for row in summary_rows if row.get("probe_id") == scenario.probe_id]
        scenario_sample_rows = [row for row in sample_rows if row.get("probe_id") == scenario.probe_id]
        scenario_summary_rows.sort(key=lambda row: int(row["step"]))
        scenario_sample_rows.sort(key=lambda row: (int(row["step"]), int(row.get("rank", 0))))
        paths = _write_plots_with_trajectory_panel(
            scenario_dir,
            scenario_summary_rows,
            scenario_sample_rows,
            config,
            scenario.metadata,
            plot_prefix=scenario.probe_id,
            plot_title=f"{scenario.title} top-3 expectation evolution",
        )
        for name, path in paths.items():
            print(f"{scenario.probe_id}/{name}: {path}", flush=True)


def _write_plots_with_trajectory_panel(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    config: Any,
    probe_metadata: dict[str, Any],
    *,
    plot_prefix: str,
    plot_title: str,
) -> dict[str, str]:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    steps = np.asarray([int(row["step"]) for row in summary_rows], dtype=float)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 14.5), constrained_layout=True)
    _plot_mean_with_std(axes[0, 0], steps, summary_rows, "shot_speed", "Average chosen shot speed", "m/s")
    _plot_mean_with_std(axes[0, 1], steps, summary_rows, "pressure", "Miss-aware pressure created", "index")
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
    _plot_trajectory_image_panel(axes[3, 1], output_dir / f"{plot_prefix}_top3_shot_trajectories_3d.png")
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


def _plot_trajectory_image_panel(ax: Any, trajectory_path: Path) -> None:
    import matplotlib.image as mpimg

    if trajectory_path.exists():
        image = mpimg.imread(trajectory_path)
        ax.imshow(image)
        ax.set_title("Latest top-3 shot trajectories")
        ax.axis("off")
        return
    ax.text(0.5, 0.5, "top-3 trajectory image not found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Latest top-3 shot trajectories")
    ax.axis("off")


def _scenario_fields(scenario: ProbeScenario) -> dict[str, Any]:
    fields = {
        "probe_id": scenario.probe_id,
        "probe_title": scenario.title,
        "contact_probe_id": scenario.metadata.get("contact_probe_id"),
        "opponent_cell_id": scenario.metadata.get("opponent_cell_id"),
    }
    for key in ("preset", "x_region", "y_region", "z_level"):
        if key in scenario.metadata:
            fields[key] = scenario.metadata[key]
    contact = scenario.metadata.get("contact_point")
    if isinstance(contact, dict):
        fields["contact_x"] = float(contact["x"])
        fields["contact_y"] = float(contact["y"])
        fields["contact_z"] = float(contact["z"])
    return fields


def _stable_index(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 131 + ord(char)) % 1_000_003
    return total


if __name__ == "__main__":
    main()
