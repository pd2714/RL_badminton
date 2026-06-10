from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionMapper
from badminton1d.dynamics import landing_position, validate_and_clip_shot_action
from badminton1d.eval_evolution import (
    build_discrete_action_config,
    build_sim_config,
    checkpoint_step,
    discover_anchor_checkpoints,
    load_anchor_model,
    load_run_config,
)
from badminton1d.evaluation import adapt_observation_to_model
from badminton1d.mpl_config import ensure_writable_matplotlib_config
from badminton1d.obs import ObservationConfig, ObservationEncoder
from badminton1d.render import (
    COURT_LINE_Z,
    COURT_SURFACE_Z,
    GROUND_MARKER_Z,
    OFFICIAL_DOUBLES_WIDTH,
    OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK,
    OFFICIAL_SHORT_SERVICE_FROM_NET,
    OFFICIAL_SINGLES_WIDTH,
    stage_colors,
)
from badminton1d.shot_generators import name_velocity_shot
from badminton1d.state import Side, StageState
from badminton1d.trajectory import simulate_trajectory
from badminton1d.utils import ensure_directory, opponent_side, recovery_bounds

TRAJECTORY_VIEW_ELEV = 18.0
TRAJECTORY_VIEW_AZIM = -62.0
TRAJECTORY_VIEW_DIST = 6.2
TRAJECTORY_VIEW_ZOOM = 1.35
TRAJECTORY_COLORBAR_PAD = 0.035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the top likely unique shot trajectories for the latest checkpoint "
            "in a controlled_contact_grid_probe directory."
        )
    )
    parser.add_argument(
        "probe_dir",
        type=Path,
        help="Directory containing controlled_contact_grid_probe_state.json.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing selfplay_config.json. Defaults to metadata or probe_dir/../...",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to visualize. Defaults to the largest numeric step in the probe summary, then run anchors.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--probe-id",
        action="append",
        default=None,
        help="Probe id to render. Can be passed more than once. Defaults to all probes.",
    )
    parser.add_argument(
        "--output-name-suffix",
        type=str,
        default="top10_shot_trajectories_3d",
        help="File suffix used inside each scenario subfolder.",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Also write a compact 27-panel overview under probe_dir/top_shot_3d_views.",
    )
    parser.add_argument(
        "--opponent-recovery-grid-3x3",
        action="store_true",
        help=(
            "Expand each contact state over a 3x3 opponent-position grid on the opponent court. "
            "The cells are the 1st, 3rd, and 5th rows/columns of the 5x5 recovery grid."
        ),
    )
    parser.add_argument(
        "--opponent-grid-side",
        choices=("left", "right"),
        default=None,
        help="Court side for the opponent-position grid. Defaults to the opponent of train_side.",
    )
    parser.add_argument(
        "--opponent-velocity-variant",
        action="append",
        choices=("neg_y", "pos_y", "pos_x", "neg_x"),
        default=None,
        help=(
            "Append an opponent velocity variant to each selected scenario before evaluating "
            "top shots. Can be passed more than once."
        ),
    )
    parser.add_argument(
        "--opponent-speed",
        type=float,
        default=5.0,
        help="Opponent speed in m/s used by --opponent-velocity-variant. Defaults to 5.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    probe_dir = args.probe_dir
    state_path = _probe_state_path(probe_dir)
    probe_state = json.loads(state_path.read_text(encoding="utf-8"))
    summary = _load_summary(probe_dir)
    run_dir = args.run_dir or _infer_run_dir(probe_dir, summary)
    run_config = load_run_config(run_dir)
    config = build_sim_config(run_config)
    discrete_config = build_discrete_action_config(run_config)
    policy_type = str(run_config.get("policy_type", "velocity_oriented"))
    train_side: Side = str(run_config.get("train_side", summary.get("train_side", "left")))  # type: ignore[assignment]

    checkpoint = args.checkpoint or _latest_probe_checkpoint(probe_dir, run_dir)
    model = load_anchor_model(checkpoint, recovery_choice_diagnostics=False)
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

    scenarios = list(probe_state["scenarios"])
    if args.opponent_recovery_grid_3x3:
        grid_side: Side = args.opponent_grid_side or opponent_side(train_side)  # type: ignore[assignment]
        scenarios = _expand_scenarios_over_opponent_recovery_grid(scenarios, grid_side, config)
    if args.probe_id is not None:
        selected_probe_ids = set(args.probe_id)
        scenarios = [scenario for scenario in scenarios if str(scenario["probe_id"]) in selected_probe_ids]
    if args.opponent_velocity_variant is not None:
        scenarios = _expand_scenarios_over_opponent_velocity(
            scenarios,
            directions=tuple(args.opponent_velocity_variant),
            speed=float(args.opponent_speed),
        )
    manifest: dict[str, Any] = {
        "probe_dir": str(probe_dir),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": checkpoint_step(checkpoint),
        "top_k": int(args.top_k),
        "opponent_recovery_grid_3x3": bool(args.opponent_recovery_grid_3x3),
        "opponent_velocity_variants": list(args.opponent_velocity_variant or []),
        "opponent_speed": float(args.opponent_speed),
        "plots": {},
        "top_shots": {},
    }
    overview_items: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for scenario in scenarios:
        probe_id = str(scenario["probe_id"])
        state = _stage_state_from_dict(scenario["response_state"])
        top_shots = _top_shots_for_state(
            model=model,
            mapper=mapper,
            encoder=encoder,
            state=state,
            agent_side=train_side,
            server_side=train_side,
            config=config,
            top_k=int(args.top_k),
        )
        if not top_shots:
            print(f"{probe_id}: no valid top shots found", flush=True)
            continue

        scenario_dir = probe_dir / _scenario_output_relative_dir(scenario)
        ensure_directory(scenario_dir)
        filename_stem = _scenario_filename_stem(scenario)
        output_path = scenario_dir / f"{filename_stem}_{args.output_name_suffix}.png"
        fig = plt.figure(figsize=(8.8, 7.4), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        mappable = _plot_top_shots(ax, scenario, top_shots, config, compact=False)
        fig.colorbar(
            mappable,
            ax=ax,
            fraction=0.035,
            pad=TRAJECTORY_COLORBAR_PAD,
            shrink=0.68,
            label="valid-renormalized marginal shot probability",
        )
        plot_title = scenario.get("plot_title")
        title = (
            f"{plot_title} {checkpoint_step(checkpoint):,}"
            if plot_title
            else f"{probe_id.replace('_', ' ')} | latest checkpoint {checkpoint_step(checkpoint):,}"
        )
        fig.suptitle(title, fontsize=13)
        fig.savefig(output_path, dpi=int(args.dpi))
        plt.close(fig)

        manifest["plots"][probe_id] = str(output_path)
        manifest["top_shots"][probe_id] = [_shot_manifest_row(row) for row in top_shots]
        overview_items.append((scenario, top_shots))
        print(f"{probe_id}: {output_path}", flush=True)

    if args.overview and overview_items:
        overview_dir = probe_dir / "top_shot_3d_views"
        ensure_directory(overview_dir)
        overview_path = overview_dir / "controlled_contact_grid_top10_shot_trajectories_3d_overview.png"
        _write_overview(overview_path, overview_items, config, dpi=int(args.dpi))
        manifest["plots"]["overview"] = str(overview_path)

    manifest_dir = probe_dir / "top_shot_3d_views"
    ensure_directory(manifest_dir)
    manifest_name = "top_shot_trajectories_3d_manifest.json"
    if args.opponent_velocity_variant is not None:
        manifest_name = f"top_shot_trajectories_3d_opponent_velocity_{_velocity_speed_label(float(args.opponent_speed))}ms_manifest.json"
    manifest_path = manifest_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")


def _expand_scenarios_over_opponent_recovery_grid(
    scenarios: list[dict[str, Any]],
    opponent_grid_side: Side,
    config: Any,
) -> list[dict[str, Any]]:
    x_cells, y_cells = _opponent_recovery_grid_cells(opponent_grid_side, config)
    expanded: list[dict[str, Any]] = []
    for scenario in scenarios:
        base_probe_id = str(scenario["probe_id"])
        for y_label, y in y_cells:
            for x_label, x in x_cells:
                variant = copy.deepcopy(scenario)
                cell_id = f"opponent_{y_label}_{x_label}"
                probe_id = f"{base_probe_id}__{cell_id}"
                state = variant["response_state"]
                if opponent_grid_side == "left":
                    state["x_left"] = float(x)
                    state["y_left"] = float(y)
                else:
                    state["x_right"] = float(x)
                    state["y_right"] = float(y)
                variant["probe_id"] = probe_id
                variant["contact_probe_id"] = base_probe_id
                variant["opponent_grid_side"] = opponent_grid_side
                variant["opponent_cell_id"] = cell_id
                variant["opponent_cell"] = {"x_region": x_label, "y_region": y_label}
                variant["opponent_position"] = {"x": float(x), "y": float(y)}
                variant["output_relative_dir"] = f"{base_probe_id}/{cell_id}"
                variant["filename_stem"] = probe_id
                expanded.append(variant)
    return expanded


def _opponent_recovery_grid_cells(
    side: Side,
    config: Any,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    (x_low, x_high), (y_low, y_high) = recovery_bounds(side, config)
    x_grid = _recovery_axis_grid(float(x_low), float(x_high), 5, lateral_motion_enabled=bool(config.court.lateral_motion_enabled))
    y_grid = _recovery_axis_grid(float(y_low), float(y_high), 5, lateral_motion_enabled=True)
    selected = (0, 2, 4)
    x_labels = ("left", "middle", "right")
    y_labels = ("backcourt", "midcourt", "frontcourt") if side == "left" else ("frontcourt", "midcourt", "backcourt")
    x_cells = tuple((label, float(x_grid[index])) for label, index in zip(x_labels, selected))
    y_cells = tuple((label, float(y_grid[index])) for label, index in zip(y_labels, selected))
    return x_cells, y_cells


def _expand_scenarios_over_opponent_velocity(
    scenarios: list[dict[str, Any]],
    *,
    directions: tuple[str, ...],
    speed: float,
) -> list[dict[str, Any]]:
    direction_vectors = {
        "neg_y": (0.0, -1.0, "-y"),
        "pos_y": (0.0, 1.0, "+y"),
        "pos_x": (1.0, 0.0, "+x"),
        "neg_x": (-1.0, 0.0, "-x"),
    }
    expanded: list[dict[str, Any]] = []
    speed_label = _velocity_speed_label(speed)
    for scenario in scenarios:
        base_probe_id = str(scenario["probe_id"])
        base_output_dir = str(scenario.get("output_relative_dir") or base_probe_id)
        base_filename_stem = str(scenario.get("filename_stem") or base_probe_id)
        for direction in directions:
            x_unit, y_unit, display_direction = direction_vectors[direction]
            variant = copy.deepcopy(scenario)
            state = variant["response_state"]
            opponent = str(variant.get("opponent_grid_side") or ("right" if state["current_hitter"] == "left" else "left"))
            vx = float(speed) * float(x_unit)
            vy = float(speed) * float(y_unit)
            state[f"v_x_{opponent}"] = vx
            state[f"v_y_{opponent}"] = vy
            velocity_id = f"opponent_v_{direction}_{speed_label}ms"
            variant["probe_id"] = f"{base_probe_id}__{velocity_id}"
            variant["base_probe_id"] = base_probe_id
            variant["opponent_velocity"] = {
                "side": opponent,
                "direction": display_direction,
                "speed": float(speed),
                "v_x": vx,
                "v_y": vy,
            }
            variant["plot_title"] = f"opponent v=({vx:g}, {vy:g}) m/s | latest checkpoint"
            variant["output_relative_dir"] = base_output_dir
            variant["filename_stem"] = f"{base_filename_stem}__{velocity_id}"
            expanded.append(variant)
    return expanded


def _velocity_speed_label(speed: float) -> str:
    if float(speed).is_integer():
        return str(int(speed))
    return f"{speed:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _recovery_axis_grid(lower: float, upper: float, count: int, *, lateral_motion_enabled: bool) -> np.ndarray:
    if count == 1:
        return np.asarray([0.5 * (lower + upper)], dtype=float)
    if lateral_motion_enabled and count in {3, 5}:
        return np.linspace(lower, upper, count + 2)[1:-1]
    return np.linspace(lower, upper, count)


def _scenario_output_relative_dir(scenario: dict[str, Any]) -> Path:
    return Path(str(scenario.get("output_relative_dir") or scenario["probe_id"]))


def _scenario_filename_stem(scenario: dict[str, Any]) -> str:
    return str(scenario.get("filename_stem") or scenario["probe_id"]).replace("/", "__")


def _probe_state_path(probe_dir: Path) -> Path:
    default_path = probe_dir / "controlled_contact_grid_probe_state.json"
    if default_path.exists():
        return default_path
    matches = sorted(probe_dir.glob("*_probe_state.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No *_probe_state.json found in {probe_dir}")
    names = ", ".join(path.name for path in matches)
    raise ValueError(f"Multiple probe state files found in {probe_dir}: {names}")


def _load_summary(probe_dir: Path) -> dict[str, Any]:
    default_path = probe_dir / "controlled_contact_grid_probe_summary.json"
    if default_path.exists():
        return json.loads(default_path.read_text(encoding="utf-8"))
    matches = sorted(probe_dir.glob("*_probe_summary.json"))
    if matches:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    return {}


def _infer_run_dir(probe_dir: Path, summary: dict[str, Any]) -> Path:
    run_dir = summary.get("run_dir")
    if run_dir:
        return Path(str(run_dir))
    if probe_dir.parent.name == "anchor_metric_eval":
        return probe_dir.parent.parent
    if probe_dir.parent.parent.name == "anchor_metric_eval":
        return probe_dir.parent.parent.parent
    raise ValueError("Could not infer run_dir; pass --run-dir.")


def _latest_probe_checkpoint(probe_dir: Path, run_dir: Path) -> Path:
    summary_csv = probe_dir / "controlled_contact_grid_probe_summary.csv"
    if not summary_csv.exists():
        matches = sorted(probe_dir.glob("*_probe_summary.csv"))
        summary_csv = matches[0] if matches else summary_csv
    if summary_csv.exists():
        rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
        valid_rows = [row for row in rows if row.get("checkpoint_path")]
        if valid_rows:
            latest = max(valid_rows, key=lambda row: int(row.get("step") or checkpoint_step(Path(row["checkpoint_path"]))))
            return Path(str(latest["checkpoint_path"]))
    return discover_anchor_checkpoints(run_dir)[-1]


def _top_shots_for_state(
    *,
    model: Any,
    mapper: DiscreteActionMapper,
    encoder: ObservationEncoder,
    state: StageState,
    agent_side: Side,
    server_side: Side,
    config: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    observation = encoder.encode(state=state, agent_side=agent_side, role="hitter", server_side=server_side)
    observation = adapt_observation_to_model(model, observation)
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    if getattr(model.policy, "output_mode", None) == "conditional_prob":
        candidates = _conditional_shot_candidates(model, obs_tensor, mapper)
    else:
        candidates = _flat_shot_candidates(model, obs_tensor, mapper, state, agent_side)

    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
    valid_probability_mass = 0.0
    for candidate in candidates:
        decoded = mapper.decode_hitter_for_agent(int(candidate["action"]), state, agent_side)
        try:
            projected = mapper.project_hitter_action(state, decoded.shot_action)
            validated = validate_and_clip_shot_action(state, projected.shot_action, config)
        except (RuntimeError, ValueError):
            continue
        action = validated.applied
        key = (
            round(float(action.v_x) * 1000),
            round(float(action.v_y) * 1000),
            round(float(action.v_z) * 1000),
        )
        probability = float(candidate["probability"])
        valid_probability_mass += probability
        previous = grouped.get(key)
        if previous is not None:
            previous["raw_probability"] += probability
            if probability <= float(previous["source_probability"]):
                continue
        trajectory = simulate_trajectory(state.x0, state.y0, state.z0, action.v_x, action.v_y, action.v_z, config)
        landing_x, landing_y = landing_position(state, action, config)
        horizontal_speed = float(np.hypot(action.v_x, action.v_y))
        theta_degrees = math.degrees(math.atan2(float(action.v_z), horizontal_speed))
        shot_type = name_velocity_shot(
            hitter=agent_side,
            contact_x=float(state.x0),
            contact_y=float(state.y0),
            landing_x=float(landing_x),
            landing_y=float(landing_y),
            theta_degrees=theta_degrees,
            config=config,
        )
        grouped[key] = {
            **candidate,
            "raw_probability": probability if previous is None else float(previous["raw_probability"]),
            "source_probability": probability,
            "projected": bool(projected.projected or validated.projected),
            "shot_type": shot_type,
            "v_x": float(action.v_x),
            "v_y": float(action.v_y),
            "v_z": float(action.v_z),
            "landing_x": float(landing_x),
            "landing_y": float(landing_y),
            "trajectory": trajectory,
        }
    if valid_probability_mass <= 0.0:
        return []

    top: list[dict[str, Any]] = []
    for row in sorted(grouped.values(), key=lambda item: float(item["raw_probability"]), reverse=True)[:top_k]:
        row = dict(row)
        row["rank"] = len(top) + 1
        row["probability"] = float(row["raw_probability"]) / valid_probability_mass
        row["valid_probability_mass"] = valid_probability_mass
        top.append(row)
    return top


def _conditional_shot_candidates(model: Any, obs_tensor: torch.Tensor, mapper: DiscreteActionMapper) -> list[dict[str, Any]]:
    policy = model.policy
    with torch.no_grad():
        features = policy.extract_features(obs_tensor)
        if policy.share_features_extractor:
            latent_pi, _ = policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        phi_logits, _, _, _, _ = policy._conditional_component_logits(obs_tensor, latent_pi)
        log_phi = torch.log_softmax(phi_logits[0], dim=0)

        candidates: list[dict[str, Any]] = []
        recovery_count = int(policy._conditional_recovery_count)
        for phi_index in range(int(policy._conditional_phi_count)):
            phi = torch.as_tensor([phi_index], dtype=torch.long, device=obs_tensor.device)
            _, theta_logits, _, _, _ = policy._conditional_component_logits(obs_tensor, latent_pi, phi=phi)
            assert theta_logits is not None
            log_theta = torch.log_softmax(theta_logits[0], dim=0)
            for theta_index in range(int(policy._conditional_theta_count)):
                theta = torch.as_tensor([theta_index], dtype=torch.long, device=obs_tensor.device)
                _, _, speed_logits, _, _ = policy._conditional_component_logits(
                    obs_tensor,
                    latent_pi,
                    phi=phi,
                    theta=theta,
                )
                assert speed_logits is not None
                log_speed = torch.log_softmax(speed_logits[0], dim=0)
                for speed_index in range(int(policy._conditional_speed_count)):
                    action = (
                        ((phi_index * int(policy._conditional_theta_count) + theta_index) * int(policy._conditional_speed_count) + speed_index)
                        * recovery_count
                    )
                    log_probability = float((log_phi[phi_index] + log_theta[theta_index] + log_speed[speed_index]).item())
                    candidates.append(
                        {
                            "action": int(action),
                            "probability": float(math.exp(log_probability)),
                            "phi_index": int(phi_index),
                            "theta_index": int(theta_index),
                            "speed_index": int(speed_index),
                        }
                    )
    return candidates


def _flat_shot_candidates(
    model: Any,
    obs_tensor: torch.Tensor,
    mapper: DiscreteActionMapper,
    state: StageState,
    agent_side: Side,
) -> list[dict[str, Any]]:
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor).distribution
        probabilities = torch.softmax(distribution.logits.squeeze(0), dim=0).detach().cpu().numpy()
    grouped: dict[int, dict[str, Any]] = {}
    recovery_count = max(int(mapper._impl._effective_x_rec_bins * mapper.discrete_config.y_rec_bins), 1)
    for action_index in range(min(int(mapper.hitter_action_count), len(probabilities))):
        decoded = mapper.decode_hitter_for_agent(action_index, state, agent_side)
        shot_action = int(decoded.flat_index // recovery_count * recovery_count)
        row = grouped.setdefault(
            shot_action,
            {
                "action": shot_action,
                "probability": 0.0,
                "phi_index": getattr(decoded, "phi_index", None),
                "theta_index": getattr(decoded, "theta_index", None),
                "speed_index": getattr(decoded, "speed_index", None),
            },
        )
        row["probability"] += float(probabilities[action_index])
    return list(grouped.values())


def _plot_top_shots(
    ax: Any,
    scenario: dict[str, Any],
    top_shots: list[dict[str, Any]],
    config: Any,
    *,
    compact: bool,
) -> Any:
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    state = _stage_state_from_dict(scenario["response_state"])
    _draw_court_3d(ax, config)
    max_probability = max(float(row["probability"]) for row in top_shots)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(max_probability, 1e-12))
    cmap = LinearSegmentedColormap.from_list(
        "probability_white_to_dark",
        [(1.0, 1.0, 1.0, 0.16), (0.58, 0.64, 0.66, 0.52), (0.02, 0.02, 0.02, 0.98)],
    )

    ax.scatter([state.x0], [state.y0], [state.z0], color="crimson", s=52 if not compact else 18, depthshade=False)
    ax.plot([state.x0, state.x0], [state.y0, state.y0], [0.0, state.z0], color="crimson", linestyle=":", linewidth=1.0)
    opponent = str(scenario.get("opponent_grid_side") or ("right" if state.current_hitter == "left" else "left"))
    opponent_x = float(state.x_right if opponent == "right" else state.x_left)
    opponent_y = float(state.y_right if opponent == "right" else state.y_left)
    ax.scatter(
        [opponent_x],
        [opponent_y],
        [0.06],
        marker="s",
        color="royalblue",
        s=42 if not compact else 15,
        depthshade=False,
        zorder=7,
    )
    if not compact:
        ax.text(
            state.x0,
            state.y0,
            state.z0 + 0.2,
            "contact",
            color="crimson",
            fontsize=8,
            ha="center",
        )
        ax.text(
            opponent_x,
            opponent_y,
            0.28,
            "opponent",
            color="royalblue",
            fontsize=8,
            ha="center",
        )

    for row in top_shots:
        trajectory = row["trajectory"]
        xs = np.asarray([point.x for point in trajectory.samples], dtype=float)
        ys = np.asarray([point.y for point in trajectory.samples], dtype=float)
        zs = np.asarray([point.z for point in trajectory.samples], dtype=float)
        probability = float(row["probability"])
        color = cmap(norm(probability))
        linewidth = (3.0 if row["rank"] == 1 else 2.15) if not compact else (1.6 if row["rank"] == 1 else 0.95)
        label = None if compact else f"#{row['rank']} {row['shot_type']} p={_format_probability(probability)}"
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=color[-1], label=label, zorder=6)
        ax.scatter(
            [row["landing_x"]],
            [row["landing_y"]],
            [GROUND_MARKER_Z],
            marker="x",
            color=color,
            s=46 if not compact else 16,
            depthshade=False,
        )
        if not compact and row["rank"] <= 3:
            ax.text(
                float(row["landing_x"]),
                float(row["landing_y"]),
                0.24 + 0.08 * (row["rank"] - 1),
                f"#{row['rank']}",
                color="0.05",
                fontsize=8,
                ha="center",
            )

    all_z = [
        point.z
        for row in top_shots
        for point in row["trajectory"].samples
    ]
    display_half_width = max(float(config.court.half_width), OFFICIAL_DOUBLES_WIDTH / 2.0)
    ax.set_xlim(-display_half_width - 0.35, display_half_width + 0.35)
    ax.set_ylim(-config.court.half_length - 0.35, config.court.half_length + 0.35)
    ax.set_zlim(0.0, max(config.render.z_max, float(np.nanmax(all_z)) + 0.5))
    box_aspect = (2.0 * display_half_width, config.court.length, 4.2)
    try:
        ax.set_box_aspect(box_aspect, zoom=TRAJECTORY_VIEW_ZOOM)
    except TypeError:
        ax.set_box_aspect(box_aspect)
    ax.view_init(elev=TRAJECTORY_VIEW_ELEV, azim=TRAJECTORY_VIEW_AZIM)
    if hasattr(ax, "dist"):
        ax.dist = TRAJECTORY_VIEW_DIST
    _hide_3d_axes(ax)
    if not compact:
        top_mass = sum(float(row["probability"]) for row in top_shots)
        ax.text2D(
            0.02,
            0.97,
            f"top-{len(top_shots)} valid-renormalized shot mass {top_mass:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )
        ax.legend(loc="upper right", fontsize=7)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def _draw_court_3d(ax: Any, config: Any) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    colors = stage_colors(False)
    half_w = max(float(config.court.half_width), OFFICIAL_DOUBLES_WIDTH / 2.0)
    half_l = float(config.court.half_length)
    net_y = float(config.court.net_y)
    net_h = float(config.court.net_height)

    court_surface = Poly3DCollection(
        [[
            (-half_w, -half_l, COURT_SURFACE_Z),
            (half_w, -half_l, COURT_SURFACE_Z),
            (half_w, half_l, COURT_SURFACE_Z),
            (-half_w, half_l, COURT_SURFACE_Z),
        ]],
        facecolors=colors["court_fill"],
        edgecolors="none",
        alpha=0.92,
        zorder=0,
    )
    court_surface.set_zsort("min")
    court_surface.set_sort_zpos(COURT_SURFACE_Z - 1.0)
    ax.add_collection3d(court_surface)

    line_z = max(COURT_LINE_Z, GROUND_MARKER_Z + 0.03)
    corners = [
        (-half_w, -half_l),
        (half_w, -half_l),
        (half_w, half_l),
        (-half_w, half_l),
        (-half_w, -half_l),
    ]
    ax.plot(
        [point[0] for point in corners],
        [point[1] for point in corners],
        [line_z for _ in corners],
        color=colors["court_line"],
        linewidth=2.4,
        zorder=1,
    )
    singles_half_width = OFFICIAL_SINGLES_WIDTH / 2.0
    short_service_y = OFFICIAL_SHORT_SERVICE_FROM_NET
    long_service_y = half_l - OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK
    line_kwargs = {"color": colors["court_line"], "linewidth": 2.2, "zorder": 1}
    for x_pos in (-singles_half_width, singles_half_width):
        ax.plot([x_pos, x_pos], [-half_l, half_l], [line_z, line_z], **line_kwargs)
    for y_pos in (-short_service_y, short_service_y, -long_service_y, long_service_y):
        ax.plot([-half_w, half_w], [y_pos, y_pos], [line_z, line_z], **line_kwargs)
    ax.plot([0.0, 0.0], [-half_l, -short_service_y], [line_z, line_z], color=colors["service_line"], linewidth=2.0, zorder=1)
    ax.plot([0.0, 0.0], [short_service_y, half_l], [line_z, line_z], color=colors["service_line"], linewidth=2.0, zorder=1)

    ax.plot([-half_w, half_w], [net_y, net_y], [net_h, net_h], color=colors["net"], linewidth=2.2)
    for x_value in (-half_w, half_w):
        ax.plot([x_value, x_value], [net_y, net_y], [0.0, net_h], color=colors["net"], linewidth=1.0)

    net = Poly3DCollection(
        [[(-half_w, net_y, 0.0), (half_w, net_y, 0.0), (half_w, net_y, net_h), (-half_w, net_y, net_h)]],
        alpha=0.16,
        facecolor=colors["net"],
        edgecolor="none",
    )
    ax.add_collection3d(net)


def _hide_3d_axes(ax: Any) -> None:
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
        if axis is None:
            continue
        pane = getattr(axis, "pane", None)
        if pane is not None:
            pane.fill = False
            pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.line.set_linewidth(0.0)
        grid_info = getattr(axis, "_axinfo", None)
        if isinstance(grid_info, dict) and "grid" in grid_info:
            grid_info["grid"]["linewidth"] = 0.0
            grid_info["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)


def _write_overview(path: Path, items: list[tuple[dict[str, Any], list[dict[str, Any]]]], config: Any, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17.0, 13.0), constrained_layout=True)
    for index, (scenario, top_shots) in enumerate(items, start=1):
        ax = fig.add_subplot(9, 3, index, projection="3d")
        _plot_top_shots(ax, scenario, top_shots, config, compact=True)
        ax.set_title(str(scenario["probe_id"]).replace("_", " "), fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.tick_params(labelsize=6)
    fig.suptitle("Controlled contact grid: top-10 latest shot trajectories", fontsize=16)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _stage_state_from_dict(values: dict[str, Any]) -> StageState:
    names = {field.name for field in fields(StageState)}
    return StageState(**{key: value for key, value in values.items() if key in names})


def _format_probability(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _shot_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(row["rank"]),
        "probability": float(row["probability"]),
        "raw_probability": float(row["raw_probability"]),
        "valid_probability_mass": float(row["valid_probability_mass"]),
        "action": int(row["action"]),
        "phi_index": None if row.get("phi_index") is None else int(row["phi_index"]),
        "theta_index": None if row.get("theta_index") is None else int(row["theta_index"]),
        "speed_index": None if row.get("speed_index") is None else int(row["speed_index"]),
        "projected": bool(row["projected"]),
        "shot_type": str(row["shot_type"]),
        "v_x": float(row["v_x"]),
        "v_y": float(row["v_y"]),
        "v_z": float(row["v_z"]),
        "landing_x": float(row["landing_x"]),
        "landing_y": float(row["landing_y"]),
    }


if __name__ == "__main__":
    main()
