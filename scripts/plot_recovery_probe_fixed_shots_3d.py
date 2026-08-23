from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.eval_evolution import build_sim_config, load_run_config
from badminton.mpl_config import ensure_writable_matplotlib_config
from badminton.state import ShotAction, StageState
from badminton.trajectory import simulate_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3D fixed-shot trajectories used by a recovery_contact_grid_probe run."
    )
    parser.add_argument(
        "probe_dir",
        type=Path,
        help="Directory containing recovery_contact_grid_probe_state.json.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing selfplay_config.json. Defaults to the run_dir saved in the probe state.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="fixed_shot_3d_views",
        help="Subdirectory under probe_dir for copies of all 3D shot plots.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Do not overlay latest recovery choice probabilities from recovery_contact_grid_probe_bins.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    state_path = _probe_state_path(args.probe_dir)
    probe_state = json.loads(state_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir or Path(str(probe_state["run_dir"]))
    config = build_sim_config(load_run_config(run_dir))

    output_dir = args.probe_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    scenarios = list(probe_state["scenarios"])
    recovery_bins = (
        {}
        if args.no_recovery
        else _load_latest_recovery_bins(_probe_bins_path(args.probe_dir))
    )
    for scenario in scenarios:
        probe_id = str(scenario["probe_id"])
        fig = plt.figure(figsize=(8.5, 7.2), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        recovery_mappable = _plot_scenario(
            ax,
            scenario,
            config,
            recovery_bins=recovery_bins.get(probe_id, []),
            compact=False,
        )
        if recovery_mappable is not None:
            fig.colorbar(
                recovery_mappable,
                ax=ax,
                fraction=0.035,
                pad=0.04,
                shrink=0.65,
                label="latest one-step response score",
            )
        fig.suptitle(_scenario_title(scenario), fontsize=13)

        scenario_dir = args.probe_dir / probe_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = scenario_dir / f"{probe_id}_fixed_shot_3d.png"
        copy_path = output_dir / f"{probe_id}_fixed_shot_3d.png"
        fig.savefig(scenario_path, dpi=int(args.dpi))
        fig.savefig(copy_path, dpi=int(args.dpi))
        plt.close(fig)
        paths[probe_id] = str(scenario_path)

    overview_path = output_dir / "recovery_contact_grid_fixed_shots_3d_overview.png"
    _write_overview(overview_path, scenarios, config, recovery_bins=recovery_bins, dpi=int(args.dpi))
    paths["overview"] = str(overview_path)

    manifest_path = output_dir / "fixed_shot_3d_manifest.json"
    manifest_path.write_text(json.dumps({"plots": paths}, indent=2), encoding="utf-8")
    print(f"wrote {len(scenarios)} fixed-shot 3D plots")
    print(f"overview: {overview_path}")
    print(f"manifest: {manifest_path}")


def _probe_state_path(probe_dir: Path) -> Path:
    default_path = probe_dir / "recovery_contact_grid_probe_state.json"
    if default_path.exists():
        return default_path
    matches = sorted(probe_dir.glob("*_probe_state.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No *_probe_state.json found in {probe_dir}")
    names = ", ".join(path.name for path in matches)
    raise ValueError(f"Multiple probe state files found in {probe_dir}: {names}")


def _probe_bins_path(probe_dir: Path) -> Path:
    default_path = probe_dir / "recovery_contact_grid_probe_bins.csv"
    if default_path.exists():
        return default_path
    matches = sorted(probe_dir.glob("*_probe_bins.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No *_probe_bins.csv found in {probe_dir}")
    names = ", ".join(path.name for path in matches)
    raise ValueError(f"Multiple probe bin files found in {probe_dir}: {names}")


def _load_latest_recovery_bins(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    latest_step_by_probe: dict[str, int] = {}
    for row in rows:
        probe_id = str(row.get("probe_id", ""))
        step = int(row.get("step", -1))
        if probe_id and step > latest_step_by_probe.get(probe_id, -1):
            latest_step_by_probe[probe_id] = step

    latest: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        probe_id = str(row.get("probe_id", ""))
        if probe_id and int(row.get("step", -1)) == latest_step_by_probe.get(probe_id):
            latest.setdefault(probe_id, []).append(row)
    return latest


def _plot_scenario(
    ax: Any,
    scenario: dict[str, Any],
    config: Any,
    *,
    recovery_bins: list[dict[str, Any]],
    compact: bool,
) -> Any | None:
    state = _stage_state_from_dict(scenario["state_before"])
    action = _shot_action_from_dict(scenario["fixed_action"])
    trajectory = simulate_trajectory(
        state.x0,
        state.y0,
        state.z0,
        action.v_x,
        action.v_y,
        action.v_z,
        config,
    )
    xs = np.asarray([point.x for point in trajectory.samples], dtype=float)
    ys = np.asarray([point.y for point in trajectory.samples], dtype=float)
    zs = np.asarray([point.z for point in trajectory.samples], dtype=float)
    target = scenario["target_point"]
    intercept = np.asarray(scenario["actual_intercept_point"], dtype=float)

    _draw_court_3d(ax, config)
    ax.plot(xs, ys, zs, color="tab:blue", linewidth=2.4, label="fixed shot trajectory", zorder=5)
    ax.scatter([state.x0], [state.y0], [state.z0], color="tab:blue", s=42, depthshade=False, label="hitter contact")
    ax.plot([state.x0, state.x0], [state.y0, state.y0], [0.0, state.z0], color="tab:blue", linestyle=":", linewidth=1.1)
    if not compact:
        ax.text(
            state.x0,
            state.y0,
            state.z0 + 0.22,
            f"hitter contact\n({state.x0:.2f}, {state.y0:.2f}, {state.z0:.2f})",
            color="tab:blue",
            fontsize=8,
            ha="center",
        )
    ax.scatter(
        [float(target["x"])],
        [float(target["y"])],
        [float(target["z"])],
        marker="o",
        facecolors="none",
        edgecolors="tab:orange",
        s=95,
        linewidths=1.7,
        depthshade=False,
        label="requested target",
    )
    ax.scatter(
        [float(target["x"])],
        [float(target["y"])],
        [0.0],
        marker="^",
        color="0.35",
        s=45,
        depthshade=False,
        label="opponent start x/y",
    )
    ax.scatter(
        [intercept[0]],
        [intercept[1]],
        [intercept[2]],
        marker="*",
        color="crimson",
        s=170,
        depthshade=False,
        label="opponent intercept",
    )
    ax.plot([intercept[0], intercept[0]], [intercept[1], intercept[1]], [0.0, intercept[2]], color="crimson", linestyle=":", linewidth=1.2)
    if not compact:
        ax.text(
            float(intercept[0]),
            float(intercept[1]),
            float(intercept[2]) + 0.22,
            f"opponent intercept\n({intercept[0]:.2f}, {intercept[1]:.2f}, {intercept[2]:.2f})",
            color="crimson",
            fontsize=8,
            ha="center",
        )
    ax.scatter(
        [trajectory.landing_x],
        [trajectory.landing_y],
        [0.0],
        marker="x",
        color="black",
        s=45,
        depthshade=False,
        label="landing",
    )
    recovery_mappable = _plot_recovery_bins(ax, recovery_bins, annotate=not compact)
    _plot_opponent_response_trajectory(ax, scenario, recovery_bins, config, compact=compact)

    intercept_index = int(scenario["intercept_index"])
    distance = float(scenario["target_distance"])
    ax.text2D(
        0.02,
        0.97,
        f"intercept index {intercept_index}  |  target error {distance:.3f} m",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    ax.set_xlabel("x across court (m)")
    ax.set_ylabel("y along court (m)")
    ax.set_zlabel("z height (m)")
    ax.set_xlim(-config.court.half_width - 0.35, config.court.half_width + 0.35)
    ax.set_ylim(-config.court.half_length - 0.35, config.court.half_length + 0.35)
    ax.set_zlim(0.0, max(config.render.z_max, float(np.nanmax(zs)) + 0.5, float(intercept[2]) + 0.5))
    ax.set_box_aspect((config.court.width, config.court.length, 4.8))
    ax.view_init(elev=24, azim=-63)
    if not compact:
        ax.legend(loc="upper right", fontsize=8)
    return recovery_mappable


def _plot_recovery_bins(ax: Any, recovery_bins: list[dict[str, Any]], *, annotate: bool) -> Any | None:
    if not recovery_bins:
        return None

    scores = np.asarray([_row_float(row, "score", default=float("nan")) for row in recovery_bins], dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    score_max = float(np.max(finite_scores)) if finite_scores.size else 1.0
    color_max = score_max if score_max > 1e-9 else 1.0
    xs = np.asarray([float(row["recovery_x"]) for row in recovery_bins], dtype=float)
    ys = np.asarray([float(row["recovery_y"]) for row in recovery_bins], dtype=float)
    z_floor = np.full_like(xs, 0.06, dtype=float)
    scatter = ax.scatter(
        xs,
        ys,
        z_floor,
        c=scores,
        cmap="magma",
        vmin=0.0,
        vmax=color_max,
        s=72.0,
        alpha=0.82,
        edgecolors="black",
        linewidths=0.35,
        depthshade=False,
        label="recovery bins",
    )
    probabilities = np.asarray([float(row["policy_probability"]) for row in recovery_bins], dtype=float)
    top_index = int(np.argmax(probabilities))
    ax.scatter(
        [xs[top_index]],
        [ys[top_index]],
        [0.18],
        marker="D",
        color="gold",
        edgecolors="black",
        s=70,
        depthshade=False,
        label="top recovery choice",
    )
    if annotate:
        ax.text(
            float(xs[top_index]),
            float(ys[top_index]),
            0.45,
            f"top recovery\np={probabilities[top_index]:.2f}",
            color="black",
            fontsize=8,
            ha="center",
        )
    best_score = float(np.max(finite_scores)) if finite_scores.size else 0.0
    best_score_rows = [
        row
        for row in recovery_bins
        if np.isclose(_row_float(row, "score", default=float("nan")), best_score, rtol=0.0, atol=1e-9)
    ]
    best_xs = [float(row["recovery_x"]) for row in best_score_rows]
    best_ys = [float(row["recovery_y"]) for row in best_score_rows]
    best_zs = [0.3] * len(best_score_rows)
    ax.scatter(
        best_xs,
        best_ys,
        best_zs,
        marker="X",
        color="tab:cyan",
        edgecolors="black",
        s=85,
        depthshade=False,
        label="best score bins",
    )
    if annotate and best_score_rows:
        label_row = max(best_score_rows, key=lambda row: _row_float(row, "policy_probability", default=0.0) or 0.0)
        best_x = float(label_row["recovery_x"])
        best_y = float(label_row["recovery_y"])
        ax.text(
            best_x,
            best_y,
            0.62,
            f"best response\n{best_score:.2f} ({len(best_score_rows)} bins)",
            color="black",
            fontsize=8,
            ha="center",
        )
    return scatter


def _plot_opponent_response_trajectory(
    ax: Any,
    scenario: dict[str, Any],
    recovery_bins: list[dict[str, Any]],
    config: Any,
    *,
    compact: bool,
) -> None:
    if not recovery_bins:
        return
    top_row = max(recovery_bins, key=lambda row: _row_float(row, "policy_probability", default=0.0))
    response_payloads = _row_response_payloads(top_row)
    valid_payloads = [
        payload
        for payload in response_payloads
        if all(_payload_float(payload, key) is not None for key in ("opponent_v_x", "opponent_v_y", "opponent_v_z"))
    ]
    if not valid_payloads:
        return
    intercept = np.asarray(scenario["actual_intercept_point"], dtype=float)
    colors = ("tab:red", "tab:pink", "tab:orange", "tab:purple", "tab:green", "tab:brown")
    for index, payload in enumerate(valid_payloads):
        values = [_payload_float(payload, key) for key in ("opponent_v_x", "opponent_v_y", "opponent_v_z")]
        assert all(value is not None for value in values)
        trajectory = simulate_trajectory(
            float(intercept[0]),
            float(intercept[1]),
            float(intercept[2]),
            float(values[0]),
            float(values[1]),
            float(values[2]),
            config,
        )
        xs = np.asarray([point.x for point in trajectory.samples], dtype=float)
        ys = np.asarray([point.y for point in trajectory.samples], dtype=float)
        zs = np.asarray([point.z for point in trajectory.samples], dtype=float)
        color = colors[index % len(colors)]
        probability = _payload_float(payload, "probability", default=0.0) or 0.0
        label = "likely opponent response" if index == 0 else f"opponent response {index + 1}"
        if not compact:
            label = f"{label} (p={probability:.2f})"
        ax.plot(
            xs,
            ys,
            zs,
            color=color,
            linestyle="--" if index == 0 else ":",
            linewidth=(2.0 if index == 0 else 1.45) if not compact else (1.25 if index == 0 else 0.9),
            alpha=0.92 if index == 0 else 0.68,
            label=label,
            zorder=6 - min(index, 3) * 0.1,
        )
        ax.scatter(
            [trajectory.landing_x],
            [trajectory.landing_y],
            [0.0],
            marker="x",
            color=color,
            s=45 if not compact else 20,
            depthshade=False,
            label="opponent response landing" if index == 0 else None,
        )
        response_intercept = tuple(
            _payload_float(payload, key)
            for key in ("response_intercept_x", "response_intercept_y", "response_intercept_z")
        )
        if all(value is not None for value in response_intercept):
            ax.scatter(
                [float(response_intercept[0])],
                [float(response_intercept[1])],
                [float(response_intercept[2])],
                marker="P",
                color=color,
                edgecolors="black",
                s=75 if (not compact and index == 0) else 35,
                depthshade=False,
                label="train response contact" if index == 0 else None,
            )
        if not compact and index == 0 and all(value is not None for value in response_intercept):
            ax.text(
                float(response_intercept[0]),
                float(response_intercept[1]),
                float(response_intercept[2]) + 0.18,
                "train response",
                color="tab:purple",
                fontsize=8,
                ha="center",
            )


def _row_response_payloads(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("opponent_responses_json")
    if raw not in (None, ""):
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            return [payload for payload in parsed if isinstance(payload, dict)]
    return [
        {
            "rank": 1,
            "probability": 1.0,
            "opponent_v_x": row.get("opponent_v_x"),
            "opponent_v_y": row.get("opponent_v_y"),
            "opponent_v_z": row.get("opponent_v_z"),
            "opponent_landing_x": row.get("opponent_landing_x"),
            "opponent_landing_y": row.get("opponent_landing_y"),
            "response_intercept_x": row.get("response_intercept_x"),
            "response_intercept_y": row.get("response_intercept_y"),
            "response_intercept_z": row.get("response_intercept_z"),
        }
    ]


def _row_float(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _payload_float(payload: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = payload.get(key)
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _axis_edges(values: list[float]) -> np.ndarray:
    if len(values) == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=float)
    centers = np.asarray(values, dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def _draw_court_3d(ax: Any, config: Any) -> None:
    half_w = float(config.court.half_width)
    half_l = float(config.court.half_length)
    net_y = float(config.court.net_y)
    net_h = float(config.court.net_height)
    service = float(config.court.service_line_distance_from_net)
    z = 0.0

    court_lines = [
        ([-half_w, half_w], [-half_l, -half_l]),
        ([-half_w, half_w], [half_l, half_l]),
        ([-half_w, -half_w], [-half_l, half_l]),
        ([half_w, half_w], [-half_l, half_l]),
        ([-half_w, half_w], [net_y, net_y]),
        ([-half_w, half_w], [-service, -service]),
        ([-half_w, half_w], [service, service]),
        ([0.0, 0.0], [-half_l, half_l]),
    ]
    for x_values, y_values in court_lines:
        ax.plot(x_values, y_values, [z, z], color="0.15", linewidth=1.1, alpha=0.75)

    ax.plot([-half_w, half_w], [net_y, net_y], [net_h, net_h], color="0.1", linewidth=2.0)
    for x_value in (-half_w, half_w):
        ax.plot([x_value, x_value], [net_y, net_y], [0.0, net_h], color="0.1", linewidth=1.0)
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        net = Poly3DCollection(
            [[(-half_w, net_y, 0.0), (half_w, net_y, 0.0), (half_w, net_y, net_h), (-half_w, net_y, net_h)]],
            alpha=0.12,
            facecolor="black",
            edgecolor="none",
        )
        ax.add_collection3d(net)
    except Exception:
        pass


def _write_overview(
    path: Path,
    scenarios: list[dict[str, Any]],
    config: Any,
    *,
    recovery_bins: dict[str, list[dict[str, Any]]],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17.0, 13.0), constrained_layout=True)
    for index, scenario in enumerate(scenarios, start=1):
        ax = fig.add_subplot(9, 3, index, projection="3d")
        _plot_scenario(
            ax,
            scenario,
            config,
            recovery_bins=recovery_bins.get(str(scenario["probe_id"]), []),
            compact=True,
        )
        ax.set_title(str(scenario["probe_id"]).replace("_", " "), fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.tick_params(labelsize=6)
    fig.suptitle("Fixed shots used by recovery_contact_grid_probe", fontsize=16)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _stage_state_from_dict(values: dict[str, Any]) -> StageState:
    names = {field.name for field in fields(StageState)}
    return StageState(**{key: value for key, value in values.items() if key in names})


def _shot_action_from_dict(values: dict[str, Any]) -> ShotAction:
    names = {field.name for field in fields(ShotAction)}
    return ShotAction(**{key: value for key, value in values.items() if key in names})


def _scenario_title(scenario: dict[str, Any]) -> str:
    return (
        "Fixed shot to "
        f"{scenario['target_y_region']} / {scenario['target_x_region']} / {scenario['target_z_level']}"
    )


if __name__ == "__main__":
    main()
