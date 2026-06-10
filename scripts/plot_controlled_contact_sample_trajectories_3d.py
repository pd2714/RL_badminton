from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.eval_evolution import build_sim_config, load_run_config
from badminton1d.mpl_config import ensure_writable_matplotlib_config
from badminton1d.render import GROUND_MARKER_Z, setup_3d_court_axes, stage_colors
from badminton1d.state import Side
from badminton1d.trajectory import simulate_trajectory
from badminton1d.utils import ensure_directory, opponent_side, recovery_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 3D trajectories reconstructed from the chosen landing samples in a "
            "controlled_contact_grid_probe samples CSV."
        )
    )
    parser.add_argument(
        "probe_dir",
        type=Path,
        help="Directory containing controlled_contact_grid_probe_samples.csv.",
    )
    parser.add_argument(
        "--probe-id",
        action="append",
        default=None,
        help="Probe id to render. Can be passed more than once. Defaults to all probes.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing selfplay_config.json. Defaults to metadata or probe_dir/../...",
    )
    parser.add_argument("--samples-csv", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--fit-z",
        action="store_true",
        help="Expand the z-axis to include every trajectory. By default the slider-court z scale is fixed.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="chosen_landing_sample_trajectories_3d",
        help="PNG suffix written under PROBE_ID/opponent_default_position.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_writable_matplotlib_config()

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    probe_dir = args.probe_dir
    samples_csv = args.samples_csv or _probe_samples_path(probe_dir)
    rows = _read_csv(samples_csv)
    selected_probe_ids = set(args.probe_id) if args.probe_id else {str(row["probe_id"]) for row in rows}
    run_dir = args.run_dir or _infer_run_dir(probe_dir)
    config = build_sim_config(load_run_config(run_dir))
    run_config = load_run_config(run_dir)
    train_side: Side = str(run_config.get("train_side", "left"))  # type: ignore[assignment]
    scenarios = _scenario_lookup(probe_dir, train_side, config)

    written: dict[str, str] = {}
    for probe_id in sorted(selected_probe_ids):
        sample_rows = [
            row
            for row in rows
            if str(row.get("probe_id")) == probe_id and _is_true(row.get("valid"))
        ]
        if not sample_rows:
            print(f"{probe_id}: no valid sample rows", flush=True)
            continue
        sample_rows.sort(key=lambda row: (int(row["step"]), int(row.get("sample_index", 0))))

        trajectories = [_trajectory_from_row(row, config) for row in sample_rows]
        steps = np.asarray([int(row["step"]) for row in sample_rows], dtype=float)
        cmap = mpl.colormaps["viridis"]
        norm = mpl.colors.Normalize(vmin=float(np.min(steps)), vmax=float(np.max(steps)))

        fig = plt.figure(figsize=(10.6, 8.6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        colors = stage_colors(monochrome=False)
        setup_3d_court_axes(ax, config, colors, show_axes=False)
        _expand_slider_court_view(ax, config, trajectories, fit_z=bool(args.fit_z))

        segments = [np.column_stack((xs, ys, zs)) for xs, ys, zs in trajectories]
        collection = Line3DCollection(
            segments,
            colors=[cmap(norm(step)) for step in steps],
            linewidths=0.75,
            alpha=0.2,
        )
        ax.add_collection3d(collection, autolim=False)

        scenario = scenarios.get(probe_id)
        contact = sample_rows[-1]
        contact_xyz = (
            float(contact["contact_x"]),
            float(contact["contact_y"]),
            float(contact["contact_z"]),
        )
        landing_xs = [float(row["landing_x"]) for row in sample_rows]
        landing_ys = [float(row["landing_y"]) for row in sample_rows]
        ax.scatter(
            landing_xs,
            landing_ys,
            [GROUND_MARKER_Z] * len(sample_rows),
            c=steps,
            cmap=cmap,
            norm=norm,
            s=12,
            alpha=0.55,
            linewidths=0.0,
            depthshade=False,
            label="chosen landings",
            zorder=7,
        )
        ax.scatter(
            [contact_xyz[0]],
            [contact_xyz[1]],
            [contact_xyz[2]],
            marker="*",
            s=150,
            color="crimson",
            edgecolors="white",
            linewidths=0.5,
            depthshade=False,
            label="fixed contact",
            zorder=8,
        )
        ax.plot(
            [contact_xyz[0], contact_xyz[0]],
            [contact_xyz[1], contact_xyz[1]],
            [GROUND_MARKER_Z, contact_xyz[2]],
            color="crimson",
            linestyle=":",
            linewidth=1.1,
            alpha=0.85,
            zorder=7,
        )
        opponent_xy = _opponent_position_from_scenario(scenario)
        if opponent_xy is not None:
            ax.scatter(
                [opponent_xy[0]],
                [opponent_xy[1]],
                [0.06],
                marker="s",
                color="royalblue",
                s=42,
                depthshade=False,
                label="opponent",
                zorder=9,
            )
            ax.text(
                opponent_xy[0],
                opponent_xy[1],
                0.28,
                "opponent",
                color="royalblue",
                fontsize=8,
                ha="center",
                zorder=9,
            )

        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(steps)
        fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.035, shrink=0.72, label="checkpoint step (M)")
        title = str(sample_rows[-1].get("probe_title") or probe_id.replace("_", " "))
        ax.set_title(f"{title}\nchosen landing sample trajectories", pad=8.0, fontsize=13)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

        output_dir = probe_dir / _output_relative_dir_for_rows(sample_rows, scenario)
        ensure_directory(output_dir)
        output_path = output_dir / f"{probe_id}_{args.output_suffix}.png"
        fig.savefig(output_path, dpi=int(args.dpi))
        plt.close(fig)
        written[probe_id] = str(output_path)
        print(f"{probe_id}: {output_path}", flush=True)

    manifest_path = probe_dir / "chosen_landing_sample_3d_manifest.json"
    manifest_plots = dict(_existing_manifest_plots(manifest_path))
    manifest_plots.update(written)
    manifest_path.write_text(
        json.dumps(
            {
                "probe_dir": str(probe_dir),
                "run_dir": str(run_dir),
                "samples_csv": str(samples_csv),
                "plots": manifest_plots,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")


def _probe_samples_path(probe_dir: Path) -> Path:
    default_path = probe_dir / "controlled_contact_grid_probe_samples.csv"
    if default_path.exists():
        return default_path
    matches = sorted(probe_dir.glob("*_probe_samples.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No *_probe_samples.csv found in {probe_dir}")
    names = ", ".join(path.name for path in matches)
    raise ValueError(f"Multiple probe sample CSVs found in {probe_dir}: {names}")


def _existing_manifest_plots(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    plots = manifest.get("plots")
    return dict(plots) if isinstance(plots, dict) else {}


def _infer_run_dir(probe_dir: Path) -> Path:
    summary_path = probe_dir / "controlled_contact_grid_probe_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary.get("run_dir")
        if run_dir:
            return Path(str(run_dir))
    if probe_dir.parent.name == "anchor_metric_eval":
        return probe_dir.parent.parent
    if probe_dir.parent.parent.name == "anchor_metric_eval":
        return probe_dir.parent.parent.parent
    raise ValueError("Could not infer run_dir; pass --run-dir.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _trajectory_from_row(row: dict[str, str], config: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = simulate_trajectory(
        float(row["contact_x"]),
        float(row["contact_y"]),
        float(row["contact_z"]),
        float(row["v_x"]),
        float(row["v_y"]),
        float(row["v_z"]),
        config,
    )
    return (
        np.asarray([point.x for point in result.samples], dtype=float),
        np.asarray([point.y for point in result.samples], dtype=float),
        np.asarray([point.z for point in result.samples], dtype=float),
    )


def _scenario_lookup(probe_dir: Path, train_side: Side, config: Any) -> dict[str, dict[str, Any]]:
    path = probe_dir / "controlled_contact_grid_probe_state.json"
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    base_scenarios = list(state.get("scenarios", []))
    scenarios = {str(item["probe_id"]): item for item in base_scenarios}
    grid_side: Side = opponent_side(train_side)  # type: ignore[assignment]
    scenarios.update(
        {
            str(item["probe_id"]): item
            for item in _expand_scenarios_over_opponent_recovery_grid(base_scenarios, grid_side, config)
        }
    )
    return scenarios


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
                response_state = variant["response_state"]
                if opponent_grid_side == "left":
                    response_state["x_left"] = float(x)
                    response_state["y_left"] = float(y)
                else:
                    response_state["x_right"] = float(x)
                    response_state["y_right"] = float(y)
                variant["probe_id"] = probe_id
                variant["contact_probe_id"] = base_probe_id
                variant["opponent_grid_side"] = opponent_grid_side
                variant["opponent_cell_id"] = cell_id
                variant["output_relative_dir"] = f"{base_probe_id}/{cell_id}"
                variant["filename_stem"] = probe_id
                expanded.append(variant)
    return expanded


def _opponent_recovery_grid_cells(
    side: Side,
    config: Any,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    (x_low, x_high), (y_low, y_high) = recovery_bounds(side, config)
    x_grid = _recovery_axis_grid(
        float(x_low),
        float(x_high),
        5,
        lateral_motion_enabled=bool(config.court.lateral_motion_enabled),
    )
    y_grid = _recovery_axis_grid(float(y_low), float(y_high), 5, lateral_motion_enabled=True)
    selected = (0, 2, 4)
    x_labels = ("left", "middle", "right")
    y_labels = ("backcourt", "midcourt", "frontcourt") if side == "left" else ("frontcourt", "midcourt", "backcourt")
    x_cells = tuple((label, float(x_grid[index])) for label, index in zip(x_labels, selected))
    y_cells = tuple((label, float(y_grid[index])) for label, index in zip(y_labels, selected))
    return x_cells, y_cells


def _recovery_axis_grid(lower: float, upper: float, count: int, *, lateral_motion_enabled: bool) -> np.ndarray:
    if count == 1:
        return np.asarray([0.5 * (lower + upper)], dtype=float)
    if lateral_motion_enabled and count in {3, 5}:
        return np.linspace(lower, upper, count + 2)[1:-1]
    return np.linspace(lower, upper, count)


def _opponent_position_from_scenario(scenario: dict[str, Any] | None) -> tuple[float, float] | None:
    if not scenario:
        return None
    response_state = scenario.get("response_state")
    if not isinstance(response_state, dict):
        return None
    hitter = str(response_state.get("current_hitter", "left"))
    opponent = str(scenario.get("opponent_grid_side") or ("right" if hitter == "left" else "left"))
    if opponent == "right" and "x_right" in response_state and "y_right" in response_state:
        return float(response_state["x_right"]), float(response_state["y_right"])
    if opponent == "left" and "x_left" in response_state and "y_left" in response_state:
        return float(response_state["x_left"]), float(response_state["y_left"])
    return None


def _output_relative_dir_for_rows(
    rows: list[dict[str, str]],
    scenario: dict[str, Any] | None,
) -> Path:
    row = rows[-1]
    scenario_relative = None if scenario is None else scenario.get("output_relative_dir")
    if scenario_relative:
        return Path(str(scenario_relative))
    contact_probe_id = str(row.get("contact_probe_id") or "").strip()
    opponent_cell_id = str(row.get("opponent_cell_id") or "").strip()
    if contact_probe_id and opponent_cell_id:
        return Path(contact_probe_id) / opponent_cell_id
    return Path(str(row["probe_id"])) / "opponent_default_position"
    return (
        np.asarray([point.x for point in result.samples], dtype=float),
        np.asarray([point.y for point in result.samples], dtype=float),
        np.asarray([point.z for point in result.samples], dtype=float),
    )


def _expand_slider_court_view(
    ax: Any,
    config: Any,
    trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    fit_z: bool,
) -> None:
    all_x = np.concatenate([trajectory[0] for trajectory in trajectories])
    all_y = np.concatenate([trajectory[1] for trajectory in trajectories])
    all_z = np.concatenate([trajectory[2] for trajectory in trajectories])
    x_pad = 0.25
    y_pad = 0.25
    ax.set_xlim(
        min(-float(config.court.half_width) - x_pad, float(np.nanmin(all_x)) - x_pad),
        max(float(config.court.half_width) + x_pad, float(np.nanmax(all_x)) + x_pad),
    )
    ax.set_ylim(
        min(-float(config.court.half_length) - y_pad, float(np.nanmin(all_y)) - y_pad),
        max(float(config.court.half_length) + y_pad, float(np.nanmax(all_y)) + y_pad),
    )
    z_max = float(config.render.z_max) * 0.82
    if fit_z:
        z_max = max(z_max, float(np.nanmax(all_z)) + 0.35)
    ax.set_zlim(0.0, z_max)
    ax.set_box_aspect((float(config.court.width), float(config.court.length), 4.2))
    ax.view_init(elev=18.0, azim=-62.0)


if __name__ == "__main__":
    main()
