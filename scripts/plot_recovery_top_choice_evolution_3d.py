from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

TARGET_CHECKPOINT_STEPS = [0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000]
OFFICIAL_DOUBLES_WIDTH = 6.10
OFFICIAL_SINGLES_WIDTH = 5.18
OFFICIAL_SHORT_SERVICE_FROM_NET = 1.98
OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK = 0.76

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.eval_evolution import build_sim_config, load_run_config
from badminton.mpl_config import ensure_writable_matplotlib_config
from badminton.trajectory import simulate_trajectory
from scripts.plot_recovery_probe_fixed_shots_3d import (
    _payload_float,
    _row_float,
    _row_response_payloads,
    _shot_action_from_dict,
    _stage_state_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the checkpoint evolution of the top two recovery choices in 3D for "
            "recovery_contact_grid_probe outputs."
        )
    )
    parser.add_argument("probe_dirs", type=Path, nargs="+", help="Probe directories containing *_probe_state.json and *_probe_bins.csv.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory with selfplay_config.json.")
    parser.add_argument("--output-subdir", type=str, default="top_recovery_evolution_3d_views")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_writable_matplotlib_config()
    for probe_dir in args.probe_dirs:
        _render_probe_dir(probe_dir, run_dir=args.run_dir, output_subdir=args.output_subdir, dpi=int(args.dpi))


def _render_probe_dir(probe_dir: Path, *, run_dir: Path | None, output_subdir: str, dpi: int) -> None:
    import matplotlib.pyplot as plt

    state_path = _single_match(probe_dir, "*_probe_state.json")
    bins_path = _single_match(probe_dir, "*_probe_bins.csv")
    probe_state = json.loads(state_path.read_text(encoding="utf-8"))
    resolved_run_dir = run_dir or Path(str(probe_state["run_dir"]))
    config = build_sim_config(load_run_config(resolved_run_dir))
    bins = _load_bins_by_probe(bins_path)
    scenarios = list(probe_state["scenarios"])

    output_dir = probe_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for scenario in scenarios:
        probe_id = str(scenario["probe_id"])
        rows = bins.get(probe_id, [])
        if not rows:
            continue

        fig = plt.figure(figsize=(8.8, 7.4), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        mappable = _plot_scenario(ax, scenario, rows, config, compact=False)
        colorbar = fig.colorbar(
            mappable,
            ax=ax,
            location="top",
            fraction=0.045,
            pad=0.02,
            shrink=0.72,
            label="checkpoint step (M)",
        )
        colorbar.set_ticks(np.arange(7, dtype=float))
        colorbar.ax.set_facecolor("white")
        colorbar.outline.set_edgecolor("black")
        colorbar.ax.tick_params(axis="x", colors="black", labelsize=9)
        colorbar.ax.xaxis.label.set_color("black")
        colorbar.ax.xaxis.set_ticks_position("top")
        colorbar.ax.xaxis.set_label_position("top")

        scenario_dir = probe_dir / probe_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = scenario_dir / f"{probe_id}_top_recovery_evolution_3d.png"
        copy_path = output_dir / f"{probe_id}_top_recovery_evolution_3d.png"
        fig.savefig(scenario_path, dpi=dpi)
        fig.savefig(copy_path, dpi=dpi)
        plt.close(fig)
        paths[probe_id] = str(scenario_path)

    overview_path = output_dir / "recovery_contact_grid_top_recovery_evolution_3d_overview.png"
    _write_overview(overview_path, scenarios, bins, config, dpi=dpi)
    paths["overview"] = str(overview_path)

    manifest_path = output_dir / "top_recovery_evolution_3d_manifest.json"
    manifest_path.write_text(json.dumps({"plots": paths}, indent=2), encoding="utf-8")
    print(f"{probe_dir}: wrote {len(paths) - 1} top-recovery evolution 3D plots")
    print(f"overview: {overview_path}")
    print(f"manifest: {manifest_path}")


def _plot_scenario(ax: Any, scenario: dict[str, Any], rows: list[dict[str, Any]], config: Any, *, compact: bool) -> Any:
    import matplotlib as mpl

    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False

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
    intercept = np.asarray(scenario["actual_intercept_point"], dtype=float)

    _draw_green_court_3d(ax, config)
    _draw_recovery_choice_grid(ax, rows, compact=compact)
    ax.plot(xs, ys, zs, color="tab:blue", linewidth=2.2 if not compact else 1.2, label="fixed shot trajectory", zorder=5)
    ax.scatter([state.x0], [state.y0], [state.z0], color="tab:blue", s=40 if not compact else 18, depthshade=False, label="hitter contact")
    ax.plot([state.x0, state.x0], [state.y0, state.y0], [0.0, state.z0], color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.scatter([intercept[0]], [intercept[1]], [intercept[2]], marker="*", color="crimson", s=150 if not compact else 55, depthshade=False, label="opponent contact")
    ax.plot([intercept[0], intercept[0]], [intercept[1], intercept[1]], [0.0, intercept[2]], color="crimson", linestyle=":", linewidth=1.0)

    ranked_rows = _top_rows_by_target_steps(rows, top_n=2)
    top_rows = [ranked["rows"][0] for ranked in ranked_rows if ranked["rows"]]
    _plot_representative_opponent_response(ax, scenario, top_rows[-1], config, compact=compact)

    norm = mpl.colors.Normalize(vmin=0.0, vmax=6.0)
    cmap = mpl.colormaps.get_cmap("viridis") if hasattr(mpl, "colormaps") else mpl.cm.get_cmap("viridis")
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.asarray([0.0, 6.0], dtype=float))
    marker_positions = _offset_recovery_marker_positions(ranked_rows, compact=compact)

    top_x = np.asarray([marker_positions[(int(ranked["target_step"]), 0)][0] for ranked in ranked_rows], dtype=float)
    top_y = np.asarray([marker_positions[(int(ranked["target_step"]), 0)][1] for ranked in ranked_rows], dtype=float)
    top_z = np.full_like(top_x, 0.26, dtype=float)
    target_steps = np.asarray([float(ranked["target_step"]) / 1_000_000.0 for ranked in ranked_rows], dtype=float)
    if len(top_rows) > 1:
        order = np.argsort(target_steps)
        ax.plot(
            top_x[order],
            top_y[order],
            top_z[order],
            color="0.15",
            linewidth=1.15 if not compact else 0.65,
            alpha=0.55,
            zorder=8,
        )
    second_ranked_rows = [ranked for ranked in ranked_rows if len(ranked["rows"]) > 1]
    if len(second_ranked_rows) > 1:
        second_x = np.asarray(
            [marker_positions[(int(ranked["target_step"]), 1)][0] for ranked in second_ranked_rows],
            dtype=float,
        )
        second_y = np.asarray(
            [marker_positions[(int(ranked["target_step"]), 1)][1] for ranked in second_ranked_rows],
            dtype=float,
        )
        second_z = np.full_like(second_x, 0.18, dtype=float)
        second_steps = np.asarray(
            [float(ranked["target_step"]) / 1_000_000.0 for ranked in second_ranked_rows],
            dtype=float,
        )
        second_order = np.argsort(second_steps)
        ax.plot(
            second_x[second_order],
            second_y[second_order],
            second_z[second_order],
            color="0.35",
            linewidth=0.9 if not compact else 0.5,
            linestyle="--",
            alpha=0.45,
            zorder=7,
        )

    top_scatter = None
    second_scatter = None
    for ranked in ranked_rows:
        target_step_m = float(ranked["target_step"]) / 1_000_000.0
        color = cmap(norm(target_step_m))
        step_rows = ranked["rows"]
        if len(step_rows) > 1:
            second = step_rows[1]
            second_x, second_y = marker_positions[(int(ranked["target_step"]), 1)]
            second_scatter = ax.scatter(
                [second_x],
                [second_y],
                [0.18],
                color=[color],
                marker="^",
                s=58 if not compact else 20,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.78,
                depthshade=False,
                label="second recovery by checkpoint" if second_scatter is None else None,
                zorder=20.1 + target_step_m,
            )
            if hasattr(second_scatter, "set_sort_zpos"):
                second_scatter.set_sort_zpos(1000.0 + target_step_m)
        top = step_rows[0]
        top_x_i, top_y_i = marker_positions[(int(ranked["target_step"]), 0)]
        top_scatter = ax.scatter(
            [top_x_i],
            [top_y_i],
            [0.26],
            color=[color],
            marker="D",
            s=72 if not compact else 24,
            edgecolors="black",
            linewidths=0.35,
            depthshade=False,
            label="top recovery by checkpoint" if top_scatter is None else None,
            zorder=20.2 + target_step_m,
        )
        if hasattr(top_scatter, "set_sort_zpos"):
            top_scatter.set_sort_zpos(1001.0 + target_step_m)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    display_half_width = max(float(config.court.half_width), OFFICIAL_DOUBLES_WIDTH / 2.0)
    ax.set_xlim(-display_half_width - 0.35, display_half_width + 0.35)
    ax.set_ylim(-config.court.half_length - 0.35, config.court.half_length + 0.35)
    ax.set_zlim(0.0, max(config.render.z_max, float(np.nanmax(zs)) + 0.5, float(intercept[2]) + 0.5))
    ax.set_box_aspect((2.0 * display_half_width, config.court.length, 4.8))
    ax.view_init(elev=24, azim=-63)
    _hide_3d_grid(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if not compact:
        ax.legend(loc="upper right", fontsize=8)
    return mappable


def _draw_recovery_choice_grid(ax: Any, rows: list[dict[str, Any]], *, compact: bool) -> None:
    points = sorted(
        {
            (round(float(row["recovery_x"]), 6), round(float(row["recovery_y"]), 6))
            for row in rows
        }
    )
    if not points:
        return

    x_values = sorted({point[0] for point in points})
    y_values = sorted({point[1] for point in points})
    point_set = set(points)
    z_floor = 0.09
    line_kwargs = {
        "color": "#f8fafc",
        "linewidth": 0.65 if not compact else 0.35,
        "alpha": 0.32 if not compact else 0.24,
        "zorder": 5.5,
    }
    for y_value in y_values:
        row_xs = [x_value for x_value in x_values if (x_value, y_value) in point_set]
        if len(row_xs) > 1:
            ax.plot(row_xs, [y_value] * len(row_xs), [z_floor] * len(row_xs), **line_kwargs)
    for x_value in x_values:
        col_ys = [y_value for y_value in y_values if (x_value, y_value) in point_set]
        if len(col_ys) > 1:
            ax.plot([x_value] * len(col_ys), col_ys, [z_floor] * len(col_ys), **line_kwargs)

    xs = np.asarray([point[0] for point in points], dtype=float)
    ys = np.asarray([point[1] for point in points], dtype=float)
    grid_scatter = ax.scatter(
        xs,
        ys,
        np.full_like(xs, z_floor + 0.015, dtype=float),
        marker="o",
        s=12 if not compact else 5,
        facecolors="#f8fafc",
        edgecolors="#111827",
        linewidths=0.25,
        alpha=0.42 if not compact else 0.32,
        depthshade=False,
        label="25 recovery choice grid" if not compact else None,
        zorder=5.8,
    )
    if hasattr(grid_scatter, "set_sort_zpos"):
        grid_scatter.set_sort_zpos(0.12)


def _plot_representative_opponent_response(
    ax: Any,
    scenario: dict[str, Any],
    top_row: dict[str, Any],
    config: Any,
    *,
    compact: bool,
) -> None:
    payloads = _row_response_payloads(top_row)
    valid_payloads = [
        payload
        for payload in payloads
        if all(_payload_float(payload, key) is not None for key in ("opponent_v_x", "opponent_v_y", "opponent_v_z"))
    ]
    if not valid_payloads:
        return
    payload = valid_payloads[0]
    intercept = np.asarray(scenario["actual_intercept_point"], dtype=float)
    trajectory = simulate_trajectory(
        float(intercept[0]),
        float(intercept[1]),
        float(intercept[2]),
        float(_payload_float(payload, "opponent_v_x")),
        float(_payload_float(payload, "opponent_v_y")),
        float(_payload_float(payload, "opponent_v_z")),
        config,
    )
    xs = np.asarray([point.x for point in trajectory.samples], dtype=float)
    ys = np.asarray([point.y for point in trajectory.samples], dtype=float)
    zs = np.asarray([point.z for point in trajectory.samples], dtype=float)
    probability = _payload_float(payload, "probability", default=0.0) or 0.0
    label = "likely opponent response" if compact else f"likely opponent response (p={probability:.2f})"
    ax.plot(xs, ys, zs, color="tab:red", linestyle="--", linewidth=1.8 if not compact else 0.95, alpha=0.9, label=label, zorder=6)
    ax.scatter(
        [trajectory.landing_x],
        [trajectory.landing_y],
        [0.0],
        marker="x",
        color="tab:red",
        s=45 if not compact else 18,
        depthshade=False,
        label="opponent response landing",
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
            color="tab:red",
            edgecolors="black",
            s=65 if not compact else 25,
            depthshade=False,
            label="train response contact",
        )


def _draw_green_court_3d(ax: Any, config: Any) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    half_w = max(float(config.court.half_width), OFFICIAL_DOUBLES_WIDTH / 2.0)
    singles_half_w = OFFICIAL_SINGLES_WIDTH / 2.0
    half_l = float(config.court.half_length)
    net_y = float(config.court.net_y)
    net_h = float(config.court.net_height)
    short_service_y = OFFICIAL_SHORT_SERVICE_FROM_NET
    long_service_doubles_y = half_l - OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK
    surface_z = -0.015
    line_z = 0.045

    court = Poly3DCollection(
        [[
            (-half_w, -half_l, surface_z),
            (half_w, -half_l, surface_z),
            (half_w, half_l, surface_z),
            (-half_w, half_l, surface_z),
        ]],
        facecolors="#15803d",
        edgecolors="none",
        alpha=0.96,
        zorder=0,
    )
    court.set_zsort("min")
    court.set_sort_zpos(surface_z - 1.0)
    ax.add_collection3d(court)

    line_kwargs = {"color": "white", "linewidth": 1.6, "alpha": 1.0, "zorder": 4}
    court_lines = [
        ([-half_w, half_w], [-half_l, -half_l]),
        ([-half_w, half_w], [half_l, half_l]),
        ([-half_w, -half_w], [-half_l, half_l]),
        ([half_w, half_w], [-half_l, half_l]),
        ([-singles_half_w, -singles_half_w], [-half_l, half_l]),
        ([singles_half_w, singles_half_w], [-half_l, half_l]),
        ([-half_w, half_w], [net_y, net_y]),
        ([-half_w, half_w], [-short_service_y, -short_service_y]),
        ([-half_w, half_w], [short_service_y, short_service_y]),
        ([-half_w, half_w], [-long_service_doubles_y, -long_service_doubles_y]),
        ([-half_w, half_w], [long_service_doubles_y, long_service_doubles_y]),
        ([0.0, 0.0], [-half_l, -short_service_y]),
        ([0.0, 0.0], [short_service_y, half_l]),
    ]
    for x_values, y_values in court_lines:
        ax.plot(x_values, y_values, [line_z, line_z], **line_kwargs)

    ax.plot([-half_w, half_w], [net_y, net_y], [net_h, net_h], color="#111827", linewidth=2.0, zorder=5)
    for x_value in (-half_w, half_w):
        ax.plot([x_value, x_value], [net_y, net_y], [0.0, net_h], color="#111827", linewidth=1.0, zorder=5)


def _hide_3d_grid(ax: Any) -> None:
    ax.grid(False)
    for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
        if axis is None:
            continue
        pane = getattr(axis, "pane", None)
        if pane is not None:
            pane.fill = False
            pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        line = getattr(axis, "line", None)
        if line is not None:
            line.set_color((1.0, 1.0, 1.0, 0.0))
            line.set_linewidth(0.0)
        grid_info = getattr(axis, "_axinfo", None)
        if isinstance(grid_info, dict) and "grid" in grid_info:
            grid_info["grid"]["linewidth"] = 0.0
            grid_info["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)


def _write_overview(
    path: Path,
    scenarios: list[dict[str, Any]],
    bins: dict[str, list[dict[str, Any]]],
    config: Any,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17.0, 13.0), constrained_layout=True)
    for index, scenario in enumerate(scenarios, start=1):
        ax = fig.add_subplot(9, 3, index, projection="3d")
        probe_id = str(scenario["probe_id"])
        rows = bins.get(probe_id, [])
        if rows:
            _plot_scenario(ax, scenario, rows, config, compact=True)
        else:
            ax.text2D(0.5, 0.5, "no bins", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(probe_id.replace("_", " "), fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.tick_params(labelsize=6)
    fig.suptitle("Checkpoint evolution of top and second recovery choices", fontsize=16)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _load_bins_by_probe(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_probe: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            probe_id = str(row.get("probe_id", ""))
            if probe_id:
                by_probe.setdefault(probe_id, []).append(row)
    return by_probe


def _top_rows_by_target_steps(rows: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        step = int(row["step"])
        by_step.setdefault(step, []).append(row)
    available_steps = sorted(by_step)
    if not available_steps:
        return []

    selected_steps: dict[int, int] = {}
    for target_step in TARGET_CHECKPOINT_STEPS:
        if target_step == 0:
            selected_steps[target_step] = available_steps[0]
        elif target_step in by_step:
            selected_steps[target_step] = target_step

    result: list[dict[str, Any]] = []
    for target_step in TARGET_CHECKPOINT_STEPS:
        step = selected_steps.get(target_step)
        if step is None:
            continue
        step_rows = sorted(
            by_step[step],
            key=lambda row: (
                -(_row_float(row, "policy_probability", default=0.0) or 0.0),
                int(row.get("recovery_flat_index", 0) or 0),
            ),
        )
        result.append(
            {
                "target_step": target_step,
                "source_step": step,
                "rows": step_rows[:top_n],
            }
        )
    return result


def _offset_recovery_marker_positions(
    ranked_rows: list[dict[str, Any]],
    *,
    compact: bool,
) -> dict[tuple[int, int], tuple[float, float]]:
    entries: list[dict[str, Any]] = []
    for ranked in ranked_rows:
        target_step = int(ranked["target_step"])
        for rank_index, row in enumerate(ranked["rows"][:2]):
            entries.append(
                {
                    "key": (target_step, rank_index),
                    "target_step": target_step,
                    "rank_index": rank_index,
                    "x": float(row["recovery_x"]),
                    "y": float(row["recovery_y"]),
                }
            )

    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault((round(float(entry["x"]), 6), round(float(entry["y"]), 6)), []).append(entry)

    positions: dict[tuple[int, int], tuple[float, float]] = {}
    radius = 0.04 if compact else 0.06
    for group in groups.values():
        group.sort(key=lambda entry: (int(entry["target_step"]), int(entry["rank_index"])))
        if len(group) == 1:
            entry = group[0]
            positions[entry["key"]] = (float(entry["x"]), float(entry["y"]))
            continue
        for index, entry in enumerate(group):
            angle = 2.0 * np.pi * float(index) / float(len(group))
            positions[entry["key"]] = (
                float(entry["x"]) + radius * float(np.cos(angle)),
                float(entry["y"]) + radius * float(np.sin(angle)),
            )
    return positions


def _single_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {pattern} found in {directory}")
    names = ", ".join(path.name for path in matches)
    raise ValueError(f"Multiple {pattern} files found in {directory}: {names}")


def _color_limits(values: np.ndarray) -> tuple[float, float]:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 0.1)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _scenario_title(scenario: dict[str, Any]) -> str:
    return (
        "Top recovery evolution after fixed shot to "
        f"{scenario['target_y_region']} / {scenario['target_x_region']} / {scenario['target_z_level']}"
    )


if __name__ == "__main__":
    main()
