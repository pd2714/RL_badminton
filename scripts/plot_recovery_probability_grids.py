from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.eval_evolution import build_sim_config, load_run_config
from badminton.mpl_config import ensure_writable_matplotlib_config
from badminton.render import setup_court_axes, stage_colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot latest recovery policy probability grids for a recovery probe.")
    parser.add_argument("probe_dir", type=Path, help="Directory containing *_probe_state.json and *_probe_bins.csv.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory with selfplay_config.json.")
    parser.add_argument("--output-subdir", type=str, default="recovery_probability_grids")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_writable_matplotlib_config()
    import matplotlib.pyplot as plt

    state_path = _single_match(args.probe_dir, "*_probe_state.json")
    bins_path = _single_match(args.probe_dir, "*_probe_bins.csv")
    probe_state = json.loads(state_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir or Path(str(probe_state["run_dir"]))
    config = build_sim_config(load_run_config(run_dir))
    latest_bins = _load_latest_bins(bins_path)

    output_dir = args.probe_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    scenarios = list(probe_state["scenarios"])
    for scenario in scenarios:
        probe_id = str(scenario["probe_id"])
        rows = latest_bins.get(probe_id, [])
        fig, ax = plt.subplots(figsize=(6.8, 7.5), constrained_layout=True)
        _plot_grid(ax, scenario, rows, config, annotate=True)
        fig.suptitle(_scenario_title(scenario), fontsize=13)

        scenario_dir = args.probe_dir / probe_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = scenario_dir / f"{probe_id}_recovery_probability_grid.png"
        copy_path = output_dir / f"{probe_id}_recovery_probability_grid.png"
        fig.savefig(scenario_path, dpi=int(args.dpi))
        fig.savefig(copy_path, dpi=int(args.dpi))
        plt.close(fig)
        paths[probe_id] = str(scenario_path)

    overview_path = output_dir / "recovery_probability_grid_overview.png"
    _write_overview(overview_path, scenarios, latest_bins, config, dpi=int(args.dpi))
    paths["overview"] = str(overview_path)

    manifest_path = output_dir / "recovery_probability_grid_manifest.json"
    manifest_path.write_text(json.dumps({"plots": paths}, indent=2), encoding="utf-8")
    print(f"wrote {len(scenarios)} recovery probability grids")
    print(f"overview: {overview_path}")
    print(f"manifest: {manifest_path}")


def _single_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {pattern} found in {directory}")
    names = ", ".join(path.name for path in matches)
    raise ValueError(f"Multiple {pattern} files found in {directory}: {names}")


def _load_latest_bins(path: Path) -> dict[str, list[dict[str, Any]]]:
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


def _plot_grid(
    ax: Any,
    scenario: dict[str, Any],
    rows: list[dict[str, Any]],
    config: Any,
    *,
    annotate: bool,
) -> None:
    setup_court_axes(ax, config, stage_colors(monochrome=True), show_axes=True)
    if not rows:
        ax.text(0.5, 0.5, "no recovery bins", transform=ax.transAxes, ha="center", va="center")
        return

    xs = sorted({float(row["recovery_x"]) for row in rows})
    ys = sorted({float(row["recovery_y"]) for row in rows})
    grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
    by_xy = {(float(row["recovery_x"]), float(row["recovery_y"])): float(row["policy_probability"]) for row in rows}
    for y_index, y_value in enumerate(ys):
        for x_index, x_value in enumerate(xs):
            grid[y_index, x_index] = by_xy.get((x_value, y_value), np.nan)

    x_edges = _edges(xs)
    y_edges = _edges(ys)
    mesh = ax.pcolormesh(x_edges, y_edges, grid, cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(grid)), 1e-6), shading="auto", alpha=0.82)
    ax.figure.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02, label="latest policy probability")

    finite = np.isfinite(grid)
    if finite.any():
        max_y, max_x = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
        ax.scatter([xs[max_x]], [ys[max_y]], marker="D", s=90, color="gold", edgecolors="black", zorder=7, label="top recovery")

    if annotate:
        for y_index, y_value in enumerate(ys):
            for x_index, x_value in enumerate(xs):
                value = grid[y_index, x_index]
                if np.isfinite(value):
                    color = "white" if value > 0.35 * max(float(np.nanmax(grid)), 1e-6) else "black"
                    ax.text(x_value, y_value, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color, zorder=8)

    state = scenario["state_before"]
    target = scenario["target_point"]
    intercept = scenario["actual_intercept_point"]
    ax.scatter([float(state["x0"])], [float(state["y0"])], color="tab:blue", s=52, zorder=8, label="hitter contact")
    ax.scatter([float(intercept[0])], [float(intercept[1])], marker="*", color="crimson", s=130, zorder=9, label="opponent intercept")
    ax.scatter(
        [float(target["x"])],
        [float(target["y"])],
        marker="o",
        facecolors="none",
        edgecolors="tab:orange",
        s=80,
        linewidths=1.5,
        zorder=9,
        label="requested target",
    )
    top_probability = max(float(row["policy_probability"]) for row in rows)
    ax.set_title(f"step {int(rows[0]['step'])} | top p={top_probability:.2f}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def _edges(values: list[float]) -> np.ndarray:
    if len(values) == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=float)
    centers = np.asarray(values, dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def _write_overview(
    path: Path,
    scenarios: list[dict[str, Any]],
    latest_bins: dict[str, list[dict[str, Any]]],
    config: Any,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(9, 3, figsize=(13.0, 28.0), constrained_layout=True)
    for ax, scenario in zip(axes.flat, scenarios):
        probe_id = str(scenario["probe_id"])
        _plot_grid(ax, scenario, latest_bins.get(probe_id, []), config, annotate=False)
        ax.set_title(probe_id.replace("_", " "), fontsize=8)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.suptitle("Latest recovery policy probability grids", fontsize=16)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _scenario_title(scenario: dict[str, Any]) -> str:
    return (
        "Recovery probability grid after fixed shot to "
        f"{scenario['target_y_region']} / {scenario['target_x_region']} / {scenario['target_z_level']}"
    )


if __name__ == "__main__":
    main()
