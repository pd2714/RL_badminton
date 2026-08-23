from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.eval_evolution import build_sim_config, load_run_config
from badminton.mpl_config import ensure_writable_matplotlib_config
from badminton.utils import ensure_directory, opponent_side, recovery_bounds

ensure_writable_matplotlib_config()
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pooled landing entropy/variance across the 9 opponent "
            "positions from top-3 controlled-contact evolution samples."
        )
    )
    parser.add_argument("probe_dir", type=Path, help="controlled_contact_grid_probe directory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Defaults to probe summary metadata.")
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=None,
        help="Defaults to top3_expectation_evolution_probe_views/top3_expectation_evolution_samples.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to probe_dir/pooled_landing_metrics.",
    )
    parser.add_argument("--expected-opponent-count", type=int, default=9)
    parser.add_argument("--entropy-eps", type=float, default=1e-9)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_dir = args.probe_dir
    samples_csv = args.samples_csv or (
        probe_dir / "top3_expectation_evolution_probe_views" / "top3_expectation_evolution_samples.csv"
    )
    output_dir = args.output_dir or (probe_dir / "pooled_landing_metrics")
    ensure_directory(output_dir)

    probe_summary = _load_probe_summary(probe_dir)
    run_dir = args.run_dir or Path(str(probe_summary.get("run_dir") or _infer_run_dir(probe_dir)))
    run_config = load_run_config(run_dir)
    config = build_sim_config(run_config)
    train_side = str(run_config.get("train_side", probe_summary.get("train_side", "left")))
    receiver_side = opponent_side(train_side)  # type: ignore[arg-type]
    opponent_positions = _opponent_position_map(receiver_side, config)

    samples = pd.read_csv(samples_csv)
    samples = samples[samples["valid"].astype(bool)].copy()
    samples["opponent_x"] = samples["opponent_cell_id"].map(lambda value: opponent_positions[str(value)][0])
    samples["opponent_y"] = samples["opponent_cell_id"].map(lambda value: opponent_positions[str(value)][1])
    samples["landing_to_opponent_distance"] = np.hypot(
        samples["landing_x"].to_numpy(dtype=float) - samples["opponent_x"].to_numpy(dtype=float),
        samples["landing_y"].to_numpy(dtype=float) - samples["opponent_y"].to_numpy(dtype=float),
    )

    rows: list[dict[str, Any]] = []
    group_cols = ["contact_probe_id", "step"]
    for (contact_probe_id, step), group in samples.groupby(group_cols, sort=True):
        opponent_count = int(group["opponent_cell_id"].nunique())
        if opponent_count != int(args.expected_opponent_count):
            raise ValueError(
                f"{contact_probe_id} step {step} has {opponent_count} opponent cells; "
                f"expected {args.expected_opponent_count}."
            )
        weights = group["top3_weight"].to_numpy(dtype=float) / float(opponent_count)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            continue
        weights = weights / weight_sum

        xy = group[["landing_x", "landing_y"]].to_numpy(dtype=float)
        mean_xy = np.average(xy, axis=0, weights=weights)
        diff = xy - mean_xy
        cov = (diff * weights[:, None]).T @ diff
        cov_eps = cov + float(args.entropy_eps) * np.eye(2)
        det_cov = float(np.linalg.det(cov))
        det_cov_eps = float(np.linalg.det(cov_eps))
        gaussian_entropy = math.log(2.0 * math.pi * math.e) + 0.5 * math.log(max(det_cov_eps, 1e-300))
        avg_distance = float(np.dot(weights, group["landing_to_opponent_distance"].to_numpy(dtype=float)))
        avg_shot_speed = float(np.dot(weights, group["shot_speed"].to_numpy(dtype=float)))
        avg_pressure = float(np.dot(weights, group["pressure"].to_numpy(dtype=float)))
        avg_reaction_miss_pressure = (
            float(np.dot(weights, group["pressure_reaction_miss_score"].to_numpy(dtype=float)))
            if "pressure_reaction_miss_score" in group
            else None
        )

        first = group.iloc[0]
        top3_mass_by_opp = group.groupby("opponent_cell_id")["top3_mass"].first()
        row = {
            "contact_probe_id": contact_probe_id,
            "step": int(step),
            "x_region": first.get("x_region"),
            "y_region": first.get("y_region"),
            "z_level": first.get("z_level"),
            "contact_x": float(first["contact_x"]),
            "contact_y": float(first["contact_y"]),
            "contact_z": float(first["contact_z"]),
            "opponent_count": opponent_count,
            "pooled_sample_count": int(len(group)),
            "pooled_weight_sum": weight_sum,
            "landing_mean_x": float(mean_xy[0]),
            "landing_mean_y": float(mean_xy[1]),
            "landing_var_x": float(cov[0, 0]),
            "landing_var_y": float(cov[1, 1]),
            "landing_cov_xy": float(cov[0, 1]),
            "landing_var_trace": float(cov[0, 0] + cov[1, 1]),
            "landing_cov_det": det_cov,
            "landing_gaussian_entropy": gaussian_entropy,
            "avg_landing_to_opponent_distance": avg_distance,
            "avg_shot_speed": avg_shot_speed,
            "avg_pressure": avg_pressure,
            "top3_mass_mean": float(top3_mass_by_opp.mean()),
            "top3_mass_min": float(top3_mass_by_opp.min()),
            "top3_mass_max": float(top3_mass_by_opp.max()),
        }
        if avg_reaction_miss_pressure is not None:
            row["avg_pressure_reaction_miss_score"] = avg_reaction_miss_pressure
        rows.append(row)

    metrics = pd.DataFrame(rows).sort_values(["y_region", "x_region", "z_level", "step"])
    metrics_path = output_dir / "pooled_landing_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    aggregate = _aggregate_over_contact_positions(metrics)
    aggregate_path = output_dir / "pooled_landing_metrics_contact_average.csv"
    aggregate.to_csv(aggregate_path, index=False)

    opponent_positions_path = output_dir / "opponent_cell_positions.csv"
    pd.DataFrame(
        [
            {"opponent_cell_id": cell_id, "opponent_x": xy[0], "opponent_y": xy[1]}
            for cell_id, xy in sorted(opponent_positions.items())
        ]
    ).to_csv(opponent_positions_path, index=False)

    plot_paths = {
        "landing_var_trace": output_dir / "pooled_landing_var_trace_facets.png",
        "landing_gaussian_entropy": output_dir / "pooled_landing_gaussian_entropy_facets.png",
        "avg_landing_to_opponent_distance": output_dir / "avg_landing_to_opponent_distance_facets.png",
        "avg_shot_speed": output_dir / "avg_shot_speed_facets.png",
        "avg_pressure": output_dir / "avg_pressure_facets.png",
        "avg_pressure_reaction_miss_score": output_dir / "avg_pressure_reaction_miss_facets.png",
        "top3_mass_mean": output_dir / "top3_mass_mean_facets.png",
        "contact_average": output_dir / "pooled_landing_metrics_contact_average.png",
    }
    _write_contact_facets(metrics, "landing_var_trace", "Pooled landing variance trace", plot_paths["landing_var_trace"], args.dpi)
    _write_contact_facets(
        metrics,
        "landing_gaussian_entropy",
        "Pooled Gaussian landing entropy",
        plot_paths["landing_gaussian_entropy"],
        args.dpi,
    )
    _write_contact_facets(
        metrics,
        "avg_landing_to_opponent_distance",
        "Average landing-to-opponent distance",
        plot_paths["avg_landing_to_opponent_distance"],
        args.dpi,
    )
    _write_contact_facets(metrics, "avg_shot_speed", "Average shot speed", plot_paths["avg_shot_speed"], args.dpi)
    _write_contact_facets(metrics, "avg_pressure", "Average pressure", plot_paths["avg_pressure"], args.dpi)
    if "avg_pressure_reaction_miss_score" in metrics:
        _write_contact_facets(
            metrics,
            "avg_pressure_reaction_miss_score",
            "Average reaction-miss pressure component",
            plot_paths["avg_pressure_reaction_miss_score"],
            args.dpi,
        )
    _write_contact_facets(metrics, "top3_mass_mean", "Mean top-3 probability mass", plot_paths["top3_mass_mean"], args.dpi)
    _write_contact_average_plot(aggregate, plot_paths["contact_average"], args.dpi)

    summary_path = output_dir / "pooled_landing_metrics_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "probe_dir": str(probe_dir),
                "samples_csv": str(samples_csv),
                "run_dir": str(run_dir),
                "train_side": train_side,
                "receiver_side": receiver_side,
                "expected_opponent_count": int(args.expected_opponent_count),
                "entropy_epsilon": float(args.entropy_eps),
                "metric_count": int(len(metrics)),
                "contact_state_count": int(metrics["contact_probe_id"].nunique()),
                "anchor_step_count": int(metrics["step"].nunique()),
                "metrics_csv": str(metrics_path),
                "contact_average_csv": str(aggregate_path),
                "opponent_positions_csv": str(opponent_positions_path),
                "plots": {key: str(path) for key, path in plot_paths.items() if path.exists()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"metrics: {metrics_path}")
    print(f"contact_average: {aggregate_path}")
    print(f"opponent_positions: {opponent_positions_path}")
    for path in plot_paths.values():
        print(f"plot: {path}")
    print(f"summary: {summary_path}")


def _load_probe_summary(probe_dir: Path) -> dict[str, Any]:
    path = probe_dir / "controlled_contact_grid_probe_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing probe summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_run_dir(probe_dir: Path) -> Path:
    if probe_dir.parent.name == "anchor_metric_eval":
        return probe_dir.parent.parent
    if probe_dir.parent.parent.name == "anchor_metric_eval":
        return probe_dir.parent.parent.parent
    raise ValueError("Could not infer run_dir; pass --run-dir.")


def _opponent_position_map(side: str, config: Any) -> dict[str, tuple[float, float]]:
    x_cells, y_cells = _opponent_recovery_grid_cells(side, config)
    return {
        f"opponent_{y_label}_{x_label}": (float(x), float(y))
        for y_label, y in y_cells
        for x_label, x in x_cells
    }


def _opponent_recovery_grid_cells(
    side: str,
    config: Any,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    (x_low, x_high), (y_low, y_high) = recovery_bounds(side, config)  # type: ignore[arg-type]
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


def _write_contact_facets(metrics: pd.DataFrame, column: str, title: str, path: Path, dpi: int) -> None:
    y_order = ["backcourt", "midcourt", "frontcourt"]
    x_order = ["left", "middle", "right"]
    z_order = ["high", "mid", "low"]
    row_keys = [(y_region, x_region) for y_region in y_order for x_region in x_order]

    fig, axes = plt.subplots(len(row_keys), len(z_order), figsize=(13.5, 20.0), sharex=True, sharey=False)
    for row_index, (y_region, x_region) in enumerate(row_keys):
        for col_index, z_level in enumerate(z_order):
            ax = axes[row_index, col_index]
            contact_rows = metrics[
                (metrics["y_region"] == y_region) & (metrics["x_region"] == x_region) & (metrics["z_level"] == z_level)
            ].sort_values("step")
            if contact_rows.empty:
                ax.axis("off")
                continue
            ax.plot(contact_rows["step"], contact_rows[column], color="#1f77b4", linewidth=1.6)
            ax.scatter(contact_rows["step"], contact_rows[column], color="#1f77b4", s=8)
            if row_index == 0:
                ax.set_title(z_level)
            if col_index == 0:
                ax.set_ylabel(f"{y_region}\n{x_region}", fontsize=8)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.tick_params(axis="both", labelsize=7)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

    fig.suptitle(title, fontsize=14)
    fig.supxlabel("anchor step")
    fig.supylabel(column)
    fig.tight_layout(rect=(0.04, 0.03, 1.0, 0.98))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _aggregate_over_contact_positions(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "landing_var_trace",
        "landing_gaussian_entropy",
        "avg_landing_to_opponent_distance",
        "avg_shot_speed",
        "avg_pressure",
        "top3_mass_mean",
    ]
    if "avg_pressure_reaction_miss_score" in metrics:
        metric_cols.insert(5, "avg_pressure_reaction_miss_score")
    rows: list[dict[str, Any]] = []
    for step, group in metrics.groupby("step", sort=True):
        row: dict[str, Any] = {
            "step": int(step),
            "contact_state_count": int(group["contact_probe_id"].nunique()),
        }
        for column in metric_cols:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values, ddof=0))
            row[f"{column}_min"] = float(np.min(values))
            row[f"{column}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_contact_average_plot(aggregate: pd.DataFrame, path: Path, dpi: int) -> None:
    panels = [
        ("landing_var_trace", "Pooled landing variance trace"),
        ("landing_gaussian_entropy", "Pooled Gaussian landing entropy"),
        ("avg_landing_to_opponent_distance", "Average landing-to-opponent distance"),
        ("avg_shot_speed", "Average shot speed"),
        ("avg_pressure", "Average pressure"),
    ]
    if "avg_pressure_reaction_miss_score_mean" in aggregate:
        panels.append(("avg_pressure_reaction_miss_score", "Average reaction-miss pressure component"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 12.0), sharex=True)
    steps = aggregate["step"].to_numpy(dtype=float)
    for ax, (column, title) in zip(axes, panels):
        mean = aggregate[f"{column}_mean"].to_numpy(dtype=float)
        std = aggregate[f"{column}_std"].to_numpy(dtype=float)
        ax.plot(steps, mean, color="#1f77b4", linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std, color="#1f77b4", alpha=0.15, linewidth=0.0)
        ax.set_ylabel("mean +/- std")
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    axes[-1].set_xlabel("anchor step")
    fig.suptitle("Average across 27 contact positions", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
