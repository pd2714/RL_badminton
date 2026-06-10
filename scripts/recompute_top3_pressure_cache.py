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

from badminton1d.dynamics import candidate_intercept_points, feasible_intercept_indices, reaction_time_for_side
from badminton1d.eval_evolution import build_sim_config, load_run_config
from badminton1d.pressure import shot_pressure_from_candidates
from badminton1d.state import ShotAction
from badminton1d.utils import player_position, player_velocity

from scripts.evaluate_controlled_lift_probe import _write_csv
from scripts.render_controlled_contact_top3_expectation_evolution_plots import (
    _expanded_scenarios,
    _load_probe_summary,
    _write_evolution_plots,
)


PRESSURE_SUMMARY_KEYS = (
    "shot_speed",
    "pressure",
    "pressure_reaction_miss_score",
    "shot_value",
    "after_win_probability",
    "anchor_shot_value",
    "anchor_after_win_probability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute pressure fields in cached top-3 controlled-contact samples "
            "using the current pressure metric, preserving the cached actions."
        )
    )
    parser.add_argument("probe_dir", type=Path, help="controlled_contact_grid_probe directory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Defaults to probe summary metadata.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Defaults to probe_dir/top3_expectation_evolution_probe_views.",
    )
    parser.add_argument("--write-cache-only", action="store_true", help="Skip regenerating per-scenario probe PNGs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_dir = args.probe_dir
    cache_dir = args.cache_dir or probe_dir / "top3_expectation_evolution_probe_views"
    samples_path = cache_dir / "top3_expectation_evolution_samples.csv"
    summary_path = cache_dir / "top3_expectation_evolution_summary.csv"
    summary_json_path = cache_dir / "top3_expectation_evolution_summary.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing cached samples: {samples_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing cached summary: {summary_path}")

    probe_summary = _load_probe_summary(probe_dir)
    run_dir = args.run_dir or Path(str(probe_summary.get("run_dir")))
    run_config = load_run_config(run_dir)
    config = build_sim_config(run_config)
    train_side = str(run_config.get("train_side", probe_summary.get("train_side", "left")))
    scenarios = _expanded_scenarios(probe_dir, probe_summary, train_side, config)
    scenario_by_probe_id = {scenario.probe_id: scenario for scenario in scenarios}

    sample_rows = _read_csv(samples_path)
    for row in sample_rows:
        scenario = scenario_by_probe_id[str(row["probe_id"])]
        pressure = _pressure_for_cached_sample(row, scenario.response_state, config)
        row["pressure"] = pressure.pressure
        row["pressure_required_speed_score"] = pressure.required_speed_score
        row["pressure_intercept_scarcity_score"] = pressure.intercept_scarcity_score
        row["pressure_low_contact_score"] = pressure.low_contact_score
        row["pressure_reaction_miss_score"] = pressure.reaction_miss_score

    summary_rows = _read_csv(summary_path)
    summaries_by_key = {(str(row["probe_id"]), int(float(row["step"]))): row for row in summary_rows}
    samples_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        samples_by_key[(str(row["probe_id"]), int(float(row["step"])))].append(row)

    for key, rows in samples_by_key.items():
        summary = summaries_by_key.get(key)
        if summary is None:
            continue
        _update_pressure_summary(summary, rows)

    _write_csv(samples_path, sample_rows)
    _write_csv(summary_path, summary_rows)
    if summary_json_path.exists():
        data = json.loads(summary_json_path.read_text(encoding="utf-8"))
        data["rows"] = summary_rows
        data["pressure_recompute"] = {
            "metric": "current",
            "reaction_miss_score": "mean_feasible_reaction_miss_probability",
        }
        summary_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if not args.write_cache_only:
        _write_evolution_plots(probe_dir, scenarios, summary_rows, sample_rows, config)

    print(f"updated samples: {samples_path}")
    print(f"updated summary: {summary_path}")
    if summary_json_path.exists():
        print(f"updated summary_json: {summary_json_path}")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _pressure_for_cached_sample(row: dict[str, Any], state: Any, config: Any) -> Any:
    action = ShotAction(
        v_x=float(row["v_x"]),
        v_y=float(row["v_y"]),
        v_z=float(row["v_z"]),
        x_rec=0.0,
        y_rec=0.0,
    )
    times, xs, ys, zs = candidate_intercept_points(state, action, config)
    feasible = feasible_intercept_indices(state, action, config, candidates=(times, xs, ys, zs))
    receiver_side = "right" if state.current_hitter == "left" else "left"
    return shot_pressure_from_candidates(
        receiver_side=receiver_side,
        receiver_start=player_position(state, receiver_side),
        receiver_velocity=player_velocity(state, receiver_side),
        receiver_reaction_time=reaction_time_for_side(state, receiver_side),
        candidate_times=times,
        candidate_xs=xs,
        candidate_ys=ys,
        candidate_zs=zs,
        feasible_indices=feasible,
        config=config,
        terminal_reason=row.get("terminal_reason") or None,
    )


def _update_pressure_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    weights = np.asarray([float(row.get("top3_weight", 0.0) or 0.0) for row in rows], dtype=float)
    if weights.size and float(np.sum(weights)) > 0.0:
        weights = weights / float(np.sum(weights))
    for key in PRESSURE_SUMMARY_KEYS:
        values = [_optional_float(row.get(key)) for row in rows]
        finite_values = np.asarray([value for value in values if value is not None], dtype=float)
        if finite_values.size == weights.size and finite_values.size:
            mean = float(np.dot(weights, finite_values))
            std = float(math.sqrt(max(float(np.dot(weights, (finite_values - mean) ** 2)), 0.0)))
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    main()
