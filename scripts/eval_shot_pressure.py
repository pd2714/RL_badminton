from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton.playback import match_trace_from_dict
from badminton.pressure import (
    ShotPressureWeights,
    evaluate_match_pressure,
    resolve_match_trace_path,
    summarize_match_pressure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate per-shot pressure for a saved match trace or match video.")
    parser.add_argument(
        "match",
        type=Path,
        help="Path to match.mp4, its video directory, or match_trace.json.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional selfplay_config.json or rally_sequence_summary.json.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path. Defaults beside match_trace.json.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path. Defaults beside match_trace.json.")
    parser.add_argument("--no-csv", action="store_true", help="Skip writing the CSV table.")
    parser.add_argument("--speed-weight", type=float, default=ShotPressureWeights.required_speed)
    parser.add_argument("--intercept-weight", type=float, default=ShotPressureWeights.intercept_scarcity)
    parser.add_argument("--height-weight", type=float, default=ShotPressureWeights.low_contact)
    parser.add_argument("--reaction-miss-weight", type=float, default=ShotPressureWeights.reaction_miss)
    parser.add_argument(
        "--reaction-time",
        type=float,
        default=None,
        help="Override receiver reaction time when old traces do not store it.",
    )
    parser.add_argument("--top", type=int, default=8, help="Number of highest-pressure shots to print.")
    return parser.parse_args()


def load_sim_config(path: Path, explicit_config: Path | None = None) -> tuple[SimulationConfig, float | None, Path | None]:
    config_path = explicit_config or _discover_config_path(path)
    if config_path is None:
        return SimulationConfig(), None, None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    reaction_time = _optional_float(data, "reaction_time")
    return _sim_config_from_mapping(data), reaction_time, config_path


def _discover_config_path(path: Path) -> Path | None:
    search_roots = [path if path.is_dir() else path.parent, *path.parents]
    for root in search_roots:
        for filename in ("selfplay_config.json", "rally_sequence_summary.json"):
            candidate = root / filename
            if candidate.exists():
                return candidate
    return None


def _sim_config_from_mapping(data: dict[str, object]) -> SimulationConfig:
    default = SimulationConfig()
    court_mode = str(data.get("court_mode", default.court.mode))
    player_speed = _float(data, "player_speed", default.player.v_max)
    racket_length = _float(data, "racket_length", default.player.r_reach)
    max_hitting_height = _float(data, "max_hitting_height", default.player.z_max)
    player_acceleration = _float(data, "player_acceleration", default.player.acceleration)
    player_deceleration = _optional_float(data, "player_deceleration")
    movement_model = str(data.get("movement_model", default.player.movement_model))
    trajectory_mode = str(data.get("trajectory_mode", default.action.trajectory_mode))
    drag_coefficient = _float(data, "drag_coefficient", default.action.drag_coefficient)
    horizontal_drag = _float(data, "horizontal_drag_coefficient", default.action.horizontal_drag_coefficient or drag_coefficient)
    vertical_drag = _float(data, "vertical_drag_coefficient", default.action.vertical_drag_coefficient or drag_coefficient)
    shuttle_speed_min = _float(data, "shuttle_speed_min", default.action.vy_min_forward)
    shuttle_speed_max = _float(data, "shuttle_speed_max", default.action.vy_max_forward)
    intercept_count = int(data.get("intercept_count", default.action.intercept_count))
    return SimulationConfig(
        court=CourtConfig(mode=court_mode),
        player=PlayerConfig(
            v_max=player_speed,
            r_reach=racket_length,
            z_max=max_hitting_height,
            acceleration=player_acceleration,
            deceleration=player_deceleration,
            movement_model=movement_model,
        ),
        action=ActionConfig(
            trajectory_mode=trajectory_mode,
            drag_coefficient=drag_coefficient,
            horizontal_drag_coefficient=horizontal_drag,
            vertical_drag_coefficient=vertical_drag,
            vy_min_forward=shuttle_speed_min,
            vy_max_forward=shuttle_speed_max,
            intercept_count=intercept_count,
            reaction_miss_fast_threshold=_float(
                data,
                "reaction_miss_fast_threshold",
                default.action.reaction_miss_fast_threshold,
            ),
            reaction_miss_fast_probability=_float(
                data,
                "reaction_miss_fast_probability",
                default.action.reaction_miss_fast_probability,
            ),
            reaction_miss_secondary_threshold=_float(
                data,
                "reaction_miss_secondary_threshold",
                default.action.reaction_miss_secondary_threshold,
            ),
            reaction_miss_secondary_probability=_float(
                data,
                "reaction_miss_secondary_probability",
                default.action.reaction_miss_secondary_probability,
            ),
            reaction_miss_zero_threshold=_float(
                data,
                "reaction_miss_zero_threshold",
                default.action.reaction_miss_zero_threshold,
            ),
        ),
    )


def _float(data: dict[str, object], key: str, default: float) -> float:
    value = data.get(key)
    if value is None:
        return float(default)
    return float(value)


def _optional_float(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return float(value)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "rally_index",
            "rally_number",
            "stage_index",
            "hitter_side",
            "receiver_side",
            "pressure",
        ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary: dict[str, object], top_rows: list[dict[str, object]]) -> None:
    print(f"shots: {summary['shot_count']}")
    print(f"avg pressure: {float(summary['avg_pressure']):.3f}")
    print(f"max pressure: {float(summary['max_pressure']):.3f}")
    print(f"avg required speed: {float(summary['avg_required_speed']):.3f} m/s")
    print(f"avg feasible intercepts: {float(summary['avg_feasible_intercepts']):.2f}")
    if top_rows:
        print("top pressure shots:")
    for row in top_rows:
        print(
            "  "
            f"rally={row['rally_number']} stage={row['stage_index']} "
            f"{row['hitter_side']}->{row['receiver_side']} "
            f"pressure={float(row['pressure']):.3f} "
            f"speed={float(row['required_speed']):.2f}m/s "
            f"miss={float(row['reaction_miss_score']):.2f} "
            f"feasible={row['feasible_intercept_count']}/{row['candidate_intercept_count']} "
            f"best_z={_format_optional(row['best_contact_height'])}"
        )


def _format_optional(value: object) -> str:
    if value is None:
        return "None"
    return f"{float(value):.2f}m"


def main() -> None:
    args = parse_args()
    trace_path = resolve_match_trace_path(args.match)
    config, discovered_reaction_time, config_path = load_sim_config(trace_path, args.config)
    reaction_time = args.reaction_time if args.reaction_time is not None else discovered_reaction_time
    trace = match_trace_from_dict(json.loads(trace_path.read_text(encoding="utf-8")))
    weights = ShotPressureWeights(
        required_speed=args.speed_weight,
        intercept_scarcity=args.intercept_weight,
        low_contact=args.height_weight,
        reaction_miss=args.reaction_miss_weight,
    )
    rows = evaluate_match_pressure(trace, config, receiver_reaction_time=reaction_time, weights=weights)
    row_dicts = [row.to_dict() for row in rows]
    summary = summarize_match_pressure(rows)
    output_json = args.output_json or trace_path.with_name("shot_pressure.json")
    output_csv = args.output_csv or trace_path.with_name("shot_pressure.csv")

    payload = {
        "match_path": str(args.match),
        "trace_path": str(trace_path),
        "config_path": None if config_path is None else str(config_path),
        "reaction_time": reaction_time,
        "weights": {
            "required_speed": weights.required_speed,
            "intercept_scarcity": weights.intercept_scarcity,
            "low_contact": weights.low_contact,
            "reaction_miss": weights.reaction_miss,
        },
        "summary": summary,
        "shots": row_dicts,
    }
    _write_json(output_json, payload)
    if not args.no_csv:
        _write_csv(output_csv, row_dicts)

    _print_summary(summary, sorted(row_dicts, key=lambda item: float(item["pressure"]), reverse=True)[: max(args.top, 0)])
    print(f"json: {output_json}")
    if not args.no_csv:
        print(f"csv: {output_csv}")


if __name__ == "__main__":
    main()
