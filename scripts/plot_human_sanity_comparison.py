from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.config import SimulationConfig
from badminton1d.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle

from badminton1d.shot_generators import name_velocity_shot


DEFAULT_RUN_DIRS = tuple(
    REPO_ROOT / "outputs/rl/ginsburg_20260622" / f"pure_cfa_seed{seed}"
    for seed in (17, 23, 31, 47, 59)
)
DEFAULT_OUTPUT = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/human_sanity_comparison.png"
DEFAULT_SOURCE_DATA = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/source_data/human_sanity_comparison_summary.json"
DEFAULT_HUMAN_EVENTS = REPO_ROOT / "data/human/shuttleset22_events.csv"
DEFAULT_PANEL_C_MODEL_PROBE_NAME = "controlled_contact_grid_opponent_grid3x3"
DEFAULT_PANEL_C_MODEL_SAMPLES = (
    REPO_ROOT
    / "outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
    / "anchor_metric_eval/controlled_contact_grid_probe/top3_expectation_evolution_probe_views/top3_expectation_evolution_samples.csv"
)
DEFAULT_EXTRA_TRACE_PATHS = (
    REPO_ROOT
    / "outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611"
    / "human_sanity_rally_eval/match_trace.json",
)

INK = "#111827"
MUTED = "#6b7280"
GRID = "#d1d5db"
MODEL = "#2563eb"
HUMAN = "#d97706"
ATTACK = "#dc2626"
NEUTRAL = "#2563eb"
DEFENSE = "#16a34a"

SHOT_FAMILIES = ("net/drop", "lift/clear", "smash/drive")
TERMINAL_ORDER = (
    "no_feasible_intercept",
    "reaction_miss",
    "illegal_shot",
    "ground",
    "opponent_no_valid_shot",
    "max_length",
)
TERMINAL_DISPLAY = {
    "no_feasible_intercept": "missed\nintercept",
    "reaction_miss": "reaction\nmiss",
    "illegal_shot": "illegal\nshot",
    "ground": "ground",
    "opponent_no_valid_shot": "no valid\nreply",
    "max_length": "max\nlength",
    "other": "unlabeled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot lightweight ShuttleArena-vs-human sanity comparisons. By default, the human "
            "overlay uses the prepared ShuttleSet22 event CSV."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Model run directory. May be repeated. Defaults to the five Ginsburg pure-CFA seeds.",
    )
    parser.add_argument("--human-events", type=Path, default=DEFAULT_HUMAN_EVENTS, help="One-row-per-shot human event CSV.")
    parser.add_argument("--human-label", default="ShuttleSet22")
    parser.add_argument("--model-label", default="ShuttleArena policy")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-data-out", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--coordinate-mode", choices=("auto", "court", "normalized", "positive-court"), default="auto")
    parser.add_argument(
        "--panel-c-model-samples",
        type=Path,
        default=DEFAULT_PANEL_C_MODEL_SAMPLES,
        help=(
            "CSV to use first for panel C model landings. Defaults to the local main-run "
            "27-contact x 9-opponent top-3 expectation samples."
        ),
    )
    parser.add_argument(
        "--panel-c-model-probe-name",
        default=DEFAULT_PANEL_C_MODEL_PROBE_NAME,
        help=(
            "Probe name to prefer for panel C model landings. If "
            "RUN_DIR/anchor_metric_eval/<name>_probe/<name>_probe_samples.csv is absent for all run dirs, "
            "panel C falls back to the standard controlled_contact_grid probe."
        ),
    )
    parser.add_argument(
        "--trace-glob",
        default="videos/human_sanity_match_traces/*/match_trace.json",
        help="Glob under --run-dir for RL match traces used for rally lengths and terminal reasons.",
    )
    parser.add_argument(
        "--extra-trace-path",
        type=Path,
        action="append",
        default=None,
        help=(
            "Additional model match_trace.json file to include for rally-length statistics. "
            "Defaults to the earlier single-run human-sanity rally-eval trace."
        ),
    )
    parser.add_argument(
        "--no-default-extra-trace",
        action="store_true",
        help="Use only traces found under --run-dir/--trace-glob, omitting the default earlier rally-eval trace.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def _int_or_none(value: Any) -> int | None:
    value_float = _float_or_none(value)
    if value_float is None:
        return None
    return int(round(value_float))


def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    lower = {key.lower(): key for key in row}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return row[key]
    return None


def _canonical_shot_family(shot_type: str | None) -> str:
    label = (shot_type or "").strip().lower()
    if any(token in label for token in ("net", "drop")):
        return "net/drop"
    if any(token in label for token in ("lift", "clear", "lob")):
        return "lift/clear"
    if any(token in label for token in ("smash", "drive", "push", "flat")):
        return "smash/drive"
    return "other"


def _is_service(shot_type: str | None) -> bool:
    label = (shot_type or "").strip().lower()
    return "service" in label or "serve" in label


def _terminal_bucket(reason: str | None) -> str:
    label = (reason or "").strip().lower()
    if not label:
        return "other"
    if "feasible" in label or "intercept" in label:
        return "no_feasible_intercept"
    if "reaction_miss" in label or "reaction miss" in label:
        return "reaction_miss"
    if "invalid" in label or "out" in label or "bound" in label or "net" in label:
        return "illegal_shot"
    if "ground" in label or "floor" in label:
        return "ground"
    if "opponent_no_valid" in label or "no_valid" in label:
        return "opponent_no_valid_shot"
    if "max" in label or "trunc" in label:
        return "max_length"
    return "other"


def _contact_zone_from_y(y: float | None, config: SimulationConfig) -> str:
    if y is None:
        return "unknown"
    absolute = abs(float(y))
    if absolute >= 0.67 * config.court.half_length:
        return "backcourt"
    if absolute <= 0.33 * config.court.half_length:
        return "frontcourt"
    return "midcourt"


def _opponent_cell_from_xy(x: float | None, y: float | None, config: SimulationConfig) -> str | None:
    if x is None or y is None:
        return None
    x_edges = np.linspace(-config.court.half_width, config.court.half_width, 4)
    y_edges = np.linspace(-config.court.half_length, config.court.half_length, 4)
    x_names = ("left", "middle", "right")
    y_names = ("backcourt", "midcourt", "frontcourt") if y < 0.0 else ("frontcourt", "midcourt", "backcourt")
    x_idx = int(np.clip(np.searchsorted(x_edges, float(x), side="right") - 1, 0, 2))
    y_idx = int(np.clip(np.searchsorted(y_edges, float(y), side="right") - 1, 0, 2))
    return f"opponent_{y_names[y_idx]}_{x_names[x_idx]}"


def _normalize_coordinates(rows: list[dict[str, Any]], mode: str, config: SimulationConfig) -> None:
    coordinate_keys = (
        "contact_x",
        "contact_y",
        "landing_x",
        "landing_y",
        "recovery_x",
        "recovery_y",
        "opponent_x",
        "opponent_y",
    )
    raw_x = [float(row[key]) for row in rows for key in coordinate_keys if key.endswith("_x") and row.get(key) is not None]
    raw_y = [float(row[key]) for row in rows for key in coordinate_keys if key.endswith("_y") and row.get(key) is not None]
    values_x = [abs(value) for value in raw_x]
    values_y = [abs(value) for value in raw_y]
    if not rows:
        return
    active_mode = mode
    if mode == "auto":
        max_x = max(values_x) if values_x else 0.0
        max_y = max(values_y) if values_y else 0.0
        min_raw_x = min(raw_x) if raw_x else 0.0
        min_raw_y = min(raw_y) if raw_y else 0.0
        if max_x <= 1.05 and max_y <= 1.05:
            active_mode = "normalized"
        elif min_raw_x >= -0.05 and min_raw_y >= -0.05 and max_x <= config.court.width + 0.5 and max_y <= config.court.length + 0.5:
            active_mode = "positive-court"
        else:
            active_mode = "court"

    for row in rows:
        for stem in ("contact", "landing", "recovery", "opponent"):
            x_key = f"{stem}_x"
            y_key = f"{stem}_y"
            x = row.get(x_key)
            y = row.get(y_key)
            if x is None or y is None:
                continue
            if active_mode == "normalized":
                row[x_key] = (float(x) - 0.5) * config.court.width
                row[y_key] = (float(y) - 0.5) * config.court.length
            elif active_mode == "positive-court":
                row[x_key] = float(x) - config.court.half_width
                row[y_key] = float(y) - config.court.half_length


def load_human_events(path: Path | None, config: SimulationConfig, coordinate_mode: str) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            event = {
                "event_index": index,
                "rally_id": _first(row, ("rally_id", "rally", "rally_number", "rally_no", "rallyid")),
                "stage_index": _int_or_none(_first(row, ("stage_index", "shot_index", "ball_round", "round", "stroke_index"))),
                "rally_length": _int_or_none(_first(row, ("rally_length", "rally_len", "shots_in_rally"))),
                "shot_type": _first(row, ("shot_type", "type", "stroke", "stroke_type", "ball_type", "shot")),
                "terminal_reason": _first(row, ("terminal_reason", "error_type", "winner_reason", "outcome", "terminal", "ending")),
                "contact_x": _float_or_none(_first(row, ("contact_x", "player_location_x", "hitter_x", "hit_x", "x0"))),
                "contact_y": _float_or_none(_first(row, ("contact_y", "player_location_y", "hitter_y", "hit_y", "y0"))),
                "landing_x": _float_or_none(_first(row, ("landing_x", "landing_location_x", "shuttle_x", "target_x"))),
                "landing_y": _float_or_none(_first(row, ("landing_y", "landing_location_y", "shuttle_y", "target_y"))),
                "recovery_x": _float_or_none(_first(row, ("recovery_x", "recover_x", "player_recovery_x", "next_player_location_x"))),
                "recovery_y": _float_or_none(_first(row, ("recovery_y", "recover_y", "player_recovery_y", "next_player_location_y"))),
                "opponent_x": _float_or_none(_first(row, ("opponent_x", "opponent_location_x", "receiver_x"))),
                "opponent_y": _float_or_none(_first(row, ("opponent_y", "opponent_location_y", "receiver_y"))),
                "opponent_cell_id": _first(row, ("opponent_cell_id", "opponent_zone", "receiver_zone")),
                "contact_zone": _first(row, ("contact_zone", "y_region", "court_region", "hit_region")),
            }
            rows.append(event)
    _normalize_coordinates(rows, coordinate_mode, config)
    for row in rows:
        if not row.get("contact_zone"):
            row["contact_zone"] = _contact_zone_from_y(row.get("contact_y"), config)
        if not row.get("opponent_cell_id"):
            row["opponent_cell_id"] = _opponent_cell_from_xy(row.get("opponent_x"), row.get("opponent_y"), config)
    return rows


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("valid", "True")).strip().lower() == "true"]


def _run_seed(run_dir: Path) -> int | None:
    stem = run_dir.name
    if "seed" not in stem:
        return None
    return _int_or_none(stem.rsplit("seed", 1)[-1])


def _model_event_from_probe_row(row: dict[str, Any], config: SimulationConfig, *, run_dir: Path | None, seed: int | None) -> dict[str, Any]:
    contact_y = _float_or_none(row.get("contact_y"))
    event = {
        "run_dir": None if run_dir is None else str(run_dir),
        "seed": seed,
        "checkpoint_step": _int_or_none(row.get("step")),
        "checkpoint_path": row.get("checkpoint_path"),
        "event_index": None,
        "rally_id": None,
        "stage_index": None,
        "rally_length": None,
        "shot_type": row.get("shot_type"),
        "terminal_reason": row.get("terminal_reason"),
        "contact_x": _float_or_none(row.get("contact_x")),
        "contact_y": contact_y,
        "landing_x": _float_or_none(row.get("landing_x")),
        "landing_y": _float_or_none(row.get("landing_y")),
        "recovery_x": _float_or_none(row.get("recovery_x")),
        "recovery_y": _float_or_none(row.get("recovery_y")),
        "opponent_x": _float_or_none(row.get("opponent_x")),
        "opponent_y": _float_or_none(row.get("opponent_y")),
        "opponent_cell_id": row.get("opponent_cell_id"),
        "contact_zone": row.get("y_region") or _contact_zone_from_y(contact_y, config),
    }
    weight = _float_or_none(row.get("top3_weight"))
    if weight is None:
        weight = _float_or_none(row.get("plot_weight"))
    if weight is not None and weight > 0.0:
        event["plot_weight"] = weight
    return event


def model_event_rows(
    run_dirs: list[Path],
    config: SimulationConfig,
    *,
    probe_name: str = "controlled_contact_grid",
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        contact_path = run_dir / "anchor_metric_eval" / f"{probe_name}_probe" / f"{probe_name}_probe_samples.csv"
        contact_rows = valid_rows(load_csv_rows(contact_path)) if contact_path.exists() else []
        seed = _run_seed(run_dir)
        for row in contact_rows:
            event = _model_event_from_probe_row(row, config, run_dir=run_dir, seed=seed)
            event["event_index"] = len(events)
            events.append(event)
    return events


def model_event_rows_from_samples_csv(path: Path, config: SimulationConfig) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = valid_rows(load_csv_rows(path))
    events: list[dict[str, Any]] = []
    for row in rows:
        event = _model_event_from_probe_row(row, config, run_dir=None, seed=None)
        event["event_index"] = len(events)
        events.append(event)
    return events


def panel_c_model_event_rows(
    run_dirs: list[Path],
    config: SimulationConfig,
    *,
    samples_path: Path | None,
    preferred_probe_name: str,
    fallback_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if samples_path is not None:
        sample_events = model_event_rows_from_samples_csv(samples_path, config)
        if sample_events:
            return sample_events, str(samples_path)
    preferred_events = model_event_rows(run_dirs, config, probe_name=preferred_probe_name)
    if preferred_events:
        return preferred_events, f"{preferred_probe_name}_probe"
    return fallback_events, "controlled_contact_grid_probe"


def _trace_paths(run_dirs: list[Path], trace_glob: str) -> list[Path]:
    return sorted(path for run_dir in run_dirs for path in run_dir.glob(trace_glob))


def model_match_rows(run_dirs: list[Path], trace_glob: str) -> tuple[list[int], Counter[str]]:
    return model_match_rows_from_trace_paths(_trace_paths(run_dirs, trace_glob))


def model_match_rows_from_trace_paths(trace_paths: list[Path]) -> tuple[list[int], Counter[str]]:
    rally_lengths: list[int] = []
    terminal_counts: Counter[str] = Counter()
    for trace_path in trace_paths:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        for rally in payload.get("rallies", []):
            stages = rally.get("stages", [])
            rally_lengths.append(len(stages))
            terminal_reason = None
            if stages:
                terminal_reason = stages[-1].get("terminal_reason")
            terminal_counts[_terminal_bucket(terminal_reason)] += 1
    return rally_lengths, terminal_counts


def _point_xy(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None
    return _float_or_none(value[0]), _float_or_none(value[1])


def _stage_shot_type(stage: dict[str, Any], config: SimulationConfig) -> str:
    if _int_or_none(stage.get("stage_index")) == 0:
        return "service"
    contact_x, contact_y = _point_xy(stage.get("shuttle_start"))
    landing_x, landing_y = _point_xy(stage.get("shuttle_landing"))
    velocity = stage.get("shuttle_velocity")
    if contact_x is None or contact_y is None or landing_x is None or landing_y is None:
        return "other"
    if not isinstance(velocity, (list, tuple)) or len(velocity) < 3:
        return "other"
    vx = _float_or_none(velocity[0])
    vy = _float_or_none(velocity[1])
    vz = _float_or_none(velocity[2])
    if vx is None or vy is None or vz is None:
        return "other"
    theta_degrees = float(np.degrees(np.arctan2(vz, max(float(np.hypot(vx, vy)), 1e-9))))
    hitter = str(stage.get("hitter_side") or "left").lower()
    if hitter not in {"left", "right"}:
        hitter = "left"
    return name_velocity_shot(
        hitter=hitter,  # type: ignore[arg-type]
        contact_x=contact_x,
        contact_y=contact_y,
        landing_x=landing_x,
        landing_y=landing_y,
        theta_degrees=theta_degrees,
        config=config,
    )


def _canonicalize_stage_y(stage: dict[str, Any], *points: tuple[float | None, float | None]) -> list[tuple[float | None, float | None]]:
    hitter = str(stage.get("hitter_side") or "").lower()
    contact_x, contact_y = points[0]
    flip_y = hitter == "right" or (hitter not in {"left", "right"} and contact_y is not None and float(contact_y) > 0.0)
    if not flip_y:
        return list(points)
    canonical: list[tuple[float | None, float | None]] = []
    for x, y in points:
        canonical.append((x, None if y is None else -float(y)))
    return canonical


def model_trace_event_rows(run_dirs: list[Path], trace_glob: str, config: SimulationConfig) -> list[dict[str, Any]]:
    return model_trace_event_rows_from_trace_paths(_trace_paths(run_dirs, trace_glob), config)


def model_trace_event_rows_from_trace_paths(trace_paths: list[Path], config: SimulationConfig) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        run_name = trace_path.parents[3].name if len(trace_path.parents) > 3 else trace_path.parent.name
        for rally in payload.get("rallies", []):
            rally_id = f"{run_name}__{trace_path.parent.name}__rally{rally.get('rally_number', len(events))}"
            stages = rally.get("stages", [])
            for stage in stages:
                contact = _point_xy(stage.get("shuttle_start"))
                landing = _point_xy(stage.get("shuttle_landing"))
                recovery = _point_xy(stage.get("hitter_end") or stage.get("recovery_target"))
                opponent = _point_xy(stage.get("receiver_start"))
                contact, landing, recovery, opponent = _canonicalize_stage_y(stage, contact, landing, recovery, opponent)
                contact_x, contact_y = contact
                landing_x, landing_y = landing
                recovery_x, recovery_y = recovery
                opponent_x, opponent_y = opponent
                events.append(
                    {
                        "event_index": len(events),
                        "rally_id": rally_id,
                        "stage_index": _int_or_none(stage.get("stage_index")),
                        "rally_length": len(stages),
                        "shot_type": _stage_shot_type(stage, config),
                        "terminal_reason": stage.get("terminal_reason"),
                        "contact_x": contact_x,
                        "contact_y": contact_y,
                        "landing_x": landing_x,
                        "landing_y": landing_y,
                        "recovery_x": recovery_x,
                        "recovery_y": recovery_y,
                        "opponent_x": opponent_x,
                        "opponent_y": opponent_y,
                        "opponent_cell_id": _opponent_cell_from_xy(opponent_x, opponent_y, config),
                        "contact_zone": _contact_zone_from_y(contact_y, config),
                        "checkpoint_step": None,
                        "checkpoint_path": None,
                    }
                )
    return events


def human_rally_lengths(events: list[dict[str, Any]]) -> list[int]:
    by_rally: dict[str, int] = defaultdict(int)
    for row in events:
        rally_id = row.get("rally_id")
        if rally_id is not None and str(rally_id).strip():
            if row.get("rally_length") is not None:
                by_rally[str(rally_id)] = max(by_rally[str(rally_id)], int(row["rally_length"]))
            else:
                by_rally[str(rally_id)] += 1
    if by_rally:
        return list(by_rally.values())
    explicit = [int(row["rally_length"]) for row in events if row.get("rally_length") is not None]
    if explicit:
        return explicit
    return []


def human_terminal_counts(events: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not events:
        return counts
    by_rally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        rally_id = row.get("rally_id")
        if rally_id is None or not str(rally_id).strip():
            reason = str(row.get("terminal_reason") or "").strip()
            if reason:
                counts[_terminal_bucket(reason)] += 1
            continue
        by_rally[str(rally_id)].append(row)
    for rows in by_rally.values():
        terminal_rows = [row for row in rows if str(row.get("terminal_reason") or "").strip()]
        if terminal_rows:
            counts[_terminal_bucket(terminal_rows[-1].get("terminal_reason"))] += 1
        else:
            rows_sorted = sorted(rows, key=lambda row: row.get("stage_index") if row.get("stage_index") is not None else -1)
            counts[_terminal_bucket(rows_sorted[-1].get("terminal_reason"))] += 1
    return counts


def _hist_percent(values: list[int], bins: np.ndarray) -> np.ndarray:
    if not values:
        return np.zeros(len(bins) - 1)
    hist, _ = np.histogram(values, bins=bins)
    return hist.astype(float) / max(float(hist.sum()), 1.0) * 100.0


def _draw_panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", pad=4)


def _draw_court(ax: plt.Axes, config: SimulationConfig) -> None:
    ax.add_patch(
        Rectangle(
            (-config.court.half_width, -config.court.half_length),
            config.court.width,
            config.court.length,
            fill=False,
            edgecolor=INK,
            linewidth=0.8,
        )
    )
    ax.axhline(0.0, color=INK, linewidth=0.7)
    ax.axvline(0.0, color=GRID, linewidth=0.5)
    ax.set_xlim(-config.court.half_width, config.court.half_width)
    ax.set_ylim(-config.court.half_length, config.court.half_length)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _events_for_zone(events: list[dict[str, Any]], zone: str) -> list[dict[str, Any]]:
    return [row for row in events if str(row.get("contact_zone") or "").lower() == zone]


def _non_service_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in events if not _is_service(str(row.get("shot_type") or ""))]


def _xy_arrays(events: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in events:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


def _xy_weight_arrays(events: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    xs: list[float] = []
    ys: list[float] = []
    weights: list[float] = []
    has_weight = False
    for row in events:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        weight = _float_or_none(row.get("plot_weight"))
        if weight is None or weight <= 0.0:
            weights.append(1.0)
        else:
            weights.append(float(weight))
            has_weight = True
    weight_array = np.asarray(weights, dtype=float) if has_weight else None
    return np.asarray(xs), np.asarray(ys), weight_array


def plot_rally_lengths(
    ax: plt.Axes,
    model_lengths: list[int],
    human_lengths: list[int],
    *,
    model_label: str,
    human_label: str,
) -> dict[str, Any]:
    max_value = max(model_lengths + human_lengths + [10])
    max_bin = min(max(max_value, 10), 60)
    bins = np.arange(1, max_bin + 2)
    centers = bins[:-1] + 0.5
    model_hist = _hist_percent(model_lengths, bins)
    ax.step(centers, model_hist, where="mid", color=MODEL, linewidth=1.8, label=f"{model_label} (n={len(model_lengths)})")
    if human_lengths:
        human_hist = _hist_percent(human_lengths, bins)
        ax.step(centers, human_hist, where="mid", color=HUMAN, linewidth=1.8, label=f"{human_label} (n={len(human_lengths)})")
    else:
        human_hist = np.zeros_like(model_hist)
    ax.set_xlabel("Shots per rally")
    ax.set_ylabel("Rallies (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    _draw_panel_label(ax, "A", "Rally length")
    return {"bins": bins.astype(int).tolist(), "model_percent": model_hist.tolist(), "human_percent": human_hist.tolist()}


def plot_terminal_reasons(
    ax: plt.Axes,
    model_counts: Counter[str],
    human_counts: Counter[str],
    *,
    model_label: str,
    human_label: str,
    model_invalid_action_rate: float | None = None,
) -> dict[str, Any]:
    labels = [label for label in TERMINAL_ORDER if model_counts.get(label, 0) or human_counts.get(label, 0)]
    if not labels:
        labels = list(TERMINAL_ORDER[:4])
    x = np.arange(len(labels))
    width = 0.36 if human_counts else 0.55
    model_total = max(sum(model_counts.values()), 1)
    model_values = np.asarray([100.0 * model_counts.get(label, 0) / model_total for label in labels])
    ax.bar(x - (width / 2 if human_counts else 0.0), model_values, width=width, color=MODEL, label=model_label)
    human_values = np.zeros_like(model_values)
    if human_counts:
        human_total = max(sum(human_counts.values()), 1)
        human_values = np.asarray([100.0 * human_counts.get(label, 0) / human_total for label in labels])
        ax.bar(x + width / 2, human_values, width=width, color=HUMAN, label=human_label)
    invalid_action_percent = None
    if model_invalid_action_rate is not None and "illegal_shot" in labels:
        invalid_action_percent = 100.0 * float(model_invalid_action_rate)
        illegal_index = labels.index("illegal_shot")
        ax.scatter(
            [x[illegal_index] - width / 2],
            [invalid_action_percent],
            marker="D",
            s=34,
            facecolors="white",
            edgecolors=MODEL,
            linewidths=1.5,
            label=f"{model_label} invalid actions",
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TERMINAL_DISPLAY.get(label, label.replace("_", "\n")) for label in labels], rotation=0)
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    _draw_panel_label(ax, "B", "Point endings")
    return {
        "labels": labels,
        "model_percent": model_values.tolist(),
        "human_percent": human_values.tolist(),
        "model_invalid_action_percent": invalid_action_percent,
    }


def plot_recovery_by_shot_family(
    ax: plt.Axes,
    model_events: list[dict[str, Any]],
    human_events: list[dict[str, Any]],
    *,
    model_label: str,
    human_label: str,
) -> dict[str, Any]:
    model_events = _non_service_events(model_events)
    human_events = _non_service_events(human_events)
    summary: dict[str, dict[str, float | int | None]] = {}

    def stats(ys: list[float]) -> dict[str, float | int | None]:
        if not ys:
            return {"n": 0, "mean": None, "sd": None, "sem": None}
        values = np.asarray(ys, dtype=float)
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        return {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "sd": sd,
            "sem": float(sd / math.sqrt(values.size)) if values.size > 0 else None,
        }

    model_means: list[float] = []
    model_sds: list[float] = []
    human_means: list[float] = []
    human_sds: list[float] = []

    for index, family in enumerate(SHOT_FAMILIES):
        model_y = [
            float(row["recovery_y"])
            for row in model_events
            if row.get("recovery_y") is not None and _canonical_shot_family(str(row.get("shot_type") or "")) == family
        ]
        human_y = [
            float(row["recovery_y"])
            for row in human_events
            if row.get("recovery_y") is not None and _canonical_shot_family(str(row.get("shot_type") or "")) == family
        ]
        model_stats = stats(model_y)
        human_stats = stats(human_y)
        model_means.append(float(model_stats["mean"]) if model_stats["mean"] is not None else np.nan)
        model_sds.append(float(model_stats["sd"]) if model_stats["sd"] is not None else 0.0)
        human_means.append(float(human_stats["mean"]) if human_stats["mean"] is not None else np.nan)
        human_sds.append(float(human_stats["sd"]) if human_stats["sd"] is not None else 0.0)
        summary[family] = {
            "model_n": int(model_stats["n"]),
            "model_mean_y": model_stats["mean"],
            "model_sd_y": model_stats["sd"],
            "model_sem_y": model_stats["sem"],
            "model_unique_y": len(set(round(float(value), 6) for value in model_y)),
            "model_y_counts": {
                f"{key:g}": value for key, value in sorted(Counter(round(float(y), 6) for y in model_y).items())
            },
            "human_n": int(human_stats["n"]),
            "human_mean_y": human_stats["mean"],
            "human_sd_y": human_stats["sd"],
            "human_sem_y": human_stats["sem"],
            "human_unique_y": len(set(round(float(value), 6) for value in human_y)),
            "human_y_counts": {
                f"{key:g}": value for key, value in sorted(Counter(round(float(y), 6) for y in human_y).items())
            },
        }
    x = np.arange(len(SHOT_FAMILIES))
    offset = 0.14 if human_events else 0.0
    ax.errorbar(
        x - offset,
        model_means,
        yerr=model_sds,
        color=MODEL,
        marker="o",
        markersize=4.5,
        capsize=3.0,
        linewidth=1.4,
        linestyle="none",
        label=f"{model_label} mean +/- SD",
    )
    if human_events:
        ax.errorbar(
            x + offset,
            human_means,
            yerr=human_sds,
            color=HUMAN,
            marker="s",
            markersize=4.5,
            capsize=3.0,
            linewidth=1.4,
            linestyle="none",
            label=f"{human_label} mean +/- SD",
        )
    ax.axhline(0.0, color=GRID, linewidth=0.8)
    ax.set_xticks(np.arange(len(SHOT_FAMILIES)))
    ax.set_xticklabels([label.replace("/", "/\n") for label in SHOT_FAMILIES])
    ax.set_ylabel("Recovery y (m)")
    ax.set_xlim(-0.5, len(SHOT_FAMILIES) - 0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    _draw_panel_label(ax, "B", "Recovery by shot")
    return summary


def plot_landing_heatmaps(
    spec: Any,
    fig: plt.Figure,
    model_events: list[dict[str, Any]],
    human_events: list[dict[str, Any]],
    config: SimulationConfig,
) -> dict[str, Any]:
    model_events = _non_service_events(model_events)
    human_events = _non_service_events(human_events)
    axes: list[plt.Axes] = []
    if isinstance(spec, tuple) and len(spec) == 4:
        panel_left, panel_bottom, panel_width, panel_height = (float(value) for value in spec)
        fig_width, fig_height = fig.get_size_inches()
        gap = 0.012
        court_width = panel_height * fig_height * (config.court.width / config.court.length) / fig_width
        total_width = 3.0 * court_width + 2.0 * gap
        if total_width > panel_width:
            court_width = (panel_width - 2.0 * gap) / 3.0
            court_height = court_width * fig_width * (config.court.length / config.court.width) / fig_height
        else:
            court_height = panel_height
        x0 = panel_left + max(panel_width - (3.0 * court_width + 2.0 * gap), 0.0) / 2.0
        y0 = panel_bottom + max(panel_height - court_height, 0.0) / 2.0
        axes = [fig.add_axes([x0 + index * (court_width + gap), y0, court_width, court_height]) for index in range(3)]
    else:
        inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.06)
        axes = [fig.add_subplot(inner[0, index]) for index in range(3)]
    zones = ("frontcourt", "midcourt", "backcourt")
    summary: dict[str, Any] = {}
    rng = np.random.default_rng(20260617)
    for index, zone in enumerate(zones):
        ax = axes[index]
        model_zone = _events_for_zone(model_events, zone)
        xs, ys, weights = _xy_weight_arrays(model_zone, "landing_x", "landing_y")
        plotted_model_n = int(xs.size)
        if xs.size:
            if xs.size > 1500:
                probabilities = None
                if weights is not None and float(np.sum(weights)) > 0.0:
                    probabilities = weights / float(np.sum(weights))
                selected = rng.choice(xs.size, size=1500, replace=False, p=probabilities)
                xs = xs[selected]
                ys = ys[selected]
            plotted_model_n = int(xs.size)
            ax.scatter(xs, ys, s=10, color=MODEL, alpha=0.55, linewidths=0.0, label="model")
        human_zone = _events_for_zone(human_events, zone)
        hx, hy = _xy_arrays(human_zone, "landing_x", "landing_y")
        human_n = int(hx.size)
        plotted_human_n = human_n
        if hx.size:
            if hx.size > 2500:
                selected = rng.choice(hx.size, size=2500, replace=False)
                hx = hx[selected]
                hy = hy[selected]
            plotted_human_n = int(hx.size)
            ax.scatter(hx, hy, s=5, color=HUMAN, alpha=0.22, linewidths=0.0, label="human")
        _draw_court(ax, config)
        ax.set_title(zone, fontsize=8, pad=2)
        if index == 0:
            ax.text(-0.26, 1.09, "C", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")
            ax.text(0.00, 1.09, "Landings by contact zone", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
            if xs.size or hx.size:
                ax.legend(frameon=False, loc="lower left", fontsize=6)
        summary[zone] = {
            "model_n": int(_xy_arrays(model_zone, "landing_x", "landing_y")[0].size),
            "human_n": human_n,
            "model_plotted_n": plotted_model_n,
            "human_plotted_n": plotted_human_n,
        }
    return summary


def make_summary_table(
    model_lengths: list[int],
    human_lengths: list[int],
    model_events: list[dict[str, Any]],
    human_events: list[dict[str, Any]],
    model_terminal: Counter[str],
    human_terminal: Counter[str],
) -> dict[str, Any]:
    def mean_or_none(values: list[float | int]) -> float | None:
        return float(np.mean(values)) if values else None

    model_landings = [row for row in model_events if row.get("landing_x") is not None and row.get("landing_y") is not None]
    human_landings = [row for row in human_events if row.get("landing_x") is not None and row.get("landing_y") is not None]
    return {
        "model": {
            "rally_count": len(model_lengths),
            "mean_rally_length": mean_or_none(model_lengths),
            "event_count": len(model_events),
            "landing_count": len(model_landings),
            "terminal_counts": dict(model_terminal),
        },
        "human": {
            "rally_count": len(human_lengths),
            "mean_rally_length": mean_or_none(human_lengths),
            "event_count": len(human_events),
            "landing_count": len(human_landings),
            "terminal_counts": dict(human_terminal),
        },
    }


def _checkpoint_step_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    steps = sorted({step for row in events if (step := _int_or_none(row.get("checkpoint_step"))) is not None})
    return {
        "count": len(steps),
        "min": steps[0] if steps else None,
        "max": steps[-1] if steps else None,
        "steps": steps,
    }


def _selected_run_dirs(args: argparse.Namespace) -> list[Path]:
    return [path.resolve() for path in (args.run_dir or list(DEFAULT_RUN_DIRS))]


def _selected_extra_trace_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if not args.no_default_extra_trace:
        paths.extend(DEFAULT_EXTRA_TRACE_PATHS)
    if args.extra_trace_path:
        paths.extend(args.extra_trace_path)
    return [path.resolve() for path in paths if path.exists()]


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    config = SimulationConfig()
    run_dirs = _selected_run_dirs(args)

    model_probe_events = model_event_rows(run_dirs, config)
    panel_c_model_events, panel_c_model_source = panel_c_model_event_rows(
        run_dirs,
        config,
        preferred_probe_name=args.panel_c_model_probe_name,
        samples_path=args.panel_c_model_samples,
        fallback_events=model_probe_events,
    )
    model_trace_paths = sorted(_trace_paths(run_dirs, args.trace_glob) + _selected_extra_trace_paths(args))
    model_lengths, model_terminal = model_match_rows_from_trace_paths(model_trace_paths)
    model_ab_source = f"combined_match_traces_{len(run_dirs)}seed_{len(model_trace_paths)}traces"
    model_trace_events = model_trace_event_rows_from_trace_paths(model_trace_paths, config)
    model_cd_events = model_probe_events if model_probe_events else model_trace_events
    model_cd_source = (
        f"controlled_contact_probe_all_valid_checkpoints_{len(run_dirs)}seed"
        if model_probe_events
        else "match_trace"
    )
    human_events = load_human_events(args.human_events, config, args.coordinate_mode)
    human_lengths = human_rally_lengths(human_events)
    human_terminal = human_terminal_counts(human_events)

    fig = plt.figure(figsize=(10.4, 3.4), constrained_layout=False)
    panel_bottom = 0.16
    panel_height = 0.68
    panel_gap = 0.055
    panel_a = (0.060, panel_bottom, 0.245, panel_height)
    panel_b = (panel_a[0] + panel_a[2] + panel_gap, panel_bottom, 0.245, panel_height)
    panel_c = (panel_b[0] + panel_b[2] + panel_gap, panel_bottom, 0.350, panel_height)

    ax_a = fig.add_axes(panel_a)
    ax_b = fig.add_axes(panel_b)

    panel_data = {
        "rally_length": plot_rally_lengths(
            ax_a,
            model_lengths,
            human_lengths,
            model_label=args.model_label,
            human_label=args.human_label,
        ),
        "recovery_by_shot_family": plot_recovery_by_shot_family(
            ax_b,
            model_cd_events,
            human_events,
            model_label=args.model_label,
            human_label=args.human_label,
        ),
        "landing_heatmaps": plot_landing_heatmaps(panel_c, fig, panel_c_model_events, human_events, config),
    }

    step_summary = _checkpoint_step_summary(model_cd_events)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    pdf_path = args.out.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")

    summary = make_summary_table(model_lengths, human_lengths, model_cd_events, human_events, model_terminal, human_terminal)
    summary["panels"] = panel_data
    summary["inputs"] = {
        "run_dirs": [str(path) for path in run_dirs],
        "trace_glob": args.trace_glob,
        "model_ab_event_source": model_ab_source,
        "model_ab_trace_count": len(model_trace_paths),
        "model_ab_traces": [str(path) for path in model_trace_paths],
        "default_extra_trace_paths": [str(path.resolve()) for path in DEFAULT_EXTRA_TRACE_PATHS if path.exists()],
        "human_events": None if args.human_events is None else str(args.human_events),
        "coordinate_mode": args.coordinate_mode,
        "model_cd_event_source": model_cd_source,
        "panel_c_model_event_source": panel_c_model_source,
        "panel_c_model_samples": None if args.panel_c_model_samples is None else str(args.panel_c_model_samples),
        "panel_c_model_probe_name": args.panel_c_model_probe_name,
        "panel_c_model_events": len(panel_c_model_events),
        "model_cd_checkpoint_steps": step_summary,
        "model_trace_events": len(model_trace_events),
        "model_probe_events": len(model_probe_events),
    }
    args.source_data_out.parent.mkdir(parents=True, exist_ok=True)
    args.source_data_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {pdf_path}")
    print(f"wrote {args.source_data_out}")


if __name__ == "__main__":
    main()
