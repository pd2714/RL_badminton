from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from badminton1d.config import SimulationConfig
from badminton1d.playback import RallyTrace, StageTrace, interpolate_stage, match_trace_from_dict, rally_trace_from_dict
from badminton1d.render import GROUND_MARKER_Z, OFFICIAL_DOUBLES_WIDTH, draw_players_3d, setup_3d_court_axes, stage_colors
from badminton1d.utils import ensure_directory
from badminton1d.video import _marker_size, _trajectory_segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 1xN static rally exhibition figure. Each panel samples one "
            "rally stage and overlays an arrow showing the shuttle flight direction."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to match.mp4/match.gif, match_trace.json, rally_trace.json, or a directory containing one of those traces.",
    )
    parser.add_argument(
        "--rallies",
        type=int,
        nargs="+",
        default=None,
        help="Rally numbers to export from a match trace. Uses trace rally_number when present, otherwise 1-based order.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-fraction", type=float, default=0.5, help="Representative time within each stage, from 0 to 1.")
    parser.add_argument("--panel-width", type=float, default=1.55)
    parser.add_argument("--panel-height", type=float, default=2.2)
    parser.add_argument("--columns", type=int, default=None, help="Number of panel columns. Defaults to one row.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--monochrome", action="store_true")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    return parser.parse_args()


def _resolve_trace_path(input_path: Path) -> Path:
    if input_path.is_dir():
        for filename in ("match_trace.json", "rally_trace.json"):
            candidate = input_path / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No match_trace.json or rally_trace.json found in {input_path}")

    if input_path.suffix.lower() == ".json":
        return input_path

    if input_path.suffix.lower() in {".mp4", ".gif", ".mov", ".avi"}:
        for filename in ("match_trace.json", "rally_trace.json"):
            candidate = input_path.with_name(filename)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No adjacent match_trace.json or rally_trace.json found next to {input_path}")

    raise ValueError(f"Unsupported input path: {input_path}")


def _load_rallies(input_path: Path) -> tuple[list[RallyTrace], Path]:
    trace_path = _resolve_trace_path(input_path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if "rallies" in payload:
        return match_trace_from_dict(payload).rallies, trace_path
    if "stages" in payload:
        return [rally_trace_from_dict(payload)], trace_path
    raise ValueError(f"{trace_path} is not a match or rally trace")


def _select_rally(rallies: list[RallyTrace], requested_number: int) -> RallyTrace:
    for rally in rallies:
        if rally.rally_number == requested_number:
            return rally
    if 1 <= requested_number <= len(rallies):
        return rallies[requested_number - 1]
    available = [r.rally_number for r in rallies if r.rally_number is not None]
    raise ValueError(f"Rally {requested_number} not found. Available rally numbers: {available or list(range(1, len(rallies) + 1))}")


def _sample_time(stage: StageTrace, fraction: float) -> float:
    fraction = min(max(float(fraction), 0.0), 1.0)
    return float(stage.playback_duration) * fraction


def _flight_arrow_points(
    stage: StageTrace,
    local_time: float,
    config: SimulationConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    duration = max(float(stage.playback_duration), 0.0)
    if duration <= 1e-9:
        start = np.asarray(stage.shuttle_start, dtype=float)
        end = np.asarray((stage.shuttle_landing[0], stage.shuttle_landing[1], GROUND_MARKER_Z), dtype=float)
    else:
        step = min(max(duration * 0.14, 0.04), 0.22)
        before_t = max(0.0, local_time - step * 0.5)
        after_t = min(duration, local_time + step * 0.5)
        if after_t - before_t < 1e-6:
            before_t = max(0.0, local_time - step)
            after_t = min(duration, local_time + step)
        before = interpolate_stage(stage, before_t, config=config).shuttle_position
        after = interpolate_stage(stage, after_t, config=config).shuttle_position
        start = np.asarray(before, dtype=float)
        end = np.asarray(after, dtype=float)

    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        vector = np.asarray((stage.shuttle_landing[0], stage.shuttle_landing[1], GROUND_MARKER_Z), dtype=float) - np.asarray(
            stage.shuttle_start,
            dtype=float,
        )
        length = float(np.linalg.norm(vector))
    if length < 1e-6:
        vector = np.asarray([0.0, 1.0, 0.0], dtype=float)
        length = 1.0

    center = np.asarray(interpolate_stage(stage, local_time, config=config).shuttle_position, dtype=float)
    direction = vector / length
    arrow_length = min(max(length, 0.65), 1.25)
    arrow_start = center - direction * arrow_length * 0.35
    arrow_end = center + direction * arrow_length * 0.65
    return (
        (float(arrow_start[0]), float(arrow_start[1]), float(arrow_start[2])),
        (float(arrow_end[0]), float(arrow_end[1]), float(arrow_end[2])),
    )


def _apply_physical_3d_view(ax: plt.Axes, config: SimulationConfig) -> None:
    ax.patch.set_alpha(0.0)
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    display_half_width = max(config.court.half_width, OFFICIAL_DOUBLES_WIDTH / 2.0)
    pad = config.render.court_padding * 0.12
    x_min = -display_half_width - pad
    x_max = display_half_width + pad
    y_min = -config.court.half_length - pad
    y_max = config.court.half_length + pad
    z_min = -0.04
    z_max = min(config.render.z_max, 4.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=21.0, azim=-58.0)
    if hasattr(ax, "dist"):
        ax.dist = 5.7
    try:
        ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min), zoom=1.34)
    except TypeError:
        ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))


def _draw_stage_panel(
    ax: plt.Axes,
    rally: RallyTrace,
    stage: StageTrace,
    *,
    config: SimulationConfig,
    sample_fraction: float,
    monochrome: bool,
) -> None:
    colors = stage_colors(monochrome)
    sample_time = _sample_time(stage, sample_fraction)
    snapshot = interpolate_stage(stage, sample_time, config=config)
    intended_point = stage.intended_intercept_point or stage.intercept_point
    chosen_intercept_point = stage.intercept_point or stage.intended_intercept_point

    setup_3d_court_axes(ax, config, colors, show_axes=False)
    _apply_physical_3d_view(ax, config)

    left_contact = stage.shuttle_start if stage.hitter_side == "left" else None
    right_contact = stage.shuttle_start if stage.hitter_side == "right" else None
    if chosen_intercept_point is not None:
        if stage.receiver_side == "left":
            left_contact = chosen_intercept_point
        else:
            right_contact = chosen_intercept_point
    draw_players_3d(
        ax,
        config,
        colors,
        left_position=snapshot.left_player_position,
        right_position=snapshot.right_player_position,
        left_contact_xyz=left_contact,
        right_contact_xyz=right_contact,
        show_player_labels=False,
    )

    traj_xs, traj_ys, traj_zs = _trajectory_segment(stage, config)
    ax.plot(traj_xs, traj_ys, traj_zs, linestyle="--", color=colors["trajectory"], linewidth=1.15, alpha=0.46, zorder=3)
    ax.scatter(
        [stage.recovery_target[0]],
        [stage.recovery_target[1]],
        [GROUND_MARKER_Z],
        color=colors["recovery"],
        s=24,
        alpha=0.28,
        zorder=4,
    )
    ax.scatter(
        [stage.shuttle_landing[0]],
        [stage.shuttle_landing[1]],
        [GROUND_MARKER_Z],
        color=colors["target"],
        s=24,
        alpha=0.78,
        zorder=4,
    )
    if intended_point is not None:
        ax.scatter(
            [intended_point[0]],
            [intended_point[1]],
            [intended_point[2]],
            color=colors["intercept"],
            s=_marker_size(float(intended_point[2]), config) * 0.32,
            alpha=0.82,
            zorder=5,
        )

    arrow_start, arrow_end = _flight_arrow_points(stage, sample_time, config)
    arrow_delta = tuple(float(end - start) for start, end in zip(arrow_start, arrow_end))
    ax.plot(
        [arrow_start[0], arrow_end[0]],
        [arrow_start[1], arrow_end[1]],
        [arrow_start[2], arrow_end[2]],
        color="#ff0000" if not monochrome else "black",
        linewidth=2.8,
        solid_capstyle="round",
        zorder=7,
    )
    ax.quiver(
        arrow_start[0],
        arrow_start[1],
        arrow_start[2],
        arrow_delta[0],
        arrow_delta[1],
        arrow_delta[2],
        color="#ff0000" if not monochrome else "black",
        linewidth=2.6,
        arrow_length_ratio=0.36,
        zorder=7,
    )
    ax.scatter(
        [snapshot.shuttle_position[0]],
        [snapshot.shuttle_position[1]],
        [snapshot.shuttle_position[2]],
        color="#ff0000" if not monochrome else "black",
        edgecolor="white" if not monochrome else "0.35",
        linewidth=0.9,
        s=_marker_size(float(snapshot.shuttle_position[2]), config) * 0.34,
        zorder=8,
    )

    title = f"stage {stage.stage_index + 1}"
    if stage.terminal and rally.winner is not None:
        title += f" | {rally.winner} wins"
    ax.text2D(0.5, 0.62, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)


def export_rally_exhibition(
    rally: RallyTrace,
    output_stem: Path,
    *,
    sample_fraction: float,
    panel_width: float,
    panel_height: float,
    columns: int | None,
    dpi: int,
    monochrome: bool,
    formats: list[str],
) -> list[Path]:
    if not rally.stages:
        raise ValueError("rally trace has no stages")

    config = SimulationConfig()
    stage_count = len(rally.stages)
    column_count = stage_count if columns is None else min(max(columns, 1), stage_count)
    row_count = int(np.ceil(stage_count / column_count))
    fig = plt.figure(figsize=(panel_width * column_count, panel_height * row_count), dpi=dpi)
    axes = [fig.add_subplot(row_count, column_count, index + 1, projection="3d") for index in range(stage_count)]
    for ax, stage in zip(axes, rally.stages):
        _draw_stage_panel(
            ax,
            rally,
            stage,
            config=config,
            sample_fraction=sample_fraction,
            monochrome=monochrome,
        )

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=-0.28, hspace=-0.28)

    written: list[Path] = []
    for suffix in formats:
        output_path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=dpi)
        written.append(output_path)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    if args.sample_fraction < 0.0 or args.sample_fraction > 1.0:
        raise ValueError("--sample-fraction must be between 0 and 1")
    if args.panel_width <= 0.0 or args.panel_height <= 0.0:
        raise ValueError("--panel-width and --panel-height must be positive")
    if args.columns is not None and args.columns <= 0:
        raise ValueError("--columns must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    rallies, trace_path = _load_rallies(args.input_path)
    output_dir = args.output_dir or (trace_path.parent / "rally_exhibition_figures")
    ensure_directory(output_dir)

    requested_rallies = args.rallies
    if requested_rallies is None:
        if len(rallies) != 1:
            raise ValueError("--rallies is required when the input is a match trace")
        requested_rallies = [rallies[0].rally_number or 1]

    for requested_number in requested_rallies:
        rally = _select_rally(rallies, requested_number)
        rally_label = rally.rally_number if rally.rally_number is not None else requested_number
        output_stem = output_dir / f"rally_{rally_label:02d}_static_exhibition"
        written = export_rally_exhibition(
            rally,
            output_stem,
            sample_fraction=args.sample_fraction,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
            columns=args.columns,
            dpi=args.dpi,
            monochrome=args.monochrome,
            formats=args.formats,
        )
        for path in written:
            print(path)


if __name__ == "__main__":
    main()
