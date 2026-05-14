from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from badminton1d.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import effective_flight_time, landing_position, sample_trajectory
from badminton1d.state import Side, StageRecord
from badminton1d.utils import ensure_directory

OFFICIAL_DOUBLES_WIDTH = 6.10
OFFICIAL_SINGLES_WIDTH = 5.18
OFFICIAL_SHORT_SERVICE_FROM_NET = 1.98
OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK = 0.76
COURT_SURFACE_Z = -0.03
COURT_LINE_Z = 0.03
GROUND_MARKER_Z = 0.04
PLAYER_FOOT_Z = 0.05


@dataclass(frozen=True)
class ScoreboardOverlay:
    score_left: int
    score_right: int
    current_server: Side
    rally_number: int
    stage_number: int
    hitter_side: Side
    shot_text: str | None = None
    flight_time_text: str | None = None
    intercept_text: str | None = None
    point_winner: Side | None = None
    match_winner: Side | None = None


def _side_label(side: Side | None) -> str:
    if side is None:
        return "-"
    return "Left" if side == "left" else "Right"


def _stage_title(record: StageRecord, config: SimulationConfig) -> str:
    action = record.validated_action.applied
    if record.terminal_reason == "invalid_serve_target":
        return (
            f"stage={record.stage_index} | hitter={record.state_before.current_hitter} | "
            f"illegal serve | winner={record.next_state.winner}"
        )
    if record.terminal_reason == "opponent_no_valid_shot":
        return (
            f"stage={record.stage_index} | hitter={record.state_before.current_hitter} | "
            f"opponent had no valid shot | winner={record.next_state.winner}"
        )
    total_flight_time = effective_flight_time(record.state_before, action, config)
    intercept_text = "None" if record.chosen_time is None else f"{record.chosen_time:.2f}s"
    landing_x, landing_y = landing_position(record.state_before, action, config)
    winner_text = ""
    if record.next_state.rally_done:
        winner_text = f" | winner={record.next_state.winner}"
    if config.court.lateral_motion_enabled:
        landing_text = f"land=({landing_x:.2f}, {landing_y:.2f})"
        velocity_text = f"vx={action.v_x:.2f} | vy={action.v_y:.2f}"
    else:
        landing_text = f"land_y={landing_y:.2f}"
        velocity_text = f"vy={action.v_y:.2f}"
    return (
        f"stage={record.stage_index} | hitter={record.state_before.current_hitter} | "
        f"{velocity_text} | vz={action.v_z:.2f} | "
        f"t_land={total_flight_time:.2f}s | {landing_text} | "
        f"t_int={intercept_text}{winner_text}"
    )


def stage_colors(monochrome: bool) -> dict[str, str]:
    if monochrome:
        return {
            "court_fill": "white",
            "court_line": "black",
            "service_line": "0.45",
            "net": "0.2",
            "left_player": "0.15",
            "right_player": "0.45",
            "left_label": "0.15",
            "right_label": "0.45",
            "player_arrow": "0.55",
            "trajectory": "0.4",
            "start": "black",
            "target": "0.6",
            "recovery": "0.8",
            "intercept": "0.5",
            "notes": "0.35",
        }
    return {
        "court_fill": "#15803d",
        "court_line": "#ffffff",
        "service_line": "#ffffff",
        "net": "#111827",
        "left_player": "#2563eb",
        "right_player": "#db2777",
        "left_label": "#1d4ed8",
        "right_label": "#be185d",
        "player_arrow": "#ea580c",
        "trajectory": "#dc2626",
        "start": "#dc2626",
        "target": "#f8fafc",
        "recovery": "#7c3aed",
        "intercept": "#fde047",
        "notes": "#111827",
    }


def draw_scoreboard_overlay(
    ax: plt.Axes,
    overlay: ScoreboardOverlay,
    colors: dict[str, str],
) -> None:
    lines = [
        f"score {overlay.score_left} - {overlay.score_right}",
        f"server {_side_label(overlay.current_server)}",
        f"rally {overlay.rally_number}",
        f"stage {overlay.stage_number}",
        f"hitter {_side_label(overlay.hitter_side)}",
    ]
    if overlay.shot_text is not None:
        lines.append(overlay.shot_text)
    if overlay.flight_time_text is not None:
        lines.append(overlay.flight_time_text)
    if overlay.intercept_text is not None:
        lines.append(overlay.intercept_text)
    if overlay.point_winner is not None:
        lines.append(f"point {_side_label(overlay.point_winner)}")
    if overlay.match_winner is not None:
        lines.append(f"match {_side_label(overlay.match_winner)}")

    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_fn(
        0.14,
        0.91,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=colors["notes"],
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.8},
    )


def setup_court_axes(
    ax: plt.Axes,
    config: SimulationConfig,
    colors: dict[str, str],
    *,
    show_axes: bool,
) -> None:
    if config.court.lateral_motion_enabled:
        display_half_width = max(config.court.half_width, OFFICIAL_DOUBLES_WIDTH / 2.0)
        x_min = -display_half_width - config.render.court_padding
        x_max = display_half_width + config.render.court_padding
        court_x = -display_half_width
        court_width = 2.0 * display_half_width
    else:
        lane_half_width = max(config.player.marker_radius * 2.2, 0.55)
        x_center = float(config.court.default_player_start_x)
        x_min = x_center - lane_half_width - config.render.court_padding
        x_max = x_center + lane_half_width + config.render.court_padding
        court_x = x_center - lane_half_width
        court_width = 2.0 * lane_half_width
    y_min = -config.court.half_length - config.render.court_padding
    y_max = config.court.half_length + config.render.court_padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    court = Rectangle(
        (court_x, -config.court.half_length),
        court_width,
        config.court.length,
        facecolor=colors["court_fill"],
        edgecolor=colors["court_line"],
        linewidth=2.0,
        zorder=0,
    )
    ax.add_patch(court)
    ax.plot(
        [court_x, court_x + court_width],
        [config.court.net_y, config.court.net_y],
        color=colors["net"],
        linewidth=2.4,
        zorder=2,
    )
    if config.court.lateral_motion_enabled:
        _draw_full_court_markings_top_view(ax, config, colors, display_half_width=display_half_width)
    else:
        _draw_service_markers_top_view(ax, config, colors, court_x=court_x, court_width=court_width)

    if show_axes:
        if config.court.lateral_motion_enabled:
            ax.set_xlabel("x across court (m)")
        else:
            ax.set_xlabel("1D lane")
        ax.set_ylabel("y along court (m)")
        ax.grid(alpha=0.18)
    else:
        ax.set_axis_off()


def _draw_full_court_markings_top_view(
    ax: plt.Axes,
    config: SimulationConfig,
    colors: dict[str, str],
    *,
    display_half_width: float,
) -> None:
    singles_half_width = OFFICIAL_SINGLES_WIDTH / 2.0
    short_service_y = OFFICIAL_SHORT_SERVICE_FROM_NET
    long_service_doubles_y = config.court.half_length - OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK
    line_kwargs = {"color": colors["court_line"], "linewidth": 2.0, "zorder": 1}

    for x_pos in (-singles_half_width, singles_half_width):
        ax.plot([x_pos, x_pos], [-config.court.half_length, config.court.half_length], **line_kwargs)
    for y_pos in (-short_service_y, short_service_y):
        ax.plot([-display_half_width, display_half_width], [y_pos, y_pos], **line_kwargs)
    for y_pos in (-long_service_doubles_y, long_service_doubles_y):
        ax.plot([-display_half_width, display_half_width], [y_pos, y_pos], **line_kwargs)
    ax.plot(
        [0.0, 0.0],
        [-config.court.half_length, -short_service_y],
        color=colors["service_line"],
        linewidth=1.8,
        zorder=1,
    )
    ax.plot(
        [0.0, 0.0],
        [short_service_y, config.court.half_length],
        color=colors["service_line"],
        linewidth=1.8,
        zorder=1,
    )


def setup_3d_court_axes(
    ax: plt.Axes,
    config: SimulationConfig,
    colors: dict[str, str],
    *,
    show_axes: bool,
) -> None:
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    pad = config.render.court_padding * 0.06
    display_half_width = max(config.court.half_width, OFFICIAL_DOUBLES_WIDTH / 2.0)
    cropped_half_width = display_half_width * 0.18
    x_min = -cropped_half_width - pad
    x_max = cropped_half_width + pad
    y_min = -config.court.half_length - pad * 0.18
    y_max = config.court.half_length + pad * 0.18
    z_min = COURT_SURFACE_Z - 0.01
    z_max = config.render.z_max * 0.66

    court_surface = Poly3DCollection(
        [[
            (-display_half_width, -config.court.half_length, COURT_SURFACE_Z),
            (display_half_width, -config.court.half_length, COURT_SURFACE_Z),
            (display_half_width, config.court.half_length, COURT_SURFACE_Z),
            (-display_half_width, config.court.half_length, COURT_SURFACE_Z),
        ]],
        facecolors=colors["court_fill"],
        edgecolors="none",
        alpha=0.9,
        zorder=0,
    )
    court_surface.set_zsort("min")
    court_surface.set_sort_zpos(COURT_SURFACE_Z - 1.0)
    ax.add_collection3d(court_surface)

    corners = [
        (-display_half_width, -config.court.half_length),
        (display_half_width, -config.court.half_length),
        (display_half_width, config.court.half_length),
        (-display_half_width, config.court.half_length),
        (-display_half_width, -config.court.half_length),
    ]
    ax.plot(
        [point[0] for point in corners],
        [point[1] for point in corners],
        [COURT_LINE_Z for _ in corners],
        color=colors["court_line"],
        linewidth=1.8,
        zorder=1,
    )
    ax.plot(
        [-display_half_width, display_half_width],
        [config.court.net_y, config.court.net_y],
        [config.court.net_height, config.court.net_height],
        color=colors["net"],
        linewidth=2.4,
        zorder=3,
    )
    for post_x in (-display_half_width, display_half_width):
        ax.plot(
            [post_x, post_x],
            [config.court.net_y, config.court.net_y],
            [PLAYER_FOOT_Z, config.court.net_height],
            color=colors["net"],
            linewidth=1.4,
            zorder=2,
        )
    _draw_full_court_markings_3d(ax, config, colors, display_half_width=display_half_width)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
    ax.view_init(elev=10.0, azim=-69.0)
    if hasattr(ax, "dist"):
        ax.dist = 6.2
    ax.grid(False)

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

    if show_axes:
        ax.set_xlabel("x across court (m)")
        ax.set_ylabel("y along court (m)")
        ax.set_zlabel("z height (m)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()


def _draw_full_court_markings_3d(
    ax: plt.Axes,
    config: SimulationConfig,
    colors: dict[str, str],
    *,
    display_half_width: float,
) -> None:
    singles_half_width = OFFICIAL_SINGLES_WIDTH / 2.0
    short_service_y = OFFICIAL_SHORT_SERVICE_FROM_NET
    long_service_doubles_y = config.court.half_length - OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK
    line_z = COURT_LINE_Z
    line_kwargs = {"color": colors["court_line"], "linewidth": 2.0, "zorder": 1}

    for x_pos in (-singles_half_width, singles_half_width):
        ax.plot([x_pos, x_pos], [-config.court.half_length, config.court.half_length], [line_z, line_z], **line_kwargs)
    for y_pos in (-short_service_y, short_service_y):
        ax.plot([-display_half_width, display_half_width], [y_pos, y_pos], [line_z, line_z], **line_kwargs)
    for y_pos in (-long_service_doubles_y, long_service_doubles_y):
        ax.plot([-display_half_width, display_half_width], [y_pos, y_pos], [line_z, line_z], **line_kwargs)
    ax.plot(
        [0.0, 0.0],
        [-config.court.half_length, -short_service_y],
        [line_z, line_z],
        color=colors["service_line"],
        linewidth=1.8,
        zorder=1,
    )
    ax.plot(
        [0.0, 0.0],
        [short_service_y, config.court.half_length],
        [line_z, line_z],
        color=colors["service_line"],
        linewidth=1.8,
        zorder=1,
    )


def draw_players_3d(
    ax: plt.Axes,
    config: SimulationConfig,
    colors: dict[str, str],
    *,
    left_position: tuple[float, float],
    right_position: tuple[float, float],
    left_contact_xyz: tuple[float, float, float] | None = None,
    right_contact_xyz: tuple[float, float, float] | None = None,
    show_player_labels: bool,
) -> None:
    body_height = min(config.player.z_max - config.player.r_reach, 1.7)
    body_height = max(body_height, config.court.net_height * 0.9)
    shoulder_height = max(body_height - 0.08, 0.0)
    racket_length = float(config.player.r_reach)

    players = [
        ("L", left_position, colors["left_player"], colors["left_label"], left_contact_xyz),
        ("R", right_position, colors["right_player"], colors["right_label"], right_contact_xyz),
    ]
    for label, (x_pos, y_pos), color, label_color, contact_xyz in players:
        ax.plot(
            [x_pos, x_pos],
            [y_pos, y_pos],
            [PLAYER_FOOT_Z, PLAYER_FOOT_Z + body_height],
            color=color,
            linewidth=5.0,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter([x_pos], [y_pos], [PLAYER_FOOT_Z + body_height], color=color, s=28, zorder=5)
        if show_player_labels:
            ax.text(x_pos, y_pos, PLAYER_FOOT_Z + body_height + 0.14, label, color=label_color, ha="center", va="bottom", fontsize=10, weight="bold")

        if contact_xyz is None:
            continue
        dx = float(contact_xyz[0] - x_pos)
        dy = float(contact_xyz[1] - y_pos)
        dz = float(contact_xyz[2] - (PLAYER_FOOT_Z + shoulder_height))
        distance = float(np.linalg.norm([dx, dy, dz]))
        if distance <= 1e-6:
            continue
        scale = min(1.0, racket_length / distance)
        ax.quiver(
            x_pos,
            y_pos,
            PLAYER_FOOT_Z + shoulder_height,
            dx * scale,
            dy * scale,
            dz * scale,
            color=colors["player_arrow"],
            linewidth=2.0,
            arrow_length_ratio=0.28,
            zorder=6,
        )


def _draw_service_markers_top_view(
    ax: plt.Axes,
    config: SimulationConfig,
    colors: dict[str, str],
    *,
    court_x: float,
    court_width: float,
) -> None:
    x_center = court_x + court_width / 2.0
    marker_half_width = min(court_width * 0.16, 0.18)
    for service_y in (-config.court.service_line_distance_from_net, config.court.service_line_distance_from_net):
        if config.court.lateral_motion_enabled:
            x_start = court_x
            x_end = court_x + court_width
            linestyle = "--"
            linewidth = 1.1
        else:
            x_start = x_center - marker_half_width
            x_end = x_center + marker_half_width
            linestyle = "-"
            linewidth = 1.4
        ax.plot(
            [x_start, x_end],
            [service_y, service_y],
            color=colors["service_line"],
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=1,
        )


def _draw_contact_arrow(
    ax: plt.Axes,
    *,
    player_xy: tuple[float, float],
    contact_xy: tuple[float, float] | None,
    color: str,
    max_length: float | None = None,
) -> None:
    if contact_xy is None:
        return
    if np.allclose(player_xy, contact_xy):
        return
    end_xy = contact_xy
    if max_length is not None:
        start = np.asarray(player_xy, dtype=float)
        end = np.asarray(contact_xy, dtype=float)
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance > max(float(max_length), 0.0) > 0.0:
            end_xy = tuple((start + delta * (float(max_length) / distance)).tolist())
    arrow = FancyArrowPatch(
        player_xy,
        end_xy,
        arrowstyle="-|>",
        mutation_scale=12.0,
        linewidth=1.8,
        color=color,
        shrinkA=6.0,
        shrinkB=6.0,
        zorder=4,
    )
    ax.add_patch(arrow)


def draw_players(
    ax: plt.Axes,
    config: SimulationConfig,
    colors: dict[str, str],
    *,
    left_position: tuple[float, float],
    right_position: tuple[float, float],
    show_player_labels: bool,
    left_contact_xy: tuple[float, float] | None = None,
    right_contact_xy: tuple[float, float] | None = None,
) -> None:
    radius = config.player.marker_radius
    left_circle = Circle(left_position, radius=radius, facecolor=colors["left_player"], edgecolor="white", linewidth=1.0, zorder=3)
    right_circle = Circle(right_position, radius=radius, facecolor=colors["right_player"], edgecolor="white", linewidth=1.0, zorder=3)
    ax.add_patch(left_circle)
    ax.add_patch(right_circle)
    _draw_contact_arrow(
        ax,
        player_xy=left_position,
        contact_xy=left_contact_xy,
        color=colors["player_arrow"],
        max_length=config.player.r_reach,
    )
    _draw_contact_arrow(
        ax,
        player_xy=right_position,
        contact_xy=right_contact_xy,
        color=colors["player_arrow"],
        max_length=config.player.r_reach,
    )
    if show_player_labels:
        ax.text(left_position[0], left_position[1] + 0.35, "L", color=colors["left_label"], ha="center", fontsize=11, weight="bold")
        ax.text(right_position[0], right_position[1] + 0.35, "R", color=colors["right_label"], ha="center", fontsize=11, weight="bold")


def _shuttle_marker_size(z_value: float, config: SimulationConfig) -> float:
    ratio = np.clip(z_value / max(config.render.z_max, 1e-6), 0.0, 1.0)
    return float(config.render.shuttle_marker_min + ratio * (config.render.shuttle_marker_max - config.render.shuttle_marker_min))


def _draw_stage(
    record: StageRecord,
    config: SimulationConfig,
    ax: plt.Axes,
    *,
    annotate: bool = True,
    show_player_labels: bool = True,
    monochrome: bool = False,
    overlay: ScoreboardOverlay | None = None,
) -> None:
    state = record.state_before
    action = record.validated_action.applied
    colors = stage_colors(monochrome)
    taus, xs, ys, zs = sample_trajectory(state, action, config)
    if config.court.lateral_motion_enabled:
        setup_3d_court_axes(ax, config, colors, show_axes=annotate)
        left_contact = (state.x0, state.y0, state.z0) if state.current_hitter == "left" else None
        right_contact = (state.x0, state.y0, state.z0) if state.current_hitter == "right" else None
        draw_players_3d(
            ax,
            config,
            colors,
            left_position=(state.x_left, state.y_left),
            right_position=(state.x_right, state.y_right),
            show_player_labels=show_player_labels,
            left_contact_xyz=left_contact,
            right_contact_xyz=right_contact,
        )
        ax.plot(xs, ys, zs, linestyle="--", color=colors["trajectory"], linewidth=2.8, alpha=0.9, zorder=2)
        ax.scatter([state.x0], [state.y0], [state.z0], color=colors["start"], s=_shuttle_marker_size(state.z0, config), zorder=7)

        if record.terminal_reason != "opponent_no_valid_shot":
            landing_x, landing_y = landing_position(state, action, config)
            ax.scatter([landing_x], [landing_y], [GROUND_MARKER_Z], color=colors["target"], s=55, alpha=0.8, zorder=4)
            ax.scatter([action.x_rec], [action.y_rec], [GROUND_MARKER_Z], color=colors["recovery"], s=65, alpha=0.28, zorder=3)

        if record.intercept_point is not None:
            ax.scatter(
                [record.intercept_point[0]],
                [record.intercept_point[1]],
                [record.intercept_point[2]],
                color=colors["intercept"],
                s=_shuttle_marker_size(record.intercept_point[2], config),
                zorder=8,
            )
    else:
        setup_court_axes(ax, config, colors, show_axes=annotate)
        left_contact = (state.x0, state.y0) if state.current_hitter == "left" else None
        right_contact = (state.x0, state.y0) if state.current_hitter == "right" else None
        draw_players(
            ax,
            config,
            colors,
            left_position=(state.x_left, state.y_left),
            right_position=(state.x_right, state.y_right),
            show_player_labels=show_player_labels,
            left_contact_xy=left_contact,
            right_contact_xy=right_contact,
        )
        ax.plot(xs, ys, linestyle="--", color=colors["trajectory"], linewidth=2.8, alpha=0.9, zorder=1)
        ax.scatter([state.x0], [state.y0], color=colors["start"], s=_shuttle_marker_size(state.z0, config), zorder=5)

        if record.terminal_reason != "opponent_no_valid_shot":
            landing_x, landing_y = landing_position(state, action, config)
            ax.scatter([landing_x], [landing_y], color=colors["target"], s=55, alpha=0.8, zorder=4)
            ax.scatter([action.x_rec], [action.y_rec], color=colors["recovery"], s=65, alpha=0.28, zorder=3)

        if record.intercept_point is not None:
            ax.scatter(
                [record.intercept_point[0]],
                [record.intercept_point[1]],
                color=colors["intercept"],
                s=_shuttle_marker_size(record.intercept_point[2], config),
                zorder=6,
            )

    if annotate:
        ax.set_title(_stage_title(record, config), fontsize=10)
    if overlay is not None:
        draw_scoreboard_overlay(ax, overlay, colors)
    if annotate and record.notes:
        text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
        text_fn(
            0.01,
            0.02,
            "\n".join(record.notes),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color=colors["notes"],
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3.0},
        )


def render_stage(
    record: StageRecord,
    config: SimulationConfig,
    output_path: Path,
    *,
    overlay: ScoreboardOverlay | None = None,
) -> None:
    ensure_directory(output_path.parent)
    subplot_kwargs = {"projection": "3d"} if config.court.lateral_motion_enabled else {}
    fig, ax = plt.subplots(figsize=config.render.figure_size, subplot_kw=subplot_kwargs)
    _draw_stage(record, config, ax, overlay=overlay)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def render_stage_image(
    record: StageRecord,
    config: SimulationConfig,
    *,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 140,
    annotate: bool = False,
    show_player_labels: bool = False,
    monochrome: bool = True,
    overlay: ScoreboardOverlay | None = None,
) -> np.ndarray:
    subplot_kwargs = {"projection": "3d"} if config.court.lateral_motion_enabled else {}
    fig, ax = plt.subplots(figsize=figure_size or config.render.figure_size, dpi=dpi, subplot_kw=subplot_kwargs)
    _draw_stage(
        record,
        config,
        ax,
        annotate=annotate,
        show_player_labels=show_player_labels,
        monochrome=monochrome,
        overlay=overlay,
    )
    if annotate:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def save_gif(image_paths: list[Path], output_path: Path, config: SimulationConfig) -> None:
    if not image_paths:
        return
    ensure_directory(output_path.parent)
    frames = [imageio.imread(path) for path in image_paths]
    duration = 1.0 / config.render.gif_fps
    imageio.mimsave(output_path, frames, duration=duration)
