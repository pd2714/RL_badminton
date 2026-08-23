from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1" / "figures" / "figure1"

from badminton.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle

from badminton.config import ActionConfig, CourtConfig, SimulationConfig
from badminton.render import (
    GROUND_MARKER_Z,
    OFFICIAL_DOUBLES_WIDTH,
    OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK,
    OFFICIAL_SHORT_SERVICE_FROM_NET,
    OFFICIAL_SINGLES_WIDTH,
    draw_players_3d,
    setup_3d_court_axes,
    stage_colors,
)
from badminton.trajectory import simulate_trajectory


ACCENT_BLUE = "#2563eb"
ACCENT_PINK = "#db2777"
ACCENT_GREEN = "#16a34a"
ACCENT_ORANGE = "#f59e0b"
ACCENT_RED = "#dc2626"
ACCENT_PURPLE = "#7c3aed"
INK = "#111827"
MUTED = "#6b7280"
LINE = "#374151"
PANEL_EDGE = "#d1d5db"
BOX_FILL = "#f9fafb"
SHOT_START = (-0.18, -2.95, 1.72)
SHOT_VELOCITY = (1.50, 6.60, 4.15)
AGENT_XY = (-0.18, -3.05)
OPPONENT_XY = (0.22, 2.75)
RECOVERY_XY = (0.95, -1.05)
PANEL_COURT_FILL = "#15803d"
PANEL_COURT_LINE = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Figure 1 schematic for the badminton RL paper.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for Figure 1 outputs.")
    parser.add_argument("--basename", default="figure1", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output resolution.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str, *, y: float = 1.02) -> None:
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_fn(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    text_fn(
        0.08,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )


def setup_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.8)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = BOX_FILL,
    edgecolor: str = LINE,
    color: str = INK,
    fontsize: float = 8.4,
    linewidth: float = 1.0,
) -> Rectangle:
    box = Rectangle(xy, width, height, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=2)
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        linespacing=1.22,
        zorder=3,
    )
    return box


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    linewidth: float = 1.2,
    mutation_scale: float = 10.0,
    connectionstyle: str = "arc3,rad=0",
    linestyle: str = "-",
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(arrow)
    return arrow


def add_loop_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, rad: float, color: str) -> None:
    add_arrow(ax, start, end, color=color, linewidth=1.4, mutation_scale=11, connectionstyle=f"arc3,rad={rad}")


def trajectory_arrays(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = simulate_trajectory(*SHOT_START, *SHOT_VELOCITY, config, sample_count=180)
    xs = np.array([point.x for point in result.samples])
    ys = np.array([point.y for point in result.samples])
    zs = np.array([point.z for point in result.samples])
    return xs, ys, zs


def intercept_point(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[float, float, float]:
    index = int(np.argmin(np.abs(ys - 1.35)))
    return float(xs[index]), float(ys[index]), float(zs[index])


def zoom_panel_a_court(ax: plt.Axes, config: SimulationConfig) -> None:
    ax.set_xlim(-3.25, 3.25)
    ax.set_ylim(-config.court.half_length - 0.30, config.court.half_length + 0.30)
    ax.set_zlim(-0.04, 3.35)
    try:
        ax.set_box_aspect((6.50, config.court.length + 0.60, 3.35), zoom=1.26)
    except TypeError:
        ax.set_box_aspect((6.50, config.court.length + 0.60, 3.35))
    if hasattr(ax, "dist"):
        ax.dist = 4.7


def draw_angle_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.2,
    mutation_scale: float = 8.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            zorder=5,
        )
    )


def draw_panel_a(ax: plt.Axes, config: SimulationConfig, *, panel_label: bool = True, annotate: bool = True) -> None:
    colors = stage_colors(monochrome=False)
    setup_3d_court_axes(ax, config, colors, show_axes=False)
    zoom_panel_a_court(ax, config)
    if panel_label:
        panel_title(ax, "A", "Court/state geometry", y=1.00)

    agent_xy = AGENT_XY
    opponent_xy = OPPONENT_XY
    contact_xyz = SHOT_START
    recovery_xy = RECOVERY_XY
    xs, ys, zs = trajectory_arrays(config)
    intercept_xyz = intercept_point(xs, ys, zs)

    draw_players_3d(
        ax,
        config,
        colors,
        left_position=agent_xy,
        right_position=opponent_xy,
        left_contact_xyz=contact_xyz,
        show_player_labels=False,
    )
    ax.plot(xs, ys, zs, color=ACCENT_RED, linewidth=2.0, alpha=0.9, zorder=7)
    arrow_index = max(12, len(xs) // 3)
    ax.quiver(
        xs[arrow_index],
        ys[arrow_index],
        zs[arrow_index],
        xs[arrow_index + 2] - xs[arrow_index],
        ys[arrow_index + 2] - ys[arrow_index],
        zs[arrow_index + 2] - zs[arrow_index],
        color=ACCENT_RED,
        linewidth=1.6,
        arrow_length_ratio=0.35,
        zorder=8,
    )
    ax.scatter([contact_xyz[0]], [contact_xyz[1]], [contact_xyz[2]], color=ACCENT_ORANGE, s=44, zorder=9)
    ax.scatter([intercept_xyz[0]], [intercept_xyz[1]], [intercept_xyz[2]], color="#fde047", edgecolors=INK, linewidths=0.4, s=52, zorder=10)
    ax.scatter([recovery_xy[0]], [recovery_xy[1]], [GROUND_MARKER_Z], color=ACCENT_PURPLE, s=68, alpha=0.95, zorder=9)
    ax.quiver(
        agent_xy[0],
        agent_xy[1],
        GROUND_MARKER_Z,
        recovery_xy[0] - agent_xy[0],
        recovery_xy[1] - agent_xy[1],
        0.0,
        color=ACCENT_PURPLE,
        linewidth=1.6,
        arrow_length_ratio=0.18,
        zorder=8,
    )

    if annotate:
        label_kwargs = {"fontsize": 7.7, "color": INK, "zorder": 10}
        ax.text(agent_xy[0] - 0.48, agent_xy[1] - 0.22, 1.28, "agent", ha="right", va="center", **label_kwargs)
        ax.text(opponent_xy[0] + 0.56, opponent_xy[1] + 0.48, 2.10, "opponent", ha="left", va="center", **label_kwargs)
        ax.text(contact_xyz[0] - 0.10, contact_xyz[1] - 0.25, contact_xyz[2] + 0.50, "contact point", ha="right", **label_kwargs)
        ax.text(xs[arrow_index] - 0.18, ys[arrow_index] + 0.08, zs[arrow_index] + 0.48, "shot trajectory", ha="center", **label_kwargs)
        ax.text(intercept_xyz[0] - 0.54, intercept_xyz[1] - 0.42, intercept_xyz[2] + 0.44, "opponent\nintercept", ha="right", va="bottom", linespacing=1.05, **label_kwargs)
        ax.text(recovery_xy[0] + 0.18, recovery_xy[1] + 0.18, 0.36, "recovery target", ha="left", va="center", **label_kwargs)


def draw_panel_a_top_court(ax: plt.Axes, config: SimulationConfig) -> None:
    display_half_width = max(float(config.court.half_width), OFFICIAL_DOUBLES_WIDTH / 2.0)
    half_length = float(config.court.half_length)
    singles_half_width = OFFICIAL_SINGLES_WIDTH / 2.0
    short_service_y = OFFICIAL_SHORT_SERVICE_FROM_NET
    long_service_doubles_y = half_length - OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK

    ax.add_patch(
        Rectangle(
            (-display_half_width, -half_length),
            2.0 * display_half_width,
            2.0 * half_length,
            facecolor=PANEL_COURT_FILL,
            edgecolor=PANEL_COURT_LINE,
            linewidth=2.0,
            zorder=0,
        )
    )

    court_line_kwargs = {"color": PANEL_COURT_LINE, "linewidth": 1.9, "zorder": 1}
    service_line_kwargs = {"color": PANEL_COURT_LINE, "linewidth": 1.7, "zorder": 1}
    for x_pos in (-singles_half_width, singles_half_width):
        ax.plot([x_pos, x_pos], [-half_length, half_length], **court_line_kwargs)
    for y_pos in (-short_service_y, short_service_y):
        ax.plot([-display_half_width, display_half_width], [y_pos, y_pos], **court_line_kwargs)
    for y_pos in (-long_service_doubles_y, long_service_doubles_y, 0.0):
        ax.plot([-display_half_width, display_half_width], [y_pos, y_pos], **court_line_kwargs)
    ax.plot([0.0, 0.0], [-half_length, -short_service_y], **service_line_kwargs)
    ax.plot([0.0, 0.0], [short_service_y, half_length], **service_line_kwargs)


def draw_panel_a_top(ax: plt.Axes, config: SimulationConfig, *, panel_label: bool = True, annotate: bool = True) -> None:
    setup_panel(ax)
    if panel_label:
        ax.text(0.0, 1.02, "A-top", transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.text(0.18, 1.02, "Top-view shot parameterization", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    xs, ys, zs = trajectory_arrays(config)
    del zs
    intercept_xyz = intercept_point(xs, ys, trajectory_arrays(config)[2])

    draw_panel_a_top_court(ax, config)

    ax.plot(xs, ys, color=ACCENT_RED, linewidth=2.3, zorder=2)
    draw_angle_arrow(ax, (xs[8], ys[8]), (xs[34], ys[34]), color=ACCENT_RED, linewidth=1.4, mutation_scale=11)
    ax.text(
        xs[28] + 0.12,
        ys[28] + 0.10,
        r"$v$",
        color=ACCENT_RED,
        fontsize=15,
        ha="left",
        va="bottom",
        math_fontfamily="cm",
    )
    ax.scatter([AGENT_XY[0]], [AGENT_XY[1]], color=ACCENT_BLUE, s=70, zorder=4)
    ax.scatter([OPPONENT_XY[0]], [OPPONENT_XY[1]], color=ACCENT_PINK, s=70, zorder=4)
    ax.scatter([SHOT_START[0]], [SHOT_START[1]], color=ACCENT_ORANGE, s=80, zorder=5)
    ax.scatter([RECOVERY_XY[0]], [RECOVERY_XY[1]], color=ACCENT_PURPLE, marker="x", s=130, linewidths=2.6, zorder=5)
    ax.scatter([intercept_xyz[0]], [intercept_xyz[1]], color="#fde047", edgecolors=INK, linewidths=0.8, s=95, zorder=5)

    phi_deg = float(np.degrees(np.arctan2(SHOT_VELOCITY[1], SHOT_VELOCITY[0])))
    ax.plot(
        [SHOT_START[0], SHOT_START[0] + 1.6],
        [SHOT_START[1], SHOT_START[1]],
        color=ACCENT_BLUE,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        zorder=6,
    )
    ax.add_patch(Arc((SHOT_START[0], SHOT_START[1]), 1.45, 1.45, theta1=0.0, theta2=phi_deg, color=ACCENT_BLUE, linewidth=1.6))
    ax.text(SHOT_START[0] + 0.72, SHOT_START[1] + 0.38, r"$\phi$", color=ACCENT_BLUE, fontsize=15, ha="center", va="center")

    if annotate:
        ax.text(AGENT_XY[0] - 0.12, AGENT_XY[1] - 0.50, "agent", color=ACCENT_BLUE, fontsize=10, ha="right", va="top")
        ax.text(OPPONENT_XY[0] + 0.18, OPPONENT_XY[1] + 0.28, "opponent", color=ACCENT_PINK, fontsize=10, ha="left", va="bottom")
        ax.text(SHOT_START[0] - 0.30, SHOT_START[1] + 0.20, "contact point", color=INK, fontsize=10, ha="right", va="bottom")
        ax.text(RECOVERY_XY[0] + 0.28, RECOVERY_XY[1] - 0.42, "recovery target", color=ACCENT_PURPLE, fontsize=10, ha="left", va="top")
        ax.text(intercept_xyz[0] + 0.22, intercept_xyz[1] + 0.18, "opponent intercept", color=INK, fontsize=10, ha="left", va="bottom")
        ax.text(1.72, -0.54, "shot trajectory", color=ACCENT_RED, fontsize=10, ha="left", va="bottom")

    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-4.65, 4.95)
    if annotate:
        ax.set_xlabel("x across court (m)")
        ax.set_ylabel("y along court (m)")
    ax.grid(False)


def rotate_top_clockwise(x: float | np.ndarray, y: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    return y, -x


def draw_panel_a_top_court_rot90(ax: plt.Axes, config: SimulationConfig) -> None:
    display_half_width = max(float(config.court.half_width), OFFICIAL_DOUBLES_WIDTH / 2.0)
    half_length = float(config.court.half_length)
    singles_half_width = OFFICIAL_SINGLES_WIDTH / 2.0
    short_service_y = OFFICIAL_SHORT_SERVICE_FROM_NET
    long_service_doubles_y = half_length - OFFICIAL_LONG_SERVICE_DOUBLES_FROM_BACK

    ax.add_patch(
        Rectangle(
            (-half_length, -display_half_width),
            2.0 * half_length,
            2.0 * display_half_width,
            facecolor=PANEL_COURT_FILL,
            edgecolor=PANEL_COURT_LINE,
            linewidth=2.0,
            zorder=0,
        )
    )

    def plot_rotated_line(x_values: list[float], y_values: list[float], **kwargs: object) -> None:
        rotated_x, rotated_y = rotate_top_clockwise(np.array(x_values), np.array(y_values))
        ax.plot(rotated_x, rotated_y, **kwargs)

    court_line_kwargs = {"color": PANEL_COURT_LINE, "linewidth": 1.9, "zorder": 1}
    service_line_kwargs = {"color": PANEL_COURT_LINE, "linewidth": 1.7, "zorder": 1}
    for x_pos in (-singles_half_width, singles_half_width):
        plot_rotated_line([x_pos, x_pos], [-half_length, half_length], **court_line_kwargs)
    for y_pos in (-short_service_y, short_service_y):
        plot_rotated_line([-display_half_width, display_half_width], [y_pos, y_pos], **court_line_kwargs)
    for y_pos in (-long_service_doubles_y, long_service_doubles_y, 0.0):
        plot_rotated_line([-display_half_width, display_half_width], [y_pos, y_pos], **court_line_kwargs)
    plot_rotated_line([0.0, 0.0], [-half_length, -short_service_y], **service_line_kwargs)
    plot_rotated_line([0.0, 0.0], [short_service_y, half_length], **service_line_kwargs)


def draw_panel_a_top_rot90(
    ax: plt.Axes,
    config: SimulationConfig,
    *,
    panel_label: bool = True,
    annotate: bool = True,
) -> None:
    setup_panel(ax)
    if panel_label:
        ax.text(0.0, 1.02, "A-top", transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.text(0.18, 1.02, "Top-view shot parameterization", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    xs, ys, zs = trajectory_arrays(config)
    intercept_xyz = intercept_point(xs, ys, zs)

    draw_panel_a_top_court_rot90(ax, config)

    rot_xs, rot_ys = rotate_top_clockwise(xs, ys)
    agent_xy = rotate_top_clockwise(AGENT_XY[0], AGENT_XY[1])
    opponent_xy = rotate_top_clockwise(OPPONENT_XY[0], OPPONENT_XY[1])
    shot_start_xy = rotate_top_clockwise(SHOT_START[0], SHOT_START[1])
    recovery_xy = rotate_top_clockwise(RECOVERY_XY[0], RECOVERY_XY[1])
    intercept_xy = rotate_top_clockwise(intercept_xyz[0], intercept_xyz[1])

    ax.plot(rot_xs, rot_ys, color=ACCENT_RED, linewidth=2.3, zorder=2)
    draw_angle_arrow(
        ax,
        rotate_top_clockwise(xs[8], ys[8]),
        rotate_top_clockwise(xs[34], ys[34]),
        color=ACCENT_RED,
        linewidth=1.4,
        mutation_scale=11,
    )
    ax.scatter([agent_xy[0]], [agent_xy[1]], color=ACCENT_BLUE, s=70, zorder=4)
    ax.scatter([opponent_xy[0]], [opponent_xy[1]], color=ACCENT_PINK, s=70, zorder=4)
    ax.scatter([shot_start_xy[0]], [shot_start_xy[1]], color=ACCENT_ORANGE, s=80, zorder=5)
    ax.scatter([recovery_xy[0]], [recovery_xy[1]], color=ACCENT_PURPLE, s=95, zorder=5)
    ax.scatter([intercept_xy[0]], [intercept_xy[1]], color="#fde047", edgecolors=INK, linewidths=0.8, s=95, zorder=5)

    phi_deg = float(np.degrees(np.arctan2(SHOT_VELOCITY[1], SHOT_VELOCITY[0])))
    baseline_start = (SHOT_START[0], SHOT_START[1])
    baseline_end = (SHOT_START[0] + 1.6, SHOT_START[1])
    ax.plot(
        [rotate_top_clockwise(*baseline_start)[0], rotate_top_clockwise(*baseline_end)[0]],
        [rotate_top_clockwise(*baseline_start)[1], rotate_top_clockwise(*baseline_end)[1]],
        color=ACCENT_BLUE,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        zorder=6,
    )
    theta = np.radians(np.linspace(0.0, phi_deg, 80))
    arc_x = SHOT_START[0] + 0.725 * np.cos(theta)
    arc_y = SHOT_START[1] + 0.725 * np.sin(theta)
    rot_arc_x, rot_arc_y = rotate_top_clockwise(arc_x, arc_y)
    ax.plot(rot_arc_x, rot_arc_y, color=ACCENT_BLUE, linewidth=1.6, zorder=6)
    phi_xy = rotate_top_clockwise(SHOT_START[0] + 0.72, SHOT_START[1] + 0.38)
    ax.text(
        phi_xy[0],
        phi_xy[1],
        r"$\phi$",
        color=ACCENT_BLUE,
        fontsize=36,
        ha="center",
        va="center",
        math_fontfamily="cm",
    )

    if annotate:
        ax.text(agent_xy[0] - 0.50, agent_xy[1] + 0.12, "agent", color=ACCENT_BLUE, fontsize=10, ha="right", va="bottom")
        ax.text(opponent_xy[0] + 0.28, opponent_xy[1] - 0.18, "opponent", color=ACCENT_PINK, fontsize=10, ha="left", va="top")
        ax.text(shot_start_xy[0] + 0.20, shot_start_xy[1] + 0.30, "contact point", color=INK, fontsize=10, ha="left", va="bottom")
        ax.text(recovery_xy[0] - 0.42, recovery_xy[1] - 0.28, "recovery target", color=ACCENT_PURPLE, fontsize=10, ha="right", va="top")
        ax.text(intercept_xy[0] + 0.18, intercept_xy[1] - 0.22, "opponent intercept", color=INK, fontsize=10, ha="left", va="top")
        ax.text(-0.54, -1.72, "shot trajectory", color=ACCENT_RED, fontsize=10, ha="right", va="bottom")

    ax.set_xlim(-6.85, 6.85)
    ax.set_ylim(-3.05, 3.05)
    if annotate:
        ax.set_xlabel("y along court (m)")
        ax.set_ylabel("-x across court (m)")
    ax.grid(False)


def draw_panel_a_side(ax: plt.Axes, config: SimulationConfig, *, panel_label: bool = True, annotate: bool = True) -> None:
    setup_panel(ax)
    if panel_label:
        ax.text(0.0, 1.02, "A-side", transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.text(0.18, 1.02, "Side-view launch parameters", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)
    xs, ys, zs = trajectory_arrays(config)
    del xs
    intercept_xyz = intercept_point(*trajectory_arrays(config))

    ax.plot(ys, zs, color=ACCENT_RED, linewidth=2.4, zorder=2)
    ax.scatter([ys[0]], [zs[0]], color=ACCENT_ORANGE, s=85, zorder=4)
    ax.scatter([intercept_xyz[1]], [intercept_xyz[2]], color="#fde047", edgecolors=INK, linewidths=0.8, s=95, zorder=5)
    ax.plot([0, 0], [0, config.court.net_height], color=LINE, linewidth=3.0)
    if annotate:
        ax.text(0.14, config.court.net_height + 0.12, "net", fontsize=10, color=MUTED, ha="left", va="bottom")

    theta_deg = float(np.degrees(np.arctan2(SHOT_VELOCITY[2], np.hypot(SHOT_VELOCITY[0], SHOT_VELOCITY[1]))))
    arrow_len = 1.25
    v_end = (ys[0] + arrow_len * np.cos(np.deg2rad(theta_deg)), zs[0] + arrow_len * np.sin(np.deg2rad(theta_deg)))
    ax.plot([ys[0], ys[0] + 1.35], [zs[0], zs[0]], color=MUTED, linewidth=1.1, linestyle="--")
    draw_angle_arrow(ax, (ys[0], zs[0]), v_end, color=ACCENT_BLUE, linewidth=1.6, mutation_scale=12)
    ax.add_patch(Arc((ys[0], zs[0]), 1.15, 1.15, theta1=0.0, theta2=theta_deg, color=ACCENT_BLUE, linewidth=1.5))
    ax.text(ys[0] + 0.60, zs[0] + 0.20, r"$\theta$", color=ACCENT_BLUE, fontsize=15, ha="center")
    ax.text(
        v_end[0] + 0.20,
        v_end[1] + 0.10,
        r"$v$",
        color=ACCENT_BLUE,
        fontsize=15,
        ha="left",
        va="bottom",
        math_fontfamily="cm",
    )

    if annotate:
        ax.text(ys[0] + 0.18, zs[0] - 0.30, "contact point", color=INK, fontsize=10, ha="left", va="top")
        ax.text(intercept_xyz[1] + 0.24, intercept_xyz[2] + 0.14, "opponent intercept", color=INK, fontsize=10, ha="left", va="bottom")
        ax.text(ys[70] + 0.10, zs[70] + 0.35, "shot trajectory", color=ACCENT_RED, fontsize=10, ha="left", va="bottom")

    ax.set_xlim(-3.45, 4.75)
    ax.set_ylim(0.0, max(3.35, float(zs.max()) + 0.35))
    if annotate:
        ax.set_xlabel("y along court (m)")
        ax.set_ylabel("z height (m)")
        ax.grid(alpha=0.12)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)


def draw_mini_court(ax: plt.Axes, xy: tuple[float, float], width: float, height: float) -> None:
    x0, y0 = xy
    court = Rectangle((x0, y0), width, height, facecolor="#ecfdf5", edgecolor=LINE, linewidth=0.9, zorder=1)
    ax.add_patch(court)
    ax.plot([x0, x0 + width], [y0 + height / 2.0, y0 + height / 2.0], color=LINE, linewidth=0.9, zorder=2)
    ax.plot([x0 + width / 2.0, x0 + width / 2.0], [y0, y0 + height], color="#9ca3af", linewidth=0.7, zorder=2)
    ax.plot([x0 + width * 0.22, x0 + width * 0.78], [y0 + height * 0.28, y0 + height * 0.28], color="#9ca3af", linewidth=0.7)
    ax.plot([x0 + width * 0.22, x0 + width * 0.78], [y0 + height * 0.72, y0 + height * 0.72], color="#9ca3af", linewidth=0.7)


def draw_panel_b(ax: plt.Axes) -> None:
    setup_panel(ax)
    panel_title(ax, "B", "Structured policy action")
    add_box(
        ax,
        (0.04, 0.57),
        0.22,
        0.24,
        r"state $s_t$ includes" + "\n"
        + "agent pos.\n"
        + "opponent pos.\n"
        + "shuttle/contact\n"
        + "rally context",
        facecolor="#eff6ff",
        edgecolor=ACCENT_BLUE,
        fontsize=6.8,
    )
    add_box(ax, (0.34, 0.61), 0.17, 0.16, "shared\nencoder", facecolor="#f8fafc", fontsize=8.0)
    add_box(ax, (0.60, 0.77), 0.15, 0.10, r"$\pi_\phi(\phi\mid s_t)$", facecolor="#fef2f2", edgecolor=ACCENT_RED, fontsize=7.2)
    add_box(ax, (0.79, 0.64), 0.16, 0.10, r"$\pi_\theta(\theta\mid s_t,\phi)$", facecolor="#fef2f2", edgecolor=ACCENT_RED, fontsize=6.8)
    add_box(ax, (0.60, 0.50), 0.16, 0.10, r"$\pi_v(v\mid s_t,\phi,\theta)$", facecolor="#fef2f2", edgecolor=ACCENT_RED, fontsize=6.6)
    add_box(
        ax,
        (0.78, 0.33),
        0.17,
        0.11,
        r"$\pi_{\mathrm{rec}}(r\mid s_t,a_{\mathrm{shot}})$",
        facecolor="#f5f3ff",
        edgecolor=ACCENT_PURPLE,
        fontsize=6.7,
    )
    add_box(ax, (0.37, 0.30), 0.25, 0.09, r"$a_{\mathrm{shot}}=(\phi,\theta,v)$", facecolor="#fff7ed", edgecolor=ACCENT_ORANGE, fontsize=8.0)
    add_box(ax, (0.37, 0.13), 0.30, 0.09, r"$a_t=(a_{\mathrm{shot}},a_{\mathrm{rec}})$", facecolor="#fffbeb", edgecolor=ACCENT_ORANGE, fontsize=8.2)
    ax.text(
        0.74,
        0.035,
        r"$p(a_t\mid s_t)=\pi_\phi\,\pi_\theta\,\pi_v\,\pi_{\mathrm{rec}}$",
        ha="center",
        va="center",
        fontsize=7.0,
        color=INK,
    )

    add_arrow(ax, (0.26, 0.69), (0.34, 0.69), color=ACCENT_BLUE)
    add_arrow(ax, (0.51, 0.70), (0.60, 0.81), color=LINE)
    add_arrow(ax, (0.75, 0.80), (0.79, 0.70), color=LINE, connectionstyle="arc3,rad=-0.10")
    add_arrow(ax, (0.79, 0.64), (0.73, 0.60), color=LINE, connectionstyle="arc3,rad=-0.08")
    add_arrow(ax, (0.68, 0.50), (0.53, 0.39), color=ACCENT_RED, connectionstyle="arc3,rad=0.05")
    add_arrow(ax, (0.62, 0.35), (0.78, 0.38), color=ACCENT_ORANGE)
    add_arrow(ax, (0.86, 0.33), (0.66, 0.20), color=ACCENT_PURPLE, connectionstyle="arc3,rad=-0.12")
    add_arrow(ax, (0.50, 0.30), (0.50, 0.22), color=ACCENT_ORANGE)

    draw_mini_court(ax, (0.07, 0.13), 0.20, 0.25)
    contact = (0.17, 0.22)
    ax.scatter([contact[0]], [contact[1]], s=24, color=ACCENT_ORANGE, zorder=5)
    for target, alpha in [((0.11, 0.34), 0.35), ((0.25, 0.33), 0.35), ((0.23, 0.15), 0.35)]:
        arrow = add_arrow(ax, contact, target, color=ACCENT_RED, linewidth=1.0, mutation_scale=8, connectionstyle="arc3,rad=0.08")
        arrow.set_alpha(alpha)
    ax.scatter([0.11], [0.17], s=38, marker="x", linewidths=1.8, color=ACCENT_PURPLE, zorder=5)
    ax.text(0.055, 0.06, "shot candidates + recovery", fontsize=6.7, color=MUTED, ha="left")


def draw_panel_c(ax: plt.Axes) -> None:
    setup_panel(ax)
    panel_title(ax, "C", "Rally transition")
    boxes = [
        ((0.04, 0.58), 0.16, 0.14, "agent\ncontact", "#eff6ff", ACCENT_BLUE),
        ((0.28, 0.58), 0.18, 0.14, "shot +\nrecovery", "#fffbeb", ACCENT_ORANGE),
        ((0.55, 0.58), 0.18, 0.14, "shuttle\nphysics", "#ecfdf5", ACCENT_GREEN),
        ((0.79, 0.58), 0.17, 0.14, "opponent\nreturn", "#fdf2f8", ACCENT_PINK),
        ((0.40, 0.24), 0.23, 0.13, "next contact\nstate", "#f8fafc", LINE),
    ]
    for xy, width, height, text, fill, edge in boxes:
        add_box(ax, xy, width, height, text, facecolor=fill, edgecolor=edge)
    add_arrow(ax, (0.20, 0.65), (0.28, 0.65), color=LINE)
    add_arrow(ax, (0.46, 0.65), (0.55, 0.65), color=LINE)
    add_arrow(ax, (0.73, 0.65), (0.79, 0.65), color=LINE)
    add_arrow(ax, (0.87, 0.58), (0.62, 0.34), color=LINE, connectionstyle="arc3,rad=-0.17")
    add_arrow(ax, (0.40, 0.34), (0.12, 0.58), color=LINE, connectionstyle="arc3,rad=-0.16", linestyle="--")
    ax.text(
        0.50,
        0.83,
        "Recovery value depends on\nopponent response.",
        ha="center",
        va="center",
        fontsize=8.0,
        color=ACCENT_PURPLE,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.11,
        r"$s_t\;-\;(a_{\mathrm{shot}},a_{\mathrm{rec}})\;-\!\!\rightarrow\;s_{t+1},\, r_t$",
        ha="center",
        va="center",
        fontsize=10.8,
        color=INK,
    )


def draw_matrix_icon(ax: plt.Axes, xy: tuple[float, float], cell: float = 0.033) -> None:
    values = np.array(
        [
            [0.50, 0.58, 0.63, 0.71],
            [0.42, 0.50, 0.55, 0.65],
            [0.37, 0.45, 0.50, 0.57],
            [0.29, 0.35, 0.43, 0.50],
        ]
    )
    x0, y0 = xy
    cmap = plt.get_cmap("RdYlGn")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.add_patch(
                Rectangle(
                    (x0 + col * cell, y0 + (values.shape[0] - 1 - row) * cell),
                    cell,
                    cell,
                    facecolor=cmap(values[row, col]),
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            )
    ax.add_patch(Rectangle((x0, y0), values.shape[1] * cell, values.shape[0] * cell, fill=False, edgecolor=LINE, linewidth=0.8, zorder=4))


def draw_panel_d(ax: plt.Axes) -> None:
    setup_panel(ax)
    panel_title(ax, "D", "Self-play training and frozen evaluation")
    add_box(ax, (0.055, 0.67), 0.215, 0.12, "current\n" + r"policy $\pi_\theta$", facecolor="#eff6ff", edgecolor=ACCENT_BLUE, fontsize=7.2)
    add_box(ax, (0.365, 0.67), 0.155, 0.12, "opponent\npool", facecolor="#fdf2f8", edgecolor=ACCENT_PINK)
    add_box(ax, (0.20, 0.43), 0.20, 0.12, "rally\nrollouts", facecolor="#ecfdf5", edgecolor=ACCENT_GREEN)
    add_box(ax, (0.06, 0.22), 0.18, 0.11, "PPO\nupdate", facecolor="#fffbeb", edgecolor=ACCENT_ORANGE)
    add_box(ax, (0.345, 0.22), 0.175, 0.11, "new\ncheckpoint", facecolor="#f8fafc", edgecolor=LINE, fontsize=7.6)
    add_loop_arrow(ax, (0.27, 0.73), (0.365, 0.73), rad=0.0, color=LINE)
    add_loop_arrow(ax, (0.43, 0.67), (0.34, 0.55), rad=0.05, color=LINE)
    add_loop_arrow(ax, (0.22, 0.55), (0.15, 0.33), rad=0.07, color=LINE)
    add_loop_arrow(ax, (0.24, 0.27), (0.345, 0.27), rad=0.0, color=LINE)
    add_loop_arrow(ax, (0.42, 0.33), (0.16, 0.67), rad=-0.34, color=ACCENT_BLUE)

    ax.plot([0.56, 0.56], [0.16, 0.84], color=PANEL_EDGE, linewidth=1.0)
    ax.text(0.30, 0.85, "training", ha="center", va="center", fontsize=8.1, color=MUTED)
    ax.text(0.77, 0.85, "evaluation", ha="center", va="center", fontsize=8.1, color=MUTED)
    add_box(
        ax,
        (0.62, 0.63),
        0.30,
        0.14,
        r"frozen checkpoints" + "\n" + r"$\{\pi_0,\pi_{200k},\pi_{1M},\pi_{3M},\pi_{6M}\}$",
        facecolor="#f8fafc",
        edgecolor=LINE,
        fontsize=6.8,
    )
    add_box(ax, (0.64, 0.41), 0.25, 0.12, "round-robin\nmatches", facecolor="#ecfdf5", edgecolor=ACCENT_GREEN)
    add_box(ax, (0.62, 0.19), 0.235, 0.12, "win-rate\nmatrix + Elo", facecolor="#fff7ed", edgecolor=ACCENT_ORANGE, fontsize=6.8)
    add_arrow(ax, (0.77, 0.63), (0.77, 0.53), color=LINE)
    add_arrow(ax, (0.77, 0.41), (0.77, 0.31), color=LINE)
    draw_matrix_icon(ax, (0.875, 0.205), cell=0.018)
    ax.text(
        0.50,
        0.06,
        "non-stationary training,\nstationary evaluation",
        ha="center",
        va="center",
        fontsize=7.7,
        color=INK,
        fontweight="bold",
        linespacing=1.05,
    )


def build_figure() -> plt.Figure:
    configure_matplotlib()
    config = make_config()
    fig = plt.figure(figsize=(7.15, 6.55), constrained_layout=False)
    fig.suptitle("Physics-based badminton self-play environment", fontsize=12.6, fontweight="bold", y=0.975)
    gs = fig.add_gridspec(2, 2, left=0.035, right=0.985, bottom=0.05, top=0.89, wspace=0.16, hspace=0.29)
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    draw_panel_a(ax_a, config)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)
    return fig


def make_config() -> SimulationConfig:
    config = SimulationConfig(
        court=CourtConfig(mode="2d"),
        action=ActionConfig(trajectory_mode="drag_square", horizontal_drag_coefficient=0.12, vertical_drag_coefficient=0.10),
    )
    return config


def build_standalone_panels() -> dict[str, plt.Figure]:
    configure_matplotlib()
    config = make_config()
    panels: dict[str, plt.Figure] = {}

    fig_a = plt.figure(figsize=(7.2, 4.0), constrained_layout=False)
    ax_a = fig_a.add_axes([0.00, 0.00, 1.00, 0.96], projection="3d")
    draw_panel_a(ax_a, config)
    panels["panel_A_main"] = fig_a

    fig_a_no_text = plt.figure(figsize=(7.2, 4.0), constrained_layout=False)
    ax_a_no_text = fig_a_no_text.add_axes([0.00, 0.00, 1.00, 0.98], projection="3d")
    draw_panel_a(ax_a_no_text, config, panel_label=False, annotate=False)
    panels["panel_A_main_no_text"] = fig_a_no_text

    fig_top, ax_top = plt.subplots(figsize=(5.5, 7.2), constrained_layout=True)
    draw_panel_a_top(ax_top, config)
    panels["panel_A_top"] = fig_top

    fig_top_no_text, ax_top_no_text = plt.subplots(figsize=(5.5, 7.2), constrained_layout=True)
    draw_panel_a_top(ax_top_no_text, config, panel_label=False, annotate=False)
    panels["panel_A_top_no_text"] = fig_top_no_text

    fig_top_no_text_rot90, ax_top_no_text_rot90 = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    draw_panel_a_top_rot90(ax_top_no_text_rot90, config, panel_label=False, annotate=False)
    panels["panel_A_top_no_text_rot90"] = fig_top_no_text_rot90

    fig_side, ax_side = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    draw_panel_a_side(ax_side, config)
    panels["panel_A_side"] = fig_side

    fig_side_no_text, ax_side_no_text = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    draw_panel_a_side(ax_side_no_text, config, panel_label=False, annotate=False)
    panels["panel_A_side_no_text"] = fig_side_no_text

    fig_b, ax_b = plt.subplots(figsize=(7.2, 4.9), constrained_layout=True)
    draw_panel_b(ax_b)
    panels["panel_B"] = fig_b

    fig_c, ax_c = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    draw_panel_c(ax_c)
    panels["panel_C"] = fig_c

    fig_d, ax_d = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    draw_panel_d(ax_d)
    panels["panel_D"] = fig_d
    return panels


def save_figure_pair(fig: plt.Figure, output_dir: Path, name: str, *, dpi: int) -> tuple[Path, Path]:
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    png_path, pdf_path = save_figure_pair(fig, args.output_dir, args.basename, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")

    for name, panel_fig in build_standalone_panels().items():
        panel_png, panel_pdf = save_figure_pair(panel_fig, args.output_dir, name, dpi=args.dpi)
        plt.close(panel_fig)
        print(f"wrote {panel_png}")
        print(f"wrote {panel_pdf}")


if __name__ == "__main__":
    main()
