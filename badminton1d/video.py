from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from badminton1d.config import SimulationConfig
from badminton1d.playback import (
    FrameSnapshot,
    MatchTrace,
    RallyTrace,
    StageTrace,
    interpolate_stage,
    match_trace_from_dict,
    match_trace_to_dict,
    rally_trace_from_dict,
    rally_trace_to_dict,
)
from badminton1d.render import (
    GROUND_MARKER_Z,
    ScoreboardOverlay,
    draw_players,
    draw_players_3d,
    draw_scoreboard_overlay,
    setup_3d_court_axes,
    setup_court_axes,
    stage_colors,
)
from badminton1d.trajectory import position_at_time
from badminton1d.utils import ensure_directory


@dataclass(frozen=True)
class VideoExportResult:
    frame_paths: list[Path]
    gif_path: Path
    mp4_path: Path | None
    trace_path: Path


@dataclass(frozen=True)
class TrainingProgressSample:
    step: int
    trace: RallyTrace
    opponent_label: str | None = None
    rally_won: bool | None = None
    invalid_action_rate: float | None = None


def _marker_size(z_value: float, config: SimulationConfig) -> float:
    ratio = np.clip(z_value / max(config.render.z_max, 1e-6), 0.0, 1.0)
    return float(config.render.shuttle_marker_min + ratio * (config.render.shuttle_marker_max - config.render.shuttle_marker_min))


def _trajectory_segment(stage: StageTrace, config: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.linspace(0.0, stage.end_time, config.render.trajectory_samples)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    local_config = SimulationConfig(
        court=config.court,
        player=config.player,
        render=config.render,
        action=replace(
            config.action,
            gravity=stage.gravity,
            trajectory_mode=stage.trajectory_mode,
            drag_coefficient=stage.drag_coefficient,
            horizontal_drag_coefficient=stage.horizontal_drag_coefficient,
            vertical_drag_coefficient=stage.vertical_drag_coefficient,
            drag_dt=stage.drag_dt,
        ),
    )
    for t in times:
        x, y, z = position_at_time(
            stage.shuttle_start[0],
            stage.shuttle_start[1],
            stage.shuttle_start[2],
            stage.shuttle_velocity[0],
            stage.shuttle_velocity[1],
            stage.shuttle_velocity[2],
            float(t),
            local_config,
        )
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(zs, dtype=float)


def _overlay_text(snapshot: FrameSnapshot) -> str:
    lines = [
        f"stage {snapshot.stage_index}",
        f"hitter {snapshot.hitter_side}",
        f"flight {snapshot.local_time:.2f}/{snapshot.playback_duration:.2f}s",
        f"z={snapshot.shuttle_position[2]:.2f}m",
    ]
    if snapshot.intended_intercept_point is not None:
        y_int = snapshot.intended_intercept_point[1]
        z_int = snapshot.intended_intercept_point[2]
        if snapshot.intended_intercept_time is not None:
            lines.append(f"intend t={snapshot.intended_intercept_time:.2f}s y={y_int:.2f} z={z_int:.2f}")
        else:
            lines.append(f"intend y={y_int:.2f} z={z_int:.2f}")
    if snapshot.terminal and snapshot.winner is not None:
        lines.append(f"winner {snapshot.winner}")
    return "\n".join(lines)


def _progress_overlay_text(sample: TrainingProgressSample, snapshot: FrameSnapshot) -> str:
    lines = [f"train step {sample.step}", _overlay_text(snapshot)]
    if sample.opponent_label:
        lines.append(f"opponent {sample.opponent_label}")
    if sample.rally_won is not None:
        lines.append(f"rally {'won' if sample.rally_won else 'lost'}")
    if sample.invalid_action_rate is not None:
        lines.append(f"invalid rate {sample.invalid_action_rate:.3f}")
    return "\n".join(lines)


def _match_intercept_text(stage: StageTrace) -> str | None:
    intended_point = stage.intended_intercept_point or stage.intercept_point
    intended_time = stage.intended_intercept_time if stage.intended_intercept_time is not None else stage.end_time if stage.intercept_point is not None else None
    if intended_point is None:
        return None
    y_int = intended_point[1]
    z_int = intended_point[2]
    if intended_time is None:
        return f"intend y={y_int:.2f} z={z_int:.2f}"
    return f"intend t={intended_time:.2f}s y={y_int:.2f} z={z_int:.2f}"


def _resolve_view(config: SimulationConfig, view: str) -> str:
    normalized = view.strip().lower()
    if normalized == "auto":
        return "3d" if config.court.lateral_motion_enabled else "side"
    if normalized not in {"top", "side", "3d"}:
        raise ValueError("view must be 'auto', 'top', 'side', or '3d'")
    return normalized


def _draw_service_markers_side_view(
    ax: plt.Axes,
    config: SimulationConfig,
    colors: dict[str, str],
) -> None:
    marker_height = min(max(config.court.net_height * 0.12, 0.12), 0.2)
    for service_y in (-config.court.service_line_distance_from_net, config.court.service_line_distance_from_net):
        if config.court.lateral_motion_enabled:
            ax.axvline(service_y, color=colors["service_line"], linewidth=1.1, linestyle="--", zorder=1)
        else:
            ax.plot(
                [service_y, service_y],
                [0.0, marker_height],
                color=colors["service_line"],
                linewidth=1.4,
                zorder=1,
            )


def _draw_1d_side_players(
    ax: plt.Axes,
    *,
    snapshot: FrameSnapshot,
    stage: StageTrace,
    config: SimulationConfig,
    colors: dict[str, str],
) -> None:
    # In the 1D side view, draw a player body plus a capped racket reach so
    # the proportions stay visually consistent with the physical net height.
    body_height = min(config.player.z_max - config.player.r_reach, 1.7)
    body_height = max(body_height, config.court.net_height * 0.9)
    racket_length = float(config.player.r_reach)
    bar_width = 0.14
    shoulder_height = max(body_height - 0.08, 0.0)
    label_height = body_height + 0.16

    players = [
        ("L", snapshot.left_player_position[1], colors["left_player"], stage.hitter_side == "left"),
        ("R", snapshot.right_player_position[1], colors["right_player"], stage.hitter_side == "right"),
    ]
    for label, player_y, color, is_hitter in players:
        racket = Rectangle(
            (player_y - bar_width / 2.0, 0.0),
            bar_width,
            body_height,
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(racket)
        ax.text(player_y, label_height, label, color=color, ha="center", va="bottom", fontsize=10, weight="bold")
        if is_hitter:
            contact_y = stage.shuttle_start[1]
            contact_z = stage.shuttle_start[2]
            dy = float(contact_y - player_y)
            dz = float(contact_z - shoulder_height)
            distance = float(np.hypot(dy, dz))
            if distance > 1e-6:
                scale = min(1.0, racket_length / distance)
                racket_tip_y = player_y + dy * scale
                racket_tip_z = shoulder_height + dz * scale
                ax.add_patch(
                    FancyArrowPatch(
                        (player_y, shoulder_height),
                        (racket_tip_y, racket_tip_z),
                        arrowstyle="-|>",
                        mutation_scale=12.0,
                        linewidth=1.8,
                        color=colors["player_arrow"],
                        shrinkA=2.0,
                        shrinkB=2.0,
                        zorder=4,
                    )
                )


def _render_frame(
    stage: StageTrace,
    snapshot: FrameSnapshot,
    config: SimulationConfig,
    *,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 140,
    monochrome: bool = False,
    overlay: ScoreboardOverlay | None = None,
    overlay_text: str | None = None,
    view: str = "auto",
) -> np.ndarray:
    resolved_view = _resolve_view(config, view)
    colors = stage_colors(monochrome)
    intended_point = stage.intended_intercept_point or stage.intercept_point
    fig, ax = plt.subplots(figsize=figure_size or config.render.figure_size, dpi=dpi)
    if resolved_view == "top":
        setup_court_axes(ax, config, colors, show_axes=False)
        left_contact = stage.shuttle_start[:2] if stage.hitter_side == "left" else None
        right_contact = stage.shuttle_start[:2] if stage.hitter_side == "right" else None
        draw_players(
            ax,
            config,
            colors,
            left_position=snapshot.left_player_position,
            right_position=snapshot.right_player_position,
            show_player_labels=False,
            left_contact_xy=left_contact,
            right_contact_xy=right_contact,
        )

        traj_xs, traj_ys, _ = _trajectory_segment(stage, config)
        ax.plot(traj_xs, traj_ys, linestyle="--", color=colors["trajectory"], linewidth=1.6, alpha=0.5)
        ax.scatter([stage.recovery_target[0]], [stage.recovery_target[1]], color=colors["recovery"], s=70, alpha=0.35, zorder=3)
        ax.scatter([stage.shuttle_landing[0]], [stage.shuttle_landing[1]], color=colors["target"], s=55, alpha=0.75, zorder=3)
        if intended_point is not None:
            ax.scatter(
                [intended_point[0]],
                [intended_point[1]],
                color=colors["intercept"],
                s=_marker_size(intended_point[2], config),
                alpha=0.85,
                zorder=4,
            )

        ax.scatter(
            [snapshot.shuttle_position[0]],
            [snapshot.shuttle_position[1]],
            color=colors["start"],
            s=_marker_size(snapshot.shuttle_position[2], config),
            zorder=5,
        )
    elif resolved_view == "3d":
        plt.close(fig)
        fig = plt.figure(figsize=figure_size or config.render.figure_size, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        setup_3d_court_axes(ax, config, colors, show_axes=False)

        left_contact = stage.shuttle_start if stage.hitter_side == "left" else None
        right_contact = stage.shuttle_start if stage.hitter_side == "right" else None
        draw_players_3d(
            ax,
            config,
            colors,
            left_position=snapshot.left_player_position,
            right_position=snapshot.right_player_position,
            show_player_labels=False,
            left_contact_xyz=left_contact,
            right_contact_xyz=right_contact,
        )

        traj_xs, traj_ys, traj_zs = _trajectory_segment(stage, config)
        ax.plot(traj_xs, traj_ys, traj_zs, linestyle="--", color=colors["trajectory"], linewidth=1.6, alpha=0.55, zorder=2)
        ax.scatter([stage.recovery_target[0]], [stage.recovery_target[1]], [GROUND_MARKER_Z], color=colors["recovery"], s=70, alpha=0.35, zorder=3)
        ax.scatter([stage.shuttle_landing[0]], [stage.shuttle_landing[1]], [GROUND_MARKER_Z], color=colors["target"], s=55, alpha=0.75, zorder=3)
        if intended_point is not None:
            ax.scatter(
                [intended_point[0]],
                [intended_point[1]],
                [intended_point[2]],
                color=colors["intercept"],
                s=_marker_size(intended_point[2], config),
                alpha=0.85,
                zorder=4,
            )

        ax.scatter(
            [snapshot.shuttle_position[0]],
            [snapshot.shuttle_position[1]],
            [snapshot.shuttle_position[2]],
            color=colors["start"],
            s=_marker_size(snapshot.shuttle_position[2], config),
            zorder=5,
        )
    elif resolved_view == "side":
        y_min = -config.court.half_length - config.render.court_padding
        y_max = config.court.half_length + config.render.court_padding
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(-0.15, config.render.z_max + 0.3)
        ax.set_facecolor(colors["court_fill"])
        ax.axhline(0.0, color=colors["court_line"], linewidth=2.0, zorder=1)
        ax.axvline(config.court.net_y, ymin=0.0, ymax=config.court.net_height / max(config.render.z_max + 0.3, 1e-6), color=colors["net"], linewidth=2.4, zorder=2)
        _draw_service_markers_side_view(ax, config, colors)
        ax.set_axis_off()

        traj_xs, traj_ys, traj_zs = _trajectory_segment(stage, config)
        ax.plot(traj_ys, traj_zs, linestyle="--", color=colors["trajectory"], linewidth=1.6, alpha=0.5)
        ax.scatter([stage.recovery_target[1]], [0.0], color=colors["recovery"], s=70, alpha=0.35, zorder=3)
        ax.scatter([stage.shuttle_landing[1]], [0.0], color=colors["target"], s=55, alpha=0.75, zorder=3)
        if intended_point is not None:
            ax.scatter(
                [intended_point[1]],
                [intended_point[2]],
                color=colors["intercept"],
                s=_marker_size(intended_point[2], config),
                alpha=0.85,
                zorder=4,
            )
        if config.court.lateral_motion_enabled:
            ax.scatter(
                [snapshot.left_player_position[1], snapshot.right_player_position[1]],
                [0.0, 0.0],
                color=[colors["left_player"], colors["right_player"]],
                s=150,
                zorder=3,
            )
        else:
            _draw_1d_side_players(
                ax,
                snapshot=snapshot,
                stage=stage,
                config=config,
                colors=colors,
            )
        ax.scatter(
            [snapshot.shuttle_position[1]],
            [snapshot.shuttle_position[2]],
            color=colors["start"],
            s=_marker_size(snapshot.shuttle_position[2], config),
            zorder=5,
        )
    else:
        raise ValueError("view must be 'auto', 'top', 'side', or '3d'")

    if overlay is not None:
        draw_scoreboard_overlay(ax, overlay, colors)
    elif overlay_text is not None:
        text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
        text_fn(
            0.015,
            0.98,
            overlay_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=colors["notes"],
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 4.0},
        )

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def render_video_frame(
    stage: StageTrace,
    snapshot: FrameSnapshot,
    config: SimulationConfig,
    *,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 140,
    monochrome: bool = False,
    view: str = "auto",
) -> np.ndarray:
    return _render_frame(
        stage,
        snapshot,
        config,
        figure_size=figure_size,
        dpi=dpi,
        monochrome=monochrome,
        overlay_text=_overlay_text(snapshot),
        view=view,
    )


def render_match_frame(
    rally_trace: RallyTrace,
    stage: StageTrace,
    snapshot: FrameSnapshot,
    config: SimulationConfig,
    *,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 140,
    monochrome: bool = False,
    view: str = "auto",
) -> np.ndarray:
    if rally_trace.server is None or rally_trace.rally_number is None:
        raise ValueError("rally_trace must include server and rally_number metadata")

    if stage.terminal:
        score_left = rally_trace.score_after_left
        score_right = rally_trace.score_after_right
    else:
        score_left = rally_trace.score_before_left
        score_right = rally_trace.score_before_right

    overlay = ScoreboardOverlay(
        score_left=score_left,
        score_right=score_right,
        current_server=rally_trace.server,
        rally_number=rally_trace.rally_number,
        stage_number=stage.stage_index + 1,
        hitter_side=snapshot.hitter_side,
        flight_time_text=f"flight {snapshot.local_time:.2f}/{stage.end_time:.2f}s",
        intercept_text=_match_intercept_text(stage),
        point_winner=rally_trace.winner if stage.terminal else None,
        match_winner=rally_trace.match_winner if stage.terminal else None,
    )
    return _render_frame(
        stage,
        snapshot,
        config,
        figure_size=figure_size,
        dpi=dpi,
        monochrome=monochrome,
        overlay=overlay,
        view=view,
    )


def _stage_sample_times(duration: float, fps: int) -> list[float]:
    frame_count = max(2, int(np.ceil(duration * fps)) + 1)
    return np.linspace(0.0, duration, frame_count).tolist()


def _write_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> Path:
    imageio.mimsave(output_path, frames, duration=1.0 / fps)
    return output_path


def _write_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> Path | None:
    try:
        imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)
        return output_path
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None or not frames:
            return None
        height, width = frames[0].shape[:2]
        command = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert process.stdin is not None
            for frame in frames:
                rgb_frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
                process.stdin.write(rgb_frame.tobytes())
            process.stdin.close()
            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            if output_path.exists():
                output_path.unlink()
            return None
        if return_code == 0 and output_path.exists():
            return output_path
        if output_path.exists():
            output_path.unlink()
        return None


def export_rally_video(
    trace: RallyTrace,
    config: SimulationConfig,
    output_dir: Path,
    *,
    fps: int = 30,
    stage_pause: float = 0.15,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 140,
    monochrome: bool = False,
    view: str = "auto",
) -> VideoExportResult:
    if not trace.stages:
        raise ValueError("trace must contain at least one stage")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if stage_pause < 0.0:
        raise ValueError("stage_pause must be zero or greater")

    ensure_directory(output_dir)
    frame_paths: list[Path] = []
    frames: list[np.ndarray] = []
    frame_index = 0
    pause_frames = int(round(stage_pause * fps))

    for stage in trace.stages:
        stage_times = _stage_sample_times(stage.playback_duration, fps)
        for local_time in stage_times:
            snapshot = interpolate_stage(stage, local_time)
            frame = render_video_frame(
                stage,
                snapshot,
                config,
                figure_size=figure_size,
                dpi=dpi,
                monochrome=monochrome,
                view=view,
            )
            frame_path = output_dir / f"frame_{frame_index:05d}.png"
            imageio.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frames.append(frame)
            frame_index += 1

        if pause_frames > 0 and frames:
            hold_frame = frames[-1]
            for _ in range(pause_frames):
                frame_path = output_dir / f"frame_{frame_index:05d}.png"
                imageio.imwrite(frame_path, hold_frame)
                frame_paths.append(frame_path)
                frames.append(hold_frame.copy())
                frame_index += 1

    trace_path = output_dir / "rally_trace.json"
    trace_path.write_text(json.dumps(rally_trace_to_dict(trace), indent=2), encoding="utf-8")

    gif_path = _write_gif(frames, output_dir / "rally.gif", fps)
    mp4_path = _write_mp4(frames, output_dir / "rally.mp4", fps)
    return VideoExportResult(frame_paths=frame_paths, gif_path=gif_path, mp4_path=mp4_path, trace_path=trace_path)


def export_match_video(
    trace: MatchTrace,
    config: SimulationConfig,
    output_dir: Path,
    *,
    fps: int = 30,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 140,
    monochrome: bool = False,
    write_mp4: bool = True,
    view: str = "auto",
) -> VideoExportResult:
    if not trace.rallies:
        raise ValueError("trace must contain at least one rally")
    if fps <= 0:
        raise ValueError("fps must be positive")

    ensure_directory(output_dir)
    frame_paths: list[Path] = []
    frames: list[np.ndarray] = []
    frame_index = 0

    for rally_trace in trace.rallies:
        for stage in rally_trace.stages:
            stage_times = _stage_sample_times(stage.playback_duration, fps)
            for local_time in stage_times:
                snapshot = interpolate_stage(stage, local_time)
                frame = render_match_frame(
                    rally_trace,
                    stage,
                    snapshot,
                    config,
                    figure_size=figure_size,
                    dpi=dpi,
                    monochrome=monochrome,
                    view=view,
                )
                frame_path = output_dir / f"frame_{frame_index:05d}.png"
                imageio.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frames.append(frame)
                frame_index += 1

        pause_frames = int(round(rally_trace.pause_duration * fps))
        if pause_frames > 0 and frames:
            hold_frame = frames[-1]
            for _ in range(pause_frames):
                frame_path = output_dir / f"frame_{frame_index:05d}.png"
                imageio.imwrite(frame_path, hold_frame)
                frame_paths.append(frame_path)
                frames.append(hold_frame.copy())
                frame_index += 1

    trace_path = output_dir / "match_trace.json"
    trace_path.write_text(json.dumps(match_trace_to_dict(trace), indent=2), encoding="utf-8")

    gif_path = _write_gif(frames, output_dir / "match.gif", fps)
    mp4_path = _write_mp4(frames, output_dir / "match.mp4", fps) if write_mp4 else None
    return VideoExportResult(frame_paths=frame_paths, gif_path=gif_path, mp4_path=mp4_path, trace_path=trace_path)


def export_training_progress_video(
    samples: list[TrainingProgressSample],
    config: SimulationConfig,
    output_dir: Path,
    *,
    fps: int = 20,
    stage_pause: float = 0.15,
    rally_pause: float = 0.9,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 140,
    monochrome: bool = False,
    view: str = "auto",
    write_frames: bool = True,
    write_gif: bool = True,
    write_mp4: bool = True,
) -> VideoExportResult:
    if not samples:
        raise ValueError("samples must contain at least one rally trace")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if stage_pause < 0.0:
        raise ValueError("stage_pause must be zero or greater")
    if rally_pause < 0.0:
        raise ValueError("rally_pause must be zero or greater")

    ensure_directory(output_dir)
    frame_paths: list[Path] = []
    frames: list[np.ndarray] = []
    frame_index = 0
    stage_pause_frames = int(round(stage_pause * fps))
    rally_pause_frames = int(round(rally_pause * fps))

    for sample in samples:
        for stage in sample.trace.stages:
            stage_times = _stage_sample_times(stage.playback_duration, fps)
            for local_time in stage_times:
                snapshot = interpolate_stage(stage, local_time)
                frame = _render_frame(
                    stage,
                    snapshot,
                    config,
                    figure_size=figure_size,
                    dpi=dpi,
                    monochrome=monochrome,
                    overlay_text=_progress_overlay_text(sample, snapshot),
                    view=view,
                )
                if write_frames:
                    frame_path = output_dir / f"frame_{frame_index:05d}.png"
                    imageio.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                frames.append(frame)
                frame_index += 1

            if stage_pause_frames > 0 and frames:
                hold_frame = frames[-1]
                for _ in range(stage_pause_frames):
                    if write_frames:
                        frame_path = output_dir / f"frame_{frame_index:05d}.png"
                        imageio.imwrite(frame_path, hold_frame)
                        frame_paths.append(frame_path)
                    frames.append(hold_frame.copy())
                    frame_index += 1

        if rally_pause_frames > 0 and frames:
            hold_frame = frames[-1]
            for _ in range(rally_pause_frames):
                if write_frames:
                    frame_path = output_dir / f"frame_{frame_index:05d}.png"
                    imageio.imwrite(frame_path, hold_frame)
                    frame_paths.append(frame_path)
                frames.append(hold_frame.copy())
                frame_index += 1

    trace_payload = {
        "sample_count": len(samples),
        "samples": [
            {
                "step": sample.step,
                "opponent_label": sample.opponent_label,
                "rally_won": sample.rally_won,
                "invalid_action_rate": sample.invalid_action_rate,
                "trace": rally_trace_to_dict(sample.trace),
            }
            for sample in samples
        ],
    }
    trace_path = output_dir / "training_progress_trace.json"
    trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")

    gif_path = _write_gif(frames, output_dir / "training_progress.gif", fps) if write_gif else output_dir / "training_progress.gif"
    mp4_path = _write_mp4(frames, output_dir / "training_progress.mp4", fps) if write_mp4 else None
    return VideoExportResult(frame_paths=frame_paths, gif_path=gif_path, mp4_path=mp4_path, trace_path=trace_path)


def export_training_progress_preview_video(
    samples: list[TrainingProgressSample],
    config: SimulationConfig,
    output_dir: Path,
    *,
    fps: int = 8,
    frames_per_stage: int = 4,
    max_stages_per_sample: int = 12,
    rally_pause: float = 0.25,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 90,
    monochrome: bool = False,
    view: str = "auto",
    write_gif: bool = False,
    write_mp4: bool = True,
) -> VideoExportResult:
    if not samples:
        raise ValueError("samples must contain at least one rally trace")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frames_per_stage <= 0:
        raise ValueError("frames_per_stage must be positive")
    if max_stages_per_sample <= 0:
        raise ValueError("max_stages_per_sample must be positive")
    if rally_pause < 0.0:
        raise ValueError("rally_pause must be zero or greater")

    ensure_directory(output_dir)
    frames: list[np.ndarray] = []
    pause_frames = int(round(rally_pause * fps))

    for sample in samples:
        stage_total = len(sample.trace.stages)
        stage_indices = np.linspace(0, stage_total - 1, num=min(stage_total, max_stages_per_sample), dtype=int)
        seen: set[int] = set()
        ordered_indices: list[int] = []
        for raw_index in stage_indices:
            index = int(raw_index)
            if index not in seen:
                ordered_indices.append(index)
                seen.add(index)

        for stage_index in ordered_indices:
            stage = sample.trace.stages[stage_index]
            local_times = np.linspace(0.0, stage.playback_duration, num=frames_per_stage).tolist()
            for local_time in local_times:
                snapshot = interpolate_stage(stage, float(local_time))
                frame = _render_frame(
                    stage,
                    snapshot,
                    config,
                    figure_size=figure_size,
                    dpi=dpi,
                    monochrome=monochrome,
                    overlay_text=_progress_overlay_text(sample, snapshot),
                    view=view,
                )
                frames.append(frame)

        if pause_frames > 0 and frames:
            hold_frame = frames[-1]
            for _ in range(pause_frames):
                frames.append(hold_frame.copy())

    trace_payload = {
        "sample_count": len(samples),
        "preview_mode": True,
        "frames_per_stage": frames_per_stage,
        "max_stages_per_sample": max_stages_per_sample,
        "samples": [
            {
                "step": sample.step,
                "opponent_label": sample.opponent_label,
                "rally_won": sample.rally_won,
                "invalid_action_rate": sample.invalid_action_rate,
                "trace": rally_trace_to_dict(sample.trace),
            }
            for sample in samples
        ],
    }
    trace_path = output_dir / "training_progress_preview_trace.json"
    trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")

    gif_path = _write_gif(frames, output_dir / "training_progress_preview.gif", fps) if write_gif else output_dir / "training_progress_preview.gif"
    mp4_path = _write_mp4(frames, output_dir / "training_progress_preview.mp4", fps) if write_mp4 else None
    return VideoExportResult(frame_paths=[], gif_path=gif_path, mp4_path=mp4_path, trace_path=trace_path)


__all__ = [
    "VideoExportResult",
    "TrainingProgressSample",
    "export_match_video",
    "export_training_progress_preview_video",
    "export_rally_video",
    "export_training_progress_video",
    "match_trace_from_dict",
    "rally_trace_from_dict",
    "render_match_frame",
    "render_video_frame",
]
