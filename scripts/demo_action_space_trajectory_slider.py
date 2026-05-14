from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib
import numpy as np

from badminton1d.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.dynamics import candidate_intercept_points, feasible_intercept_indices, valid_hitter_action
from badminton1d.render import (
    GROUND_MARKER_Z,
    OFFICIAL_DOUBLES_WIDTH,
    PLAYER_FOOT_Z,
    setup_3d_court_axes,
    stage_colors,
)
from badminton1d.shot_generators.shot_naming import name_velocity_shot
from badminton1d.state import Side, StageState
from badminton1d.trajectory import simulate_trajectory
from badminton1d.utils import default_player_position, opponent_side, service_court_x, side_y_bounds, x_bounds


DEFAULT_RUN_DIR = Path(
    "outputs/rl/"
    "selfplay_2d_accel65_theta65_speed50_rt015_ps40_reach150_zmax26_feasible003_nointercept002_oppint001_attack002_deflift02_ifratio01_reaction_continuation_20k_equal_sides_20260506"
)


def configure_interactive_backend() -> None:
    if "--no-show" in sys.argv:
        return
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        return
    for candidate in ("MacOSX", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


configure_interactive_backend()

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize trajectories from the discrete velocity-oriented action space used by a self-play run."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Self-play output directory containing selfplay_config.json.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional explicit selfplay_config.json path.")
    parser.add_argument("--hitter", choices=("left", "right"), default="left", help="Current hitter side.")
    parser.add_argument("--stage-index", type=int, default=1, help="Use 0 for serve bins, 1 or higher for rally-shot bins.")
    parser.add_argument("--x0", type=float, default=None, help="Initial contact x. Defaults to hitter start x.")
    parser.add_argument("--y0", type=float, default=None, help="Initial contact y. Defaults to hitter start y.")
    parser.add_argument("--z0", type=float, default=1.7, help="Initial contact height.")
    parser.add_argument("--opponent-x", type=float, default=None, help="Initial opponent/receiver x. Defaults to receiver start x.")
    parser.add_argument("--opponent-y", type=float, default=None, help="Initial opponent/receiver y. Defaults to receiver start y.")
    parser.add_argument("--phi-bin-init", type=int, default=None, help="Initial horizontal-angle bin.")
    parser.add_argument("--theta-bin-init", type=int, default=None, help="Initial vertical-angle bin.")
    parser.add_argument("--speed-bin-init", type=int, default=None, help="Initial total-speed bin.")
    parser.add_argument("--x-rec-bin-init", type=int, default=1, help="Initial recovery x bin.")
    parser.add_argument("--y-rec-bin-init", type=int, default=1, help="Initial recovery y bin.")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional output path for saving the current figure.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening an interactive window.")
    return parser.parse_args()


def _load_run_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find run config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Run config must be a JSON object: {path}")
    return data


def _number(data: dict[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _integer(data: dict[str, object], *keys: str, default: int) -> int:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return int(value)
    return int(default)


def _build_configs(run_data: dict[str, object]) -> tuple[SimulationConfig, DiscreteActionConfig, float, str]:
    default_discrete = DiscreteActionConfig()
    default_sim = SimulationConfig()
    sim_config = SimulationConfig(
        court=CourtConfig(mode=str(run_data.get("court_mode", "2d"))),
        player=PlayerConfig(
            v_max=_number(run_data, "player_speed", default_sim.player.v_max),
            acceleration=_number(run_data, "player_acceleration", default_sim.player.acceleration),
            r_reach=_number(run_data, "racket_length", default_sim.player.r_reach),
            z_max=_number(run_data, "max_hitting_height", default_sim.player.z_max),
            movement_model=str(run_data.get("movement_model", default_sim.player.movement_model)),
        ),
        action=ActionConfig(
            trajectory_mode=str(run_data.get("trajectory_mode", "drag_square")),
            drag_coefficient=_number(run_data, "drag_coefficient", default_sim.action.drag_coefficient),
            horizontal_drag_coefficient=_number(
                run_data,
                "horizontal_drag_coefficient",
                default_sim.action.effective_horizontal_drag_coefficient,
            ),
            vertical_drag_coefficient=_number(
                run_data,
                "vertical_drag_coefficient",
                default_sim.action.effective_vertical_drag_coefficient,
            ),
            vy_min_forward=_number(run_data, "shuttle_speed_min", default_sim.action.vy_min_forward),
            vy_max_forward=_number(run_data, "shuttle_speed_max", default_sim.action.vy_max_forward),
            intercept_count=_integer(run_data, "intercept_count", default=default_sim.action.intercept_count),
        ),
    )
    discrete_config = DiscreteActionConfig(
        phi_bins=_integer(run_data, "phi_bins", "vx_bins", "v_x_bins", default=default_discrete.phi_bins),
        theta_bins=_integer(run_data, "theta_bins", "vy_bins", "v_y_bins", default=default_discrete.theta_bins),
        speed_bins=_integer(run_data, "speed_bins", "vz_bins", "v_z_bins", default=default_discrete.speed_bins),
        x_rec_bins=_integer(run_data, "x_rec_bins", "x-rec-bins", default=default_discrete.x_rec_bins),
        y_rec_bins=_integer(run_data, "y_rec_bins", "y-rec-bins", default=default_discrete.y_rec_bins),
    )
    reaction_time = _number(run_data, "reaction_time", 0.0)
    policy_type = str(run_data.get("policy_type", "velocity_oriented"))
    return sim_config, discrete_config, reaction_time, policy_type


def _clip_index(value: int | None, count: int, *, default: int) -> int:
    if value is None:
        value = default
    return int(np.clip(int(value), 0, max(count - 1, 0)))


def _flat_hitter_index(
    *,
    phi_index: int,
    theta_index: int,
    speed_index: int,
    x_rec_index: int,
    y_rec_index: int,
    mapper: DiscreteActionMapper,
) -> int:
    cfg = mapper.discrete_config
    return (
        ((((phi_index * cfg.theta_bins) + theta_index) * cfg.speed_bins + speed_index)
         * mapper._effective_x_rec_bins + x_rec_index)
        * cfg.y_rec_bins
        + y_rec_index
    )


def _set_scatter3d(artist, x: float | None, y: float | None, z: float | None) -> None:
    if x is None or y is None or z is None:
        artist._offsets3d = ([], [], [])
        return
    artist._offsets3d = ([x], [y], [z])


def _set_scatter3d_many(artist, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
    artist._offsets3d = (xs.tolist(), ys.tolist(), zs.tolist())


def _use_full_width_3d_view(ax: plt.Axes, config: SimulationConfig) -> None:
    display_half_width = max(config.court.half_width, OFFICIAL_DOUBLES_WIDTH / 2.0)
    pad = config.render.court_padding * 0.06
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()
    x_min = -display_half_width - pad
    x_max = display_half_width + pad
    ax.set_xlim(x_min, x_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))


def _player_body_height(config: SimulationConfig) -> float:
    body_height = min(config.player.z_max - config.player.r_reach, 1.7)
    return max(body_height, config.court.net_height * 0.9)


def _update_player_artist(
    body,
    head,
    position: tuple[float, float],
    *,
    body_height: float,
) -> None:
    x_pos, y_pos = position
    body.set_data_3d(
        [x_pos, x_pos],
        [y_pos, y_pos],
        [PLAYER_FOOT_Z, PLAYER_FOOT_Z + body_height],
    )
    _set_scatter3d(head, x_pos, y_pos, PLAYER_FOOT_Z + body_height)


def _state_from_sliders(
    sliders: dict[str, Slider],
    *,
    config: SimulationConfig,
    hitter: Side,
    reaction_time: float,
) -> StageState:
    stage_index = max(int(sliders["stage"].val), 0)
    if stage_index == 0:
        _, default_service_y = default_player_position(hitter, config)
        x0 = float(service_court_x(hitter, config))
        y0 = float(default_service_y)
    else:
        x0 = float(sliders["x0"].val)
        y0 = float(sliders["y0"].val)
    opponent_x = float(sliders["opponent_x"].val)
    opponent_y = float(sliders["opponent_y"].val)
    if hitter == "left":
        x_left, y_left = x0, y0
        x_right, y_right = opponent_x, opponent_y
    else:
        x_left, y_left = opponent_x, opponent_y
        x_right, y_right = x0, y0
    return StageState(
        x_left=float(x_left),
        y_left=float(y_left),
        x_right=float(x_right),
        y_right=float(y_right),
        current_hitter=hitter,
        x0=x0,
        y0=y0,
        z0=float(sliders["z0"].val),
        v_x_left=0.0,
        v_y_left=0.0,
        v_x_right=0.0,
        v_y_right=0.0,
        reaction_time_left=reaction_time,
        reaction_time_right=reaction_time,
        stage_index=max(int(stage_index), 0),
    )


def _receiver_feasibility_config(config: SimulationConfig) -> SimulationConfig:
    return replace(config, player=replace(config.player, movement_model="accelerated"))


def build_plot(
    args: argparse.Namespace,
    config: SimulationConfig,
    discrete_config: DiscreteActionConfig,
    *,
    reaction_time: float,
) -> tuple[plt.Figure, dict[str, Slider]]:
    if not config.court.lateral_motion_enabled:
        raise ValueError("This action-space slider is for the 2D velocity-oriented action space.")

    mapper = DiscreteActionMapper(config, discrete_config, policy_type="velocity_oriented")
    colors = stage_colors(monochrome=False)
    hitter: Side = args.hitter
    receiver = opponent_side(hitter)
    default_hitter_x, default_hitter_y = default_player_position(hitter, config)
    default_receiver_x, default_receiver_y = default_player_position(receiver, config)

    x_low, x_high = x_bounds(config)
    y_low, y_high = side_y_bounds(hitter, config)
    opponent_y_low, opponent_y_high = side_y_bounds(receiver, config)
    z_low, z_high = config.player.z_min, config.player.z_max
    stage_init = max(int(args.stage_index), 0)
    if stage_init == 0:
        x0_init = float(np.clip(service_court_x(hitter, config), x_low, x_high))
        y0_init = float(np.clip(default_hitter_y, y_low, y_high))
    else:
        x0_init = float(np.clip(default_hitter_x if args.x0 is None else args.x0, x_low, x_high))
        y0_init = float(np.clip(default_hitter_y if args.y0 is None else args.y0, y_low, y_high))
    z0_init = float(np.clip(args.z0, z_low, z_high))
    opponent_x_init = float(np.clip(default_receiver_x if args.opponent_x is None else args.opponent_x, x_low, x_high))
    opponent_y_init = float(np.clip(default_receiver_y if args.opponent_y is None else args.opponent_y, opponent_y_low, opponent_y_high))

    phi_init = _clip_index(args.phi_bin_init, mapper._effective_phi_bins, default=mapper._effective_phi_bins // 2)
    theta_init = _clip_index(args.theta_bin_init, discrete_config.theta_bins, default=discrete_config.theta_bins // 2)
    speed_init = _clip_index(args.speed_bin_init, discrete_config.speed_bins, default=discrete_config.speed_bins // 2)
    xrec_init = _clip_index(args.x_rec_bin_init, mapper._effective_x_rec_bins, default=mapper._effective_x_rec_bins // 2)
    yrec_init = _clip_index(args.y_rec_bin_init, discrete_config.y_rec_bins, default=discrete_config.y_rec_bins // 2)

    fig = plt.figure(figsize=(12.4, 10.0))
    ax = fig.add_axes([0.02, 0.28, 0.70, 0.68], projection="3d")
    setup_3d_court_axes(ax, config, colors, show_axes=False)
    _use_full_width_3d_view(ax, config)
    ax.set_title("Discrete Action-Space Trajectory Slider", pad=4.0)

    body_height = _player_body_height(config)
    (left_player_body,) = ax.plot([], [], [], color=colors["left_player"], linewidth=5.0, solid_capstyle="round", zorder=4)
    (right_player_body,) = ax.plot([], [], [], color=colors["right_player"], linewidth=5.0, solid_capstyle="round", zorder=4)
    left_player_head = ax.scatter([], [], [], color=colors["left_player"], s=28, zorder=5)
    right_player_head = ax.scatter([], [], [], color=colors["right_player"], s=28, zorder=5)

    (trajectory_line,) = ax.plot([], [], [], color=colors["trajectory"], linewidth=2.2, alpha=0.8, label="trajectory")
    (aim_line,) = ax.plot([], [], [], color="tab:cyan", linewidth=1.8, alpha=0.9, label="phi aim")
    start_point = ax.scatter([], [], [], color=colors["start"], s=70, label="contact", zorder=5)
    landing_point = ax.scatter([], [], [], color=colors["target"], s=70, label="landing", zorder=5)
    net_point = ax.scatter([], [], [], color=colors["intercept"], s=80, label="net crossing", zorder=5)
    recovery_point = ax.scatter([], [], [], color=colors["recovery"], s=75, alpha=0.85, label="recovery", zorder=5)
    candidate_points = ax.scatter([], [], [], color="0.7", s=22, alpha=0.45, label="intercept samples", zorder=4)
    feasible_points = ax.scatter([], [], [], color="gold", s=42, alpha=0.9, label="feasible intercepts", zorder=5)

    info_text = fig.text(0.75, 0.93, "", ha="left", va="top", fontsize=9.5, linespacing=1.32)
    fig.legend(loc="upper left", bbox_to_anchor=(0.75, 0.41), frameon=False, fontsize=9, borderaxespad=0.0)

    slider_specs = [
        ("stage", 0, max(30, stage_init), stage_init, 1),
        ("x0", x_low, x_high, x0_init, 0.01),
        ("y0", y_low, y_high, y0_init, 0.01),
        ("z0", z_low, z_high, z0_init, 0.01),
        ("opponent_x", x_low, x_high, opponent_x_init, 0.01),
        ("opponent_y", opponent_y_low, opponent_y_high, opponent_y_init, 0.01),
        ("phi_bin", 0, mapper._effective_phi_bins - 1, phi_init, 1),
        ("theta_bin", 0, discrete_config.theta_bins - 1, theta_init, 1),
        ("speed_bin", 0, discrete_config.speed_bins - 1, speed_init, 1),
        ("x_rec_bin", 0, mapper._effective_x_rec_bins - 1, xrec_init, 1),
        ("y_rec_bin", 0, discrete_config.y_rec_bins - 1, yrec_init, 1),
    ]
    sliders: dict[str, Slider] = {}
    top = 0.255
    step = 0.025
    for index, (label, lower, upper, init, valstep) in enumerate(slider_specs):
        axis = fig.add_axes([0.16, top - index * step, 0.56, 0.018])
        sliders[label] = Slider(axis, label, lower, upper, valinit=init, valstep=valstep)

    def update(_: float) -> None:
        stage_index = max(int(sliders["stage"].val), 0)
        if stage_index == 0:
            _, default_service_y = default_player_position(hitter, config)
            for slider_name, service_value in (
                ("x0", service_court_x(hitter, config)),
                ("y0", default_service_y),
            ):
                slider = sliders[slider_name]
                previous_eventson = slider.eventson
                slider.eventson = False
                slider.set_val(float(service_value))
                slider.eventson = previous_eventson

        state = _state_from_sliders(
            sliders,
            config=config,
            hitter=hitter,
            reaction_time=reaction_time,
        )
        phi_index = int(sliders["phi_bin"].val)
        theta_index = int(sliders["theta_bin"].val)
        speed_index = int(sliders["speed_bin"].val)
        x_rec_index = int(sliders["x_rec_bin"].val)
        y_rec_index = int(sliders["y_rec_bin"].val)
        flat_index = _flat_hitter_index(
            phi_index=phi_index,
            theta_index=theta_index,
            speed_index=speed_index,
            x_rec_index=x_rec_index,
            y_rec_index=y_rec_index,
            mapper=mapper,
        )
        decoded = mapper.decode_hitter(flat_index, state)
        action = decoded.shot_action

        _update_player_artist(
            left_player_body,
            left_player_head,
            (state.x_left, state.y_left),
            body_height=body_height,
        )
        _update_player_artist(
            right_player_body,
            right_player_head,
            (state.x_right, state.y_right),
            body_height=body_height,
        )

        result = simulate_trajectory(state.x0, state.y0, state.z0, action.v_x, action.v_y, action.v_z, config)
        xs = np.asarray([point.x for point in result.samples], dtype=float)
        ys = np.asarray([point.y for point in result.samples], dtype=float)
        zs = np.asarray([point.z for point in result.samples], dtype=float)
        crossing = result.net_crossing
        legal = valid_hitter_action(state, action, config, result=result)
        feasibility_config = _receiver_feasibility_config(config)
        candidate_ts, candidate_xs, candidate_ys, candidate_zs = candidate_intercept_points(state, action, feasibility_config)
        feasible = feasible_intercept_indices(state, action, feasibility_config)

        color = "tab:blue" if legal else "tab:red"
        trajectory_line.set_data_3d(xs, ys, zs)
        trajectory_line.set_color(color)
        phi = float(mapper.phi_grid(state)[phi_index])
        theta = float(mapper.theta_grid(state, phi_index)[theta_index])
        speed = float(mapper.speed_grid(state, phi_index, theta_index)[speed_index])
        action_phi_values = np.rad2deg(mapper.phi_grid(state))
        action_phi_low = float(np.min(action_phi_values))
        action_phi_high = float(np.max(action_phi_values))
        shot_name = name_velocity_shot(
            hitter=hitter,
            contact_x=state.x0,
            contact_y=state.y0,
            landing_x=result.landing_x,
            landing_y=result.landing_y,
            theta_degrees=float(np.degrees(theta)),
            config=config,
        )
        aim_length = 3.2
        aim_line.set_data_3d(
            [state.x0, state.x0 + aim_length * np.cos(phi)],
            [state.y0, state.y0 + aim_length * np.sin(phi)],
            [state.z0, state.z0],
        )
        _set_scatter3d(start_point, state.x0, state.y0, state.z0)
        _set_scatter3d(landing_point, result.landing_x, result.landing_y, GROUND_MARKER_Z)
        _set_scatter3d(recovery_point, action.x_rec, action.y_rec, GROUND_MARKER_Z)
        if crossing is None:
            _set_scatter3d(net_point, None, None, None)
        else:
            _set_scatter3d(net_point, crossing.x, crossing.y, crossing.z)
            net_point.set_color(color)
        _set_scatter3d_many(candidate_points, candidate_xs, candidate_ys, candidate_zs)
        if feasible:
            feasible_indices = np.asarray(feasible, dtype=int)
            _set_scatter3d_many(
                feasible_points,
                candidate_xs[feasible_indices],
                candidate_ys[feasible_indices],
                candidate_zs[feasible_indices],
            )
        else:
            _set_scatter3d_many(feasible_points, np.asarray([]), np.asarray([]), np.asarray([]))

        z_net = crossing.z if crossing is not None else float("nan")
        info_text.set_text(
            "\n".join(
                [
                    f"run = {args.run_dir.name}",
                    f"hitter = {hitter}  receiver = {receiver}  stage = {stage_index}",
                    f"mode = {config.action.trajectory_mode}  drag = "
                    f"({config.action.effective_horizontal_drag_coefficient:.3f}, "
                    f"{config.action.effective_vertical_drag_coefficient:.3f})",
                    f"receiver reach = accelerated from rest, v_max {feasibility_config.player.v_max:.2f} m/s, "
                    f"a {feasibility_config.player.acceleration:.2f} m/s^2",
                    f"bins = phi {discrete_config.phi_bins}, theta {discrete_config.theta_bins}, "
                    f"speed {discrete_config.speed_bins}, rec {discrete_config.x_rec_bins}x{discrete_config.y_rec_bins}",
                    f"action index = {flat_index}",
                    f"shot = {shot_name}",
                    f"contact = ({state.x0:.2f}, {state.y0:.2f}, {state.z0:.2f})",
                    f"opponent = ({sliders['opponent_x'].val:.2f}, {sliders['opponent_y'].val:.2f})",
                    f"bin indices = ({phi_index}, {theta_index}, {speed_index}, {x_rec_index}, {y_rec_index})",
                    f"phi = {np.degrees(phi):.1f} deg  action span [{action_phi_low:.1f}, {action_phi_high:.1f}]",
                    f"theta = {np.degrees(theta):.1f} deg",
                    f"speed = {speed:.2f} m/s",
                    f"launch = ({action.v_x:.2f}, {action.v_y:.2f}, {action.v_z:.2f}) m/s",
                    f"recovery = ({action.x_rec:.2f}, {action.y_rec:.2f})",
                    f"landing = ({result.landing_x:.2f}, {result.landing_y:.2f}) in {result.landing_time:.2f}s",
                    f"z(net) = {z_net:.2f} m",
                    f"intercepts = {len(feasible)}/{len(candidate_ts)} feasible",
                    f"status: {'legal' if legal else 'invalid'}",
                ]
            )
        )
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)

    update(0.0)
    return fig, sliders


def main() -> None:
    args = parse_args()
    config_path = args.config_path or args.run_dir / "selfplay_config.json"
    run_data = _load_run_config(config_path)
    sim_config, discrete_config, reaction_time, policy_type = _build_configs(run_data)
    if policy_type != "velocity_oriented":
        raise ValueError(f"This slider expects policy_type='velocity_oriented', got {policy_type!r}.")

    fig, sliders = build_plot(args, sim_config, discrete_config, reaction_time=reaction_time)

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=160, bbox_inches="tight")
        print(f"saved figure to {args.save_path}")

    print(
        "action-space trajectory sliders ready: "
        f"x0={sliders['x0'].val:.2f}, "
        f"y0={sliders['y0'].val:.2f}, "
        f"z0={sliders['z0'].val:.2f}, "
        f"stage={int(sliders['stage'].val)}, "
        f"opponent_x={sliders['opponent_x'].val:.2f}, "
        f"opponent_y={sliders['opponent_y'].val:.2f}, "
        f"phi_bin={int(sliders['phi_bin'].val)}, "
        f"theta_bin={int(sliders['theta_bin'].val)}, "
        f"speed_bin={int(sliders['speed_bin'].val)}, "
        f"x_rec_bin={int(sliders['x_rec_bin'].val)}, "
        f"y_rec_bin={int(sliders['y_rec_bin'].val)}"
    )

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
