from __future__ import annotations

from dataclasses import dataclass, replace

from badminton1d.agents import StageAgent
from badminton1d.config import SimulationConfig
from badminton1d.dynamics import feasible_intercept_indices, validate_and_clip_shot_action
from badminton1d.env import Badminton1DEnv
from badminton1d.state import Side, StageRecord, StageState
from badminton1d.utils import default_player_position, opponent_side, service_court_x, side_center_y


@dataclass(frozen=True)
class MatchScore:
    left: int = 0
    right: int = 0

    def award_point(self, winner: Side) -> "MatchScore":
        if winner == "left":
            return MatchScore(left=self.left + 1, right=self.right)
        return MatchScore(left=self.left, right=self.right + 1)


@dataclass(frozen=True)
class MatchConfig:
    target_score: int = 11
    max_stages_per_rally: int = 30
    serve_z0: float = 1.15
    left_service_x: float | None = None
    left_service_y: float | None = None
    right_service_x: float | None = None
    right_service_y: float | None = None


@dataclass(frozen=True)
class RallyResult:
    rally_number: int
    server: Side
    score_before: MatchScore
    score_after: MatchScore
    initial_state: StageState
    records: list[StageRecord]
    winner: Side
    final_state: StageState


@dataclass(frozen=True)
class MatchResult:
    rallies: list[RallyResult]
    target_score: int
    initial_server: Side
    final_score: MatchScore
    winner: Side


def side_midpoints(config: SimulationConfig) -> tuple[float, float]:
    return side_center_y("left", config), side_center_y("right", config)


def default_start_positions(config: SimulationConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    return default_player_position("left", config), default_player_position("right", config)


def service_receive_position(side: Side, config: SimulationConfig) -> tuple[float, float]:
    x, y = default_player_position(side, config)
    if not config.court.lateral_motion_enabled:
        return x, y

    offset = config.court.service_receive_x_offset_toward_center
    if side == "left":
        return min(x + offset, 0.0), y
    return max(x - offset, 0.0), y


def service_line_positions(config: SimulationConfig) -> tuple[float, float]:
    line_distance = config.court.service_line_distance_from_net
    return config.court.net_y - line_distance, config.court.net_y + line_distance


def is_legal_service_target(y_aim: float, receiver: Side, config: SimulationConfig) -> bool:
    left_line_y, right_line_y = service_line_positions(config)
    if receiver == "left":
        return y_aim <= left_line_y
    return y_aim >= right_line_y


def serve_positions(
    config: SimulationConfig,
    match_config: MatchConfig | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    default_left, default_right = default_start_positions(config)
    active_match_config = match_config or MatchConfig()
    left_position = (
        service_court_x("left", config)
        if active_match_config.left_service_x is None
        else active_match_config.left_service_x,
        default_left[1] if active_match_config.left_service_y is None else active_match_config.left_service_y,
    )
    right_position = (
        service_court_x("right", config)
        if active_match_config.right_service_x is None
        else active_match_config.right_service_x,
        default_right[1] if active_match_config.right_service_y is None else active_match_config.right_service_y,
    )
    return left_position, right_position


def reset_for_serve(
    server: Side,
    config: SimulationConfig,
    match_config: MatchConfig | None = None,
) -> StageState:
    active_match_config = match_config or MatchConfig()
    (left_x, left_y), (right_x, right_y) = serve_positions(config, active_match_config)
    if server == "left":
        default_right_receive_x, default_right_receive_y = service_receive_position("right", config)
        right_x = default_right_receive_x if active_match_config.right_service_x is None else right_x
        right_y = default_right_receive_y if active_match_config.right_service_y is None else right_y
        x0, y0 = left_x, left_y
    else:
        default_left_receive_x, default_left_receive_y = service_receive_position("left", config)
        left_x = default_left_receive_x if active_match_config.left_service_x is None else left_x
        left_y = default_left_receive_y if active_match_config.left_service_y is None else left_y
        x0, y0 = right_x, right_y
    return StageState(
        x_left=left_x,
        y_left=left_y,
        x_right=right_x,
        y_right=right_y,
        current_hitter=server,
        x0=x0,
        y0=y0,
        z0=active_match_config.serve_z0,
        rally_done=False,
        winner=None,
        stage_index=0,
    )


def with_service_court_x_side(
    state: StageState,
    config: SimulationConfig,
    *,
    server_x_side: Side,
) -> StageState:
    if not config.court.lateral_motion_enabled:
        return state
    server_x = service_court_x(server_x_side, config)
    receiver_x, _ = service_receive_position(opponent_side(server_x_side), config)
    if state.current_hitter == "left":
        return replace(state, x_left=server_x, x_right=receiver_x, x0=server_x)
    return replace(state, x_left=receiver_x, x_right=server_x, x0=server_x)


def run_rally(
    env: Badminton1DEnv,
    left_agent: StageAgent,
    right_agent: StageAgent,
    config: SimulationConfig,
    *,
    server: Side,
    match_config: MatchConfig | None = None,
    rally_number: int = 1,
    score_before: MatchScore | None = None,
) -> RallyResult:
    active_match_config = match_config or MatchConfig()
    initial_state = reset_for_serve(server, config, active_match_config)
    initial_state = replace(
        initial_state,
        reaction_time_left=float(left_agent.reaction_time),
        reaction_time_right=float(right_agent.reaction_time),
    )
    env.reset(initial_state)

    records: list[StageRecord] = []
    for _ in range(active_match_config.max_stages_per_rally):
        state = env.state
        if state.rally_done:
            break

        hitter_agent = left_agent if state.current_hitter == "left" else right_agent
        receiver_agent = right_agent if state.current_hitter == "left" else left_agent

        proposed_action = hitter_agent.choose_shot_action(state, config)
        validated = validate_and_clip_shot_action(state, proposed_action, config)
        feasible = feasible_intercept_indices(state, validated.applied, config)
        chosen_index = receiver_agent.choose_intercept_index(state, validated.applied, feasible, config)
        records.append(env.step(proposed_action, chosen_index))

    if not env.state.rally_done:
        raise RuntimeError(
            f"Rally {rally_number} exceeded max_stages_per_rally={active_match_config.max_stages_per_rally} "
            "without a point winner."
        )

    winner = env.state.winner
    if winner is None:
        raise RuntimeError(f"Rally {rally_number} finished without a winner.")

    start_score = score_before or MatchScore()
    return RallyResult(
        rally_number=rally_number,
        server=server,
        score_before=start_score,
        score_after=start_score.award_point(winner),
        initial_state=initial_state,
        records=records,
        winner=winner,
        final_state=env.state,
    )


def run_match(
    left_agent: StageAgent,
    right_agent: StageAgent,
    config: SimulationConfig,
    *,
    match_config: MatchConfig | None = None,
    initial_server: Side = "left",
) -> MatchResult:
    active_match_config = match_config or MatchConfig()
    if active_match_config.target_score <= 0:
        raise ValueError("target_score must be positive")

    env = Badminton1DEnv(config=config)
    score = MatchScore()
    server = initial_server
    rallies: list[RallyResult] = []

    rally_number = 1
    while score.left < active_match_config.target_score and score.right < active_match_config.target_score:
        rally = run_rally(
            env,
            left_agent,
            right_agent,
            config,
            server=server,
            match_config=active_match_config,
            rally_number=rally_number,
            score_before=score,
        )
        rallies.append(rally)
        score = rally.score_after
        server = rally.winner
        rally_number += 1

    winner: Side = "left" if score.left >= active_match_config.target_score else "right"
    return MatchResult(
        rallies=rallies,
        target_score=active_match_config.target_score,
        initial_server=initial_server,
        final_score=score,
        winner=winner,
    )
