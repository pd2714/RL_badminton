from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from badminton1d.config import PlayerConfig, SimulationConfig
from badminton1d.dynamics import effective_flight_time, landing_position, reaction_time_for_side
from badminton1d.match import MatchResult
from badminton1d.movement import advance_player_toward, intercept_body_target_after_reaction
from badminton1d.state import Side, StageRecord
from badminton1d.trajectory import position_at_time
from badminton1d.utils import player_position, player_velocity


@dataclass(frozen=True)
class StageTrace:
    stage_index: int
    hitter_side: Side
    receiver_side: Side
    shuttle_start: tuple[float, float, float]
    shuttle_velocity: tuple[float, float, float]
    shuttle_landing: tuple[float, float]
    total_flight_time: float
    playback_duration: float
    end_time: float
    intercepted: bool
    left_start: tuple[float, float]
    right_start: tuple[float, float]
    left_end: tuple[float, float]
    right_end: tuple[float, float]
    hitter_start: tuple[float, float]
    hitter_end: tuple[float, float]
    receiver_start: tuple[float, float]
    receiver_end: tuple[float, float]
    receiver_reaction_time: float
    recovery_target: tuple[float, float]
    intercept_point: tuple[float, float, float] | None
    intended_intercept_time: float | None
    intended_intercept_point: tuple[float, float, float] | None
    trajectory_mode: str
    gravity: float
    drag_coefficient: float
    drag_dt: float
    terminal: bool
    winner: Side | None
    terminal_reason: str | None
    horizontal_drag_coefficient: float | None = None
    vertical_drag_coefficient: float | None = None
    left_start_velocity: tuple[float, float] = (0.0, 0.0)
    right_start_velocity: tuple[float, float] = (0.0, 0.0)
    player_v_max: float = PlayerConfig().v_max
    player_acceleration: float = PlayerConfig().acceleration
    player_deceleration: float | None = None
    player_movement_model: str = "accelerated"
    player_r_reach: float = PlayerConfig().r_reach


@dataclass(frozen=True)
class RallyTrace:
    stages: list[StageTrace]
    rally_done: bool
    winner: Side | None
    total_playback_time: float
    rally_number: int | None = None
    server: Side | None = None
    score_before_left: int = 0
    score_before_right: int = 0
    score_after_left: int = 0
    score_after_right: int = 0
    pause_duration: float = 0.0
    match_winner: Side | None = None


@dataclass(frozen=True)
class MatchTrace:
    rallies: list[RallyTrace]
    target_score: int
    score_left: int
    score_right: int
    winner: Side | None
    total_playback_time: float


@dataclass(frozen=True)
class FrameSnapshot:
    stage_index: int
    hitter_side: Side
    local_time: float
    playback_duration: float
    time_ratio: float
    shuttle_position: tuple[float, float, float]
    left_player_position: tuple[float, float]
    right_player_position: tuple[float, float]
    recovery_target: tuple[float, float]
    intercept_point: tuple[float, float, float] | None
    intended_intercept_time: float | None
    intended_intercept_point: tuple[float, float, float] | None
    terminal: bool
    winner: Side | None


def _is_valid_intercept(record: StageRecord) -> bool:
    return record.chosen_index in record.feasible_indices and record.intercept_point is not None and record.chosen_time is not None


def _receiver_attempt_end_position(record: StageRecord, config: SimulationConfig) -> tuple[float, float]:
    receiver_start = player_position(record.state_before, record.receiver_side)
    receiver_speed = player_velocity(record.state_before, record.receiver_side)
    receiver_reaction_time = reaction_time_for_side(record.state_before, record.receiver_side)
    if record.terminal_reason == "opponent_no_valid_shot":
        return receiver_start
    if _is_valid_intercept(record):
        assert record.intercept_point is not None
        assert record.chosen_time is not None
        receiver_target = intercept_body_target_after_reaction(
            receiver_start,
            receiver_speed,
            (float(record.intercept_point[0]), float(record.intercept_point[1])),
            record.receiver_side,
            config,
            target_z=float(record.intercept_point[2]),
            reaction_time=receiver_reaction_time,
        )
        receiver_motion = advance_player_toward(
            receiver_start,
            receiver_speed,
            receiver_target,
            float(record.chosen_time),
            config,
            reaction_time=receiver_reaction_time,
            stop_when_early=True,
        )
        return receiver_motion.position
    action = record.validated_action.applied
    total_flight_time = effective_flight_time(record.state_before, action, config)
    landing_xy = landing_position(record.state_before, action, config)
    target_point = record.intended_intercept_point or record.intercept_point
    if target_point is None:
        chased = advance_player_toward(
            receiver_start,
            receiver_speed,
            landing_xy,
            total_flight_time,
            config,
            reaction_time=receiver_reaction_time,
            stop_when_early=False,
        )
        return chased.position
    chased = advance_player_toward(
        receiver_start,
        receiver_speed,
        (float(target_point[0]), float(target_point[1])),
        total_flight_time,
        config,
        reaction_time=receiver_reaction_time,
        stop_when_early=False,
    )
    return chased.position


def build_rally_trace(records: list[StageRecord], config: SimulationConfig) -> RallyTrace:
    stages: list[StageTrace] = []
    total_playback_time = 0.0

    for record in records:
        state = record.state_before
        action = record.validated_action.applied
        intercepted = _is_valid_intercept(record)
        if record.terminal_reason == "opponent_no_valid_shot":
            total_flight_time = 0.0
            playback_duration = 0.0
            end_time = 0.0
            shuttle_landing = (state.x0, state.y0)
        else:
            total_flight_time = effective_flight_time(state, action, config)
            playback_duration = float(record.chosen_time) if intercepted else total_flight_time
            end_time = float(record.chosen_time) if intercepted else total_flight_time
            shuttle_landing = landing_position(state, action, config)

        hitter_start = player_position(state, state.current_hitter)
        receiver_start = player_position(state, record.receiver_side)
        receiver_reaction_time = reaction_time_for_side(state, record.receiver_side)
        hitter_motion = advance_player_toward(
            hitter_start,
            player_velocity(state, state.current_hitter),
            (action.x_rec, action.y_rec),
            playback_duration,
            config,
            stop_when_early=True,
        )
        hitter_end = hitter_motion.position
        receiver_end = _receiver_attempt_end_position(record, config)

        if record.receiver_side == "left":
            left_end = receiver_end
            right_end = hitter_end
        else:
            left_end = hitter_end
            right_end = receiver_end

        stages.append(
            StageTrace(
                stage_index=record.stage_index,
                hitter_side=state.current_hitter,
                receiver_side=record.receiver_side,
                shuttle_start=(state.x0, state.y0, state.z0),
                shuttle_velocity=(action.v_x, action.v_y, action.v_z),
                shuttle_landing=shuttle_landing,
                total_flight_time=total_flight_time,
                playback_duration=playback_duration,
                end_time=end_time,
                intercepted=intercepted,
                left_start=(state.x_left, state.y_left),
                right_start=(state.x_right, state.y_right),
                left_end=left_end,
                right_end=right_end,
                left_start_velocity=player_velocity(state, "left"),
                right_start_velocity=player_velocity(state, "right"),
                hitter_start=hitter_start,
                hitter_end=hitter_end,
                receiver_start=receiver_start,
                receiver_end=receiver_end,
                receiver_reaction_time=receiver_reaction_time,
                recovery_target=(action.x_rec, action.y_rec),
                intercept_point=record.intercept_point,
                intended_intercept_time=record.intended_intercept_time,
                intended_intercept_point=record.intended_intercept_point,
                trajectory_mode=config.action.trajectory_mode,
                gravity=config.action.gravity,
                drag_coefficient=config.action.drag_coefficient,
                drag_dt=config.action.drag_dt,
                terminal=record.next_state.rally_done,
                winner=record.next_state.winner,
                terminal_reason=record.terminal_reason,
                horizontal_drag_coefficient=config.action.horizontal_drag_coefficient,
                vertical_drag_coefficient=config.action.vertical_drag_coefficient,
                player_v_max=config.player.v_max,
                player_acceleration=config.player.acceleration,
                player_deceleration=config.player.deceleration,
                player_movement_model=config.player.movement_model,
                player_r_reach=config.player.r_reach,
            )
        )
        total_playback_time += playback_duration

    final_record = records[-1] if records else None
    return RallyTrace(
        stages=stages,
        rally_done=bool(final_record and final_record.next_state.rally_done),
        winner=final_record.next_state.winner if final_record else None,
        total_playback_time=total_playback_time,
    )


def build_match_trace(
    match_result: MatchResult,
    config: SimulationConfig,
    *,
    rally_pause: float = 0.6,
) -> MatchTrace:
    if rally_pause < 0.0:
        raise ValueError("rally_pause must be zero or greater")

    rallies: list[RallyTrace] = []
    total_playback_time = 0.0

    for rally_result in match_result.rallies:
        rally_trace = build_rally_trace(rally_result.records, config)
        annotated_trace = RallyTrace(
            stages=rally_trace.stages,
            rally_done=rally_trace.rally_done,
            winner=rally_trace.winner,
            total_playback_time=rally_trace.total_playback_time,
            rally_number=rally_result.rally_number,
            server=rally_result.server,
            score_before_left=rally_result.score_before.left,
            score_before_right=rally_result.score_before.right,
            score_after_left=rally_result.score_after.left,
            score_after_right=rally_result.score_after.right,
            pause_duration=rally_pause,
            match_winner=match_result.winner if rally_result.winner == match_result.winner and (
                rally_result.score_after.left >= match_result.target_score
                or rally_result.score_after.right >= match_result.target_score
            ) else None,
        )
        rallies.append(annotated_trace)
        total_playback_time += annotated_trace.total_playback_time + annotated_trace.pause_duration

    return MatchTrace(
        rallies=rallies,
        target_score=match_result.target_score,
        score_left=match_result.final_score.left,
        score_right=match_result.final_score.right,
        winner=match_result.winner,
        total_playback_time=total_playback_time,
    )


def _movement_config_for_stage(stage: StageTrace, config: SimulationConfig | None) -> SimulationConfig:
    base_config = SimulationConfig() if config is None else config
    return replace(
        base_config,
        player=PlayerConfig(
            v_max=stage.player_v_max,
            acceleration=stage.player_acceleration,
            deceleration=stage.player_deceleration,
            movement_model=stage.player_movement_model,
            r_reach=stage.player_r_reach,
            z_min=base_config.player.z_min,
            z_max=base_config.player.z_max,
            marker_radius=base_config.player.marker_radius,
        ),
    )


def _receiver_target_and_stop(
    stage: StageTrace,
    config: SimulationConfig,
    reaction_time: float,
) -> tuple[tuple[float, float], bool]:
    if stage.terminal_reason == "opponent_no_valid_shot":
        return stage.receiver_start, True
    if stage.intercepted and stage.intercept_point is not None:
        if stage.receiver_side == "left":
            receiver_velocity = stage.left_start_velocity
        else:
            receiver_velocity = stage.right_start_velocity
        target = intercept_body_target_after_reaction(
            stage.receiver_start,
            receiver_velocity,
            (float(stage.intercept_point[0]), float(stage.intercept_point[1])),
            stage.receiver_side,
            config,
            target_z=float(stage.intercept_point[2]),
            reaction_time=reaction_time,
        )
        return target, True
    target_point = stage.intended_intercept_point or stage.intercept_point
    if target_point is None:
        return stage.shuttle_landing, False
    return (float(target_point[0]), float(target_point[1])), False


def _player_position_at_time(
    stage: StageTrace,
    side: Side,
    local_time: float,
    config: SimulationConfig,
    *,
    apply_receiver_reaction_delay: bool,
) -> tuple[float, float]:
    duration = min(max(float(local_time), 0.0), max(float(stage.playback_duration), 0.0))
    if side == "left":
        start = stage.left_start
        velocity = stage.left_start_velocity
    else:
        start = stage.right_start
        velocity = stage.right_start_velocity

    if side == stage.hitter_side:
        target = stage.recovery_target
        reaction_time = 0.0
        stop_when_early = True
    else:
        reaction_time = stage.receiver_reaction_time if apply_receiver_reaction_delay else 0.0
        target, stop_when_early = _receiver_target_and_stop(stage, config, reaction_time)

    motion = advance_player_toward(
        start,
        velocity,
        target,
        duration,
        config,
        reaction_time=reaction_time,
        stop_when_early=stop_when_early,
    )
    return motion.position


def interpolate_stage(
    stage: StageTrace,
    local_time: float,
    *,
    apply_receiver_reaction_delay: bool = True,
    config: SimulationConfig | None = None,
) -> FrameSnapshot:
    if stage.playback_duration <= 0.0:
        ratio = 1.0
    else:
        clamped_time = min(max(local_time, 0.0), stage.playback_duration)
        ratio = clamped_time / stage.playback_duration

    time_now = stage.end_time * ratio
    clamped_local_time = min(max(local_time, 0.0), stage.playback_duration)
    local_config = SimulationConfig()
    local_config = SimulationConfig(
        court=local_config.court,
        player=local_config.player,
        render=local_config.render,
        action=replace(
            local_config.action,
            trajectory_mode=stage.trajectory_mode,
            gravity=stage.gravity,
            drag_coefficient=stage.drag_coefficient,
            horizontal_drag_coefficient=stage.horizontal_drag_coefficient,
            vertical_drag_coefficient=stage.vertical_drag_coefficient,
            drag_dt=stage.drag_dt,
        ),
    )
    movement_config = _movement_config_for_stage(stage, config)

    shuttle_x, shuttle_y, shuttle_z = position_at_time(
        stage.shuttle_start[0],
        stage.shuttle_start[1],
        stage.shuttle_start[2],
        stage.shuttle_velocity[0],
        stage.shuttle_velocity[1],
        stage.shuttle_velocity[2],
        time_now,
        local_config,
    )

    return FrameSnapshot(
        stage_index=stage.stage_index,
        hitter_side=stage.hitter_side,
        local_time=clamped_local_time,
        playback_duration=stage.playback_duration,
        time_ratio=ratio,
        shuttle_position=(shuttle_x, shuttle_y, shuttle_z),
        left_player_position=_player_position_at_time(
            stage,
            "left",
            clamped_local_time,
            movement_config,
            apply_receiver_reaction_delay=apply_receiver_reaction_delay,
        ),
        right_player_position=_player_position_at_time(
            stage,
            "right",
            clamped_local_time,
            movement_config,
            apply_receiver_reaction_delay=apply_receiver_reaction_delay,
        ),
        recovery_target=stage.recovery_target,
        intercept_point=stage.intercept_point,
        intended_intercept_time=stage.intended_intercept_time,
        intended_intercept_point=stage.intended_intercept_point,
        terminal=stage.terminal,
        winner=stage.winner,
    )


def rally_trace_to_dict(trace: RallyTrace) -> dict[str, object]:
    payload = asdict(trace)
    payload["stage_count"] = len(trace.stages)
    return payload


def match_trace_to_dict(trace: MatchTrace) -> dict[str, object]:
    payload = asdict(trace)
    payload["rally_count"] = len(trace.rallies)
    payload["stage_count"] = sum(len(rally.stages) for rally in trace.rallies)
    return payload


def stage_trace_from_dict(payload: dict[str, object]) -> StageTrace:
    trace_payload = dict(payload)
    trace_payload.setdefault("horizontal_drag_coefficient", None)
    trace_payload.setdefault("vertical_drag_coefficient", None)
    trace_payload.setdefault("intended_intercept_time", None)
    trace_payload.setdefault("intended_intercept_point", None)
    trace_payload.setdefault("receiver_reaction_time", 0.0)
    trace_payload.setdefault("left_start_velocity", (0.0, 0.0))
    trace_payload.setdefault("right_start_velocity", (0.0, 0.0))
    trace_payload.setdefault("player_v_max", PlayerConfig().v_max)
    trace_payload.setdefault("player_acceleration", PlayerConfig().acceleration)
    trace_payload.setdefault("player_deceleration", None)
    trace_payload.setdefault("player_movement_model", "accelerated")
    trace_payload.setdefault("player_r_reach", PlayerConfig().r_reach)
    return StageTrace(**trace_payload)


def rally_trace_from_dict(payload: dict[str, object]) -> RallyTrace:
    trace_payload = dict(payload)
    trace_payload.pop("stage_count", None)
    stages = [stage_trace_from_dict(stage) for stage in trace_payload.pop("stages", [])]
    return RallyTrace(stages=stages, **trace_payload)


def match_trace_from_dict(payload: dict[str, object]) -> MatchTrace:
    trace_payload = dict(payload)
    trace_payload.pop("rally_count", None)
    trace_payload.pop("stage_count", None)
    rallies = [rally_trace_from_dict(rally) for rally in trace_payload.pop("rallies", [])]
    return MatchTrace(rallies=rallies, **trace_payload)
