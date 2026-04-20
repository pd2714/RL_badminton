from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import effective_flight_time, landing_position
from badminton1d.match import MatchResult
from badminton1d.state import Side, StageRecord
from badminton1d.trajectory import position_at_time
from badminton1d.utils import move_toward, player_position


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


def _lerp(start: float, end: float, ratio: float) -> float:
    return start + ratio * (end - start)


def _lerp_point(start: tuple[float, float], end: tuple[float, float], ratio: float) -> tuple[float, float]:
    return (_lerp(start[0], end[0], ratio), _lerp(start[1], end[1], ratio))


def _is_valid_intercept(record: StageRecord) -> bool:
    return record.chosen_index in record.feasible_indices and record.intercept_point is not None and record.chosen_time is not None


def _receiver_attempt_end_position(record: StageRecord, config: SimulationConfig) -> tuple[float, float]:
    receiver_start = player_position(record.state_before, record.receiver_side)
    if record.terminal_reason == "opponent_no_valid_shot":
        return receiver_start
    if _is_valid_intercept(record):
        assert record.intercept_point is not None
        return float(record.intercept_point[0]), float(record.intercept_point[1])
    action = record.validated_action.applied
    total_flight_time = effective_flight_time(record.state_before, action, config)
    landing_xy = landing_position(record.state_before, action, config)
    target_point = record.intended_intercept_point or record.intercept_point
    if target_point is None:
        chased = move_toward(receiver_start, landing_xy, config.player.v_max * total_flight_time)
        assert isinstance(chased, tuple)
        return chased
    chased = move_toward(
        receiver_start,
        (float(target_point[0]), float(target_point[1])),
        config.player.v_max * total_flight_time,
    )
    assert isinstance(chased, tuple)
    return chased


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
        hitter_end = move_toward(hitter_start, (action.x_rec, action.y_rec), config.player.v_max * playback_duration)
        assert isinstance(hitter_end, tuple)
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
                hitter_start=hitter_start,
                hitter_end=hitter_end,
                receiver_start=receiver_start,
                receiver_end=receiver_end,
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


def interpolate_stage(stage: StageTrace, local_time: float) -> FrameSnapshot:
    if stage.playback_duration <= 0.0:
        ratio = 1.0
    else:
        clamped_time = min(max(local_time, 0.0), stage.playback_duration)
        ratio = clamped_time / stage.playback_duration

    time_now = stage.end_time * ratio
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
        local_time=min(max(local_time, 0.0), stage.playback_duration),
        playback_duration=stage.playback_duration,
        time_ratio=ratio,
        shuttle_position=(shuttle_x, shuttle_y, shuttle_z),
        left_player_position=_lerp_point(stage.left_start, stage.left_end, ratio),
        right_player_position=_lerp_point(stage.right_start, stage.right_end, ratio),
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
