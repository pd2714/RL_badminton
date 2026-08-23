from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from badminton.config import SimulationConfig


@dataclass(frozen=True)
class MovementResult:
    position: tuple[float, float]
    velocity: tuple[float, float]
    arrived: bool
    arrival_time: float | None = None
    brake_start_time: float | None = None


def player_velocity_from_state(state, side: str) -> tuple[float, float]:
    if side == "left":
        return float(state.v_x_left), float(state.v_y_left)
    return float(state.v_x_right), float(state.v_y_right)


def earliest_arrival_time(
    start: tuple[float, float],
    velocity: tuple[float, float],
    target: tuple[float, float],
    config: SimulationConfig,
    *,
    reach_radius: float | None = None,
    reach_side: str | None = None,
    target_z: float | None = None,
    reaction_time: float = 0.0,
) -> float:
    reaction = max(float(reaction_time), 0.0)
    reaction_motion = advance_player_during_reaction(start, velocity, reaction, config)
    reaction_position = np.asarray(reaction_motion.position, dtype=float)
    reaction_velocity = np.asarray(reaction_motion.velocity, dtype=float)
    if reach_radius is None:
        motion_target = np.asarray(
            closest_intercept_body_target(reaction_motion.position, target, reach_side, config, target_z=target_z),
            dtype=float,
        )
        delta = motion_target - reaction_position
        distance = float(np.linalg.norm(delta))
    else:
        radius = max(float(reach_radius), 0.0)
        delta = np.asarray(target, dtype=float) - reaction_position
        distance = max(float(np.linalg.norm(delta)) - radius, 0.0)
    if distance <= 1e-9:
        return reaction
    if config.player.movement_model == "constant_velocity":
        return reaction + distance / max(float(config.player.v_max), 1e-9)
    return reaction + _accelerated_travel_time(distance, reaction_velocity, delta, config)


def can_arrive_by_time(
    start: tuple[float, float],
    velocity: tuple[float, float],
    target: tuple[float, float],
    duration: float,
    config: SimulationConfig,
    *,
    reach_radius: float | None = None,
    reach_side: str | None = None,
    target_z: float | None = None,
    reaction_time: float = 0.0,
) -> bool:
    return earliest_arrival_time(
        start,
        velocity,
        target,
        config,
        reach_radius=reach_radius,
        reach_side=reach_side,
        target_z=target_z,
        reaction_time=reaction_time,
    ) <= float(duration) + 1e-9


def horizontal_racket_reach_at_height(target_z: float | None, config: SimulationConfig) -> float:
    reach = max(float(config.player.r_reach), 0.0)
    if target_z is None or reach <= 0.0:
        return reach
    height_above_full_reach = float(target_z) - (float(config.player.z_max) - reach)
    if height_above_full_reach <= 0.0:
        return reach
    if height_above_full_reach >= reach:
        return 0.0
    return float(np.sqrt(max(reach * reach - height_above_full_reach * height_above_full_reach, 0.0)))


def usable_racket_reach(
    start: tuple[float, float],
    target: tuple[float, float],
    side: str | None,
    config: SimulationConfig,
    *,
    target_z: float | None = None,
) -> float:
    horizontal_reach = horizontal_racket_reach_at_height(target_z, config)
    if side is None:
        return horizontal_reach
    delta_y = float(target[1]) - float(start[1])
    if side == "left":
        return horizontal_reach if delta_y >= -1e-9 else 0.0
    if side == "right":
        return horizontal_reach if delta_y <= 1e-9 else 0.0
    raise ValueError(f"Unknown side: {side!r}")


def closest_intercept_body_target(
    start: tuple[float, float],
    target: tuple[float, float],
    side: str | None,
    config: SimulationConfig,
    *,
    target_z: float | None = None,
) -> tuple[float, float]:
    reach = horizontal_racket_reach_at_height(target_z, config)
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    delta_from_target = start_array - target_array
    distance = float(np.linalg.norm(delta_from_target))
    if reach <= 0.0:
        return float(target_array[0]), float(target_array[1])
    if side is None:
        if distance <= reach:
            return float(start_array[0]), float(start_array[1])
        body_target = target_array + delta_from_target * (reach / max(distance, 1e-9))
        return float(body_target[0]), float(body_target[1])
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown side: {side!r}")

    mapped_delta = delta_from_target.copy()
    if side == "right":
        mapped_delta[1] *= -1.0
    mapped_distance = float(np.linalg.norm(mapped_delta))
    if mapped_distance <= reach and mapped_delta[1] <= 1e-9:
        return float(start_array[0]), float(start_array[1])

    if mapped_distance > 1e-9:
        disk_projection = mapped_delta * (reach / mapped_distance)
        if disk_projection[1] <= 1e-9:
            if side == "right":
                disk_projection[1] *= -1.0
            body_target = target_array + disk_projection
            return float(body_target[0]), float(body_target[1])

    mapped_body_delta = np.asarray([np.clip(mapped_delta[0], -reach, reach), 0.0], dtype=float)
    if side == "right":
        mapped_body_delta[1] *= -1.0
    body_target = target_array + mapped_body_delta
    return float(body_target[0]), float(body_target[1])


def intercept_body_target_after_reaction(
    start: tuple[float, float],
    velocity: tuple[float, float],
    target: tuple[float, float],
    side: str,
    config: SimulationConfig,
    *,
    target_z: float | None = None,
    reaction_time: float = 0.0,
) -> tuple[float, float]:
    reaction = max(float(reaction_time), 0.0)
    reaction_motion = advance_player_during_reaction(start, velocity, reaction, config)
    return closest_intercept_body_target(reaction_motion.position, target, side, config, target_z=target_z)


def earliest_stop_arrival_time(
    start: tuple[float, float],
    velocity: tuple[float, float],
    target: tuple[float, float],
    config: SimulationConfig,
    *,
    reaction_time: float = 0.0,
) -> float:
    reaction = max(float(reaction_time), 0.0)
    reaction_motion = advance_player_during_reaction(start, velocity, reaction, config)
    reaction_position = np.asarray(reaction_motion.position, dtype=float)
    reaction_velocity = np.asarray(reaction_motion.velocity, dtype=float)
    delta = np.asarray(target, dtype=float) - reaction_position
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9 and float(np.linalg.norm(reaction_velocity)) <= 1e-9:
        return reaction
    if config.player.movement_model == "constant_velocity":
        return reaction + distance / max(float(config.player.v_max), 1e-9)
    total_time, _ = _accelerated_stop_profile(distance, reaction_velocity, delta, config)
    return reaction + total_time


def brake_start_time(
    start: tuple[float, float],
    velocity: tuple[float, float],
    target: tuple[float, float],
    duration: float,
    config: SimulationConfig,
    *,
    reaction_time: float = 0.0,
) -> float | None:
    arrival = earliest_arrival_time(start, velocity, target, config, reach_radius=0.0, reaction_time=reaction_time)
    stop_arrival = earliest_stop_arrival_time(start, velocity, target, config, reaction_time=reaction_time)
    if stop_arrival > float(duration) + 1e-9:
        return None
    if config.player.movement_model == "constant_velocity":
        return None
    reaction = max(float(reaction_time), 0.0)
    reaction_motion = advance_player_during_reaction(start, velocity, reaction, config)
    reaction_position = np.asarray(reaction_motion.position, dtype=float)
    reaction_velocity = np.asarray(reaction_motion.velocity, dtype=float)
    delta = np.asarray(target, dtype=float) - reaction_position
    _, brake_duration = _accelerated_stop_profile(
        float(np.linalg.norm(delta)),
        reaction_velocity,
        delta,
        config,
    )
    return max(reaction, float(duration) - brake_duration)


def advance_player_during_reaction(
    start: tuple[float, float],
    velocity: tuple[float, float],
    duration: float,
    config: SimulationConfig,
) -> MovementResult:
    """Continue existing motion during reaction time without steering to the new target."""
    start_array = np.asarray(start, dtype=float)
    velocity_array = _clip_speed(np.asarray(velocity, dtype=float), float(config.player.v_max))
    reaction = max(float(duration), 0.0)
    if reaction <= 0.0:
        return MovementResult(
            position=(float(start_array[0]), float(start_array[1])),
            velocity=(float(velocity_array[0]), float(velocity_array[1])),
            arrived=False,
            arrival_time=None,
            brake_start_time=None,
        )

    speed = float(np.linalg.norm(velocity_array))
    if speed <= 1e-12:
        return MovementResult(
            position=(float(start_array[0]), float(start_array[1])),
            velocity=(0.0, 0.0),
            arrived=False,
            arrival_time=None,
            brake_start_time=None,
        )

    direction = velocity_array / speed
    if config.player.movement_model == "constant_velocity":
        displacement = velocity_array * reaction
        end_velocity = velocity_array
    else:
        deceleration = float(config.player.effective_deceleration)
        brake_time = min(reaction, speed / max(deceleration, 1e-9))
        distance = speed * brake_time - 0.5 * deceleration * brake_time * brake_time
        end_speed = max(speed - deceleration * reaction, 0.0)
        displacement = direction * distance
        end_velocity = direction * end_speed

    position = start_array + displacement
    return MovementResult(
        position=(float(position[0]), float(position[1])),
        velocity=(float(end_velocity[0]), float(end_velocity[1])),
        arrived=False,
        arrival_time=None,
        brake_start_time=0.0 if config.player.movement_model == "accelerated" else None,
    )


def advance_player_toward(
    start: tuple[float, float],
    velocity: tuple[float, float],
    target: tuple[float, float],
    duration: float,
    config: SimulationConfig,
    *,
    reaction_time: float = 0.0,
    stop_when_early: bool = True,
) -> MovementResult:
    reaction = max(float(reaction_time), 0.0)
    reaction_motion = advance_player_during_reaction(start, velocity, min(reaction, max(float(duration), 0.0)), config)
    reaction_start = reaction_motion.position
    reaction_velocity = reaction_motion.velocity
    active_duration = max(float(duration) - reaction, 0.0)
    if config.player.movement_model == "constant_velocity":
        return _advance_constant_velocity(reaction_start, target, active_duration, config)

    start_array = np.asarray(reaction_start, dtype=float)
    velocity_array = _clip_speed(np.asarray(reaction_velocity, dtype=float), float(config.player.v_max))
    target_array = np.asarray(target, dtype=float)

    arrival = earliest_arrival_time(start, velocity, target, config, reach_radius=0.0, reaction_time=reaction_time)
    stop_arrival = earliest_stop_arrival_time(start, velocity, target, config, reaction_time=reaction_time)
    if stop_when_early and stop_arrival <= float(duration) + 1e-9:
        brake_at = brake_start_time(start, velocity, target, duration, config, reaction_time=reaction_time)
        return MovementResult(
            position=(float(target_array[0]), float(target_array[1])),
            velocity=(0.0, 0.0),
            arrived=True,
            arrival_time=arrival,
            brake_start_time=brake_at,
        )

    if active_duration <= 0.0:
        return MovementResult(
            position=(float(start_array[0]), float(start_array[1])),
            velocity=(float(velocity_array[0]), float(velocity_array[1])),
            arrived=False,
            arrival_time=arrival,
            brake_start_time=None,
        )

    result = _simulate_seek(start_array, velocity_array, target_array, active_duration, config)
    if arrival <= float(duration) + 1e-9:
        return MovementResult(
            position=(float(target_array[0]), float(target_array[1])),
            velocity=result.velocity,
            arrived=True,
            arrival_time=arrival,
            brake_start_time=None,
        )
    return result


def _advance_constant_velocity(
    start: tuple[float, float],
    target: tuple[float, float],
    duration: float,
    config: SimulationConfig,
) -> MovementResult:
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    delta = target_array - start_array
    distance = float(np.linalg.norm(delta))
    max_distance = max(float(config.player.v_max) * float(duration), 0.0)
    if distance <= max_distance or distance <= 1e-12:
        position = target_array
        arrived = True
    else:
        position = start_array + delta * (max_distance / distance)
        arrived = False
    return MovementResult(
        position=(float(position[0]), float(position[1])),
        velocity=(0.0, 0.0),
        arrived=arrived,
        arrival_time=0.0 if distance <= 1e-12 else distance / max(float(config.player.v_max), 1e-9),
        brake_start_time=None,
    )


def _accelerated_travel_time(
    distance: float,
    velocity: np.ndarray,
    delta: np.ndarray,
    config: SimulationConfig,
) -> float:
    if distance <= 1e-9:
        return 0.0
    direction = delta / max(float(np.linalg.norm(delta)), 1e-9)
    v_parallel = float(np.dot(velocity, direction))
    v_perp = float(np.linalg.norm(velocity - v_parallel * direction))
    a = float(config.player.acceleration)
    v_max = float(config.player.v_max)
    penalty = v_perp / a
    if v_parallel < 0.0:
        distance += (v_parallel * v_parallel) / (2.0 * a)
        penalty += -v_parallel / a
        v_parallel = 0.0
    v_parallel = min(v_parallel, v_max)
    accel_distance = max((v_max * v_max - v_parallel * v_parallel) / (2.0 * a), 0.0)
    if distance <= accel_distance:
        return penalty + max((-v_parallel + np.sqrt(max(v_parallel * v_parallel + 2.0 * a * distance, 0.0))) / a, 0.0)
    return penalty + (v_max - v_parallel) / a + (distance - accel_distance) / v_max


def _accelerated_stop_profile(
    distance: float,
    velocity: np.ndarray,
    delta: np.ndarray,
    config: SimulationConfig,
) -> tuple[float, float]:
    b = float(config.player.effective_deceleration)
    if distance <= 1e-9:
        brake_duration = float(np.linalg.norm(velocity)) / b
        return brake_duration, brake_duration
    direction = delta / max(float(np.linalg.norm(delta)), 1e-9)
    v_parallel = float(np.dot(velocity, direction))
    v_perp = float(np.linalg.norm(velocity - v_parallel * direction))
    a = float(config.player.acceleration)
    v_max = float(config.player.v_max)
    penalty = v_perp / b
    if v_parallel < 0.0:
        distance += (v_parallel * v_parallel) / (2.0 * b)
        penalty += -v_parallel / b
        v_parallel = 0.0
    v_parallel = min(v_parallel, v_max)

    peak_sq = max((2.0 * a * b * distance + b * v_parallel * v_parallel) / (a + b), 0.0)
    peak = float(np.sqrt(peak_sq))
    if peak <= v_max:
        brake_duration = peak / b
        return penalty + max((peak - v_parallel) / a, 0.0) + brake_duration, brake_duration

    accel_distance = max((v_max * v_max - v_parallel * v_parallel) / (2.0 * a), 0.0)
    brake_distance = v_max * v_max / (2.0 * b)
    cruise_distance = max(distance - accel_distance - brake_distance, 0.0)
    brake_duration = v_max / b
    return penalty + max((v_max - v_parallel) / a, 0.0) + cruise_distance / v_max + brake_duration, brake_duration


def _simulate_seek(
    position: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    duration: float,
    config: SimulationConfig,
) -> MovementResult:
    remaining = float(duration)
    dt_base = 0.01
    a = float(config.player.acceleration)
    b = float(config.player.effective_deceleration)
    v_max = float(config.player.v_max)
    while remaining > 1e-9:
        dt = min(dt_base, remaining)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        speed = float(np.linalg.norm(velocity))
        if distance <= 1e-9 and speed <= b * dt:
            velocity = np.zeros(2, dtype=float)
            position = target.copy()
            remaining -= dt
            continue
        if distance <= 1e-9:
            accel = -velocity / max(speed, 1e-9) * b
        else:
            direction = delta / distance
            closing_speed = max(float(np.dot(velocity, direction)), 0.0)
            stopping_distance = closing_speed * closing_speed / (2.0 * b)
            if speed > 1e-9 and distance <= stopping_distance:
                accel = -velocity / speed * b
            else:
                desired_velocity = direction * v_max
                steer = desired_velocity - velocity
                steer_norm = float(np.linalg.norm(steer))
                accel = np.zeros(2, dtype=float) if steer_norm <= 1e-9 else steer / steer_norm * a
        velocity = _clip_speed(velocity + accel * dt, v_max)
        position = position + velocity * dt
        remaining -= dt
    arrived = float(np.linalg.norm(target - position)) <= float(config.player.r_reach)
    return MovementResult(
        position=(float(position[0]), float(position[1])),
        velocity=(float(velocity[0]), float(velocity[1])),
        arrived=arrived,
        arrival_time=None,
        brake_start_time=None,
    )


def _clip_speed(velocity: np.ndarray, v_max: float) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    if speed <= v_max or speed <= 1e-12:
        return velocity
    return velocity * (v_max / speed)
