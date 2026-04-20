from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from badminton1d.config import SimulationConfig


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float
    z: float
    v_x: float
    v_y: float
    v_z: float


@dataclass(frozen=True)
class NetCrossing:
    t: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TrajectoryResult:
    mode: str
    samples: tuple[TrajectoryPoint, ...]
    landing_time: float
    landing_x: float
    landing_y: float
    net_crossing: NetCrossing | None


def ballistic_position(
    x0: float,
    y0: float,
    z0: float,
    v_x: float,
    v_y: float,
    v_z: float,
    t: float,
    g: float = 9.81,
) -> tuple[float, float, float]:
    x = x0 + v_x * t
    y = y0 + v_y * t
    z = z0 + v_z * t - 0.5 * g * t * t
    return x, y, z


def ballistic_landing_time(z0: float, v_z: float, g: float = 9.81) -> float:
    return float((v_z + np.sqrt(v_z * v_z + 2.0 * g * z0)) / g)


def ballistic_landing_point(
    x0: float,
    y0: float,
    z0: float,
    v_x: float,
    v_y: float,
    v_z: float,
    g: float = 9.81,
) -> tuple[float, float]:
    landing_t = ballistic_landing_time(z0, v_z, g)
    return float(x0 + v_x * landing_t), float(y0 + v_y * landing_t)


def ballistic_net_crossing(
    x0: float,
    y0: float,
    z0: float,
    v_x: float,
    v_y: float,
    v_z: float,
    net_y: float,
    *,
    g: float = 9.81,
) -> NetCrossing | None:
    if np.isclose(v_y, 0.0):
        return None
    t_net = (net_y - y0) / v_y
    if t_net <= 0.0:
        return None
    x_net, y_net, z_net = ballistic_position(x0, y0, z0, v_x, v_y, v_z, t_net, g)
    return NetCrossing(t=t_net, x=float(x_net), y=float(y_net), z=float(z_net))


def simulate_drag_trajectory(
    x0: float,
    y0: float,
    z0: float,
    v_x: float,
    v_y: float,
    v_z: float,
    *,
    g: float = 9.81,
    c: float | None = None,
    kh: float | None = None,
    kv: float | None = None,
    dt: float = 0.01,
    net_y: float = 0.0,
    max_time: float = 10.0,
) -> TrajectoryResult:
    horizontal_drag = float(c if kh is None and c is not None else (0.0 if kh is None else kh))
    vertical_drag = float(c if kv is None and c is not None else (0.0 if kv is None else kv))
    t = 0.0
    x = x0
    y = y0
    z = z0
    vx_now = v_x
    vy_now = v_y
    vz_now = v_z
    samples = [TrajectoryPoint(t=t, x=x, y=y, z=z, v_x=vx_now, v_y=vy_now, v_z=vz_now)]
    net_crossing: NetCrossing | None = None

    while t < max_time:
        prev_t = t
        prev_x = x
        prev_y = y
        prev_z = z
        prev_vx = vx_now
        prev_vy = vy_now
        prev_vz = vz_now

        x = x + vx_now * dt
        y = y + vy_now * dt
        z = z + vz_now * dt
        speed = float(np.sqrt(vx_now * vx_now + vy_now * vy_now + vz_now * vz_now))
        vx_now = vx_now + (-horizontal_drag * speed * vx_now) * dt
        vy_now = vy_now + (-horizontal_drag * speed * vy_now) * dt
        vz_now = vz_now + (-g - vertical_drag * speed * vz_now) * dt
        t = t + dt

        if net_crossing is None and (prev_y - net_y) * (y - net_y) <= 0.0 and not np.isclose(prev_y, y):
            ratio = (net_y - prev_y) / (y - prev_y)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            net_crossing = NetCrossing(
                t=prev_t + ratio * (t - prev_t),
                x=prev_x + ratio * (x - prev_x),
                y=net_y,
                z=prev_z + ratio * (z - prev_z),
            )

        if z <= 0.0:
            if np.isclose(prev_z, z):
                ratio = 1.0
            else:
                ratio = prev_z / (prev_z - z)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            landing_t = prev_t + ratio * (t - prev_t)
            landing_x = prev_x + ratio * (x - prev_x)
            landing_y = prev_y + ratio * (y - prev_y)
            landing_vx = prev_vx + ratio * (vx_now - prev_vx)
            landing_vy = prev_vy + ratio * (vy_now - prev_vy)
            landing_vz = prev_vz + ratio * (vz_now - prev_vz)
            samples.append(
                TrajectoryPoint(
                    t=landing_t,
                    x=landing_x,
                    y=landing_y,
                    z=0.0,
                    v_x=landing_vx,
                    v_y=landing_vy,
                    v_z=landing_vz,
                )
            )
            return TrajectoryResult(
                mode="drag_square",
                samples=tuple(samples),
                landing_time=float(landing_t),
                landing_x=float(landing_x),
                landing_y=float(landing_y),
                net_crossing=net_crossing,
            )

        samples.append(TrajectoryPoint(t=t, x=x, y=y, z=z, v_x=vx_now, v_y=vy_now, v_z=vz_now))

    raise RuntimeError("Drag trajectory did not land within max_time.")


def build_ballistic_trajectory(
    x0: float,
    y0: float,
    z0: float,
    v_x: float,
    v_y: float,
    v_z: float,
    *,
    config: SimulationConfig,
    sample_count: int | None = None,
) -> TrajectoryResult:
    landing_time = ballistic_landing_time(z0, v_z, config.action.gravity)
    landing_x, landing_y = ballistic_landing_point(x0, y0, z0, v_x, v_y, v_z, config.action.gravity)
    count = max(sample_count or config.render.trajectory_samples, 2)
    times = np.linspace(0.0, landing_time, count)
    samples = []
    for t in times:
        x, y, z = ballistic_position(x0, y0, z0, v_x, v_y, v_z, float(t), config.action.gravity)
        samples.append(
            TrajectoryPoint(
                t=float(t),
                x=float(x),
                y=float(y),
                z=max(float(z), 0.0),
                v_x=float(v_x),
                v_y=float(v_y),
                v_z=float(v_z - config.action.gravity * t),
            )
        )

    net_crossing = ballistic_net_crossing(
        x0,
        y0,
        z0,
        v_x,
        v_y,
        v_z,
        config.court.net_y,
        g=config.action.gravity,
    )
    if net_crossing is not None and not (0.0 < net_crossing.t < landing_time):
        net_crossing = None

    return TrajectoryResult(
        mode="ballistic",
        samples=tuple(samples),
        landing_time=float(landing_time),
        landing_x=landing_x,
        landing_y=landing_y,
        net_crossing=net_crossing,
    )


def simulate_trajectory(
    x0: float,
    y0: float,
    z0: float,
    v_x: float,
    v_y: float,
    v_z: float,
    config: SimulationConfig,
    *,
    sample_count: int | None = None,
) -> TrajectoryResult:
    if config.action.uses_square_drag:
        return simulate_drag_trajectory(
            x0,
            y0,
            z0,
            v_x,
            v_y,
            v_z,
            g=config.action.gravity,
            kh=config.action.effective_horizontal_drag_coefficient,
            kv=config.action.effective_vertical_drag_coefficient,
            dt=config.action.drag_dt,
            net_y=config.court.net_y,
        )
    return build_ballistic_trajectory(
        x0,
        y0,
        z0,
        v_x,
        v_y,
        v_z,
        config=config,
        sample_count=sample_count,
    )


def position_at_time(
    x0: float,
    y0: float,
    z0: float,
    v_x: float,
    v_y: float,
    v_z: float,
    t: float,
    config: SimulationConfig,
    *,
    trajectory: TrajectoryResult | None = None,
) -> tuple[float, float, float]:
    if config.action.effective_trajectory_mode == "ballistic":
        return ballistic_position(x0, y0, z0, v_x, v_y, v_z, t, config.action.gravity)

    active = trajectory or simulate_trajectory(x0, y0, z0, v_x, v_y, v_z, config)
    if t <= 0.0:
        first = active.samples[0]
        return first.x, first.y, first.z
    if t >= active.landing_time:
        return active.landing_x, active.landing_y, 0.0

    points = active.samples
    for left, right in zip(points[:-1], points[1:]):
        if left.t <= t <= right.t:
            if np.isclose(left.t, right.t):
                return right.x, right.y, right.z
            ratio = (t - left.t) / (right.t - left.t)
            return (
                float(left.x + ratio * (right.x - left.x)),
                float(left.y + ratio * (right.y - left.y)),
                float(left.z + ratio * (right.z - left.z)),
            )
    last = active.samples[-1]
    return last.x, last.y, last.z
