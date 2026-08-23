from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from badminton.config import SimulationConfig
from badminton.dynamics import trajectory_result, valid_hitter_action
from badminton.state import ShotAction, StageState
from badminton.shot_generators.shot_naming import infer_shot_name
from badminton.shot_generators.tactic_lookup_common import (
    ANGLE_BIN_NAMES,
    SHOT_NAME_ORDER,
    TacticAction2D,
    TacticRuntimeConfig,
    angle_bin_centers_deg,
    canonicalize_state,
    contact_height_centers,
    contact_x_centers,
    contact_y_centers,
    landing_col_centers,
    landing_row_centers,
    landing_zone_names_2d,
    nearest_bin,
    power_speed_targets_2d,
    repo_root,
    resolve_lookup_dir,
    restore_velocity_from_canonical,
)

LOOKUP_FILENAME = "tactic_2d_lookup.npz"


@dataclass(frozen=True)
class LookupEntry2D:
    velocity: tuple[float, float, float]
    valid: bool
    fallback_used: bool
    landing_position: tuple[float, float]
    net_crossing_height: float | None
    flight_time: float
    score: float
    inferred_shot_name: str
    contact_bins: tuple[int, int, int]


class TacticLookup2D:
    def __init__(
        self,
        config: SimulationConfig,
        runtime_config: TacticRuntimeConfig | None = None,
    ) -> None:
        self.config = config
        self.runtime_config = runtime_config or TacticRuntimeConfig()
        self.lookup_dir = resolve_lookup_dir(self.runtime_config)
        self.lookup_path = self.lookup_dir / LOOKUP_FILENAME
        self.contact_x_centers = contact_x_centers(config)
        self.contact_y_centers = contact_y_centers(config)
        self.contact_height_centers = contact_height_centers(config)
        self.landing_row_centers = landing_row_centers(config)
        self.landing_col_centers = landing_col_centers(config)
        self.zone_names = landing_zone_names_2d()
        self.angle_names = ANGLE_BIN_NAMES
        self.power_names = ("soft", "normal", "hard")
        self._loaded = False

    @property
    def action_count(self) -> int:
        return 3 * 3 * 5 * 3

    @property
    def contact_shape(self) -> tuple[int, int, int]:
        return (
            len(self.contact_x_centers),
            len(self.contact_y_centers),
            len(self.contact_height_centers),
        )

    @property
    def table_shape(self) -> tuple[int, ...]:
        return self.contact_shape + (3, 3, 5, 3)

    def flat_to_action(self, flat_index: int) -> TacticAction2D:
        bounded = int(flat_index) % self.action_count
        power_bin = bounded % 3
        bounded //= 3
        angle_bin = bounded % 5
        bounded //= 5
        landing_col = bounded % 3
        landing_row = bounded // 3
        return TacticAction2D(
            landing_row=int(landing_row),
            landing_col=int(landing_col),
            angle_bin=int(angle_bin),
            power_bin=int(power_bin),
        )

    def action_to_flat(self, action: TacticAction2D) -> int:
        return ((((int(action.landing_row) * 3) + int(action.landing_col)) * 5) + int(action.angle_bin)) * 3 + int(action.power_bin)

    def ensure_loaded(self) -> None:
        if self._loaded and not self.runtime_config.regenerate_lookup_table:
            return
        if self.runtime_config.regenerate_lookup_table or not self.lookup_path.exists():
            self.build()
        else:
            try:
                self._load()
            except ValueError:
                self.build()

    def build(self) -> None:
        self.lookup_dir.mkdir(parents=True, exist_ok=True)
        shape = self.table_shape
        velocities = np.zeros(shape + (3,), dtype=np.float32)
        valid = np.zeros(shape, dtype=bool)
        fallback_used = np.zeros(shape, dtype=bool)
        landing_x = np.zeros(shape, dtype=np.float32)
        landing_y = np.zeros(shape, dtype=np.float32)
        net_crossing_height = np.full(shape, np.nan, dtype=np.float32)
        flight_time = np.zeros(shape, dtype=np.float32)
        score = np.full(shape, -1e9, dtype=np.float32)
        shot_name_index = np.full(shape, SHOT_NAME_ORDER.index("generic"), dtype=np.int16)

        for contact_x_bin, x0 in enumerate(self.contact_x_centers):
            for contact_y_bin, y0 in enumerate(self.contact_y_centers):
                for contact_height_bin, z0 in enumerate(self.contact_height_centers):
                    state = self._state_for_contact(x0, y0, z0)
                    for landing_row, landing_y_center in enumerate(self.landing_row_centers):
                        for landing_col, landing_x_center in enumerate(self.landing_col_centers):
                            for angle_bin in range(5):
                                for power_bin in range(3):
                                    best = self._best_entry(
                                        state=state,
                                        landing_x=float(landing_x_center),
                                        landing_y=float(landing_y_center),
                                        landing_row=landing_row,
                                        landing_col=landing_col,
                                        angle_bin=angle_bin,
                                        power_bin=power_bin,
                                    )
                                    key = (
                                        contact_x_bin,
                                        contact_y_bin,
                                        contact_height_bin,
                                        landing_row,
                                        landing_col,
                                        angle_bin,
                                        power_bin,
                                    )
                                    velocities[key] = np.asarray(best.velocity, dtype=np.float32)
                                    valid[key] = bool(best.valid)
                                    fallback_used[key] = bool(best.fallback_used)
                                    landing_x[key] = float(best.landing_position[0])
                                    landing_y[key] = float(best.landing_position[1])
                                    if best.net_crossing_height is not None:
                                        net_crossing_height[key] = float(best.net_crossing_height)
                                    flight_time[key] = float(best.flight_time)
                                    score[key] = float(best.score)
                                    shot_name_index[key] = int(SHOT_NAME_ORDER.index(best.inferred_shot_name))

        np.savez_compressed(
            self.lookup_path,
            velocities=velocities,
            valid=valid,
            fallback_used=fallback_used,
            landing_x=landing_x,
            landing_y=landing_y,
            net_crossing_height=net_crossing_height,
            flight_time=flight_time,
            score=score,
            shot_name_index=shot_name_index,
            contact_x_centers=self.contact_x_centers.astype(np.float32),
            contact_y_centers=self.contact_y_centers.astype(np.float32),
            contact_height_centers=self.contact_height_centers.astype(np.float32),
            landing_row_centers=self.landing_row_centers.astype(np.float32),
            landing_col_centers=self.landing_col_centers.astype(np.float32),
            zone_names=np.asarray(self.zone_names),
            angle_names=np.asarray(self.angle_names),
            power_names=np.asarray(self.power_names),
            shot_names=np.asarray(SHOT_NAME_ORDER),
            version=np.asarray([1], dtype=np.int16),
        )
        self._load()

    def _load(self) -> None:
        payload = np.load(self.lookup_path, allow_pickle=False)
        velocities = np.asarray(payload["velocities"], dtype=np.float32)
        valid = np.asarray(payload["valid"], dtype=bool)
        fallback_used = np.asarray(payload["fallback_used"], dtype=bool)
        expected_velocity_shape = self.table_shape + (3,)
        if velocities.shape != expected_velocity_shape or valid.shape != self.table_shape or fallback_used.shape != self.table_shape:
            raise ValueError("Cached 2D tactic lookup has an incompatible shape and must be regenerated.")
        self.velocities = velocities
        self.valid = valid
        self.fallback_used = fallback_used
        self.landing_x = np.asarray(payload["landing_x"], dtype=np.float32)
        self.landing_y = np.asarray(payload["landing_y"], dtype=np.float32)
        self.net_crossing_height = np.asarray(payload["net_crossing_height"], dtype=np.float32)
        self.flight_time = np.asarray(payload["flight_time"], dtype=np.float32)
        self.score = np.asarray(payload["score"], dtype=np.float32)
        self.shot_name_index = np.asarray(payload["shot_name_index"], dtype=np.int16)
        self._loaded = True

    def lookup(self, state: StageState, action: TacticAction2D) -> LookupEntry2D:
        self.ensure_loaded()
        canonical = canonicalize_state(state)
        contact_bins = (
            nearest_bin(canonical.x0, self.contact_x_centers),
            nearest_bin(canonical.y0, self.contact_y_centers),
            nearest_bin(canonical.z0, self.contact_height_centers),
        )
        key = contact_bins + (action.landing_row, action.landing_col, action.angle_bin, action.power_bin)
        canonical_velocity = tuple(float(value) for value in self.velocities[key].tolist())
        velocity = restore_velocity_from_canonical(
            canonical_velocity,
            mirrored_longitudinal=canonical.mirrored_longitudinal,
        )
        canonical_landing_y = float(self.landing_y[key])
        landing_position = (
            float(self.landing_x[key]),
            -canonical_landing_y if canonical.mirrored_longitudinal else canonical_landing_y,
        )
        crossing = float(self.net_crossing_height[key])
        shot_name = infer_shot_name(
            contact_y=canonical.y0,
            landing_row=action.landing_row,
            angle_bin=action.angle_bin,
            power_bin=action.power_bin,
            config=self.config,
        )
        return LookupEntry2D(
            velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
            valid=bool(self.valid[key]),
            fallback_used=bool(self.fallback_used[key]),
            landing_position=landing_position,
            net_crossing_height=None if np.isnan(crossing) else crossing,
            flight_time=float(self.flight_time[key]),
            score=float(self.score[key]),
            inferred_shot_name=shot_name,
            contact_bins=contact_bins,
        )

    def best_flat_action_for_velocity(self, state: StageState, shot_action: ShotAction) -> int:
        self.ensure_loaded()
        canonical = canonicalize_state(state)
        contact_bins = (
            nearest_bin(canonical.x0, self.contact_x_centers),
            nearest_bin(canonical.y0, self.contact_y_centers),
            nearest_bin(canonical.z0, self.contact_height_centers),
        )
        velocity = np.asarray(
            [
                float(shot_action.v_x),
                float(-shot_action.v_y if canonical.mirrored_longitudinal else shot_action.v_y),
                float(shot_action.v_z),
            ],
            dtype=np.float32,
        )
        velocities = self.velocities[contact_bins].reshape(self.action_count, 3)
        valid = self.valid[contact_bins].reshape(self.action_count)
        fallback = self.fallback_used[contact_bins].reshape(self.action_count)
        squared_error = np.sum((velocities - velocity[None, :]) ** 2, axis=1)
        squared_error = squared_error + np.where(valid, 0.0, 10.0) + np.where(fallback, 2.0, 0.0)
        return int(np.argmin(squared_error))

    def summary(self) -> dict[str, float]:
        self.ensure_loaded()
        return {
            "entries": float(self.valid.size),
            "valid_fraction": float(np.mean(self.valid)),
            "fallback_fraction": float(np.mean(self.fallback_used)),
            "lookup_path": str(self.lookup_path),
        }

    def _state_for_contact(self, x0: float, y0: float, z0: float) -> StageState:
        return StageState(
            x_left=float(x0),
            y_left=float(y0),
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=float(x0),
            y0=float(y0),
            z0=float(z0),
            stage_index=1,
        )

    def _best_entry(
        self,
        *,
        state: StageState,
        landing_x: float,
        landing_y: float,
        landing_row: int,
        landing_col: int,
        angle_bin: int,
        power_bin: int,
    ) -> LookupEntry2D:
        requested_angle = float(angle_bin_centers_deg()[angle_bin])
        requested_speed = float(power_speed_targets_2d()[power_bin])
        shot_name = infer_shot_name(
            contact_y=state.y0,
            landing_row=landing_row,
            angle_bin=angle_bin,
            power_bin=power_bin,
            config=self.config,
        )
        best: LookupEntry2D | None = None
        for velocity in self._candidate_velocities(
            state=state,
            landing_x=landing_x,
            landing_y=landing_y,
            landing_row=landing_row,
            angle_bin=angle_bin,
            power_bin=power_bin,
        ):
            shot = ShotAction(v_x=velocity[0], v_y=velocity[1], v_z=velocity[2], x_rec=0.0, y_rec=0.0)
            result = trajectory_result(state, shot, self.config)
            is_valid = valid_hitter_action(state, shot, self.config, result=result)
            ground_speed = max(float(np.hypot(velocity[0], velocity[1])), 1e-6)
            launch_angle = float(np.degrees(np.arctan2(velocity[2], ground_speed)))
            launch_speed = float(np.linalg.norm(np.asarray(velocity, dtype=float)))
            landing_error = float(np.hypot(result.landing_x - landing_x, result.landing_y - landing_y))
            crossing_height = None if result.net_crossing is None else float(result.net_crossing.z)
            score = (
                (260.0 if is_valid else -260.0)
                - 42.0 * landing_error
                - 2.0 * abs(launch_angle - requested_angle)
                - 0.45 * abs(launch_speed - requested_speed)
            )
            if crossing_height is not None:
                score += 6.0 * min(crossing_height, 3.0)
            candidate = LookupEntry2D(
                velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
                valid=is_valid,
                fallback_used=False,
                landing_position=(float(result.landing_x), float(result.landing_y)),
                net_crossing_height=crossing_height,
                flight_time=float(result.landing_time),
                score=float(score),
                inferred_shot_name=shot_name,
                contact_bins=(0, 0, 0),
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None and best.valid:
            return best

        fallback = self._fallback_entry(
            state=state,
            landing_row=landing_row,
            angle_bin=angle_bin,
            power_bin=power_bin,
        )
        if best is None or fallback.score >= best.score:
            return fallback
        return LookupEntry2D(
            velocity=best.velocity,
            valid=False,
            fallback_used=True,
            landing_position=best.landing_position,
            net_crossing_height=best.net_crossing_height,
            flight_time=best.flight_time,
            score=best.score,
            inferred_shot_name=best.inferred_shot_name,
            contact_bins=(0, 0, 0),
        )

    def _fallback_entry(
        self,
        *,
        state: StageState,
        landing_row: int,
        angle_bin: int,
        power_bin: int,
    ) -> LookupEntry2D:
        shot_name = infer_shot_name(
            contact_y=state.y0,
            landing_row=landing_row,
            angle_bin=angle_bin,
            power_bin=power_bin,
            config=self.config,
        )
        safe_velocity = self._find_any_fallback_velocity(state)
        safe_action = ShotAction(v_x=safe_velocity[0], v_y=safe_velocity[1], v_z=safe_velocity[2], x_rec=0.0, y_rec=0.0)
        result = trajectory_result(state, safe_action, self.config)
        return LookupEntry2D(
            velocity=(float(safe_action.v_x), float(safe_action.v_y), float(safe_action.v_z)),
            valid=False,
            fallback_used=True,
            landing_position=(float(result.landing_x), float(result.landing_y)),
            net_crossing_height=None if result.net_crossing is None else float(result.net_crossing.z),
            flight_time=float(result.landing_time),
            score=-125.0,
            inferred_shot_name=shot_name,
            contact_bins=(0, 0, 0),
        )

    def _find_any_fallback_velocity(self, state: StageState) -> tuple[float, float, float]:
        best_velocity = (0.0, 6.0, 4.0)
        best_score = float("-inf")
        for landing_row, landing_y in enumerate(reversed(self.landing_row_centers)):
            for landing_col, landing_x in enumerate(self.landing_col_centers):
                safe_row = 2 - landing_row
                for angle_bin in (4, 3, 2, 1, 0):
                    for power_bin in (2, 1, 0):
                        for velocity in self._candidate_velocities(
                            state=state,
                            landing_x=float(landing_x),
                            landing_y=float(landing_y),
                            landing_row=safe_row,
                            angle_bin=angle_bin,
                            power_bin=power_bin,
                        ):
                            shot = ShotAction(v_x=velocity[0], v_y=velocity[1], v_z=velocity[2], x_rec=0.0, y_rec=0.0)
                            result = trajectory_result(state, shot, self.config)
                            score = -float(np.hypot(result.landing_x - landing_x, result.landing_y - landing_y))
                            if valid_hitter_action(state, shot, self.config, result=result):
                                return (float(velocity[0]), float(velocity[1]), float(velocity[2]))
                            if score > best_score:
                                best_score = score
                                best_velocity = (float(velocity[0]), float(velocity[1]), float(velocity[2]))
        return best_velocity

    def _candidate_velocities(
        self,
        *,
        state: StageState,
        landing_x: float,
        landing_y: float,
        landing_row: int,
        angle_bin: int,
        power_bin: int,
    ) -> list[tuple[float, float, float]]:
        dx = float(landing_x - state.x0)
        dy = float(landing_y - state.y0)
        ground_distance = max(float(np.hypot(dx, dy)), 1e-6)
        direction = np.asarray([dx / ground_distance, dy / ground_distance], dtype=float)
        requested_angle = float(angle_bin_centers_deg()[angle_bin])
        base_speed = float(power_speed_targets_2d()[power_bin]) * (1.0 + 0.15 * landing_row + 0.05 * ground_distance)
        speeds = base_speed * np.asarray([0.7, 0.85, 1.0, 1.2, 1.4, 1.65], dtype=float)
        angle_offsets = np.asarray([-14.0, -7.0, 0.0, 7.0, 14.0], dtype=float)
        velocities: list[tuple[float, float, float]] = []
        seen: set[tuple[int, int, int]] = set()

        for speed in speeds:
            for angle_deg in requested_angle + angle_offsets:
                ground_speed = max(float(speed), 0.2)
                radians = float(np.deg2rad(angle_deg))
                vx = float(direction[0] * ground_speed)
                vy = float(direction[1] * ground_speed)
                vz = float(np.tan(radians) * ground_speed)
                signature = (round(vx * 100), round(vy * 100), round(vz * 100))
                if signature not in seen:
                    seen.add(signature)
                    velocities.append((vx, vy, vz))

        time_center = {0: 1.0, 1: 0.78, 2: 0.58}[power_bin] * (1.0 + 0.18 * landing_row)
        for flight_time in time_center * np.asarray([0.7, 0.9, 1.1, 1.3], dtype=float):
            for scale in (1.0, 1.12, 1.25, 1.38):
                vx = float((dx / flight_time) * scale)
                vy = float((dy / flight_time) * scale)
                vz = float((((-state.z0) + 0.5 * self.config.action.gravity * (flight_time ** 2)) / flight_time) * scale)
                signature = (round(vx * 100), round(vy * 100), round(vz * 100))
                if signature not in seen:
                    seen.add(signature)
                    velocities.append((vx, vy, vz))
        return velocities


def default_lookup_path_2d() -> Path:
    return repo_root() / "lookup_tables" / LOOKUP_FILENAME
