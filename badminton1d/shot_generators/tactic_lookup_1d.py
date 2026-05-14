from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import trajectory_result, valid_hitter_action
from badminton1d.shot_generators.shot_naming import infer_shot_name
from badminton1d.shot_generators.tactic_lookup_common import (
    ANGLE_BIN_COUNT_1D,
    ANGLE_BIN_NAMES_1D,
    LANDING_ZONE_COUNT_1D,
    POWER_BIN_NAMES_1D,
    SHOT_NAME_ORDER,
    TacticAction1D,
    TacticRuntimeConfig,
    angle_bin_centers_deg_1d,
    canonicalize_state,
    contact_height_centers,
    contact_y_centers_1d,
    landing_row_centers,
    landing_zone_names_1d,
    nearest_bin,
    power_speed_targets_1d,
    repo_root,
    resolve_lookup_dir,
    restore_velocity_from_canonical,
)
from badminton1d.state import ShotAction, StageState

LOOKUP_FILENAME = "tactic_1d_lookup.npz"


@dataclass(frozen=True)
class LookupEntry1D:
    velocity: tuple[float, float]
    valid: bool
    fallback_used: bool
    landing_position: float
    net_crossing_height: float | None
    flight_time: float
    score: float
    inferred_shot_name: str
    contact_bins: tuple[int, int]


class TacticLookup1D:
    def __init__(
        self,
        config: SimulationConfig,
        runtime_config: TacticRuntimeConfig | None = None,
    ) -> None:
        self.config = config
        self.runtime_config = runtime_config or TacticRuntimeConfig()
        self.lookup_dir = resolve_lookup_dir(self.runtime_config)
        self.lookup_path = self.lookup_dir / LOOKUP_FILENAME
        self.contact_y_centers = contact_y_centers_1d(config)
        self.contact_height_centers = contact_height_centers(config)
        self.landing_zone_centers = landing_row_centers(config, bins=LANDING_ZONE_COUNT_1D)
        self.zone_names = landing_zone_names_1d()
        self.angle_names = ANGLE_BIN_NAMES_1D
        self.power_names = POWER_BIN_NAMES_1D
        self._loaded = False

    @property
    def action_count(self) -> int:
        return len(self.zone_names) * len(self.angle_names) * len(self.power_names)

    @property
    def table_shape(self) -> tuple[int, ...]:
        return (
            len(self.contact_y_centers),
            len(self.contact_height_centers),
            len(self.zone_names),
            len(self.angle_names),
            len(self.power_names),
        )

    def flat_to_action(self, flat_index: int) -> TacticAction1D:
        bounded = int(flat_index) % self.action_count
        power_bin = bounded % len(self.power_names)
        bounded //= len(self.power_names)
        angle_bin = bounded % len(self.angle_names)
        landing_zone = bounded // len(self.angle_names)
        return TacticAction1D(
            landing_zone=int(landing_zone),
            angle_bin=int(angle_bin),
            power_bin=int(power_bin),
        )

    def action_to_flat(self, action: TacticAction1D) -> int:
        return ((int(action.landing_zone) * len(self.angle_names)) + int(action.angle_bin)) * len(self.power_names) + int(action.power_bin)

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
        velocities = np.zeros(shape + (2,), dtype=np.float32)
        valid = np.zeros(shape, dtype=bool)
        fallback_used = np.zeros(shape, dtype=bool)
        landing_y = np.zeros(shape, dtype=np.float32)
        net_crossing_height = np.full(shape, np.nan, dtype=np.float32)
        flight_time = np.zeros(shape, dtype=np.float32)
        score = np.full(shape, -1e9, dtype=np.float32)
        shot_name_index = np.full(shape, SHOT_NAME_ORDER.index("generic"), dtype=np.int16)

        for contact_y_bin, y0 in enumerate(self.contact_y_centers):
            for contact_height_bin, z0 in enumerate(self.contact_height_centers):
                state = self._state_for_contact(y0, z0)
                for landing_zone, landing_center in enumerate(self.landing_zone_centers):
                    for angle_bin in range(len(self.angle_names)):
                        for power_bin in range(len(self.power_names)):
                            best = self._best_entry(
                                state=state,
                                landing_y=float(landing_center),
                                landing_row=landing_zone,
                                angle_bin=angle_bin,
                                power_bin=power_bin,
                            )
                            key = (contact_y_bin, contact_height_bin, landing_zone, angle_bin, power_bin)
                            velocities[key] = np.asarray(best.velocity, dtype=np.float32)
                            valid[key] = bool(best.valid)
                            fallback_used[key] = bool(best.fallback_used)
                            landing_y[key] = float(best.landing_position)
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
            landing_y=landing_y,
            net_crossing_height=net_crossing_height,
            flight_time=flight_time,
            score=score,
            shot_name_index=shot_name_index,
            contact_y_centers=self.contact_y_centers.astype(np.float32),
            contact_height_centers=self.contact_height_centers.astype(np.float32),
            landing_zone_centers=self.landing_zone_centers.astype(np.float32),
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
        expected_velocity_shape = self.table_shape + (2,)
        if velocities.shape != expected_velocity_shape or valid.shape != self.table_shape or fallback_used.shape != self.table_shape:
            raise ValueError("Cached 1D tactic lookup has an incompatible shape and must be regenerated.")
        self.velocities = velocities
        self.valid = valid
        self.fallback_used = fallback_used
        self.landing_y = np.asarray(payload["landing_y"], dtype=np.float32)
        self.net_crossing_height = np.asarray(payload["net_crossing_height"], dtype=np.float32)
        self.flight_time = np.asarray(payload["flight_time"], dtype=np.float32)
        self.score = np.asarray(payload["score"], dtype=np.float32)
        self.shot_name_index = np.asarray(payload["shot_name_index"], dtype=np.int16)
        self._loaded = True

    def lookup(self, state: StageState, action: TacticAction1D) -> LookupEntry1D:
        self.ensure_loaded()
        canonical = canonicalize_state(state)
        contact_bins = (
            nearest_bin(canonical.y0, self.contact_y_centers),
            nearest_bin(canonical.z0, self.contact_height_centers),
        )
        key = contact_bins + (action.landing_zone, action.angle_bin, action.power_bin)
        canonical_velocity = tuple(float(value) for value in self.velocities[key].tolist())
        velocity = restore_velocity_from_canonical(
            canonical_velocity,
            mirrored_longitudinal=canonical.mirrored_longitudinal,
        )
        landing_y = float(self.landing_y[key])
        shot_name = infer_shot_name(
            contact_y=canonical.y0,
            landing_row=action.landing_zone,
            angle_bin=action.angle_bin,
            power_bin=action.power_bin,
            config=self.config,
            angle_names=self.angle_names,
            power_names=self.power_names,
            landing_row_count=len(self.zone_names),
            angle_degrees=float(
                angle_bin_centers_deg_1d(
                    canonical.y0,
                    canonical.z0,
                    self.config,
                    bins=ANGLE_BIN_COUNT_1D,
                )[action.angle_bin]
            ),
        )
        crossing = float(self.net_crossing_height[key])
        return LookupEntry1D(
            velocity=(float(velocity[0]), float(velocity[1])),
            valid=bool(self.valid[key]),
            fallback_used=bool(self.fallback_used[key]),
            landing_position=-landing_y if canonical.mirrored_longitudinal else landing_y,
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
            nearest_bin(canonical.y0, self.contact_y_centers),
            nearest_bin(canonical.z0, self.contact_height_centers),
        )
        velocity = np.asarray(
            [
                float(-shot_action.v_y if canonical.mirrored_longitudinal else shot_action.v_y),
                float(shot_action.v_z),
            ],
            dtype=np.float32,
        )
        velocities = self.velocities[contact_bins].reshape(self.action_count, 2)
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

    def _state_for_contact(self, y0: float, z0: float) -> StageState:
        return StageState(
            x_left=0.0,
            y_left=float(y0),
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=float(y0),
            z0=float(z0),
            stage_index=1,
        )

    def _best_entry(
        self,
        *,
        state: StageState,
        landing_y: float,
        landing_row: int,
        angle_bin: int,
        power_bin: int,
    ) -> LookupEntry1D:
        requested_angle = float(
            angle_bin_centers_deg_1d(
                state.y0,
                state.z0,
                self.config,
                bins=ANGLE_BIN_COUNT_1D,
            )[angle_bin]
        )
        requested_speed = float(power_speed_targets_1d()[power_bin])
        shot_name = infer_shot_name(
            contact_y=state.y0,
            landing_row=landing_row,
            angle_bin=angle_bin,
            power_bin=power_bin,
            config=self.config,
            angle_names=self.angle_names,
            power_names=self.power_names,
            landing_row_count=len(self.zone_names),
            angle_degrees=requested_angle,
        )
        best: LookupEntry1D | None = None
        for velocity in self._candidate_velocities(
            state=state,
            landing_y=landing_y,
            landing_row=landing_row,
            angle_bin=angle_bin,
            power_bin=power_bin,
        ):
            shot = ShotAction(v_x=0.0, v_y=velocity[0], v_z=velocity[1], x_rec=0.0, y_rec=0.0)
            result = trajectory_result(state, shot, self.config)
            is_valid = valid_hitter_action(state, shot, self.config, result=result)
            ground_speed = max(abs(float(velocity[0])), 1e-6)
            launch_angle = float(np.degrees(np.arctan2(velocity[1], ground_speed)))
            launch_speed = float(np.linalg.norm(np.asarray(velocity, dtype=float)))
            landing_error = abs(float(result.landing_y) - landing_y)
            crossing_height = None if result.net_crossing is None else float(result.net_crossing.z)
            candidate_score = (
                - 7.5 * landing_error
                - 8.0 * abs(launch_angle - requested_angle)
                - 0.5 * abs(launch_speed - requested_speed)
            )
            candidate = LookupEntry1D(
                velocity=(float(velocity[0]), float(velocity[1])),
                valid=is_valid,
                fallback_used=False,
                landing_position=float(result.landing_y),
                net_crossing_height=crossing_height,
                flight_time=float(result.landing_time),
                score=float(candidate_score),
                inferred_shot_name=shot_name,
                contact_bins=(0, 0),
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None:
            return best
        raise RuntimeError("1D tactic lookup could not produce any candidate velocities.")

    def _candidate_velocities(
        self,
        *,
        state: StageState,
        landing_y: float,
        landing_row: int,
        angle_bin: int,
        power_bin: int,
    ) -> list[tuple[float, float]]:
        dy = float(landing_y - state.y0)
        requested_angle = float(
            angle_bin_centers_deg_1d(
                state.y0,
                state.z0,
                self.config,
                bins=ANGLE_BIN_COUNT_1D,
            )[angle_bin]
        )
        base_speed = float(power_speed_targets_1d()[power_bin]) * (1.0 + 0.16 * landing_row + 0.06 * abs(dy))
        speeds = base_speed * np.asarray([0.7, 0.85, 1.0, 1.18, 1.4, 1.6], dtype=float)
        angle_offsets = np.asarray([-14.0, -7.0, 0.0, 7.0, 14.0], dtype=float)
        velocities: list[tuple[float, float]] = []
        seen: set[tuple[int, int]] = set()
        direction = 1.0 if dy >= 0.0 else -1.0

        for speed in speeds:
            for angle_deg in requested_angle + angle_offsets:
                ground_speed = max(float(speed), 0.2) * direction
                vz = float(np.tan(np.deg2rad(angle_deg)) * abs(ground_speed))
                signature = (round(ground_speed * 100), round(vz * 100))
                if signature not in seen:
                    seen.add(signature)
                    velocities.append((ground_speed, vz))

        power_ratio = 0.0 if len(self.power_names) == 1 else float(power_bin) / float(len(self.power_names) - 1)
        time_center = (1.1 - 0.55 * power_ratio) * (1.0 + 0.18 * landing_row)
        for flight_time in time_center * np.asarray([0.7, 0.9, 1.1, 1.3], dtype=float):
            for scale in (1.0, 1.12, 1.25, 1.38):
                vy = float((dy / flight_time) * scale)
                vz = float((((-state.z0) + 0.5 * self.config.action.gravity * (flight_time ** 2)) / flight_time) * scale)
                signature = (round(vy * 100), round(vz * 100))
                if signature not in seen:
                    seen.add(signature)
                    velocities.append((vy, vz))
        return velocities


def default_lookup_path_1d() -> Path:
    return repo_root() / "lookup_tables" / LOOKUP_FILENAME
