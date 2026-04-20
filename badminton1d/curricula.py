from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.state import Side, StageState
from badminton1d.utils import opponent_side, side_x_bounds, side_y_bounds

DEFAULT_DEFENSIVE_CURRICULUM_NAME = "defensive-backcourt-attack-best"
DEFAULT_DEFENSIVE_CURRICULUM_OPPONENT_PATH = Path(
    "outputs/rl/selfplay_1d_dragsquare_ps35_rt03_ic50_recency_masked_rm03_20260419_20k_100ktotal_attack_best/best_model.zip"
)


@dataclass(frozen=True)
class DefensiveBackcourtPhase:
    name: str
    start_episode: int
    attacker_depth_range: tuple[float, float]
    attacker_lateral_span: float
    defender_depth_range: tuple[float, float]
    defender_lateral_span: float
    hit_height_range: tuple[float, float]

    def __post_init__(self) -> None:
        if self.start_episode < 0:
            raise ValueError("start_episode must be zero or greater.")
        for lower, upper, label in (
            (*self.attacker_depth_range, "attacker_depth_range"),
            (*self.defender_depth_range, "defender_depth_range"),
            (*self.hit_height_range, "hit_height_range"),
        ):
            if lower > upper:
                raise ValueError(f"{label} must be increasing.")
        for lower, upper, label in (
            (*self.attacker_depth_range, "attacker_depth_range"),
            (*self.defender_depth_range, "defender_depth_range"),
        ):
            if lower < 0.0 or upper > 1.0:
                raise ValueError(f"{label} values must lie in [0, 1].")
        for value, label in (
            (self.attacker_lateral_span, "attacker_lateral_span"),
            (self.defender_lateral_span, "defender_lateral_span"),
        ):
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"{label} must lie in (0, 1].")


@dataclass(frozen=True)
class DefensiveBackcourtCurriculumConfig:
    name: str = DEFAULT_DEFENSIVE_CURRICULUM_NAME
    stage_index: int = 1
    phases: tuple[DefensiveBackcourtPhase, ...] = field(
        default_factory=lambda: (
            DefensiveBackcourtPhase(
                name="stabilize_center_lane",
                start_episode=0,
                attacker_depth_range=(0.72, 0.84),
                attacker_lateral_span=0.35,
                defender_depth_range=(0.52, 0.66),
                defender_lateral_span=0.30,
                hit_height_range=(1.75, 2.05),
            ),
            DefensiveBackcourtPhase(
                name="expand_attack_angles",
                start_episode=1_500,
                attacker_depth_range=(0.78, 0.90),
                attacker_lateral_span=0.65,
                defender_depth_range=(0.42, 0.72),
                defender_lateral_span=0.60,
                hit_height_range=(1.95, 2.30),
            ),
            DefensiveBackcourtPhase(
                name="full_backcourt_pressure",
                start_episode=4_500,
                attacker_depth_range=(0.82, 0.98),
                attacker_lateral_span=1.00,
                defender_depth_range=(0.25, 0.78),
                defender_lateral_span=1.00,
                hit_height_range=(2.10, 2.55),
            ),
        )
    )

    def __post_init__(self) -> None:
        if self.stage_index <= 0:
            raise ValueError("stage_index must be positive so the curriculum starts mid-rally.")
        if not self.phases:
            raise ValueError("At least one curriculum phase is required.")
        ordered = tuple(sorted(self.phases, key=lambda phase: phase.start_episode))
        if ordered != self.phases:
            raise ValueError("Curriculum phases must be sorted by start_episode.")


@dataclass(frozen=True)
class TrainingCurriculumSpec:
    name: str
    description: str
    opponent_checkpoint_path: Path
    sampler_config: DefensiveBackcourtCurriculumConfig
    initial_server: str = "opponent"
    opponent_hitter_deterministic: bool = False
    opponent_receiver_deterministic: bool = True


class DefensiveBackcourtCurriculumSampler:
    def __init__(
        self,
        *,
        sim_config: SimulationConfig,
        curriculum_config: DefensiveBackcourtCurriculumConfig,
        seed: int | None = None,
    ) -> None:
        self.sim_config = sim_config
        self.curriculum_config = curriculum_config
        self.rng = np.random.default_rng(seed)
        self.episodes_sampled = 0

    def sample_initial_state(self, *, train_side: Side) -> tuple[StageState, Side]:
        phase = self._active_phase()
        self.episodes_sampled += 1

        defender_side = train_side
        attacker_side = opponent_side(train_side)

        attacker_x = self._sample_side_x(attacker_side, phase.attacker_lateral_span)
        defender_x = self._sample_side_x(defender_side, phase.defender_lateral_span)
        attacker_y = self._sample_side_depth(attacker_side, phase.attacker_depth_range)
        defender_y = self._sample_side_depth(defender_side, phase.defender_depth_range)
        z0 = float(self.rng.uniform(*phase.hit_height_range))

        if attacker_side == "left":
            x_left, y_left = attacker_x, attacker_y
            x_right, y_right = defender_x, defender_y
        else:
            x_left, y_left = defender_x, defender_y
            x_right, y_right = attacker_x, attacker_y

        state = StageState(
            x_left=float(x_left),
            y_left=float(y_left),
            x_right=float(x_right),
            y_right=float(y_right),
            current_hitter=attacker_side,
            x0=float(attacker_x),
            y0=float(attacker_y),
            z0=z0,
            rally_done=False,
            winner=None,
            stage_index=self.curriculum_config.stage_index,
        )
        return state, attacker_side

    def _active_phase(self) -> DefensiveBackcourtPhase:
        active = self.curriculum_config.phases[0]
        for phase in self.curriculum_config.phases[1:]:
            if self.episodes_sampled < phase.start_episode:
                break
            active = phase
        return active

    def _sample_side_x(self, side: Side, span_ratio: float) -> float:
        x_low, x_high = side_x_bounds(side, self.sim_config)
        center = 0.5 * (x_low + x_high)
        half_span = 0.5 * (x_high - x_low) * span_ratio
        low = max(x_low, center - half_span)
        high = min(x_high, center + half_span)
        return float(self.rng.uniform(low, high))

    def _sample_side_depth(self, side: Side, depth_range: tuple[float, float]) -> float:
        low_ratio, high_ratio = depth_range
        depth_ratio = float(self.rng.uniform(low_ratio, high_ratio))
        y_low, y_high = side_y_bounds(side, self.sim_config)
        if side == "right":
            return float(y_low + depth_ratio * (y_high - y_low))
        return float(y_high - depth_ratio * (y_high - y_low))


def available_training_curricula() -> tuple[str, ...]:
    return (DEFAULT_DEFENSIVE_CURRICULUM_NAME,)


def build_training_curriculum(
    name: str,
    *,
    opponent_checkpoint_path: Path | None = None,
) -> TrainingCurriculumSpec:
    normalized = name.strip().lower()
    if normalized != DEFAULT_DEFENSIVE_CURRICULUM_NAME:
        raise ValueError(f"Unsupported curriculum: {name}")
    return TrainingCurriculumSpec(
        name=DEFAULT_DEFENSIVE_CURRICULUM_NAME,
        description=(
            "Start each rally with the opponent holding a high-contact back-court attack position and "
            "sample its opening shot stochastically to train defensive coverage."
        ),
        opponent_checkpoint_path=(opponent_checkpoint_path or DEFAULT_DEFENSIVE_CURRICULUM_OPPONENT_PATH).resolve(),
        sampler_config=DefensiveBackcourtCurriculumConfig(),
        initial_server="opponent",
        opponent_hitter_deterministic=False,
        opponent_receiver_deterministic=True,
    )
