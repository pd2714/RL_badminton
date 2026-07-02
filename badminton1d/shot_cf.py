from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from badminton1d.state import ShotAction


@dataclass(frozen=True)
class ShotCFCandidate:
    flat_index: int
    shot_key: tuple[int, int, int]
    log_probability: float
    probability: float
    shot_action: ShotAction
    landing_x: float
    landing_y: float


@dataclass(frozen=True)
class ShotCFSelection:
    candidates: tuple[ShotCFCandidate, ...]
    chosen_index: int
    skipped: bool
    skip_reason: str | None = None
    mean_landing_distance: float = 0.0


def mean_pairwise_landing_distance(candidates: Sequence[ShotCFCandidate]) -> float:
    if len(candidates) < 2:
        return 0.0
    distances: list[float] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            distances.append(float(np.hypot(left.landing_x - right.landing_x, left.landing_y - right.landing_y)))
    return float(np.mean(distances)) if distances else 0.0


def select_diverse_shot_candidates(
    candidates: Sequence[ShotCFCandidate],
    *,
    chosen_candidate: ShotCFCandidate | None,
    num_modes: int = 3,
    min_landing_dist: float = 1.0,
    include_chosen: bool = True,
    skip_low_diversity: bool = True,
    min_modes: int = 2,
) -> ShotCFSelection:
    """Greedily keep high-probability shot modes separated in landing space."""

    max_modes = max(int(num_modes), 1)
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (float(candidate.log_probability), float(candidate.probability)),
        reverse=True,
    )

    selected: list[ShotCFCandidate] = []
    for candidate in sorted_candidates:
        if any(candidate.flat_index == existing.flat_index for existing in selected):
            continue
        if selected and any(
            np.hypot(candidate.landing_x - existing.landing_x, candidate.landing_y - existing.landing_y)
            < float(min_landing_dist)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_modes:
            break

    chosen_index = -1
    if include_chosen and chosen_candidate is not None:
        for index, candidate in enumerate(selected):
            if candidate.flat_index == chosen_candidate.flat_index or (
                np.hypot(
                    candidate.landing_x - chosen_candidate.landing_x,
                    candidate.landing_y - chosen_candidate.landing_y,
                )
                < float(min_landing_dist)
            ):
                chosen_index = index
                break
        if chosen_index < 0:
            if len(selected) >= max_modes:
                selected = selected[: max_modes - 1]
            selected.insert(0, chosen_candidate)
            chosen_index = 0

    if chosen_index < 0 and chosen_candidate is not None:
        for index, candidate in enumerate(selected):
            if candidate.flat_index == chosen_candidate.flat_index:
                chosen_index = index
                break

    if chosen_index < 0:
        return ShotCFSelection(tuple(selected), -1, True, "chosen_missing")

    if skip_low_diversity and len(selected) < max(int(min_modes), 1):
        return ShotCFSelection(
            tuple(selected),
            chosen_index,
            True,
            "low_diversity",
            mean_landing_distance=mean_pairwise_landing_distance(selected),
        )

    return ShotCFSelection(
        tuple(selected),
        chosen_index,
        False,
        mean_landing_distance=mean_pairwise_landing_distance(selected),
    )
