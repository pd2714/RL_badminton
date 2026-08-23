from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PairwiseRecord:
    agent_a: str
    agent_b: str
    agent_a_score: float
    games: float


def expected_score(rating_a: float, rating_b: float, *, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def calculate_elo(
    records: Iterable[PairwiseRecord | Mapping[str, object]],
    *,
    initial_rating: float = 1500.0,
    scale: float = 400.0,
    prior_std: float = 400.0,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    """Estimate fixed-pool Elo ratings from aggregate pairwise outcomes.

    The estimator fits a Bradley-Terry/Elo logistic model to all pair results
    at once. A weak Gaussian prior keeps ratings finite when a pool contains
    one-sided matchups.
    """
    normalized = [_normalize_record(record) for record in records]
    if not normalized:
        raise ValueError("records must not be empty")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if prior_std <= 0.0:
        raise ValueError("prior_std must be positive")

    agents = sorted({record.agent_a for record in normalized} | {record.agent_b for record in normalized})
    index = {agent: i for i, agent in enumerate(agents)}
    ratings = np.full(len(agents), float(initial_rating), dtype=float)
    beta = math.log(10.0) / float(scale)
    prior_precision = 1.0 / (float(prior_std) ** 2)

    for _ in range(max(int(max_iterations), 1)):
        gradient = np.zeros_like(ratings)
        neg_hessian = np.eye(len(agents), dtype=float) * prior_precision

        for record in normalized:
            i = index[record.agent_a]
            j = index[record.agent_b]
            games = float(record.games)
            score_a = float(record.agent_a_score)
            probability_a = 1.0 / (1.0 + math.exp(-beta * float(ratings[i] - ratings[j])))
            residual = score_a - games * probability_a
            gradient[i] += beta * residual
            gradient[j] -= beta * residual

            curvature = beta * beta * games * probability_a * (1.0 - probability_a)
            neg_hessian[i, i] += curvature
            neg_hessian[j, j] += curvature
            neg_hessian[i, j] -= curvature
            neg_hessian[j, i] -= curvature

        gradient -= (ratings - initial_rating) * prior_precision
        step = np.linalg.solve(neg_hessian, gradient)
        ratings += step
        if float(np.max(np.abs(step))) < tolerance:
            break

    return {agent: float(rating) for agent, rating in zip(agents, ratings.tolist())}


def _normalize_record(record: PairwiseRecord | Mapping[str, object]) -> PairwiseRecord:
    if isinstance(record, PairwiseRecord):
        normalized = record
    else:
        agent_a = str(record["agent_a"])
        agent_b = str(record["agent_b"])
        if "agent_a_score" in record and "games" in record:
            normalized = PairwiseRecord(
                agent_a=agent_a,
                agent_b=agent_b,
                agent_a_score=float(record["agent_a_score"]),
                games=float(record["games"]),
            )
        elif "agent_a_win_rate" in record and "episodes" in record:
            games = float(record["episodes"])
            normalized = PairwiseRecord(
                agent_a=agent_a,
                agent_b=agent_b,
                agent_a_score=float(record["agent_a_win_rate"]) * games,
                games=games,
            )
        else:
            raise ValueError("record mappings need agent_a/agent_b plus agent_a_score/games or agent_a_win_rate/episodes")

    if normalized.agent_a == normalized.agent_b:
        raise ValueError("agent_a and agent_b must be different")
    if normalized.games <= 0.0:
        raise ValueError("games must be positive")
    if normalized.agent_a_score < 0.0 or normalized.agent_a_score > normalized.games:
        raise ValueError("agent_a_score must be between 0 and games")
    return normalized


def ratings_table(ratings: Mapping[str, float], *, descending: bool = True) -> list[dict[str, object]]:
    ordered: Sequence[tuple[str, float]] = sorted(ratings.items(), key=lambda item: item[1], reverse=descending)
    return [
        {
            "rank": rank,
            "agent": agent,
            "elo": float(rating),
        }
        for rank, (agent, rating) in enumerate(ordered, start=1)
    ]
