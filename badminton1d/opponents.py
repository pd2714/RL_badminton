from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from badminton1d.action_space import DiscreteActionMapper
from badminton1d.agents import GreedyReceiver, RandomValidHitter, SafeHitter
from badminton1d.config import SimulationConfig
from badminton1d.state import ShotAction, Side, StageState


class OpponentPolicy(Protocol):
    def on_episode_start(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        ...

    def refresh(self) -> None:
        ...

    def label(self) -> str:
        ...

    def choose_hitter_action(self, state: StageState, config: SimulationConfig, server_side: Side) -> ShotAction:
        ...

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
        server_side: Side,
    ) -> int | None:
        ...


@dataclass
class SafeHeuristicOpponent:
    hitter: SafeHitter = field(default_factory=SafeHitter)
    receiver: GreedyReceiver = field(default_factory=lambda: GreedyReceiver(mode="earliest"))

    def on_episode_start(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        return None

    def refresh(self) -> None:
        return None

    def label(self) -> str:
        return "heuristic_safe"

    def choose_hitter_action(self, state: StageState, config: SimulationConfig, server_side: Side) -> ShotAction:
        return self.hitter.choose_action(state, config)

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
        server_side: Side,
    ) -> int | None:
        return self.receiver.choose_intercept_index(state, action, feasible_indices, config)


@dataclass
class RandomValidOpponent:
    seed: int | None = None
    hitter: RandomValidHitter = field(init=False)
    receiver: GreedyReceiver = field(default_factory=lambda: GreedyReceiver(mode="earliest"))

    def __post_init__(self) -> None:
        self.hitter = RandomValidHitter(seed=self.seed)

    def on_episode_start(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        return None

    def refresh(self) -> None:
        return None

    def label(self) -> str:
        return "random_valid"

    def choose_hitter_action(self, state: StageState, config: SimulationConfig, server_side: Side) -> ShotAction:
        return self.hitter.choose_action(state, config)

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
        server_side: Side,
    ) -> int | None:
        return self.receiver.choose_intercept_index(state, action, feasible_indices, config)


@dataclass
class GreedyInterceptOpponent:
    hitter: SafeHitter = field(default_factory=SafeHitter)
    receiver: GreedyReceiver = field(default_factory=lambda: GreedyReceiver(mode="highest"))

    def on_episode_start(
        self,
        *,
        train_side: Side,
        opponent_side: Side,
        server_side: Side,
        config: SimulationConfig,
    ) -> None:
        return None

    def refresh(self) -> None:
        return None

    def label(self) -> str:
        return "heuristic_greedy"

    def choose_hitter_action(self, state: StageState, config: SimulationConfig, server_side: Side) -> ShotAction:
        return self.hitter.choose_action(state, config)

    def choose_intercept_index(
        self,
        state: StageState,
        action: ShotAction,
        feasible_indices: list[int],
        config: SimulationConfig,
        server_side: Side,
    ) -> int | None:
        return self.receiver.choose_intercept_index(state, action, feasible_indices, config)


def make_opponent(name: str, seed: int | None = None) -> OpponentPolicy:
    normalized = name.strip().lower()
    if normalized == "safe":
        return SafeHeuristicOpponent()
    if normalized == "random":
        return RandomValidOpponent(seed=seed)
    if normalized == "greedy":
        return GreedyInterceptOpponent()
    raise ValueError(f"Unsupported opponent type: {name}")


@dataclass(frozen=True)
class DecisionContext:
    state: StageState
    role: str
    pending_action: ShotAction | None
    feasible_indices: list[int]
    receiver_action_count: int


class BaselinePolicy(Protocol):
    def choose_action(self, context: DecisionContext) -> int:
        ...


@dataclass
class RandomDiscretePolicy:
    action_mapper: DiscreteActionMapper
    seed: int | None = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def choose_action(self, context: DecisionContext) -> int:
        if context.role == "hitter":
            return int(self.rng.integers(self.action_mapper.hitter_action_count))
        if context.feasible_indices:
            return int(self.rng.choice(context.feasible_indices))
        return 0


@dataclass
class SafeDiscretePolicy:
    action_mapper: DiscreteActionMapper
    hitter: SafeHitter = field(default_factory=SafeHitter)
    receiver: GreedyReceiver = field(default_factory=lambda: GreedyReceiver(mode="earliest"))

    def choose_action(self, context: DecisionContext) -> int:
        if context.role == "hitter":
            action = self.hitter.choose_action(context.state, self.action_mapper.config)
            return self.action_mapper.encode_hitter(action, context.state)
        if context.pending_action is None:
            return 0
        chosen = self.receiver.choose_intercept_index(
            context.state,
            context.pending_action,
            context.feasible_indices,
            self.action_mapper.config,
        )
        return 0 if chosen is None else self.action_mapper.encode_receiver(chosen)


@dataclass
class GreedyDiscretePolicy:
    action_mapper: DiscreteActionMapper
    hitter: SafeHitter = field(default_factory=SafeHitter)
    receiver: GreedyReceiver = field(default_factory=lambda: GreedyReceiver(mode="highest"))

    def choose_action(self, context: DecisionContext) -> int:
        if context.role == "hitter":
            action = self.hitter.choose_action(context.state, self.action_mapper.config)
            return self.action_mapper.encode_hitter(action, context.state)
        if context.pending_action is None:
            return 0
        chosen = self.receiver.choose_intercept_index(
            context.state,
            context.pending_action,
            context.feasible_indices,
            self.action_mapper.config,
        )
        return 0 if chosen is None else self.action_mapper.encode_receiver(chosen)


def make_baseline_policy(name: str, action_mapper: DiscreteActionMapper, seed: int | None = None) -> BaselinePolicy:
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomDiscretePolicy(action_mapper=action_mapper, seed=seed)
    if normalized == "safe":
        return SafeDiscretePolicy(action_mapper=action_mapper)
    if normalized == "greedy":
        return GreedyDiscretePolicy(action_mapper=action_mapper)
    raise ValueError(f"Unsupported baseline policy: {name}")
