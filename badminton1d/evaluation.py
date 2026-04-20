from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from stable_baselines3 import PPO

from badminton1d.opponents import BaselinePolicy, DecisionContext
from badminton1d.policy import apply_receiver_action_mask
from badminton1d.rl_env import BadmintonRLEnv


class ActionSelector(Protocol):
    def choose_action(self, observation: np.ndarray, context: DecisionContext) -> int:
        ...


def choose_model_action(
    model: PPO,
    observation: np.ndarray,
    context: DecisionContext,
    *,
    deterministic: bool = True,
) -> int:
    if context.role != "receiver":
        action, _ = model.predict(observation, deterministic=deterministic)
        return int(action)

    if context.feasible_indices:
        legal_indices = [int(index) for index in context.feasible_indices]
    else:
        legal_indices = list(range(max(int(context.receiver_action_count), 1)))

    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor).distribution
        logits = distribution.logits
        logits = apply_receiver_action_mask(
            logits,
            obs_tensor,
            receiver_action_count=max(int(context.receiver_action_count), 1),
        ).squeeze(0)
        legal_tensor = torch.as_tensor(legal_indices, dtype=torch.long, device=logits.device)
        legal_logits = logits.index_select(0, legal_tensor)
        if deterministic:
            selected = int(torch.argmax(legal_logits).item())
        else:
            probs = torch.softmax(legal_logits, dim=0)
            selected = int(torch.multinomial(probs, 1).item())
    return legal_indices[selected]


@dataclass
class ModelSelector:
    model: PPO
    deterministic: bool = True

    def choose_action(self, observation: np.ndarray, context: DecisionContext) -> int:
        return choose_model_action(self.model, observation, context, deterministic=self.deterministic)


@dataclass
class BaselineSelector:
    policy: BaselinePolicy

    def choose_action(self, observation: np.ndarray, context: DecisionContext) -> int:
        return int(self.policy.choose_action(context))


def rollout_episode(
    env: BadmintonRLEnv,
    selector: ActionSelector,
    *,
    seed: int | None = None,
) -> dict[str, object]:
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action = selector.choose_action(observation, env.current_decision_context())
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    metrics = info["badminton_metrics"]
    return {
        "reward": total_reward,
        "winner": metrics["winner"],
        "rally_won": metrics["rally_won"],
        "rally_length": metrics["rally_length"],
        "invalid_action_rate": metrics["invalid_action_rate"],
        "metrics": metrics,
        "records": list(env.records),
        "server": info["server"],
        "train_side": info["train_side"],
        "opponent_label": info.get("opponent_label"),
        "truncated": truncated,
    }


def summarize_episodes(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("results must not be empty")
    first_metrics = results[0].get("metrics", {})
    hitter_size = len(first_metrics.get("hitter_action_hist", [])) if isinstance(first_metrics, dict) else 0
    intercept_size = len(first_metrics.get("intercept_hist", [])) if isinstance(first_metrics, dict) else 0
    hitter_hist = np.zeros(hitter_size, dtype=np.int64)
    intercept_hist = np.zeros(intercept_size, dtype=np.int64)
    max_streaks: list[float] = []
    avg_streaks: list[float] = []
    for item in results:
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        if hitter_size:
            hitter_hist += np.asarray(metrics.get("hitter_action_hist", []), dtype=np.int64)
        if intercept_size:
            intercept_hist += np.asarray(metrics.get("intercept_hist", []), dtype=np.int64)
        max_streaks.append(float(metrics.get("max_repeated_action_streak", 0.0)))
        avg_streaks.append(float(metrics.get("avg_repeated_action_streak", 0.0)))
    return {
        "episodes": len(results),
        "win_rate": float(np.mean([float(item["rally_won"]) for item in results])),
        "avg_reward": float(np.mean([float(item["reward"]) for item in results])),
        "avg_rally_length": float(np.mean([float(item["rally_length"]) for item in results])),
        "avg_invalid_action_rate": float(np.mean([float(item["invalid_action_rate"]) for item in results])),
        "truncation_rate": float(np.mean([1.0 if bool(item["truncated"]) else 0.0 for item in results])),
        "hitter_action_hist": hitter_hist.astype(int).tolist(),
        "intercept_hist": intercept_hist.astype(int).tolist(),
        "avg_max_repeated_action_streak": float(np.mean(max_streaks)) if max_streaks else 0.0,
        "avg_repeated_action_streak": float(np.mean(avg_streaks)) if avg_streaks else 0.0,
    }


def evaluate_selector(
    name: str,
    selector: ActionSelector,
    env: BadmintonRLEnv,
    episodes: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    results = [rollout_episode(env, selector, seed=seed + episode) for episode in range(episodes)]
    summary = summarize_episodes(results)
    summary["name"] = name
    return summary, results
