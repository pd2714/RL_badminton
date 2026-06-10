from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from stable_baselines3 import PPO
from gymnasium import spaces

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import landing_position
from badminton1d.opponents import BaselinePolicy, DecisionContext
from badminton1d.policy import apply_receiver_action_mask
from badminton1d.rl_env import BadmintonRLEnv


class ActionSelector(Protocol):
    def choose_action(self, observation: np.ndarray, context: DecisionContext) -> int:
        ...


def adapt_observation_to_model(model: PPO, observation: np.ndarray) -> np.ndarray:
    expected_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not isinstance(expected_shape, (tuple, list)) or not expected_shape:
        return observation
    expected = int(expected_shape[0])
    obs = np.asarray(observation, dtype=np.float32)
    actual = int(obs.shape[-1])
    if actual == expected:
        return obs

    if actual > expected:
        if actual - expected == 4:
            return obs[..., :expected]
        if actual - expected == 8:
            return np.concatenate([obs[..., :29], obs[..., 33:-4]], axis=-1)
        return obs[..., :expected]

    adapted_shape = (*obs.shape[:-1], expected)
    adapted = np.zeros(adapted_shape, dtype=np.float32)
    if expected - actual == 4:
        adapted[..., :actual] = obs
    elif expected - actual == 8:
        adapted[..., :29] = obs[..., :29]
        adapted[..., 33:-4] = obs[..., 29:]
    else:
        adapted[..., :actual] = obs
    return adapted


def choose_model_action(
    model: PPO,
    observation: np.ndarray,
    context: DecisionContext,
    *,
    deterministic: bool = True,
) -> int:
    observation = adapt_observation_to_model(model, observation)
    if isinstance(model.action_space, spaces.Box):
        action, _ = model.predict(observation, deterministic=deterministic)
        if context.role == "receiver":
            values = np.asarray(action, dtype=float).reshape(-1)
            receiver_value = float(values[5]) if values.size > 5 else 0.0
            unit = float(np.clip(0.5 * (receiver_value + 1.0), 0.0, 1.0))
            return int(np.clip(round(unit * (context.receiver_action_count - 1)), 0, context.receiver_action_count - 1))
        return action
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

    def choose_action(self, observation: np.ndarray, context: DecisionContext) -> object:
        if isinstance(self.model.action_space, spaces.Box):
            observation = adapt_observation_to_model(self.model, observation)
            action, _ = self.model.predict(observation, deterministic=self.deterministic)
            return action
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
        "config": env.config,
        "server": info["server"],
        "train_side": info["train_side"],
        "opponent_label": info.get("opponent_label"),
        "truncated": truncated,
    }


def _histogram_from_metrics(metrics: dict[str, object], prefix: str) -> np.ndarray:
    dense = metrics.get(prefix)
    if isinstance(dense, list) and dense:
        return np.asarray(dense, dtype=np.int64)

    size = int(metrics.get(f"{prefix}_size", 0) or 0)
    hist = np.zeros(size, dtype=np.int64)
    indices = np.asarray(metrics.get(f"{prefix}_indices", []), dtype=np.int64)
    counts = np.asarray(metrics.get(f"{prefix}_counts", []), dtype=np.int64)
    if indices.size and counts.size:
        valid = (indices >= 0) & (indices < size)
        hist[indices[valid]] = counts[valid]
    return hist


def summarize_episodes(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("results must not be empty")
    first_metrics = results[0].get("metrics", {})
    first_hitter_hist = _histogram_from_metrics(first_metrics, "hitter_action_hist") if isinstance(first_metrics, dict) else np.zeros(0, dtype=np.int64)
    first_intercept_hist = _histogram_from_metrics(first_metrics, "intercept_hist") if isinstance(first_metrics, dict) else np.zeros(0, dtype=np.int64)
    hitter_size = int(first_hitter_hist.size)
    intercept_size = int(first_intercept_hist.size)
    hitter_hist = np.zeros(hitter_size, dtype=np.int64)
    intercept_hist = np.zeros(intercept_size, dtype=np.int64)
    tactic_zone_names = list(first_metrics.get("tactic_zone_names", [])) if isinstance(first_metrics, dict) else []
    tactic_angle_names = list(first_metrics.get("tactic_angle_names", [])) if isinstance(first_metrics, dict) else []
    tactic_power_names = list(first_metrics.get("tactic_power_names", [])) if isinstance(first_metrics, dict) else []
    tactic_shot_names = list(first_metrics.get("tactic_shot_names", [])) if isinstance(first_metrics, dict) else []
    tactic_zone_hist = np.zeros(len(tactic_zone_names), dtype=np.int64)
    tactic_angle_hist = np.zeros(len(tactic_angle_names), dtype=np.int64)
    tactic_power_hist = np.zeros(len(tactic_power_names), dtype=np.int64)
    tactic_shot_hist = np.zeros(len(tactic_shot_names), dtype=np.int64)
    tactic_lookup_valid = 0.0
    tactic_lookup_fallback = 0.0
    max_streaks: list[float] = []
    avg_streaks: list[float] = []
    for item in results:
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        if hitter_size:
            hitter_hist += _histogram_from_metrics(metrics, "hitter_action_hist")
        if intercept_size:
            intercept_hist += _histogram_from_metrics(metrics, "intercept_hist")
        if tactic_zone_hist.size:
            tactic_zone_hist += np.asarray(metrics.get("tactic_zone_hist", []), dtype=np.int64)
        if tactic_angle_hist.size:
            tactic_angle_hist += np.asarray(metrics.get("tactic_angle_hist", []), dtype=np.int64)
        if tactic_power_hist.size:
            tactic_power_hist += np.asarray(metrics.get("tactic_power_hist", []), dtype=np.int64)
        if tactic_shot_hist.size:
            tactic_shot_hist += np.asarray(metrics.get("tactic_shot_hist", []), dtype=np.int64)
        tactic_lookup_valid += float(metrics.get("tactic_lookup_valid_count", 0.0))
        tactic_lookup_fallback += float(metrics.get("tactic_lookup_fallback_count", 0.0))
        max_streaks.append(float(metrics.get("max_repeated_action_streak", 0.0)))
        avg_streaks.append(float(metrics.get("avg_repeated_action_streak", 0.0)))
    summary = {
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
    summary["tactic_zone_names"] = tactic_zone_names
    summary["tactic_angle_names"] = tactic_angle_names
    summary["tactic_power_names"] = tactic_power_names
    summary["tactic_shot_names"] = tactic_shot_names
    summary["tactic_zone_hist"] = tactic_zone_hist.astype(int).tolist()
    summary["tactic_angle_hist"] = tactic_angle_hist.astype(int).tolist()
    summary["tactic_power_hist"] = tactic_power_hist.astype(int).tolist()
    summary["tactic_shot_hist"] = tactic_shot_hist.astype(int).tolist()
    summary["avg_tactic_lookup_valid_count"] = float(tactic_lookup_valid / max(len(results), 1))
    summary["avg_tactic_lookup_fallback_count"] = float(tactic_lookup_fallback / max(len(results), 1))
    summary["tactic_zone_frequency"] = _hist_to_frequency_dict(tactic_zone_names, tactic_zone_hist)
    summary["tactic_angle_frequency"] = _hist_to_frequency_dict(tactic_angle_names, tactic_angle_hist)
    summary["tactic_power_frequency"] = _hist_to_frequency_dict(tactic_power_names, tactic_power_hist)
    summary["tactic_shot_frequency"] = _hist_to_frequency_dict(tactic_shot_names, tactic_shot_hist)
    summary.update(recovery_intended_landing_metrics(results))
    return summary


def recovery_intended_landing_metrics(
    results: list[dict[str, object]],
    config: SimulationConfig | None = None,
) -> dict[str, object]:
    recovery_x: list[float] = []
    recovery_y: list[float] = []
    landing_x: list[float] = []
    landing_y: list[float] = []

    for item in results:
        item_config = config or item.get("config") or SimulationConfig()
        if not isinstance(item_config, SimulationConfig):
            item_config = SimulationConfig()
        records = item.get("records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            state = getattr(record, "state_before", None)
            validated_action = getattr(record, "validated_action", None)
            action = getattr(validated_action, "applied", None)
            if state is None or action is None:
                continue
            x_land, y_land = landing_position(state, action, item_config)
            recovery_x.append(float(action.x_rec))
            recovery_y.append(float(action.y_rec))
            landing_x.append(float(x_land))
            landing_y.append(float(y_land))

    distances = np.hypot(np.asarray(recovery_x) - np.asarray(landing_x), np.asarray(recovery_y) - np.asarray(landing_y))
    return {
        "recovery_intended_landing_pair_count": len(recovery_x),
        "recovery_intended_landing_x_corr": _pearson_or_none(recovery_x, landing_x),
        "recovery_intended_landing_y_corr": _pearson_or_none(recovery_y, landing_y),
        "recovery_intended_landing_distance_mean": None if distances.size == 0 else float(np.mean(distances)),
    }


def _hist_to_frequency_dict(names: list[str], hist: np.ndarray) -> dict[str, float]:
    if hist.size == 0 or not names:
        return {}
    total = max(float(hist.sum()), 1.0)
    return {name: float(count / total) for name, count in zip(names, hist.tolist())}


def _pearson_or_none(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


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
