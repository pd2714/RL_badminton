from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from badminton1d.evaluation import ModelSelector, evaluate_selector
from badminton1d.utils import ensure_directory


def _histogram_from_metrics(metrics: dict[str, Any], prefix: str) -> np.ndarray:
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


class RallyDiagnosticsCallback(BaseCallback):
    def __init__(self, output_dir: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.output_dir = output_dir
        self.completed_episodes = 0
        self.win_sum = 0.0
        self.length_sum = 0.0
        self.invalid_sum = 0.0
        self.reward_sum = 0.0
        self.loop_penalty_sum = 0.0
        self.pressure_reward_sum = 0.0
        self.attack_reward_sum = 0.0
        self.defensive_lift_reward_sum = 0.0
        self.intercept_flight_ratio_reward_sum = 0.0
        self.feasible_pressure_reward_sum = 0.0
        self.no_feasible_intercept_bonus_sum = 0.0
        self.opponent_intercept_penalty_sum = 0.0
        self.stage_penalty_sum = 0.0
        self.stall_penalty_sum = 0.0
        self.max_streak_sum = 0.0
        self.avg_streak_sum = 0.0
        self.hitter_hist: np.ndarray | None = None
        self.intercept_hist: np.ndarray | None = None
        self.tactic_zone_names: list[str] = []
        self.tactic_angle_names: list[str] = []
        self.tactic_power_names: list[str] = []
        self.tactic_shot_names: list[str] = []
        self.tactic_zone_hist: np.ndarray | None = None
        self.tactic_angle_hist: np.ndarray | None = None
        self.tactic_power_hist: np.ndarray | None = None
        self.tactic_shot_hist: np.ndarray | None = None
        self.tactic_lookup_valid_sum = 0.0
        self.tactic_lookup_fallback_sum = 0.0
        self.recovery_counterfactual_count = 0
        self.recovery_rank_sum = 0.0
        self.recovery_rank_fraction_sum = 0.0
        self.recovery_chosen_above_average_sum = 0.0
        self.recovery_chosen_best_sum = 0.0
        self.recovery_a_rec_sum = 0.0
        self.recovery_a_rec_sq_sum = 0.0
        self.recovery_a_rec_min = float("inf")
        self.recovery_a_rec_max = float("-inf")
        self.recovery_training_advantage_sum = 0.0
        self.recovery_training_advantage_sq_sum = 0.0
        self.recovery_training_advantage_min = float("inf")
        self.recovery_training_advantage_max = float("-inf")
        self.recovery_bin_x_count = 0
        self.recovery_bin_y_count = 0
        self.recovery_no_feasible_counts: np.ndarray | None = None
        self.recovery_bin_counts: np.ndarray | None = None
        self.recovery_grid_samples: list[dict[str, Any]] = []
        self.max_recovery_grid_samples = 8
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            self._accumulate_recovery_diagnostics(info)
            metrics = info.get("badminton_metrics")
            if metrics is None:
                continue
            self.completed_episodes += 1
            self.win_sum += float(metrics["rally_won"])
            self.length_sum += float(metrics["rally_length"])
            self.invalid_sum += float(metrics["invalid_action_rate"])
            self.loop_penalty_sum += float(metrics.get("loop_penalty_total", 0.0))
            self.pressure_reward_sum += float(metrics.get("pressure_reward_total", 0.0))
            self.attack_reward_sum += float(metrics.get("attack_reward_total", 0.0))
            self.defensive_lift_reward_sum += float(metrics.get("defensive_lift_reward_total", 0.0))
            self.intercept_flight_ratio_reward_sum += float(metrics.get("intercept_flight_ratio_reward_total", 0.0))
            self.feasible_pressure_reward_sum += float(metrics.get("feasible_pressure_reward_total", 0.0))
            self.no_feasible_intercept_bonus_sum += float(metrics.get("no_feasible_intercept_bonus_total", 0.0))
            self.opponent_intercept_penalty_sum += float(metrics.get("opponent_intercept_penalty_total", 0.0))
            self.stage_penalty_sum += float(metrics.get("stage_penalty_total", 0.0))
            self.stall_penalty_sum += float(metrics.get("stall_penalty_total", 0.0))
            self.max_streak_sum += float(metrics.get("max_repeated_action_streak", 0.0))
            self.avg_streak_sum += float(metrics.get("avg_repeated_action_streak", 0.0))
            episode_info = info.get("episode", {})
            self.reward_sum += float(episode_info.get("r", 0.0))

            hitter_hist = _histogram_from_metrics(metrics, "hitter_action_hist")
            intercept_hist = _histogram_from_metrics(metrics, "intercept_hist")
            if self.hitter_hist is None:
                self.hitter_hist = np.zeros_like(hitter_hist)
            if self.intercept_hist is None:
                self.intercept_hist = np.zeros_like(intercept_hist)
            self.hitter_hist += hitter_hist
            self.intercept_hist += intercept_hist
            self._accumulate_tactic_metrics(metrics)
        return True

    def _on_rollout_end(self) -> None:
        if self.completed_episodes == 0:
            return
        diagnostics = {
            "timesteps": int(self.num_timesteps),
            "episodes": self.completed_episodes,
            "rally_win_rate": self.win_sum / self.completed_episodes,
            "avg_rally_length": self.length_sum / self.completed_episodes,
            "avg_invalid_action_rate": self.invalid_sum / self.completed_episodes,
            "avg_episode_reward": self.reward_sum / self.completed_episodes,
            "avg_loop_penalty": self.loop_penalty_sum / self.completed_episodes,
            "avg_pressure_reward": self.pressure_reward_sum / self.completed_episodes,
            "avg_attack_reward": self.attack_reward_sum / self.completed_episodes,
            "avg_defensive_lift_reward": self.defensive_lift_reward_sum / self.completed_episodes,
            "avg_intercept_flight_ratio_reward": self.intercept_flight_ratio_reward_sum / self.completed_episodes,
            "avg_feasible_pressure_reward": self.feasible_pressure_reward_sum / self.completed_episodes,
            "avg_no_feasible_intercept_bonus": self.no_feasible_intercept_bonus_sum / self.completed_episodes,
            "avg_opponent_intercept_penalty": self.opponent_intercept_penalty_sum / self.completed_episodes,
            "avg_stage_penalty": self.stage_penalty_sum / self.completed_episodes,
            "avg_stall_penalty": self.stall_penalty_sum / self.completed_episodes,
            "avg_max_repeated_action_streak": self.max_streak_sum / self.completed_episodes,
            "avg_repeated_action_streak": self.avg_streak_sum / self.completed_episodes,
            "hitter_action_hist": [] if self.hitter_hist is None else self.hitter_hist.astype(int).tolist(),
            "intercept_hist": [] if self.intercept_hist is None else self.intercept_hist.astype(int).tolist(),
            "tactic_zone_names": list(self.tactic_zone_names),
            "tactic_angle_names": list(self.tactic_angle_names),
            "tactic_power_names": list(self.tactic_power_names),
            "tactic_shot_names": list(self.tactic_shot_names),
            "tactic_zone_hist": [] if self.tactic_zone_hist is None else self.tactic_zone_hist.astype(int).tolist(),
            "tactic_angle_hist": [] if self.tactic_angle_hist is None else self.tactic_angle_hist.astype(int).tolist(),
            "tactic_power_hist": [] if self.tactic_power_hist is None else self.tactic_power_hist.astype(int).tolist(),
            "tactic_shot_hist": [] if self.tactic_shot_hist is None else self.tactic_shot_hist.astype(int).tolist(),
            "avg_tactic_lookup_valid_count": self.tactic_lookup_valid_sum / self.completed_episodes,
            "avg_tactic_lookup_fallback_count": self.tactic_lookup_fallback_sum / self.completed_episodes,
        }
        diagnostics.update(self._recovery_diagnostics_payload())
        diagnostics["tactic_zone_frequency"] = self._hist_to_frequency_dict(self.tactic_zone_names, self.tactic_zone_hist)
        diagnostics["tactic_angle_frequency"] = self._hist_to_frequency_dict(self.tactic_angle_names, self.tactic_angle_hist)
        diagnostics["tactic_power_frequency"] = self._hist_to_frequency_dict(self.tactic_power_names, self.tactic_power_hist)
        diagnostics["tactic_shot_frequency"] = self._hist_to_frequency_dict(self.tactic_shot_names, self.tactic_shot_hist)
        self.logger.record("badminton/rally_win_rate", diagnostics["rally_win_rate"])
        self.logger.record("badminton/avg_rally_length", diagnostics["avg_rally_length"])
        self.logger.record("badminton/avg_invalid_action_rate", diagnostics["avg_invalid_action_rate"])
        self.logger.record("badminton/avg_episode_reward", diagnostics["avg_episode_reward"])
        self.logger.record("badminton/avg_loop_penalty", diagnostics["avg_loop_penalty"])
        self.logger.record("badminton/avg_pressure_reward", diagnostics["avg_pressure_reward"])
        self.logger.record("badminton/avg_attack_reward", diagnostics["avg_attack_reward"])
        self.logger.record("badminton/avg_defensive_lift_reward", diagnostics["avg_defensive_lift_reward"])
        self.logger.record("badminton/avg_intercept_flight_ratio_reward", diagnostics["avg_intercept_flight_ratio_reward"])
        self.logger.record("badminton/avg_feasible_pressure_reward", diagnostics["avg_feasible_pressure_reward"])
        self.logger.record("badminton/avg_no_feasible_intercept_bonus", diagnostics["avg_no_feasible_intercept_bonus"])
        self.logger.record("badminton/avg_opponent_intercept_penalty", diagnostics["avg_opponent_intercept_penalty"])
        self.logger.record("badminton/avg_stage_penalty", diagnostics["avg_stage_penalty"])
        self.logger.record("badminton/avg_stall_penalty", diagnostics["avg_stall_penalty"])
        self.logger.record("badminton/avg_max_repeated_action_streak", diagnostics["avg_max_repeated_action_streak"])
        self.logger.record("badminton/avg_repeated_action_streak", diagnostics["avg_repeated_action_streak"])
        self.logger.record("badminton/avg_tactic_lookup_valid_count", diagnostics["avg_tactic_lookup_valid_count"])
        self.logger.record("badminton/avg_tactic_lookup_fallback_count", diagnostics["avg_tactic_lookup_fallback_count"])
        for name, value in diagnostics["tactic_shot_frequency"].items():
            self.logger.record(f"badminton/tactic_shot_{name}", value)
        if self.recovery_counterfactual_count > 0:
            self.logger.record("badminton/recovery_chosen_mean_rank", diagnostics["recovery_chosen_mean_rank"])
            self.logger.record(
                "badminton/recovery_chosen_mean_rank_fraction",
                diagnostics["recovery_chosen_mean_rank_fraction"],
            )
            self.logger.record(
                "badminton/recovery_chosen_above_average_fraction",
                diagnostics["recovery_chosen_above_average_fraction"],
            )
            self.logger.record("badminton/recovery_chosen_best_fraction", diagnostics["recovery_chosen_best_fraction"])
            self.logger.record("badminton/recovery_a_rec_mean", diagnostics["recovery_a_rec_mean"])
            self.logger.record("badminton/recovery_a_rec_std", diagnostics["recovery_a_rec_std"])
            self.logger.record("badminton/recovery_a_rec_min", diagnostics["recovery_a_rec_min"])
            self.logger.record("badminton/recovery_a_rec_max", diagnostics["recovery_a_rec_max"])

        ensure_directory(self.output_dir)
        self.history.append(diagnostics)
        (self.output_dir / "rollout_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        (self.output_dir / "rollout_diagnostics_history.json").write_text(
            json.dumps(self.history, indent=2),
            encoding="utf-8",
        )

    def _accumulate_tactic_metrics(self, metrics: dict[str, Any]) -> None:
        zone_names = list(metrics.get("tactic_zone_names", []))
        angle_names = list(metrics.get("tactic_angle_names", []))
        power_names = list(metrics.get("tactic_power_names", []))
        shot_names = list(metrics.get("tactic_shot_names", []))
        zone_hist = np.asarray(metrics.get("tactic_zone_hist", []), dtype=np.int64)
        angle_hist = np.asarray(metrics.get("tactic_angle_hist", []), dtype=np.int64)
        power_hist = np.asarray(metrics.get("tactic_power_hist", []), dtype=np.int64)
        shot_hist = np.asarray(metrics.get("tactic_shot_hist", []), dtype=np.int64)

        if zone_names:
            self.tactic_zone_names = zone_names
            if self.tactic_zone_hist is None:
                self.tactic_zone_hist = np.zeros_like(zone_hist)
            self.tactic_zone_hist += zone_hist
        if angle_names:
            self.tactic_angle_names = angle_names
            if self.tactic_angle_hist is None:
                self.tactic_angle_hist = np.zeros_like(angle_hist)
            self.tactic_angle_hist += angle_hist
        if power_names:
            self.tactic_power_names = power_names
            if self.tactic_power_hist is None:
                self.tactic_power_hist = np.zeros_like(power_hist)
            self.tactic_power_hist += power_hist
        if shot_names:
            self.tactic_shot_names = shot_names
            if self.tactic_shot_hist is None:
                self.tactic_shot_hist = np.zeros_like(shot_hist)
            self.tactic_shot_hist += shot_hist
        self.tactic_lookup_valid_sum += float(metrics.get("tactic_lookup_valid_count", 0.0))
        self.tactic_lookup_fallback_sum += float(metrics.get("tactic_lookup_fallback_count", 0.0))

    def _accumulate_recovery_diagnostics(self, info: dict[str, Any]) -> None:
        recovery = info.get("recovery_factorized_diagnostics")
        if not isinstance(recovery, dict):
            return
        self.recovery_counterfactual_count += 1
        self.recovery_rank_sum += float(recovery.get("chosen_rank", 0.0))
        self.recovery_rank_fraction_sum += float(recovery.get("chosen_rank_fraction", 0.0))
        self.recovery_chosen_above_average_sum += 1.0 if recovery.get("chosen_above_average") else 0.0
        self.recovery_chosen_best_sum += 1.0 if recovery.get("chosen_best") else 0.0

        a_rec = float(recovery.get("a_rec", 0.0))
        self.recovery_a_rec_sum += a_rec
        self.recovery_a_rec_sq_sum += a_rec * a_rec
        self.recovery_a_rec_min = min(self.recovery_a_rec_min, a_rec)
        self.recovery_a_rec_max = max(self.recovery_a_rec_max, a_rec)

        training_advantage = float(recovery.get("training_recovery_advantage", a_rec))
        self.recovery_training_advantage_sum += training_advantage
        self.recovery_training_advantage_sq_sum += training_advantage * training_advantage
        self.recovery_training_advantage_min = min(self.recovery_training_advantage_min, training_advantage)
        self.recovery_training_advantage_max = max(self.recovery_training_advantage_max, training_advantage)

        no_feasible_grid = np.asarray(recovery.get("no_feasible_grid", []), dtype=np.float64)
        if no_feasible_grid.ndim == 2 and no_feasible_grid.size > 0:
            flat_no_feasible = no_feasible_grid.reshape(-1)
            if self.recovery_no_feasible_counts is None or self.recovery_no_feasible_counts.shape != flat_no_feasible.shape:
                self.recovery_no_feasible_counts = np.zeros_like(flat_no_feasible)
                self.recovery_bin_counts = np.zeros_like(flat_no_feasible)
            self.recovery_no_feasible_counts += flat_no_feasible
            assert self.recovery_bin_counts is not None
            self.recovery_bin_counts += 1.0
            self.recovery_bin_x_count = int(no_feasible_grid.shape[0])
            self.recovery_bin_y_count = int(no_feasible_grid.shape[1])

        if len(self.recovery_grid_samples) < self.max_recovery_grid_samples:
            sample = {
                "timesteps": int(self.num_timesteps),
                "chosen_flat_index": int(recovery.get("chosen_flat_index", -1)),
                "chosen_x_index": int(recovery.get("chosen_x_index", -1)),
                "chosen_y_index": int(recovery.get("chosen_y_index", -1)),
                "chosen_rank": int(recovery.get("chosen_rank", 0)),
                "chosen_above_average": bool(recovery.get("chosen_above_average", False)),
                "chosen_best": bool(recovery.get("chosen_best", False)),
                "a_rec": a_rec,
                "score_grid": recovery.get("score_grid", []),
                "policy_probability_grid": recovery.get("policy_probability_grid", []),
                "no_feasible_grid": recovery.get("no_feasible_grid", []),
            }
            self.recovery_grid_samples.append(sample)

    def _recovery_diagnostics_payload(self) -> dict[str, Any]:
        count = self.recovery_counterfactual_count
        if count <= 0:
            return {
                "recovery_counterfactual_count": 0,
                "recovery_grid_samples": [],
            }

        a_rec_mean = self.recovery_a_rec_sum / count
        a_rec_var = max(self.recovery_a_rec_sq_sum / count - a_rec_mean * a_rec_mean, 0.0)
        training_mean = self.recovery_training_advantage_sum / count
        training_var = max(self.recovery_training_advantage_sq_sum / count - training_mean * training_mean, 0.0)
        payload: dict[str, Any] = {
            "recovery_counterfactual_count": count,
            "recovery_chosen_mean_rank": self.recovery_rank_sum / count,
            "recovery_chosen_mean_rank_fraction": self.recovery_rank_fraction_sum / count,
            "recovery_chosen_above_average_fraction": self.recovery_chosen_above_average_sum / count,
            "recovery_chosen_best_fraction": self.recovery_chosen_best_sum / count,
            "recovery_a_rec_mean": a_rec_mean,
            "recovery_a_rec_std": float(np.sqrt(a_rec_var)),
            "recovery_a_rec_min": self.recovery_a_rec_min,
            "recovery_a_rec_max": self.recovery_a_rec_max,
            "recovery_training_advantage_mean": training_mean,
            "recovery_training_advantage_std": float(np.sqrt(training_var)),
            "recovery_training_advantage_min": self.recovery_training_advantage_min,
            "recovery_training_advantage_max": self.recovery_training_advantage_max,
            "recovery_grid_samples": list(self.recovery_grid_samples),
        }
        if self.recovery_no_feasible_counts is not None and self.recovery_bin_counts is not None:
            rates = np.divide(
                self.recovery_no_feasible_counts,
                np.maximum(self.recovery_bin_counts, 1.0),
            )
            payload["recovery_no_feasible_count_by_bin"] = self.recovery_no_feasible_counts.astype(float).tolist()
            payload["recovery_bin_count_by_bin"] = self.recovery_bin_counts.astype(float).tolist()
            payload["recovery_no_feasible_rate_by_bin"] = rates.astype(float).tolist()
            if self.recovery_bin_x_count > 0 and self.recovery_bin_y_count > 0:
                shape = (self.recovery_bin_x_count, self.recovery_bin_y_count)
                payload["recovery_no_feasible_rate_grid"] = rates.reshape(shape).astype(float).tolist()
        return payload

    def _hist_to_frequency_dict(self, names: list[str], hist: np.ndarray | None) -> dict[str, float]:
        if hist is None or hist.size == 0 or not names:
            return {}
        total = max(float(hist.sum()), 1.0)
        return {name: float(count / total) for name, count in zip(names, hist.tolist())}


class EntropyScheduleCallback(BaseCallback):
    def __init__(
        self,
        *,
        ent_coef_initial: float,
        ent_coef_final: float,
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.ent_coef_initial = float(ent_coef_initial)
        self.ent_coef_final = float(ent_coef_final)
        self.total_timesteps = max(int(total_timesteps), 1)

    def _current_ent_coef(self) -> float:
        progress = min(max(self.num_timesteps / self.total_timesteps, 0.0), 1.0)
        return float(self.ent_coef_initial + progress * (self.ent_coef_final - self.ent_coef_initial))

    def _on_step(self) -> bool:
        current_ent_coef = self._current_ent_coef()
        self.model.ent_coef = current_ent_coef
        self.logger.record("train/ent_coef", current_ent_coef)
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("train/ent_coef", self._current_ent_coef())


class SafeWinRateEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: Path,
        log_path: Path,
        deterministic: bool = False,
        eval_seed: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = max(int(n_eval_episodes), 1)
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.eval_seed = int(eval_seed)
        self.best_win_rate = float("-inf")
        self.last_eval_timestep = 0
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps

        selector = ModelSelector(model=self.model, deterministic=self.deterministic)
        summary, _ = evaluate_selector(
            "ppo_model",
            selector,
            self.eval_env,
            self.n_eval_episodes,
            self.eval_seed + self.num_timesteps,
        )
        win_rate = float(summary["win_rate"])
        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/avg_reward", float(summary["avg_reward"]))
        self.logger.record("eval/avg_rally_length", float(summary["avg_rally_length"]))
        self.logger.record("eval/avg_invalid_action_rate", float(summary["avg_invalid_action_rate"]))

        ensure_directory(self.log_path)
        payload = {
            "num_timesteps": int(self.num_timesteps),
            "best_win_rate": float(max(self.best_win_rate, win_rate)),
            "summary": summary,
        }
        self.history.append(payload)
        (self.log_path / "safe_eval_history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")

        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            ensure_directory(self.best_model_save_path)
            self.model.save(self.best_model_save_path / "best_model.zip")
        return True
