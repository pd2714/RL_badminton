from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from badminton1d.evaluation import ModelSelector, evaluate_selector
from badminton1d.utils import ensure_directory


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
        self.stage_penalty_sum = 0.0
        self.stall_penalty_sum = 0.0
        self.max_streak_sum = 0.0
        self.avg_streak_sum = 0.0
        self.hitter_hist: np.ndarray | None = None
        self.intercept_hist: np.ndarray | None = None
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            metrics = info.get("badminton_metrics")
            if metrics is None:
                continue
            self.completed_episodes += 1
            self.win_sum += float(metrics["rally_won"])
            self.length_sum += float(metrics["rally_length"])
            self.invalid_sum += float(metrics["invalid_action_rate"])
            self.loop_penalty_sum += float(metrics.get("loop_penalty_total", 0.0))
            self.pressure_reward_sum += float(metrics.get("pressure_reward_total", 0.0))
            self.stage_penalty_sum += float(metrics.get("stage_penalty_total", 0.0))
            self.stall_penalty_sum += float(metrics.get("stall_penalty_total", 0.0))
            self.max_streak_sum += float(metrics.get("max_repeated_action_streak", 0.0))
            self.avg_streak_sum += float(metrics.get("avg_repeated_action_streak", 0.0))
            episode_info = info.get("episode", {})
            self.reward_sum += float(episode_info.get("r", 0.0))

            hitter_hist = np.asarray(metrics["hitter_action_hist"], dtype=np.int64)
            intercept_hist = np.asarray(metrics["intercept_hist"], dtype=np.int64)
            if self.hitter_hist is None:
                self.hitter_hist = np.zeros_like(hitter_hist)
            if self.intercept_hist is None:
                self.intercept_hist = np.zeros_like(intercept_hist)
            self.hitter_hist += hitter_hist
            self.intercept_hist += intercept_hist
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
            "avg_stage_penalty": self.stage_penalty_sum / self.completed_episodes,
            "avg_stall_penalty": self.stall_penalty_sum / self.completed_episodes,
            "avg_max_repeated_action_streak": self.max_streak_sum / self.completed_episodes,
            "avg_repeated_action_streak": self.avg_streak_sum / self.completed_episodes,
            "hitter_action_hist": [] if self.hitter_hist is None else self.hitter_hist.astype(int).tolist(),
            "intercept_hist": [] if self.intercept_hist is None else self.intercept_hist.astype(int).tolist(),
        }
        self.logger.record("badminton/rally_win_rate", diagnostics["rally_win_rate"])
        self.logger.record("badminton/avg_rally_length", diagnostics["avg_rally_length"])
        self.logger.record("badminton/avg_invalid_action_rate", diagnostics["avg_invalid_action_rate"])
        self.logger.record("badminton/avg_episode_reward", diagnostics["avg_episode_reward"])
        self.logger.record("badminton/avg_loop_penalty", diagnostics["avg_loop_penalty"])
        self.logger.record("badminton/avg_pressure_reward", diagnostics["avg_pressure_reward"])
        self.logger.record("badminton/avg_stage_penalty", diagnostics["avg_stage_penalty"])
        self.logger.record("badminton/avg_stall_penalty", diagnostics["avg_stall_penalty"])
        self.logger.record("badminton/avg_max_repeated_action_streak", diagnostics["avg_max_repeated_action_streak"])
        self.logger.record("badminton/avg_repeated_action_streak", diagnostics["avg_repeated_action_streak"])

        ensure_directory(self.output_dir)
        self.history.append(diagnostics)
        (self.output_dir / "rollout_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        (self.output_dir / "rollout_diagnostics_history.json").write_text(
            json.dumps(self.history, indent=2),
            encoding="utf-8",
        )


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
