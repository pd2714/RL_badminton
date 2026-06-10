from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot how recovery-position choice quality evolves during a self-play run."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Self-play output directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument(
        "--window",
        type=int,
        default=12,
        help="Number of rollout snapshots to smooth recent-window metrics over.",
    )
    return parser.parse_args()


def _as_float_array(history: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(item.get(key, np.nan)) for item in history], dtype=np.float64)


def _per_window_mean(
    counts: np.ndarray,
    means: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    sums = counts * means
    previous_counts = np.zeros_like(counts)
    previous_sums = np.zeros_like(sums)
    if window < counts.size:
        previous_counts[window:] = counts[:-window]
        previous_sums[window:] = sums[:-window]
    delta_counts = counts - previous_counts
    delta_sums = sums - previous_sums
    return np.divide(delta_sums, delta_counts, out=np.full_like(delta_sums, np.nan), where=delta_counts > 0)


def _per_window_rate(counts: np.ndarray, rates: np.ndarray, *, window: int) -> np.ndarray:
    return _per_window_mean(counts, rates, window=window)


def _last_grid(history: Iterable[dict], key: str) -> np.ndarray | None:
    for item in reversed(list(history)):
        grid = item.get(key)
        if isinstance(grid, list) and grid:
            arr = np.asarray(grid, dtype=np.float64)
            if arr.ndim == 2 and arr.size:
                return arr
    return None


def main() -> None:
    args = parse_args()
    history_path = args.run_dir / "rollout_diagnostics_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing rollout diagnostics history: {history_path}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"No diagnostics entries found in {history_path}")

    config_path = args.run_dir / "selfplay_config.json"
    timestep_offset = 0
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        timestep_offset = int(config.get("resume_step_offset", 0) or 0)

    steps = _as_float_array(history, "timesteps")
    counts = _as_float_array(history, "recovery_counterfactual_count")
    valid = np.isfinite(steps) & np.isfinite(counts) & (counts > 0)
    if not np.any(valid):
        raise ValueError("No recovery counterfactual diagnostics found in history.")

    history = [item for item, keep in zip(history, valid.tolist()) if keep]
    steps = steps[valid] + timestep_offset
    counts = counts[valid]
    window = max(int(args.window), 1)

    rank_fraction = _as_float_array(history, "recovery_chosen_mean_rank_fraction")
    above_average = _as_float_array(history, "recovery_chosen_above_average_fraction")
    best = _as_float_array(history, "recovery_chosen_best_fraction")
    a_rec = _as_float_array(history, "recovery_a_rec_mean")
    training_advantage = _as_float_array(history, "recovery_training_advantage_mean")
    recent_rank_fraction = _per_window_mean(counts, rank_fraction, window=window)
    recent_above_average = _per_window_rate(counts, above_average, window=window)
    recent_best = _per_window_rate(counts, best, window=window)
    recent_a_rec = _per_window_mean(counts, a_rec, window=window)
    recent_training_advantage = _per_window_mean(counts, training_advantage, window=window)
    final_no_feasible_grid = _last_grid(history, "recovery_no_feasible_rate_grid")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    ax_rank, ax_fraction, ax_advantage, ax_grid = axes.flat

    ax_rank.plot(steps, rank_fraction, color="0.65", linewidth=1.6, label="cumulative")
    ax_rank.plot(steps, recent_rank_fraction, color="tab:blue", linewidth=2.2, label=f"recent {window} rollouts")
    ax_rank.set_title("Chosen recovery rank fraction")
    ax_rank.set_ylabel("Rank fraction (lower is better)")
    ax_rank.set_ylim(0.0, 1.0)
    ax_rank.grid(True, alpha=0.3)
    ax_rank.legend()

    ax_fraction.plot(steps, recent_above_average, color="tab:green", linewidth=2.2, label="above average")
    ax_fraction.plot(steps, recent_best, color="tab:purple", linewidth=2.2, label="best bin")
    ax_fraction.set_title("Recent choice quality rates")
    ax_fraction.set_ylabel("Fraction")
    ax_fraction.set_ylim(0.0, 1.0)
    ax_fraction.grid(True, alpha=0.3)
    ax_fraction.legend()

    ax_advantage.plot(steps, a_rec, color="0.65", linewidth=1.6, label="a_rec cumulative")
    ax_advantage.plot(steps, recent_a_rec, color="tab:orange", linewidth=2.2, label="a_rec recent")
    ax_advantage.plot(
        steps,
        recent_training_advantage,
        color="tab:red",
        linewidth=2.0,
        linestyle="--",
        label="training advantage recent",
    )
    ax_advantage.axhline(0.0, color="0.2", linewidth=1.0, alpha=0.6)
    ax_advantage.set_title("Recovery advantage")
    ax_advantage.set_xlabel("Model timesteps")
    ax_advantage.set_ylabel("Advantage")
    ax_advantage.grid(True, alpha=0.3)
    ax_advantage.legend()

    if final_no_feasible_grid is None:
        ax_grid.axis("off")
        ax_grid.text(0.5, 0.5, "No recovery bin grid recorded", ha="center", va="center")
    else:
        image = ax_grid.imshow(final_no_feasible_grid.T, origin="lower", vmin=0.0, vmax=1.0, cmap="magma")
        ax_grid.set_title("Final no-feasible rate by recovery bin")
        ax_grid.set_xlabel("x recovery bin")
        ax_grid.set_ylabel("y recovery bin")
        ax_grid.set_xticks(range(final_no_feasible_grid.shape[0]))
        ax_grid.set_yticks(range(final_no_feasible_grid.shape[1]))
        fig.colorbar(image, ax=ax_grid, fraction=0.046, pad=0.04)

    fig.suptitle(f"Recovery position choice evolution: {args.run_dir.name}")
    output_path = args.out or args.run_dir / "recovery_choice_evolution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    print(output_path)


if __name__ == "__main__":
    main()
