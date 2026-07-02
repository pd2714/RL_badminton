from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze temporal dominance from a win-rate matrix: later checkpoints "
            "against earlier checkpoints after a cutoff."
        )
    )
    parser.add_argument("matrix_json", type=Path, help="Path to win_rate_matrix.json.")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=3.0,
        help="Checkpoint cutoff. Defaults to 3.0, interpreted as million steps unless --cutoff-unit raw is set.",
    )
    parser.add_argument(
        "--cutoff-unit",
        choices=("million", "raw"),
        default="million",
        help="Whether --cutoff is in millions of steps or raw step counts.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--lag-csv", type=Path, default=None)
    parser.add_argument("--lag-plot", type=Path, default=None)
    return parser.parse_args()


def finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def sample_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    if finite.size == 1 or samples <= 0:
        mean = float(finite[0])
        return mean, mean
    draws = rng.choice(finite, size=(samples, finite.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def standard_error(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 0.0 if finite.size == 1 else None
    return float(np.std(finite, ddof=1) / math.sqrt(finite.size))


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    required = {"row_steps", "col_steps", "win_rate_matrix"}
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    return report


def collect_later_earlier_pairs(
    report: dict[str, Any],
    *,
    cutoff_steps: int,
) -> list[dict[str, Any]]:
    row_steps = [int(step) for step in report["row_steps"]]
    col_steps = [None if step is None else int(step) for step in report["col_steps"]]
    row_labels = list(report.get("row_labels", [str(step) for step in row_steps]))
    col_labels = list(report.get("col_labels", [str(step) for step in col_steps]))
    matrix = np.asarray(report["win_rate_matrix"], dtype=float)
    if matrix.shape != (len(row_steps), len(col_steps)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match row/column steps "
            f"({len(row_steps)}, {len(col_steps)})"
        )

    pairs: list[dict[str, Any]] = []
    for i, later_step in enumerate(row_steps):
        if later_step < cutoff_steps:
            continue
        for j, earlier_step in enumerate(col_steps):
            if earlier_step is None or earlier_step < cutoff_steps or later_step <= earlier_step:
                continue
            value = float(matrix[i, j])
            if not math.isfinite(value):
                continue
            pairs.append(
                {
                    "later_label": row_labels[i],
                    "later_step": later_step,
                    "earlier_label": col_labels[j],
                    "earlier_step": earlier_step,
                    "lag_steps": later_step - earlier_step,
                    "win_rate": value,
                }
            )
    return pairs


def write_lag_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "lag_steps",
        "lag_million_steps",
        "pair_count",
        "mean_win_rate",
        "standard_error",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_lag_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = np.asarray([row["lag_steps"] / 1_000_000.0 for row in rows], dtype=float)
    y = np.asarray([row["mean_win_rate"] for row in rows], dtype=float)
    low = np.asarray([row["bootstrap_ci_low"] for row in rows], dtype=float)
    high = np.asarray([row["bootstrap_ci_high"] for row in rows], dtype=float)
    yerr = np.vstack([y - low, high - y])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0.5, color="0.25", linewidth=1.2, linestyle="--", label="Even")
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        capsize=3,
        color="tab:blue",
        ecolor="tab:blue",
        alpha=0.95,
        label="Later vs earlier",
    )
    ax.set_xlabel("Temporal lag (million steps)")
    ax.set_ylabel("Mean win rate")
    ax.set_title("Lagged checkpoint improvement")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(
    report: dict[str, Any],
    *,
    cutoff_steps: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = collect_later_earlier_pairs(report, cutoff_steps=cutoff_steps)
    values = np.asarray([pair["win_rate"] for pair in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    mean_ci_low, mean_ci_high = sample_mean_ci(values, rng=rng, samples=bootstrap_samples)

    lag_rows: list[dict[str, Any]] = []
    for lag in sorted({int(pair["lag_steps"]) for pair in pairs}):
        lag_values = np.asarray([pair["win_rate"] for pair in pairs if pair["lag_steps"] == lag], dtype=float)
        ci_low, ci_high = sample_mean_ci(lag_values, rng=rng, samples=bootstrap_samples)
        lag_rows.append(
            {
                "lag_steps": lag,
                "lag_million_steps": lag / 1_000_000.0,
                "pair_count": int(lag_values.size),
                "mean_win_rate": finite_mean(lag_values),
                "standard_error": standard_error(lag_values),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )

    summary = {
        "definition": (
            "For all finite pairs (i, j) with step_i > step_j and both steps >= cutoff, "
            "M[i, j] is treated as the later checkpoint's win rate against the earlier checkpoint."
        ),
        "cutoff_steps": cutoff_steps,
        "pair_count": int(values.size),
        "later_beats_earlier_mean_win_rate": finite_mean(values),
        "later_beats_earlier_bootstrap_ci_low": mean_ci_low,
        "later_beats_earlier_bootstrap_ci_high": mean_ci_high,
        "temporal_dominance_fraction_gt_0_5": float(np.mean(values > 0.5)) if values.size else None,
        "temporal_dominance_fraction_gt_0_55": float(np.mean(values > 0.55)) if values.size else None,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "lag_count": len(lag_rows),
    }
    return summary, lag_rows


def main() -> None:
    args = parse_args()
    cutoff_steps = int(round(args.cutoff * 1_000_000)) if args.cutoff_unit == "million" else int(round(args.cutoff))
    report = load_report(args.matrix_json)
    output_dir = args.matrix_json.parent
    summary_path = args.summary_json or (output_dir / "temporal_dominance_summary.json")
    lag_csv_path = args.lag_csv or (output_dir / "temporal_lag_curve.csv")
    lag_plot_path = args.lag_plot or (output_dir / "temporal_lag_curve.png")
    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    summary, lag_rows = analyze(
        report,
        cutoff_steps=cutoff_steps,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    write_lag_csv(lag_csv_path, lag_rows)
    plot_lag_curve(lag_plot_path, lag_rows)

    print(f"pairs: {summary['pair_count']}")
    print(f"mean: {summary['later_beats_earlier_mean_win_rate']:.6f}")
    print(
        "95% bootstrap CI: "
        f"[{summary['later_beats_earlier_bootstrap_ci_low']:.6f}, "
        f"{summary['later_beats_earlier_bootstrap_ci_high']:.6f}]"
    )
    print(f"fraction > 0.50: {summary['temporal_dominance_fraction_gt_0_5']:.6f}")
    print(f"fraction > 0.55: {summary['temporal_dominance_fraction_gt_0_55']:.6f}")
    print(f"summary json: {summary_path}")
    print(f"lag csv: {lag_csv_path}")
    print(f"lag plot: {lag_plot_path}")


if __name__ == "__main__":
    main()
