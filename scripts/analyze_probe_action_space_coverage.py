from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.mpl_config import ensure_writable_matplotlib_config
from badminton1d.utils import ensure_directory

ensure_writable_matplotlib_config()
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate controlled shot/recovery probe samples into action-space "
            "coverage and distribution-shift statistics across frozen checkpoints."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Self-play run directory.")
    parser.add_argument(
        "--anchor-metric-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/anchor_metric_eval.",
    )
    parser.add_argument(
        "--shot-probe-dir",
        type=Path,
        default=None,
        help="Defaults to ANCHOR_METRIC_DIR/controlled_contact_grid_probe.",
    )
    parser.add_argument(
        "--recovery-probe-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Recovery probe directory. May be repeated. Defaults to all "
            "ANCHOR_METRIC_DIR/recovery_contact_grid_probe_*_latest folders."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to ANCHOR_METRIC_DIR/probe_action_space_coverage.",
    )
    parser.add_argument("--min-probability", type=float, default=0.01)
    parser.add_argument("--smoothing", type=float, default=1e-6)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor_metric_dir = args.anchor_metric_dir or (args.run_dir / "anchor_metric_eval")
    shot_probe_dir = args.shot_probe_dir or (anchor_metric_dir / "controlled_contact_grid_probe")
    recovery_probe_dirs = args.recovery_probe_dir or sorted(anchor_metric_dir.glob("recovery_contact_grid_probe_*_latest"))
    output_dir = args.output_dir or (anchor_metric_dir / "probe_action_space_coverage")
    ensure_directory(output_dir)

    if not recovery_probe_dirs:
        raise FileNotFoundError(f"No recovery probe dirs found under {anchor_metric_dir}")

    shot_state = _shot_state_metrics(
        shot_probe_dir / "controlled_contact_grid_probe_summary.csv",
        shot_probe_dir / "controlled_contact_grid_probe_samples.csv",
        min_probability=float(args.min_probability),
        smoothing=float(args.smoothing),
    )
    recovery_state = _recovery_state_metrics(
        recovery_probe_dirs,
        min_probability=float(args.min_probability),
        smoothing=float(args.smoothing),
    )

    shot_aggregate = _aggregate_state_metrics(shot_state, state_col="probe_id")
    recovery_aggregate = _aggregate_state_metrics(recovery_state, state_col="recovery_context_id")

    shot_state_path = output_dir / "shot_probe_action_space_by_state.csv"
    recovery_state_path = output_dir / "recovery_probe_action_space_by_state.csv"
    shot_aggregate_path = output_dir / "shot_probe_action_space_by_step.csv"
    recovery_aggregate_path = output_dir / "recovery_probe_action_space_by_step.csv"
    shot_state.to_csv(shot_state_path, index=False)
    recovery_state.to_csv(recovery_state_path, index=False)
    shot_aggregate.to_csv(shot_aggregate_path, index=False)
    recovery_aggregate.to_csv(recovery_aggregate_path, index=False)

    plot_path = output_dir / "probe_action_space_coverage_summary.png"
    _write_summary_plot(shot_aggregate, recovery_aggregate, plot_path, dpi=int(args.dpi))

    summary_path = output_dir / "probe_action_space_coverage_summary.json"
    summary = _write_json_summary(
        output_dir=output_dir,
        run_dir=args.run_dir,
        shot_probe_dir=shot_probe_dir,
        recovery_probe_dirs=recovery_probe_dirs,
        shot_state=shot_state,
        recovery_state=recovery_state,
        shot_aggregate=shot_aggregate,
        recovery_aggregate=recovery_aggregate,
        paths={
            "shot_state_csv": shot_state_path,
            "recovery_state_csv": recovery_state_path,
            "shot_aggregate_csv": shot_aggregate_path,
            "recovery_aggregate_csv": recovery_aggregate_path,
            "plot": plot_path,
        },
        min_probability=float(args.min_probability),
        smoothing=float(args.smoothing),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"shot state metrics: {shot_state_path}")
    print(f"recovery state metrics: {recovery_state_path}")
    print(f"shot aggregate: {shot_aggregate_path}")
    print(f"recovery aggregate: {recovery_aggregate_path}")
    print(f"plot: {plot_path}")
    print(f"summary: {summary_path}")


def _shot_state_metrics(
    summary_csv: Path,
    samples_csv: Path,
    *,
    min_probability: float,
    smoothing: float,
) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing shot probe summary: {summary_csv}")
    if not samples_csv.exists():
        raise FileNotFoundError(f"Missing shot probe samples: {samples_csv}")

    summary = pd.read_csv(summary_csv)
    samples = pd.read_csv(samples_csv)
    valid_samples = samples[samples["valid"].astype(bool)].copy()

    shot_cols = [col for col in summary.columns if col.startswith("shot_type_freq_")]
    landing_cols = [col for col in summary.columns if col.startswith("landing_zone_freq_")]

    rows: list[dict[str, Any]] = []
    for (step, probe_id), group in summary.groupby(["step", "probe_id"], sort=True):
        first = group.iloc[0]
        sample_group = valid_samples[(valid_samples["step"] == step) & (valid_samples["probe_id"] == probe_id)]

        shot_dist = group[shot_cols].mean().to_numpy(dtype=float)
        landing_dist = group[landing_cols].mean().to_numpy(dtype=float)
        pair_dist = _categorical_distribution(sample_group, ["shot_type", "landing_zone"])
        xy = sample_group[["landing_x", "landing_y"]].to_numpy(dtype=float) if not sample_group.empty else np.empty((0, 2))

        rows.append(
            {
                "step": int(step),
                "probe_id": str(probe_id),
                "sample_count": int(first.get("sample_count", len(sample_group))),
                "valid_sample_count": int(first.get("valid_sample_count", len(sample_group))),
                "x_region": first.get("x_region"),
                "y_region": first.get("y_region"),
                "z_level": first.get("z_level"),
                "shot_type_entropy_nats": _entropy(shot_dist, smoothing),
                "shot_type_effective_count": _effective_count(shot_dist, smoothing),
                "shot_type_support_count": _support_count(shot_dist, min_probability),
                "landing_zone_entropy_nats": _entropy(landing_dist, smoothing),
                "landing_zone_effective_count": _effective_count(landing_dist, smoothing),
                "landing_zone_support_count": _support_count(landing_dist, min_probability),
                "shot_landing_entropy_nats": _entropy(pair_dist, smoothing),
                "shot_landing_effective_count": _effective_count(pair_dist, smoothing),
                "shot_landing_support_count": _support_count(pair_dist, min_probability),
                "landing_cov_area_95": _covariance_ellipse_area(xy),
                "landing_var_trace": _variance_trace(xy),
                "landing_x_range": _axis_range(xy, 0),
                "landing_y_range": _axis_range(xy, 1),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["probe_id", "step"]).reset_index(drop=True)
    metrics = _add_shift_metrics(
        metrics,
        state_col="probe_id",
        dist_tables={
            "shot_type": _distribution_table(summary, ["step", "probe_id"], shot_cols),
            "landing_zone": _distribution_table(summary, ["step", "probe_id"], landing_cols),
            "shot_landing": _sample_distribution_table(valid_samples, ["step", "probe_id"], ["shot_type", "landing_zone"]),
        },
        smoothing=smoothing,
    )
    return metrics


def _recovery_state_metrics(
    recovery_probe_dirs: list[Path],
    *,
    min_probability: float,
    smoothing: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    distribution_frames: list[pd.DataFrame] = []

    for probe_dir in recovery_probe_dirs:
        summary_paths = sorted(probe_dir.glob("*_probe_summary.csv"))
        sample_paths = sorted(probe_dir.glob("*_probe_samples.csv"))
        if len(summary_paths) != 1 or len(sample_paths) != 1:
            raise FileNotFoundError(f"Expected one summary and one samples CSV in {probe_dir}")
        summary = pd.read_csv(summary_paths[0])
        samples = pd.read_csv(sample_paths[0])
        probe_family = probe_dir.name.replace("recovery_contact_grid_probe_", "").replace("_latest", "")
        samples = samples.copy()
        samples["probe_family"] = probe_family
        samples["recovery_context_id"] = samples["probe_family"].astype(str) + "__" + samples["probe_id"].astype(str)
        summary = summary.copy()
        summary["probe_family"] = probe_family
        summary["recovery_context_id"] = summary["probe_family"].astype(str) + "__" + summary["probe_id"].astype(str)

        distribution_frames.append(
            _sample_distribution_table(samples, ["step", "recovery_context_id"], ["recovery_flat_index"])
        )

        for (step, context_id), group in summary.groupby(["step", "recovery_context_id"], sort=True):
            sample_group = samples[(samples["step"] == step) & (samples["recovery_context_id"] == context_id)]
            rec_dist = _categorical_distribution(sample_group, ["recovery_flat_index"])
            xy = sample_group[["recovery_x", "recovery_y"]].to_numpy(dtype=float) if not sample_group.empty else np.empty((0, 2))
            first = group.iloc[0]
            rows.append(
                {
                    "step": int(step),
                    "recovery_context_id": str(context_id),
                    "probe_family": str(first["probe_family"]),
                    "probe_id": str(first["probe_id"]),
                    "sample_count": int(first.get("sample_count", len(sample_group))),
                    "target_x_region": first.get("target_x_region"),
                    "target_y_region": first.get("target_y_region"),
                    "target_z_level": first.get("target_z_level"),
                    "policy_entropy_nats": float(group["policy_entropy"].mean()),
                    "top_probability": float(group["top_probability"].mean()),
                    "recovery_entropy_nats": _entropy(rec_dist, smoothing),
                    "recovery_effective_count": _effective_count(rec_dist, smoothing),
                    "recovery_support_count": _support_count(rec_dist, min_probability),
                    "recovery_cov_area_95": _covariance_ellipse_area(xy),
                    "recovery_var_trace": _variance_trace(xy),
                    "recovery_x_range": _axis_range(xy, 0),
                    "recovery_y_range": _axis_range(xy, 1),
                    "sampled_best_frequency": float(group["sampled_best_frequency"].mean()),
                    "sampled_rank_fraction_mean": float(group["sampled_rank_fraction_mean"].mean()),
                }
            )

    metrics = pd.DataFrame(rows).sort_values(["recovery_context_id", "step"]).reset_index(drop=True)
    rec_distribution = pd.concat(distribution_frames, ignore_index=True)
    metrics = _add_shift_metrics(
        metrics,
        state_col="recovery_context_id",
        dist_tables={"recovery": rec_distribution},
        smoothing=smoothing,
    )
    return metrics


def _aggregate_state_metrics(metrics: pd.DataFrame, *, state_col: str) -> pd.DataFrame:
    ignore_cols = {"step", state_col}
    numeric_cols = [
        col
        for col in metrics.select_dtypes(include=[np.number]).columns
        if col not in ignore_cols and not col.endswith("_count")
    ]
    count_cols = [
        col
        for col in metrics.select_dtypes(include=[np.number]).columns
        if col.endswith("_count") and col not in {"sample_count", "valid_sample_count"}
    ]
    rows: list[dict[str, Any]] = []
    for step, group in metrics.groupby("step", sort=True):
        row: dict[str, Any] = {"step": int(step), "state_count": int(group[state_col].nunique())}
        for col in numeric_cols + count_cols:
            values = group[col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0))
            row[f"{col}_sem"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{col}_median"] = float(np.median(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def _add_shift_metrics(
    metrics: pd.DataFrame,
    *,
    state_col: str,
    dist_tables: dict[str, pd.DataFrame],
    smoothing: float,
) -> pd.DataFrame:
    out = metrics.copy()
    for name, dist_table in dist_tables.items():
        initial: dict[str, np.ndarray] = {}
        previous: dict[str, np.ndarray] = {}
        by_state_step = {
            (str(row[state_col]), int(row["step"])): row.drop(labels=[state_col, "step"]).to_numpy(dtype=float)
            for _, row in dist_table.iterrows()
        }
        kl_initial: list[float] = []
        js_initial: list[float] = []
        kl_previous: list[float] = []
        entropy_delta: list[float] = []
        for _, row in out.iterrows():
            state = str(row[state_col])
            step = int(row["step"])
            dist = by_state_step.get((state, step))
            if dist is None:
                kl_initial.append(float("nan"))
                js_initial.append(float("nan"))
                kl_previous.append(float("nan"))
                entropy_delta.append(float("nan"))
                continue
            dist = _normalize(dist, smoothing)
            if state not in initial:
                initial[state] = dist
            base = initial[state]
            prev = previous.get(state, dist)
            kl_initial.append(_kl_divergence(dist, base, smoothing))
            js_initial.append(_js_divergence(dist, base, smoothing))
            kl_previous.append(_kl_divergence(dist, prev, smoothing))
            entropy_delta.append(_entropy(dist, smoothing) - _entropy(base, smoothing))
            previous[state] = dist
        out[f"{name}_kl_to_initial_nats"] = kl_initial
        out[f"{name}_js_to_initial_nats"] = js_initial
        out[f"{name}_kl_to_previous_nats"] = kl_previous
        out[f"{name}_entropy_delta_vs_initial_nats"] = entropy_delta
    return out


def _distribution_table(df: pd.DataFrame, group_cols: list[str], dist_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        dist = group[dist_cols].mean().to_numpy(dtype=float)
        row = {col: key for col, key in zip(group_cols, keys)}
        for col, value in zip(dist_cols, dist):
            row[col] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_distribution_table(df: pd.DataFrame, group_cols: list[str], category_cols: list[str]) -> pd.DataFrame:
    categories = sorted(_category_keys(df, category_cols))
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts = group[category_cols].astype(str).agg("|".join, axis=1).value_counts(normalize=True)
        row = {col: key for col, key in zip(group_cols, keys)}
        for category in categories:
            row[category] = float(counts.get(category, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _categorical_distribution(df: pd.DataFrame, category_cols: list[str]) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=float)
    counts = df[category_cols].astype(str).agg("|".join, axis=1).value_counts(normalize=True)
    return counts.to_numpy(dtype=float)


def _category_keys(df: pd.DataFrame, category_cols: list[str]) -> set[str]:
    if df.empty:
        return set()
    return set(df[category_cols].astype(str).agg("|".join, axis=1).unique())


def _normalize(values: np.ndarray, smoothing: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0) + float(smoothing)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(values.shape, 1.0 / len(values), dtype=float)
    return values / total


def _entropy(values: np.ndarray, smoothing: float) -> float:
    probs = _normalize(values, smoothing)
    if probs.size == 0:
        return float("nan")
    return float(-np.sum(probs * np.log(probs)))


def _effective_count(values: np.ndarray, smoothing: float) -> float:
    entropy = _entropy(values, smoothing)
    return float(math.exp(entropy)) if math.isfinite(entropy) else float("nan")


def _support_count(values: np.ndarray, threshold: float) -> int:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0
    total = float(values.sum())
    if total <= 0.0:
        return 0
    probs = values / total
    return int(np.sum(probs >= float(threshold)))


def _kl_divergence(p: np.ndarray, q: np.ndarray, smoothing: float) -> float:
    p = _normalize(p, smoothing)
    q = _normalize(q, smoothing)
    if p.size == 0 or q.size == 0:
        return float("nan")
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _js_divergence(p: np.ndarray, q: np.ndarray, smoothing: float) -> float:
    p = _normalize(p, smoothing)
    q = _normalize(q, smoothing)
    if p.size == 0 or q.size == 0:
        return float("nan")
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, smoothing) + 0.5 * _kl_divergence(q, m, smoothing)


def _covariance_ellipse_area(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return float("nan")
    cov = np.cov(xy, rowvar=False, ddof=0)
    det = float(np.linalg.det(cov))
    if det < 0.0 and abs(det) < 1e-12:
        det = 0.0
    chi2_95 = 5.991464547107979
    return float(math.pi * chi2_95 * math.sqrt(max(det, 0.0)))


def _variance_trace(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return float("nan")
    cov = np.cov(xy, rowvar=False, ddof=0)
    return float(cov[0, 0] + cov[1, 1])


def _axis_range(xy: np.ndarray, axis: int) -> float:
    if xy.shape[0] == 0:
        return float("nan")
    values = xy[:, axis]
    return float(values.max() - values.min())


def _write_summary_plot(shot: pd.DataFrame, recovery: pd.DataFrame, path: Path, *, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharex=True)
    panels = [
        (
            axes[0, 0],
            shot,
            "shot_landing_effective_count",
            "Shot action support",
            "effective shot x landing bins",
            "#2f6fbb",
        ),
        (
            axes[0, 1],
            shot,
            "shot_landing_js_to_initial_nats",
            "Shot distribution shift",
            "JS divergence vs 0-step",
            "#b85c38",
        ),
        (
            axes[1, 0],
            recovery,
            "recovery_effective_count",
            "Recovery-grid support",
            "effective recovery bins",
            "#3d8b62",
        ),
        (
            axes[1, 1],
            recovery,
            "recovery_js_to_initial_nats",
            "Recovery distribution shift",
            "JS divergence vs 0-step",
            "#7d5fb2",
        ),
    ]
    for ax, frame, metric, title, ylabel, color in panels:
        mean_col = f"{metric}_mean"
        sem_col = f"{metric}_sem"
        steps = frame["step"].to_numpy(dtype=float) / 1_000_000.0
        mean = frame[mean_col].to_numpy(dtype=float)
        sem = frame[sem_col].to_numpy(dtype=float) if sem_col in frame else np.zeros_like(mean)
        ax.plot(steps, mean, color=color, linewidth=2.0)
        ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0.0)
        ax.scatter(steps, mean, color=color, s=14)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel("checkpoint step (millions)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _write_json_summary(
    *,
    output_dir: Path,
    run_dir: Path,
    shot_probe_dir: Path,
    recovery_probe_dirs: list[Path],
    shot_state: pd.DataFrame,
    recovery_state: pd.DataFrame,
    shot_aggregate: pd.DataFrame,
    recovery_aggregate: pd.DataFrame,
    paths: dict[str, Path],
    min_probability: float,
    smoothing: float,
) -> dict[str, Any]:
    def row_for_step(frame: pd.DataFrame, step: int) -> dict[str, Any]:
        rows = frame[frame["step"] == step]
        return {} if rows.empty else _json_safe(rows.iloc[0].to_dict())

    shot_steps = sorted(int(step) for step in shot_aggregate["step"].unique())
    recovery_steps = sorted(int(step) for step in recovery_aggregate["step"].unique())
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "shot_probe_dir": str(shot_probe_dir),
        "recovery_probe_dirs": [str(path) for path in recovery_probe_dirs],
        "min_probability_for_support_count": float(min_probability),
        "distribution_smoothing": float(smoothing),
        "shot": {
            "state_count": int(shot_state["probe_id"].nunique()),
            "step_count": int(shot_state["step"].nunique()),
            "first_step": row_for_step(shot_aggregate, min(shot_steps)),
            "last_step": row_for_step(shot_aggregate, max(shot_steps)),
        },
        "recovery": {
            "state_count": int(recovery_state["recovery_context_id"].nunique()),
            "probe_family_count": int(recovery_state["probe_family"].nunique()),
            "step_count": int(recovery_state["step"].nunique()),
            "first_step": row_for_step(recovery_aggregate, min(recovery_steps)),
            "last_step": row_for_step(recovery_aggregate, max(recovery_steps)),
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


if __name__ == "__main__":
    main()
