from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cross-run fixed-pool win-rate and Elo evolution.")
    parser.add_argument("report", type=Path, help="fixed_pool_eval_report.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.report.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_plot_rows(report)
    win_path = output_dir / "mean_fixed_pool_win_rate_evolution.png"
    elo_path = output_dir / "elo_evolution.png"
    combined_path = output_dir / "fixed_pool_winrate_elo_evolution.png"

    plot_metric(
        rows,
        key="mean_fixed_pool_win_rate",
        ylabel="Mean win rate vs fixed pool",
        title="Fixed-pool win-rate evolution",
        output_path=win_path,
        y_limits=(0.0, 1.0),
        as_percent=True,
    )
    plot_metric(
        rows,
        key="elo",
        ylabel="Elo",
        title="Fixed-pool Elo evolution",
        output_path=elo_path,
    )
    plot_combined(rows, output_path=combined_path)

    print(f"win_rate_plot: {win_path}")
    print(f"elo_plot: {elo_path}")
    print(f"combined_plot: {combined_path}")


def build_plot_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    eval_agents = {str(item["label"]): item for item in report["eval_agents"]}
    opponent_labels = {str(item["label"]) for item in report["opponent_pool"]}
    pair_by_agent: dict[str, dict[str, float]] = defaultdict(dict)
    for row in report["pair_results"]:
        pair_by_agent[str(row["agent"])][str(row["opponent"])] = float(row["agent_win_rate"])

    elo_by_agent = {str(row["agent"]): float(row["elo"]) for row in report["elo_ratings"]}
    rows: list[dict[str, Any]] = []
    for label, agent in eval_agents.items():
        values = []
        for opponent_label in sorted(opponent_labels):
            if opponent_label == label:
                values.append(0.5)
            else:
                values.append(pair_by_agent[label][opponent_label])
        rows.append(
            {
                "agent": label,
                "run_label": str(agent["run_label"]),
                "step": int(agent["step"]),
                "step_millions": float(agent["step"]) / 1_000_000.0,
                "mean_fixed_pool_win_rate": sum(values) / len(values),
                "elo": elo_by_agent[label],
            }
        )
    rows.sort(key=lambda item: (str(item["run_label"]), float(item["step_millions"])))
    return rows


def rows_by_run(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["run_label"])].append(row)
    return dict(sorted(grouped.items()))


def pretty_label(label: str) -> str:
    return {
        "norecoverycfadv": "No CRA",
        "recoverycfdefault": "CRA",
    }.get(label, label)


def plot_metric(
    rows: list[dict[str, Any]],
    *,
    key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
    as_percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    draw_metric_lines(ax, rows, key=key, as_percent=as_percent)
    ax.set_title(title)
    ax.set_xlabel("Training step (M)")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_combined(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    draw_metric_lines(axes[0], rows, key="mean_fixed_pool_win_rate", as_percent=True)
    axes[0].set_title("Fixed-pool win rate")
    axes[0].set_xlabel("Training step (M)")
    axes[0].set_ylabel("Mean win rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(frameon=False)

    draw_metric_lines(axes[1], rows, key="elo", as_percent=False)
    axes[1].set_title("Fixed-pool Elo")
    axes[1].set_xlabel("Training step (M)")
    axes[1].set_ylabel("Elo")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_metric_lines(ax: Any, rows: list[dict[str, Any]], *, key: str, as_percent: bool) -> None:
    colors = {
        "norecoverycfadv": "#2f7fba",
        "recoverycfdefault": "#d0693a",
    }
    for run_label, run_rows in rows_by_run(rows).items():
        x = [float(row["step_millions"]) for row in run_rows]
        y = [float(row[key]) for row in run_rows]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=5.0,
            color=colors.get(run_label),
            label=pretty_label(run_label),
        )
    if as_percent:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


if __name__ == "__main__":
    main()
