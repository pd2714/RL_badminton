from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.elo import PairwiseRecord, calculate_elo
from scripts.plot_new_0to6m_pairwise_pure_split_elo import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_DIR


DEFAULT_CROSS_ROOT = Path("outputs/rl/ginsburg_20260622/same_june11_fixed_pool_eval_200r")
PREFIX = "pure_split_0_to_6m_new_pairwise_crossfit_benchmarked"
BRANCH_STEP = 3_000_000
COLORS = {"pure": "#4c78a8", "split": "#f58518"}
FAMILY_LABELS = {"pure": "pure recency", "split": "pure+linear recency"}
FAMILY_DIRS = {
    "pure": "cfa_purerecency_0_to_6m_pool_elo_200r",
    "split": "cfa_splitlinear_0_to_6m_pool_elo_200r",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a seed-wise pure/split Elo graph using within-branch pairwise "
            "records plus shared fixed-pool bridge records, with one shared "
            "0-3M prefix node per checkpoint step."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--cross-root", type=Path, default=DEFAULT_CROSS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--branch-step", type=int, default=BRANCH_STEP)
    parser.add_argument(
        "--include-step-zero",
        action="store_true",
        help="Keep step-0 records and plot rows. By default step 0 is omitted.",
    )
    parser.add_argument(
        "--skip-base-bridge",
        action="store_true",
        help="Do not include the June11 fixed-pool base report's own records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, source_rows = build_rows(
        args.input_root,
        args.cross_root,
        branch_step=int(args.branch_step),
        include_step_zero=bool(args.include_step_zero),
        include_base_bridge=not bool(args.skip_base_bridge),
    )
    if not rows:
        raise SystemExit("No crossfit benchmarked Elo rows were produced")

    summary_rows = summarize_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_path = args.output_dir / f"{PREFIX}_elo_rows.csv"
    summary_path = args.output_dir / f"{PREFIX}_elo_summary.csv"
    source_path = args.output_dir / f"{PREFIX}_source_counts.csv"
    mean_path = args.output_dir / f"{PREFIX}_elo_mean_sem.png"
    panel_path = args.output_dir / f"{PREFIX}_elo_seed_panels.png"
    overlay_path = args.output_dir / f"{PREFIX}_elo_overlay.png"

    write_rows(rows_path, rows)
    write_summary(summary_path, summary_rows)
    write_source_counts(source_path, source_rows)
    plot_mean_sem(rows, summary_rows, mean_path, branch_step=int(args.branch_step))
    plot_seed_panels(rows, panel_path, branch_step=int(args.branch_step))
    plot_overlay(rows, overlay_path, branch_step=int(args.branch_step))

    print(f"rows: {rows_path}")
    print(f"summary: {summary_path}")
    print(f"sources: {source_path}")
    print(f"mean_sem: {mean_path}")
    print(f"seed_panels: {panel_path}")
    print(f"overlay: {overlay_path}")


def build_rows(
    input_root: Path,
    cross_root: Path,
    *,
    branch_step: int,
    include_step_zero: bool,
    include_base_bridge: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    seeds = sorted(
        int(path.name.replace("seed_", ""))
        for path in input_root.glob("seed_*")
        if path.is_dir() and path.name.replace("seed_", "").isdigit()
    )
    base_records_by_report: dict[str, list[PairwiseRecord]] = {}

    for seed in seeds:
        records: list[PairwiseRecord] = []
        step_nodes: dict[tuple[str, int], str] = {}
        source_counts: dict[str, int] = {}

        for family in ("pure", "split"):
            report_path = input_root / f"seed_{seed}" / FAMILY_DIRS[family] / "elo_rating_report.json"
            if not report_path.exists():
                continue
            report_records, report_step_nodes = records_from_pairwise_report(
                report_path,
                family=family,
                branch_step=branch_step,
                include_step_zero=include_step_zero,
            )
            records.extend(report_records)
            step_nodes.update(report_step_nodes)
            source_counts[f"within_{family}"] = source_counts.get(f"within_{family}", 0) + len(report_records)

        cross_report_path = cross_root / f"seed_{seed}" / "split_vs_pure_cfa" / "fixed_pool_eval_report.json"
        if cross_report_path.exists():
            cross_records, cross_step_nodes, base_report = records_from_cross_report(
                cross_report_path,
                branch_step=branch_step,
                include_step_zero=include_step_zero,
            )
            records.extend(cross_records)
            step_nodes.update(cross_step_nodes)
            source_counts["same_fixed_pool_bridge"] = len(cross_records)
            if include_base_bridge and base_report:
                if base_report not in base_records_by_report:
                    base_records_by_report[base_report] = records_from_base_bridge_report(Path(base_report))
                records.extend(base_records_by_report[base_report])
                source_counts["base_fixed_pool_anchor"] = len(base_records_by_report[base_report])
        else:
            source_counts["same_fixed_pool_bridge"] = 0

        if not records:
            continue
        ratings = calculate_elo(records, initial_rating=1500.0, scale=400.0, prior_std=400.0)
        rows.extend(rows_from_ratings(seed, ratings, step_nodes, branch_step=branch_step))
        for source, count in sorted(source_counts.items()):
            source_rows.append({"seed": seed, "source": source, "record_count": count})

    return sorted(rows, key=lambda row: (int(row["seed"]), str(row["family"]), int(row["step"]))), source_rows


def records_from_pairwise_report(
    report_path: Path,
    *,
    family: str,
    branch_step: int,
    include_step_zero: bool,
) -> tuple[list[PairwiseRecord], dict[tuple[str, int], str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records: list[PairwiseRecord] = []
    step_nodes: dict[tuple[str, int], str] = {}
    for pair in report.get("pair_summaries", []):
        step_a = step_from_agent(str(pair["agent_a"]))
        step_b = step_from_agent(str(pair["agent_b"]))
        if not include_step_zero and (step_a == 0 or step_b == 0):
            continue
        agent_a = node_for(family, step_a, branch_step=branch_step)
        agent_b = node_for(family, step_b, branch_step=branch_step)
        if agent_a == agent_b:
            continue
        step_nodes[(family, step_a)] = agent_a
        step_nodes[(family, step_b)] = agent_b
        games = float(pair["episodes"])
        records.append(
            PairwiseRecord(
                agent_a=agent_a,
                agent_b=agent_b,
                agent_a_score=float(pair["agent_a_win_rate"]) * games,
                games=games,
            )
        )
    return records, step_nodes


def records_from_cross_report(
    report_path: Path,
    *,
    branch_step: int,
    include_step_zero: bool,
) -> tuple[list[PairwiseRecord], dict[tuple[str, int], str], str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records: list[PairwiseRecord] = []
    step_nodes: dict[tuple[str, int], str] = {}
    for pair in report.get("pair_results", []):
        family = family_from_run_label(str(pair["agent_run_label"]))
        if family is None:
            continue
        step = int(pair["agent_step"])
        if not include_step_zero and step == 0:
            continue
        agent = node_for(family, step, branch_step=branch_step)
        opponent = external_node(str(pair["opponent"]))
        if agent == opponent:
            continue
        step_nodes[(family, step)] = agent
        records.append(
            PairwiseRecord(
                agent_a=agent,
                agent_b=opponent,
                agent_a_score=float(pair["agent_wins"]),
                games=float(pair["episodes"]),
            )
        )
    return records, step_nodes, str(report.get("base_report", ""))


def records_from_base_bridge_report(report_path: Path) -> list[PairwiseRecord]:
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records: list[PairwiseRecord] = []
    for pair in report.get("pair_results", []):
        agent = external_node(str(pair["agent"]))
        opponent = external_node(str(pair["opponent"]))
        if agent == opponent:
            continue
        records.append(
            PairwiseRecord(
                agent_a=agent,
                agent_b=opponent,
                agent_a_score=float(pair["agent_wins"]),
                games=float(pair["episodes"]),
            )
        )
    return records


def rows_from_ratings(
    seed: int,
    ratings: dict[str, float],
    step_nodes: dict[tuple[str, int], str],
    *,
    branch_step: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in ("pure", "split"):
        family_steps = sorted(step for fam, step in step_nodes if fam == family)
        for step in family_steps:
            node = step_nodes[(family, step)]
            if node not in ratings:
                continue
            note = "shared_prefix_node" if step <= branch_step else "branch_specific_node"
            rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "family_label": FAMILY_LABELS[family],
                    "step": step,
                    "step_millions": step / 1_000_000.0,
                    "elo": ratings[node],
                    "fit_node": node,
                    "benchmark_note": note,
                }
            )
    return rows


def node_for(family: str, step: int, *, branch_step: int) -> str:
    if step <= branch_step:
        return f"shared_step{step:07d}"
    return f"{family}_step{step:07d}"


def external_node(label: str) -> str:
    return f"bridge_{label}"


def family_from_run_label(run_label: str) -> str | None:
    if run_label.startswith("pure_cfa") or run_label.startswith("cfa_purerecency"):
        return "pure"
    if run_label.startswith("split_cfa") or run_label.startswith("cfa_splitlinear"):
        return "split"
    return None


def step_from_agent(agent: str) -> int:
    if "step" in agent:
        return int(agent.rsplit("step", 1)[1])
    match = re.search(r"_(\d+)$", agent)
    if match:
        return int(match.group(1))
    raise ValueError(f"cannot parse step from agent label: {agent}")


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), int(row["step"])), []).append(float(row["elo"]))
    summary: list[dict[str, Any]] = []
    for (family, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(values)
        sd = stdev(values) if n > 1 else 0.0
        sem = sd / math.sqrt(n) if n > 1 else 0.0
        summary.append(
            {
                "family": family,
                "family_label": FAMILY_LABELS[family],
                "step": step,
                "step_millions": step / 1_000_000.0,
                "n": n,
                "mean_elo": mean(values),
                "sd_elo": sd,
                "sem_elo": sem,
            }
        )
    return summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_dicts(
        path,
        rows,
        ["seed", "family", "family_label", "step", "step_millions", "elo", "fit_node", "benchmark_note"],
    )


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    write_dicts(
        path,
        rows,
        ["family", "family_label", "step", "step_millions", "n", "mean_elo", "sd_elo", "sem_elo"],
    )


def write_source_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    write_dicts(path, rows, ["seed", "source", "record_count"])


def write_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_mean_sem(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    branch_step: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in rows}):
        for family in ("pure", "split"):
            seed_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in seed_rows],
                [float(row["elo"]) for row in seed_rows],
                linewidth=1.0,
                alpha=0.18,
                color=COLORS[family],
                zorder=1,
            )

    for family in ("pure", "split"):
        family_rows = sorted(
            [row for row in summary_rows if row["family"] == family],
            key=lambda row: int(row["step"]),
        )
        x = [float(row["step_millions"]) for row in family_rows]
        y = [float(row["mean_elo"]) for row in family_rows]
        sem = [float(row["sem_elo"]) for row in family_rows]
        ax.fill_between(
            x,
            [value - err for value, err in zip(y, sem)],
            [value + err for value, err in zip(y, sem)],
            color=COLORS[family],
            alpha=0.16,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.2,
            linewidth=2.4,
            color=COLORS[family],
            label=FAMILY_LABELS[family],
            zorder=3,
        )

    ax.axvline(branch_step / 1_000_000.0, color="#333333", linewidth=1.0, alpha=0.28)
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(0.15, 6.05)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_seed_panels(rows: list[dict[str, Any]], output_path: Path, *, branch_step: int) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.2), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.ravel())
    for ax, seed in zip(flat_axes, seeds):
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in family_rows],
                [float(row["elo"]) for row in family_rows],
                marker="o",
                markersize=3.8,
                linewidth=2.0,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.axvline(branch_step / 1_000_000.0, color="#333333", linewidth=0.9, alpha=0.24)
        ax.set_title(f"seed {seed}", fontsize=14)
        ax.set_xlim(0.15, 6.05)
        ax.grid(True, alpha=0.28)

    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Checkpoint step (M)")
    for ax in flat_axes[::3]:
        ax.set_ylabel("Cross-fit benchmarked Elo")
    flat_axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Cross-fit benchmarked Elo by seed")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overlay(rows: list[dict[str, Any]], output_path: Path, *, branch_step: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in rows}):
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [float(row["step_millions"]) for row in family_rows],
                [float(row["elo"]) for row in family_rows],
                marker="o",
                markersize=3.4,
                linewidth=1.4,
                alpha=0.58,
                color=COLORS[family],
                label=FAMILY_LABELS[family] if seed == 17 else None,
            )
    ax.axvline(branch_step / 1_000_000.0, color="#333333", linewidth=1.0, alpha=0.28)
    ax.set_title("Cross-fit benchmarked Elo")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Cross-fit benchmarked Elo")
    ax.set_xlim(0.15, 6.05)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
