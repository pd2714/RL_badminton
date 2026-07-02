from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/rl/ginsburg_20260622/eval"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "pure_split_0_to_6m_new_pairwise_plots"
DEFAULT_FAMILY_DIR = "cfa_splitlinear_0_to_6m_pool_elo_200r"
DEFAULT_PREFIX = "split_cfa_0_to_6m_pairwise_win_rate_matrix_5seed_no0"
FONT_SCALE = 0.9 / 1.3 * 1.1


def _fs(size: float) -> float:
    return size * FONT_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average split-CFA 0-6M pairwise win-rate matrices across seeds and "
            "plot the Fig. 2A-style heatmap without the step-0 checkpoint."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--family-dir", default=DEFAULT_FAMILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--include-step-zero", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_average_matrix_report(
        args.input_root,
        family_dir=str(args.family_dir),
        include_step_zero=bool(args.include_step_zero),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / f"{args.prefix}.json"
    mean_csv_path = args.output_dir / f"{args.prefix}.csv"
    sem_csv_path = args.output_dir / f"{args.prefix}_sem.csv"
    png_path = args.output_dir / f"{args.prefix}.png"

    write_json(json_path, report)
    write_matrix_csv(mean_csv_path, report, key="mean_win_rate_matrix")
    write_matrix_csv(sem_csv_path, report, key="sem_win_rate_matrix")
    plot_matrix(report, png_path)

    print(f"json: {json_path}")
    print(f"mean_csv: {mean_csv_path}")
    print(f"sem_csv: {sem_csv_path}")
    print(f"png: {png_path}")


def build_average_matrix_report(input_root: Path, *, family_dir: str, include_step_zero: bool) -> dict[str, Any]:
    seed_reports = sorted(input_root.glob(f"seed_*/{family_dir}/elo_rating_report.json"), key=seed_sort_key)
    if not seed_reports:
        raise FileNotFoundError(f"no elo_rating_report.json files found under {input_root}/seed_*/{family_dir}")

    seed_matrices: list[np.ndarray] = []
    seeds: list[int] = []
    reference_steps: list[int] | None = None
    reference_labels: list[str] | None = None
    source_rows: list[dict[str, Any]] = []

    for report_path in seed_reports:
        seed = seed_from_path(report_path)
        matrix_report = matrix_from_pairwise_report(report_path, include_step_zero=include_step_zero)
        if reference_steps is None:
            reference_steps = matrix_report["steps"]
            reference_labels = matrix_report["labels"]
        elif matrix_report["steps"] != reference_steps:
            raise ValueError(
                f"step grid mismatch for seed {seed}: {matrix_report['steps']} != {reference_steps}"
            )
        seed_matrices.append(np.asarray(matrix_report["matrix"], dtype=float))
        seeds.append(seed)
        source_rows.append(
            {
                "seed": seed,
                "source_path": str(report_path),
                "pair_summary_count": matrix_report["pair_summary_count"],
                "used_pair_summary_count": matrix_report["used_pair_summary_count"],
            }
        )

    stack = np.stack(seed_matrices, axis=0)
    mean_matrix = np.nanmean(stack, axis=0)
    sd_matrix = np.nanstd(stack, axis=0, ddof=1) if len(seed_matrices) > 1 else np.zeros_like(mean_matrix)
    sem_matrix = sd_matrix / np.sqrt(len(seed_matrices))

    return {
        "definition": (
            "P_ij = Pr(split-CFA checkpoint i beats split-CFA checkpoint j), "
            "averaged elementwise across seed-specific 0-6M pairwise reports."
        ),
        "input_root": str(input_root),
        "family_dir": family_dir,
        "include_step_zero": include_step_zero,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "row_labels": reference_labels,
        "col_labels": reference_labels,
        "row_steps": reference_steps,
        "col_steps": reference_steps,
        "mean_win_rate_matrix": mean_matrix.tolist(),
        "sem_win_rate_matrix": sem_matrix.tolist(),
        "source_reports": source_rows,
    }


def matrix_from_pairwise_report(report_path: Path, *, include_step_zero: bool) -> dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    agents = report.get("agents", [])
    labels_and_steps = []
    for agent in agents:
        label = str(agent["label"])
        step = int(agent.get("step", step_from_agent(label)))
        if not include_step_zero and step == 0:
            continue
        labels_and_steps.append((label, step))
    labels_and_steps = sorted(labels_and_steps, key=lambda item: item[1])
    labels = [label for label, _ in labels_and_steps]
    steps = [step for _, step in labels_and_steps]
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 0.5)

    used_count = 0
    for pair in report.get("pair_summaries", []):
        agent_a = str(pair["agent_a"])
        agent_b = str(pair["agent_b"])
        if agent_a not in index or agent_b not in index:
            continue
        i = index[agent_a]
        j = index[agent_b]
        win_rate_a = float(pair["agent_a_win_rate"])
        matrix[i, j] = win_rate_a
        matrix[j, i] = 1.0 - win_rate_a
        used_count += 1

    return {
        "labels": labels,
        "steps": steps,
        "matrix": matrix.tolist(),
        "pair_summary_count": len(report.get("pair_summaries", [])),
        "used_pair_summary_count": used_count,
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def write_matrix_csv(path: Path, report: dict[str, Any], *, key: str) -> None:
    col_labels = list(report["col_labels"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "step", *col_labels])
        for label, step, values in zip(report["row_labels"], report["row_steps"], report[key]):
            writer.writerow([label, step, *values])


def plot_matrix(report: dict[str, Any], output_path: Path) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    matrix = np.asarray(report["mean_win_rate_matrix"], dtype=float)
    row_steps_m = np.asarray(report["row_steps"], dtype=float) / 1_000_000.0
    col_steps_m = np.asarray(report["col_steps"], dtype=float) / 1_000_000.0
    dx = float(np.median(np.diff(col_steps_m)))
    dy = float(np.median(np.diff(row_steps_m)))
    extent = [
        float(col_steps_m[0] - dx / 2),
        float(col_steps_m[-1] + dx / 2),
        float(row_steps_m[-1] + dy / 2),
        float(row_steps_m[0] - dy / 2),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": _fs(33),
            "axes.titlesize": _fs(31),
            "xtick.labelsize": _fs(27),
            "ytick.labelsize": _fs(27),
            "axes.linewidth": 2.0,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.major.size": 9,
            "ytick.major.size": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(1050 / 180, 1050 / 180), dpi=180)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.76)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdBu", aspect="equal", extent=extent)
    ax.plot([0.2, 6.0], [0.2, 6.0], color="0.55", linewidth=2.0, alpha=0.8)
    ax.set_xlabel("Opponent (M)")
    ax.set_ylabel("Checkpoint (M)")
    ax.set_xticks(np.arange(1, 7, 1))
    ax.set_yticks(np.arange(1, 7, 1))
    ax.set_xlim(0.1, 6.1)
    ax.set_ylim(6.1, 0.1)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    cax = ax.inset_axes([0.21, 1.045, 0.58, 0.075])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0.0, 0.5, 1.0])
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Win rate", labelpad=5, fontsize=_fs(31))
    cbar.ax.tick_params(labelsize=_fs(27), width=2.0, length=8)
    cbar.outline.set_linewidth(2.0)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def configure_matplotlib() -> None:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/private/tmp")) / "rl_badminton_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)


def seed_sort_key(path: Path) -> int:
    return seed_from_path(path)


def seed_from_path(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"could not parse seed from {path}")


def step_from_agent(agent: str) -> int:
    return int(agent.rsplit("step", 1)[1])


if __name__ == "__main__":
    main()
