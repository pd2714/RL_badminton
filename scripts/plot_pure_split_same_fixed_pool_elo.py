from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rl_badminton_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_ROOT = Path(
    "outputs/rl/ginsburg_20260622/same_main_fig2b_fixed_pool_eval_200r"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "split_vs_pure_cfa_aggregate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pure-CFA vs split-CFA Elo on the same fixed opponent pool."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_rows, base_report = read_seed_rows(args.input_root)
    validate_actual_prefix(plot_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        args.output_dir / "pure_split_same_fixed_pool_seed_rows.csv",
        plot_rows,
    )
    (args.output_dir / "fixed_pool_source.json").write_text(
        json.dumps(
            {
                "input_root": str(args.input_root),
                "base_report": base_report,
                "seed_count": len({row["seed"] for row in plot_rows}),
                "note": (
                    "pure is plotted as old; split is plotted as new. "
                    "Rows are plotted only from actual same-fixed-pool Ginsburg evaluations; "
                    "no local/shared prefix is spliced in."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    overlay_path = args.output_dir / "elo_vs_checkpoint_two_lines_pure_split_same_fixed_pool_by_seed.png"
    panel_path = args.output_dir / "elo_vs_checkpoint_two_lines_pure_split_same_fixed_pool_seed_panels.png"
    plot_elo_overlay(plot_rows, overlay_path)
    plot_elo_seed_panels(plot_rows, panel_path)
    print(overlay_path)
    print(panel_path)


def read_seed_rows(input_root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    base_reports: set[str] = set()
    for metrics_path in sorted(input_root.glob("seed_*/split_vs_pure_cfa/mean_win_rate_elo.csv")):
        seed = metrics_path.parts[-3].replace("seed_", "")
        report_path = metrics_path.with_name("fixed_pool_eval_report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("remaining_pair_count", -1)) != 0:
            raise ValueError(f"incomplete fixed-pool report: {report_path}")
        base_reports.add(str(report.get("base_report", "")))

        with metrics_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                run_label = str(row["run_label"])
                if run_label.startswith("pure_cfa"):
                    family = "pure"
                elif run_label.startswith("split_cfa"):
                    family = "split"
                else:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "family": family,
                        "run_label": run_label,
                        "label": row["label"],
                        "step": int(row["step"]),
                        "step_millions": int(row["step"]) / 1_000_000.0,
                        "mean_pool_win_rate": float(row["mean_pool_win_rate"]),
                        "elo": float(row["elo"]),
                        "evaluated_pair_count": int(row["evaluated_pair_count"]),
                    }
                )

    if not rows:
        raise FileNotFoundError(f"no split_vs_pure_cfa metrics found under {input_root}")
    if len(base_reports) != 1:
        raise ValueError(f"expected one fixed-pool base report, found {sorted(base_reports)}")
    return rows, next(iter(base_reports))


def validate_actual_prefix(rows: list[dict[str, Any]]) -> None:
    missing: list[str] = []
    for seed in sorted({str(row["seed"]) for row in rows}, key=int):
        for family in ("pure", "split"):
            family_steps = {
                int(row["step"])
                for row in rows
                if str(row["seed"]) == seed and row["family"] == family
            }
            if 0 not in family_steps:
                missing.append(f"seed {seed} {family}")
    if missing:
        raise ValueError(
            "missing actual step-0 same-fixed-pool rows; wait for prefix evals: "
            + ", ".join(missing)
        )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_elo_overlay(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = {"pure": "pure (old)", "split": "split (new)"}
    colors = {"pure": "#4c78a8", "split": "#f58518"}

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for seed in sorted({str(row["seed"]) for row in rows}, key=int):
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in rows if str(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            x = [float(row["step_millions"]) for row in family_rows]
            y = [float(row["elo"]) for row in family_rows]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=3.6,
                alpha=0.62,
                color=colors[family],
                label=labels[family] if seed == "17" else None,
            )

    ax.set_title("Fixed-pool Elo")
    ax.set_xlabel("Checkpoint step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_elo_seed_panels(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = {"pure": "pure (old)", "split": "split (new)"}
    colors = {"pure": "#4c78a8", "split": "#f58518"}
    seeds = sorted({str(row["seed"]) for row in rows}, key=int)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = list(axes.ravel())
    for ax, seed in zip(flat_axes, seeds):
        for family in ("pure", "split"):
            family_rows = sorted(
                [row for row in rows if str(row["seed"]) == seed and row["family"] == family],
                key=lambda row: int(row["step"]),
            )
            x = [float(row["step_millions"]) for row in family_rows]
            y = [float(row["elo"]) for row in family_rows]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=3.8,
                color=colors[family],
                label=labels[family],
            )
        ax.set_title(f"seed {seed}")
        ax.set_xlim(left=0.0)
        ax.grid(True, alpha=0.28)
    for ax in flat_axes[len(seeds) :]:
        ax.axis("off")
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Checkpoint step (M)")
    for ax in flat_axes[::3]:
        ax.set_ylabel("Elo")
    flat_axes[0].legend(frameon=False)
    fig.suptitle("Fixed-pool Elo")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
