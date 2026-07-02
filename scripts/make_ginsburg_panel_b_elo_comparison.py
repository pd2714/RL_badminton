from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/eval"
DEFAULT_LOCAL_ELO_CSV = (
    REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/legacy_local_panel_b_20260611/elo_ratings.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/panel_b_pure_vs_nocfa_elo_6run.png"
DEFAULT_SUMMARY = REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/panel_b_pure_vs_nocfa_elo_6run_summary.csv"
DEFAULT_PER_SEED_OUTPUT_DIR = REPO_ROOT / "outputs/rl/ginsburg_panel_b_20260626/per_seed"
EXPECTED_SEEDS = [17, 23, 31, 47, 59]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Ginsburg panel-B pure-CFA vs no-CFA Elo across seeds.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--per-seed-output-dir", type=Path, default=DEFAULT_PER_SEED_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=EXPECTED_SEEDS)
    parser.add_argument("--local-elo-csv", type=Path, default=DEFAULT_LOCAL_ELO_CSV)
    parser.add_argument("--local-replicate", default="local_20260611")
    parser.add_argument("--no-local", action="store_true", help="Do not include the legacy local panel-B replicate.")
    parser.add_argument("--require-all", action="store_true", help="Fail if any requested seed has no Elo CSV.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/private/tmp")) / "rl_badminton_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)


def load_seed_rows(eval_root: Path, seeds: list[int]) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in seeds:
        csv_path = eval_root / f"seed_{seed}" / "panel_b_shared_pool_elo_200r" / "elo_ratings.csv"
        if not csv_path.exists():
            missing.append(seed)
            continue
        rows.extend(load_elo_rows(csv_path, f"seed_{seed}"))
    return rows, missing


def load_elo_rows(csv_path: Path, replicate: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            family = family_from_run_label(str(row["run_label"]))
            if family is None:
                continue
            rows.append(
                {
                    "replicate": replicate,
                    "family": family,
                    "step": int(row["step"]),
                    "step_millions": float(row["step_millions"]),
                    "elo": float(row["elo"]),
                }
            )
    return rows


def family_from_run_label(run_label: str) -> str | None:
    if run_label.startswith("purecfa_seed"):
        return "purecfa"
    if run_label.startswith("nocfa_seed"):
        return "nocfa"
    if run_label == "recoverycfdefault":
        return "purecfa"
    if run_label == "norecoverycfadv":
        return "nocfa"
    return None


def family_label(family: str) -> str:
    return {
        "purecfa": "Pure CFA",
        "nocfa": "No CFA",
    }.get(family, family)


def rows_by_replicate_family(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["replicate"]), str(row["family"]))].append(row)
    return {key: sorted(value, key=lambda row: int(row["step"])) for key, value in grouped.items()}


def replicate_sort_key(replicate: str) -> tuple[int, int | str]:
    if replicate.startswith("seed_"):
        try:
            return (0, int(replicate.removeprefix("seed_")))
        except ValueError:
            pass
    return (1, replicate)


def replicate_title(replicate: str) -> str:
    if replicate.startswith("seed_"):
        return f"seed {replicate.removeprefix('seed_')}"
    return replicate.replace("_", " ")


def replicate_file_suffix(replicate: str) -> str:
    if replicate.startswith("seed_"):
        return f"seed{replicate.removeprefix('seed_')}"
    return replicate


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    step_millions: dict[int, float] = {}
    for row in rows:
        step = int(row["step"])
        grouped[(str(row["family"]), step)].append(float(row["elo"]))
        step_millions[step] = float(row["step_millions"])

    summary: list[dict[str, Any]] = []
    for (family, step), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(values)
        sd = stdev(values) if n > 1 else 0.0
        sem = sd / math.sqrt(n) if n > 1 else 0.0
        summary.append(
            {
                "family": family,
                "label": family_label(family),
                "step": step,
                "step_millions": step_millions[step],
                "n": n,
                "mean_elo": mean(values),
                "sd_elo": sd,
                "sem_elo": sem,
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["family", "label", "step", "step_millions", "n", "mean_elo", "sd_elo", "sem_elo"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def rounded_ylim(values: list[float]) -> tuple[float, float]:
    low = math.floor(min(values) / 50.0) * 50.0
    high = math.ceil(max(values) / 50.0) * 50.0
    if high <= max(values):
        high += 50.0
    if low >= min(values):
        low -= 50.0
    return low, high


def plot(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output: Path) -> None:
    configure_matplotlib()

    import matplotlib.pyplot as plt

    colors = {
        "purecfa": "#d0693a",
        "nocfa": "#2f7fba",
    }
    order = ["purecfa", "nocfa"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 24,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 18,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.18, top=0.97)

    for (_replicate, family), seed_rows in rows_by_replicate_family(rows).items():
        ax.plot(
            [float(row["step_millions"]) for row in seed_rows],
            [float(row["elo"]) for row in seed_rows],
            color=colors.get(family, "0.35"),
            linewidth=1.15,
            alpha=0.22,
            zorder=1,
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_family[str(row["family"])].append(row)

    for family in [family for family in order if family in by_family]:
        family_rows = sorted(by_family[family], key=lambda row: int(row["step"]))
        x = [float(row["step_millions"]) for row in family_rows]
        y = [float(row["mean_elo"]) for row in family_rows]
        sem = [float(row["sem_elo"]) for row in family_rows]
        ax.fill_between(
            x,
            [value - err for value, err in zip(y, sem)],
            [value + err for value, err in zip(y, sem)],
            color=colors[family],
            alpha=0.16,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.4,
            markersize=5.8,
            color=colors[family],
            label=family_label(family),
            zorder=3,
        )

    y_values = [float(row["elo"]) for row in rows]
    y_values.extend(float(row["mean_elo"]) + float(row["sem_elo"]) for row in summary_rows)
    y_values.extend(float(row["mean_elo"]) - float(row["sem_elo"]) for row in summary_rows)
    y_low, y_high = rounded_ylim(y_values)

    ax.set_xlabel("Training step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(0.25, 3.35)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_ylim(y_low, y_high)
    ax.set_yticks(range(int(y_low), int(y_high) + 1, 100))
    ax.grid(True, color="0.72", linewidth=0.8, alpha=0.35)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.05, 0.995), handlelength=1.25, borderaxespad=0.0)
    ax.set_box_aspect(0.72)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_per_seed(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    configure_matplotlib()

    import matplotlib.pyplot as plt

    colors = {
        "purecfa": "#d0693a",
        "nocfa": "#2f7fba",
    }
    order = ["purecfa", "nocfa"]
    grouped = rows_by_replicate_family(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 24,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 18,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
        }
    )

    for replicate in sorted({str(row["replicate"]) for row in rows}, key=replicate_sort_key):
        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.18, top=0.92)
        seed_values: list[float] = []

        for family in order:
            seed_rows = grouped.get((replicate, family), [])
            if not seed_rows:
                continue
            x = [float(row["step_millions"]) for row in seed_rows]
            y = [float(row["elo"]) for row in seed_rows]
            seed_values.extend(y)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.4,
                markersize=5.8,
                color=colors[family],
                label=family_label(family),
                zorder=3,
            )

        y_low, y_high = rounded_ylim(seed_values)
        ax.set_title(replicate_title(replicate), fontsize=24, pad=8)
        ax.set_xlabel("Training step (M)")
        ax.set_ylabel("Elo")
        ax.set_xlim(0.25, 3.35)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_ylim(y_low, y_high)
        ax.set_yticks(range(int(y_low), int(y_high) + 1, 100))
        ax.grid(True, color="0.72", linewidth=0.8, alpha=0.35)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.05, 0.995), handlelength=1.25, borderaxespad=0.0)
        ax.set_box_aspect(0.72)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

        output_path = output_dir / f"panel_b_pure_vs_nocfa_elo_{replicate_file_suffix(replicate)}.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    args = parse_args()
    rows, missing = load_seed_rows(args.eval_root, [int(seed) for seed in args.seeds])
    missing_local = False
    if not args.no_local:
        if args.local_elo_csv.exists():
            rows.extend(load_elo_rows(args.local_elo_csv, str(args.local_replicate)))
        else:
            missing_local = True
    if missing and args.require_all:
        raise SystemExit(f"Missing Elo CSV for seed(s): {', '.join(str(seed) for seed in missing)}")
    if missing_local and args.require_all:
        raise SystemExit(f"Missing local Elo CSV: {args.local_elo_csv}")
    if not rows:
        raise SystemExit(f"No Elo rows found under {args.eval_root}")

    summary_rows = summarize_rows(rows)
    write_summary(args.summary_csv, summary_rows)
    plot(rows, summary_rows, args.output)
    per_seed_outputs = plot_per_seed(rows, args.per_seed_output_dir)

    complete_replicates = sorted({str(row["replicate"]) for row in rows}, key=replicate_sort_key)
    print(f"output: {args.output}")
    print(f"summary_csv: {args.summary_csv}")
    for output_path in per_seed_outputs:
        print(f"per_seed_output: {output_path}")
    print(f"complete_replicate_count: {len(complete_replicates)}")
    print(f"complete_replicates: {','.join(complete_replicates)}")
    if missing:
        print(f"missing_elo_seeds: {','.join(str(seed) for seed in missing)}")
    if missing_local:
        print(f"missing_local_elo_csv: {args.local_elo_csv}")


if __name__ == "__main__":
    main()
