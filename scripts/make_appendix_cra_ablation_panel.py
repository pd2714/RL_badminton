from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ELO_CSV = REPO_ROOT / "outputs/rl/cross_run_fixed_pool_0p4m_to_3p2m_200r_20260611/elo_ratings.csv"
DEFAULT_PAIR_WIN_RATES_CSV = (
    REPO_ROOT / "outputs/rl/cross_run_fixed_pool_0p4m_to_3p2m_200r_20260611/pair_win_rates.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/appendix_cra_ablation_panel.png"
DEFAULT_SOURCE_DATA = (
    REPO_ROOT / "6a19f5382c36b7ba5e5cf0b1/figures/source_data/appendix_cra_ablation_direct_match_win_rates.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the appendix CRA/no-CRA ablation panel.")
    parser.add_argument("--elo-csv", type=Path, default=DEFAULT_ELO_CSV)
    parser.add_argument("--pair-win-rates-csv", type=Path, default=DEFAULT_PAIR_WIN_RATES_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-data-csv", type=Path, default=DEFAULT_SOURCE_DATA)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/private/tmp")) / "rl_badminton_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_by_key(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(sorted(grouped.items()))


def _run_label(run_label: str) -> str:
    return {
        "norecoverycfadv": "No CRA",
        "recoverycfdefault": "CRA",
    }.get(run_label, run_label)


def _rounded_ylim(values: list[float]) -> tuple[float, float]:
    low = math.floor(min(values) / 50.0) * 50.0
    high = math.ceil(max(values) / 50.0) * 50.0
    if high <= max(values):
        high += 50.0
    if low >= min(values):
        low -= 50.0
    return low, high


def _wilson_interval(wins: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0.0:
        raise ValueError("Wilson interval requires a positive sample size")
    p_hat = wins / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denom
    half_width = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def matched_direct_rows(pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    direct_rows: list[dict[str, Any]] = []
    for row in pair_rows:
        if row["agent_run_label"] != "recoverycfdefault":
            continue
        if row["opponent_run_label"] != "norecoverycfadv":
            continue
        if int(row["agent_step"]) != int(row["opponent_step"]):
            continue

        episodes = float(row["episodes"])
        wins = float(row["agent_wins"])
        win_rate = float(row["agent_win_rate"])
        ci_low, ci_high = _wilson_interval(wins, episodes)
        direct_rows.append(
            {
                "step": int(row["agent_step"]),
                "step_millions": float(row["agent_step"]) / 1_000_000.0,
                "episodes": int(episodes),
                "cra_wins": wins,
                "cra_win_rate": win_rate,
                "no_cra_win_rate": 1.0 - win_rate,
                "wilson95_low": ci_low,
                "wilson95_high": ci_high,
            }
        )
    direct_rows.sort(key=lambda row: int(row["step"]))
    return direct_rows


def write_direct_source_data(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "step_millions",
        "episodes",
        "cra_wins",
        "cra_win_rate",
        "no_cra_win_rate",
        "wilson95_low",
        "wilson95_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot_elo_panel(ax: Any, elo_rows: list[dict[str, str]]) -> list[float]:
    colors = {
        "norecoverycfadv": "#2f7fba",
        "recoverycfdefault": "#d0693a",
    }
    plotted: list[float] = []
    for run_label, run_rows in _rows_by_key(elo_rows, "run_label").items():
        run_rows = sorted(run_rows, key=lambda row: float(row["step_millions"]))
        x = [float(row["step_millions"]) for row in run_rows]
        y = [float(row["elo"]) for row in run_rows]
        plotted.extend(y)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.35,
            markersize=5.7,
            color=colors.get(run_label),
            label=_run_label(run_label),
        )

    ax.set_xlabel("Training step (M)")
    ax.set_ylabel("Elo")
    ax.set_xlim(0.25, 3.35)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.06, 0.995), handlelength=1.2, borderaxespad=0.0)
    return plotted


def plot_direct_bar_panel(ax: Any, direct_rows: list[dict[str, Any]]) -> None:
    x = [float(row["step_millions"]) for row in direct_rows]
    y = [float(row["cra_win_rate"]) for row in direct_rows]
    yerr_low = [value - float(row["wilson95_low"]) for value, row in zip(y, direct_rows)]
    yerr_high = [float(row["wilson95_high"]) - value for value, row in zip(y, direct_rows)]

    ax.bar(
        x,
        y,
        width=0.22,
        color="#d0693a",
        edgecolor="black",
        linewidth=0.9,
        yerr=[yerr_low, yerr_high],
        error_kw={"elinewidth": 1.1, "ecolor": "0.15", "capsize": 3.2, "capthick": 1.1},
        zorder=3,
    )
    ax.axhline(0.5, color="0.25", linestyle="--", linewidth=1.15, zorder=2)
    for xpos, value in zip(x, y):
        ax.text(xpos, value + 0.052, f"{value:.1%}", ha="center", va="bottom", fontsize=15)

    ax.set_xlabel("Checkpoint (M)")
    ax.set_ylabel("CRA win rate")
    ax.set_xlim(0.55, 3.45)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.1f}" for value in x])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def plot_figure(elo_csv: Path, pair_win_rates_csv: Path, output: Path, source_data_csv: Path) -> None:
    configure_matplotlib()

    import matplotlib.pyplot as plt

    elo_rows = load_csv_rows(elo_csv)
    direct_rows = matched_direct_rows(load_csv_rows(pair_win_rates_csv))
    if not direct_rows:
        raise SystemExit(f"No matched CRA-vs-noCRA direct rows found in {pair_win_rates_csv}")
    write_direct_source_data(source_data_csv, direct_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.subplots_adjust(left=0.095, right=0.992, bottom=0.195, top=0.955, wspace=0.36)

    elo_values = plot_elo_panel(axes[0], elo_rows)
    plot_direct_bar_panel(axes[1], direct_rows)

    y_low, y_high = _rounded_ylim(elo_values)
    axes[0].set_ylim(y_low, y_high)
    axes[0].set_yticks(range(int(y_low), int(y_high) + 1, 100))

    for ax in axes:
        ax.grid(True, color="0.72", linewidth=0.8, alpha=0.35, zorder=0)
        ax.set_box_aspect(0.78)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    label_box = dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="0.82", linewidth=0.75)
    axes[0].text(0.025, 0.965, "A", transform=axes[0].transAxes, fontsize=24, fontweight="bold", va="top", bbox=label_box)
    axes[1].text(0.025, 0.965, "B", transform=axes[1].transAxes, fontsize=24, fontweight="bold", va="top", bbox=label_box)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_figure(args.elo_csv, args.pair_win_rates_csv, args.output, args.source_data_csv)
    print(f"output: {args.output}")
    print(f"source_data_csv: {args.source_data_csv}")


if __name__ == "__main__":
    main()
