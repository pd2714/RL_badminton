from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from badminton.config import ActionConfig, SimulationConfig
from badminton.trajectory import TrajectoryResult, simulate_trajectory


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "trajectory_validation"
BASE_KH = 0.20
BASE_KV = 0.16
DRAG_DT = 0.002
COURT_HALF_LENGTH = 6.70
NET_Y = 0.0
NET_HEIGHT_M = 1.55


@dataclass(frozen=True)
class ShotSpec:
    name: str
    start: tuple[float, float, float]
    velocity: tuple[float, float, float]
    color: str
    title: str
    reference_note: str


@dataclass(frozen=True)
class MetricBand:
    shot: str
    metric: str
    label: str
    low: float
    high: float
    unit: str
    source: str
    note: str


@dataclass(frozen=True)
class MetricRow:
    shot: str
    metric: str
    label: str
    simulated: float
    low: float
    high: float
    unit: str
    source: str
    note: str

    @property
    def passes(self) -> bool:
        return self.low <= self.simulated <= self.high


SHOT_SPECS: tuple[ShotSpec, ...] = (
    ShotSpec(
        name="clear_lift",
        start=(0.0, -6.70, 1.20),
        velocity=(0.0, 40.0, 22.0),
        color="#2563eb",
        title="Clear / lift",
        reference_note="BWF speed-test landing window",
    ),
    ShotSpec(
        name="drop",
        start=(0.0, -5.70, 2.35),
        velocity=(0.0, 18.0, 2.0),
        color="#16a34a",
        title="Drop",
        reference_note="short front-court landing and near-terminal speed",
    ),
    ShotSpec(
        name="smash",
        start=(0.0, -6.00, 2.80),
        velocity=(0.0, 83.0, -12.0),
        color="#dc2626",
        title="Smash",
        reference_note="Collet 2026 speed decay and time-of-flight",
    ),
    ShotSpec(
        name="drive",
        start=(0.0, -3.20, 1.80),
        velocity=(0.0, 24.0, 1.0),
        color="#7c3aed",
        title="Drive",
        reference_note="low, flat net-crossing corridor",
    ),
)


REFERENCE_BANDS: tuple[MetricBand, ...] = (
    MetricBand(
        shot="clear_lift",
        metric="landing_shortfall_m",
        label="landing short of far back line",
        low=0.53,
        high=0.99,
        unit="m",
        source="BWF Laws 3.1-3.2",
        note="Correct-speed shuttle lands 0.53-0.99 m short after full underhand stroke.",
    ),
    MetricBand(
        shot="clear_lift",
        metric="max_height_m",
        label="maximum height",
        low=4.50,
        high=8.50,
        unit="m",
        source="canonical high-clear check",
        note="High clears/lifts should be high enough to reset court position without ceiling-scale arcs.",
    ),
    MetricBand(
        shot="drop",
        metric="landing_y_m",
        label="landing past net",
        low=0.50,
        high=2.00,
        unit="m",
        source="drop-shot tactical definition",
        note="A drop should land just over and close to the net.",
    ),
    MetricBand(
        shot="drop",
        metric="impact_speed_mps",
        label="impact speed",
        low=5.50,
        high=8.00,
        unit="m/s",
        source="Collet 2026 terminal-velocity range",
        note="Slow terminal-region speed, consistent with measured feather shuttle terminal velocities.",
    ),
    MetricBand(
        shot="smash",
        metric="time_to_10m_s",
        label="time to travel 10 m",
        low=0.35,
        high=0.55,
        unit="s",
        source="Collet 2026 Figure 11",
        note="Published 300 km/h smash reaches 10 m in about 0.44 s.",
    ),
    MetricBand(
        shot="smash",
        metric="speed_ratio_3p35",
        label="speed ratio after 3.35 m",
        low=0.45,
        high=0.58,
        unit="ratio",
        source="Collet 2026",
        note="Feather-shuttle speed is approximately halved every 3.35 m.",
    ),
    MetricBand(
        shot="smash",
        metric="landing_y_m",
        label="landing past net",
        low=3.00,
        high=5.50,
        unit="m",
        source="court-geometry smash check",
        note="A steep rear-court smash should land in the opponent mid/back court.",
    ),
    MetricBand(
        shot="drive",
        metric="max_height_m",
        label="maximum height",
        low=1.55,
        high=2.40,
        unit="m",
        source="drive-shot tactical definition",
        note="A drive is a fast, flat shot around eye/net height rather than a high arc.",
    ),
    MetricBand(
        shot="drive",
        metric="net_crossing_z_m",
        label="net crossing height",
        low=1.60,
        high=2.20,
        unit="m",
        source="BWF net height plus flat-drive check",
        note="The shot should clear the 1.55 m net but remain low.",
    ),
)


REFERENCE_URLS = (
    ("BWF Laws of Badminton", "https://www.worldbadminton.com/rules/"),
    ("Collet 2026, Shuttlecock velocity decay", "https://arxiv.org/abs/2601.01412"),
    ("Drop shot definition", "https://en.wikipedia.org/wiki/Drop_shot"),
    ("Drive shot definition", "https://de.wikipedia.org/wiki/Drive_%28Schlagtechnik%29"),
)
LATEX_ROW_END = r"\\"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ShuttleArena shuttle trajectories against published reference ranges.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated validation artifacts.")
    parser.add_argument("--basename", default="shuttle_trajectory_validation", help="Filename stem for figure/table outputs.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def make_config(*, kh: float = BASE_KH, kv: float = BASE_KV) -> SimulationConfig:
    return SimulationConfig(
        action=ActionConfig(
            trajectory_mode="drag_square",
            horizontal_drag_coefficient=kh,
            vertical_drag_coefficient=kv,
            drag_dt=DRAG_DT,
        )
    )


def simulate_spec(spec: ShotSpec, config: SimulationConfig) -> TrajectoryResult:
    return simulate_trajectory(*spec.start, *spec.velocity, config, sample_count=500)


def trajectory_array(result: TrajectoryResult) -> np.ndarray:
    return np.asarray(
        [(point.t, point.x, point.y, point.z, point.v_x, point.v_y, point.v_z) for point in result.samples],
        dtype=float,
    )


def speed_array(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, 4:7], axis=1)


def value_at_y(points: np.ndarray, values: np.ndarray, target_y: float) -> float:
    ys = points[:, 2]
    for idx in range(1, len(points)):
        if (ys[idx - 1] - target_y) * (ys[idx] - target_y) <= 0.0:
            if np.isclose(ys[idx - 1], ys[idx]):
                return float(values[idx])
            ratio = (target_y - ys[idx - 1]) / (ys[idx] - ys[idx - 1])
            return float(values[idx - 1] + ratio * (values[idx] - values[idx - 1]))
    return float("nan")


def summarize_shot(spec: ShotSpec, result: TrajectoryResult) -> dict[str, float]:
    points = trajectory_array(result)
    speeds = speed_array(points)
    initial_speed = float(np.linalg.norm(np.asarray(spec.velocity, dtype=float)))
    y0 = float(spec.start[1])
    at_3p35_y = y0 + 3.35
    at_10m_y = y0 + 10.0
    net_z = result.net_crossing.z if result.net_crossing is not None else float("nan")
    return {
        "flight_time_s": float(result.landing_time),
        "landing_y_m": float(result.landing_y),
        "landing_shortfall_m": float(COURT_HALF_LENGTH - result.landing_y),
        "travel_y_m": float(result.landing_y - y0),
        "max_height_m": float(np.max(points[:, 3])),
        "net_crossing_z_m": float(net_z),
        "initial_speed_mps": initial_speed,
        "impact_speed_mps": float(speeds[-1]),
        "speed_ratio_3p35": value_at_y(points, speeds, at_3p35_y) / initial_speed,
        "time_to_10m_s": value_at_y(points, points[:, 0], at_10m_y),
    }


def baseline_metric_rows(config: SimulationConfig | None = None) -> list[MetricRow]:
    active = make_config() if config is None else config
    specs = {spec.name: spec for spec in SHOT_SPECS}
    bands = {(band.shot, band.metric): band for band in REFERENCE_BANDS}
    rows: list[MetricRow] = []
    for shot_name in dict.fromkeys(band.shot for band in REFERENCE_BANDS):
        spec = specs[shot_name]
        metrics = summarize_shot(spec, simulate_spec(spec, active))
        for band in (candidate for candidate in REFERENCE_BANDS if candidate.shot == shot_name):
            rows.append(
                MetricRow(
                    shot=band.shot,
                    metric=band.metric,
                    label=band.label,
                    simulated=float(metrics[band.metric]),
                    low=band.low,
                    high=band.high,
                    unit=band.unit,
                    source=band.source,
                    note=band.note,
                )
            )
    return rows


def robustness_rows(scales: Iterable[float] = (0.8, 0.9, 1.0, 1.1, 1.2)) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    specs = {spec.name: spec for spec in SHOT_SPECS}
    for scale in scales:
        kh = BASE_KH * scale
        kv = BASE_KV * scale
        config = make_config(kh=kh, kv=kv)
        clear = summarize_shot(specs["clear_lift"], simulate_spec(specs["clear_lift"], config))
        drop = summarize_shot(specs["drop"], simulate_spec(specs["drop"], config))
        smash = summarize_shot(specs["smash"], simulate_spec(specs["smash"], config))
        drive = summarize_shot(specs["drive"], simulate_spec(specs["drive"], config))
        rows.append(
            {
                "drag_scale": float(scale),
                "kh": kh,
                "kv": kv,
                "speed_halving_distance_m": math.log(2.0) / kh,
                "vertical_terminal_speed_mps": math.sqrt(9.81 / kv),
                "clear_shortfall_m": clear["landing_shortfall_m"],
                "drop_landing_y_m": drop["landing_y_m"],
                "smash_time_to_10m_s": smash["time_to_10m_s"],
                "drive_max_height_m": drive["max_height_m"],
            }
        )
    return rows


def format_value(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if abs(value) >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_metric_csv(rows: Iterable[MetricRow], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot", "metric", "label", "simulated", "reference_low", "reference_high", "unit", "pass", "source", "note"])
        for row in rows:
            writer.writerow([row.shot, row.metric, row.label, row.simulated, row.low, row.high, row.unit, int(row.passes), row.source, row.note])


def write_robustness_csv(rows: Iterable[dict[str, float]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_latex(rows: Iterable[MetricRow], path: Path) -> None:
    with path.open("w") as handle:
        handle.write("\\begin{tabular}{lllll}\n")
        handle.write("\\toprule\n")
        handle.write(f"Shot & Metric & ShuttleArena & Reference & Source {LATEX_ROW_END}\n")
        handle.write("\\midrule\n")
        for row in rows:
            status = "yes" if row.passes else "no"
            reference = f"{format_value(row.low)}--{format_value(row.high)} {row.unit}"
            simulated = f"{format_value(row.simulated)} {row.unit}"
            handle.write(
                f"{escape_latex(row.shot)} & {escape_latex(row.label)} & "
                f"{escape_latex(simulated)} & {escape_latex(reference)} & "
                f"{escape_latex(row.source)} ({status}) {LATEX_ROW_END}\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def write_robustness_latex(rows: Iterable[dict[str, float]], path: Path) -> None:
    with path.open("w") as handle:
        handle.write("\\begin{tabular}{rrrrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write(f"$k$ scale & $k_h$ & $k_v$ & half-dist. & $V_T$ & clear short & drop land & smash $t_{{10}}$ {LATEX_ROW_END}\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(
                f"{row['drag_scale']:.1f} & {row['kh']:.3f} & {row['kv']:.3f} & "
                f"{row['speed_halving_distance_m']:.2f} & {row['vertical_terminal_speed_mps']:.2f} & "
                f"{row['clear_shortfall_m']:.2f} & {row['drop_landing_y_m']:.2f} & "
                f"{row['smash_time_to_10m_s']:.3f} {LATEX_ROW_END}\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate_panel(ax: plt.Axes, spec: ShotSpec, metrics: dict[str, float]) -> None:
    lines = [
        f"flight {metrics['flight_time_s']:.2f} s",
        f"max z {metrics['max_height_m']:.2f} m",
        f"land y {metrics['landing_y_m']:.2f} m",
    ]
    if spec.name == "smash":
        lines = [
            f"t(10 m) {metrics['time_to_10m_s']:.2f} s",
            f"V(3.35 m)/V0 {metrics['speed_ratio_3p35']:.2f}",
            f"impact {metrics['impact_speed_mps']:.1f} m/s",
        ]
    elif spec.name == "clear_lift":
        lines.append(f"short {metrics['landing_shortfall_m']:.2f} m")
    elif spec.name == "drop":
        lines.append(f"impact {metrics['impact_speed_mps']:.1f} m/s")
    elif spec.name == "drive":
        lines.append(f"net z {metrics['net_crossing_z_m']:.2f} m")
    ax.text(
        0.03,
        0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.7,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "square,pad=0.28", "alpha": 0.95},
    )


def add_reference_overlay(ax: plt.Axes, spec: ShotSpec) -> None:
    if spec.name == "clear_lift":
        ax.axvspan(COURT_HALF_LENGTH - 0.99, COURT_HALF_LENGTH - 0.53, color="#93c5fd", alpha=0.24, label="BWF landing window")
    elif spec.name == "drop":
        ax.axvspan(0.50, 2.00, color="#86efac", alpha=0.24, label="close-to-net landing")
    elif spec.name == "smash":
        ax.axvspan(4.00, 5.50, color="#fecaca", alpha=0.24, label="steep in-court landing")
    elif spec.name == "drive":
        ax.axhspan(1.60, 2.20, color="#ddd6fe", alpha=0.26, label="flat net corridor")


def plot_validation_figure(config: SimulationConfig, output_png: Path, output_pdf: Path, *, dpi: int) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, spec in zip(axes_flat, SHOT_SPECS):
        result = simulate_spec(spec, config)
        points = trajectory_array(result)
        metrics = summarize_shot(spec, result)
        add_reference_overlay(ax, spec)
        ax.plot(points[:, 2], points[:, 3], color=spec.color, linewidth=2.2, label="ShuttleArena")
        ax.scatter([spec.start[1]], [spec.start[2]], s=28, color=spec.color, edgecolor="white", zorder=5)
        ax.scatter([result.landing_y], [0.0], s=34, color=spec.color, marker="v", edgecolor="white", zorder=5)
        ax.axvline(NET_Y, color="#111827", linewidth=1.0, alpha=0.75)
        ax.axhline(NET_HEIGHT_M, color="#111827", linewidth=0.9, linestyle="--", alpha=0.70)
        ax.set_xlim(-6.9, 6.9)
        ax.set_ylim(0.0, max(6.6, float(np.max(points[:, 3])) + 0.35))
        ax.set_title(f"{spec.title}\n{spec.reference_note}", loc="left", fontweight="bold", fontsize=9.4)
        ax.set_xlabel("court y position (m)")
        ax.set_ylabel("shuttle height z (m)")
        ax.grid(True, color="#e5e7eb", linewidth=0.7)
        ax.legend(loc="upper right", frameon=False)
        annotate_panel(ax, spec, metrics)
    fig.suptitle("Shuttle trajectory validation against published/reference badminton ranges", fontsize=12, fontweight="bold")
    fig.savefig(output_png, dpi=dpi)
    fig.savefig(output_pdf)
    plt.close(fig)


def write_reference_notes(path: Path) -> None:
    with path.open("w") as handle:
        handle.write("# Shuttle trajectory validation references\n\n")
        for name, url in REFERENCE_URLS:
            handle.write(f"- {name}: {url}\n")
        handle.write("\n")
        handle.write("The clear/lift hard landing window comes from BWF Laws 3.1-3.2. ")
        handle.write("The smash speed-decay and time-of-flight checks use Collet 2026, which reports a velocity-halving distance near 3.35 m and about 0.44 s to travel 10 m for a 300 km/h smash. ")
        handle.write("Drop and drive checks are intentionally lower precision: they encode canonical shot geometry and terminal-speed plausibility rather than full biomechanical calibration.\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = make_config()
    metric_rows = baseline_metric_rows(config)
    sensitivity_rows = robustness_rows()

    write_metric_csv(metric_rows, args.output_dir / f"{args.basename}_metrics.csv")
    write_metric_latex(metric_rows, args.output_dir / f"{args.basename}_metrics.tex")
    write_robustness_csv(sensitivity_rows, args.output_dir / f"{args.basename}_robustness.csv")
    write_robustness_latex(sensitivity_rows, args.output_dir / f"{args.basename}_robustness.tex")
    write_reference_notes(args.output_dir / f"{args.basename}_references.md")
    plot_validation_figure(
        config,
        args.output_dir / f"{args.basename}.png",
        args.output_dir / f"{args.basename}.pdf",
        dpi=args.dpi,
    )

    passed = sum(row.passes for row in metric_rows)
    print(f"wrote trajectory validation artifacts to {args.output_dir}")
    print(f"baseline reference checks passed: {passed}/{len(metric_rows)}")


if __name__ == "__main__":
    main()
