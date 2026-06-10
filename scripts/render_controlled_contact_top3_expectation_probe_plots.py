from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.eval_evolution import (
    LANDING_ZONE_NAMES,
    SHOT_TYPE_ORDER,
    build_sim_config,
    landing_zone_name,
    load_run_config,
)
from badminton1d.mpl_config import ensure_writable_matplotlib_config
from badminton1d.state import Side
from badminton1d.utils import ensure_directory, opponent_side


DEFAULT_POSITION_DIR = "opponent_default_position"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render lightweight controlled-contact opponent-position probe plots from "
            "the existing top-3 shot manifest."
        )
    )
    parser.add_argument("probe_dir", type=Path, help="controlled_contact_grid_probe directory.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to PROBE_DIR/top_shot_3d_views/top_shot_trajectories_3d_manifest.json.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Defaults to manifest run_dir.")
    parser.add_argument(
        "--contact-state",
        action="append",
        default=None,
        help="Optional base contact probe_id filter. Can be passed more than once.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_writable_matplotlib_config()

    probe_dir = args.probe_dir
    manifest_path = args.manifest or (probe_dir / "top_shot_3d_views" / "top_shot_trajectories_3d_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir or Path(str(manifest["run_dir"]))
    run_config = load_run_config(run_dir)
    config = build_sim_config(run_config)
    train_side: Side = str(run_config.get("train_side", "left"))  # type: ignore[assignment]
    receiver_side = opponent_side(train_side)
    checkpoint_step = int(manifest.get("checkpoint_step", -1))

    selected = None if args.contact_state is None else set(args.contact_state)
    rows: list[dict[str, Any]] = []
    for probe_id, top_shots in sorted(manifest["top_shots"].items()):
        if "__" not in probe_id:
            continue
        contact_id, opponent_cell_id = probe_id.split("__", 1)
        if selected is not None and contact_id not in selected:
            continue
        if not top_shots:
            continue

        summary = _summarize_top_shots(
            probe_id=probe_id,
            contact_id=contact_id,
            opponent_cell_id=opponent_cell_id,
            top_shots=top_shots,
            receiver_side=receiver_side,
            config=config,
            checkpoint_step=checkpoint_step,
        )
        rows.append(summary)

        scenario_dir = probe_dir / contact_id / opponent_cell_id
        ensure_directory(scenario_dir)
        trajectory_path = scenario_dir / f"{probe_id}_top3_shot_trajectories_3d.png"
        _write_probe_plot(
            scenario_dir / f"{probe_id}_probe.png",
            summary,
            top_shots,
            trajectory_path=trajectory_path,
            config=config,
            dpi=int(args.dpi),
        )
        _write_distribution_plot(
            scenario_dir / f"{probe_id}_shot_type_frequency.png",
            title=f"{_title(probe_id)}: top-3 expected shot type",
            values=summary["shot_type_weights"],
            ordered_names=SHOT_TYPE_ORDER,
            xlabel="top-3 normalized probability",
            dpi=int(args.dpi),
        )
        _write_distribution_plot(
            scenario_dir / f"{probe_id}_landing_zone_distribution.png",
            title=f"{_title(probe_id)}: top-3 expected landing zone",
            values=summary["landing_zone_weights"],
            ordered_names=LANDING_ZONE_NAMES,
            xlabel="top-3 normalized probability",
            dpi=int(args.dpi),
        )
        print(f"{probe_id}: {scenario_dir / f'{probe_id}_probe.png'}", flush=True)

    _move_default_position_pngs(probe_dir)
    _write_summary_files(probe_dir, rows)


def _summarize_top_shots(
    *,
    probe_id: str,
    contact_id: str,
    opponent_cell_id: str,
    top_shots: list[dict[str, Any]],
    receiver_side: Side,
    config: Any,
    checkpoint_step: int,
) -> dict[str, Any]:
    probabilities = np.asarray([float(row.get("probability", 0.0)) for row in top_shots], dtype=float)
    top3_mass = float(np.sum(probabilities))
    if top3_mass > 0.0:
        weights = probabilities / top3_mass
    else:
        weights = np.full(len(top_shots), 1.0 / max(len(top_shots), 1), dtype=float)

    speeds = np.asarray(
        [
            np.linalg.norm([float(row["v_x"]), float(row["v_y"]), float(row["v_z"])])
            for row in top_shots
        ],
        dtype=float,
    )
    landing_x = np.asarray([float(row["landing_x"]) for row in top_shots], dtype=float)
    landing_y = np.asarray([float(row["landing_y"]) for row in top_shots], dtype=float)
    vx = np.asarray([float(row["v_x"]) for row in top_shots], dtype=float)
    vy = np.asarray([float(row["v_y"]) for row in top_shots], dtype=float)
    vz = np.asarray([float(row["v_z"]) for row in top_shots], dtype=float)

    shot_type_weights: dict[str, float] = defaultdict(float)
    landing_zone_weights: dict[str, float] = defaultdict(float)
    shot_rows: list[dict[str, Any]] = []
    for weight, row in zip(weights, top_shots):
        shot_type = str(row["shot_type"])
        zone = landing_zone_name(receiver_side, (float(row["landing_x"]), float(row["landing_y"])), config)
        shot_type_weights[shot_type] += float(weight)
        landing_zone_weights[zone] += float(weight)
        shot_rows.append(
            {
                "rank": int(row["rank"]),
                "probability": float(row["probability"]),
                "top3_weight": float(weight),
                "shot_type": shot_type,
                "landing_zone": zone,
                "shot_speed": float(np.linalg.norm([float(row["v_x"]), float(row["v_y"]), float(row["v_z"])])),
                "landing_x": float(row["landing_x"]),
                "landing_y": float(row["landing_y"]),
            }
        )

    return {
        "probe_id": probe_id,
        "contact_id": contact_id,
        "opponent_cell_id": opponent_cell_id,
        "checkpoint_step": checkpoint_step,
        "top3_mass": top3_mass,
        "expected_shot_speed": float(np.dot(weights, speeds)),
        "expected_landing_x": float(np.dot(weights, landing_x)),
        "expected_landing_y": float(np.dot(weights, landing_y)),
        "expected_v_x": float(np.dot(weights, vx)),
        "expected_v_y": float(np.dot(weights, vy)),
        "expected_v_z": float(np.dot(weights, vz)),
        "shot_type_weights": dict(shot_type_weights),
        "landing_zone_weights": dict(landing_zone_weights),
        "shots": shot_rows,
    }


def _write_probe_plot(
    path: Path,
    summary: dict[str, Any],
    top_shots: list[dict[str, Any]],
    *,
    trajectory_path: Path,
    config: Any,
    dpi: int,
) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)
    fig.suptitle(f"{_title(summary['probe_id'])}: top-3 expectation probe")

    ax = axes[0, 0]
    if trajectory_path.exists():
        image = mpimg.imread(trajectory_path)
        ax.imshow(image)
        ax.set_title("Existing latest top-3 trajectories")
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "top-3 trajectory image not found", ha="center", va="center")
        ax.axis("off")

    ax = axes[0, 1]
    ax.axis("off")
    metrics = [
        ("checkpoint", f"{int(summary['checkpoint_step']):,}" if int(summary["checkpoint_step"]) >= 0 else "n/a"),
        ("top-3 mass", f"{float(summary['top3_mass']):.3f}"),
        ("E speed", f"{float(summary['expected_shot_speed']):.2f} m/s"),
        ("E landing x", f"{float(summary['expected_landing_x']):.2f} m"),
        ("E landing y", f"{float(summary['expected_landing_y']):.2f} m"),
        ("E velocity", f"({summary['expected_v_x']:.2f}, {summary['expected_v_y']:.2f}, {summary['expected_v_z']:.2f})"),
    ]
    table = ax.table(
        cellText=[[name, value] for name, value in metrics],
        colLabels=["metric", "expected value"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.03, 0.47, 0.94, 0.46],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    top_rows = [
        [
            f"#{int(row['rank'])}",
            f"{float(row['probability']):.3f}",
            str(row["shot_type"]),
            str(row["landing_zone"]),
            f"({float(row['landing_x']):.2f}, {float(row['landing_y']):.2f})",
        ]
        for row in summary["shots"]
    ]
    shot_table = ax.table(
        cellText=top_rows,
        colLabels=["rank", "p", "type", "zone", "landing"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.03, 0.03, 0.94, 0.36],
    )
    shot_table.auto_set_font_size(False)
    shot_table.set_fontsize(8)
    ax.set_title("Expected summary")

    _plot_distribution_axis(
        axes[1, 0],
        summary["shot_type_weights"],
        SHOT_TYPE_ORDER,
        title="Shot type distribution",
        xlabel="top-3 normalized probability",
    )
    _plot_landing_axis(axes[1, 1], summary, top_shots, config)

    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_landing_axis(ax: Any, summary: dict[str, Any], top_shots: list[dict[str, Any]], config: Any) -> None:
    half_w = float(config.court.half_width)
    half_l = float(config.court.half_length)
    service = float(config.court.service_line_distance_from_net)
    ax.set_xlim(-half_w - 0.25, half_w + 0.25)
    ax.set_ylim(-half_l - 0.25, half_l + 0.25)
    ax.set_aspect("equal", adjustable="box")
    for x_values, y_values in (
        ([-half_w, half_w], [-half_l, -half_l]),
        ([-half_w, half_w], [half_l, half_l]),
        ([-half_w, -half_w], [-half_l, half_l]),
        ([half_w, half_w], [-half_l, half_l]),
        ([-half_w, half_w], [0.0, 0.0]),
        ([-half_w, half_w], [-service, -service]),
        ([-half_w, half_w], [service, service]),
        ([0.0, 0.0], [-half_l, half_l]),
    ):
        ax.plot(x_values, y_values, color="0.2", linewidth=1.0, alpha=0.75)

    probabilities = np.asarray([float(row["probability"]) for row in top_shots], dtype=float)
    top3_mass = max(float(probabilities.sum()), 1e-12)
    weights = probabilities / top3_mass
    for weight, row in zip(weights, top_shots):
        ax.scatter(
            [float(row["landing_x"])],
            [float(row["landing_y"])],
            s=80 + 420 * float(weight),
            alpha=0.5,
            label=f"#{int(row['rank'])} {row['shot_type']}",
        )
        ax.text(float(row["landing_x"]), float(row["landing_y"]), f"#{int(row['rank'])}", ha="center", va="center", fontsize=8)
    ax.scatter(
        [float(summary["expected_landing_x"])],
        [float(summary["expected_landing_y"])],
        marker="*",
        s=240,
        color="black",
        label="expectation",
        zorder=5,
    )
    ax.set_title("Landing points and expectation")
    ax.set_xlabel("x across court (m)")
    ax.set_ylabel("y along court (m)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.18)


def _write_distribution_plot(
    path: Path,
    *,
    title: str,
    values: dict[str, float],
    ordered_names: tuple[str, ...],
    xlabel: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    _plot_distribution_axis(ax, values, ordered_names, title=title, xlabel=xlabel)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_distribution_axis(
    ax: Any,
    values: dict[str, float],
    ordered_names: tuple[str, ...],
    *,
    title: str,
    xlabel: str,
) -> None:
    names = [name for name in ordered_names if float(values.get(name, 0.0)) > 1e-12]
    names.extend(sorted(name for name in values if name not in names and float(values.get(name, 0.0)) > 1e-12))
    if not names:
        ax.text(0.5, 0.5, "no top-3 mass", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    y = np.arange(len(names))
    widths = [float(values.get(name, 0.0)) for name in names]
    ax.barh(y, widths, color="C0", alpha=0.82)
    ax.set_yticks(y, labels=[name.replace("_", " ") for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(1.0, max(widths) * 1.08))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    for y_index, value in zip(y, widths):
        ax.text(value + 0.015, y_index, f"{value:.2f}", va="center", fontsize=8)


def _move_default_position_pngs(probe_dir: Path) -> None:
    for scenario_dir in sorted(path for path in probe_dir.iterdir() if path.is_dir() and not path.name.startswith("top_shot")):
        if scenario_dir.name in {"opponent_position_probe_cache"}:
            continue
        pngs = sorted(scenario_dir.glob(f"{scenario_dir.name}_*.png"))
        if not pngs:
            continue
        default_dir = scenario_dir / DEFAULT_POSITION_DIR
        ensure_directory(default_dir)
        for source in pngs:
            target = default_dir / source.name
            if source.resolve() == target.resolve():
                continue
            source.replace(target)


def _write_summary_files(probe_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_dir = probe_dir / "top3_expectation_probe_views"
    ensure_directory(summary_dir)
    json_rows = rows
    (summary_dir / "top3_expectation_probe_summary.json").write_text(
        json.dumps(json_rows, indent=2),
        encoding="utf-8",
    )
    flat_rows = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"shot_type_weights", "landing_zone_weights", "shots"}
        }
        for name, value in row["shot_type_weights"].items():
            flat[f"shot_type_weight_{_field_name(name)}"] = float(value)
        for name, value in row["landing_zone_weights"].items():
            flat[f"landing_zone_weight_{name}"] = float(value)
        flat_rows.append(flat)
    if flat_rows:
        fieldnames = sorted({key for row in flat_rows for key in row})
        with (summary_dir / "top3_expectation_probe_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def _field_name(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_")


def _title(probe_id: str) -> str:
    return probe_id.replace("__", " / ").replace("_", " ")


if __name__ == "__main__":
    main()
