from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.mpl_config import ensure_writable_matplotlib_config

ensure_writable_matplotlib_config()

import matplotlib.pyplot as plt


_STEP_PATTERN = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot self-play win-rate and invalid-rate evolution.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _report_sort_key(path: Path) -> int:
    matches = _STEP_PATTERN.findall(path.stem)
    if not matches:
        return -1
    return int(matches[-1])


def main() -> None:
    args = parse_args()
    reports = sorted(args.eval_dir.glob("selfplay_eval_*.json"), key=_report_sort_key)
    if not reports:
        raise FileNotFoundError(f"No self-play eval reports found in {args.eval_dir}")

    labels = {
        "current_vs_newest_checkpoint": "vs newest checkpoint",
        "current_vs_mirror_self": "vs mirror self",
        "current_vs_anchor_checkpoint": "vs anchor checkpoint",
    }
    win_series: dict[str, list[tuple[int, float]]] = {key: [] for key in labels}
    invalid_series: dict[str, list[tuple[int, float]]] = {key: [] for key in labels}

    for report_path in reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        step = int(payload["num_timesteps"])
        summaries = payload.get("current_matchup_summaries", {})
        for key in labels:
            summary = summaries.get(key)
            if not isinstance(summary, dict):
                continue
            if "win_rate" in summary:
                win_series[key].append((step, float(summary["win_rate"])))
            if "avg_invalid_action_rate" in summary:
                invalid_series[key].append((step, float(summary["avg_invalid_action_rate"])))

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    for key, label in labels.items():
        points = sorted(win_series[key], key=lambda item: item[0])
        if points:
            axes[0].plot(
                [step for step, _ in points],
                [value for _, value in points],
                marker="o",
                linewidth=2.0,
                label=label,
            )
        invalid_points = sorted(invalid_series[key], key=lambda item: item[0])
        if invalid_points:
            axes[1].plot(
                [step for step, _ in invalid_points],
                [value for _, value in invalid_points],
                marker="o",
                linewidth=2.0,
                label=label,
            )

    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Self-play evaluation evolution")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Training timesteps")
    axes[1].set_ylabel("Invalid action rate")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
