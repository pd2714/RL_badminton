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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot self-play win-rate evolution from evaluation JSON files.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


_STEP_PATTERN = re.compile(r"(\d+)")


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

    series: dict[str, list[tuple[int, float]]] = {
        "current_vs_newest_checkpoint": [],
        "current_vs_mirror_self": [],
        "current_vs_anchor_checkpoint": [],
    }
    for report_path in reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        step = int(payload["num_timesteps"])
        summaries = payload.get("current_matchup_summaries", {})
        for key in series:
            summary = summaries.get(key)
            if isinstance(summary, dict) and "win_rate" in summary:
                series[key].append((step, float(summary["win_rate"])))

    plt.figure(figsize=(8, 4.8))
    labels = {
        "current_vs_newest_checkpoint": "vs newest checkpoint",
        "current_vs_mirror_self": "vs mirror self",
        "current_vs_anchor_checkpoint": "vs anchor checkpoint",
    }
    for key, points in series.items():
        if not points:
            continue
        points = sorted(points, key=lambda item: item[0])
        xs = [step for step, _ in points]
        ys = [value for _, value in points]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=labels[key])

    plt.ylim(0.0, 1.0)
    plt.xlabel("Training timesteps")
    plt.ylabel("Win rate")
    plt.title("Self-play win-rate evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
