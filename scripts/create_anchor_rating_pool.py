from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.eval_evolution import checkpoint_step, discover_anchor_checkpoints, filter_anchor_checkpoints
from badminton.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed rating pool from a run's anchor checkpoints.")
    parser.add_argument("run_dir", type=Path, help="Self-play run directory containing anchor_checkpoints.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Rating-pool directory to create.")
    parser.add_argument("--anchor-step-min", type=int, default=None)
    parser.add_argument("--anchor-step-max", type=int, default=None)
    parser.add_argument("--anchor-step-interval", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.anchor_step_interval is not None and args.anchor_step_interval <= 0:
        raise ValueError("--anchor-step-interval must be positive")

    run_dir = args.run_dir
    output_dir = args.output_dir
    ensure_directory(output_dir)

    config_path = run_dir / "selfplay_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    checkpoints = filter_anchor_checkpoints(
        discover_anchor_checkpoints(run_dir),
        step_min=args.anchor_step_min,
        step_max=args.anchor_step_max,
        step_interval=args.anchor_step_interval,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints selected from {run_dir}")

    agents: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        label = f"anchor_{step}"
        agent_dir = output_dir / label
        ensure_directory(agent_dir)
        _replace_symlink(agent_dir / "final_model.zip", checkpoint)
        _replace_symlink(agent_dir / "selfplay_config.json", config_path)
        agents.append({"run_dir": label, "label": label, "step": int(step)})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"agents": agents}, indent=2), encoding="utf-8")
    print(f"rating_pool: {output_dir}")
    print(f"agents: {len(agents)}")
    print(f"steps: {checkpoint_step(checkpoints[0])}..{checkpoint_step(checkpoints[-1])}")


def _replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        return
    relative_target = Path(os.path.relpath(target, start=path.parent))
    path.symlink_to(relative_target)


if __name__ == "__main__":
    main()
