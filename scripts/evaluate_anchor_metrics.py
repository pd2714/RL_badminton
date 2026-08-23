from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.eval_evolution import AnchorEvaluationConfig, evaluate_anchor_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shot and pressure metrics across anchor checkpoints.")
    parser.add_argument("run_dir", type=Path, help="Self-play run directory containing selfplay_config.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to RUN_DIR/anchor_metric_eval.")
    parser.add_argument("--episodes", type=int, default=24, help="Rallies per anchor, or per fixed-pool opponent when --rating-pool-dir is set.")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-anchors", type=int, default=None, help="Evaluate only the first N sorted anchors.")
    parser.add_argument("--anchor-stride", type=int, default=1, help="Evaluate every Nth sorted anchor.")
    parser.add_argument("--anchor-step-min", type=int, default=None)
    parser.add_argument("--anchor-step-max", type=int, default=None)
    parser.add_argument("--anchor-step-interval", type=int, default=None)
    parser.add_argument("--rating-pool-dir", type=Path, default=None, help="Fixed rating-pool directory to evaluate against.")
    parser.add_argument("--pool-checkpoint-name", default="final_model.zip")
    parser.add_argument(
        "--skip-recovery-choice-diagnostics",
        action="store_true",
        help="Do not recompute recovery-bin rank/best-bin diagnostics during anchor evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.anchor_stride <= 0:
        raise ValueError("--anchor-stride must be positive")
    if args.anchor_step_interval is not None and args.anchor_step_interval <= 0:
        raise ValueError("--anchor-step-interval must be positive")
    output_dir = args.output_dir or (args.run_dir / "anchor_metric_eval")
    report = evaluate_anchor_folder(
        args.run_dir,
        output_dir,
        AnchorEvaluationConfig(
            episodes=args.episodes,
            seed=args.seed,
            deterministic=args.deterministic,
            max_anchors=args.max_anchors,
            anchor_stride=args.anchor_stride,
            anchor_step_min=args.anchor_step_min,
            anchor_step_max=args.anchor_step_max,
            anchor_step_interval=args.anchor_step_interval,
            rating_pool_dir=args.rating_pool_dir,
            pool_checkpoint_name=args.pool_checkpoint_name,
            recovery_choice_diagnostics=not args.skip_recovery_choice_diagnostics,
        ),
    )
    print(f"anchors: {report['anchor_count']}")
    print(f"json: {output_dir / 'anchor_metric_evolution.json'}")
    print(f"csv: {output_dir / 'anchor_metric_evolution.csv'}")
    for name, path in report.get("plots", {}).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
