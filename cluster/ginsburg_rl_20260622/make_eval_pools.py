from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


FAMILIES = {
    "split_cfa": ("cfa_splitlinear", 6_000_000),
    "pure_cfa": ("cfa_purerecency", 6_000_000),
    "noncfa": ("noncfa_ablation", 3_000_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create symlinked rating pools for Ginsburg RL badminton jobs.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step-interval", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    eval_root = repo / "outputs/rl/ginsburg_20260622/eval" / f"seed_{args.seed}"
    eval_root.mkdir(parents=True, exist_ok=True)

    for family, (label, target_step) in FAMILIES.items():
        run_dir = repo / "outputs/rl/ginsburg_20260622" / f"{family}_seed{args.seed}"
        pool_dir = eval_root / f"{label}_0_to_{target_step // 1_000_000}m_pool"
        create_pool(
            pool_dir=pool_dir,
            entries=family_entries(
                run_dir=run_dir,
                label=label,
                step_min=0,
                step_max=target_step,
                step_interval=args.step_interval,
            ),
        )
        print(pool_dir)

    combined = []
    for family, (label, _target_step) in FAMILIES.items():
        run_dir = repo / "outputs/rl/ginsburg_20260622" / f"{family}_seed{args.seed}"
        combined.extend(
            family_entries(
                run_dir=run_dir,
                label=label,
                step_min=0,
                step_max=3_000_000,
                step_interval=args.step_interval,
            )
        )
    combined_pool = eval_root / "combined_cfa_vs_noncfa_0_to_3m_pool"
    create_pool(pool_dir=combined_pool, entries=combined)
    print(combined_pool)


def family_entries(
    *,
    run_dir: Path,
    label: str,
    step_min: int,
    step_max: int,
    step_interval: int,
) -> list[dict[str, object]]:
    config_path = run_dir / "selfplay_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    entries: list[dict[str, object]] = []
    for step in range(step_min, step_max + 1, step_interval):
        checkpoint_path = run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip"
        if not checkpoint_path.exists() and step == 0:
            checkpoint_path = run_dir / "compatible_base_model.zip"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {label} seed pool: {checkpoint_path}")
        entries.append(
            {
                "label": f"{label}_step{step:07d}",
                "step": step,
                "checkpoint_path": checkpoint_path,
                "config_path": config_path,
            }
        )
    return entries


def create_pool(*, pool_dir: Path, entries: list[dict[str, object]]) -> None:
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    manifest_agents: list[dict[str, object]] = []
    for entry in entries:
        label = str(entry["label"])
        agent_dir = pool_dir / label
        agent_dir.mkdir(parents=True, exist_ok=True)
        _replace_symlink(agent_dir / "final_model.zip", Path(entry["checkpoint_path"]))
        _replace_symlink(agent_dir / "selfplay_config.json", Path(entry["config_path"]))
        manifest_agents.append({"run_dir": label, "label": label, "step": int(entry["step"])})

    (pool_dir / "manifest.json").write_text(
        json.dumps({"agents": manifest_agents}, indent=2),
        encoding="utf-8",
    )


def _replace_symlink(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
    relative_target = Path(os.path.relpath(target.resolve(), start=path.parent.resolve()))
    path.symlink_to(relative_target)


if __name__ == "__main__":
    main()
