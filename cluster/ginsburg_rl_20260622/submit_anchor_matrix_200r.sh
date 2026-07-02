#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROOT="outputs/rl/ginsburg_20260622"
FAMILIES=(split_cfa pure_cfa noncfa)
SEEDS=(17 23 31 47 59)

for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="$ROOT/${family}_seed${seed}"
    snapshot="$run_dir/anchor_pairwise_200r/anchor_steps.txt"
    mkdir -p "$(dirname "$snapshot")"
    find "$run_dir/anchor_checkpoints" -maxdepth 1 -type f -name 'anchor_step_*.zip' -print \
      | sed -E 's/.*anchor_step_([0-9]+)\.zip/\1/' \
      | sort -n > "$snapshot"
    count="$(wc -l < "$snapshot" | tr -d ' ')"
    if (( count < 2 )); then
      echo "Expected at least two anchors for $run_dir; found $count." >&2
      exit 2
    fi
    echo "Frozen $count anchors for $family seed $seed in $snapshot"
  done
done

mkdir -p cluster/ginsburg_rl_20260622/logs
sbatch --array=0-14 cluster/ginsburg_rl_20260622/eval_anchor_matrix.sbatch
