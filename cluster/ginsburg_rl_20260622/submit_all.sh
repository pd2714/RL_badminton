#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p cluster/ginsburg_rl_20260622/logs

echo "Submitting 5-seed split CFA jobs: 0-3M recency, 3-6M linear-recency historical anchors"
sbatch --array=0-4 cluster/ginsburg_rl_20260622/train_family.sbatch split_cfa

echo "Submitting 5-seed pure-recency CFA jobs: 0-6M recency"
sbatch --array=0-4 cluster/ginsburg_rl_20260622/train_family.sbatch pure_cfa

echo "Submitting 5-seed non-CFA ablation jobs: 0-3M recency, exact disabled-CFA coefficients"
sbatch --array=0-4 cluster/ginsburg_rl_20260622/train_family.sbatch noncfa

echo "Submitting evaluation watcher array; each seed waits until all three family targets are present"
sbatch --begin=now+12hours cluster/ginsburg_rl_20260622/eval_when_ready.sbatch
