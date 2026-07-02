#!/usr/bin/env bash
# Evaluate same-seed split-CFA and pure-CFA checkpoints against the June 11 fixed pool.
# Submit with: sbatch --array=0-4 cluster/ginsburg_rl_20260622/eval_split_pure_same_fixed_pool.sbatch
#SBATCH --job-name=rlbdm_sp_eval
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_rl_20260622/logs/%x_%A_%a.out
#SBATCH --error=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_rl_20260622/logs/%x_%A_%a.err

set -euo pipefail

REPO="${REPO:-/burg-archive/home/pd2714/quantum/RL_badminton}"
cd "$REPO"

mkdir -p cluster/ginsburg_rl_20260622/logs
module load anaconda/3-2023.09
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl_badminton_matplotlib_${USER}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$MPLCONFIGDIR"

SEEDS=(17 23 31 47 59)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]}"
ROOT="outputs/rl/ginsburg_20260622"
OUT_DIR="$ROOT/same_june11_fixed_pool_eval_200r/seed_${SEED}/split_vs_pure_cfa"
BASE_REPORT="outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611/cross_run_fixed_pool_eval_200r/fixed_pool_eval_report.json"

if [[ -f "$OUT_DIR/fixed_pool_eval_report.json" ]]; then
  remaining="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("remaining_pair_count", 0))' "$OUT_DIR/fixed_pool_eval_report.json")"
  if [[ "$remaining" == "0" ]]; then
    echo "already complete: $OUT_DIR"
    exit 0
  fi
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  next_job="$(sbatch --parsable --array="$TASK_ID" --dependency="afterany:${SLURM_JOB_ID}" "$0")"
  echo "scheduled_continuation_job: $next_job afterany:$SLURM_JOB_ID"
fi

python scripts/evaluate_runs_same_fixed_pool.py \
  --base-report "$BASE_REPORT" \
  --output-dir "$OUT_DIR" \
  --run "split_cfa_seed${SEED}" "$ROOT/split_cfa_seed${SEED}" \
  --run "pure_cfa_seed${SEED}" "$ROOT/pure_cfa_seed${SEED}" \
  --step-min 3200000 \
  --step-interval 400000 \
  --auto-step-mode intersection \
  --freeze-steps-file "$OUT_DIR/frozen_steps.json" \
  --eval-rallies 200 \
  --seed "$((2026062200 + SEED))"
