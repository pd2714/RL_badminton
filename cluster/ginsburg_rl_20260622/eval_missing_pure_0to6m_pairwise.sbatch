#!/usr/bin/env bash
# Finish the missing pure-CFA 0-6M pairwise Elo ratings for seeds 47 and 59.
# Submit with: sbatch --array=0-1 cluster/ginsburg_rl_20260622/eval_missing_pure_0to6m_pairwise.sbatch
# Array index 0 -> seed 47, index 1 -> seed 59.
# Step 0 is intentionally still present in the raw pool for reproducibility;
# downstream plotting refits Elo without step-0 matchups.
#SBATCH --job-name=rlbdm_pure6m_elo
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

SEEDS=(47 59)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]}"
ROOT="outputs/rl/ginsburg_20260622"
POOL_DIR="$ROOT/eval/seed_${SEED}/cfa_purerecency_0_to_6m_pool"
OUT_DIR="$ROOT/eval/seed_${SEED}/cfa_purerecency_0_to_6m_pool_elo_200r"

if [[ ! -f "$POOL_DIR/manifest.json" ]]; then
  echo "Missing pool manifest: $POOL_DIR/manifest.json" >&2
  exit 2
fi

if [[ -f "$OUT_DIR/elo_rating_report.json" ]]; then
  echo "Already complete: $OUT_DIR/elo_rating_report.json"
  exit 0
fi

echo "Started pure-CFA 0-6M pairwise Elo for seed $SEED at $(date)"
python scripts/rate_fixed_pool_elo.py "$POOL_DIR" \
  --output-dir "$OUT_DIR" \
  --eval-rallies-per-side 100 \
  --seed "$((2026062200 + SEED))"
echo "Finished pure-CFA 0-6M pairwise Elo for seed $SEED at $(date)"
