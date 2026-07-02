#!/usr/bin/env bash
# Five-seed Figure 6A recovery ablation: learned recovery vs centered recovery.
# Submit with:
#   sbatch cluster/ginsburg_6a_20260701/eval_learned_centered_5seed.sbatch
# Optional:
#   EVAL_RALLIES=50 sbatch --array=0 cluster/ginsburg_6a_20260701/eval_learned_centered_5seed.sbatch
#SBATCH --job-name=rlbdm_6a_rec
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@600
#SBATCH --array=0-4
#SBATCH --output=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_6a_20260701/logs/%x_%A_%a.out
#SBATCH --error=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_6a_20260701/logs/%x_%A_%a.err

set -euo pipefail

REPO="${REPO:-/burg-archive/home/pd2714/quantum/RL_badminton}"
cd "$REPO"

module load anaconda/3-2023.09
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl_badminton_matplotlib_${USER}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$MPLCONFIGDIR" cluster/ginsburg_6a_20260701/logs

SEEDS=(17 23 31 47 59)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]}"

ROOT="outputs/rl/ginsburg_20260622"
RUN_DIR="$ROOT/pure_cfa_seed${SEED}"
OUT_DIR="$ROOT/eval/seed_${SEED}/panel_6a_learned_centered_recovery_200r"
EVAL_RALLIES="${EVAL_RALLIES:-200}"

resubmit_self() {
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sbatch --array="$TASK_ID" --dependency="afterany:${SLURM_JOB_ID}" "$0"
  else
    sbatch --array="$TASK_ID" "$0"
  fi
}

on_time_signal() {
  echo "Received pre-timeout signal at $(date); resubmitting seed $SEED"
  resubmit_self
}
trap on_time_signal USR1

latest_step() {
  find "$1/checkpoint_pool" -maxdepth 1 -type f -name 'selfplay_step_*.zip' 2>/dev/null \
    | sed -E 's/.*selfplay_step_([0-9]+)\.zip/\1/' \
    | sort -n \
    | tail -n 1
}

step="$(latest_step "$RUN_DIR" || true)"
step="${step:-0}"
echo "readiness pure_cfa seed $SEED: step=$step target=6000000"
if (( step < 6000000 )); then
  echo "Training not complete for seed $SEED; resubmitting for later."
  sbatch --array="$TASK_ID" --begin=now+2hours "$0"
  exit 0
fi

echo "Evaluation started for seed $SEED at $(date)"
python scripts/evaluate_recovery_ablation_fixed_pool.py "$RUN_DIR" \
  --output-dir "$OUT_DIR" \
  --recovery-modes learned centered \
  --no-heuristic-safe-opponent \
  --steps 200000 1000000 2000000 3000000 4000000 5000000 6000000 \
  --pool-steps 2000000 4000000 6000000 \
  --eval-rallies "$EVAL_RALLIES" \
  --seed 20260611

echo "Evaluation finished for seed $SEED at $(date)"
