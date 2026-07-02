#!/usr/bin/env bash
# One seed-paired, shared-pool Elo evaluation for Fig. ablations panel B.
#SBATCH --job-name=rlbdm_pB_eval
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@600
#SBATCH --array=0-4
#SBATCH --output=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_panel_b_20260626/logs/%x_%A_%a.out
#SBATCH --error=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_panel_b_20260626/logs/%x_%A_%a.err

set -euo pipefail
REPO="${REPO:-/burg-archive/home/pd2714/quantum/RL_badminton}"
cd "$REPO"
module load anaconda/3-2023.09
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl_badminton_matplotlib_${USER}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$MPLCONFIGDIR" cluster/ginsburg_panel_b_20260626/logs

SEEDS=(17 23 31 47 59)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"; SEED="${SEEDS[$TASK_ID]}"
ROOT="outputs/rl/ginsburg_panel_b_20260626"
OUT_DIR="$ROOT/eval/seed_${SEED}/panel_b_shared_pool_elo_200r"

resubmit() {
  sbatch --array="$TASK_ID" --dependency="afterany:${SLURM_JOB_ID}" "$0"
}
trap 'echo "pre-timeout; resubmitting"; resubmit' USR1
latest_step() {
  find "$1/checkpoint_pool" -maxdepth 1 -type f -name 'selfplay_step_*.zip' 2>/dev/null \
    | sed -E 's/.*selfplay_step_([0-9]+)\.zip/\1/' | sort -n | tail -n 1
}
for family in purecfa nocfa; do
  step="$(latest_step "$ROOT/${family}_seed${SEED}" || true)"; step="${step:-0}"
  echo "readiness: $family seed $SEED -> $step / 3200000"
  if (( step < 3200000 )); then
    sbatch --array="$TASK_ID" --begin=now+2hours "$0"
    exit 0
  fi
done

if [[ -f "$OUT_DIR/fixed_pool_eval_report.json" ]]; then
  echo "already complete: $OUT_DIR"
  exit 0
fi
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/evaluate_cross_run_fixed_pool.py \
  "$ROOT/purecfa_seed${SEED}" "$ROOT/nocfa_seed${SEED}" \
  --output-dir "$OUT_DIR" \
  --eval-steps 400000 800000 1200000 1600000 2000000 2400000 2800000 3200000 \
  --pool-steps 400000 800000 1200000 1600000 2000000 2400000 2800000 3200000 \
  --eval-rallies 200 --seed 20260611
