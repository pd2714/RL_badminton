#!/usr/bin/env bash
#SBATCH --job-name=rlbdm_cfa_bench
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=0-7
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
mkdir -p "$MPLCONFIGDIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAMILIES=(split_cfa pure_cfa)
VARIANT_NAMES=(current_threads pinned_env8 pinned_env12 pinned_env16)
VARIANT_N_ENVS=(8 8 12 16)
VARIANT_THREADS=(16 1 1 1)

FAMILY_INDEX=$((TASK_ID / ${#VARIANT_NAMES[@]}))
VARIANT_INDEX=$((TASK_ID % ${#VARIANT_NAMES[@]}))
FAMILY="${FAMILIES[$FAMILY_INDEX]}"
VARIANT="${VARIANT_NAMES[$VARIANT_INDEX]}"
N_ENVS="${VARIANT_N_ENVS[$VARIANT_INDEX]}"
THREADS="${VARIANT_THREADS[$VARIANT_INDEX]}"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

BASE_HEURISTIC="outputs/rl/velocity_2d_heuristic_1k_fast/final_model.zip"
ROOT="outputs/rl/ginsburg_20260623_cfa_speed_bench"
RUN_DIR="$ROOT/${FAMILY}_${VARIANT}_job${SLURM_JOB_ID:-manual}_${TASK_ID}"
LOG_DIR="$ROOT/logs"
TRAIN_LOG="$LOG_DIR/${FAMILY}_${VARIANT}_job${SLURM_JOB_ID:-manual}_${TASK_ID}.log"
TIMESTEPS="${BENCH_TIMESTEPS:-32768}"
SEED=$((9017 + TASK_ID))

mkdir -p "$LOG_DIR"

if [[ ! -f "$BASE_HEURISTIC" ]]; then
  echo "Missing heuristic base checkpoint: $BASE_HEURISTIC" >&2
  exit 2
fi

common_args=(
  --court-mode 2d
  --policy-type velocity_oriented
  --trajectory-mode drag_square
  --reaction-time 0.15
  --player-speed 5.0
  --movement-model accelerated
  --player-acceleration 8.0
  --racket-length 1.6
  --max-hitting-height 2.6
  --shuttle-speed-min 0.1
  --shuttle-speed-max 100.0
  --recovery-x-margin 0.25
  --recovery-net-margin 0.3
  --recovery-back-margin 0.5
  --intercept-count 20
  --reaction-miss-fast-threshold 0.1
  --reaction-miss-fast-probability 0.8
  --reaction-miss-secondary-threshold 0.5
  --reaction-miss-secondary-probability 0.0
  --reaction-miss-zero-threshold 0.5
  --phi-bins 11
  --theta-bins 8
  --speed-bins 5
  --x-rec-bins 5
  --y-rec-bins 5
  --max-rally-stages 120
  --max-rally-penalty 1.0
  --loop-penalty 0.1
  --loop-window 4
  --checkpoint-recency-power 3.0
  --recent-opponent-weight 0.9
  --older-opponent-weight 0.1
  --save-interval 1000000000
  --eval-freq 1000000000
  --eval-episodes 1
  --anchor-eval-interval 0
  --n-envs "$N_ENVS"
  --vec-env subproc
  --seed "$SEED"
  --ent-coef 0.002
  --ent-coef-final 0.002
  --no-use-shot-cf
  --heuristic-opponent-prob 0.10
  --opponent-sampling-mode recency
  --use-recovery-factorized-advantage
  --recovery-counterfactual-baseline average
  --recovery-counterfactual-other-sample-count 24
  --recovery-counterfactual-advantage-coef 0.05
  --recovery-counterfactual-distribution-coef 0.0
  --recovery-counterfactual-distribution-temperature 0.25
  --counterfactual-opponent-response-samples 1
  --recovery-counterfactual-expected-response-target
  --recovery-full-diagnostics-probability 0.0
)

{
  echo "started: $(date)"
  echo "hostname: $(hostname)"
  echo "family_label: $FAMILY"
  echo "variant: $VARIANT"
  echo "run_dir: $RUN_DIR"
  echo "timesteps: $TIMESTEPS"
  echo "n_envs: $N_ENVS"
  echo "threads: $THREADS"
  echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-unset}"
  echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-unset}"
  which python
  python --version
  python -c "import os, torch; print('torch_threads', torch.get_num_threads(), 'interop', torch.get_num_interop_threads(), 'omp', os.environ.get('OMP_NUM_THREADS'))"

  python scripts/train_selfplay.py \
    --base-checkpoint-path "$BASE_HEURISTIC" \
    --output-dir "$RUN_DIR" \
    --selfplay-total-timesteps "$TIMESTEPS" \
    "${common_args[@]}"

  echo "training_finished: $(date)"
} > "$TRAIN_LOG" 2>&1
