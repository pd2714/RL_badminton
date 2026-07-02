#!/usr/bin/env bash
# Fig. ablations, panel B: identical recency self-play except for CRA.
# Submit with: sbatch --array=0-4 train_panel_b.sbatch {purecfa|nocfa}
#SBATCH --job-name=rlbdm_pB_train
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_panel_b_20260626/logs/%x_%A_%a.out
#SBATCH --error=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_panel_b_20260626/logs/%x_%A_%a.err

set -euo pipefail

REPO="${REPO:-/burg-archive/home/pd2714/quantum/RL_badminton}"
cd "$REPO"
module load anaconda/3-2023.09
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl_badminton_matplotlib_${USER}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" cluster/ginsburg_panel_b_20260626/logs

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch train_panel_b.sbatch purecfa|nocfa [SEED]" >&2
  exit 2
fi
FAMILY="$1"
SEEDS=(17 23 31 47 59)
SEED="${2:-${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}}"
case "$FAMILY" in purecfa|nocfa) ;; *) echo "Unknown FAMILY: $FAMILY" >&2; exit 2 ;; esac

BASE_HEURISTIC="outputs/rl/velocity_2d_heuristic_1k_fast/final_model.zip"
ROOT="outputs/rl/ginsburg_panel_b_20260626"
RUN_DIR="$ROOT/${FAMILY}_seed${SEED}"
TARGET=3200000
ANCHOR_INTERVAL=400000

latest_step() {
  find "$1/checkpoint_pool" -maxdepth 1 -type f -name 'selfplay_step_*.zip' 2>/dev/null \
    | sed -E 's/.*selfplay_step_([0-9]+)\.zip/\1/' | sort -n | tail -n 1
}
step="$(latest_step "$RUN_DIR" || true)"; step="${step:-0}"
if (( step >= TARGET )); then
  echo "$FAMILY seed $SEED already reached $step steps (target $TARGET)"
  exit 0
fi
if [[ "$step" == 0 ]]; then base_checkpoint="$BASE_HEURISTIC"; else base_checkpoint="$RUN_DIR/checkpoint_pool/selfplay_step_${step}.zip"; fi
if [[ ! -f "$base_checkpoint" ]]; then echo "Missing base checkpoint: $base_checkpoint" >&2; exit 2; fi

# Continue in 12-hour allocations until the 3.2M panel-B target is complete.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  next_job="$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" "$0" "$FAMILY" "$SEED")"
  echo "scheduled_continuation_job: $next_job"
fi

common_args=(
  --court-mode 2d --policy-type velocity_oriented --trajectory-mode drag_square
  --reaction-time 0.15 --player-speed 5.0 --movement-model accelerated --player-acceleration 8.0
  --racket-length 1.6 --max-hitting-height 2.6 --shuttle-speed-min 0.1 --shuttle-speed-max 100.0
  --recovery-x-margin 0.25 --recovery-net-margin 0.3 --recovery-back-margin 0.5 --intercept-count 20
  --reaction-miss-fast-threshold 0.1 --reaction-miss-fast-probability 0.8
  --reaction-miss-secondary-threshold 0.5 --reaction-miss-secondary-probability 0.0
  --reaction-miss-zero-threshold 0.5 --phi-bins 11 --theta-bins 8 --speed-bins 5 --x-rec-bins 5 --y-rec-bins 5
  --max-rally-stages 120 --max-rally-penalty 1.0 --loop-penalty 0.1 --loop-window 4
  --checkpoint-recency-power 3.0 --recent-opponent-weight 0.9 --older-opponent-weight 0.1
  --save-interval 2000 --eval-freq 100000 --eval-episodes 12 --anchor-eval-interval "$ANCHOR_INTERVAL"
  --n-envs 16 --vec-env subproc --seed "$SEED" --ent-coef 0.002 --ent-coef-final 0.002 --no-use-shot-cf
  --heuristic-opponent-prob 0.10 --opponent-sampling-mode recency
)
cfa_args=(
  --use-recovery-factorized-advantage --recovery-counterfactual-baseline average
  --recovery-counterfactual-other-sample-count 24 --recovery-counterfactual-advantage-coef 0.05
  --recovery-counterfactual-distribution-coef 0.0 --recovery-counterfactual-distribution-temperature 0.25
  --counterfactual-opponent-response-samples 1 --recovery-counterfactual-expected-response-target
  --recovery-full-diagnostics-probability 0.0
)
nocfa_args=(
  --use-recovery-factorized-advantage --recovery-counterfactual-baseline average
  --recovery-counterfactual-other-sample-count 0 --recovery-counterfactual-advantage-coef 0.0
  --recovery-counterfactual-distribution-coef 0.0 --recovery-counterfactual-distribution-temperature 0.25
  --counterfactual-opponent-response-samples 1 --no-recovery-counterfactual-expected-response-target
  --recovery-full-diagnostics-probability 0.0
)
if [[ "$FAMILY" == purecfa ]]; then family_args=("${cfa_args[@]}"); else family_args=("${nocfa_args[@]}"); fi

LOG="$RUN_DIR/logs/train_${FAMILY}_seed${SEED}_${step}_to_${TARGET}_${SLURM_JOB_ID:-manual}.log"
mkdir -p "$(dirname "$LOG")"
{
  echo "started: $(date)"; echo "family: $FAMILY"; echo "seed: $SEED"; echo "resume_step: $step"; echo "target: $TARGET"
  python scripts/train_selfplay.py --base-checkpoint-path "$base_checkpoint" --output-dir "$RUN_DIR" \
    --selfplay-total-timesteps "$((TARGET - step))" --resume-step-offset "$step" "${common_args[@]}" "${family_args[@]}"
} > "$LOG" 2>&1
