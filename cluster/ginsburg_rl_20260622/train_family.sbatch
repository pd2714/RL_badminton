#!/usr/bin/env bash
#SBATCH --job-name=rlbdm_train
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_rl_20260622/logs/%x_%A_%a.out
#SBATCH --error=/burg-archive/home/pd2714/quantum/RL_badminton/cluster/ginsburg_rl_20260622/logs/%x_%A_%a.err

set -euo pipefail

REPO="${REPO:-/burg-archive/home/pd2714/quantum/RL_badminton}"
cd "$REPO"

mkdir -p cluster/ginsburg_rl_20260622/logs
module load anaconda/3-2023.09
if [[ -f .venv/bin/activate ]]; then
  # Use the project-local environment on Ginsburg; the base module lacks SB3.
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl_badminton_matplotlib_${USER}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

FAMILY="${1:?usage: sbatch train_family.sbatch FAMILY [SEED]}"
SEEDS=(17 23 31 47 59)
if [[ $# -ge 2 ]]; then
  SEED="$2"
else
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  SEED="${SEEDS[$TASK_ID]}"
fi

BASE_HEURISTIC="outputs/rl/velocity_2d_heuristic_1k_fast/final_model.zip"
ROOT="outputs/rl/ginsburg_20260622"
RUN_DIR="$ROOT/${FAMILY}_seed${SEED}"
LOG_DIR="$RUN_DIR/logs"
ANCHOR_INTERVAL=200000
SAVE_INTERVAL=2000
EVAL_FREQ=100000
N_ENVS=16
mkdir -p "$LOG_DIR"

if [[ ! -f "$BASE_HEURISTIC" ]]; then
  echo "Missing heuristic base checkpoint: $BASE_HEURISTIC" >&2
  exit 2
fi

schedule_afterany_continuation() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    return
  fi
  local continuation_id
  continuation_id="$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" "$0" "$FAMILY" "$SEED")"
  echo "scheduled_continuation_job: $continuation_id afterany:$SLURM_JOB_ID"
}

latest_step() {
  local run_dir="$1"
  find "$run_dir/checkpoint_pool" -maxdepth 1 -type f -name 'selfplay_step_*.zip' 2>/dev/null \
    | sed -E 's/.*selfplay_step_([0-9]+)\.zip/\1/' \
    | sort -n \
    | tail -n 1
}

latest_checkpoint() {
  local run_dir="$1"
  local step="$2"
  if [[ -n "$step" && -f "$run_dir/checkpoint_pool/selfplay_step_${step}.zip" ]]; then
    printf '%s\n' "$run_dir/checkpoint_pool/selfplay_step_${step}.zip"
  elif [[ -f "$run_dir/latest_model.zip" ]]; then
    printf '%s\n' "$run_dir/latest_model.zip"
  else
    printf '%s\n' "$BASE_HEURISTIC"
  fi
}

checkpoint_step_from_path() {
  local checkpoint_path="$1"
  local checkpoint_name="${checkpoint_path##*/}"
  case "$checkpoint_name" in
    selfplay_step_*.zip)
      local checkpoint_step="${checkpoint_name#selfplay_step_}"
      checkpoint_step="${checkpoint_step%.zip}"
      case "$checkpoint_step" in
        ''|*[!0-9]*) ;;
        *) printf '%s\n' "$checkpoint_step" ;;
      esac
      ;;
  esac
}

prepare_split_historical_anchors() {
  local run_dir="$1"
  local anchor_dir="$run_dir/historical_anchors"
  mkdir -p "$anchor_dir"
  # The heuristic base checkpoint uses the pre-factorized action head and cannot
  # be loaded as a factorized PPO opponent during split-CFA stage 2.
  rm -f "$anchor_dir/anchor_step_0.zip"
  local anchor_step
  for anchor_step in $(seq "$ANCHOR_INTERVAL" "$ANCHOR_INTERVAL" 3000000); do
    local src="$run_dir/anchor_checkpoints/anchor_step_${anchor_step}.zip"
    local dst="$anchor_dir/anchor_step_${anchor_step}.zip"
    if [[ -f "$src" && ! -f "$dst" ]]; then
      cp -p "$src" "$dst"
    fi
  done
}

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
  --save-interval "$SAVE_INTERVAL"
  --eval-freq "$EVAL_FREQ"
  --eval-episodes 12
  --anchor-eval-interval "$ANCHOR_INTERVAL"
  --n-envs "$N_ENVS"
  --vec-env subproc
  --seed "$SEED"
  --ent-coef 0.002
  --ent-coef-final 0.002
  --no-use-shot-cf
)

cfa_args=(
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

noncfa_args=(
  --use-recovery-factorized-advantage
  --recovery-counterfactual-baseline average
  --recovery-counterfactual-other-sample-count 0
  --recovery-counterfactual-advantage-coef 0.0
  --recovery-counterfactual-distribution-coef 0.0
  --recovery-counterfactual-distribution-temperature 0.25
  --counterfactual-opponent-response-samples 1
  --no-recovery-counterfactual-expected-response-target
  --recovery-full-diagnostics-probability 0.0
)

step="$(latest_step "$RUN_DIR" || true)"
step="${step:-0}"
base_checkpoint="$(latest_checkpoint "$RUN_DIR" "$step")"
checkpoint_step="$(checkpoint_step_from_path "$base_checkpoint" || true)"
checkpoint_step="${checkpoint_step:-0}"
if (( checkpoint_step > step )); then
  echo "correcting_resume_step_from_checkpoint: latest_step=$step checkpoint_step=$checkpoint_step checkpoint=$base_checkpoint"
  step="$checkpoint_step"
fi

target=0
sampling_args=()
family_args=()
case "$FAMILY" in
  split_cfa)
    family_args=(--heuristic-opponent-prob 0.10 "${cfa_args[@]}")
    if (( step < 3000000 )); then
      target=3000000
      sampling_args=(--opponent-sampling-mode recency)
    else
      target=6000000
      prepare_split_historical_anchors "$RUN_DIR"
      sampling_args=(
        --opponent-sampling-mode variety
        --historical-anchor-dir "$RUN_DIR/historical_anchors"
        --historical-anchor-weight 0.70
        --recent-continuation-weight 0.15
        --newest-continuation-weight 0.05
      )
    fi
    ;;
  pure_cfa)
    target=6000000
    family_args=(--heuristic-opponent-prob 0.10 "${cfa_args[@]}")
    sampling_args=(--opponent-sampling-mode recency)
    ;;
  noncfa)
    target=3000000
    family_args=(--heuristic-opponent-prob 0.05 "${noncfa_args[@]}")
    sampling_args=(--opponent-sampling-mode recency)
    ;;
  *)
    echo "Unknown FAMILY: $FAMILY" >&2
    exit 2
    ;;
esac

if (( step >= target )); then
  echo "$FAMILY seed $SEED already reached target $target at step $step"
  if [[ "$FAMILY" == "split_cfa" && "$target" == "3000000" ]]; then
    sbatch "$0" "$FAMILY" "$SEED"
  fi
  exit 0
fi

remaining=$((target - step))
train_log="$LOG_DIR/train_${FAMILY}_seed${SEED}_${step}_to_${target}_${SLURM_JOB_ID:-manual}.log"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run: family=$FAMILY seed=$SEED step=$step target=$target remaining=$remaining base_checkpoint=$base_checkpoint"
  exit 0
fi
schedule_afterany_continuation

{
  echo "started: $(date)"
  echo "hostname: $(hostname)"
  echo "family: $FAMILY"
  echo "seed: $SEED"
  echo "run_dir: $RUN_DIR"
  echo "base_checkpoint: $base_checkpoint"
  echo "resume_step_offset: $step"
  echo "target: $target"
  echo "remaining_selfplay_timesteps: $remaining"
  echo "anchor_eval_interval: $ANCHOR_INTERVAL"
  echo "shot_cf: disabled"
  which python
  python --version
  python -c "import stable_baselines3, torch, numpy; print('imports_ok', stable_baselines3.__version__, torch.__version__, numpy.__version__)"

  python scripts/train_selfplay.py \
    --base-checkpoint-path "$base_checkpoint" \
    --output-dir "$RUN_DIR" \
    --selfplay-total-timesteps "$remaining" \
    --resume-step-offset "$step" \
    "${common_args[@]}" \
    "${sampling_args[@]}" \
    "${family_args[@]}"

  echo "training_finished: $(date)"
} > "$train_log" 2>&1

final_step="$(latest_step "$RUN_DIR" || true)"
final_step="${final_step:-0}"
if [[ "$FAMILY" == "split_cfa" && "$target" == "3000000" && "$final_step" -ge 3000000 ]]; then
  prepare_split_historical_anchors "$RUN_DIR"
  echo "Submitting split_cfa stage 2 for seed $SEED"
  sbatch "$0" "$FAMILY" "$SEED"
fi
