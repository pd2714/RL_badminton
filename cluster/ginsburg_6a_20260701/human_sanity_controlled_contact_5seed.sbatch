#!/usr/bin/env bash
# Five-seed controlled-contact probe source data for human_sanity_comparison.
# Submit with:
#   sbatch --array=0-4 cluster/ginsburg_6a_20260701/human_sanity_controlled_contact_5seed.sbatch
# Optional:
#   SAMPLES=32 sbatch --array=0-4 cluster/ginsburg_6a_20260701/human_sanity_controlled_contact_5seed.sbatch
#   FORCE=1 sbatch --array=0-4 cluster/ginsburg_6a_20260701/human_sanity_controlled_contact_5seed.sbatch
#SBATCH --job-name=rlbdm_hsan_probe
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
OUT_DIR="$RUN_DIR/anchor_metric_eval/controlled_contact_grid_probe"
SAMPLES="${SAMPLES:-64}"
FORCE="${FORCE:-0}"
SAMPLES_CSV="$OUT_DIR/controlled_contact_grid_probe_samples.csv"

echo "Started human-sanity controlled-contact probe at $(date)"
echo "seed=$SEED"
echo "run_dir=$RUN_DIR"
echo "out_dir=$OUT_DIR"
echo "samples=$SAMPLES"

if [[ ! -f "$RUN_DIR/selfplay_config.json" ]]; then
  echo "Missing run config: $RUN_DIR/selfplay_config.json" >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/anchor_checkpoints/anchor_step_6000000.zip" ]]; then
  echo "Missing 6M anchor checkpoint for seed $SEED" >&2
  exit 2
fi
if [[ "$FORCE" != "1" && -s "$SAMPLES_CSV" ]]; then
  echo "Probe already exists: $SAMPLES_CSV"
  wc -l "$SAMPLES_CSV"
  exit 0
fi

python3 -u scripts/evaluate_controlled_lift_probe.py "$RUN_DIR" \
  --output-dir "$OUT_DIR" \
  --probe-preset contact-grid \
  --probe-name controlled_contact_grid \
  --samples "$SAMPLES" \
  --seed "$((20260529 + SEED))" \
  --anchor-step-min 0 \
  --anchor-step-max 6000000 \
  --anchor-step-interval 200000

echo "Finished human-sanity controlled-contact probe at $(date)"
wc -l "$SAMPLES_CSV"
