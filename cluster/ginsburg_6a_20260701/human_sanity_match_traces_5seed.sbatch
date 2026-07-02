#!/usr/bin/env bash
# Five-seed trace-only checkpoint match rollouts for human_sanity_comparison panels A/B.
# Submit with:
#   sbatch --array=0-4 cluster/ginsburg_6a_20260701/human_sanity_match_traces_5seed.sbatch
# Optional:
#   TARGET_SCORE=7 sbatch --array=0-4 cluster/ginsburg_6a_20260701/human_sanity_match_traces_5seed.sbatch
#   DETERMINISTIC=1 sbatch --array=0-4 cluster/ginsburg_6a_20260701/human_sanity_match_traces_5seed.sbatch
#SBATCH --job-name=rlbdm_hsan_trace
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$MPLCONFIGDIR" cluster/ginsburg_6a_20260701/logs

SEEDS=(17 23 31 47 59)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]}"

ROOT="outputs/rl/ginsburg_20260622"
RUN_DIR="$ROOT/pure_cfa_seed${SEED}"
OUT_DIR="$RUN_DIR/videos/human_sanity_match_traces"
TARGET_SCORE="${TARGET_SCORE:-5}"
DETERMINISTIC="${DETERMINISTIC:-0}"

echo "Started human-sanity trace rollouts at $(date)"
echo "seed=$SEED"
echo "run_dir=$RUN_DIR"
echo "out_dir=$OUT_DIR"
echo "target_score=$TARGET_SCORE"
echo "deterministic=$DETERMINISTIC"

if [[ ! -f "$RUN_DIR/selfplay_config.json" ]]; then
  echo "Missing run config: $RUN_DIR/selfplay_config.json" >&2
  exit 2
fi
for step in 3000000 4000000 5000000 6000000; do
  if [[ ! -f "$RUN_DIR/anchor_checkpoints/anchor_step_${step}.zip" ]]; then
    echo "Missing anchor checkpoint: $RUN_DIR/anchor_checkpoints/anchor_step_${step}.zip" >&2
    exit 2
  fi
done

deterministic_flag=("--no-deterministic")
if [[ "$DETERMINISTIC" == "1" ]]; then
  deterministic_flag=("--deterministic")
fi

python3 -u scripts/export_human_sanity_match_traces.py "$RUN_DIR" \
  --output-dir "$OUT_DIR" \
  --target-score "$TARGET_SCORE" \
  --seed "$((2026070200 + SEED))" \
  "${deterministic_flag[@]}"

echo "Finished human-sanity trace rollouts at $(date)"
find "$OUT_DIR" -name match_trace.json -maxdepth 2 -print
