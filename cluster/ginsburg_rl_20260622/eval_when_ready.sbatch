#!/usr/bin/env bash
#SBATCH --job-name=rlbdm_eval
#SBATCH --account=rqlab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@600
#SBATCH --array=0-4
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$MPLCONFIGDIR"

SEEDS=(17 23 31 47 59)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
ROOT="outputs/rl/ginsburg_20260622"
LOG_DIR="$ROOT/eval/seed_${SEED}/logs"
mkdir -p "$LOG_DIR"

resubmit_self() {
  local array_arg=()
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    array_arg=(--array="${SLURM_ARRAY_TASK_ID}")
  fi
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sbatch "${array_arg[@]}" --dependency="afterany:${SLURM_JOB_ID}" "$0"
  else
    sbatch "${array_arg[@]}" "$0"
  fi
}

on_time_signal() {
  echo "Received pre-timeout signal at $(date); resubmitting eval watcher for seed $SEED"
  resubmit_self
}
trap on_time_signal USR1

latest_step() {
  local run_dir="$1"
  find "$run_dir/checkpoint_pool" -maxdepth 1 -type f -name 'selfplay_step_*.zip' 2>/dev/null \
    | sed -E 's/.*selfplay_step_([0-9]+)\.zip/\1/' \
    | sort -n \
    | tail -n 1
}

ready=1
for item in "split_cfa:6000000" "pure_cfa:6000000" "noncfa:3000000"; do
  family="${item%%:*}"
  target="${item##*:}"
  run_dir="$ROOT/${family}_seed${SEED}"
  step="$(latest_step "$run_dir" || true)"
  step="${step:-0}"
  echo "readiness $family seed $SEED: step=$step target=$target"
  if (( step < target )); then
    ready=0
  fi
done

if (( ready == 0 )); then
  echo "Training not complete for seed $SEED; resubmitting eval watcher for later."
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    sbatch --array="${SLURM_ARRAY_TASK_ID}" --begin=now+2hours "$0"
  else
    sbatch --begin=now+2hours "$0"
  fi
  exit 0
fi

echo "Evaluation started for seed $SEED at $(date)"
python cluster/ginsburg_rl_20260622/make_eval_pools.py --repo "$REPO" --seed "$SEED" --step-interval 200000

declare -a POOLS=(
  "cfa_splitlinear_0_to_6m_pool"
  "cfa_purerecency_0_to_6m_pool"
  "noncfa_ablation_0_to_3m_pool"
  "combined_cfa_vs_noncfa_0_to_3m_pool"
)

for pool_name in "${POOLS[@]}"; do
  pool_dir="$ROOT/eval/seed_${SEED}/${pool_name}"
  out_dir="$ROOT/eval/seed_${SEED}/${pool_name}_elo_200r"
  if [[ -f "$out_dir/elo_rating_report.json" ]]; then
    echo "Skipping completed rating: $out_dir"
    continue
  fi
  echo "Rating $pool_dir -> $out_dir"
  python scripts/rate_fixed_pool_elo.py "$pool_dir" \
    --output-dir "$out_dir" \
    --eval-rallies-per-side 100 \
    --seed "$((2026062200 + SEED))"
done

python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("outputs/rl/ginsburg_20260622/eval")
rows = []
for report_path in sorted(root.glob("seed_*/*_elo_200r/elo_rating_report.json")):
    report = json.loads(report_path.read_text())
    seed = report_path.parts[-3].replace("seed_", "")
    pool = report_path.parts[-2].replace("_elo_200r", "")
    for item in report["standings"]:
        agent = str(item["agent"])
        rows.append(
            {
                "seed": seed,
                "pool": pool,
                "rank": item["rank"],
                "agent": agent,
                "elo": item["elo"],
            }
        )

out = root / "elo_summary_all_available.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["seed", "pool", "rank", "agent", "elo"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

echo "Evaluation finished for seed $SEED at $(date)"
