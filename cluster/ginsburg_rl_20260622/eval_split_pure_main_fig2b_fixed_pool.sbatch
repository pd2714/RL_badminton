#!/usr/bin/env bash
# Evaluate same-seed split-CFA vs pure-CFA against the full main Fig2B matrix pool.
# Existing pairs in pair_results.jsonl are reused and not recomputed.
# Submit with: sbatch --array=0-4 cluster/ginsburg_rl_20260622/eval_split_pure_main_fig2b_fixed_pool.sbatch
#SBATCH --job-name=rlbdm_sp_2b
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
OUT_DIR="$ROOT/same_main_fig2b_fixed_pool_eval_200r/seed_${SEED}/split_vs_pure_cfa"
BASE_REPORT="outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611/anchor_metric_eval/cached_win_rate_matrix_200r/fixed_pool_eval_report.json"
ZERO_MODEL="outputs/rl/final_selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611/anchor_checkpoints/anchor_step_0.zip"

mapfile -t STEPS < <(
  python - "$BASE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
for step in sorted({int(row["step"]) for row in report["agents"]}):
    print(step)
PY
)

MISSING_COUNT="$(
  python - "$OUT_DIR/pair_results.jsonl" "$BASE_REPORT" "$ROOT/split_cfa_seed${SEED}" "$ROOT/pure_cfa_seed${SEED}" "$ZERO_MODEL" "${STEPS[@]}" <<'PY'
import json
import sys
from pathlib import Path

partial_path = Path(sys.argv[1])
base_report = json.loads(Path(sys.argv[2]).read_text())
runs = [("split_cfa", Path(sys.argv[3])), ("pure_cfa", Path(sys.argv[4]))]
zero_model = Path(sys.argv[5])
steps = [int(step) for step in sys.argv[6:]]

def resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    for old, new in [
        (
            "/selfplay_2d_recoverycfdefault_resp1_3m_",
            "/final_selfplay_2d_recoverycfdefault_resp1_3m_",
        ),
    ]:
        candidate = Path(text.replace(old, new))
        if candidate.exists():
            return candidate
    return path

completed = set()
if partial_path.exists():
    for line in partial_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add((str(row["agent"]), str(row["opponent"])))

if "opponent_pool" in base_report:
    opponents = [
        str(row["label"])
        for row in base_report["opponent_pool"]
        if resolve_existing_path(Path(row["model_path"])).exists()
    ]
else:
    opponents = [
        str(row["label"])
        for row in base_report["agents"]
        if resolve_existing_path(Path(row["model_path"])).exists()
    ]

available_pairs = set()
for family, run_dir in runs:
    seed = run_dir.name.rsplit("seed", 1)[1]
    run_label = f"{family}_seed{seed}"
    for step in steps:
        if step == 0:
            available = zero_model.exists()
        else:
            available = (run_dir / "anchor_checkpoints" / f"anchor_step_{step}.zip").exists()
        if not available:
            continue
        label = f"{run_label}_{step}"
        for opponent in opponents:
            available_pairs.add((label, opponent))

print(len(available_pairs - completed))
PY
)"

echo "missing_pairs_for_main_fig2b_pool: $MISSING_COUNT"
if [[ "$MISSING_COUNT" == "0" ]]; then
  echo "already complete for available selected steps: $OUT_DIR"
  exit 0
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
  --steps "${STEPS[@]}" \
  --zero-model "$ZERO_MODEL" \
  --eval-rallies 200 \
  --seed "$((2026062200 + SEED))"
