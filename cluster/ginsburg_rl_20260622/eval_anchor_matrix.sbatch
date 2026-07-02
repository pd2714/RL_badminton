#!/usr/bin/env bash
#SBATCH --job-name=rlbdm_anchor_matrix
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

FAMILIES=(split_cfa pure_cfa noncfa)
SEEDS=(17 23 31 47 59)
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as an array.}"
FAMILY_INDEX=$((TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))
FAMILY="${FAMILIES[$FAMILY_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

ROOT="outputs/rl/ginsburg_20260622"
RUN_DIR="$ROOT/${FAMILY}_seed${SEED}"
OUT_DIR="$RUN_DIR/anchor_pairwise_200r"
SNAPSHOT="$OUT_DIR/anchor_steps.txt"
POOL_DIR="$OUT_DIR/pool"
CONFIG_PATH="$RUN_DIR/selfplay_config.json"

if [[ ! -s "$SNAPSHOT" ]]; then
  echo "Missing frozen anchor snapshot: $SNAPSHOT" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" && "$FAMILY" == "pure_cfa" ]]; then
  # All three sweep families use the same environment and action-space settings.
  # Pure-CFA is still training and has not yet persisted its own config, so use
  # the completed split-CFA configuration for the same seed to evaluate its
  # already-written anchors.
  CONFIG_PATH="$ROOT/split_cfa_seed${SEED}/selfplay_config.json"
  echo "Pure-CFA run config unavailable; using compatible config: $CONFIG_PATH"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing evaluation configuration: $CONFIG_PATH" >&2
  exit 2
fi
if [[ -f "$OUT_DIR/win_rate_matrix.json" ]]; then
  echo "Completed matrix already exists: $OUT_DIR/win_rate_matrix.json"
  exit 0
fi

mapfile -t STEPS < "$SNAPSHOT"
if (( ${#STEPS[@]} < 2 )); then
  echo "Need at least two anchors; got ${#STEPS[@]} in $SNAPSHOT" >&2
  exit 2
fi

mkdir -p "$POOL_DIR"
for step in "${STEPS[@]}"; do
  checkpoint="$RUN_DIR/anchor_checkpoints/anchor_step_${step}.zip"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Frozen anchor is missing: $checkpoint" >&2
    exit 2
  fi
  agent_dir="$POOL_DIR/anchor_${step}"
  mkdir -p "$agent_dir"
  ln -sfn "$REPO/$checkpoint" "$agent_dir/final_model.zip"
  ln -sfn "$REPO/$CONFIG_PATH" "$agent_dir/selfplay_config.json"
done

python - "$POOL_DIR/manifest.json" "${STEPS[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
steps = [int(step) for step in sys.argv[2:]]
path.write_text(
    json.dumps(
        {"agents": [{"run_dir": f"anchor_{step}", "label": f"anchor_{step}", "step": step} for step in steps]},
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY

echo "Started: $(date)"
echo "Run: $RUN_DIR"
echo "Anchors: ${STEPS[*]}"
python scripts/rate_fixed_pool_elo.py "$POOL_DIR" \
  --output-dir "$OUT_DIR/elo_rating" \
  --eval-rallies-per-side 100 \
  --seed "$((2026062600 + TASK_ID))"

python - "$OUT_DIR/elo_rating/elo_rating_report.json" "$OUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))
labels = [str(agent["label"]) for agent in report["agents"]]
steps = [int(label.removeprefix("anchor_")) for label in labels]
values = {(label, label): 0.5 for label in labels}
rallies = {(label, label): 0 for label in labels}
for pair in report["pair_summaries"]:
    a = str(pair["agent_a"])
    b = str(pair["agent_b"])
    wr = float(pair["agent_a_win_rate"])
    episodes = int(pair["episodes"])
    values[(a, b)] = wr
    values[(b, a)] = 1.0 - wr
    rallies[(a, b)] = episodes
    rallies[(b, a)] = episodes

matrix = [[values[(a, b)] for b in labels] for a in labels]
rally_matrix = [[rallies[(a, b)] for b in labels] for a in labels]
payload = {
    "definition": "P_ij = Pr(anchor i beats anchor j), with each unordered matchup side-balanced over 200 rallies (100 per side).",
    "eval_rallies_per_pair": 200,
    "eval_rallies_per_side": 100,
    "row_labels": labels,
    "row_steps": steps,
    "col_labels": labels,
    "col_steps": steps,
    "win_rate_matrix": matrix,
    "rally_count_matrix": rally_matrix,
    "elo_rating_report": str(report_path),
}
(out_dir / "win_rate_matrix.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
with (out_dir / "win_rate_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["checkpoint", "step", *labels])
    for label, step, row in zip(labels, steps, matrix):
        writer.writerow([label, step, *row])
print(out_dir / "win_rate_matrix.json")
print(out_dir / "win_rate_matrix.csv")
PY

echo "Finished: $(date)"
