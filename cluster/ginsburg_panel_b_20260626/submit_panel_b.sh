#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p cluster/ginsburg_panel_b_20260626/logs
echo "Submitting 5 seed-matched pure-CFA panel-B trainings"
sbatch --array=0-4 cluster/ginsburg_panel_b_20260626/train_panel_b.sbatch purecfa
echo "Submitting 5 seed-matched no-CFA panel-B trainings"
sbatch --array=0-4 cluster/ginsburg_panel_b_20260626/train_panel_b.sbatch nocfa
echo "Submitting 5 fixed-pool panel-B evaluation watchers"
sbatch --array=0-4 cluster/ginsburg_panel_b_20260626/eval_panel_b.sbatch
