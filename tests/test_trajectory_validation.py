from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_shuttle_trajectories.py"


def load_validation_module():
    spec = importlib.util.spec_from_file_location("validate_shuttle_trajectories", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_baseline_trajectory_validation_ranges() -> None:
    module = load_validation_module()

    rows = module.baseline_metric_rows()

    assert rows
    assert all(row.passes for row in rows)


def test_robustness_rows_are_ordered_by_drag_scale() -> None:
    module = load_validation_module()

    rows = module.robustness_rows(scales=(0.9, 1.0, 1.1))

    assert [row["drag_scale"] for row in rows] == [0.9, 1.0, 1.1]
    assert rows[0]["speed_halving_distance_m"] > rows[1]["speed_halving_distance_m"] > rows[2]["speed_halving_distance_m"]
    assert rows[0]["vertical_terminal_speed_mps"] > rows[1]["vertical_terminal_speed_mps"] > rows[2]["vertical_terminal_speed_mps"]
