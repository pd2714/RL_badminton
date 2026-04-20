from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import feasible_intercept_indices, step_stage
from badminton1d.playback import build_rally_trace
from badminton1d.state import ShotAction, StageState
from badminton1d.video import (
    TrainingProgressSample,
    _draw_service_markers_side_view,
    _resolve_view,
    export_training_progress_video,
)


class TrainingProgressVideoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()

    def test_export_training_progress_video_writes_manifest_and_media(self) -> None:
        state = StageState(
            x_left=-0.5,
            y_left=-2.5,
            x_right=0.5,
            y_right=2.5,
            current_hitter="left",
            x0=-0.5,
            y0=-2.5,
            z0=1.7,
            stage_index=1,
        )

        action_a = ShotAction(v_x=0.9, v_y=5.8, v_z=5.0, x_rec=0.0, y_rec=-1.8)
        intercept_a = feasible_intercept_indices(state, action_a, self.config)[0]
        record_a = step_stage(state, action_a, intercept_a, self.config)

        slow_receiver_config = SimulationConfig(player=type(self.config.player)(v_max=1.0))
        action_b = ShotAction(v_x=1.0, v_y=5.4, v_z=5.0, x_rec=0.0, y_rec=-2.0)
        record_b = step_stage(state, action_b, intercept_index=None, config=slow_receiver_config)

        sample_a = TrainingProgressSample(
            step=0,
            trace=build_rally_trace([record_a], self.config),
            opponent_label="bootstrap",
            rally_won=False,
            invalid_action_rate=0.25,
        )
        sample_b = TrainingProgressSample(
            step=10_000,
            trace=build_rally_trace([record_b], self.config),
            opponent_label="selfplay_step_10000",
            rally_won=True,
            invalid_action_rate=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "progress_video"
            result = export_training_progress_video(
                [sample_a, sample_b],
                self.config,
                output_dir,
                fps=6,
                stage_pause=0.0,
                rally_pause=0.0,
            )

            self.assertTrue(result.gif_path.exists())
            self.assertTrue(result.trace_path.exists())
            self.assertGreater(len(result.frame_paths), 0)

            payload = json.loads(result.trace_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["sample_count"], 2)
            self.assertEqual(payload["samples"][0]["step"], 0)
            self.assertEqual(payload["samples"][1]["step"], 10_000)

    def test_auto_view_uses_side_view_for_one_dimensional_court(self) -> None:
        config = SimulationConfig(court=type(self.config.court)(mode="1d"))
        self.assertEqual(_resolve_view(config, "auto"), "side")

    def test_auto_view_uses_rotated_3d_view_for_two_dimensional_court(self) -> None:
        self.assertEqual(_resolve_view(self.config, "auto"), "3d")

    def test_one_dimensional_side_view_uses_short_service_markers(self) -> None:
        config = SimulationConfig(court=type(self.config.court)(mode="1d"))
        fig, ax = plt.subplots()
        try:
            _draw_service_markers_side_view(ax, config, {"service_line": "black"})
            service_lines = [line for line in ax.lines if np.allclose(np.diff(line.get_xdata()), 0.0)]

            self.assertEqual(len(service_lines), 2)
            for line in service_lines:
                self.assertEqual(line.get_linestyle(), "-")
                self.assertGreater(float(line.get_ydata()[1]), 0.0)
                self.assertLess(float(line.get_ydata()[1]), config.court.net_height)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
