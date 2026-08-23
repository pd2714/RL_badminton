from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from badminton.config import CourtConfig, SimulationConfig
from badminton.render import _stage_title, draw_players, render_stage_image, setup_court_axes, stage_colors
from badminton.state import ShotAction, StageState
from badminton.dynamics import feasible_intercept_indices, step_stage
from badminton.trajectory import ballistic_landing_time


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()
        self.colors = stage_colors(monochrome=False)

    def test_draw_players_adds_contact_arrow_for_each_requested_contact_point(self) -> None:
        fig, ax = plt.subplots()
        try:
            draw_players(
                ax,
                self.config,
                self.colors,
                left_position=(-0.5, -2.5),
                right_position=(0.5, 2.5),
                show_player_labels=False,
                left_contact_xy=(0.0, -1.0),
                right_contact_xy=(0.2, 1.5),
            )

            arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
            self.assertEqual(len(arrows), 2)
        finally:
            plt.close(fig)

    def test_draw_players_skips_contact_arrow_when_contact_matches_player_position(self) -> None:
        fig, ax = plt.subplots()
        try:
            draw_players(
                ax,
                self.config,
                self.colors,
                left_position=(-0.5, -2.5),
                right_position=(0.5, 2.5),
                show_player_labels=False,
                left_contact_xy=(-0.5, -2.5),
            )

            arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
            self.assertEqual(len(arrows), 0)
        finally:
            plt.close(fig)

    def test_setup_court_axes_collapses_to_lane_in_one_dimensional_mode(self) -> None:
        config = SimulationConfig(court=CourtConfig(mode="1d"))
        fig, ax = plt.subplots()
        try:
            setup_court_axes(ax, config, self.colors, show_axes=True)
            x_min, x_max = ax.get_xlim()
            self.assertLess(x_max - x_min, config.court.width)
            self.assertEqual(ax.get_xlabel(), "1D lane")
        finally:
            plt.close(fig)

    def test_setup_court_axes_draws_center_service_lines_in_two_dimensional_mode(self) -> None:
        fig, ax = plt.subplots()
        try:
            setup_court_axes(ax, self.config, self.colors, show_axes=False)
            center_service_lines = [
                line
                for line in ax.lines
                if np.allclose(line.get_xdata(), [0.0, 0.0])
            ]

            self.assertEqual(len(center_service_lines), 2)
            y_spans = sorted(tuple(line.get_ydata()) for line in center_service_lines)
            self.assertIn((-self.config.court.half_length, -1.98), y_spans)
            self.assertIn((1.98, self.config.court.half_length), y_spans)
        finally:
            plt.close(fig)

    def test_setup_court_axes_uses_short_service_markers_in_one_dimensional_mode(self) -> None:
        config = SimulationConfig(court=CourtConfig(mode="1d"))
        fig, ax = plt.subplots()
        try:
            setup_court_axes(ax, config, self.colors, show_axes=False)

            net_line = next(
                line for line in ax.lines if np.allclose(line.get_ydata(), [config.court.net_y, config.court.net_y])
            )
            net_span = abs(float(net_line.get_xdata()[1] - net_line.get_xdata()[0]))
            service_lines = [
                line
                for line in ax.lines
                if np.allclose(np.diff(line.get_ydata()), 0.0) and not np.allclose(line.get_ydata(), [config.court.net_y, config.court.net_y])
            ]

            self.assertEqual(len(service_lines), 2)
            for line in service_lines:
                self.assertLess(abs(float(line.get_xdata()[1] - line.get_xdata()[0])), net_span)
                self.assertEqual(line.get_linestyle(), "-")
        finally:
            plt.close(fig)

    def test_render_stage_image_supports_rotated_3d_view_for_two_dimensional_court(self) -> None:
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
        action = ShotAction(v_x=0.9, v_y=5.8, v_z=5.0, x_rec=0.0, y_rec=-1.8)
        record = step_stage(state, action, feasible_intercept_indices(state, action, self.config)[0], self.config)

        image = render_stage_image(record, self.config, annotate=False, show_player_labels=False, monochrome=False)

        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)

    def test_stage_title_marks_invalid_serve_as_illegal(self) -> None:
        state = StageState(
            x_left=-0.875,
            y_left=-3.5,
            x_right=0.875,
            y_right=3.5,
            current_hitter="left",
            x0=-0.875,
            y0=-3.5,
            z0=1.15,
            stage_index=0,
        )
        v_z = 5.0
        flight_time = ballistic_landing_time(state.z0, v_z, self.config.action.gravity)
        action = ShotAction(
            v_x=(-0.5 - state.x0) / flight_time,
            v_y=(3.0 - state.y0) / flight_time,
            v_z=v_z,
            x_rec=state.x0,
            y_rec=state.y0,
        )
        record = step_stage(state, action, None, self.config)

        self.assertEqual(record.terminal_reason, "invalid_serve_target")
        self.assertIn("illegal serve", _stage_title(record, self.config))


if __name__ == "__main__":
    unittest.main()
