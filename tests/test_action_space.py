from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from badminton1d.action_space import (
    VELOCITY_ORIENTED_MIDRALLY_LEFT_COURT_PHI_TRIM_DEG,
    VELOCITY_ORIENTED_MIDRALLY_RIGHT_COURT_PHI_TRIM_DEG,
    VELOCITY_ORIENTED_PHI_CLUSTER_POWER,
    VELOCITY_ORIENTED_SERVICE_PHI_BOUNDARY_TRIM_DEG,
    VELOCITY_ORIENTED_THETA_HIGH_DEG,
    DiscreteActionConfig,
    DiscreteActionMapper,
)
from badminton1d.agents import SafeHitter
from badminton1d.config import ActionConfig, CourtConfig, SimulationConfig
from badminton1d.evaluation import choose_model_action
from badminton1d.dynamics import landing_position, validate_and_clip_shot_action
from badminton1d.opponents import DecisionContext
from badminton1d.policy import FEASIBLE_MASK_START_INDEX, apply_hitter_action_mask, apply_receiver_action_mask
from badminton1d.state import ShotAction, StageState
from badminton1d.utils import opponent_side, recovery_bounds, service_target_bounds_for_receiver_state


def _expected_clustered_angle_grid(lower: float, upper: float, center: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([float(np.clip(center, lower, upper))], dtype=float)
    center = float(np.clip(center, lower, upper))
    if np.isclose(center, lower) or np.isclose(center, upper):
        return np.linspace(lower, upper, count)
    interval_count = count - 1
    center_fraction = (center - lower) / (upper - lower)
    left_intervals = int(round(interval_count * center_fraction))
    left_intervals = int(np.clip(left_intervals, 1, interval_count - 1))
    right_intervals = interval_count - left_intervals
    left_span = center - lower
    right_span = upper - center
    left = center - left_span * (np.linspace(1.0, 0.0, left_intervals + 1) ** VELOCITY_ORIENTED_PHI_CLUSTER_POWER)
    right = center + right_span * (np.linspace(0.0, 1.0, right_intervals + 1) ** VELOCITY_ORIENTED_PHI_CLUSTER_POWER)
    return np.concatenate((left[:-1], right)).astype(float)


class DiscreteActionMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()
        self.mapper = DiscreteActionMapper(
            self.config,
            DiscreteActionConfig(phi_bins=5, theta_bins=7, speed_bins=4, x_rec_bins=5, y_rec_bins=5),
        )

    def test_stage_zero_hitter_velocities_point_to_opponent_side(self) -> None:
        state = StageState(
            x_left=0.0,
            y_left=-2.5,
            x_right=0.0,
            y_right=2.5,
            current_hitter="left",
            x0=0.0,
            y0=-2.5,
            z0=1.15,
            stage_index=1,
        )
        decoded_vy = [self.mapper.decode_hitter(action, state).shot_action.v_y for action in range(self.mapper.hitter_action_count)]
        self.assertTrue(np.all(np.isfinite(decoded_vy)))
        self.assertGreaterEqual(min(decoded_vy), -1e-9)

    def test_two_dimensional_defaults_use_phi_theta_speed_velocity(self) -> None:
        mapper = DiscreteActionMapper(self.config)
        state = StageState(
            x_left=0.0,
            y_left=-3.5,
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=-3.5,
            z0=1.15,
            stage_index=1,
        )

        discrete_config = DiscreteActionConfig()
        self.assertEqual(
            mapper.hitter_action_count,
            (
                discrete_config.phi_bins
                * discrete_config.theta_bins
                * discrete_config.speed_bins
                * discrete_config.x_rec_bins
                * discrete_config.y_rec_bins
            ),
        )
        phi_low, phi_high = sorted(
            (
                np.arctan2(self.config.court.net_y - state.y0, -self.config.court.half_width - state.x0),
                np.arctan2(self.config.court.net_y - state.y0, self.config.court.half_width - state.x0),
            )
        )
        lower_trim = np.deg2rad(VELOCITY_ORIENTED_MIDRALLY_RIGHT_COURT_PHI_TRIM_DEG)
        upper_trim = np.deg2rad(VELOCITY_ORIENTED_MIDRALLY_LEFT_COURT_PHI_TRIM_DEG)
        expected_phi = _expected_clustered_angle_grid(
            phi_low + lower_trim,
            phi_high - upper_trim,
            np.pi / 2.0,
            discrete_config.phi_bins,
        )
        self.assertTrue(np.allclose(mapper.phi_grid(state), expected_phi))

        x_bounds, y_bounds = recovery_bounds(state.current_hitter, self.config)
        self.assertTrue(np.allclose(mapper.x_rec_grid(state), np.linspace(*x_bounds, 7)[1:6]))
        self.assertTrue(np.allclose(mapper.y_rec_grid(state), np.linspace(*y_bounds, 7)[1:6]))

        for phi_index, phi in enumerate(mapper.phi_grid(state)):
            theta_grid = mapper.theta_grid(state, phi_index)
            horizontal_distance_to_net = abs((self.config.court.net_y - state.y0) / np.sin(phi))
            expected_theta_low = np.arctan2(
                self.config.court.net_height + self.config.action.net_clearance_margin - state.z0,
                horizontal_distance_to_net,
            )
            expected_theta_grid = expected_theta_low + (
                np.deg2rad(VELOCITY_ORIENTED_THETA_HIGH_DEG) - expected_theta_low
            ) * (np.linspace(0.0, 1.0, discrete_config.theta_bins + 2) ** 2.0)[2:]
            self.assertEqual(theta_grid.size, discrete_config.theta_bins)
            self.assertAlmostEqual(float(theta_grid[0]), float(expected_theta_grid[0]))
            self.assertAlmostEqual(float(theta_grid[-1]), np.deg2rad(VELOCITY_ORIENTED_THETA_HIGH_DEG))
            self.assertTrue(float(theta_grid[0]) > expected_theta_low)
            self.assertTrue(np.all(np.diff(np.diff(theta_grid)) >= -1e-12))

            for theta_index, theta in enumerate(theta_grid):
                valid_speed_range = mapper.valid_speed_range(state, phi_index, theta_index)
                speed_grid = mapper.speed_grid(state, phi_index, theta_index)
                self.assertEqual(speed_grid.size, discrete_config.speed_bins)
                if valid_speed_range is None:
                    continue
                lower, upper = valid_speed_range
                self.assertAlmostEqual(float(speed_grid[0]), lower)
                self.assertAlmostEqual(float(speed_grid[-1]), upper)
                self.assertGreaterEqual(lower, 0.0)
                self.assertLessEqual(upper, 100.0 + 1e-6)
                for candidate_speed in speed_grid:
                    self.assertTrue(mapper._speed_valid_for_phi_theta(state, phi, theta, float(candidate_speed)))

    def test_discrete_recovery_bins_are_conditioned_on_shot_context(self) -> None:
        discrete_config = DiscreteActionConfig(phi_bins=5, theta_bins=5, speed_bins=5, x_rec_bins=5, y_rec_bins=5)
        mapper = DiscreteActionMapper(self.config, discrete_config)
        state = StageState(
            x_left=0.0,
            y_left=-3.5,
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=-3.5,
            z0=1.15,
            stage_index=1,
        )

        def flat_action(speed_index: int) -> int:
            phi_index = discrete_config.phi_bins // 2
            theta_index = discrete_config.theta_bins // 2
            x_rec_index = discrete_config.x_rec_bins // 2
            y_rec_index = discrete_config.y_rec_bins // 2
            return (
                ((((phi_index * discrete_config.theta_bins + theta_index) * discrete_config.speed_bins + speed_index)
                  * discrete_config.x_rec_bins + x_rec_index)
                 * discrete_config.y_rec_bins)
                + y_rec_index
            )

        short = mapper.decode_hitter(flat_action(0), state).shot_action
        deep = mapper.decode_hitter(flat_action(discrete_config.speed_bins - 1), state).shot_action
        short_landing = landing_position(state, short, self.config)
        deep_landing = landing_position(state, deep, self.config)

        self.assertNotAlmostEqual(short_landing[1], deep_landing[1])
        self.assertNotAlmostEqual(short.y_rec, deep.y_rec)

    def test_angle_speed_ranges_stay_inside_total_speed_bounds(self) -> None:
        discrete_config = DiscreteActionConfig(phi_bins=5, theta_bins=7, speed_bins=5, x_rec_bins=5, y_rec_bins=5)
        mapper = DiscreteActionMapper(self.config, discrete_config)
        state = StageState(
            x_left=0.0,
            y_left=-3.5,
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=-3.5,
            z0=1.15,
            stage_index=1,
        )

        for phi_index in range(discrete_config.phi_bins):
            for theta_index in range(discrete_config.theta_bins):
                speed_range = mapper.valid_speed_range(state, phi_index, theta_index)
                if speed_range is None:
                    continue
                lower, upper = speed_range
                self.assertGreaterEqual(lower, 0.0)
                self.assertLessEqual(upper, 100.0 + 1e-6)
                self.assertLessEqual(lower, upper)

    def test_drag_angle_speed_range_endpoints_are_valid(self) -> None:
        config = SimulationConfig(action=ActionConfig(trajectory_mode="drag_square"))
        discrete_config = DiscreteActionConfig(phi_bins=5, theta_bins=7, speed_bins=5, x_rec_bins=5, y_rec_bins=5)
        mapper = DiscreteActionMapper(config, discrete_config)
        state = StageState(
            x_left=0.0,
            y_left=-3.5,
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=-3.5,
            z0=1.15,
            stage_index=1,
        )

        valid_ranges = 0
        for phi_index in range(discrete_config.phi_bins):
            phi = float(mapper.phi_grid(state)[phi_index])
            for theta_index in range(discrete_config.theta_bins):
                theta = float(mapper.theta_grid(state, phi_index)[theta_index])
                speed_range = mapper.valid_speed_range(state, phi_index, theta_index)
                if speed_range is None:
                    continue
                valid_ranges += 1
                lower, upper = speed_range
                self.assertTrue(mapper._speed_valid_for_phi_theta(state, phi, theta, lower))
                self.assertTrue(mapper._speed_valid_for_phi_theta(state, phi, theta, upper))
                for speed in mapper.speed_grid(state, phi_index, theta_index):
                    self.assertTrue(mapper._speed_valid_for_phi_theta(state, phi, theta, float(speed)))

        self.assertGreater(valid_ranges, 0)

    def test_fast_drag_speed_status_matches_full_drag_landing_status(self) -> None:
        config = SimulationConfig(action=ActionConfig(trajectory_mode="drag_square"))
        discrete_config = DiscreteActionConfig(phi_bins=5, theta_bins=7, speed_bins=5, x_rec_bins=5, y_rec_bins=5)
        mapper = DiscreteActionMapper(config, discrete_config)
        state = StageState(
            x_left=0.0,
            y_left=-3.5,
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=-3.5,
            z0=1.15,
            stage_index=1,
        )

        for phi_index in range(discrete_config.phi_bins):
            phi = float(mapper.phi_grid(state)[phi_index])
            for theta_index in range(discrete_config.theta_bins):
                theta = float(mapper.theta_grid(state, phi_index)[theta_index])
                context = mapper._speed_validation_context(state, phi, theta)
                target_interval = mapper._target_ray_progress_interval(context)
                if target_interval is None:
                    continue
                target_low, target_high = target_interval
                for speed in np.linspace(0.0, 100.0, 9):
                    action = mapper._shot_action_from_speed_context(context, float(speed))
                    horizontal_speed = float(np.hypot(action.v_x, action.v_y))
                    fast_status = -1
                    if horizontal_speed > 1e-9:
                        fast_status = mapper._drag_speed_status(context, float(speed), target_low, target_high)

                    metrics = mapper._fast_drag_landing_metrics(state, action)
                    if metrics is None or metrics.net_z is None or metrics.net_z < context.required_net_z:
                        full_status = -1
                    else:
                        progress = (
                            (metrics.landing_x - state.x0) * action.v_x
                            + (metrics.landing_y - state.y0) * action.v_y
                        ) / horizontal_speed
                        full_status = -1 if progress < target_low else (1 if progress > target_high else 0)
                    self.assertEqual(fast_status, full_status)

    def test_serve_phi_grid_uses_trimmed_service_target_half_span(self) -> None:
        mapper = DiscreteActionMapper(self.config)
        state = StageState(
            x_left=-self.config.court.half_width / 2.0,
            y_left=-3.5,
            x_right=self.config.court.half_width / 2.0,
            y_right=3.5,
            current_hitter="left",
            x0=-self.config.court.half_width / 2.0,
            y0=-3.5,
            z0=1.15,
            stage_index=0,
        )
        (x_low, x_high), (y_low, y_high) = service_target_bounds_for_receiver_state(state, "right", self.config)
        corner_angles = [
            np.arctan2(corner_y - state.y0, corner_x - state.x0)
            for corner_x in (x_low, x_high)
            for corner_y in (y_low, y_high)
        ]
        phi_low, phi_high = min(corner_angles), max(corner_angles)
        trim = np.deg2rad(VELOCITY_ORIENTED_SERVICE_PHI_BOUNDARY_TRIM_DEG)
        expected_phi = _expected_clustered_angle_grid(
            phi_low + trim,
            phi_high - trim,
            np.pi / 2.0,
            DiscreteActionConfig().phi_bins,
        )

        phi_grid = mapper.phi_grid(state)

        self.assertTrue(np.allclose(phi_grid, expected_phi))

    def test_safe_hitter_serves_to_receiver_physical_x_half(self) -> None:
        state = StageState(
            x_left=self.config.court.half_width / 2.0,
            y_left=-3.5,
            x_right=-self.config.court.half_width / 2.0,
            y_right=3.5,
            current_hitter="left",
            x0=self.config.court.half_width / 2.0,
            y0=-3.5,
            z0=1.15,
            stage_index=0,
        )
        action = SafeHitter(stochastic=False).choose_action(state, self.config)
        landing_x, landing_y = landing_position(state, action, self.config)
        receiver = opponent_side(state.current_hitter)
        (x_low, x_high), (y_low, y_high) = service_target_bounds_for_receiver_state(state, receiver, self.config)

        self.assertGreaterEqual(landing_x, x_low - 1e-9)
        self.assertLessEqual(landing_x, x_high + 1e-9)
        self.assertGreaterEqual(landing_y, y_low - 1e-9)
        self.assertLessEqual(landing_y, y_high + 1e-9)

    def test_projection_clips_illegal_velocity_to_valid_range(self) -> None:
        state = StageState(
            x_left=0.0,
            y_left=-2.5,
            x_right=0.0,
            y_right=2.5,
            current_hitter="left",
            x0=0.0,
            y0=-2.5,
            z0=1.15,
            stage_index=1,
        )
        valid_action = SafeHitter().choose_action(state, self.config)

        projected = self.mapper.project_hitter_action(
            state,
            ShotAction(
                v_x=valid_action.v_x,
                v_y=valid_action.v_y,
                v_z=valid_action.v_z,
                x_rec=valid_action.x_rec + 10.0,
                y_rec=valid_action.y_rec - 10.0,
            ),
        )

        self.assertTrue(projected.projected)
        self.assertLessEqual(projected.shot_action.v_x, self.config.action.vx_max)
        self.assertLessEqual(projected.shot_action.v_z, self.config.action.vz_max)
        validate_and_clip_shot_action(state, projected.shot_action, self.config)

    def test_one_dimensional_mode_collapses_lateral_action_bins(self) -> None:
        config = SimulationConfig(court=CourtConfig(mode="1d"))
        mapper = DiscreteActionMapper(
            config,
            DiscreteActionConfig(phi_bins=5, theta_bins=7, speed_bins=4, x_rec_bins=5, y_rec_bins=5),
        )
        state = StageState(
            x_left=0.0,
            y_left=-2.5,
            x_right=0.0,
            y_right=2.5,
            current_hitter="left",
            x0=0.0,
            y0=-2.5,
            z0=1.15,
            stage_index=0,
        )

        self.assertEqual(mapper.hitter_action_count, 7 * 4 * 5)
        decoded = mapper.decode_hitter(mapper.hitter_action_count - 1, state).shot_action
        self.assertAlmostEqual(decoded.v_x, 0.0)
        self.assertAlmostEqual(decoded.x_rec, 0.0)

    def test_agent_canonical_decode_mirrors_actions_for_right_hitter(self) -> None:
        state = StageState(
            x_left=-0.3,
            y_left=-2.8,
            x_right=0.5,
            y_right=3.1,
            current_hitter="left",
            x0=-0.3,
            y0=-2.8,
            z0=1.15,
            stage_index=1,
        )
        mirrored_state = StageState(
            x_left=state.x_right,
            y_left=-state.y_right,
            x_right=state.x_left,
            y_right=-state.y_left,
            current_hitter="right",
            x0=state.x0,
            y0=-state.y0,
            z0=state.z0,
            stage_index=state.stage_index,
        )

        for action in (0, 17, 100, self.mapper.hitter_action_count - 1):
            left = self.mapper.decode_hitter_for_agent(action, state, "left").shot_action
            right = self.mapper.decode_hitter_for_agent(action, mirrored_state, "right").shot_action
            self.assertAlmostEqual(left.v_x, right.v_x)
            self.assertAlmostEqual(left.v_y, -right.v_y)
            self.assertAlmostEqual(left.v_z, right.v_z)
            self.assertAlmostEqual(left.x_rec, right.x_rec)
            self.assertAlmostEqual(left.y_rec, -right.y_rec)

    def test_receiver_decode_wraps_raw_policy_action_into_legal_range(self) -> None:
        config = SimulationConfig(court=CourtConfig(mode="1d"))
        mapper = DiscreteActionMapper(
            config,
            DiscreteActionConfig(phi_bins=5, theta_bins=7, speed_bins=4, x_rec_bins=5, y_rec_bins=5),
        )

        self.assertEqual(mapper.receiver_action_count, config.action.intercept_count)
        self.assertEqual(mapper.decode_receiver(18), 18)
        self.assertEqual(mapper.decode_receiver(318), 318 % config.action.intercept_count)

    def test_choose_model_action_restricts_receiver_to_feasible_indices(self) -> None:
        model = Mock()
        logits = torch.full((1, 825), -1000.0)
        logits[0, 318] = 10.0
        logits[0, 17] = 1.0
        logits[0, 18] = 3.0
        logits[0, 23] = 2.0
        model.policy.obs_to_tensor.return_value = (torch.zeros((1, FEASIBLE_MASK_START_INDEX + 50), dtype=torch.float32), None)
        model.policy.get_distribution.return_value = Mock(distribution=Mock(logits=logits))

        state = StageState(
            x_left=0.0,
            y_left=-2.5,
            x_right=0.0,
            y_right=2.5,
            current_hitter="right",
            x0=0.0,
            y0=2.5,
            z0=1.0,
            stage_index=1,
        )
        context = DecisionContext(
            state=state,
            role="receiver",
            pending_action=None,
            feasible_indices=[13, 17, 18, 23],
            receiver_action_count=50,
        )

        observation = np.zeros(FEASIBLE_MASK_START_INDEX + 50, dtype=np.float32)
        observation[14] = 1.0
        observation[FEASIBLE_MASK_START_INDEX + 13] = 1.0
        observation[FEASIBLE_MASK_START_INDEX + 17] = 1.0
        observation[FEASIBLE_MASK_START_INDEX + 18] = 1.0
        observation[FEASIBLE_MASK_START_INDEX + 23] = 1.0

        chosen = choose_model_action(model, observation, context, deterministic=True)
        self.assertEqual(chosen, 18)

    def test_apply_receiver_action_mask_blocks_out_of_range_and_infeasible_logits(self) -> None:
        logits = torch.zeros((1, 60), dtype=torch.float32)
        logits[0, 55] = 12.0
        logits[0, 18] = 4.0
        logits[0, 22] = 6.0
        obs = torch.zeros((1, FEASIBLE_MASK_START_INDEX + 50), dtype=torch.float32)
        obs[0, 14] = 1.0
        obs[0, FEASIBLE_MASK_START_INDEX + 18] = 1.0
        obs[0, FEASIBLE_MASK_START_INDEX + 22] = 1.0

        masked = apply_receiver_action_mask(logits, obs, receiver_action_count=50)

        self.assertLess(masked[0, 55].item(), -1e8)
        self.assertLess(masked[0, 17].item(), -1e8)
        self.assertEqual(masked[0, 18].item(), 4.0)
        self.assertEqual(masked[0, 22].item(), 6.0)

    def test_legal_serve_hitter_mask_filters_out_illegal_serve_targets(self) -> None:
        serve_mapper = DiscreteActionMapper(
            self.config,
            DiscreteActionConfig(phi_bins=11, theta_bins=15, speed_bins=11, x_rec_bins=5, y_rec_bins=5),
        )
        state = StageState(
            x_left=-self.config.court.half_width / 2.0,
            y_left=-3.5,
            x_right=self.config.court.half_width / 2.0,
            y_right=3.5,
            current_hitter="left",
            x0=-self.config.court.half_width / 2.0,
            y0=-3.5,
            z0=1.15,
            stage_index=0,
        )

        legal_mask = serve_mapper.legal_serve_hitter_mask(state)

        self.assertTrue(legal_mask.any())
        self.assertFalse(legal_mask.all())

    def test_apply_hitter_action_mask_blocks_illegal_stage_zero_serves(self) -> None:
        serve_mapper = DiscreteActionMapper(
            self.config,
            DiscreteActionConfig(phi_bins=11, theta_bins=15, speed_bins=11, x_rec_bins=5, y_rec_bins=5),
        )
        state = StageState(
            x_left=-self.config.court.half_width / 2.0,
            y_left=-3.5,
            x_right=self.config.court.half_width / 2.0,
            y_right=3.5,
            current_hitter="left",
            x0=-self.config.court.half_width / 2.0,
            y0=-3.5,
            z0=1.15,
            stage_index=0,
        )
        obs = torch.zeros((1, FEASIBLE_MASK_START_INDEX + self.config.action.intercept_count), dtype=torch.float32)
        obs[0, 0] = state.x_left / self.config.court.half_width
        obs[0, 1] = state.y_left / self.config.court.half_length
        obs[0, 2] = state.x_right / self.config.court.half_width
        obs[0, 3] = state.y_right / self.config.court.half_length
        obs[0, 4] = state.x0 / self.config.court.half_width
        obs[0, 5] = state.y0 / self.config.court.half_length
        obs[0, 6] = state.z0 / self.config.render.z_max
        obs[0, 7] = 1.0
        obs[0, 13] = 1.0
        obs[0, 17] = 0.0

        logits = torch.zeros((1, serve_mapper.action_count), dtype=torch.float32)
        legal_mask = serve_mapper.legal_serve_hitter_mask(state)
        illegal_index = int(np.flatnonzero(~legal_mask)[0])
        legal_index = int(np.flatnonzero(legal_mask)[0])
        logits[0, illegal_index] = 5.0
        logits[0, legal_index] = 4.0

        masked = apply_hitter_action_mask(logits, obs, mapper=serve_mapper)

        self.assertLess(masked[0, illegal_index].item(), -1e8)
        self.assertEqual(masked[0, legal_index].item(), 4.0)

    def test_apply_hitter_action_mask_can_block_illegal_mid_rally_shots(self) -> None:
        mapper = DiscreteActionMapper(
            self.config,
            DiscreteActionConfig(phi_bins=3, theta_bins=5, speed_bins=11, x_rec_bins=1, y_rec_bins=1),
        )
        state = StageState(
            x_left=0.0,
            y_left=-2.0,
            x_right=0.0,
            y_right=3.5,
            current_hitter="left",
            x0=0.0,
            y0=-2.0,
            z0=1.6,
            stage_index=1,
        )
        legal_mask = mapper.legal_hitter_mask(state)
        self.assertTrue(legal_mask.any())
        self.assertFalse(legal_mask.all())
        illegal_index = int(np.flatnonzero(~legal_mask)[0])
        legal_index = int(np.flatnonzero(legal_mask)[0])

        obs = torch.zeros((1, FEASIBLE_MASK_START_INDEX + self.config.action.intercept_count), dtype=torch.float32)
        obs[0, 0] = state.x_left / self.config.court.half_width
        obs[0, 1] = state.y_left / self.config.court.half_length
        obs[0, 2] = state.x_right / self.config.court.half_width
        obs[0, 3] = state.y_right / self.config.court.half_length
        obs[0, 4] = state.x0 / self.config.court.half_width
        obs[0, 5] = state.y0 / self.config.court.half_length
        obs[0, 6] = state.z0 / self.config.render.z_max
        obs[0, 7] = 1.0
        obs[0, 13] = 1.0
        obs[0, 17] = 1.0 / 30.0

        logits = torch.zeros((1, mapper.action_count), dtype=torch.float32)
        logits[0, illegal_index] = 5.0
        logits[0, legal_index] = 4.0

        masked = apply_hitter_action_mask(logits, obs, mapper=mapper, mask_mid_rally=True)

        self.assertLess(masked[0, illegal_index].item(), -1e8)
        self.assertEqual(masked[0, legal_index].item(), 4.0)


if __name__ == "__main__":
    unittest.main()
