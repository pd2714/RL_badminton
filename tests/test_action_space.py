from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from badminton1d.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton1d.agents import SafeHitter
from badminton1d.config import CourtConfig, SimulationConfig
from badminton1d.evaluation import choose_model_action
from badminton1d.dynamics import validate_and_clip_shot_action, vy_bounds_for_hitter
from badminton1d.opponents import DecisionContext
from badminton1d.policy import apply_hitter_action_mask, apply_receiver_action_mask
from badminton1d.state import ShotAction, StageState


class DiscreteActionMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig()
        self.mapper = DiscreteActionMapper(
            self.config,
            DiscreteActionConfig(v_x_bins=5, v_y_bins=7, v_z_bins=4, x_rec_bins=5, y_rec_bins=5),
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
        vy_low, vy_high = vy_bounds_for_hitter("left", self.config)
        self.assertGreaterEqual(min(decoded_vy), vy_low)
        self.assertLessEqual(max(decoded_vy), vy_high)

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
            DiscreteActionConfig(v_x_bins=5, v_y_bins=7, v_z_bins=4, x_rec_bins=5, y_rec_bins=5),
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

    def test_receiver_decode_wraps_raw_policy_action_into_legal_range(self) -> None:
        config = SimulationConfig(court=CourtConfig(mode="1d"))
        mapper = DiscreteActionMapper(
            config,
            DiscreteActionConfig(v_x_bins=5, v_y_bins=7, v_z_bins=4, x_rec_bins=5, y_rec_bins=5),
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
        model.policy.obs_to_tensor.return_value = (torch.zeros((1, 29 + 50), dtype=torch.float32), None)
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

        observation = np.zeros(29 + 50, dtype=np.float32)
        observation[14] = 1.0
        observation[29 + 13] = 1.0
        observation[29 + 17] = 1.0
        observation[29 + 18] = 1.0
        observation[29 + 23] = 1.0

        chosen = choose_model_action(model, observation, context, deterministic=True)
        self.assertEqual(chosen, 18)

    def test_apply_receiver_action_mask_blocks_out_of_range_and_infeasible_logits(self) -> None:
        logits = torch.zeros((1, 60), dtype=torch.float32)
        logits[0, 55] = 12.0
        logits[0, 18] = 4.0
        logits[0, 22] = 6.0
        obs = torch.zeros((1, 29 + 50), dtype=torch.float32)
        obs[0, 14] = 1.0
        obs[0, 29 + 18] = 1.0
        obs[0, 29 + 22] = 1.0

        masked = apply_receiver_action_mask(logits, obs, receiver_action_count=50)

        self.assertLess(masked[0, 55].item(), -1e8)
        self.assertLess(masked[0, 17].item(), -1e8)
        self.assertEqual(masked[0, 18].item(), 4.0)
        self.assertEqual(masked[0, 22].item(), 6.0)

    def test_legal_serve_hitter_mask_filters_out_illegal_serve_targets(self) -> None:
        serve_mapper = DiscreteActionMapper(
            self.config,
            DiscreteActionConfig(v_x_bins=11, v_y_bins=15, v_z_bins=11, x_rec_bins=5, y_rec_bins=5),
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
            DiscreteActionConfig(v_x_bins=11, v_y_bins=15, v_z_bins=11, x_rec_bins=5, y_rec_bins=5),
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
        obs = torch.zeros((1, 29 + self.config.action.intercept_count), dtype=torch.float32)
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


if __name__ == "__main__":
    unittest.main()
