from __future__ import annotations

import math
import unittest

import numpy as np

from badminton1d.config import SimulationConfig
from badminton1d.dynamics import landing_position
from badminton1d.eval_evolution import summarize_match_trace_metrics
from badminton1d.evaluation import recovery_intended_landing_metrics, summarize_episodes
from badminton1d.playback import MatchTrace, RallyTrace, build_rally_trace
from badminton1d.state import ShotAction, StageRecord, StageState, ValidatedShotAction
from badminton1d.trajectory import ballistic_landing_time


def _record_for_landing(
    *,
    stage_index: int,
    landing_x: float,
    landing_y: float,
    recovery_x: float,
    recovery_y: float,
    config: SimulationConfig,
) -> StageRecord:
    state = StageState(
        x_left=0.0,
        y_left=-1.0,
        x_right=0.0,
        y_right=1.0,
        current_hitter="left",
        x0=0.0,
        y0=-1.0,
        z0=1.0,
        stage_index=stage_index,
    )
    v_z = 4.0
    flight_time = ballistic_landing_time(state.z0, v_z, config.action.gravity)
    action = ShotAction(
        v_x=(landing_x - state.x0) / flight_time,
        v_y=(landing_y - state.y0) / flight_time,
        v_z=v_z,
        x_rec=recovery_x,
        y_rec=recovery_y,
    )
    x_land, y_land = landing_position(state, action, config)
    next_state = StageState(
        x_left=recovery_x,
        y_left=recovery_y,
        x_right=x_land,
        y_right=y_land,
        current_hitter="right",
        x0=x_land,
        y0=y_land,
        z0=1.0,
        stage_index=stage_index + 1,
    )
    return StageRecord(
        stage_index=stage_index,
        state_before=state,
        validated_action=ValidatedShotAction(requested=action, applied=action, projected=False),
        receiver_side="right",
        candidate_times=np.asarray([0.2, 0.4], dtype=float),
        feasible_indices=[0],
        chosen_index=0,
        chosen_time=0.2,
        intercept_point=(x_land, y_land, 1.0),
        intended_intercept_time=0.2,
        intended_intercept_point=(x_land, y_land, 1.0),
        next_state=next_state,
        reward_left=0.0,
        reward_right=0.0,
    )


class EvaluationMetricsTests(unittest.TestCase):
    def test_recovery_intended_landing_metrics_are_reported_for_rollout_records(self) -> None:
        config = SimulationConfig()
        records = [
            _record_for_landing(stage_index=0, landing_x=0.0, landing_y=2.0, recovery_x=-1.0, recovery_y=-2.0, config=config),
            _record_for_landing(stage_index=1, landing_x=1.0, landing_y=3.0, recovery_x=0.0, recovery_y=-1.0, config=config),
            _record_for_landing(stage_index=2, landing_x=2.0, landing_y=4.0, recovery_x=1.0, recovery_y=0.0, config=config),
        ]
        result = {
            "reward": 0.0,
            "winner": "left",
            "rally_won": 1.0,
            "rally_length": len(records),
            "invalid_action_rate": 0.0,
            "truncated": False,
            "metrics": {},
            "records": records,
            "config": config,
        }

        metrics = recovery_intended_landing_metrics([result])
        summary = summarize_episodes([result])

        self.assertEqual(metrics["recovery_intended_landing_pair_count"], 3)
        self.assertAlmostEqual(metrics["recovery_intended_landing_x_corr"], 1.0)
        self.assertAlmostEqual(metrics["recovery_intended_landing_y_corr"], 1.0)
        self.assertTrue(math.isfinite(float(metrics["recovery_intended_landing_distance_mean"])))
        self.assertEqual(summary["recovery_intended_landing_pair_count"], 3)

    def test_recovery_intended_landing_metrics_are_reported_for_match_traces(self) -> None:
        config = SimulationConfig()
        records = [
            _record_for_landing(stage_index=0, landing_x=0.0, landing_y=2.0, recovery_x=-1.0, recovery_y=-2.0, config=config),
            _record_for_landing(stage_index=1, landing_x=1.0, landing_y=3.0, recovery_x=0.0, recovery_y=-1.0, config=config),
            _record_for_landing(stage_index=2, landing_x=2.0, landing_y=4.0, recovery_x=1.0, recovery_y=0.0, config=config),
        ]
        rally_trace = build_rally_trace(records, config)
        trace = MatchTrace(
            rallies=[RallyTrace(stages=rally_trace.stages, rally_done=False, winner=None, total_playback_time=1.0)],
            target_score=1,
            score_left=0,
            score_right=0,
            winner=None,
            total_playback_time=1.0,
        )

        metrics = summarize_match_trace_metrics(
            trace,
            config,
            speed_bins=(0.0, 10.0, 30.0),
            height_bins=(0.0, 1.5, 3.0),
        )

        self.assertEqual(metrics["recovery_intended_landing_pair_count"], 3)
        self.assertAlmostEqual(metrics["recovery_intended_landing_x_corr"], 1.0)
        self.assertAlmostEqual(metrics["recovery_intended_landing_y_corr"], 1.0)


if __name__ == "__main__":
    unittest.main()
