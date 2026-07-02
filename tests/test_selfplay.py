from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from badminton1d.selfplay import CheckpointPool, SelfPlayEvalCallback


class CheckpointPoolTests(unittest.TestCase):
    def test_newest_sampling_mode_always_picks_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            older = checkpoint_dir / "selfplay_step_1000.zip"
            newer = checkpoint_dir / "selfplay_step_2000.zip"
            older.write_bytes(b"older")
            newer.write_bytes(b"newer")

            pool = CheckpointPool(
                checkpoint_dir=checkpoint_dir,
                pool_size=2,
                sampling_mode="newest",
                seed=3,
            )

            sampled = {pool.sample_path().name for _ in range(5)}
            self.assertEqual(sampled, {"selfplay_step_2000.zip"})

    def test_linear_recency_sampling_prefers_newer_checkpoints_linearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            for step in (1000, 2000, 3000):
                (checkpoint_dir / f"anchor_step_{step}.zip").write_bytes(str(step).encode("ascii"))

            pool = CheckpointPool(
                checkpoint_dir=checkpoint_dir,
                pool_size=3,
                sampling_mode="linear_recency",
                seed=9,
            )

            counts = {path.name: 0 for path in pool.checkpoints}
            for _ in range(600):
                counts[pool.sample_path().name] += 1

            self.assertGreater(counts["anchor_step_3000.zip"], counts["anchor_step_2000.zip"])
            self.assertGreater(counts["anchor_step_2000.zip"], counts["anchor_step_1000.zip"])

    def test_recent_sampling_can_exclude_newest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            for step in (1000, 2000, 3000, 4000):
                (checkpoint_dir / f"selfplay_step_{step}.zip").write_bytes(str(step).encode("ascii"))

            pool = CheckpointPool(
                checkpoint_dir=checkpoint_dir,
                pool_size=4,
                sampling_mode="newest",
                recent_fraction=0.5,
                seed=3,
            )

            sampled = {pool.sample_recent_path(exclude_newest=True).name for _ in range(5)}
            self.assertEqual(sampled, {"selfplay_step_3000.zip"})

    def test_model_cache_can_be_limited_to_one_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            first = checkpoint_dir / "anchor_step_1000.zip"
            second = checkpoint_dir / "anchor_step_2000.zip"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            pool = CheckpointPool(
                checkpoint_dir=checkpoint_dir,
                pool_size=2,
                max_cached_models=1,
                seed=3,
            )
            second_resolved = second.resolve()

            with patch("badminton1d.selfplay.PPO.load", side_effect=[object(), object()]):
                pool.load_model(first)
                pool.load_model(second)

            self.assertLessEqual(len(pool.cached_models), 1)
            self.assertEqual(list(pool.cached_models), [second_resolved])

    def test_model_load_falls_back_for_flat_action_head_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            checkpoint = checkpoint_dir / "anchor_step_1000.zip"
            checkpoint.write_bytes(b"old")
            loaded_model = object()

            pool = CheckpointPool(
                checkpoint_dir=checkpoint_dir,
                pool_size=1,
                seed=3,
            )

            load_error = RuntimeError(
                "Error(s) in loading state_dict for MaskedBadmintonPolicy: "
                'Missing key(s) in state_dict: "phi_head.weight". '
                'Unexpected key(s) in state_dict: "action_net.weight".'
            )
            with (
                patch("badminton1d.selfplay.PPO.load", side_effect=load_error),
                patch("badminton1d.selfplay._load_ppo_with_compatible_policy_state", return_value=loaded_model),
            ):
                self.assertIs(pool.load_model(checkpoint), loaded_model)

            self.assertEqual(pool.cached_models[checkpoint.resolve()], loaded_model)

    def test_eval_schedule_aligns_resume_to_absolute_interval(self) -> None:
        self.assertEqual(
            SelfPlayEvalCallback._initial_last_eval_timestep_for_offset(
                timestep_offset=3_000_000,
                eval_freq=100_000,
            ),
            0,
        )
        self.assertEqual(
            SelfPlayEvalCallback._initial_last_eval_timestep_for_offset(
                timestep_offset=3_218_000,
                eval_freq=100_000,
            ),
            -18_000,
        )


if __name__ == "__main__":
    unittest.main()
