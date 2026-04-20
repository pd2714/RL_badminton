from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from badminton1d.selfplay import CheckpointPool


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


if __name__ == "__main__":
    unittest.main()
