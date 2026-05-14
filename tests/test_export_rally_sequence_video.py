from __future__ import annotations

import unittest

import numpy as np

from scripts.export_rally_sequence_video import _resolve_initial_server, _resolve_next_server


class ExportRallySequenceVideoTests(unittest.TestCase):
    def test_random_initial_server_samples_both_sides(self) -> None:
        observed = {
            _resolve_initial_server("random", "left", np.random.default_rng(seed))
            for seed in range(12)
        }
        self.assertEqual(observed, {"left", "right"})

    def test_symbolic_initial_servers_resolve_relative_to_train_side(self) -> None:
        rng = np.random.default_rng(1)
        self.assertEqual(_resolve_initial_server("train", "right", rng), "right")
        self.assertEqual(_resolve_initial_server("opponent", "right", rng), "left")

    def test_next_server_can_follow_winner(self) -> None:
        server = _resolve_next_server(
            winner="right",
            current_server="left",
            train_side="left",
            random_server_each_rally=False,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(server, "right")

    def test_next_server_can_be_sampled_independently(self) -> None:
        observed = {
            _resolve_next_server(
                winner="right",
                current_server="right",
                train_side="left",
                random_server_each_rally=True,
                rng=np.random.default_rng(seed),
            )
            for seed in range(12)
        }
        self.assertEqual(observed, {"left", "right"})


if __name__ == "__main__":
    unittest.main()
