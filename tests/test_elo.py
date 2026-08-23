from __future__ import annotations

import unittest

from badminton.elo import PairwiseRecord, calculate_elo, expected_score


class EloTests(unittest.TestCase):
    def test_expected_score_is_symmetric(self) -> None:
        self.assertAlmostEqual(expected_score(1500.0, 1500.0), 0.5)
        self.assertAlmostEqual(expected_score(1600.0, 1400.0), 1.0 - expected_score(1400.0, 1600.0))

    def test_calculate_elo_orders_fixed_pool(self) -> None:
        ratings = calculate_elo(
            [
                PairwiseRecord("a", "b", agent_a_score=80.0, games=100.0),
                PairwiseRecord("b", "c", agent_a_score=70.0, games=100.0),
                PairwiseRecord("a", "c", agent_a_score=90.0, games=100.0),
            ]
        )

        self.assertGreater(ratings["a"], ratings["b"])
        self.assertGreater(ratings["b"], ratings["c"])


if __name__ == "__main__":
    unittest.main()
