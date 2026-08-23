from __future__ import annotations

import importlib
import unittest

import badminton
from badminton.policy import MaskedBadmintonPolicy


class PackageNameTests(unittest.TestCase):
    def test_canonical_package_name_is_badminton(self) -> None:
        self.assertEqual(badminton.__name__, "badminton")

    def test_legacy_checkpoint_policy_path_resolves_to_canonical_class(self) -> None:
        legacy_policy = importlib.import_module("badminton1d.policy")
        self.assertIs(legacy_policy.MaskedBadmintonPolicy, MaskedBadmintonPolicy)


if __name__ == "__main__":
    unittest.main()
