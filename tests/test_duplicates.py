import unittest

import numpy as np

import tests.helpers  # noqa: F401
from barnacles.ml import DuplicateIndex, select, select_count, suppress_duplicates
from barnacles.review import model_set, neighbours_from_pairs


class TestDuplicateSuppression(unittest.TestCase):
    def test_keeps_best_of_overlapping(self):
        xy = np.array([[0, 0], [3, 0], [30, 0]], float)
        keep = suppress_duplicates(xy, np.array([0.5, 0.9, 0.1]), np.array([5, 5, 5.]), frac=1.0)
        self.assertEqual(list(keep), [1, 2])

    def test_separated_points_untouched(self):
        xy = np.array([[0, 0], [11, 0]], float)
        self.assertEqual(len(suppress_duplicates(xy, np.array([1, 1.]), np.array([5, 5.]), frac=1.5)), 2)
        self.assertEqual(len(suppress_duplicates(xy, np.array([1, 1.]), np.array([8, 8.]), frac=1.5)), 1)

    def test_review_rule_matches_ml_rule(self):
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 500, (3000, 2))
        r = rng.uniform(3, 9, 3000)
        s = rng.uniform(0, 1, 3000)
        dup = DuplicateIndex(xy, r, 1.5)
        for thr in (0.0, 0.3, 0.8):
            self.assertEqual(list(select(dup, s, thr)), model_set(s, thr, dup.nbrs))

    def test_select_count(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform(0, 1000, (2000, 2))
        dup = DuplicateIndex(xy, np.full(2000, 4.0), 1.5)
        s = rng.uniform(0, 1, 2000)
        idx, thr = select_count(dup, s, 300)
        self.assertGreaterEqual(len(idx), 300)
        self.assertLess(len(idx), 310)
        self.assertTrue((s[idx] >= thr).all())

    def test_pairs_roundtrip(self):
        nb = neighbours_from_pairs(4, [(0, 1), (2, 3)])
        self.assertEqual(nb, [[1], [0], [3], [2]])


if __name__ == "__main__":
    unittest.main()
