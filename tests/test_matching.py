import unittest

import numpy as np

import tests.helpers  # noqa: F401
from barnacles.evaluation import match, evaluate


class TestMatching(unittest.TestCase):
    def test_optimal_not_greedy(self):
        # Greedy nearest-first would pair P0-G1 (dist 1) and leave G0 unmatched.
        pred = [[0, 0], [3, 0]]
        gt = [[-1.9, 0], [1, 0]]
        pairs, up, ug = match(pred, gt, 2.0)
        self.assertEqual(sorted((p, g) for p, g, _ in pairs), [(0, 0), (1, 1)])
        self.assertEqual((up, ug), ([], []))

    def test_one_to_one_and_radius(self):
        pairs, up, ug = match([[0, 0], [0.5, 0]], [[0, 0]], 1.0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(up), 1)
        pairs, up, ug = match([[0, 0]], [[5, 0]], 1.0)
        self.assertEqual((pairs, up, ug), ([], [0], [0]))

    def test_empty(self):
        self.assertEqual(match([], [[1, 1]], 1.0), ([], [], [0]))
        self.assertEqual(match([[1, 1]], [], 1.0), ([], [0], []))

    def test_metrics(self):
        m, _ = evaluate([[0, 0], [10, 10], [20, 20]], [[0, 1], [10, 10], [50, 50], [60, 60]], 2.0)
        self.assertEqual(m["n_matched"], 2)
        self.assertAlmostEqual(m["annotated_recall"], 0.5)
        self.assertAlmostEqual(m["annotation_precision_lower_bound"], 2 / 3)
        self.assertAlmostEqual(m["f1_vs_annotations"], 4 / 7)
        self.assertEqual(m["count_minus_annotations"], -1)

    def test_large_random_is_valid(self):
        rng = np.random.default_rng(0)
        g = rng.uniform(0, 1000, (800, 2))
        p = np.r_[g + rng.normal(0, 2, g.shape), rng.uniform(0, 1000, (400, 2))]
        pairs, up, ug = match(p, g, 6.0)
        self.assertEqual(len({a for a, _, _ in pairs}), len(pairs))
        self.assertEqual(len({b for _, b, _ in pairs}), len(pairs))
        self.assertTrue(all(d <= 6.0 for _, _, d in pairs))
        self.assertEqual(len(pairs) + len(up), len(p))
        self.assertEqual(len(pairs) + len(ug), len(g))


if __name__ == "__main__":
    unittest.main()
