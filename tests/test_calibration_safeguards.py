import json
import unittest

import numpy as np

import tests.helpers  # noqa: F401
from barnacles import calibration as cal
from barnacles.review import replay, neighbours_from_pairs


class TestSampleSize(unittest.TestCase):
    def test_exact_budget(self):
        s = np.random.default_rng(0).uniform(0, 1, 5000)
        for k in (9, 20, 40, 72, 80, 81, 150):
            idx, w, strat = cal.stratified_sample(s, k, np.random.default_rng(1))
            self.assertEqual(len(idx), k)
            self.assertEqual(len(set(idx.tolist())), k)
        self.assertEqual(len(cal.stratified_sample(s, 80, np.random.default_rng(1), legacy=True)[0]), 72)

    def test_small_population(self):
        s = np.random.default_rng(0).uniform(0, 1, 30)
        self.assertEqual(len(cal.stratified_sample(s, 80, np.random.default_rng(1))[0]), 30)

    def test_weights_sum_to_population(self):
        s = np.random.default_rng(2).uniform(0, 1, 7000)
        idx, w, strat = cal.stratified_sample(s, 80, np.random.default_rng(3))
        self.assertAlmostEqual(w.sum(), 7000, delta=1e-6)


class TestIsotonicSafeguard(unittest.TestCase):
    def test_monotone_rates(self):
        r = cal.isotonic_stratum_rates(np.array([1, 0, 0, 0, 1, 1, 0, 1]), np.array([0, 0, 1, 1, 2, 2, 3, 3]), 4)
        self.assertTrue(np.all(np.diff(r) >= 0))

    def test_single_low_stratum_error_is_damped(self):
        # 9 strata, 8 labels each; a big low-score stratum with one mistaken positive
        sizes = np.array([3000, 1200, 600, 300, 300, 300, 180, 180, 120])
        strat = np.repeat(np.arange(9), 8)
        labels = np.r_[np.zeros(40, int), [0, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0], np.ones(16, int)]
        bad = labels.copy(); bad[0] = 1                      # mistake in the 3000-candidate stratum
        ht = lambda l: sum(sizes[s] * l[strat == s].mean() for s in range(9))
        iso = lambda l: cal.isotonic_count(l, strat, sizes)
        self.assertGreater(ht(bad) - ht(labels), 300)       # naive estimate jumps by 375
        self.assertLess(iso(bad) - iso(labels), 0.6 * (ht(bad) - ht(labels)))
        base, infl = cal.label_influence(labels, strat, sizes)
        self.assertGreater(np.abs(infl).max(), 0)


class TestExportImport(unittest.TestCase):
    def test_session_round_trip(self):
        c = {"id": [f"c{i}" for i in range(4)], "x": [0, 3, 40, 80], "y": [0, 0, 40, 80],
             "u": [0.1, 0.11, 0.5, 0.9], "v": [0.1, 0.1, 0.5, 0.9], "ml_score": [0.9, 0.8, 0.6, 0.3]}
        nb = neighbours_from_pairs(4, [(0, 1)])
        sess = {"operating_point": {"threshold": 0.5, "initial_threshold": 0.5}, "grid": {"rows": 3, "cols": 3},
                "actions": [{"action": "remove", "id": "c2", "seq": 1},
                            {"action": "add", "id": "a1", "x": 5, "y": 5, "u": 0.2, "v": 0.2, "seq": 2},
                            {"action": "set_calibration", "a": 1.2, "b": -0.3, "threshold": 0.25, "n_hat": 3,
                             "labels": [], "seq": 3}]}
        r1 = replay(c, nb, sess)
        r2 = replay(c, nb, json.loads(json.dumps(sess)))
        self.assertEqual(r1, r2)
        self.assertEqual(r1["corrected_count"], 3)    # c0, c3 (after threshold drop), a1; c2 removed
        self.assertEqual(sum(r1["per_cell_corrected"].values()), r1["corrected_count"])


if __name__ == "__main__":
    unittest.main()
