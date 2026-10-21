import json
import pickle
import unittest

import numpy as np

from tests.helpers import ROOT
from barnacles.ml import make_model, blocked_oof_scores
from barnacles import calibration as cal


def data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.8, n) > 0.8).astype(int)
    uv = rng.uniform(0, 1, (n, 2))
    return X, y, uv


class TestReproducibility(unittest.TestCase):
    def test_models_deterministic(self):
        X, y, _ = data()
        for kind in ("logreg", "hgb"):
            a = make_model(kind).fit(X, y).predict_proba(X)[:, 1]
            b = make_model(kind).fit(X, y).predict_proba(X)[:, 1]
            np.testing.assert_array_equal(a, b)

    def test_blocked_cv_deterministic(self):
        X, y, uv = data(1)
        np.testing.assert_array_equal(blocked_oof_scores(X, y, uv, "hgb"), blocked_oof_scores(X, y, uv, "hgb"))

    def test_calibration_sample_deterministic(self):
        s = np.random.default_rng(2).uniform(0, 1, 5000)
        a = cal.stratified_sample(s, 80, np.random.default_rng(1000))
        b = cal.stratified_sample(s, 80, np.random.default_rng(1000))
        np.testing.assert_array_equal(a[0], b[0])

    def test_saved_model_reproduces_phase3_scores(self):
        out = ROOT / "phase3_output"
        if not (out / "review_candidates.json").exists() or not (out / "cache").exists():
            self.skipTest("phase3 outputs or cache not present (run run_phase3.py)")
        rc = json.loads((out / "review_candidates.json").read_text())
        for test, train in (("img2", "img1"), ("img1", "img2")):
            with open(out / "models" / f"model_trained_on_{train}.pkl", "rb") as fh:
                m = pickle.load(fh)
            with open(out / "cache" / f"{test}_candidates_features.pkl", "rb") as fh:
                _, Xc, names, _ = pickle.load(fh)
            X = Xc[:, [i for i, n in enumerate(names) if not n.startswith("color_")]]
            s = m["model"].predict_proba(X)[:, 1]
            ref = np.array([c["ml_score"] for c in rc[f"{test}.png"]["candidates"]])
            np.testing.assert_allclose(s, ref, rtol=0, atol=1e-12)


class TestCalibration(unittest.TestCase):
    def test_horvitz_thompson_close_to_truth(self):
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 20000)
        truth = (rng.uniform(0, 1, 20000) < s ** 4).astype(int)
        est = []
        for k in range(200):
            idx, w, strat = cal.stratified_sample(s, 80, np.random.default_rng(k))
            est.append(cal.horvitz_thompson_total(truth[idx], w, strat[idx])[0])
        self.assertLess(abs(np.mean(est) - truth.sum()) / truth.sum(), 0.05)

    def test_platt_threshold_recovers_midpoint(self):
        rng = np.random.default_rng(1)
        s = rng.uniform(-6, 6, 4000)
        y = (rng.uniform(0, 1, 4000) < 1 / (1 + np.exp(-(2 * s - 1)))).astype(int)
        ab = cal.fit_platt(s, y, np.ones(4000), is_prob=False)
        self.assertAlmostEqual(cal.threshold_from_platt(ab, False), 0.5, delta=0.15)


if __name__ == "__main__":
    unittest.main()
