"""Static cross-photo demo of the review interface (phase4_output/review.html).

Each development photo is shown with the SELECTED model configuration (C7)
trained only on the OTHER photo, so the errors on screen are honest
cross-photo errors. The operating point uses 80 labels simulated from the
annotations (exact budget, isotonic count, count-matched threshold, seed 0);
the "Fill the rest from annotations (demo)" button reuses those labels.
Requires the experiment cache (scripts/run_experiments.py prepare + classifiers).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiments as rx  # noqa: E402
from barnacles.ml import DuplicateIndex, DUP_FRAC, label_candidates, select_count  # noqa: E402
from barnacles import calibration as cal  # noqa: E402
from barnacles.review import static_signals  # noqa: E402
from barnacles.inference import viewer_payload  # noqa: E402
from barnacles.review_viewer import write_review_viewer  # noqa: E402


def main(out=ROOT / "phase4_output" / "review.html"):
    payloads = []
    for name, other in (("img1.png", "img2.png"), ("img2.png", "img1.png")):
        d = rx.load(name)
        img = cv2.imread(str(rx.DATA / name), cv2.IMREAD_COLOR)
        c = d["sets"]["hough"]
        xy = rx.to_img(d, c)
        uv = d["rect"].rect_to_uv(np.c_[c["x"], c["y"]])
        r = c["radius"] * d["s"]
        s = np.load(rx.CACHE / f"scores_C7_{name}.npy")
        y, _ = label_candidates(xy, d["gt"], d["R"])
        lab = (y == 1).astype(int)
        dup = DuplicateIndex(xy, r, DUP_FRAC)
        idx, w, strat = cal.stratified_sample(s, 80, np.random.default_rng(1000))
        sizes = np.bincount(strat, minlength=len(cal.STRATA_Q) - 1)
        n_hat = cal.isotonic_count(lab[idx], strat[idx], sizes)
        det, thr = select_count(dup, s, int(round(n_hat)))
        ab = cal.fit_platt(s[idx], lab[idx], w, True)
        acc = np.zeros(len(xy), bool); acc[det] = True
        sig = static_signals(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), d["frame"], xy[:, 0], xy[:, 1], r, acc, c["score"])
        art = {"trained_on": [other], "model_kind": "hgb", "version": 2, "variant": "C7 (cross-photo demo)"}
        ids = [f"{Path(name).stem}_d{i:05d}" for i in range(len(xy))]
        p = viewer_payload(img, d["frame"], Path(name).stem, ids, xy, uv, r, s, sig, dup, thr, idx, w, strat, art)
        p["file"] = name
        p["ann"] = [[round(float(a), 2), round(float(b), 2)] for a, b in d["gt"]]
        n = 150
        g = (np.arange(n) + 0.5) / n
        U, Vv = np.meshgrid(g, g)
        dist, _ = cKDTree(d["gt"]).query(d["frame"].to_image(np.c_[U.ravel(), Vv.ravel()]))
        p["cov"] = {"n": n, "bits": "".join("1" if q > 0.9 * d["spacing"] else "0" for q in dist)}
        p["op"].update(thr=float(thr), a=float(ab[0]), b=float(ab[1]), calibrated=True, n_hat=float(n_hat),
                       method="Calibrated from 80 labels simulated from the annotations (demo)",
                       sample=[[int(i), int(lab[i]), float(wi), int(strat[i])] for i, wi in zip(idx, w)])
        p["frame"]["confirmed"] = True
        p["upload"] = False
        payloads.append(p)
        print(name, "detections", len(det), "n_hat %.0f" % n_hat, "threshold %.3f" % thr)
    write_review_viewer(payloads, out)


if __name__ == "__main__":
    main()
