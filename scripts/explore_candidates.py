"""Which classical detector fits these photos? Reproducible comparison.

Each variant produces scale-space peaks (3 x 5 x 5 local maxima, interior
scales, non-maximum suppression) over the same radius range as the chosen
generator. Reported per photo: number of candidates and annotated recall
with all of them kept (the ceiling a classifier could reach), plus the best
F1 against annotations over a threshold sweep on the SAME photo (an
optimistic number, used only to choose the method).
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from barnacles.frame import detect  # noqa: E402
from barnacles.rectify import rectify  # noqa: E402
from barnacles.candidate_detection import (CandidateParams, normalize, log_stack, ring_stack,  # noqa: E402
                                           nms)
from barnacles.evaluation import evaluate  # noqa: E402
from barnacles.paths import data_dir  # noqa: E402


def peaks(S, radii, rect):
    mx = ndi.maximum_filter(S, size=(3, 5, 5), mode="nearest")
    pk = (S == mx) & (S > 0)
    pk[0] = pk[-1] = False
    k, yy, xx = np.nonzero(pk)
    xy = np.c_[xx, yy].astype(float)
    uv = rect.rect_to_uv(xy)
    ins = (uv >= 0).all(1) & (uv <= 1).all(1)
    xy, sc, r = xy[ins], S[k, yy, xx][ins], radii[k[ins]]
    keep = nms(xy, r, sc, 0.8)
    return xy[keep], sc[keep]


def hough(rect, dmin, dmax):
    g = cv2.GaussianBlur(cv2.cvtColor(rect.image, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=dmin * 0.8, param1=80, param2=14,
                         minRadius=int(dmin / 2), maxRadius=int(dmax / 2))
    c = np.zeros((0, 3)) if c is None else c[0]
    uv = rect.rect_to_uv(c[:, :2]) if len(c) else np.zeros((0, 2))
    c = c[(uv >= 0).all(1) & (uv <= 1).all(1)] if len(c) else c
    return c[:, :2], np.arange(len(c), 0, -1).astype(float)


def best_f1(xy, sc, gt, radius):
    best = 0.0
    for t in np.quantile(sc, np.linspace(0, 0.95, 30)) if len(sc) else []:
        m, _ = evaluate(xy[sc >= t], gt, radius)
        best = max(best, m["f1_vs_annotations"])
    return best


def main(data=str(data_dir()), phase1="phase1_output/annotations.json", out="phase3_output"):
    p = CandidateParams()
    res = {}
    for e in json.load(open(phase1))["images"]:
        img = cv2.imread(str(Path(data) / e["file"]), cv2.IMREAD_COLOR)
        f, _ = detect(img)
        rect = rectify(img, f)
        rec = [r for r in e["annotations"] if r["counted"]]
        gt = rect.image_to_rect([[r["x"], r["y"]] for r in rec])
        diam = np.median([r["equiv_diameter_px"] for r in rec]) / rect.px_per_rect_px()
        radius = 0.5 * diam
        gray = cv2.cvtColor(rect.image, cv2.COLOR_BGR2GRAY)
        z = normalize(gray, rect.size, p.bg_sigma_frac)
        raw = (gray.astype(np.float32) - gray.mean()) / (gray.std() + 1e-6)
        radii = np.geomspace(p.radius_min_frac, p.radius_max_frac, p.n_scales) * rect.size
        sig = radii / np.sqrt(2)
        variants = {
            "LoG bright, contrast-normalized": lambda: peaks(log_stack(z, sig, "bright"), radii, rect),
            "LoG bright, raw grey": lambda: peaks(log_stack(raw, sig, "bright"), radii, rect),
            "LoG dark, contrast-normalized": lambda: peaks(log_stack(z, sig, "dark"), radii, rect),
            "Hough circles": lambda: hough(rect, 0.4 * diam, 2.5 * diam),
            "ring only": lambda: peaks(ring_stack(z, radii), radii, rect),
            "ring + LoG bright (chosen)": lambda: peaks(
                (lambda R, L: R / R.std() + L / L.std())(ring_stack(z, radii), log_stack(z, sig, "bright")), radii, rect),
        }
        res[e["file"]] = {"n_annotations": len(gt), "match_radius_rect_px": float(radius)}
        for name, fn in variants.items():
            t0 = time.time()
            xy, sc = fn()
            dt = time.time() - t0
            m, _ = evaluate(xy, gt, radius)
            res[e["file"]][name] = {"n_candidates": int(len(xy)), "recall_all": m["annotated_recall"],
                                    "best_f1_same_photo": best_f1(xy, sc, gt, radius), "seconds": round(dt, 1)}
            print(e["file"], name, json.dumps(res[e["file"]][name]), flush=True)
    Path(out).mkdir(exist_ok=True)
    with open(Path(out) / "candidate_method_exploration.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
