"""Phase 4: build the human-in-the-loop review prototype.

Usage:
    python run_phase4.py --phase1 phase1_output/annotations.json \
        --phase3 phase3_output --data /path/to/uploads --out phase4_output

Reads Phase 3's cross-photo predictions (each photo scored by the model
trained on the other photo) and writes review.html plus phase4_report.json,
which includes a simulated review: how many model errors a reviewer finds
per item checked when following the review queue, versus checking random
detections, using the annotations as a stand-in reviewer.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from barnacles.paths import data_dir
from barnacles.io import load_image
from barnacles.frame import detect
from barnacles.ml import DuplicateIndex, DUP_FRAC
from barnacles.review import static_signals, model_set, simulate_review
from barnacles.grid_viewer import image_payload
from barnacles.review_viewer import write_review_viewer


def build(entry, rc, data):
    img = load_image(data / entry["file"])
    frame, _ = detect(img)
    C = rc["candidates"]
    x = np.array([c["x"] for c in C]); y = np.array([c["y"] for c in C])
    r = np.array([c["radius_px"] for c in C]); s = np.array([c["ml_score"] for c in C])
    cvs = np.array([c["cv_score"] for c in C])
    dup = DuplicateIndex(np.c_[x, y], r, DUP_FRAC)
    pairs = sorted({(int(min(a, b)), int(max(a, b))) for a in range(len(x)) for b in dup.nbrs[a]})
    nbrs = [[int(j) for j in nb] for nb in dup.nbrs]
    op = rc["recommended_ml"]
    mset = model_set(s, op["threshold"], nbrs)
    phase3_set = [i for i, c in enumerate(C) if c["ml_accepted"]]
    assert mset == phase3_set, "review model set differs from Phase 3 selection"
    acc = np.zeros(len(x), bool); acc[mset] = True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sig = static_signals(gray, frame, x, y, r, acc, cvs)
    rec = [a for a in entry["annotations"] if a["counted"]]
    ann = np.array([[a["x"], a["y"]] for a in rec])
    spacing = entry["summary"]["median_nn_spacing_px"]
    n = 150
    g = (np.arange(n) + 0.5) / n
    U, Vv = np.meshgrid(g, g)
    pts = frame.to_image(np.c_[U.ravel(), Vv.ravel()])
    dist, _ = cKDTree(ann).query(pts)
    bits = "".join("1" if d > 0.9 * spacing else "0" for d in dist)
    base = image_payload(entry["file"], img, frame, [{"id": "", "x": 0, "y": 0, "u": 0, "v": 0}],
                         __import__("barnacles.grid", fromlist=["GridSpec"]).GridSpec(10, 10))
    payload = {
        "name": Path(entry["file"]).stem, "file": entry["file"], "img": base["img"], "crop": base["crop"],
        "H": frame.H.tolist(), "Hinv": frame.H_inv.tolist(), "corners": frame.corners.tolist(), "rows": 10, "cols": 10,
        "c": {"id": [c["id"] for c in C], "x": [round(v, 2) for v in x], "y": [round(v, 2) for v in y],
              "u": [round(c["u"], 6) for c in C], "v": [round(c["v"], 6) for c in C], "r": [round(v, 2) for v in r],
              "s": s.tolist(), "cvp": [round(v, 4) for v in sig["cv_pct"]],
              "clar": sig["low_clarity"].astype(int).tolist(), "dense": sig["dense"].astype(int).tolist(),
              "edge": sig["near_frame"].astype(int).tolist()},
        "pairs": [v for p in pairs for v in p],
        "ann": [[round(a, 2), round(b, 2)] for a, b in ann],
        "cov": {"n": n, "bits": bits},
        "op": {"thr": op["threshold"], "a": op["platt_a_b"][0], "b": op["platt_a_b"][1],
               "method": "Count-matched calibration from 80 labels (simulated from the annotations in this demo)",
               "n_hat": op["n_hat"], "ht": op["ht_estimate"], "ht_se": op["ht_se"],
               "sample": [[int(q["index"]), int(q["label"]), float(q["weight"]), int(q["stratum"])] for q in op["sample"]]},
        "model": {"trained_on": rc["trained_on"], "kind": rc["ml_model"]},
    }
    # simulated review
    cands = {"ml_score": s, "x": x, "y": y}
    R = 0.5 * float(np.median([a["equiv_diameter_px"] for a in rec]))
    sims = []
    for budget in (25, 50, 100, 200):
        res = simulate_review(cands, nbrs, op["threshold"], op["platt_a_b"], sig, ann, R, budget, np.random.default_rng(0))
        rnd = [simulate_review(cands, nbrs, op["threshold"], op["platt_a_b"], sig, ann, R, budget,
                               np.random.default_rng(k))["errors_found_random"] for k in range(1, 21)]
        res["errors_found_random_mean"] = float(np.mean(rnd))
        res["errors_found_random_sd"] = float(np.std(rnd))
        sims.append(res)
    flags = {k: int(sig[k][acc].sum()) for k in ("low_clarity", "dense", "near_frame")}
    return payload, {"n_candidates": len(x), "n_model_detections": len(mset), "n_duplicate_pairs": len(pairs),
                     "static_flags_on_model_detections": flags, "simulated_review": sims}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="phase1_output/annotations.json")
    ap.add_argument("--phase3", default="phase3_output")
    ap.add_argument("--data", default=str(data_dir()), help="folder with the challenge photos and masks (default: data/ or $BARNACLE_DATA)")
    ap.add_argument("--out", default="phase4_output")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p1 = {e["file"]: e for e in json.load(open(args.phase1))["images"]}
    rc = json.load(open(Path(args.phase3) / "review_candidates.json"))
    payloads, report = [], {}
    for f in sorted(rc):
        pl, rep = build(p1[f], rc[f], Path(args.data))
        payloads.append(pl); report[f] = rep
        for sres in rep["simulated_review"]:
            print(f, f"budget {sres['budget']}: errors found, queue {sres['errors_found_priority']} vs random "
                     f"{sres['errors_found_random_mean']:.1f} (sd {sres['errors_found_random_sd']:.1f}); "
                     f"model {sres['n_model']}, unmatched {sres['n_model_unmatched']}, suggestions {sres['n_suggestions']} "
                     f"({sres['n_suggestions_matching_annotations']} match annotations)")
    # The current cross-photo demo (review.html) is built by scripts/build_demo.py;
    # this script keeps the original C0-model version under a separate name.
    write_review_viewer(payloads, out / "review_c0.html")
    with open(out / "phase4_report.json", "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
