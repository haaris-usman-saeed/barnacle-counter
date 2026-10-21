"""Train the deployable detector on both development photos.

Usage:
    python train_model.py --data /path/to/uploads --phase1 phase1_output/annotations.json

Configuration = experiment C7 (selected in experiments/classifiers.json by
the pre-declared rule: highest mean cross-photo AP). Hough-circle proposals,
base + wide-HOG + radial-gradient features, gradient boosting.

The evaluation numbers for this configuration come from the cross-photo
experiments (each photo scored by a model trained on the other). This final
model is trained on BOTH photos, so it has no held-out evaluation of its
own; its default threshold is chosen by spatially blocked cross-validation
(2 x 2 blocks per photo, 8 folds) and is only a starting point. On a new
photo the reviewer should calibrate (review UI) before trusting the count.
"""
import argparse
import datetime
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import sklearn

from barnacles.paths import data_dir
from barnacles.frame import detect
from barnacles.rectify import rectify
from barnacles.candidate_detection import CandidateParams, score_maps, hough_candidates
from barnacles.features import candidate_features
from barnacles.ml import (DuplicateIndex, DUP_FRAC, SEED, label_candidates, make_model, spatial_blocks, select,
                          best_threshold)
from barnacles.evaluation import evaluate
from barnacles.inference import ARTIFACT_VERSION, pixel_hash
from barnacles.fingerprint import write_refs

CONFIG = {"variant": "C7: Hough candidates, base + wide HOG + radial gradients, gradient boosting",
          "generator": "hough", "extra_features": True, "per_photo_standardize": False, "model_kind": "hgb"}


def photo(entry, data):
    img = cv2.imread(str(data / entry["file"]), cv2.IMREAD_COLOR)
    f, _ = detect(img)
    rect = rectify(img, f)
    maps = score_maps(rect)
    c = hough_candidates(rect, maps)
    X, names = candidate_features(rect, c, maps, extra=True)
    del maps
    xy = rect.rect_to_image(np.c_[c["x"], c["y"]])
    uv = rect.rect_to_uv(np.c_[c["x"], c["y"]])
    rec = [r for r in entry["annotations"] if r["counted"]]
    gt = np.array([[r["x"], r["y"]] for r in rec])
    R = 0.5 * float(np.median([r["equiv_diameter_px"] for r in rec]))
    y, _ = label_candidates(xy, gt, R)
    dup = DuplicateIndex(xy, c["radius"] * rect.px_per_rect_px(), DUP_FRAC)
    return {"file": entry["file"], "X": X, "names": names, "y": y, "uv": uv, "xy": xy, "gt": gt, "R": R,
            "dup": dup, "hash": pixel_hash(img)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(data_dir()), help="folder with the challenge photos and masks (default: data/ or $BARNACLE_DATA)")
    ap.add_argument("--phase1", default="phase1_output/annotations.json")
    ap.add_argument("--out", default="models/barnacle_detector.pkl")
    args = ap.parse_args()
    t0 = time.time()
    P = [photo(e, Path(args.data)) for e in json.load(open(args.phase1))["images"]]
    names = P[0]["names"]
    # pooled blocked CV: 4 spatial blocks per photo = 8 folds
    oof = [np.zeros(len(p["X"])) for p in P]
    for pi, p in enumerate(P):
        blk = spatial_blocks(p["uv"], 2)
        for b in range(4):
            test = blk == b
            r, cc = divmod(b, 2)
            u0, u1, v0, v1 = cc / 2 - 0.03, (cc + 1) / 2 + 0.03, r / 2 - 0.03, (r + 1) / 2 + 0.03
            near = (p["uv"][:, 0] > u0) & (p["uv"][:, 0] < u1) & (p["uv"][:, 1] > v0) & (p["uv"][:, 1] < v1)
            Xtr = [p["X"][(~near) & (p["y"] >= 0)]] + [q["X"][q["y"] >= 0] for qi, q in enumerate(P) if qi != pi]
            ytr = [p["y"][(~near) & (p["y"] >= 0)]] + [q["y"][q["y"] >= 0] for qi, q in enumerate(P) if qi != pi]
            m = make_model(CONFIG["model_kind"]).fit(np.vstack(Xtr), np.concatenate(ytr))
            oof[pi][test] = m.predict_proba(p["X"][test])[:, 1]
            print(f"fold {p['file']} block {b} done", flush=True)
    # threshold maximizing the mean F1 over the two photos
    grid = np.unique(np.quantile(np.concatenate(oof), np.linspace(0.5, 0.999, 120)))
    best = (-1, None)
    for t in grid:
        f1s = []
        for p, s in zip(P, oof):
            idx = select(p["dup"], s, t)
            f1s.append(evaluate(p["xy"][idx], p["gt"], p["R"])[0]["f1_vs_annotations"])
        if np.mean(f1s) > best[0]:
            best = (float(np.mean(f1s)), float(t), f1s)
    X = np.vstack([p["X"][p["y"] >= 0] for p in P])
    y = np.concatenate([p["y"][p["y"] >= 0] for p in P])
    model = make_model(CONFIG["model_kind"]).fit(X, y)
    exp = json.load(open("experiments/classifiers.json")) if Path("experiments/classifiers.json").exists() else {}
    art = {"version": ARTIFACT_VERSION, **CONFIG, "model": model, "feature_names": names,
           "candidate_params": CandidateParams().to_dict(), "dup_frac": DUP_FRAC, "rect_size": 1600,
           "default_threshold": best[1], "default_threshold_cv_f1_per_photo": dict(zip([p["file"] for p in P], best[2])),
           "trained_on": [p["file"] for p in P], "n_train": {"positive": int((y == 1).sum()), "negative": int((y == 0).sum())},
           "training_hashes": {p["hash"]: p["file"] for p in P},
           "cross_photo_evaluation": {k: v for k, v in exp.items() if k.startswith("C7")},
           "sklearn_version": sklearn.__version__, "seed": SEED,
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "train_seconds": round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(art, fh)
    write_refs({p["file"]: cv2.imread(str(Path(args.data) / p["file"]), cv2.IMREAD_COLOR) for p in P})
    print("saved", args.out, "default threshold %.3f" % best[1], "blocked-CV F1 per photo", best[2],
          "train seconds", art["train_seconds"])


if __name__ == "__main__":
    main()
