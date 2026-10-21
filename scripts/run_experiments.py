"""Bounded detection-improvement experiments (Priority 1).

Stages, each writing experiments/<stage>.json:
  prepare      per photo: score maps, candidate sets (ring+LoG, Hough, union),
               base and extra features, contrast-jittered training features
  generators   candidate recall ceiling, separately from classification
  classifiers  cross-photo AP and calibrated detection quality per variant
  refine       center-refinement variants on the selected classifier
  calibration  count-estimator stability under single reviewer mistakes

Protocol: every photo is scored by a model trained only on the other photo.
The selection rule for the classifier was fixed before running: highest mean
cross-photo AP; differences under 0.01 go to the simpler variant. Because
only two photos exist, selecting on them is still tuning on evaluation data,
and is disclosed as such.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from barnacles.frame import detect  # noqa: E402
from barnacles.rectify import rectify  # noqa: E402
from barnacles.candidate_detection import (CandidateParams, score_maps, detect_candidates,  # noqa: E402
                                           hough_candidates, merge_candidates)
from barnacles.features import candidate_features  # noqa: E402
from barnacles.ml import (DuplicateIndex, DUP_FRAC, label_candidates, make_model, select, select_count,  # noqa: E402
                          precision_recall_curve_vs_annotations, average_precision_vs_annotations)
from barnacles.evaluation import evaluate, local_density  # noqa: E402
from barnacles import calibration as cal  # noqa: E402
from barnacles.paths import data_dir, experiment_cache  # noqa: E402

DATA = data_dir()
OUT = ROOT / "experiments"
CACHE = experiment_cache()
PHOTOS = ("img1.png", "img2.png")
SEEDS = 10
K = 80


def jdump(o, p):
    json.dump(o, open(p, "w"), indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v))


def annotations(name):
    e = next(x for x in json.load(open(ROOT / "phase1_output/annotations.json"))["images"] if x["file"] == name)
    rec = [r for r in e["annotations"] if r["counted"]]
    gt = np.array([[r["x"], r["y"]] for r in rec])
    diam = float(np.median([r["equiv_diameter_px"] for r in rec]))
    return gt, diam, float(e["summary"]["median_nn_spacing_px"])


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in PHOTOS:
        path = CACHE / f"{name}.pkl"
        if path.exists():
            continue
        img = cv2.imread(str(DATA / name), cv2.IMREAD_COLOR)
        f, _ = detect(img)
        rect = rectify(img, f)
        t = time.time(); maps = score_maps(rect); t_maps = time.time() - t
        t = time.time(); g0 = detect_candidates(rect, maps=maps); t_g0 = time.time() - t
        t = time.time(); gh = hough_candidates(rect, maps); t_h = time.time() - t
        gu, from_h = merge_candidates(g0, gh)
        sets = {"ringlog": g0, "hough": gh, "union": gu}
        feats = {}
        for gname, c in sets.items():
            t = time.time()
            X, names = candidate_features(rect, c, maps, extra=True)
            feats[gname] = (X, names, time.time() - t)
        # contrast / illumination jitter of the photo, same candidate positions (ringlog only)
        jit = {}
        for tag, gamma, gain in (("gamma0.7", 0.7, 1.0), ("gamma1.4", 1.4, 1.0), ("gain0.75", 1.0, 0.75)):
            im = np.clip(255 * ((rect.image / 255.0) ** gamma) * gain, 0, 255).astype(np.uint8)
            r2 = type(rect)(im, rect.size, rect.margin, rect.M, rect.frame)
            m2 = score_maps(r2)
            c2 = dict(g0)
            k, yy, xx = c2["level"], np.round(c2["y"]).astype(int), np.round(c2["x"]).astype(int)
            c2.update(score=m2["combined"][k, yy, xx], ring_score=m2["ring"][k, yy, xx], log_score=m2["log"][k, yy, xx])
            jit[tag] = candidate_features(r2, c2, m2, extra=True)[0]
            del m2
        # center-refinement support: ring-map local maximum within +/- 0.6 r at the candidate scale
        ring = maps["ring"]
        rx, ry = g0["x"].copy(), g0["y"].copy()
        for i in range(len(g0["x"])):
            k = g0["level"][i]; h = max(1, int(round(0.6 * g0["radius"][i])))
            x0, y0 = int(g0["x"][i]), int(g0["y"][i])
            win = ring[k, max(y0 - h, 0):y0 + h + 1, max(x0 - h, 0):x0 + h + 1]
            dy, dx = np.unravel_index(np.argmax(win), win.shape)
            rx[i], ry[i] = max(x0 - h, 0) + dx, max(y0 - h, 0) + dy
        pickle.dump({"rect_M": rect.M, "frame_corners": f.corners, "size": rect.size, "margin": rect.margin,
                     "sets": sets, "from_hough": from_h, "feats": feats, "jitter": jit,
                     "ring_refined_xy": np.c_[rx, ry],
                     "timing": {"maps_s": t_maps, "ringlog_s": t_g0, "hough_s": t_h}},
                    open(path, "wb"))
        del maps
        print("prepared", name, flush=True)


def load(name):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    img = cv2.imread(str(DATA / name), cv2.IMREAD_COLOR)
    f, _ = detect(img)
    d["frame"] = f
    rect = rectify(img, f)
    d["rect"] = rect
    d["s"] = rect.px_per_rect_px()
    gt, diam, sp = annotations(name)
    d.update(gt=gt, diam=diam, R=0.5 * diam, spacing=sp)
    dg = local_density(gt, gt, 2.5 * sp)
    d["cuts"] = np.quantile(dg, [1 / 3, 2 / 3])
    d["gt_stratum"] = np.digitize(dg, d["cuts"])
    return d


def to_img(d, c):
    return d["rect"].rect_to_image(np.c_[c["x"], c["y"]])


def generators():
    res = {}
    for name in PHOTOS:
        d = load(name)
        res[name] = {}
        for g, c in d["sets"].items():
            xy = to_img(d, c)
            row = {"n_candidates": int(len(xy))}
            for frac in (0.25, 0.5):
                m, (pairs, _, _) = evaluate(xy, d["gt"], frac * d["diam"])
                row[f"recall_ceiling_r{frac}"] = m["annotated_recall"]
                if frac == 0.5:
                    row["median_dist_px"] = m["match_dist_median_px"]
                    mg = np.zeros(len(d["gt"]), bool); mg[[q for _, q, _ in pairs]] = True
                    row["ceiling_by_density"] = {nm: float(mg[d["gt_stratum"] == k].mean())
                                                 for k, nm in enumerate(("sparse", "medium", "dense"))}
            row["seconds"] = (d["timing"]["maps_s"] + d["timing"]["ringlog_s"]) if g == "ringlog" else (
                d["timing"]["maps_s"] + d["timing"]["hough_s"] if g == "hough" else
                d["timing"]["maps_s"] + d["timing"]["ringlog_s"] + d["timing"]["hough_s"])
            row["feature_seconds"] = d["feats"][g][2]
            res[name][g] = row
            print(name, g, json.dumps(row), flush=True)
    jdump(res, OUT / "generators.json")


def feature_matrix(d, gen, fset, stdz):
    X, names, _ = d["feats"][gen]
    if fset == "base":
        cols = [i for i, n in enumerate(names) if not (n.startswith("hog_wide") or n.startswith("radial_grad")
                                                        or n.startswith("radial_absgrad"))]
        X = X[:, cols]
    if stdz:
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return X


def calibrated_runs(d, xy, dup, score, labels, seeds=SEEDS, k=K, estimator="platt"):
    rows = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        idx, w, strat = cal.stratified_sample(score, k, rng)
        if estimator == "platt":
            ab = cal.fit_platt(score[idx], labels[idx], w, True)
            n_hat = float(cal.calibrated_prob(score, ab, True).sum()) if ab else 0.0
        else:
            sizes = np.bincount(strat, minlength=len(cal.STRATA_Q) - 1)
            n_hat = cal.isotonic_count(labels[idx], strat[idx], sizes)
        det, thr = select_count(dup, score, int(round(n_hat)))
        m, (pairs, um, mg) = evaluate(xy[det], d["gt"], d["R"])
        rows.append({"n_hat": n_hat, "n_det": m["n_detections"], "f1": m["f1_vs_annotations"],
                     "recall": m["annotated_recall"], "prec_lb": m["annotation_precision_lower_bound"],
                     "med_dist": m["match_dist_median_px"], "det": det, "pairs": pairs})
    return rows


def summarize(rows, d):
    f = lambda k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
    r0 = rows[0]
    mg = np.zeros(len(d["gt"]), bool); mg[[q for _, q, _ in r0["pairs"]]] = True
    return {"f1": f("f1"), "n_det": f("n_det"), "recall": f("recall"), "prec_lb": f("prec_lb"),
            "median_dist_px": f("med_dist"), "n_annotations": len(d["gt"]),
            "seed0_recall_by_density": {nm: float(mg[d["gt_stratum"] == k].mean())
                                        for k, nm in enumerate(("sparse", "medium", "dense"))},
            "seed0_unmatched": int(r0["n_det"] - len(r0["pairs"]))}


VARIANTS = {
    "C0 current: ring+LoG, base features, HGB": dict(gen="ringlog", fset="base", stdz=False, model="hgb"),
    "C1 + per-photo feature standardization": dict(gen="ringlog", fset="base", stdz=True, model="hgb"),
    "C2 + wide HOG and radial gradients": dict(gen="ringlog", fset="extra", stdz=False, model="hgb"),
    "C3 C2 + per-photo standardization": dict(gen="ringlog", fset="extra", stdz=True, model="hgb"),
    "C4 C1 features, random forest": dict(gen="ringlog", fset="base", stdz=True, model="rf"),
    "C5 C1 + contrast-jitter augmentation": dict(gen="ringlog", fset="base", stdz=True, model="hgb", aug=True),
    "C6 C1 on union (ring+LoG + Hough) candidates": dict(gen="union", fset="base", stdz=True, model="hgb"),
    # added after C0 to C4 had run: C1 (standardization) hurt, C2 helped, and the
    # generator stage showed Hough proposals are better centred, so the remaining
    # variants build on C2 instead of C1.
    "C7 C2 features on Hough candidates": dict(gen="hough", fset="extra", stdz=False, model="hgb"),
    "C8 C2 features on union candidates": dict(gen="union", fset="extra", stdz=False, model="hgb"),
    "C9 C2 + contrast-jitter augmentation": dict(gen="ringlog", fset="extra", stdz=False, model="hgb", aug=True),
}


def model_of(kind):
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features="sqrt",
                                      class_weight="balanced_subsample", random_state=0, n_jobs=1)
    return make_model(kind)


def jitter_matrix(d, tag, fset, stdz):
    X = d["jitter"][tag]
    names = d["feats"]["ringlog"][1]
    if fset == "base":
        X = X[:, [i for i, n in enumerate(names) if not (n.startswith("hog_wide") or n.startswith("radial_grad")
                                                          or n.startswith("radial_absgrad"))]]
    return (X - X.mean(0)) / (X.std(0) + 1e-6) if stdz else X


def classifiers(only=None):
    P = {n: load(n) for n in PHOTOS}
    out_path = OUT / "classifiers.json"
    res = json.load(open(out_path)) if out_path.exists() else {}
    for vname, v in VARIANTS.items():
        if only and vname not in only:
            continue
        if vname in res:
            continue
        res[vname] = {"config": v}
        for A_n, B_n in (("img1.png", "img2.png"), ("img2.png", "img1.png")):
            A, B = P[A_n], P[B_n]
            ca, cb = A["sets"][v["gen"]], B["sets"][v["gen"]]
            xa, xb = to_img(A, ca), to_img(B, cb)
            ya, _ = label_candidates(xa, A["gt"], A["R"])
            yb, _ = label_candidates(xb, B["gt"], B["R"])
            XA, XB = feature_matrix(A, v["gen"], v["fset"], v["stdz"]), feature_matrix(B, v["gen"], v["fset"], v["stdz"])
            lab = ya >= 0
            Xtr, ytr = XA[lab], ya[lab]
            if v.get("aug"):
                for tag in A["jitter"]:
                    Xtr = np.vstack([Xtr, jitter_matrix(A, tag, v["fset"], v["stdz"])[lab]])
                    ytr = np.r_[ytr, ya[lab]]
            t = time.time(); m = model_of(v["model"]).fit(Xtr, ytr); t_fit = time.time() - t
            t = time.time(); sB = m.predict_proba(XB)[:, 1]; t_pred = time.time() - t
            dup = DuplicateIndex(xb, cb["radius"] * B["s"], DUP_FRAC)
            ap = average_precision_vs_annotations(precision_recall_curve_vs_annotations(dup, sB, B["gt"], B["R"]))
            rows = calibrated_runs(B, xb, dup, sB, (yb == 1).astype(int))
            key = f"test {B_n}"
            res[vname][key] = {"ap": ap, "n_candidates": int(len(xb)),
                               "candidate_ceiling": evaluate(xb, B["gt"], B["R"])[0]["annotated_recall"],
                               "calibrated_k80": summarize(rows, B), "fit_s": t_fit, "predict_s": t_pred,
                               "n_features": int(XA.shape[1])}
            np.save(CACHE / f"scores_{vname[:2]}_{B_n}.npy", sB)
            print(vname, key, "AP %.3f F1 %.3f+-%.3f n %.0f+-%.0f" % (
                ap, *res[vname][key]["calibrated_k80"]["f1"], *res[vname][key]["calibrated_k80"]["n_det"]), flush=True)
        jdump(res, out_path)


def _variant(prefix):
    return next(k for k in VARIANTS if k.startswith(prefix))


def refine(prefix):
    """Center refinement on the selected variant's seed-0 calibrated detections.

    R0 none; R1 ring-map local maximum within +/- 0.6 r (opening margin
    contrast); R2 classifier local maximum: re-score the detection at 8
    offsets of +/- 0.4 r and move to the best-scoring position if it beats
    the original. After moving, duplicates are suppressed again so any
    merging of neighbours is counted, not hidden."""
    vname = _variant(prefix)
    v = VARIANTS[vname]
    from barnacles.ml import suppress_duplicates
    res = {"variant": vname}
    P = {n: load(n) for n in PHOTOS}
    for A_n, B_n in (("img1.png", "img2.png"), ("img2.png", "img1.png")):
        A, B = P[A_n], P[B_n]
        cb = B["sets"][v["gen"]]
        xb = to_img(B, cb)
        yb, _ = label_candidates(xb, B["gt"], B["R"])
        sB = np.load(CACHE / f"scores_{vname[:2]}_{B_n}.npy")
        dup = DuplicateIndex(xb, cb["radius"] * B["s"], DUP_FRAC)
        row0 = calibrated_runs(B, xb, dup, sB, (yb == 1).astype(int), seeds=1)[0]
        det = row0["det"]
        # R1
        r1 = B["rect"].rect_to_image(B["ring_refined_xy"][det]) if v["gen"] == "ringlog" else xb[det]
        # R2: classifier local maximum (needs score maps for the shifted features)
        ca = A["sets"][v["gen"]]
        xa = to_img(A, ca)
        ya, _ = label_candidates(xa, A["gt"], A["R"])
        XA = feature_matrix(A, v["gen"], v["fset"], v["stdz"])
        X_all_B, names, _ = B["feats"][v["gen"]]
        mu, sd = None, None
        if v["stdz"]:
            Xb_full = feature_matrix(B, v["gen"], v["fset"], False)
            mu, sd = Xb_full.mean(0), Xb_full.std(0) + 1e-6
        model = model_of(v["model"]).fit(XA[ya >= 0], ya[ya >= 0])
        maps = score_maps(B["rect"])
        offs = [(dx, dy) for dx in (-0.4, 0, 0.4) for dy in (-0.4, 0, 0.4) if (dx, dy) != (0, 0)]
        best_xy = np.c_[cb["x"][det], cb["y"][det]].copy()
        best_s = sB[det].copy()
        t = time.time()
        for dx, dy in offs:
            c2 = {k_: cb[k_][det].copy() for k_ in cb}
            c2["x"] = c2["x"] + dx * c2["radius"]
            c2["y"] = c2["y"] + dy * c2["radius"]
            k_, yy, xx = c2["level"], np.clip(np.round(c2["y"]).astype(int), 0, maps["combined"].shape[1] - 1), \
                np.clip(np.round(c2["x"]).astype(int), 0, maps["combined"].shape[2] - 1)
            c2.update(score=maps["combined"][k_, yy, xx], ring_score=maps["ring"][k_, yy, xx], log_score=maps["log"][k_, yy, xx])
            X2, n2 = candidate_features(B["rect"], c2, maps, extra=(v["fset"] == "extra"))
            if v["fset"] == "base":
                X2 = X2  # extra=False already gives the base columns
            if v["stdz"]:
                X2 = (X2 - mu) / sd
            s2 = model.predict_proba(X2)[:, 1]
            better = s2 > best_s
            best_s[better] = s2[better]
            best_xy[better] = np.c_[c2["x"], c2["y"]][better]
        t_r2 = time.time() - t
        del maps
        r2 = B["rect"].rect_to_image(best_xy)
        out = {}
        for tag, xy in (("R0 none", xb[det]), ("R1 ring-map local max", r1), ("R2 classifier local max", r2)):
            keep = suppress_duplicates(xy, sB[det], cb["radius"][det] * B["s"], DUP_FRAC)
            xy_k = xy[keep]
            row = {"n_before": int(len(det)), "n_after_resuppression": int(len(keep))}
            for frac in (0.25, 0.5):
                m, _ = evaluate(xy_k, B["gt"], frac * B["diam"])
                row[f"f1_r{frac}"] = m["f1_vs_annotations"]
                row[f"recall_r{frac}"] = m["annotated_recall"]
                if frac == 0.5:
                    row["median_dist_px"] = m["match_dist_median_px"]
                    row["p90_dist_px"] = m["match_dist_p90_px"]
            out[tag] = row
            print(B_n, tag, json.dumps(row), flush=True)
        out["R2_seconds"] = t_r2
        res[f"test {B_n}"] = out
    jdump(res, OUT / "refine.json")


def calibration(prefix):
    """Count-estimator accuracy and stability under single reviewer mistakes."""
    vname = _variant(prefix)
    v = VARIANTS[vname]
    res = {"variant": vname}
    for B_n in PHOTOS:
        B = load(B_n)
        cb = B["sets"][v["gen"]]
        xb = to_img(B, cb)
        yb, _ = label_candidates(xb, B["gt"], B["R"])
        lab = (yb == 1).astype(int)
        sB = np.load(CACHE / f"scores_{vname[:2]}_{B_n}.npy")
        dup = DuplicateIndex(xb, cb["radius"] * B["s"], DUP_FRAC)
        out = {"candidate_positives": int(lab.sum())}
        for est in ("platt", "isotonic"):
            rows = calibrated_runs(B, xb, dup, sB, lab, seeds=20, estimator=est)
            # stability: flip each label of the first 10 draws, recompute the count estimate
            rel, rel_unflagged = [], []
            for s in range(10):
                rng = np.random.default_rng(1000 + s)
                idx, w, strat = cal.stratified_sample(sB, K, rng)
                sizes = np.bincount(strat, minlength=len(cal.STRATA_Q) - 1)
                def n_of(l):
                    if est == "platt":
                        ab = cal.fit_platt(sB[idx], l, w, True)
                        return float(cal.calibrated_prob(sB, ab, True).sum()) if ab else 0.0
                    return cal.isotonic_count(l, strat[idx], sizes)
                base = n_of(lab[idx])
                infl = []
                for j in range(len(idx)):
                    l2 = lab[idx].copy(); l2[j] = 1 - l2[j]
                    infl.append(abs(n_of(l2) - base) / max(base, 1))
                infl = np.array(infl)
                rel.append(infl.max())
                rel_unflagged.append(infl[infl <= 0.10].max() if (infl <= 0.10).any() else 0.0)
            f = lambda k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
            out[est] = {"f1": f("f1"), "n_det": f("n_det"), "n_hat": f("n_hat"),
                        "count_abs_rel_error_mean": float(np.mean([abs(r["n_hat"] - lab.sum()) / lab.sum() for r in rows])),
                        "single_flip_max_rel_change_mean": float(np.mean(rel)),
                        "single_flip_max_rel_change_worst": float(np.max(rel)),
                        "max_rel_change_after_confirming_flagged_labels": float(np.mean(rel_unflagged))}
            print(B_n, est, json.dumps(out[est]), flush=True)
        res[B_n] = out
    jdump(res, OUT / "calibration.json")


if __name__ == "__main__":
    stage = sys.argv[1]
    OUT.mkdir(exist_ok=True)
    if stage == "prepare":
        prepare()
    elif stage == "generators":
        generators()
    elif stage == "classifiers":
        classifiers(sys.argv[2:] or None)
    elif stage == "refine":
        refine(sys.argv[2])
    elif stage == "calibration":
        calibration(sys.argv[2])
