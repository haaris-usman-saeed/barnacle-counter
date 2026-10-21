"""Phase 3: OpenCV baseline vs ML candidate classifier, cross-image evaluation.

Usage:
    python run_phase3.py --phase1 phase1_output/annotations.json \
        --data /path/to/uploads --out phase3_output

Protocol (no evaluation photo is used for any choice that is then scored on it):
  * Candidates: the same classical generator (ring + LoG) on both photos.
  * OpenCV baseline: candidate score >= threshold, threshold chosen on the
    OTHER photo (best F1 against its annotations).
  * ML: features -> classifier trained on the OTHER photo. Model family
    (logistic regression vs gradient boosting) and threshold are chosen by
    spatially blocked cross-validation on the training photo only.
  * Each test photo is scored by a model that never saw it.
Secondary, clearly labelled: within-photo spatially blocked CV, an oracle
(best threshold picked on the test photo itself), and a colour ablation.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from barnacles.paths import data_dir
from barnacles.io import load_image
from barnacles.frame import detect
from barnacles.rectify import rectify
from barnacles.candidate_detection import CandidateParams, score_maps, detect_candidates
from barnacles.features import candidate_features
from barnacles.ml import (SEED, label_candidates, make_model, blocked_oof_scores, DuplicateIndex,
                          select, select_count, best_threshold, precision_recall_curve_vs_annotations,
                          average_precision_vs_annotations)
from barnacles.evaluation import evaluate, local_density
from barnacles.grid import GridSpec, assign
from barnacles import calibration as cal
from barnacles import phase3_viz as pv

KINDS = ("logreg", "hgb")
CACHE = Path("phase3_cache")
_OOF = {}


def oof_scores(A, feats, kind):
    """Blocked out-of-fold scores, computed once per (photo, features, model)."""
    key = (A["name"], feats, kind)
    path = CACHE / f"oof_{A['name']}_{feats}_{kind}.npz"
    if key not in _OOF and path.exists():
        z = np.load(path)
        _OOF[key] = (z["s"], float(z["t"]))
    if key not in _OOF:
        t = time.time()
        _OOF[key] = (blocked_oof_scores(A[feats], A["y"], A["uv"], kind), time.time() - t)
        np.savez(path, s=_OOF[key][0], t=_OOF[key][1])
        print(f"  oof {key} {_OOF[key][1]:.0f}s", flush=True)
    return _OOF[key]
DENSITY_NAMES = ["sparse", "medium", "dense"]


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def prepare(entry, data, timings):
    name = Path(entry["file"]).stem
    t = time.time()
    img = load_image(data / entry["file"])
    frame, _ = detect(img)
    rect = rectify(img, frame)
    t_frame = time.time() - t
    cache = CACHE / f"{name}_candidates_features.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            c, Xc, names, timings[name] = pickle.load(fh)
        timings[name]["frame_and_rectify_s"] = t_frame
    else:
        t = time.time()
        maps = score_maps(rect)
        c = detect_candidates(rect, maps=maps)
        t_cand = time.time() - t
        t = time.time()
        Xc, names = candidate_features(rect, c, maps, with_color=True)
        t_feat = time.time() - t
        timings[name] = {"frame_and_rectify_s": t_frame, "candidates_s": t_cand, "features_s": t_feat}
        with open(cache, "wb") as fh:
            pickle.dump((c, Xc, names, timings[name]), fh)
    n_base = sum(not n.startswith("color_") for n in names)

    s = rect.px_per_rect_px()
    cxy = rect.rect_to_image(np.c_[c["x"], c["y"]])
    uv = rect.rect_to_uv(np.c_[c["x"], c["y"]])
    rec = [r for r in entry["annotations"] if r["counted"]]
    gt = np.array([[r["x"], r["y"]] for r in rec])
    gt_uv = frame.to_norm(gt)
    diam = float(np.median([r["equiv_diameter_px"] for r in rec]))
    spacing = float(entry["summary"]["median_nn_spacing_px"])
    R = 0.5 * diam
    y, dnear = label_candidates(cxy, gt, R)
    dens_r = 2.5 * spacing
    dg = local_density(gt, gt, dens_r)
    q = np.quantile(dg, [1 / 3, 2 / 3])
    strat = lambda d: np.digitize(d, q)     # 0 sparse, 1 medium, 2 dense
    return {"name": name, "file": entry["file"], "img": img, "frame": frame, "rect": rect,
            "cands": c, "X": Xc[:, :n_base], "Xc": Xc, "feat_names": names, "cxy": cxy, "uv": uv,
            "radius_px": c["radius"] * s, "gt": gt, "gt_uv": gt_uv, "gt_ids": [r["id"] for r in rec],
            "diam": diam, "spacing": spacing, "R": R, "y": y, "dnear": dnear,
            "dup": DuplicateIndex(cxy, c["radius"] * s, frac=DUP_FRAC),
            "gt_stratum": strat(dg), "cand_stratum": strat(local_density(gt, cxy, dens_r)),
            "density_radius_px": dens_r, "density_cuts": q.tolist()}


def score_detections(P, idx, spec=GridSpec(10, 10)):
    det = P["cxy"][idx]
    m, (pairs, ud, mg) = evaluate(det, P["gt"], P["R"],
                                  strata=(P["gt_stratum"], P["cand_stratum"][idx], DENSITY_NAMES))
    r1, c1, _ = assign(P["uv"][idx], spec)
    r0, c0, _ = assign(P["gt_uv"], spec)
    cd = np.bincount(r1 * spec.cols + c1, minlength=spec.n_cells)
    ca = np.bincount(r0 * spec.cols + c0, minlength=spec.n_cells)
    m["grid_10x10"] = {"mean_abs_cell_diff": float(np.abs(cd - ca).mean()),
                       "max_abs_cell_diff": int(np.abs(cd - ca).max()),
                       "pearson_r": float(np.corrcoef(cd, ca)[0, 1]) if cd.std() > 0 else float("nan")}
    # Missed annotations: did the generator have a candidate there at all?
    has_cand = local_density(P["cxy"], P["gt"][mg], P["R"]) > 0 if len(mg) else np.array([], bool)
    m["missed_with_no_candidate"] = int((~has_cand).sum())
    m["missed_candidate_rejected"] = int(has_cand.sum())
    m["match_dist_median_px"] = m["match_dist_median_px"]
    return m, (pairs, ud, mg)


def train_direction(A, B, out, timings, feats="X", kinds=None):
    """Model selection + threshold on A by blocked CV; fit on A; score B."""
    res = {"train": A["file"], "test": B["file"], "features": feats}
    oof = {}
    t_cv = 0.0
    for k in (kinds or KINDS):
        oof[k], dt = oof_scores(A, feats, k)
        t_cv += dt
        curve = precision_recall_curve_vs_annotations(A["dup"], oof[k], A["gt"], A["R"])
        res[f"oof_ap_{k}"] = average_precision_vs_annotations(curve)
    kind = max(oof, key=lambda k: res[f"oof_ap_{k}"])
    thr, m_oof = best_threshold(A["dup"], oof[kind], A["gt"], A["R"])
    t = time.time()
    lab = A["y"] >= 0
    mpath = CACHE / f"model_{A['name']}_{feats}_{kind}.pkl"
    if mpath.exists():
        with open(mpath, "rb") as fh:
            model, t_fit = pickle.load(fh)
    else:
        model = make_model(kind).fit(A[feats][lab], A["y"][lab])
        t_fit = time.time() - t
        with open(mpath, "wb") as fh:
            pickle.dump((model, t_fit), fh)
    t = time.time()
    sB = model.predict_proba(B[feats])[:, 1]
    idx = select(B["dup"], sB, thr)
    t_pred = time.time() - t
    res.update({"model": kind, "threshold_from_train_cv": thr, "train_oof_f1_at_threshold": m_oof["f1_vs_annotations"],
                "n_train_pos": int((A["y"] == 1).sum()), "n_train_neg": int((A["y"] == 0).sum()),
                "n_train_ignored": int((A["y"] == -1).sum()),
                "timing_s": {"blocked_cv": t_cv, "fit": t_fit, "predict_and_select": t_pred}})
    return res, model, sB, idx, oof


REC_K, REC_SEED = 80, 0
from barnacles.ml import DUP_FRAC  # noqa: E402  (1.5, see barnacles/ml.py)


def recommended_point(B, score, is_prob, k=REC_K, seed=REC_SEED):
    """The single, deterministic operating point used for figures and the
    review prototype: count-matched few-shot calibration, k labels, seed 0
    (one draw of the 20 summarized in operating_point)."""
    oracle = (B["y"] == 1).astype(int)
    rng = np.random.default_rng(1000 + seed)
    idx, w, strat = cal.stratified_sample(score, k, rng)
    ab = cal.fit_platt(score[idx], oracle[idx], w, is_prob)
    n_hat = float(cal.calibrated_prob(score, ab, is_prob).sum())
    det, thr = select_count(B["dup"], score, int(round(n_hat)))
    ht, se = cal.horvitz_thompson_total(oracle[idx], w, strat[idx])
    return det, {"method": f"count-matched few-shot calibration, k={k}, seed={seed}",
                 "threshold": thr, "platt_a_b": list(ab), "n_hat": n_hat, "ht_estimate": ht, "ht_se": se,
                 "sample": [{"index": int(i), "label": int(oracle[i]), "weight": float(wi), "stratum": int(strat[i])}
                            for i, wi in zip(idx, w)]}


def operating_points(B, scorers, ks=(20, 40, 80), seeds=20):
    """Few-shot review calibration (simulated reviewer = annotations) and
    label-free Otsu thresholds, for each scorer on test photo B."""
    oracle = (B["y"] == 1).astype(int)          # ambiguous (-1) counts as "not a new opening"
    out = {"simulated_reviewer": "candidate matched one-to-one to an annotation", "n_candidate_positives": int(oracle.sum())}
    for name, (score, is_prob) in scorers.items():
        res = {}
        t_otsu = cal.otsu_threshold(score, is_prob)
        m, _ = score_detections(B, select(B["dup"], score, t_otsu))
        res["label_free_otsu"] = {"threshold": t_otsu, "metrics": m}
        for k in ks:
            rows = []
            for s in range(seeds):
                rng = np.random.default_rng(1000 + s)
                idx, w, strat = cal.stratified_sample(score, k, rng)
                ab = cal.fit_platt(score[idx], oracle[idx], w, is_prob)
                t = cal.threshold_from_platt(ab, is_prob)
                ht, se = cal.horvitz_thompson_total(oracle[idx], w, strat[idx])
                if t is None:
                    rows.append({"failed": True, "ht_total": ht, "ht_se": se})
                    continue
                det = select(B["dup"], score, t)
                mm, _ = evaluate(B["cxy"][det], B["gt"], B["R"])
                # count-matched rule: accept the top N, N = sum of calibrated probabilities
                n_hat = float(cal.calibrated_prob(score, ab, is_prob).sum())
                det_c, t_c = select_count(B["dup"], score, int(round(n_hat)))
                mc, _ = evaluate(B["cxy"][det_c], B["gt"], B["R"])
                rows.append({"threshold": t, "n_detections": mm["n_detections"], "f1": mm["f1_vs_annotations"],
                             "recall": mm["annotated_recall"], "precision_lb": mm["annotation_precision_lower_bound"],
                             "ht_total": ht, "ht_se": se, "n_hat_calibrated": n_hat,
                             "cm_threshold": t_c, "cm_n_detections": mc["n_detections"], "cm_f1": mc["f1_vs_annotations"],
                             "cm_recall": mc["annotated_recall"], "cm_precision_lb": mc["annotation_precision_lower_bound"]})
            ok = [r_ for r_ in rows if not r_.get("failed")]
            summ = lambda key: (float(np.mean([r_[key] for r_ in ok])), float(np.std([r_[key] for r_ in ok]))) if ok else (None, None)
            res[f"few_shot_k{k}"] = {"n_labels": k, "n_seeds": seeds, "n_failed_fits": len(rows) - len(ok),
                                     "f1_mean_sd": summ("f1"), "recall_mean_sd": summ("recall"),
                                     "precision_lb_mean_sd": summ("precision_lb"),
                                     "n_detections_mean_sd": summ("n_detections"),
                                     "count_matched": {"f1_mean_sd": summ("cm_f1"), "recall_mean_sd": summ("cm_recall"),
                                                       "precision_lb_mean_sd": summ("cm_precision_lb"),
                                                       "n_detections_mean_sd": summ("cm_n_detections"),
                                                       "n_hat_mean_sd": summ("n_hat_calibrated")},
                                     "ht_count_estimate_mean_sd": (float(np.mean([r_["ht_total"] for r_ in rows])),
                                                                   float(np.std([r_["ht_total"] for r_ in rows]))),
                                     "ht_se_median": float(np.median([r_["ht_se"] for r_ in rows])),
                                     "runs": rows}
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="phase1_output/annotations.json")
    ap.add_argument("--data", default=str(data_dir()), help="folder with the challenge photos and masks (default: data/ or $BARNACLE_DATA)")
    ap.add_argument("--out", default="phase3_output")
    args = ap.parse_args()
    out = Path(args.out)
    global CACHE
    CACHE = out / "cache"
    CACHE.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "galleries").mkdir(exist_ok=True)
    phase1 = json.load(open(args.phase1))
    timings = {}
    P = {e["file"]: prepare(e, Path(args.data), timings) for e in phase1["images"]}
    files = list(P)
    report = {"protocol": __doc__, "candidate_params": CandidateParams().to_dict(), "duplicate_frac": DUP_FRAC,
              "images": {}, "directions": {}}

    for f in files:
        p = P[f]
        m_all, _ = evaluate(p["cxy"], p["gt"], p["R"])
        report["images"][f] = {
            "n_annotations": len(p["gt"]), "n_candidates": len(p["cxy"]),
            "candidate_recall_ceiling": m_all["annotated_recall"],
            "match_radius_px": p["R"], "median_opening_diameter_px": p["diam"],
            "labels": {"positive": int((p["y"] == 1).sum()), "negative": int((p["y"] == 0).sum()),
                       "ignored": int((p["y"] == -1).sum())},
            "density_strata": {"radius_px": p["density_radius_px"], "cuts_annotations_within_radius": p["density_cuts"]},
            "timing_s": timings[p["name"]]}

    review_payload = {}
    curves, points, strata_tab = {}, {}, {}
    for A_f, B_f in ((files[0], files[1]), (files[1], files[0])):
        A, B = P[A_f], P[B_f]
        key = f"train {A['name']} -> test {B['name']}"
        print("direction", key, flush=True)
        d = {}
        # --- OpenCV baseline: threshold transferred from A
        t = time.time()
        thr_cv, _ = best_threshold(A["dup"], A["cands"]["score"], A["gt"], A["R"])
        idx_cv = select(B["dup"], B["cands"]["score"], thr_cv)
        t_cv = time.time() - t
        m_cv, r_cv = score_detections(B, idx_cv)
        d["opencv_baseline"] = {"threshold_from_train_image": thr_cv, "metrics": m_cv,
                                "timing_s": {"threshold_search_and_select": t_cv}}
        # --- ML
        res, model, sB, idx_ml, oofA = train_direction(A, B, out, timings)
        m_ml, r_ml = score_detections(B, idx_ml)
        d["ml_classifier"] = {**res, "metrics": m_ml}
        with open(out / "models" / f"model_trained_on_{A['name']}.pkl", "wb") as fh:
            pickle.dump({"model": model, "kind": res["model"], "threshold": res["threshold_from_train_cv"],
                         "feature_names": [n for n in A["feat_names"] if not n.startswith("color_")],
                         "candidate_params": CandidateParams().to_dict(), "seed": SEED}, fh)
        # --- secondary: oracle thresholds on the test photo (optimistic)
        thr_o_cv, m_o_cv = best_threshold(B["dup"], B["cands"]["score"], B["gt"], B["R"])
        thr_o_ml, m_o_ml = best_threshold(B["dup"], sB, B["gt"], B["R"])
        d["oracle_threshold_on_test_photo"] = {
            "note": "threshold picked on the test photo itself: optimistic, not a valid estimate",
            "opencv_baseline": m_o_cv, "ml_classifier": m_o_ml}
        c_cv = precision_recall_curve_vs_annotations(B["dup"], B["cands"]["score"], B["gt"], B["R"])
        c_ml = precision_recall_curve_vs_annotations(B["dup"], sB, B["gt"], B["R"])
        d["ap_vs_annotations"] = {"opencv_baseline": average_precision_vs_annotations(c_cv),
                                  "ml_classifier": average_precision_vs_annotations(c_ml)}
        # --- operating point on the test photo without its annotations
        d["operating_point"] = operating_points(B, {"opencv_baseline": (B["cands"]["score"], False),
                                                    "ml_classifier": (sB, True)})
        idx_cv_r, rp_cv = recommended_point(B, B["cands"]["score"], False)
        idx_ml_r, rp_ml = recommended_point(B, sB, True)
        m_cv_r, r_cv_r = score_detections(B, idx_cv_r)
        m_ml_r, r_ml_r = score_detections(B, idx_ml_r)
        d["recommended_point"] = {"opencv_baseline": {**{k_: v_ for k_, v_ in rp_cv.items() if k_ != "sample"}, "metrics": m_cv_r},
                                  "ml_classifier": {**{k_: v_ for k_, v_ in rp_ml.items() if k_ != "sample"}, "metrics": m_ml_r}}
        # --- colour ablation (reported only)
        res_c, _, sBc, idx_c, _ = train_direction(A, B, out, timings, feats="Xc", kinds=(res["model"],))
        m_c, _ = score_detections(B, idx_c)
        c_c = precision_recall_curve_vs_annotations(B["dup"], sBc, B["gt"], B["R"])
        d["ablation_with_colour"] = {"model": res_c["model"], "metrics": m_c,
                                     "ap_vs_annotations": average_precision_vs_annotations(c_c)}
        report["directions"][key] = d
        curves[f"test on {B['name']} (trained on {A['name']})"] = {"OpenCV baseline": c_cv, "ML classifier": c_ml}
        points[f"test on {B['name']} (trained on {A['name']})"] = {
            "OpenCV baseline @ transferred threshold": (m_cv["annotated_recall"], m_cv["annotation_precision_lower_bound"], "transferred"),
            "ML classifier @ transferred threshold": (m_ml["annotated_recall"], m_ml["annotation_precision_lower_bound"], "transferred"),
            "OpenCV baseline @ 80-label calibration": (m_cv_r["annotated_recall"], m_cv_r["annotation_precision_lower_bound"], "calibrated"),
            "ML classifier @ 80-label calibration": (m_ml_r["annotated_recall"], m_ml_r["annotation_precision_lower_bound"], "calibrated")}
        strata_tab[f"{B['name']} OpenCV"] = m_cv_r["by_stratum"]
        strata_tab[f"{B['name']} ML"] = m_ml_r["by_stratum"]

        # --- visuals for test photo B
        names = ["OpenCV baseline", "ML classifier"]
        for tag, sets in (("calibrated", ((names[0], idx_cv_r, r_cv_r, m_cv_r), (names[1], idx_ml_r, r_ml_r, m_ml_r))),
                          ("transferred", ((names[0], idx_cv, r_cv, m_cv), (names[1], idx_ml, r_ml, m_ml)))):
            for nm, idx, (pairs, ud, mg), m in sets:
                how = "80-label calibration" if tag == "calibrated" else "threshold from training photo"
                ttl = (f"{B['file']}, {nm} trained on {A['file']}, {how}: {m['n_detections']} detections, "
                       f"{m['n_matched']} of {m['n_annotations']} annotations matched")
                ov = pv.outcome_overlay(B["img"], B["frame"], B["cxy"][idx], B["gt"], pairs, ud, mg, title=ttl)
                ov = np.concatenate([ov, pv.legend_strip(ov.shape[1])], 0)
                cv2.imwrite(str(out / f"{B['name']}_{'opencv' if nm == names[0] else 'ml'}_{tag}_outcomes.png"), ov)
        idx_cv, r_cv, idx_ml, r_ml = idx_cv_r, r_cv_r, idx_ml_r, r_ml_r   # figures below use the calibrated point
        # zoom windows: densest and sparsest annotated 10x10 cells
        rr, cc, _ = assign(B["gt_uv"], GridSpec(10, 10))
        cnt = np.bincount(rr * 10 + cc, minlength=100)
        dense_k, sparse_k = int(np.argmax(cnt)), int(np.argmin(np.where(cnt > 0, cnt, 999)))
        centers = [((k % 10 + 0.5) / 10, (k // 10 + 0.5) / 10) for k in (dense_k, sparse_k)]
        half = int(1.4 * 0.1 * np.mean([np.linalg.norm(B["frame"].corners[1] - B["frame"].corners[0]),
                                       np.linalg.norm(B["frame"].corners[3] - B["frame"].corners[0])]) / 2)
        zp = pv.zoom_pair(B["img"], B["frame"],
                          [(B["cxy"][idx_cv], B["gt"], *r_cv), (B["cxy"][idx_ml], B["gt"], *r_ml)],
                          centers, half, names)
        cv2.imwrite(str(out / f"{B['name']}_comparison_zoom.png"), zp)
        # failure galleries (ML): highest-scoring unmatched detections, and missed annotations
        pairs, ud, mg = r_ml
        ud = np.array(ud, int)
        top = ud[np.argsort(-sB[idx_ml][ud])][:32] if len(ud) else ud
        g = pv.gallery(B["img"], B["cxy"][idx_ml][top], [f"u{k} s={sB[idx_ml][i]:.2f}" for k, i in enumerate(top)], B["gt"])
        if g is not None:
            cv2.imwrite(str(out / "galleries" / f"{B['name']}_ml_top_unmatched.png"), g)
        rng = np.random.default_rng(SEED)
        mg_s = rng.choice(mg, min(32, len(mg)), replace=False) if len(mg) else []
        g = pv.gallery(B["img"], B["gt"][mg_s], [f"m{k}" for k in range(len(mg_s))], B["gt"])
        if g is not None:
            cv2.imwrite(str(out / "galleries" / f"{B['name']}_ml_missed_sample.png"), g)

        # --- review payload for Phase 4 (cross-image predictions: honest errors)
        ml_set, cv_set = set(idx_ml_r.tolist()), set(idx_cv_r.tolist())
        review_payload[B["file"]] = {
            "trained_on": A["file"], "ml_model": res["model"],
            "ml_threshold_transferred": res["threshold_from_train_cv"], "cv_threshold_transferred": thr_cv,
            "recommended_ml": rp_ml, "recommended_cv": {k_: v_ for k_, v_ in rp_cv.items() if k_ != "sample"},
            "candidates": [{"id": f"{B['name']}_c{i:05d}", "x": float(B["cxy"][i, 0]), "y": float(B["cxy"][i, 1]),
                            "u": float(B["uv"][i, 0]), "v": float(B["uv"][i, 1]),
                            "radius_px": float(B["radius_px"][i]), "ml_score": float(sB[i]),
                            "cv_score": float(B["cands"]["score"][i]),
                            "ml_accepted": i in ml_set, "cv_accepted": i in cv_set}
                           for i in range(len(B["cxy"]))],
        }
        # within-photo blocked CV (secondary): B's own OOF, threshold tuned in-sample
        res_B_self = {}
        for k in KINDS:
            oofB, _ = oof_scores(B, "X", k)
            thr_s, m_s = best_threshold(B["dup"], oofB, B["gt"], B["R"])
            res_B_self[k] = {"metrics": m_s, "threshold": thr_s,
                             "ap": average_precision_vs_annotations(
                                 precision_recall_curve_vs_annotations(B["dup"], oofB, B["gt"], B["R"]))}
        report["images"][B["file"]]["within_photo_blocked_cv"] = {
            "note": "2x2 spatial blocks with a 0.03 buffer; threshold tuned on the same out-of-fold scores. "
                    "Measures within-photo separability, NOT generalization to new photos.", **res_B_self}

    pv.pr_figure(curves, points, out / "pr_curves.png")
    pv.strata_figure(strata_tab, out / "density_strata.png")
    jdump(report, out / "phase3_report.json")
    jdump(review_payload, out / "review_candidates.json")
    # compact console summary
    for key, d in report["directions"].items():
        for meth in ("opencv_baseline", "ml_classifier"):
            m = d[meth]["metrics"]
            print(f"{key} | {meth:16s} | det {m['n_detections']:5d} | matched {m['n_matched']:4d}/{m['n_annotations']} "
                  f"| recall {m['annotated_recall']:.3f} | prec_lb {m['annotation_precision_lower_bound']:.3f} "
                  f"| F1 {m['f1_vs_annotations']:.3f} | med dist {m['match_dist_median_px']:.1f}px "
                  f"| cell MAE {m['grid_10x10']['mean_abs_cell_diff']:.1f}")
        print(f"   AP: {d['ap_vs_annotations']} | model {d['ml_classifier']['model']} "
              f"| oracle F1 cv {d['oracle_threshold_on_test_photo']['opencv_baseline']['f1_vs_annotations']:.3f} "
              f"ml {d['oracle_threshold_on_test_photo']['ml_classifier']['f1_vs_annotations']:.3f} "
              f"| colour ablation F1 {d['ablation_with_colour']['metrics']['f1_vs_annotations']:.3f} AP {d['ablation_with_colour']['ap_vs_annotations']:.3f}")
        op = d["operating_point"]
        for meth in ("opencv_baseline", "ml_classifier"):
            o = op[meth]
            print(f"   {meth}: otsu F1 {o['label_free_otsu']['metrics']['f1_vs_annotations']:.3f} "
                  f"n {o['label_free_otsu']['metrics']['n_detections']} | " +
                  " | ".join(f"k{k}: p.5 F1 {o[f'few_shot_k{k}']['f1_mean_sd'][0]:.3f} n {o[f'few_shot_k{k}']['n_detections_mean_sd'][0]:.0f}; "
                             f"count-matched F1 {o[f'few_shot_k{k}']['count_matched']['f1_mean_sd'][0]:.3f}+-{o[f'few_shot_k{k}']['count_matched']['f1_mean_sd'][1]:.3f} "
                             f"n {o[f'few_shot_k{k}']['count_matched']['n_detections_mean_sd'][0]:.0f}+-{o[f'few_shot_k{k}']['count_matched']['n_detections_mean_sd'][1]:.0f}; "
                             f"HT {o[f'few_shot_k{k}']['ht_count_estimate_mean_sd'][0]:.0f}+-{o[f'few_shot_k{k}']['ht_count_estimate_mean_sd'][1]:.0f}"
                             for k in (20, 40, 80)) + f" (candidate positives {op['n_candidate_positives']})")


if __name__ == "__main__":
    main()
