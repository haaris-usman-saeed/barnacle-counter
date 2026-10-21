"""Phase 4: review signals, correction replay and export.

The review prototype (review_viewer.py) and this module implement the SAME
rules, so an exported session can be replayed here and checked:

  model set   candidates with ml_score >= threshold, then greedy duplicate
              suppression in descending score (ties: lower index first)
              using the precomputed duplicate pairs.
  counted     (model set minus "remove" overrides) plus "keep" overrides
              plus reviewer-added points.

Review signals are kept in three separate families, because they answer
different questions:
  detection confidence   is this particular detection likely right?
                         (ambiguous calibrated score, a competing
                         overlapping candidate, classical/ML disagreement,
                         crowding, poor local clarity, touching the frame)
  detection density      how many detections a cell has, relative to the photo
  annotation coverage    whether the source masks annotated this area at all
                         (only exists for the challenge photos; a new upload
                         has no annotations)
"""
import numpy as np
from scipy.spatial import cKDTree

from .grid import GridSpec, assign, cell_id

SCHEMA = "barnacle-review/1"
AMBIGUOUS = (0.25, 0.75)


def logit(s):
    s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def pcal(score, ab):
    return 1 / (1 + np.exp(-(ab[0] * logit(score) + ab[1])))


def neighbours_from_pairs(n, pairs):
    nb = [[] for _ in range(n)]
    for a, b in pairs:
        nb[a].append(b)
        nb[b].append(a)
    return nb


def model_set(score, thr, nbrs):
    """Greedy duplicate suppression over score >= thr (same rule as ml.DuplicateIndex)."""
    score = np.asarray(score, float)
    idx = np.flatnonzero(score >= thr)
    order = idx[np.argsort(-score[idx], kind="stable")]
    alive = np.zeros(len(score), bool)
    alive[idx] = True
    keep = []
    for i in order:
        if alive[i]:
            keep.append(int(i))
            for j in nbrs[i]:
                alive[j] = False
    return sorted(keep)


def static_signals(gray, frame, x, y, r, accepted, cv_score):
    """Signals that do not depend on the threshold slider."""
    n = len(x)
    # local clarity: grey-level standard deviation in a window of +/- 1.5 r
    integ = np.cumsum(np.cumsum(np.pad(gray.astype(np.float64), ((1, 0), (1, 0))), 0), 1)
    integ2 = np.cumsum(np.cumsum(np.pad(gray.astype(np.float64) ** 2, ((1, 0), (1, 0))), 0), 1)
    H, W = gray.shape
    h = np.maximum(2, np.round(1.5 * r)).astype(int)
    x0 = np.clip(np.round(x).astype(int) - h, 0, W - 1)
    x1 = np.clip(np.round(x).astype(int) + h, 1, W)
    y0 = np.clip(np.round(y).astype(int) - h, 0, H - 1)
    y1 = np.clip(np.round(y).astype(int) + h, 1, H)
    area = (x1 - x0) * (y1 - y0)
    s1 = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
    s2 = integ2[y1, x1] - integ2[y0, x1] - integ2[y1, x0] + integ2[y0, x0]
    contrast = np.sqrt(np.maximum(s2 / area - (s1 / area) ** 2, 0))
    acc = np.asarray(accepted, bool)
    low_clarity = contrast < np.percentile(contrast[acc], 10) if acc.any() else np.zeros(n, bool)
    # crowding among accepted detections
    dense = np.zeros(n, bool)
    if acc.sum() > 10:
        pts = np.c_[x, y]
        rad = 3 * np.median(r[acc])
        cnt = np.array([len(v) for v in cKDTree(pts[acc]).query_ball_point(pts, rad)])
        dense = cnt > np.percentile(cnt[acc], 95)
    near_frame = frame.signed_distance_px(np.c_[x, y]) < r
    cv_pct = (np.argsort(np.argsort(cv_score)) + 0.5) / n
    return {"contrast": contrast, "low_clarity": low_clarity, "dense": dense,
            "near_frame": near_frame, "cv_pct": cv_pct}


def replay(cands, nbrs, session, frame=None):
    """Recompute the corrected detections and counts from a session.

    cands: dict with arrays id, x, y, u, v, ml_score.  Returns a result dict
    shaped like the viewer's export "result" block."""
    ids = list(cands["id"])
    pos = {k: i for i, k in enumerate(ids)}
    op = session["operating_point"]
    thr = float(op.get("initial_threshold", op["threshold"]))   # actions below move it
    overrides, added = {}, {}
    for a in session["actions"]:
        act, k = a["action"], a.get("id")
        if act == "keep":
            overrides[k] = "keep"
        elif act == "remove":
            if k in added:
                del added[k]
            else:
                overrides[k] = "remove"
        elif act == "restore":
            overrides.pop(k, None)
        elif act == "add":
            added[k] = (float(a["x"]), float(a["y"]), float(a["u"]), float(a["v"]))
        elif act in ("set_threshold", "set_calibration"):
            thr = float(a["threshold"])
    mset = model_set(cands["ml_score"], thr, nbrs)
    counted = {i for i in mset if overrides.get(ids[i]) != "remove"}
    counted |= {pos[k] for k, v in overrides.items() if v == "keep" and k in pos}
    dets = []
    for i in sorted(counted):
        src = "reviewer_kept" if overrides.get(ids[i]) == "keep" else "model"
        if src == "reviewer_kept" and i not in set(mset):
            src = "reviewer_accepted_suggestion"
        dets.append({"id": ids[i], "x": float(cands["x"][i]), "y": float(cands["y"][i]),
                     "u": float(cands["u"][i]), "v": float(cands["v"][i]), "source": src})
    for k, (x, y, u, v) in added.items():
        dets.append({"id": k, "x": x, "y": y, "u": u, "v": v, "source": "reviewer_added"})
    g = session["grid"]
    spec = GridSpec(int(g["rows"]), int(g["cols"]))
    uv = np.array([[d["u"], d["v"]] for d in dets]) if dets else np.zeros((0, 2))
    rr, cc, ins = assign(uv, spec)
    per_cell = {}
    for d, a, b, i in zip(dets, rr, cc, ins):
        d["cell_id"] = cell_id(a, b) if i else None
        if i:
            per_cell[d["cell_id"]] = per_cell.get(d["cell_id"], 0) + 1
    mu = np.array([[cands["u"][i], cands["v"][i]] for i in mset]) if mset else np.zeros((0, 2))
    r2, c2, i2 = assign(mu, spec)
    model_cell = {}
    for a, b, i in zip(r2, c2, i2):
        if i:
            model_cell[cell_id(a, b)] = model_cell.get(cell_id(a, b), 0) + 1
    ms = set(mset)
    bd = {"selected_model": len(mset), "confirmed": 0, "removed": 0, "accepted_suggestions": 0,
          "rejected_suggestions": 0, "added": len(added)}
    for k, v in overrides.items():
        if k not in pos:
            continue
        m = pos[k] in ms
        bd[("confirmed" if m else "accepted_suggestions") if v == "keep" else ("removed" if m else "rejected_suggestions")] += 1
    bd["reviewed"] = bd["confirmed"] + bd["removed"] + bd["accepted_suggestions"] + bd["rejected_suggestions"]
    assert bd["selected_model"] - bd["removed"] + bd["accepted_suggestions"] + bd["added"] == len(dets)
    return {"threshold": thr, "model_count": len(mset), "corrected_count": len(dets), "breakdown": bd,
            "corrected_inside_frame": int(ins.sum()) if len(dets) else 0,
            "detections": sorted(dets, key=lambda d: d["id"]),
            "per_cell_corrected": dict(sorted(per_cell.items())),
            "per_cell_model": dict(sorted(model_cell.items()))}


def review_queue(score, thr, ab, nbrs, sig, counted=None, overrides=None, sugg_frac=0.25):
    """Items worth a reviewer's look, most likely counting errors first.

    Priority is the calibrated probability that the item is currently a
    counting error: 1 - p for a counted model detection (it may be wrong),
    p for a suggestion (a rejected candidate just below the threshold, not a
    duplicate of a counted detection, that may be a miss). Each static flag
    adds 0.05 as a tie-breaker. Flags are also returned as human-readable
    reasons. Mirrored in review_viewer.py.
    """
    s = np.asarray(score, float)
    overrides = overrides or {}
    mset = model_set(s, thr, nbrs)
    in_m = np.zeros(len(s), bool)
    in_m[mset] = True
    if counted is None:
        counted = in_m.copy()
    p = pcal(s, ab)
    covered = np.zeros(len(s), bool)
    for i in np.flatnonzero(counted):
        covered[nbrs[i]] = True
    below = np.flatnonzero((~counted) & (~covered) & (s < thr))
    below = below[np.argsort(-s[below], kind="stable")][:int(np.ceil(sugg_frac * max(len(mset), 1)))]
    sugg = np.zeros(len(s), bool)
    sugg[below] = True
    nflag = (sig["dense"].astype(int) + sig["low_clarity"] + sig["near_frame"] + (sig["cv_pct"] < 0.5))
    # a detection awaits review if a flag applies or its score is below 0.5 (the viewer also
    # flags ambiguity and competing overlaps, which only add items with p in [0.25, 0.75])
    strong = sig["dense"].astype(int) + sig["low_clarity"] + sig["near_frame"]
    det_item = in_m & counted & ((strong > 0) | (p < 0.5) | ((p >= AMBIGUOUS[0]) & (p <= AMBIGUOUS[1])))
    prio = np.where(det_item, 1 - p + 0.05 * nflag, np.where(sugg, p, -1.0))
    items = np.flatnonzero(prio >= 0)
    return items[np.lexsort((items, -prio[items]))], prio, sugg, in_m


def simulate_review(cands, nbrs, thr, ab, sig, gt_xy, radius, budget, rng):
    """Use the annotations as a stand-in reviewer to measure how many
    counting errors a review budget finds when following the queue versus
    checking items in random order. An error is a counted detection not
    matched to any annotation, or a suggestion that matches an annotation
    the model missed."""
    from .evaluation import match
    s = np.asarray(cands["ml_score"], float)
    order, prio, sugg, in_m = review_queue(s, thr, ab, nbrs, sig)
    mset = np.flatnonzero(in_m)
    xy = np.c_[cands["x"], cands["y"]]
    _, um, _ = match(xy[mset], gt_xy, radius)
    wrong = set(mset[um].tolist())
    sidx = np.flatnonzero(sugg)
    # a suggestion is right if it matches an annotation not matched by the model set
    pairs_m, _, ug = match(xy[mset], gt_xy, radius)
    pairs_s, _, _ = match(xy[sidx], gt_xy[ug], radius) if len(sidx) and len(ug) else ([], [], [])
    sugg_right = set(sidx[[a for a, _, _ in pairs_s]].tolist())
    is_err = lambda i: (i in wrong) or (i in sugg_right)
    rnd = rng.permutation(order)
    return {"budget": budget, "n_model": int(len(mset)), "n_model_unmatched": len(wrong),
            "n_suggestions": int(sugg.sum()), "n_suggestions_matching_annotations": len(sugg_right),
            "n_queue": int(len(order)),
            "errors_found_priority": int(sum(is_err(i) for i in order[:budget])),
            "errors_found_random": int(sum(is_err(i) for i in rnd[:budget]))}
