"""Candidate classification: labels, models, spatially blocked CV, thresholds.

Label policy (see TECHNICAL_REPORT.md for the reasoning and the audit):
  positive  candidate matched one-to-one to an annotation center (within
            half the image's median opening diameter).
  negative  unmatched candidate farther than that radius from every
            annotation: interstices, shell walls, off-centre peaks on an
            annotated barnacle, rock, wire. These are "unlabelled treated as
            negative". A by-eye audit of 96 random negatives (48 per photo,
            half near annotations, half in voids) found no clear live
            opening in img1 and at most 2 to 3 ambiguous ones in img2, so the
            label noise is small but not zero, and larger in img2.
  ignored   unmatched but within the radius of an annotation (lost the
            one-to-one assignment to a closer candidate).
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import match, evaluate

SEED = 0
# Two distinct openings cannot be closer than the sum of their radii (a shell
# wall separates them), so detections closer than 1.5 x their mean radius are
# treated as the same barnacle. Raised from 1.0 after the img2 error audit
# found duplicate detections on large openings (see TECHNICAL_REPORT.md).
DUP_FRAC = 1.5


def label_candidates(cand_xy, gt_xy, match_radius):
    """Return labels (1, 0, -1) and nearest-annotation distance per candidate.

    1   matched one-to-one to an annotation within match_radius
    0   unmatched and at least match_radius from every annotation
    -1  unmatched but within match_radius of an annotation (it lost the
        one-to-one assignment to a closer candidate): ambiguous, ignored
    """
    pairs, _, _ = match(cand_xy, gt_xy, match_radius)
    y = np.full(len(cand_xy), -1, int)
    d, _ = cKDTree(gt_xy).query(cand_xy)
    y[d >= match_radius] = 0
    y[[p for p, _, _ in pairs]] = 1
    return y, d


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                              l2_regularization=1.0, class_weight="balanced",
                                              random_state=SEED)
    raise ValueError(kind)


def spatial_blocks(uv, n=2):
    """Quadrant (n x n block) index of each point in normalized frame coords."""
    c = np.minimum((np.asarray(uv)[:, 0] * n).astype(int), n - 1)
    r = np.minimum((np.asarray(uv)[:, 1] * n).astype(int), n - 1)
    return r * n + c


def blocked_oof_scores(X, y, uv, kind, n=2, buffer=0.03):
    """Out-of-fold scores for EVERY candidate with spatially blocked folds.

    Each block is scored by a model trained on the other blocks, excluding
    training samples within `buffer` (normalized units) of the held-out
    block, so overlapping patches never straddle the train/test boundary.
    """
    blk = spatial_blocks(uv, n)
    oof = np.zeros(len(X))
    for b in range(n * n):
        test = blk == b
        r, c = divmod(b, n)
        u0, u1, v0, v1 = c / n - buffer, (c + 1) / n + buffer, r / n - buffer, (r + 1) / n + buffer
        near = (uv[:, 0] > u0) & (uv[:, 0] < u1) & (uv[:, 1] > v0) & (uv[:, 1] < v1)
        tr = (~near) & (y >= 0)
        m = make_model(kind).fit(X[tr], y[tr])
        oof[test] = m.predict_proba(X[test])[:, 1]
    return oof


class DuplicateIndex:
    """Precomputed "same barnacle" pairs for fast repeated suppression.

    Two detections are duplicates when their centers are closer than
    `frac` x the mean of their radii, i.e. one sits inside the other's
    opening. Built once per image; `select` then runs for any threshold.
    """

    def __init__(self, xy, radius, frac=1.0):
        self.xy = np.asarray(xy, float)
        self.r = np.asarray(radius, float)
        tree = cKDTree(self.xy)
        pairs = tree.query_pairs(frac * self.r.max(), output_type="ndarray") if len(xy) else np.zeros((0, 2), int)
        if len(pairs):
            d = np.hypot(*(self.xy[pairs[:, 0]] - self.xy[pairs[:, 1]]).T)
            pairs = pairs[d < frac * 0.5 * (self.r[pairs[:, 0]] + self.r[pairs[:, 1]])]
        self.nbrs = [[] for _ in range(len(self.xy))]
        for a, b in pairs:
            self.nbrs[a].append(b)
            self.nbrs[b].append(a)

    def suppress(self, idx, score):
        """Greedy over `idx` in descending score; returns kept indices."""
        idx = np.asarray(idx, int)
        alive = np.zeros(len(self.xy), bool)
        alive[idx] = True
        keep = []
        for i in idx[np.argsort(-score[idx], kind="stable")]:
            if not alive[i]:
                continue
            keep.append(i)
            for j in self.nbrs[i]:
                alive[j] = False
        return np.array(sorted(keep), int)


def suppress_duplicates(xy, score, radius, frac=1.0):
    """Stand-alone duplicate suppression (see DuplicateIndex)."""
    return DuplicateIndex(xy, radius, frac).suppress(np.arange(len(xy)), np.asarray(score, float))


def select(dup: DuplicateIndex, score, thr):
    """Indices of candidates with score >= thr after duplicate suppression."""
    idx = np.flatnonzero(score >= thr)
    return dup.suppress(idx, score) if len(idx) else idx


def best_threshold(dup, score, gt_xy, radius, n=60):
    """Threshold maximizing F1 against annotations on the image it is given."""
    best = (-1, None, None)
    for t in np.unique(np.quantile(score, np.linspace(0.0, 0.995, n))):
        idx = select(dup, score, t)
        m, _ = evaluate(dup.xy[idx], gt_xy, radius)
        if m["f1_vs_annotations"] > best[0]:
            best = (m["f1_vs_annotations"], float(t), m)
    return best[1], best[2]


def precision_recall_curve_vs_annotations(dup, score, gt_xy, radius, n=40):
    rows = []
    for t in np.unique(np.quantile(score, np.linspace(0.0, 0.995, n))):
        idx = select(dup, score, t)
        m, _ = evaluate(dup.xy[idx], gt_xy, radius)
        rows.append({"threshold": float(t), "n": len(idx), "recall": m["annotated_recall"],
                     "precision_lb": m["annotation_precision_lower_bound"]})
    return rows


def average_precision_vs_annotations(curve):
    """Area under the (recall, precision lower bound) curve, step-wise."""
    pts = sorted((c["recall"], c["precision_lb"]) for c in curve if c["n"] > 0)
    ap, prev_r = 0.0, 0.0
    for r, p in pts:
        best_p = max(pp for rr, pp in pts if rr >= r)
        ap += (r - prev_r) * best_p
        prev_r = r
    return ap


def select_count(dup: DuplicateIndex, score, n):
    """Highest-scoring detections, after duplicate suppression, totalling
    about n (binary search on the threshold; suppression is monotone enough
    in practice). Returns (indices, threshold)."""
    s = np.sort(np.unique(score))[::-1]
    if n <= 0:
        return np.array([], int), float(s[0]) + 1e-9
    lo, hi = 0, len(s) - 1
    best = (select(dup, score, s[hi]), float(s[hi]))
    while lo <= hi:
        mid = (lo + hi) // 2
        idx = select(dup, score, s[mid])
        if len(idx) >= n:
            best, hi = (idx, float(s[mid])), mid - 1
        else:
            lo = mid + 1
    return best
