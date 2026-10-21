"""One-to-one matching of detections to annotation centers, and metrics.

Annotations are incomplete, so the vocabulary here is deliberate:
  annotated recall      matched annotations / annotations
  annotation precision  matched detections / detections. A LOWER BOUND on
                        true precision: an unmatched detection may be a real,
                        unannotated barnacle.
  unmatched detections  detections with no annotation within the radius.
                        Not called false positives.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def match(pred_xy, gt_xy, radius):
    """Minimum-total-distance one-to-one matching with a distance cap.

    Pairs farther than `radius` are never matched. Solved exactly with the
    Hungarian algorithm on each connected component of the
    "within radius" graph, which keeps it fast for thousands of points.
    Returns (pairs [(i_pred, j_gt, dist)], unmatched_pred_idx, unmatched_gt_idx).
    """
    pred_xy = np.asarray(pred_xy, float).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, float).reshape(-1, 2)
    P, G = len(pred_xy), len(gt_xy)
    if P == 0 or G == 0:
        return [], list(range(P)), list(range(G))
    tp, tg = cKDTree(pred_xy), cKDTree(gt_xy)
    sp = tp.sparse_distance_matrix(tg, radius, output_type="coo_matrix")
    pairs = []
    if sp.nnz:
        # bipartite graph: nodes 0..P-1 preds, P..P+G-1 annotations
        rows, cols = sp.row, sp.col + P
        A = coo_matrix((np.ones(sp.nnz), (rows, cols)), shape=(P + G, P + G))
        ncomp, lab = connected_components(A, directed=False)
        comp_of_edge = lab[rows]
        order = np.argsort(comp_of_edge, kind="stable")
        bounds = np.flatnonzero(np.diff(comp_of_edge[order])) + 1
        for idx in np.split(order, bounds):
            pi, gi, d = sp.row[idx], sp.col[idx], sp.data[idx]
            up, ug = np.unique(pi), np.unique(gi)
            C = np.full((len(up), len(ug)), 1e9)
            C[np.searchsorted(up, pi), np.searchsorted(ug, gi)] = d
            r, c = linear_sum_assignment(C)
            for a, b in zip(r, c):
                if C[a, b] < 1e9:
                    pairs.append((int(up[a]), int(ug[b]), float(C[a, b])))
    mp = {p for p, _, _ in pairs}
    mg = {g for _, g, _ in pairs}
    return (pairs, [i for i in range(P) if i not in mp], [j for j in range(G) if j not in mg])


def local_density(points_xy, query_xy, radius):
    """Number of `points_xy` within `radius` of each query point."""
    if len(points_xy) == 0:
        return np.zeros(len(query_xy), int)
    t = cKDTree(np.asarray(points_xy, float))
    return np.array([len(x) for x in t.query_ball_point(np.asarray(query_xy, float), radius)])


def evaluate(pred_xy, gt_xy, radius, density_radius=None, strata=None):
    """Matching metrics. `strata`: optional (gt_labels, pred_labels, names)
    for per-stratum recall and unmatched rates (e.g. dense vs sparse)."""
    pairs, up, ug = match(pred_xy, gt_xy, radius)
    d = np.array([p[2] for p in pairs]) if pairs else np.array([])
    n_p, n_g, n_m = len(pred_xy), len(gt_xy), len(pairs)
    out = {
        "n_annotations": n_g, "n_detections": n_p, "n_matched": n_m,
        "n_missed_annotations": len(ug), "n_unmatched_detections": len(up),
        "annotated_recall": n_m / n_g if n_g else float("nan"),
        "annotation_precision_lower_bound": n_m / n_p if n_p else float("nan"),
        "f1_vs_annotations": 2 * n_m / (n_p + n_g) if (n_p + n_g) else float("nan"),
        "count_minus_annotations": n_p - n_g,
        "count_ratio": n_p / n_g if n_g else float("nan"),
        "match_dist_median_px": float(np.median(d)) if len(d) else float("nan"),
        "match_dist_p90_px": float(np.percentile(d, 90)) if len(d) else float("nan"),
        "match_radius_px": float(radius),
    }
    if strata is not None:
        g_lab, p_lab, names = strata
        g_lab, p_lab = np.asarray(g_lab), np.asarray(p_lab)
        matched_g = np.zeros(n_g, bool)
        matched_g[[g for _, g, _ in pairs]] = True
        matched_p = np.zeros(n_p, bool)
        matched_p[[p for p, _, _ in pairs]] = True
        out["by_stratum"] = {}
        for k, nm in enumerate(names):
            gs, ps = g_lab == k, p_lab == k
            out["by_stratum"][nm] = {
                "n_annotations": int(gs.sum()), "n_detections": int(ps.sum()),
                "annotated_recall": float(matched_g[gs].mean()) if gs.any() else float("nan"),
                "unmatched_detection_rate": float((~matched_p[ps]).mean()) if ps.any() else float("nan"),
            }
    return out, (pairs, up, ug)
