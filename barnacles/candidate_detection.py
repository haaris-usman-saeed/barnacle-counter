"""Classical candidate generation for barnacle openings.

Measured appearance (scripts/explore_candidates.py, TECHNICAL_REPORT.md):
  img1  grey operculum, outlined by a thin dark margin, inside a bright
        white shell. A bright-blob LoG fires on the white shell walls next
        to the opening, not on it (recall 37% at best).
  img2  the operculum plates are the brightest spot; the surround is dark.
        A bright-blob LoG works (77% recall) but misses dimmer openings.
Two complementary, hand-designed responses are combined:
  ring  centre-ring-surround contrast: mean(inner disk) and mean(outer
        annulus) minus mean(ring at the opening margin). Fires where a dark
        margin encloses a lighter centre (img1 openings).
  LoG   scale-normalized bright-blob Laplacian of Gaussian (img2 openings).
Both are z-scaled and summed; scale-space peaks of the sum, after
non-maximum suppression, are the candidates. Dark-blob LoG and Hough
circles were tried and rejected. This generator is tuned for recall: it
returns many more candidates than barnacles, and the classifier
(barnacles.ml) does the selection.

All processing happens in the rectified frame (barnacles.rectify), so the
blob scales are fractions of the frame side and do not depend on camera
resolution.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class CandidateParams:
    radius_min_frac: float = 0.0025  # opening radius range as a fraction of the frame side
    radius_max_frac: float = 0.0120  # (0.0025 to 0.012 of 1600 px = 4 to 19 px)
    n_scales: int = 9
    bg_sigma_frac: float = 0.03      # background removal scale for contrast normalization
    threshold: float = 0.0           # keep peaks of the combined score above this
    nms_frac: float = 0.8            # suppress weaker peaks closer than nms_frac * mean radius

    def to_dict(self):
        return asdict(self)


def normalize(gray, size, bg_sigma_frac):
    """Local contrast normalization: subtract a large-scale background and
    divide by the local standard deviation. Removes glare and exposure
    gradients (img2 is wet and unevenly lit)."""
    g = gray.astype(np.float32) / 255.0
    s = bg_sigma_frac * size
    mu = cv2.GaussianBlur(g, (0, 0), s)
    var = cv2.GaussianBlur((g - mu) ** 2, (0, 0), s)
    return (g - mu) / np.sqrt(var + 1e-4)


def log_stack(img, sigmas, polarity="bright"):
    sign = 1.0 if polarity == "bright" else -1.0
    return np.stack([-sign * s ** 2 * ndi.gaussian_laplace(img, s) for s in sigmas])


def nms(xy, r, score, frac):
    """Greedy suppression: keep the highest score, drop anything whose
    center is closer than frac * mean radius of the pair."""
    order = np.argsort(-score)
    keep = []
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    removed = np.zeros(len(xy), bool)
    rmax = r.max() if len(r) else 0
    for i in order:
        if removed[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(xy[i], frac * rmax):
            if j != i and not removed[j]:
                if np.hypot(*(xy[i] - xy[j])) < frac * 0.5 * (r[i] + r[j]):
                    removed[j] = True
    return np.array(sorted(keep), int)


def _disk_mean(img, rho):
    R = int(np.ceil(rho))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    k = (np.hypot(xx, yy) <= rho).astype(np.float32)
    return cv2.filter2D(img, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT), float(k.sum())


def _annulus_mean(img, r1, r2):
    m2, a2 = _disk_mean(img, r2)
    m1, a1 = _disk_mean(img, r1)
    return (m2 * a2 - m1 * a1) / (a2 - a1)


def ring_stack(z, radii):
    """Centre-ring-surround contrast at each radius (positive = dark ring)."""
    out = []
    for r in radii:
        inner, _ = _disk_mean(z, 0.6 * r)
        ring = _annulus_mean(z, 0.75 * r, 1.25 * r)
        outer = _annulus_mean(z, 1.4 * r, 2.0 * r)
        out.append(0.5 * (inner + outer) - ring)
    return np.stack(out)


def score_maps(rect, p: CandidateParams = CandidateParams()):
    gray = cv2.cvtColor(rect.image, cv2.COLOR_BGR2GRAY)
    z = normalize(gray, rect.size, p.bg_sigma_frac)
    radii = np.geomspace(p.radius_min_frac, p.radius_max_frac, p.n_scales) * rect.size
    ring = ring_stack(z, radii)
    log = log_stack(z, radii / np.sqrt(2), "bright")
    combined = ring / ring.std() + log / log.std()
    return {"z": z, "radii": radii, "ring": ring, "log": log, "combined": combined}


def detect_candidates(rect, p: CandidateParams = CandidateParams(), maps=None):
    """Scale-space peaks of the combined score inside the frame.

    Returns a dict of arrays in rectified coordinates: x, y, radius, level
    (scale index), score (combined), ring_score, log_score.
    """
    m = maps if maps is not None else score_maps(rect, p)
    S = m["combined"]
    mx = ndi.maximum_filter(S, size=(3, 5, 5), mode="nearest")
    pk = (S == mx) & (S > p.threshold)
    pk[0] = pk[-1] = False   # interior scales only: the true scale is bracketed
    k, yy, xx = np.nonzero(pk)
    xy = np.c_[xx, yy].astype(float)
    uv = rect.rect_to_uv(xy)
    ins = (uv >= 0).all(1) & (uv <= 1).all(1)
    k, xy = k[ins], xy[ins]
    yy, xx = yy[ins], xx[ins]
    sc = S[k, yy, xx]
    r = m["radii"][k]
    keep = nms(xy, r, sc, p.nms_frac) if len(xy) else np.array([], int)
    k, yy, xx = k[keep], yy[keep], xx[keep]
    return {"x": xy[keep, 0], "y": xy[keep, 1], "radius": r[keep], "level": k,
            "score": sc[keep], "ring_score": m["ring"][k, yy, xx], "log_score": m["log"][k, yy, xx]}


def to_records(cands, rect, image_name, prefix="cand"):
    """Candidate dicts with ids, raw-image pixel coordinates and (u, v)."""
    xy = np.c_[cands["x"], cands["y"]]
    img_xy = rect.rect_to_image(xy) if len(xy) else np.zeros((0, 2))
    uv = rect.rect_to_uv(xy) if len(xy) else np.zeros((0, 2))
    s = rect.px_per_rect_px()
    recs = []
    for i in range(len(xy)):
        recs.append({"id": f"{image_name}_{prefix}{i:05d}",
                     "x": float(img_xy[i, 0]), "y": float(img_xy[i, 1]),
                     "u": float(uv[i, 0]), "v": float(uv[i, 1]),
                     "rect_x": float(xy[i, 0]), "rect_y": float(xy[i, 1]),
                     "radius_px": float(cands["radius"][i] * s),
                     "bbox_xywh": [float(img_xy[i, 0] - cands["radius"][i] * s),
                                   float(img_xy[i, 1] - cands["radius"][i] * s),
                                   float(2 * cands["radius"][i] * s), float(2 * cands["radius"][i] * s)],
                     "cv_score": float(cands["score"][i]),
                     "ring_score": float(cands["ring_score"][i]),
                     "log_score": float(cands["log_score"][i])})
    return recs


def hough_candidates(rect, maps, p: CandidateParams = CandidateParams()):
    """Hough-circle proposals over the same frame-relative radius range,
    scored by sampling the ring/LoG maps at the nearest scale, so they can be
    classified with exactly the same features as the main generator."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rect.image, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    radii = maps["radii"]
    rmin, rmax = radii[0], radii[-1]
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=max(2.0, 1.6 * rmin), param1=80,
                         param2=14, minRadius=int(np.floor(rmin)), maxRadius=int(np.ceil(rmax)))
    c = np.zeros((0, 3)) if c is None else c[0].astype(float)
    uv = rect.rect_to_uv(c[:, :2]) if len(c) else np.zeros((0, 2))
    c = c[(uv >= 0).all(1) & (uv <= 1).all(1)] if len(c) else c
    k = np.abs(radii[None, :] - c[:, 2:3]).argmin(1) if len(c) else np.zeros(0, int)
    k = np.clip(k, 1, len(radii) - 2)
    yy = np.clip(np.round(c[:, 1]).astype(int), 0, maps["combined"].shape[1] - 1)
    xx = np.clip(np.round(c[:, 0]).astype(int), 0, maps["combined"].shape[2] - 1)
    return {"x": c[:, 0], "y": c[:, 1], "radius": radii[k], "level": k,
            "score": maps["combined"][k, yy, xx], "ring_score": maps["ring"][k, yy, xx],
            "log_score": maps["log"][k, yy, xx]}


def merge_candidates(a, b, frac=0.5):
    """Union of two candidate sets: keep all of `a`, add the members of `b`
    farther than frac x radius from every member of `a`."""
    from scipy.spatial import cKDTree
    if len(b["x"]) == 0:
        return dict(a), np.zeros(len(a["x"]), bool)
    d, _ = cKDTree(np.c_[a["x"], a["y"]]).query(np.c_[b["x"], b["y"]])
    add = d > frac * b["radius"]
    out = {k: np.concatenate([a[k], b[k][add]]) for k in a}
    return out, np.r_[np.zeros(len(a["x"]), bool), np.ones(add.sum(), bool)]
