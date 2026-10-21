"""Inner quadrat frame: detection, manual override, and normalized coordinates.

The frame is the opening bounded by the inner edges of the four wire bars
around the central cell. Detection uses raw pixels only (see select_opening);
annotations are never needed. Normalized coordinates (u, v) are defined by the
homography that sends the frame corners TL, TR, BR, BL to (0,0), (1,0),
(1,1), (0,1). u runs left to right along the frame, v runs top to bottom.
Because the mapping is defined by the corners, (u, v) does not depend on
image resolution: scale the image and the corners together and every
point keeps the same (u, v).

A homography is exact for a flat frame seen by a pinhole camera. It does not
model lens distortion; the side-fit residuals reported by `detect` measure
how much that matters for a given photo.
"""
from dataclasses import dataclass, field
import json

import cv2
import numpy as np
from scipy import ndimage as ndi

from .diagnostics import frame_mask

CORNER_NAMES = ("TL", "TR", "BR", "BL")
UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)


@dataclass
class Frame:
    corners: np.ndarray            # (4, 2) float64, TL TR BR BL, raw-image pixels
    image_size: tuple              # (width, height) the corners refer to
    source: str                    # "auto" or "manual"
    quality: dict = field(default_factory=dict)

    def __post_init__(self):
        self.corners = np.asarray(self.corners, np.float64).reshape(4, 2)
        self.H = cv2.getPerspectiveTransform(self.corners.astype(np.float32),
                                             UNIT_SQUARE.astype(np.float32)).astype(np.float64)
        self.H_inv = np.linalg.inv(self.H)

    @staticmethod
    def _apply(M, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        ph = np.c_[pts, np.ones(len(pts))] @ M.T
        return ph[:, :2] / ph[:, 2:3]

    def to_norm(self, xy):
        """Image pixels -> normalized frame coordinates (u, v)."""
        return self._apply(self.H, xy)

    def to_image(self, uv):
        """Normalized frame coordinates -> image pixels."""
        return self._apply(self.H_inv, uv)

    def scaled_to(self, width, height):
        """Same physical frame for a resized copy of the image."""
        sx, sy = width / self.image_size[0], height / self.image_size[1]
        return Frame(self.corners * [sx, sy], (width, height), self.source, dict(self.quality))

    def contains(self, xy):
        """Inside test in image space (independent of the homography)."""
        poly = self.corners.astype(np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                         for x, y in np.asarray(xy).reshape(-1, 2)])

    def signed_distance_px(self, xy):
        """Positive inside the frame, negative outside, in image pixels."""
        poly = self.corners.astype(np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True)
                         for x, y in np.asarray(xy).reshape(-1, 2)])

    def to_config(self):
        return {"corners_px": {n: [round(float(x), 2), round(float(y), 2)]
                               for n, (x, y) in zip(CORNER_NAMES, self.corners)},
                "image_size": list(self.image_size), "source": self.source}


def frame_from_config(entry: dict, actual_size: tuple) -> Frame:
    """Build a Frame from a config entry, rescaling if the image size differs."""
    c = np.array([entry["corners_px"][n] for n in CORNER_NAMES], np.float64)
    f = Frame(c, tuple(entry["image_size"]), entry.get("source", "manual"))
    if tuple(actual_size) != tuple(entry["image_size"]):
        f = f.scaled_to(*actual_size)
    return f


def load_config(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------- detection

def wire_mask(img, h_range=(75, 100), s_min=80, v_min=18):
    """Teal wire mask for frame detection, robust to exposure.

    Same hue/saturation band as Phase 1's frame_mask, but brightness is
    gain-normalized first (99th percentile of V mapped to 255), so an
    under-exposed photo does not lose the dark parts of the wire, and the
    morphology kernels scale with image size. Phase 1's frame_mask is left
    unchanged because Phase 1 and 2 diagnostics depend on it.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2].astype(np.float32)
    gain = 255.0 / max(float(np.percentile(v, 99)), 1.0)
    v = np.clip(v * gain, 0, 255)
    m = ((h >= h_range[0]) & (h <= h_range[1]) & (s_ >= s_min) & (v >= v_min)).astype(np.uint8)
    scale = np.hypot(*img.shape[:2]) / 4080.0      # 4080 = diagonal of the 3264 x 2448 photos
    k1 = max(3, int(round(9 * scale)) | 1)
    k2 = max(3, int(round(11 * scale)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k1, k1), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k2, k2), np.uint8))
    return m.astype(bool)


def select_opening(img, close_px=31, min_area_frac=0.01, min_rectangularity=0.9,
                   max_aspect=1.3, seed_xy=None):
    """Find the inner frame opening from raw pixels only.

    The quadrat's wire grid splits the photo into closed cells. The inner
    frame is the largest closed cell that is (a) not touching the image
    border, (b) nearly rectangular (area / min-area-rectangle area), and
    (c) roughly square. On both photos it is 7x to 9x larger than any
    other qualifying cell, which `selection.margin` reports. `seed_xy`
    is optional and only used to pick a cell when several qualify; it is
    never required.
    """
    wire = wire_mask(img).astype(np.uint8)
    close_px = max(5, int(round(close_px * np.hypot(*img.shape[:2]) / 4080.0)) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    sealed = cv2.morphologyEx(wire, cv2.MORPH_CLOSE, k).astype(bool)
    lab, n = ndi.label(~sealed)
    H, W = lab.shape
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    cands = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or areas[i] < min_area_frac * H * W:
            continue
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W:
            continue
        filled = ndi.binary_fill_holes(lab[sl] == i)
        cnts, _ = cv2.findContours(filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        rect = float(filled.sum() / max(w * h, 1))
        aspect = float(max(w, h) / max(min(w, h), 1))
        if rect >= min_rectangularity and aspect <= max_aspect:
            cands.append({"label": i, "area": int(filled.sum()), "rectangularity": rect,
                          "aspect": aspect, "slice": sl})
    if not cands:
        raise RuntimeError("no closed, square, rectangular wire cell found; "
                           "configure the frame corners manually")
    if seed_xy is not None:
        sx, sy = int(round(seed_xy[0])), int(round(seed_xy[1]))
        hit = [c for c in cands if c["label"] == lab[sy, sx]]
        cands = hit or cands
    cands.sort(key=lambda c: -c["area"])
    best = cands[0]
    opening = np.zeros((H, W), bool)
    opening[best["slice"]] = ndi.binary_fill_holes(lab[best["slice"]] == best["label"])
    margin = best["area"] / cands[1]["area"] if len(cands) > 1 else float("inf")
    selection = {"n_qualifying_cells": len(cands), "area_px": best["area"],
                 "rectangularity": best["rectangularity"], "aspect": best["aspect"],
                 "margin": float(margin), "used_seed": seed_xy is not None}
    return opening, wire.astype(bool), selection


def _fit_line(pts, schedule=(40, 20, 10, 5, 2.5)):
    """Total-least-squares line with a shrinking trim threshold.

    Starting wide and tightening lets the fit walk away from protrusions
    (weld bumps, debris on the wire) instead of being anchored by them.
    The final inlier set is points within 2.5 px of the line.
    """
    keep = np.ones(len(pts), bool)
    for thr in schedule:
        p = pts[keep]
        c = p.mean(0)
        _, _, vt = np.linalg.svd(p - c)
        d = vt[0]
        n = np.array([-d[1], d[0]])
        res = (pts - c) @ n
        # Recentre on the dominant edge: the mode of residuals, not the mean.
        hist, edges = np.histogram(res, bins=np.arange(res.min() - 1, res.max() + 2, 1.0))
        mode = edges[np.argmax(hist)] + 0.5
        keep = np.abs(res - mode) <= thr
    p = pts[keep]
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    res = (pts - c) @ n
    keep = np.abs(res) <= 2.5
    return c, d, n, res, keep


def _intersect(c1, d1, c2, d2):
    A = np.array([d1, -d2]).T
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


def detect(img, seed_xy=None, corner_margin=0.08):
    """Detect the inner frame as four fitted lines and their intersections.

    Returns (Frame, debug). Quality metrics, per side:
      inlier_frac   share of mid-side boundary points within 2.5 px of the line
      rms_px        RMS residual of inliers
      inward_max_px largest excursion into the opening (weld bumps, debris)
      bow_px        median residual of the middle third minus median of the
                    outer thirds: a crude curvature (lens distortion) check
    """
    opening, wire, selection = select_opening(img, seed_xy=seed_xy)
    touches_border = False  # enforced by select_opening
    cnts, _ = cv2.findContours(opening.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    s, d = cnt.sum(1), cnt[:, 0] - cnt[:, 1]
    rough = {"TL": cnt[np.argmin(s)], "BR": cnt[np.argmax(s)],
             "TR": cnt[np.argmax(d)], "BL": cnt[np.argmin(d)]}

    sides = {"top": ("TL", "TR"), "right": ("TR", "BR"),
             "bottom": ("BR", "BL"), "left": ("BL", "TL")}
    lines, quality, debug_pts = {}, {}, {}
    center = cnt.mean(0)
    for name, (a, b) in sides.items():
        pa, pb = rough[a], rough[b]
        ab = pb - pa
        L = np.linalg.norm(ab)
        t = (cnt - pa) @ ab / L ** 2
        dist_to_ab = np.abs((cnt - pa) @ np.array([-ab[1], ab[0]]) / L)
        sel = (t > corner_margin) & (t < 1 - corner_margin) & (dist_to_ab < 0.15 * L)
        pts = cnt[sel]
        c, dvec, n, res, keep = _fit_line(pts)
        if (center - c) @ n < 0:    # orient the normal into the opening
            n, res = -n, -res
        tt = ((pts - pa) @ ab / L ** 2)
        mid = (tt > 1 / 3) & (tt < 2 / 3)
        outer = ~mid
        lines[name] = (c, dvec)
        debug_pts[name] = (pts, res, keep, (pts - pa) @ ab / L ** 2)
        # Protrusions: contiguous runs of edge points more than 5 px inside the
        # line. On these frames they are the weld nubs where outer grid bars
        # join, and bolts; they hide a strip of the frame interior.
        order = np.argsort(tt)
        inward = res[order] > 5
        runs, start = [], None
        for i, flag in enumerate(np.r_[inward, False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= 10:
                    seg = order[start:i]
                    runs.append({"from_frac": float(tt[seg].min()), "to_frac": float(tt[seg].max()),
                                 "depth_px": float(res[seg].max())})
                start = None
        quality[name] = {
            "protrusions": runs,
            "n_points": int(len(pts)),
            "inlier_frac": float(keep.mean()),
            "rms_px": float(np.sqrt(np.mean(res[keep] ** 2))),
            "inward_max_px": float(res.max()),
            "outward_max_px": float(-res.min()),
            "bow_px": float(np.median(res[mid & keep]) - np.median(res[outer & keep])),
        }

    corners = np.array([
        _intersect(*lines["left"], *lines["top"]),
        _intersect(*lines["top"], *lines["right"]),
        _intersect(*lines["right"], *lines["bottom"]),
        _intersect(*lines["bottom"], *lines["left"]),
    ])
    angles = []
    for i in range(4):
        p0, p1, p2 = corners[i - 1], corners[i], corners[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        angles.append(float(np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))))
    side_len = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    quad_area = float(cv2.contourArea(corners.astype(np.float32)))
    quad_mask = np.zeros(opening.shape, np.uint8)
    cv2.fillPoly(quad_mask, [np.round(corners).astype(np.int32)], 1)
    quad_mask = quad_mask.astype(bool)
    q = {"selection": selection,
         "sides": quality,
         "quad_px_not_open": int((quad_mask & ~opening).sum()),
         "open_px_outside_quad": int((opening & ~quad_mask).sum()),
         "corner_angles_deg": dict(zip(CORNER_NAMES, angles)),
         "side_lengths_px": dict(zip(("top", "right", "bottom", "left"), side_len)),
         "opening_area_px": int(opening.sum()),
         "quad_area_px": quad_area,
         "opening_over_quad": float(opening.sum() / quad_area),
         "opening_touches_image_border": bool(touches_border)}
    warnings = []
    if selection["margin"] < 2:
        warnings.append(f"frame cell is only {selection['margin']:.1f}x larger than the next "
                        "candidate cell; check the selection")
    if touches_border:
        warnings.append("opening leaks to the image border: a wire gap was not sealed")
    for nme, sq in quality.items():
        if sq["inlier_frac"] < 0.6:
            warnings.append(f"{nme} side: only {sq['inlier_frac']:.0%} of edge points fit a line")
        if sq["rms_px"] > 2.0:
            warnings.append(f"{nme} side: inlier RMS {sq['rms_px']:.1f} px, edge is not clean")
        if abs(sq["bow_px"]) > 3:
            warnings.append(f"{nme} side bows {sq['bow_px']:+.1f} px (lens distortion or bent wire)")
    if any(abs(a - 90) > 8 for a in angles):
        warnings.append("corner angles deviate more than 8 deg from square")
    q["warnings"] = warnings
    h, w = img.shape[:2]
    f = Frame(corners, (w, h), "auto", q)
    return f, {"opening": opening, "wire": wire, "contour": cnt, "rough": rough,
               "lines": lines, "side_points": debug_pts}
