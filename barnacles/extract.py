"""Turn a binary stroke mask into per-barnacle annotation records.

Method (see README for the reasoning):
 1. Remove specks: 8-connected stroke blobs below `min_component_area`.
 2. Fill pinholes: enclosed background regions below `max_pinhole_area`
    are texture inside a ragged stroke, not barnacle interiors.
 3. Classify every remaining enclosed background region ("hole").
    A hole is a barnacle interior if it is large enough and thick enough;
    thin holes are slivers left where a stroke overlaps itself, and are
    filled back into the stroke.
 4. Split strokes along their midline: every stroke pixel is assigned to the
    nearest background region (an interior, or the outside). The region for
    interior k is that interior plus the inner half of its ring. This gives
    one polygon per barnacle even when rings touch and share stroke pixels.
 5. Stroke components with no interior at all (rings drawn so thick the
    middle closed up, or rings left open) become one annotation each,
    with the convex hull as geometry.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)
FOUR = ndi.generate_binary_structure(2, 1)


@dataclass
class ExtractionParams:
    min_component_area: int = 15         # px; smaller stroke blobs are specks
    max_pinhole_area: int = 10           # px; smaller enclosed holes are stroke texture
    min_interior_inradius: float = 1.5   # px; thinner holes are self-overlap slivers
    fragment_area_frac: float = 0.25     # no-interior blob below this x median area -> fragment
    open_fill_ratio: float = 0.6         # no-interior blob filling less of its hull -> open ring

    def to_dict(self):
        return asdict(self)


def _outside_labels(lab: np.ndarray) -> set:
    edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    return set(int(v) for v in np.unique(edge) if v > 0)


def clean_stroke(stroke: np.ndarray, p: ExtractionParams):
    """Steps 1 and 2. Returns cleaned stroke and a report of what was removed."""
    lab, n = ndi.label(stroke, structure=EIGHT)
    areas = ndi.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    small = np.where(areas < p.min_component_area)[0] + 1
    specks = []
    if len(small):
        cents = ndi.center_of_mass(np.ones_like(lab), lab, small)
        specks = [{"x": float(c[1]), "y": float(c[0]), "area_px": int(areas[i - 1])}
                  for i, c in zip(small, cents)]
    clean = stroke & ~np.isin(lab, small)

    bg_lab, nb = ndi.label(~clean, structure=FOUR)
    outside = _outside_labels(bg_lab)
    bg_areas = ndi.sum_labels(np.ones_like(bg_lab), bg_lab, index=np.arange(1, nb + 1))
    pin = [i + 1 for i, a in enumerate(bg_areas)
           if a < p.max_pinhole_area and (i + 1) not in outside]
    clean = clean | np.isin(bg_lab, pin)
    # Specks hugging a kept stroke are ragged-brush debris; isolated ones are
    # stray marks worth reporting (they can also be a lone dot annotation).
    if specks:
        d_to_stroke = ndi.distance_transform_edt(~clean)
        for s in specks:
            s["dist_to_stroke_px"] = float(d_to_stroke[int(round(s["y"])), int(round(s["x"]))])
    isolated = [s for s in specks if s["dist_to_stroke_px"] > 10]
    report = {"n_raw_components": int(n), "specks": specks, "specks_isolated": isolated,
              "n_pinholes_filled": len(pin),
              "pinhole_px_filled": int(sum(bg_areas[i - 1] for i in pin))}
    return clean, report


def _polygon(region: np.ndarray, x0: int, y0: int) -> list:
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :]
    return (c + [x0, y0]).astype(int).tolist()


def _region_stats(region: np.ndarray, x0: int, y0: int) -> dict:
    m = cv2.moments(region.astype(np.uint8), binaryImage=True)
    area = m["m00"]
    cx, cy = m["m10"] / area + x0, m["m01"] / area + y0
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    filled_area = cv2.contourArea(c)
    if len(c) >= 5:
        (_, _), (a, b), ang = cv2.fitEllipse(c)
        major, minor = max(a, b), min(a, b)
    else:
        major = minor = np.sqrt(4 * area / np.pi)
        ang = 0.0
    return {"x": float(cx), "y": float(cy), "area_px": int(area),
            "equiv_diameter_px": float(np.sqrt(4 * area / np.pi)),
            "ellipse_major_px": float(major), "ellipse_minor_px": float(minor),
            "aspect_ratio": float(major / max(minor, 1e-6)),
            "solidity": float(filled_area / hull_area) if hull_area > 0 else 1.0}


def extract_annotations(stroke: np.ndarray, p: ExtractionParams,
                        color_index: np.ndarray = None):
    """Steps 3 to 5. Returns (records, region_label_map, info)."""
    H, W = stroke.shape
    comp_lab, n_comp = ndi.label(stroke, structure=EIGHT)
    bg_lab, n_bg = ndi.label(~stroke, structure=FOUR)
    outside = _outside_labels(bg_lab)

    # Distance from each background pixel to the nearest stroke pixel: the
    # max over a hole is its inscribed radius (thickness test for slivers).
    dist_bg = ndi.distance_transform_edt(~stroke)
    idx = np.arange(1, n_bg + 1)
    inrad = ndi.maximum(dist_bg, bg_lab, idx)
    bg_area = ndi.sum_labels(np.ones_like(bg_lab), bg_lab, idx)
    holes = [i for i in idx if i not in outside]
    interiors = [i for i in holes if inrad[i - 1] >= p.min_interior_inradius]
    slivers = [i for i in holes if inrad[i - 1] < p.min_interior_inradius]

    # Fill slivers back into the stroke, then relabel background so that
    # outside -> 0 and interiors -> 1..K.
    stroke2 = stroke | np.isin(bg_lab, slivers)
    remap = np.zeros(n_bg + 1, np.int32)
    for k, i in enumerate(interiors, start=1):
        remap[i] = k
    bg_k = remap[bg_lab]  # 0 on stroke and outside, k inside interior k

    # Midline split: each stroke pixel inherits the label of its nearest
    # background pixel. Outside counts as label 0, so the outer half of each
    # ring is discarded and the inner half joins the interior.
    _, (iy, ix) = ndi.distance_transform_edt(stroke2, return_indices=True)
    region = np.where(stroke2, bg_k[iy, ix], bg_k)

    # Which stroke component encloses each interior: the component touching it.
    comp_of = {}
    nested = set()
    bg_objs = ndi.find_objects(bg_lab)
    for k, i in enumerate(interiors, start=1):
        sl = bg_objs[i - 1]
        sl = (slice(max(sl[0].start - 1, 0), min(sl[0].stop + 1, H)),
              slice(max(sl[1].start - 1, 0), min(sl[1].stop + 1, W)))
        hole = bg_lab[sl] == i
        ring = ndi.binary_dilation(hole, FOUR) & ~hole
        touching = comp_lab[sl][ring]
        touching = touching[touching > 0]
        labs, cnt = np.unique(touching, return_counts=True)
        comp_of[k] = int(labs[np.argmax(cnt)])
        if len(labs) > 1:
            nested.add(k)  # another stroke component sits inside this interior

    per_comp = {}
    for k, c in comp_of.items():
        per_comp.setdefault(c, []).append(k)

    records = []
    objs = ndi.find_objects(region)
    for k in range(1, len(interiors) + 1):
        sl = objs[k - 1]
        reg = region[sl] == k
        y0, x0 = sl[0].start, sl[1].start
        st = _region_stats(reg, x0, y0)
        c = comp_of[k]
        siblings = len(per_comp[c])
        rec = {"kind": "closed" if siblings == 1 else "merged", **st,
               "bbox_xywh": [int(x0), int(y0), int(reg.shape[1]), int(reg.shape[0])],
               "interior_area_px": int(bg_area[interiors[k - 1] - 1]),
               "interior_inradius_px": float(inrad[interiors[k - 1] - 1]),
               "component_id": c, "n_in_component": siblings,
               "polygon": _polygon(reg, x0, y0), "flags": []}
        if k in nested:
            rec["flags"].append("encloses_other_stroke")
        records.append(rec)

    # Components that own no interior.
    comp_objs = ndi.find_objects(comp_lab)
    no_interior = [c for c in range(1, n_comp + 1) if c not in per_comp]
    for c in no_interior:
        sl = comp_objs[c - 1]
        comp = comp_lab[sl] == c
        y0, x0 = sl[0].start, sl[1].start
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        pts = np.vstack([q[:, 0, :] for q in cnts])
        hull = cv2.convexHull(pts)
        hull_mask = np.zeros(comp.shape, np.uint8)
        cv2.fillPoly(hull_mask, [hull], 1)
        st = _region_stats(hull_mask.astype(bool), x0, y0)
        stroke_area = int(comp.sum())
        hull_area = int(hull_mask.sum())
        rec = {"kind": "no_interior", **st,
               "bbox_xywh": [int(x0), int(y0), int(comp.shape[1]), int(comp.shape[0])],
               "interior_area_px": 0, "interior_inradius_px": 0.0,
               "component_id": c, "n_in_component": 1,
               "stroke_area_px": stroke_area,
               "stroke_fill_ratio": stroke_area / max(hull_area, 1),
               "polygon": (hull[:, 0, :] + [x0, y0]).astype(int).tolist(),
               "flags": []}
        records.append(rec)

    # Stroke-colour provenance (mask1 was drawn in two blues).
    if color_index is not None:
        for r in records:
            x, y, w, h = r["bbox_xywh"]
            ci = color_index[y:y + h, x:x + w]
            cl = comp_lab[y:y + h, x:x + w] == r["component_id"]
            vals = ci[cl & (ci >= 0)]
            r["stroke_color_index"] = int(np.bincount(vals).argmax()) if len(vals) else -1

    # Stroke width per component: twice the distance-transform value sampled
    # on the stroke skeleton. Width matters because the midline polygon sits
    # half a stroke inside the outer edge, so thick and thin pens give
    # slightly different geometry for the same barnacle.
    from skimage.morphology import skeletonize
    dt_stroke = ndi.distance_transform_edt(stroke2)
    widths = {}
    for c in set(r["component_id"] for r in records):
        sl = comp_objs[c - 1]
        comp = (comp_lab[sl] == c)
        sk = skeletonize(comp)
        vals = dt_stroke[sl][sk]
        widths[c] = float(2 * np.median(vals)) if len(vals) else float("nan")
    for r in records:
        r["stroke_width_px"] = widths[r["component_id"]]

    info = {"n_components": int(n_comp), "n_holes": len(holes),
            "n_interiors": len(interiors), "n_slivers_filled": len(slivers),
            "n_no_interior_components": len(no_interior)}
    return records, region, info


def classify(records: list, p: ExtractionParams) -> list:
    """Assign kinds/flags that depend on image-level statistics.

    `counted` marks whether a record contributes to the annotated count.
    Only fragments (tiny partial strokes) are excluded; everything else is
    counted but may carry review flags.
    """
    ref = [r["area_px"] for r in records if r["kind"] in ("closed", "merged")]
    med = float(np.median(ref))
    q1, q3 = np.percentile(ref, [25, 75])
    hi = q3 + 3 * (q3 - q1)
    for r in records:
        if r["kind"] == "no_interior":
            is_open = r["stroke_fill_ratio"] < p.open_fill_ratio
            if is_open and r["area_px"] < p.fragment_area_frac * med:
                r["kind"] = "fragment"   # short partial stroke, not a ring
            elif is_open:
                r["kind"] = "open"       # ring with a gap; hull approximates it
            else:
                r["kind"] = "solid"      # ring drawn so thick it has no hole
                r["stroke_width_px"] = float("nan")  # blob radius, not a pen width
        if r["area_px"] > hi:
            r["flags"].append("size_outlier_large")
        if r["area_px"] < 0.15 * med:
            r["flags"].append("size_outlier_small")
        if r["kind"] == "merged" and r["solidity"] < 0.8:
            r["flags"].append("concave_merged_region")
        r["counted"] = r["kind"] != "fragment"

    # Near-duplicate centers (ring drawn twice, nested rings).
    pts = np.array([[r["x"], r["y"]] for r in records])
    diam = np.median([r["equiv_diameter_px"] for r in records if r["counted"]])
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    for i, j in tree.query_pairs(0.35 * diam):
        for a in (i, j):
            if "near_duplicate" not in records[a]["flags"]:
                records[a]["flags"].append("near_duplicate")
    return records
