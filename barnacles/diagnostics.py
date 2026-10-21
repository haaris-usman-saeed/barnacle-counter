"""Image-level diagnostics: summary stats, spacing, and unannotated voids.

Nothing here decides what a barnacle *is*. Void detection only finds space
inside the annotated region that is large enough to hold a barnacle at the
local packing density but contains no annotation. Whether that space holds
an unannotated barnacle or bare rock is a judgement for a human (Phase 1)
or a model (later phases).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def annotation_roi(records: list, pad: int = 0) -> tuple:
    """Bounding box (x0, y0, x1, y1) of all annotation geometry.

    This is the only region where "no annotation" carries any information.
    Outside it, the masks say nothing at all.
    """
    x0 = min(r["bbox_xywh"][0] for r in records) - pad
    y0 = min(r["bbox_xywh"][1] for r in records) - pad
    x1 = max(r["bbox_xywh"][0] + r["bbox_xywh"][2] for r in records) + pad
    y1 = max(r["bbox_xywh"][1] + r["bbox_xywh"][3] for r in records) + pad
    return int(x0), int(y0), int(x1), int(y1)


def frame_mask(img: np.ndarray, h_range=(75, 100), s_min=80, v_min=18) -> np.ndarray:
    """Pixels belonging to the teal quadrat frame.

    The wire is a saturated teal (OpenCV hue about 89 to 96 in both photos)
    and is the only object in either scene in that hue band; the green algae
    in img2 sits at a yellow-green hue well below it. Closing bridges specular
    highlights on the wire; opening drops small teal-ish specks. Used only to
    keep the frame out of the void analysis and to flag edge annotations.
    """
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= h_range[0]) & (h <= h_range[1]) & (s_ >= s_min) & (v >= v_min)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    return m.astype(bool)


def evaluable_region(records: list, roi: tuple, frame: np.ndarray, spacing: float) -> np.ndarray:
    """Where "no annotation here" is informative, as a mask over the ROI box.

    Convex hull of annotation centers (the annotators clearly worked inside
    it), minus the frame and a half-spacing margin around it (barnacles cut
    by the wire are ambiguous and are not a counting question for Phase 1).
    """
    import cv2
    x0, y0, x1, y1 = roi
    pts = np.array([[r["x"] - x0, r["y"] - y0] for r in records if r["counted"]], np.float32)
    hull = cv2.convexHull(pts).astype(np.int32)
    reg = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(reg, [hull], 1)
    fr = frame[y0:y1, x0:x1]
    near_frame = ndi.distance_transform_edt(~fr) < 0.5 * spacing
    return reg.astype(bool) & ~near_frame


def flag_frame_contact(records: list, frame: np.ndarray, margin: int = 2) -> int:
    """Flag annotations whose polygon touches the frame (likely partial barnacles)."""
    import cv2
    near = ndi.binary_dilation(frame, iterations=margin)
    n = 0
    for r in records:
        x, y, w, h = r["bbox_xywh"]
        poly = np.array(r["polygon"]) - [x, y]
        m = np.zeros((h + 1, w + 1), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        if (m.astype(bool) & near[y:y + h + 1, x:x + w + 1]).any():
            r["flags"].append("touches_frame")
            n += 1
    return n


def nearest_neighbor(records: list) -> np.ndarray:
    pts = np.array([[r["x"], r["y"]] for r in records if r["counted"]])
    d, _ = cKDTree(pts).query(pts, k=2)
    return d[:, 1]


def find_voids(records: list, roi: tuple, spacing: float, region: np.ndarray = None,
               frame: np.ndarray = None,
               radius_factor: float = 0.9, min_area_factor: float = 0.5) -> list:
    """Regions inside the ROI farther than `radius_factor * spacing` from any
    annotation center.

    Reasoning: in a densely packed patch with center spacing s, no point is
    farther than about 0.58 s from a center. If one barnacle is skipped, the
    point where it sits is about 1.0 s from its neighbours. So a clearance of
    0.9 s means "there is room for at least one more barnacle here".
    """
    x0, y0, x1, y1 = roi
    occ = np.ones((y1 - y0, x1 - x0), bool)
    for r in records:
        if r["counted"]:
            cx, cy = int(round(r["x"])) - x0, int(round(r["y"])) - y0
            if 0 <= cy < occ.shape[0] and 0 <= cx < occ.shape[1]:
                occ[cy, cx] = False
    dist = ndi.distance_transform_edt(occ)
    void = dist >= radius_factor * spacing
    if region is not None:
        void &= region
    lab, n = ndi.label(void)
    d_frame = (ndi.distance_transform_edt(~frame[y0:y1, x0:x1])
               if frame is not None else None)
    voids = []
    min_area = min_area_factor * np.pi * (0.5 * spacing) ** 2
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        reg = lab[sl] == i
        area = int(reg.sum())
        if area < min_area:
            continue
        d = np.where(reg, dist[sl], 0)
        yy, xx = np.unravel_index(np.argmax(d), d.shape)
        voids.append({
            "void_id": len(voids),
            "x": float(xx + sl[1].start + x0), "y": float(yy + sl[0].start + y0),
            "max_clearance_px": float(d.max()),
            "clearance_over_spacing": float(d.max() / spacing),
            "area_px": area,
            "bbox_xywh": [int(sl[1].start + x0), int(sl[0].start + y0),
                          int(reg.shape[1]), int(reg.shape[0])],
            # Barnacles half under the wire are often skipped or unreadable.
            "frame_adjacent": bool(d_frame is not None and
                                   d_frame[sl][reg].min() < 1.0 * spacing),
        })
    voids.sort(key=lambda v: -v["area_px"])
    for k, v in enumerate(voids):
        v["void_id"] = k
        # Rough capacity: barnacles that would fit at the local packing density.
        v["approx_barnacle_slots"] = round(v["area_px"] / (spacing ** 2 * np.sqrt(3) / 2), 1)
    return voids, (void, (x0, y0))


def summarize(records: list, clean_report: dict, info: dict, roi: tuple,
              voids: list, void_mask, palette: list, region_px: int = 0) -> dict:
    counted = [r for r in records if r["counted"]]
    kinds, flags = {}, {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    nn = nearest_neighbor(records)
    diam = np.array([r["equiv_diameter_px"] for r in counted])
    sw = np.array([r["stroke_width_px"] for r in counted])
    sw = sw[np.isfinite(sw)]
    x0, y0, x1, y1 = roi
    vm, _ = void_mask
    by_color = {}
    for r in counted:
        by_color.setdefault(r.get("stroke_color_index", -1), []).append(r)
    return {
        "annotated_count": len(counted),
        "excluded_records": len(records) - len(counted),
        "kinds": kinds,
        "flags": flags,
        "artifacts": {
            "specks_removed": len(clean_report["specks"]),
            "isolated_specks": [s for s in clean_report.get("specks_isolated", [])],
            "pinholes_filled": clean_report["n_pinholes_filled"],
            "sliver_holes_filled": info["n_slivers_filled"],
        },
        "roi_xyxy": list(roi),
        "roi_area_px": int((x1 - x0) * (y1 - y0)),
        "annotation_density_per_100k_px": 1e5 * len(counted) / ((x1 - x0) * (y1 - y0)),
        "median_equiv_diameter_px": float(np.median(diam)),
        "equiv_diameter_p5_p95_px": [float(np.percentile(diam, 5)), float(np.percentile(diam, 95))],
        "median_nn_spacing_px": float(np.median(nn)),
        "stroke_width_p10_p50_p90_px": [float(v) for v in np.percentile(sw, [10, 50, 90])],
        "stroke_colors": {
            str(k): {"rgb": list(palette[k]) if k >= 0 else None, "n": len(v),
                     "median_stroke_width_px": float(np.nanmedian([r["stroke_width_px"] for r in v])),
                     "y_range": [float(min(r["y"] for r in v)), float(max(r["y"] for r in v))]}
            for k, v in sorted(by_color.items())
        },
        "voids": {"n": len(voids),
                  "n_frame_adjacent": sum(v["frame_adjacent"] for v in voids),
                  "n_clearance_ge_1p5_spacing": sum(v["clearance_over_spacing"] >= 1.5 for v in voids),
                  "n_clearance_ge_2p5_spacing": sum(v["clearance_over_spacing"] >= 2.5 for v in voids),
                  "fraction_of_evaluable_region": float(vm.sum() / max(region_px, 1)),
                  "evaluable_region_px": int(region_px)},
    }
