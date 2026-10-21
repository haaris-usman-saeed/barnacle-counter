"""Grid over normalized frame coordinates and assignment of centers to cells.

Assignment rule (the only one in the system; the HTML viewer mirrors it):
  inside  <=>  0 <= u <= 1 and 0 <= v <= 1
  col = min(floor(u * cols), cols - 1)
  row = min(floor(v * rows), rows - 1)
Cells are half-open [k/n, (k+1)/n) except the last row/column, which is
closed on its far edge. Every inside point therefore lands in exactly one
cell, including points exactly on an internal grid line or on the frame edge.

Cells are uniform in (u, v), so they are equal areas of the physical frame
plane. In the photo they are perspective-projected quadrilaterals.

Counts here are ANNOTATED counts: the number of extracted outlines whose
center falls in the cell. They are not barnacle counts; see README.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int

    def __post_init__(self):
        if self.rows < 1 or self.cols < 1:
            raise ValueError("grid needs at least one row and one column")

    @property
    def n_cells(self):
        return self.rows * self.cols

    def tag(self):
        return f"{self.rows}x{self.cols}"


def cell_id(row: int, col: int) -> str:
    return f"r{row:02d}c{col:02d}"


def assign(uv: np.ndarray, spec: GridSpec):
    """Return (row, col, inside) arrays; row = col = -1 outside the frame."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    col = np.minimum(np.floor(u * spec.cols), spec.cols - 1)
    row = np.minimum(np.floor(v * spec.rows), spec.rows - 1)
    col = np.where(inside, col, -1).astype(int)
    row = np.where(inside, row, -1).astype(int)
    return row, col, inside


def map_centers(records: list, frame) -> list:
    """Normalized coordinates for every counted Phase 1 annotation."""
    rec = [r for r in records if r["counted"]]
    xy = np.array([[r["x"], r["y"]] for r in rec])
    uv = frame.to_norm(xy)
    sd = frame.signed_distance_px(xy)
    return [{"id": r["id"], "x": float(x), "y": float(y), "u": float(u), "v": float(v),
             "signed_dist_px": float(s), "flags": r["flags"]}
            for r, (x, y), (u, v), s in zip(rec, xy, uv, sd)]


def pixel_cell_labels(frame, spec: GridSpec, shape):
    """Cell index (row * cols + col) for every pixel center in the frame's
    bounding box, -1 outside. Returns (labels, (x0, y0))."""
    H_img, W_img = shape[:2]
    c = frame.corners
    x0, y0 = max(int(np.floor(c[:, 0].min())) - 1, 0), max(int(np.floor(c[:, 1].min())) - 1, 0)
    x1, y1 = min(int(np.ceil(c[:, 0].max())) + 2, W_img), min(int(np.ceil(c[:, 1].max())) + 2, H_img)
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
    uv = frame.to_norm(np.c_[xs.ravel(), ys.ravel()])
    row, col, inside = assign(uv, spec)
    lab = np.where(inside, row * spec.cols + col, -1).reshape(xs.shape)
    return lab, (x0, y0)


def build_cells(frame, spec: GridSpec, mapped: list, labels=None, offset=None,
                wire=None, void=None):
    """One record per cell, in row-major order.

    wire: bool mask (full image) of the frame wire; gives occluded_fraction.
    void: (mask, (vx, vy)) from Phase 1 find_voids; gives void_fraction.
    """
    uv = np.array([[m["u"], m["v"]] for m in mapped])
    row, col, inside = assign(uv, spec)
    for m, r, c, ins in zip(mapped, row, col, inside):
        m["inside"] = bool(ins)
        m["row"], m["col"] = int(r), int(c)
        m["cell_id"] = cell_id(r, c) if ins else None

    members = {}
    for m in mapped:
        if m["inside"]:
            members.setdefault((m["row"], m["col"]), []).append(m["id"])

    area = occl = vfrac = None
    if labels is not None:
        flat = labels.ravel()
        ok = flat >= 0
        area = np.bincount(flat[ok], minlength=spec.n_cells)
        x0, y0 = offset
        h, w = labels.shape
        if wire is not None:
            wsub = wire[y0:y0 + h, x0:x0 + w].ravel()
            occl = np.bincount(flat[ok], weights=wsub[ok], minlength=spec.n_cells) / np.maximum(area, 1)
        if void is not None:
            vm, (vx, vy) = void
            vfull = np.zeros(labels.shape, bool)
            # paste the void mask (ROI coords) into the label-map frame
            ys0, xs0 = vy - y0, vx - x0
            sy0, sx0 = max(ys0, 0), max(xs0, 0)
            sy1, sx1 = min(ys0 + vm.shape[0], h), min(xs0 + vm.shape[1], w)
            vfull[sy0:sy1, sx0:sx1] = vm[sy0 - ys0:sy1 - ys0, sx0 - xs0:sx1 - xs0]
            vfrac = np.bincount(flat[ok], weights=vfull.ravel()[ok], minlength=spec.n_cells) / np.maximum(area, 1)

    cells = []
    for r in range(spec.rows):
        for c in range(spec.cols):
            k = r * spec.cols + c
            u0, u1, v0, v1 = c / spec.cols, (c + 1) / spec.cols, r / spec.rows, (r + 1) / spec.rows
            poly = frame.to_image([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
            ids = members.get((r, c), [])
            cells.append({
                "cell_id": cell_id(r, c), "row": r, "col": c,
                "u_range": [u0, u1], "v_range": [v0, v1],
                "polygon_px": [[round(float(x), 2), round(float(y), 2)] for x, y in poly],
                "area_px": int(area[k]) if area is not None else None,
                "annotated_count": len(ids),
                "barnacle_ids": ids,
                "occluded_fraction": round(float(occl[k]), 4) if occl is not None else None,
                "void_fraction": round(float(vfrac[k]), 4) if vfrac is not None else None,
                "verified_count": None,
            })
    return cells


def verify(frame, spec: GridSpec, mapped: list, cells: list, labels=None) -> dict:
    """Bookkeeping checks. Raises AssertionError on any hard failure."""
    ids_inside = [m["id"] for m in mapped if m["inside"]]
    ids_cells = [i for c in cells for i in c["barnacle_ids"]]
    total = sum(c["annotated_count"] for c in cells)
    n_out = sum(not m["inside"] for m in mapped)
    checks = {
        "n_extracted_centers": len(mapped),
        "n_inside_frame": len(ids_inside),
        "n_outside_frame": n_out,
        "sum_of_cell_counts": total,
        "sum_equals_inside": total == len(ids_inside),
        "no_duplicate_ids": len(ids_cells) == len(set(ids_cells)),
        "cell_ids_equal_inside_ids": set(ids_cells) == set(ids_inside),
        "every_cell_present": len(cells) == spec.n_cells,
    }
    # Homography inside test vs. polygon inside test (independent code paths).
    poly_in = frame.contains([[m["x"], m["y"]] for m in mapped])
    disagree = [m["id"] for m, p in zip(mapped, poly_in) if p != m["inside"]]
    checks["inside_test_disagreements"] = disagree
    checks["min_abs_dist_to_frame_edge_px"] = float(min(abs(m["signed_dist_px"]) for m in mapped))
    rt = frame.to_image(frame.to_norm(frame.corners)) - frame.corners
    checks["corner_roundtrip_err_px"] = float(np.abs(rt).max())
    if labels is not None:
        area_sum = int((labels >= 0).sum())
        checks["pixel_partition_area_px"] = area_sum
        checks["pixel_partition_vs_quad_area"] = area_sum / frame.quality.get("quad_area_px", area_sum) \
            if frame.quality.get("quad_area_px") else None
    hard = ["sum_equals_inside", "no_duplicate_ids", "cell_ids_equal_inside_ids", "every_cell_present"]
    for h in hard:
        assert checks[h], f"grid verification failed: {h}"
    assert not disagree, f"inside tests disagree for {disagree}"
    checks["all_passed"] = True
    return checks


def boundary_selftest(spec: GridSpec) -> dict:
    """Synthetic points on every grid line, corner and edge: each must land in
    exactly one cell, and points just outside must land in none."""
    us = np.r_[np.arange(spec.cols + 1) / spec.cols, 0.5 / spec.cols]
    vs = np.r_[np.arange(spec.rows + 1) / spec.rows, 0.5 / spec.rows]
    U, V = np.meshgrid(us, vs)
    pts = np.c_[U.ravel(), V.ravel()]
    row, col, inside = assign(pts, spec)
    eps = 1e-12
    outside_pts = np.array([[-eps, 0.5], [1 + eps, 0.5], [0.5, -eps], [0.5, 1 + eps]])
    _, _, ins_out = assign(outside_pts, spec)
    ok = bool(inside.all() and (row >= 0).all() and (row < spec.rows).all()
              and (col >= 0).all() and (col < spec.cols).all() and not ins_out.any())
    return {"grid": spec.tag(), "n_points": int(len(pts)), "ok": ok}
