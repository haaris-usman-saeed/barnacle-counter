"""Phase 2 entry point: inner frame, normalized coordinates, grid aggregation.

Usage:
    python run_phase2.py --phase1 phase1_output/annotations.json \
        --data /path/to/uploads --out phase2_output --rows 10 --cols 10 \
        [--frame-config frame_config.json]

Reads Phase 1 output; does not re-run extraction. Frame corners come from
--frame-config when it has an entry for an image, otherwise from automatic
detection. Detected corners are always written to frame_config_detected.json,
which can be edited (set "source": "manual") and passed back in.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from barnacles.paths import data_dir
from barnacles.io import load_image
from barnacles.diagnostics import frame_mask, find_voids
from barnacles.frame import detect, frame_from_config, load_config, CORNER_NAMES
from barnacles.grid import (GridSpec, assign, map_centers, pixel_cell_labels, build_cells,
                            verify, boundary_selftest)
from barnacles import grid_viz as gv
from barnacles.grid_viewer import image_payload, write_viewer

SWEEP = [(1, 1), (2, 3), (4, 4), (5, 7), (10, 10), (16, 16), (25, 20), (33, 29), (60, 60)]


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def resolution_checks(img, frame, mapped, spec, auto_frame):
    """(a) Exact invariance: scale corners and centers together, (u, v) and
    cells must not change. (b) Empirical: detect on a half-size copy and
    compare corners with full-size detection."""
    out = {}
    xy = np.array([[m["x"], m["y"]] for m in mapped])
    uv = frame.to_norm(xy)
    r0, c0, _ = assign(uv, spec)
    W, H = frame.image_size
    for s in (0.5, 0.37, 2.0):
        f2 = frame.scaled_to(W * s, H * s)
        uv2 = f2.to_norm(xy * s)
        r2, c2, _ = assign(uv2, spec)
        out[f"scale_{s}"] = {"max_uv_change": float(np.abs(uv2 - uv).max()),
                             "cells_identical": bool((r2 == r0).all() and (c2 == c0).all())}
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    try:
        fs, _ = detect(small)
        dev = fs.scaled_to(W, H).corners - auto_frame.corners
        out["detect_on_half_size"] = {
            "corner_shift_px_fullres": {n: round(float(np.hypot(*d)), 2) for n, d in zip(CORNER_NAMES, dev)},
            "max_corner_shift_px_fullres": float(np.hypot(dev[:, 0], dev[:, 1]).max())}
    except Exception as e:  # report, don't hide
        out["detect_on_half_size"] = {"error": str(e)}
    return out


def process(entry, data, out, spec, config, compare_auto=False):
    name = Path(entry["file"]).stem
    img = load_image(data / entry["file"])
    h, w = img.shape[:2]
    records = entry["annotations"]
    counted = [r for r in records if r["counted"]]
    # Manual corners bypass detection entirely; detection then runs only on
    # request (--compare-auto) to report the difference.
    auto = debug = None
    if entry["file"] in config:
        frame = frame_from_config(config[entry["file"]], (w, h))
        frame.quality = {"warnings": []}
        if compare_auto:
            auto, debug = detect(img)
            d = frame.corners - auto.corners
            frame.quality["manual_minus_auto_px"] = {n: [round(float(a), 2), round(float(b), 2)]
                                                     for n, (a, b) in zip(CORNER_NAMES, d)}
    else:
        auto, debug = detect(img)          # raw pixels only, no annotation seed
        frame = auto

    mapped = map_centers(records, frame)
    labels, offset = pixel_cell_labels(frame, spec, img.shape)
    wire = frame_mask(img)

    # Voids over the whole frame (Phase 1 limited them to the annotation hull).
    x0, y0 = offset
    roi = (x0, y0, x0 + labels.shape[1], y0 + labels.shape[0])
    spacing = entry["summary"]["median_nn_spacing_px"]
    near_wire = ndi.distance_transform_edt(~wire[y0:roi[3], x0:roi[2]]) < 0.5 * spacing
    region = (labels >= 0) & ~near_wire
    voids, void_mask = find_voids(records, roi, spacing, region, wire)

    cells = build_cells(frame, spec, mapped, labels, offset, wire, void_mask)
    checks = verify(frame, spec, mapped, cells, labels)

    sweep = []
    for rr, cc in SWEEP:
        sp = GridSpec(rr, cc)
        m2 = [dict(m) for m in mapped]
        c2 = build_cells(frame, sp, m2)
        v2 = verify(frame, sp, m2, c2)
        sweep.append({"grid": sp.tag(), "sum": v2["sum_of_cell_counts"],
                      "inside": v2["n_inside_frame"], "ok": v2["all_passed"]})
    res_checks = resolution_checks(img, frame, mapped, spec, auto) if auto is not None else {}

    # Coverage: how much of the frame Phase 1's annotations even reach.
    hull = cv2.convexHull(np.array([[m["x"], m["y"]] for m in mapped], np.float32)).astype(np.int32)
    hull_mask = np.zeros(labels.shape, np.uint8)
    cv2.fillPoly(hull_mask, [hull - [x0, y0]], 1)
    frame_px = (labels >= 0)
    coverage = {"frame_area_px": int(frame_px.sum()),
                "frame_outside_annotation_hull_frac": float((frame_px & ~hull_mask.astype(bool)).sum() / frame_px.sum()),
                "frame_void_frac": float((void_mask[0] & frame_px).sum() / frame_px.sum())}

    # Exports
    tag = spec.tag()
    with open(out / f"{name}_centers_normalized.csv", "w") as f:
        f.write("id,x,y,u,v,inside,signed_dist_to_frame_px,row,col,cell_id\n")
        for m in mapped:
            f.write(f"{m['id']},{m['x']:.2f},{m['y']:.2f},{m['u']:.6f},{m['v']:.6f},{m['inside']},"
                    f"{m['signed_dist_px']:.2f},{m['row']},{m['col']},{m['cell_id'] or ''}\n")
    with open(out / f"{name}_cells_{tag}.csv", "w") as f:
        f.write("cell_id,row,col,u0,u1,v0,v1,area_px,annotated_count,occluded_fraction,void_fraction,barnacle_ids\n")
        for c in cells:
            f.write(f"{c['cell_id']},{c['row']},{c['col']},{c['u_range'][0]:.6f},{c['u_range'][1]:.6f},"
                    f"{c['v_range'][0]:.6f},{c['v_range'][1]:.6f},{c['area_px']},{c['annotated_count']},"
                    f"{c['occluded_fraction']},{c['void_fraction']},{';'.join(c['barnacle_ids'])}\n")
    jdump({"image": entry["file"], "image_size": [w, h],
           "count_semantics": "annotated_count = extracted Phase 1 outlines whose center is in the cell. "
                              "Not a barnacle count. verified_count is reserved for human review.",
           "assignment_rule": "inside iff 0<=u<=1 and 0<=v<=1; col=min(floor(u*cols),cols-1); "
                              "row=min(floor(v*rows),rows-1)",
           "frame": {**frame.to_config(), "quality": frame.quality},
           "grid": {"rows": spec.rows, "cols": spec.cols},
           "verification": checks, "coverage": coverage, "cells": cells},
          out / f"{name}_grid_{tag}.json")

    cv2.imwrite(str(out / f"{name}_grid_overlay.png"), gv.draw_grid_overlay(img, frame, spec, mapped))
    cv2.imwrite(str(out / f"{name}_rectified.png"), gv.rectified_view(img, frame, spec, mapped))
    if auto is not None:
        cv2.imwrite(str(out / f"{name}_frame_check.png"), gv.frame_check(img, auto, debug, mapped))
    gv.summary_figure(img, frame, spec, cells, mapped,
                      f"{entry['file']}: {checks['n_inside_frame']} annotated centers inside the frame, "
                      f"{spec.rows} x {spec.cols} grid, sum of cells = {checks['sum_of_cell_counts']}",
                      out / f"{name}_grid_summary.png")
    payload = image_payload(entry["file"], img, frame, mapped, spec)
    return {"file": entry["file"], "frame": frame, "checks": checks, "sweep": sweep,
            "resolution": res_checks, "coverage": coverage, "payload": payload}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="phase1_output/annotations.json")
    ap.add_argument("--data", default=str(data_dir()), help="folder with the challenge photos and masks (default: data/ or $BARNACLE_DATA)")
    ap.add_argument("--out", default="phase2_output")
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--frame-config", default=None)
    ap.add_argument("--compare-auto", action="store_true",
                    help="also run detection on manually configured frames and report the difference")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec(args.rows, args.cols)
    config = load_config(args.frame_config) if args.frame_config else {}
    phase1 = json.load(open(args.phase1))

    results = [process(e, Path(args.data), out, spec, config, args.compare_auto)
               for e in phase1["images"]]

    jdump({r["file"]: r["frame"].to_config() for r in results}, out / "frame_config_detected.json")
    write_viewer([r["payload"] for r in results], out / "grid_viewer.html")
    report = {
        "grid": spec.tag(),
        "boundary_selftests": [boundary_selftest(GridSpec(a, b)) for a, b in SWEEP],
        "images": {r["file"]: {"frame": {**r["frame"].to_config(), "quality": r["frame"].quality},
                               "verification": r["checks"], "grid_sweep": r["sweep"],
                               "resolution": r["resolution"], "coverage": r["coverage"]}
                   for r in results},
    }
    jdump(report, out / "phase2_report.json")
    for r in results:
        c = r["checks"]
        print(f"{r['file']}: frame {r['frame'].source}, corners "
              f"{np.round(r['frame'].corners, 1).tolist()}")
        print(f"   {c['n_inside_frame']} inside + {c['n_outside_frame']} outside = "
              f"{c['n_extracted_centers']} | sum of {spec.n_cells} cells = {c['sum_of_cell_counts']} "
              f"| sweep ok: {all(s['ok'] for s in r['sweep'])} | warnings: {r['frame'].quality['warnings']}")


if __name__ == "__main__":
    main()
