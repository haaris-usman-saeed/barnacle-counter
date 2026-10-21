"""Phase 1 entry point: extract barnacle annotations from contour masks.

Usage:
    python run_phase1.py --data /path/to/uploads --out ./phase1_output

Expects imgN.png + maskN.png pairs (same pixel grid). If masked_imgN.png is
present it is audited, never used for coordinates.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from barnacles.paths import data_dir
from barnacles.io import load_image, load_mask, check_alignment, locate_crop
from barnacles.extract import ExtractionParams, clean_stroke, extract_annotations, classify
from barnacles.diagnostics import (annotation_roi, nearest_neighbor, find_voids, summarize,
                                   frame_mask, evaluable_region, flag_frame_contact)
from barnacles.export import write_csv, write_json, write_coco
from barnacles import visualize as viz


def audit_overlay(img, mask, overlay_path):
    """Check how the provided overlay relates to the raw photo and mask."""
    ov = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    loc = locate_crop(img, ov)
    changed = loc.pop("changed")
    x0, y0, h, w = loc["x0"], loc["y0"], loc["h"], loc["w"]
    best = (0.0, 0, 0)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            sub = mask.stroke[y0 + dy:y0 + dy + h, x0 + dx:x0 + dx + w]
            if sub.shape != changed.shape:
                continue
            iou = (sub & changed).sum() / max((sub | changed).sum(), 1)
            if iou > best[0]:
                best = (float(iou), dx, dy)
    loc.update(painted_px=int(changed.sum()),
               contour_shift_vs_mask_px={"dx": best[1], "dy": best[2], "iou": best[0]},
               mask_px_outside_crop=int(mask.stroke.sum() -
                                        mask.stroke[y0:y0 + h, x0:x0 + w].sum()))
    return loc


def process(img_path, mask_path, overlay_path, p, out):
    name = Path(img_path).stem
    img = load_image(img_path)
    mask = load_mask(mask_path)
    dims = check_alignment(img, mask)

    clean, clean_rep = clean_stroke(mask.stroke, p)
    records, _, info = extract_annotations(clean, p, mask.color_index)
    records = classify(records, p)
    records.sort(key=lambda r: (round(r["y"] / 20), r["x"]))  # reading order, stable
    for k, r in enumerate(records, start=1):
        r["id"] = f"{name}_{k:04d}"
        r["image"] = Path(img_path).name

    roi = annotation_roi(records)
    nn = nearest_neighbor(records)
    spacing = float(np.median(nn))
    frame = frame_mask(img)
    n_touch = flag_frame_contact(records, frame)
    region = evaluable_region(records, roi, frame, spacing)
    voids, void_mask = find_voids(records, roi, spacing, region, frame)
    summary = summarize(records, clean_rep, info, roi, voids, void_mask, mask.palette,
                        region_px=int(region.sum()))
    summary["annotations_touching_frame"] = n_touch
    summary["extraction_info"] = info
    if overlay_path and Path(overlay_path).exists():
        summary["overlay_audit"] = audit_overlay(img, mask, overlay_path)

    # Exports
    write_csv(records, out / f"{name}_centers.csv")
    cv2.imwrite(str(out / f"{name}_centers.png"), viz.draw_centers(img, records))
    cv2.imwrite(str(out / f"{name}_review.png"),
                viz.draw_review(img, records, roi, voids, void_mask, region))
    gal_dir = out / "review_galleries"
    gal_dir.mkdir(exist_ok=True)
    groups = {
        "merged": [r for r in records if r["kind"] == "merged"],
        "solid_open_fragment": [r for r in records if r["kind"] in ("solid", "open", "fragment")],
        "flagged": [r for r in records if [f for f in r["flags"] if f != "touches_frame"]],
        "touches_frame": [r for r in records if "touches_frame" in r["flags"]],
    }
    for g, rs in groups.items():
        tile = viz.gallery(img, mask.stroke, rs, f"{name}: {g} ({len(rs)})")
        if tile is not None:
            cv2.imwrite(str(gal_dir / f"{name}_{g}.png"), tile)
    # Two void galleries: the largest (room for several barnacles) and a seeded
    # random sample of small interior voids (room for about one), which is
    # where single skipped barnacles would show up.
    vg = viz.void_gallery(img, records, voids, spacing)
    if vg is not None:
        cv2.imwrite(str(gal_dir / f"{name}_voids_largest.png"), vg)
    small = [v for v in voids if v["clearance_over_spacing"] < 1.5 and not v["frame_adjacent"]]
    rng = np.random.default_rng(0)
    pick = [small[i] for i in sorted(rng.choice(len(small), min(16, len(small)), replace=False))]
    vg = viz.void_gallery(img, records, pick, spacing)
    if vg is not None:
        cv2.imwrite(str(gal_dir / f"{name}_voids_small_sample.png"), vg)

    sc, sc_meta = viz.spot_check(img, records, spacing)
    cv2.imwrite(str(gal_dir / f"{name}_spot_check.png"), sc)
    summary["spot_check_patches"] = sc_meta

    return {"file": Path(img_path).name, "mask": Path(mask_path).name, **dims,
            "summary": summary, "artifacts": {"specks": clean_rep["specks"]},
            "voids": voids, "annotations": records, "nn": nn.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(data_dir()), help="folder with the challenge photos and masks (default: data/ or $BARNACLE_DATA)")
    ap.add_argument("--out", default="phase1_output")
    ap.add_argument("--pairs", nargs="+", default=["1", "2"])
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = ExtractionParams()

    images = []
    for k in args.pairs:
        res = process(data / f"img{k}.png", data / f"mask{k}.png",
                      data / f"masked_img{k}.png", p, out)
        images.append(res)
        s = res["summary"]
        print(f"{res['file']}: {s['annotated_count']} annotated barnacles "
              f"| kinds {s['kinds']} | flags {s['flags']} | voids {s['voids']['n']}")

    payload = {"phase": 1, "params": p.to_dict(),
               "coordinate_convention": "pixels in the raw photo; x = column, y = row, "
                                        "origin at top-left pixel center",
               "images": [{k: v for k, v in im.items() if k != "nn"} for im in images]}
    write_json(payload, out / "annotations.json")
    write_coco(images, out / "annotations_coco.json")
    viz.stats_figure({im["file"]: im for im in images}, out / "stats.png")
    write_json({im["file"]: im["summary"] for im in images}, out / "summary.json")


if __name__ == "__main__":
    main()
