"""Write annotation records to CSV (centers), JSON (full), and COCO (training)."""
import csv
import json

SCALAR_FIELDS = ["id", "image", "x", "y", "counted", "kind", "flags",
                 "area_px", "equiv_diameter_px", "ellipse_major_px", "ellipse_minor_px",
                 "aspect_ratio", "solidity", "interior_area_px", "stroke_width_px",
                 "stroke_color_index", "component_id", "n_in_component",
                 "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def _row(r):
    x, y, w, h = r["bbox_xywh"]
    row = {k: r.get(k) for k in SCALAR_FIELDS}
    row.update(flags=";".join(r["flags"]), bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
               x=round(r["x"], 2), y=round(r["y"], 2))
    for k in ("equiv_diameter_px", "ellipse_major_px", "ellipse_minor_px",
              "aspect_ratio", "solidity", "stroke_width_px"):
        row[k] = round(row[k], 3) if row[k] == row[k] else ""  # NaN -> blank
    return row


def write_csv(records, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCALAR_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in records:
            w.writerow(_row(r))


def write_json(payload, path):
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=float)


def write_coco(images: list, path):
    """COCO instance-segmentation file with one category.

    Only counted records are written. Each image carries the annotation ROI;
    any training code must treat pixels outside it as *unlabelled*, not as
    background, and should treat voids inside it with suspicion, because the
    masks are incomplete. COCO tooling assumes exhaustive labels by default.
    """
    coco = {"info": {"description": "DALI barnacles, Phase 1 extraction",
                     "note": "Annotations are incomplete. See roi_xyxy and README."},
            "categories": [{"id": 1, "name": "barnacle",
                            "supercategory": "organism"}],
            "images": [], "annotations": []}
    ann_id = 1
    for img_id, im in enumerate(images, start=1):
        coco["images"].append({"id": img_id, "file_name": im["file"],
                               "width": im["width"], "height": im["height"],
                               "roi_xyxy": im["summary"]["roi_xyxy"]})
        for r in im["annotations"]:
            if not r["counted"]:
                continue
            seg = [float(v) for pt in r["polygon"] for v in pt]
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [seg], "area": float(r["area_px"]),
                "bbox": [float(v) for v in r["bbox_xywh"]], "iscrowd": 0,
                "attributes": {"center_xy": [r["x"], r["y"]], "kind": r["kind"],
                               "flags": r["flags"], "source_id": r["id"]}})
            ann_id += 1
    with open(path, "w") as f:
        json.dump(coco, f)
