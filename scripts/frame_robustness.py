"""Stress-test seedless frame detection with known image transforms.

For each transform the true corners are known (the reference corners pushed
through the same transform), so corner error is measurable. Writes
frame_robustness.json in the output folder.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from barnacles.frame import detect  # noqa: E402
from barnacles.paths import data_dir  # noqa: E402


def transforms(img):
    h, w = img.shape[:2]
    I = np.eye(3)
    yield "identity", img, I
    yield "darker x0.6", np.clip(img * 0.6, 0, 255).astype(np.uint8), I
    yield "brighter x1.3", np.clip(img * 1.3, 0, 255).astype(np.uint8), I
    yield "gaussian blur s=2", cv2.GaussianBlur(img, (0, 0), 2), I
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    yield "jpeg q50", cv2.imdecode(buf, cv2.IMREAD_COLOR), I
    S = np.diag([0.5, 0.5, 1])
    yield "downscale 0.5", cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), S
    for ang in (4, -7):
        R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0), [0, 0, 1]])
        yield f"rotate {ang} deg", cv2.warpAffine(img, R[:2], (w, h), borderValue=(128, 128, 128)), R
    T = np.array([[1, 0, -300], [0, 1, -150], [0, 0, 1]], float)
    yield "crop (-300,-150)", img[150:, 300:].copy(), T


def run(data, out):
    report = {}
    for name in ("img1.png", "img2.png"):
        img = cv2.imread(str(Path(data) / name), cv2.IMREAD_COLOR)
        ref, _ = detect(img)
        rows = []
        for tname, timg, M in transforms(img):
            truth = cv2.perspectiveTransform(ref.corners.reshape(-1, 1, 2), M).reshape(-1, 2)
            try:
                f, _ = detect(timg)
                err = np.hypot(*(f.corners - truth).T)
                scale = 0.5 if tname.startswith("downscale") else 1.0
                rows.append({"transform": tname, "max_corner_err_px": float(err.max() / scale),
                             "margin": f.quality["selection"]["margin"],
                             "warnings": f.quality["warnings"]})
            except Exception as e:
                rows.append({"transform": tname, "error": str(e)})
        report[name] = rows
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "frame_robustness.json", "w") as fh:
        json.dump(report, fh, indent=1)
    return report


if __name__ == "__main__":
    rep = run(sys.argv[1] if len(sys.argv) > 1 else str(data_dir()),
              sys.argv[2] if len(sys.argv) > 2 else "phase2_output")
    for k, rows in rep.items():
        for r in rows:
            print(k, r)
