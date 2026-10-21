"""Inference on a newly uploaded photo (no annotations needed).

upload bytes -> validate/decode -> frame (automatic, or manual corners)
-> rectify -> candidates -> features -> classifier score -> optional center
refinement -> duplicate suppression at the default operating point ->
calibration sample for the reviewer -> review signals -> viewer payload.

Every step is the same code path used in training and evaluation
(rectify, candidate_detection, features, ml.DuplicateIndex), with the
settings stored in the model artifact, so a new photo is processed exactly
like the training photos were.
"""
import base64
import hashlib
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .frame import Frame, detect, CORNER_NAMES
from .rectify import rectify
from .candidate_detection import CandidateParams, score_maps, detect_candidates, hough_candidates, merge_candidates
from .features import candidate_features
from .ml import DuplicateIndex, select
from .calibration import stratified_sample
from .review import static_signals
from .grid import GridSpec, assign, cell_id
from .fingerprint import recognize

ARTIFACT_VERSION = 2
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "barnacle_detector.pkl"


class InputError(ValueError):
    """The upload itself is unusable (format, size, decoding)."""
    status = 400


class FrameError(RuntimeError):
    """The quadrat frame could not be located; manual corners are needed."""
    status = 422


@dataclass
class Limits:
    max_bytes: int = 40 * 1024 * 1024
    max_pixels: int = 60_000_000
    min_side: int = 500
    detect_max_side: int = 3300      # frame detection runs on a copy no larger than this
    formats: tuple = ("png", "jpeg")  # public deployment accepts PNG and JPEG only


MAGIC = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpeg", b"II*\x00": "tiff", b"MM\x00*": "tiff",
         b"BM": "bmp", b"GIF8": "gif"}


def sniff(data: bytes):
    for m, kind in MAGIC.items():
        if data.startswith(m):
            return kind
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def header_size(data: bytes, kind: str):
    """Width and height read from the file header WITHOUT decoding, so an
    oversized ("decompression bomb") image is rejected before any pixel
    buffer is allocated. Returns None if the header cannot be parsed."""
    try:
        if kind == "png":
            if data[12:16] != b"IHDR":
                return None
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if kind == "jpeg":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    return None
                marker = data[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg = int.from_bytes(data[i + 2:i + 4], "big")
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
                i += 2 + seg
    except (IndexError, ValueError):
        return None
    return None


def decode_image(data: bytes, limits: Limits = Limits()):
    """Validate and decode an upload. The bytes are only ever handed to
    OpenCV's image decoder; nothing in them is executed or written to disk."""
    if not data:
        raise InputError("The upload is empty.")
    if len(data) > limits.max_bytes:
        e = InputError(f"The file is {len(data) / 1e6:.1f} MB; the limit is {limits.max_bytes / 1e6:.0f} MB.")
        e.status = 413
        raise e
    kind = sniff(data)
    if kind not in limits.formats:
        e = InputError("Unsupported file type. Upload a PNG or JPEG photo." if kind is None else
                       f"{kind.upper()} files are not accepted. Upload a PNG or JPEG photo.")
        e.status = 415
        raise e
    dims = header_size(data, kind)
    if dims is None:
        raise InputError(f"The {kind.upper()} header could not be read; the file may be corrupted or truncated.")
    w, h = dims
    if w * h > limits.max_pixels:
        raise InputError(f"The photo is {w} x {h} px ({w * h / 1e6:.0f} MP); the limit is {limits.max_pixels / 1e6:.0f} MP.")
    if min(w, h) < limits.min_side:
        raise InputError(f"The photo is {w} x {h} px; at least {limits.min_side} px on the short side is needed.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(f"The {kind.upper()} file could not be decoded; it may be corrupted.")
    if img.shape[:2] != (h, w):
        raise InputError("The image header and its pixel data disagree; the file may be corrupted.")
    return img, kind


def pixel_hash(img):
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def load_artifact(path=DEFAULT_ARTIFACT):
    with open(path, "rb") as fh:
        art = pickle.load(fh)
    if art.get("version") != ARTIFACT_VERSION:
        raise RuntimeError(f"model artifact version {art.get('version')} does not match {ARTIFACT_VERSION}")
    return art


def find_frame(img, corners=None, limits: Limits = Limits()):
    """Manual corners bypass detection. Otherwise detect on a size-capped copy
    and scale the corners back to the original resolution."""
    h, w = img.shape[:2]
    if corners is not None:
        c = np.asarray(corners, float).reshape(4, 2)
        if not ((c[:, 0] >= 0) & (c[:, 0] < w) & (c[:, 1] >= 0) & (c[:, 1] < h)).all():
            raise FrameError("Manual corners must lie inside the photo.")
        f = Frame(c, (w, h), "manual", {"warnings": []})
        if cv2.contourArea(c.astype(np.float32)) < 0.02 * w * h:
            raise FrameError("The manual frame covers less than 2% of the photo; check the corner order (TL, TR, BR, BL).")
        return f
    s = min(1.0, limits.detect_max_side / max(h, w))
    small = img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    try:
        f, _ = detect(small)
    except RuntimeError as e:
        raise FrameError(f"No quadrat frame was found automatically ({e}). Mark the four inner corners by hand.")
    q = f.quality
    if s != 1.0:
        f = f.scaled_to(w, h)
        f.quality = q
    return f


def generate(rect, art):
    p = CandidateParams(**art["candidate_params"])
    maps = score_maps(rect, p)
    g = art["generator"]
    if g == "ringlog":
        c = detect_candidates(rect, p, maps)
    elif g == "hough":
        c = hough_candidates(rect, maps, p)
    elif g == "union":
        c, _ = merge_candidates(detect_candidates(rect, p, maps), hough_candidates(rect, maps, p))
    else:
        raise ValueError(g)
    return c, maps


def features_for(rect, c, maps, art):
    X, names = candidate_features(rect, c, maps, extra=art["extra_features"])
    if names != art["feature_names"]:
        raise RuntimeError("feature layout differs from the one the model was trained with")
    if art.get("per_photo_standardize"):
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return X


def run(img, art, corners=None, k_calibration=80, seed=0, limits: Limits = Limits(), name="upload"):
    """Process one photo. Returns (payload for the viewer, detection records, timings)."""
    t = {}
    t0 = time.time()
    frame = find_frame(img, corners, limits)
    t["frame_s"] = time.time() - t0
    t0 = time.time()
    rect = rectify(img, frame, size=art["rect_size"])
    c, maps = generate(rect, art)
    t["candidates_s"] = time.time() - t0
    if len(c["x"]) == 0:
        raise FrameError("No candidate barnacles were found inside the frame; check the frame corners.")
    t0 = time.time()
    X = features_for(rect, c, maps, art)
    del maps
    t["features_s"] = time.time() - t0
    t0 = time.time()
    score = art["model"].predict_proba(X)[:, 1]
    t["classify_s"] = time.time() - t0
    s_px = rect.px_per_rect_px()
    xy = rect.rect_to_image(np.c_[c["x"], c["y"]])
    uv = rect.rect_to_uv(np.c_[c["x"], c["y"]])
    r = c["radius"] * s_px
    dup = DuplicateIndex(xy, r, art["dup_frac"])
    thr = float(art["default_threshold"])
    mset = select(dup, score, thr)
    rng = np.random.default_rng(seed)
    sidx, sw, sstrat = stratified_sample(score, k_calibration, rng)
    acc = np.zeros(len(xy), bool)
    acc[mset] = True
    sig = static_signals(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), frame, xy[:, 0], xy[:, 1], r, acc, c["score"])
    spec = GridSpec(10, 10)
    rr, cc, ins = assign(uv, spec)
    ids = [f"{name}_d{i:05d}" for i in range(len(xy))]
    records = [{"id": ids[i], "x": float(xy[i, 0]), "y": float(xy[i, 1]), "u": float(uv[i, 0]), "v": float(uv[i, 1]),
                "cell_id_10x10": cell_id(rr[i], cc[i]) if ins[i] else None, "score": float(score[i]),
                "radius_px": float(r[i]), "bbox_xywh": [float(xy[i, 0] - r[i]), float(xy[i, 1] - r[i]),
                                                        float(2 * r[i]), float(2 * r[i])],
                "review_status": "unreviewed"} for i in mset]
    payload = viewer_payload(img, frame, name, ids, xy, uv, r, score, sig, dup, thr, sidx, sw, sstrat, art)
    payload["timing_s"] = {k: round(v, 2) for k, v in t.items()}
    payload["image_size"] = [int(img.shape[1]), int(img.shape[0])]
    payload["training_photo"] = recognize(img)      # None, or {level, photo, message, definitive, evidence}
    return payload, records, t


def viewer_payload(img, frame, name, ids, xy, uv, r, score, sig, dup, thr, sidx, sw, sstrat, art, pad=60, max_px=1400):
    H, W = img.shape[:2]
    cr = frame.corners
    x0, y0 = max(int(cr[:, 0].min()) - pad, 0), max(int(cr[:, 1].min()) - pad, 0)
    x1, y1 = min(int(cr[:, 0].max()) + pad, W), min(int(cr[:, 1].max()) + pad, H)
    crop = img[y0:y1, x0:x1]
    s = min(1.0, max_px / max(crop.shape[:2]))
    disp = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 86])
    pairs = sorted({(int(min(a, b)), int(max(a, b))) for a in range(len(xy)) for b in dup.nbrs[a]})
    q = frame.quality or {}
    return {
        "name": name, "file": name, "img": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
        "crop": {"x0": x0, "y0": y0, "scale": s, "w": disp.shape[1], "h": disp.shape[0]},
        "H": frame.H.tolist(), "Hinv": frame.H_inv.tolist(), "corners": frame.corners.tolist(), "rows": 10, "cols": 10,
        "c": {"id": ids, "x": [round(float(v), 2) for v in xy[:, 0]], "y": [round(float(v), 2) for v in xy[:, 1]],
              "u": [round(float(v), 6) for v in uv[:, 0]], "v": [round(float(v), 6) for v in uv[:, 1]],
              "r": [round(float(v), 2) for v in r], "s": [float(v) for v in score],
              "cvp": [round(float(v), 4) for v in sig["cv_pct"]], "clar": sig["low_clarity"].astype(int).tolist(),
              "dense": sig["dense"].astype(int).tolist(), "edge": sig["near_frame"].astype(int).tolist()},
        "pairs": [v for p in pairs for v in p], "ann": [], "cov": None,
        "op": {"thr": thr, "a": 1.0, "b": 0.0, "calibrated": False,
               "method": "Default threshold from the training photos (not calibrated for this photo)",
               "n_hat": None, "ht": None, "ht_se": None,
               "sample": [[int(i), None, float(w), int(sstrat[i])] for i, w in zip(sidx, sw)],
               "stratum_sizes": np.bincount(sstrat, minlength=int(sstrat.max()) + 1).tolist()},
        "model": {"trained_on": ", ".join(art["trained_on"]), "kind": art["model_kind"], "version": art["version"],
                  "variant": art["variant"]},
        "frame": {"source": frame.source, "warnings": q.get("warnings", []), "confirmed": frame.source == "manual",
                  "corners_px": {n: [round(float(a), 2), round(float(b), 2)] for n, (a, b) in zip(CORNER_NAMES, frame.corners)}},
        "upload": True,
    }
