"""Recognize uploads that may be the development (training) photos.

Three levels, deliberately conservative:
  exact     identical decoded pixels (SHA-256). Definitive.
  likely    same aspect ratio and very close perceptual hashes and thumbnail
            correlation: probably a resized or re-compressed copy.
  possible  looser similarity: may overlap with a training photo.
"likely" and "possible" are similarity evidence, never proof; the messages
say so. Crops of a small part of a photo, strong rotations or mirrored
copies are not reliably recognized.

Thresholds were set from transformed copies of the two development photos
versus each other and versus unrelated images (tests/test_fingerprint.py):
resized and re-compressed copies sit at pHash distance 0 to 2 and thumbnail
correlation 1.00; a 5% crop or a 3 degree rotation at 6 to 10 and 0.76 to
0.83; the two different development photos at 28 and 0.52; noise at 32+.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

REF_PATH = Path(__file__).resolve().parents[1] / "models" / "training_fingerprints.json"
THUMB = (64, 48)
LIKELY = {"phash": 8, "dhash": 10, "corr": 0.97, "aspect": 0.02}
POSSIBLE = {"phash": 12, "corr": 0.70, "aspect": 0.05}


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _bits_to_hex(bits):
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def phash(img):
    g = cv2.resize(_gray(img), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].ravel()
    return _bits_to_hex(d > np.median(d[1:]))


def dhash(img):
    g = cv2.resize(_gray(img), (9, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _bits_to_hex((g[:, 1:] > g[:, :-1]).ravel())


def thumb(img):
    t = cv2.resize(_gray(img), THUMB, interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-9)


def sha256(img):
    a = np.ascontiguousarray(img)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def fingerprint(img):
    h, w = img.shape[:2]
    return {"sha256": sha256(img), "aspect": w / h, "phash": phash(img), "dhash": dhash(img),
            "thumb": [round(float(v), 5) for v in thumb(img)]}


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def compare(fp, ref):
    return {"exact": fp["sha256"] == ref["sha256"],
            "phash": hamming(fp["phash"], ref["phash"]), "dhash": hamming(fp["dhash"], ref["dhash"]),
            "corr": float(np.dot(fp["thumb"], ref["thumb"])),
            "aspect_diff": abs(fp["aspect"] - ref["aspect"]) / ref["aspect"]}


def load_refs(path=REF_PATH):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


MESSAGES = {
    "exact": ("This is exactly one of the photos the model was trained on ({p}). Its predictions are in-sample: "
              "they will look better than on a new photo and cannot demonstrate that the model generalizes."),
    "likely": ("This looks like a resized or re-compressed copy of a training photo ({p}). If it is, its predictions "
               "are in-sample and cannot demonstrate generalization. This is based on image similarity, not proof."),
    "possible": ("This photo resembles a training photo ({p}) and may overlap with the training data. "
                 "Similarity alone does not prove it."),
}


def recognize(img, refs=None):
    """Best match against the training photos, or None."""
    refs = load_refs() if refs is None else refs
    if not refs:
        return None
    fp = fingerprint(img)
    best = None
    for name, ref in refs.items():
        c = compare(fp, ref)
        if c["exact"]:
            level = "exact"
        elif (c["aspect_diff"] <= LIKELY["aspect"] and c["phash"] <= LIKELY["phash"]
              and c["dhash"] <= LIKELY["dhash"] and c["corr"] >= LIKELY["corr"]):
            level = "likely"
        elif c["aspect_diff"] <= POSSIBLE["aspect"] and c["phash"] <= POSSIBLE["phash"] and c["corr"] >= POSSIBLE["corr"]:
            level = "possible"
        else:
            continue
        rank = {"exact": 3, "likely": 2, "possible": 1}[level]
        if best is None or rank > best[0] or (rank == best[0] and c["corr"] > best[1]["evidence"]["corr"]):
            best = (rank, {"level": level, "photo": name, "evidence": {k: (round(v, 4) if isinstance(v, float) else v)
                                                                         for k, v in c.items()},
                           "definitive": level == "exact", "message": MESSAGES[level].format(p=name)})
    return best[1] if best else None


def write_refs(photos, path=REF_PATH):
    """photos: {name: BGR image}."""
    Path(path).write_text(json.dumps({n: fingerprint(im) for n, im in photos.items()}))
