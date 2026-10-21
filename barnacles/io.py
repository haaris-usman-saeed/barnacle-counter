"""Loading and validating raw photographs and contour masks."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MaskData:
    stroke: np.ndarray        # bool (H, W): True where an annotation stroke was drawn
    color_index: np.ndarray   # int16 (H, W): palette index per stroke pixel, -1 elsewhere
    palette: list             # list of RGB tuples, ordered by pixel count
    path: str


def load_image(path) -> np.ndarray:
    """Load a photo as 3-channel BGR uint8, dropping alpha if present."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_mask(path) -> MaskData:
    """Load an RGBA contour mask and return a binary stroke map.

    The supplied masks are hard-edged: alpha is exactly 0 (white background)
    or 255 (blue stroke). Alpha is therefore the cleanest stroke signal. We
    fall back to a "not white" test if a mask has no alpha channel. Any
    violation of the binary assumption raises, so a re-exported mask with
    anti-aliasing cannot silently change results.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[..., 3]
        if not np.isin(alpha, (0, 255)).all():
            raise ValueError(f"{path}: alpha is not binary; mask may be anti-aliased")
        stroke = alpha > 0
        bgr = raw[..., :3]
    else:
        bgr = raw if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        stroke = ~np.all(bgr >= 250, axis=2)

    colors, inverse, counts = np.unique(
        bgr[stroke].reshape(-1, 3), axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(-counts)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    color_index = np.full(stroke.shape, -1, np.int16)
    color_index[stroke] = rank[inverse.ravel()]
    palette = [tuple(int(c) for c in colors[i][::-1]) for i in order]  # RGB
    return MaskData(stroke=stroke, color_index=color_index, palette=palette, path=str(path))


def check_alignment(img: np.ndarray, mask: MaskData) -> dict:
    """Raw photo and mask must share a pixel grid (same H, W)."""
    ok = img.shape[:2] == mask.stroke.shape
    if not ok:
        raise ValueError(f"shape mismatch: image {img.shape[:2]} vs mask {mask.stroke.shape}")
    return {"height": int(img.shape[0]), "width": int(img.shape[1])}


def locate_crop(full: np.ndarray, crop: np.ndarray) -> dict:
    """Find where an overlay crop sits inside the full photo (verification only).

    Used to audit the provided overlays, never to derive coordinates.
    Returns the offset and the fraction of crop pixels identical to the photo.
    """
    g_full = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
    g_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(g_full, g_crop, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x0, y0) = cv2.minMaxLoc(res)
    h, w = crop.shape[:2]
    same = np.all(full[y0:y0 + h, x0:x0 + w] == crop, axis=2)
    return {"x0": int(x0), "y0": int(y0), "w": int(w), "h": int(h),
            "match_score": float(score), "identical_px_frac": float(same.mean()),
            "changed": ~same}
