import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from barnacles.paths import data_dir, has_development_data  # noqa: E402

DATA = data_dir()                      # <repo>/data or $BARNACLE_DATA; tests that need it skip without it
HAVE_DATA = has_development_data()


def synthetic_quadrat(seed=0, H=900, W=1200, bar=18, warp=None):
    """Noise background with a teal wire grid around a central square opening.
    Returns (image, true inner-frame corners TL, TR, BR, BL)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 160, (H, W, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    teal = (112, 118, 28)                      # BGR, hue ~ 90 in OpenCV units
    x0, y0, s = 350, 200, 500                  # inner opening
    for x in (x0 - bar - 250, x0 - bar, x0 + s, x0 + s + 250):
        cv2.rectangle(img, (x, 40), (x + bar, H - 40), teal, -1)
    for y in (y0 - bar - 150, y0 - bar, y0 + s, y0 + s + 150):
        cv2.rectangle(img, (40, y), (W - 40, y + bar), teal, -1)
    corners = np.array([[x0, y0], [x0 + s - 1, y0], [x0 + s - 1, y0 + s - 1], [x0, y0 + s - 1]], float)
    if warp is not None:
        img = cv2.warpPerspective(img, warp, (W, H), borderValue=(90, 90, 90))
        corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), warp).reshape(-1, 2)
    return img, corners
