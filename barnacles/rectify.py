"""Canonical rectified frame: all detection works in this space.

The inner frame is warped to a square of `size` pixels (plus a margin so
patches near the wire are complete). Working here removes perspective,
fixes the scale relative to the frame, and makes the pipeline independent
of camera resolution. Results are mapped back to raw-image pixels and to
normalized (u, v) through the frame homography.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Rectified:
    image: np.ndarray      # BGR, (size + 2 margin) square
    size: int
    margin: int
    M: np.ndarray          # raw image px -> rectified px
    frame: object

    def uv_to_rect(self, uv):
        return np.asarray(uv, float) * self.size + self.margin

    def rect_to_uv(self, xy):
        return (np.asarray(xy, float) - self.margin) / self.size

    def rect_to_image(self, xy):
        return self.frame.to_image(self.rect_to_uv(xy))

    def image_to_rect(self, xy):
        return self.uv_to_rect(self.frame.to_norm(xy))

    def px_per_rect_px(self):
        """Mean raw-image pixels per rectified pixel (for reporting sizes)."""
        side = np.mean([np.linalg.norm(self.frame.corners[(i + 1) % 4] - self.frame.corners[i])
                        for i in range(4)])
        return side / self.size


def rectify(img, frame, size=1600, margin=48):
    T = np.array([[size, 0, margin], [0, size, margin], [0, 0, 1]], float)
    M = T @ frame.H
    out = cv2.warpPerspective(img, M, (size + 2 * margin, size + 2 * margin),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Rectified(out, size, margin, M, frame)


@dataclass
class NativeRectified:
    """Native-pixel close-up view with reflected context (no magnification).

    Candidate radii retain the trained 1600-reference pixel values. Only the
    photographed region plus a reflection halo is processed; this is equivalent
    to ignoring a large empty canvas, without artificial black-border edges.
    """
    image: np.ndarray
    frame: object
    size: int = 1600
    margin: int = 88

    def uv_to_rect(self, uv):
        return self.frame.to_image(uv) + self.margin

    def rect_to_uv(self, xy):
        return self.frame.to_norm(self.rect_to_image(xy))

    def rect_to_image(self, xy):
        return np.asarray(xy, float) - self.margin

    def image_to_rect(self, xy):
        return np.asarray(xy, float) + self.margin

    def px_per_rect_px(self):
        return 1.0


def native_rectify(img, frame, size=1600, margin=88):
    """Reflect pixels outside the real crop to preserve local feature context.

    The recorded ROI itself is never resized. Proposals outside it are excluded
    by the existing rect_to_uv/inside logic in candidate generation.
    """
    out = cv2.copyMakeBorder(img, margin, margin, margin, margin,
                             cv2.BORDER_REFLECT_101)
    return NativeRectified(out, frame, size, margin)
