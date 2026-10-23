"""Bounded tiled second-scale inference for high-resolution quadrats.

A single 1600px warp may erase small openings in a large physical frame.
This module provides 4 core tiles at a 3200px rectified scale, with real
neighboring pixels in each tile's context halo. Each tile still advertises
a 1600px reference scale to the existing candidate generator and classifier.
The inference pipeline maps all detections to the original photo and applies
one global duplicate-suppression pass across both scales and all four tiles.

This is experimental: new image scales still require independent validation.
"""
from dataclasses import dataclass

import numpy as np

from .rectify import rectify

REFERENCE_SIDE = 1600
DETAIL_SIDE = 3200
DETAIL_MARGIN = 88
MIN_FRAME_SIDE_FOR_DETAIL = 2400


def use_detail(frame, min_side=MIN_FRAME_SIDE_FOR_DETAIL):
    """Decide from the photographed quadrat's own pixel footprint, not the upload size."""
    corners = np.asarray(frame.corners, dtype=float)
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    return float(np.mean(sides)) >= min_side


@dataclass
class DetailTile:
    """A non-overlapping 1600px core with an 88px *real-image* context halo."""
    image: np.ndarray
    frame: object
    x0: int
    y0: int
    full_size: int = DETAIL_SIDE
    size: int = REFERENCE_SIDE
    margin: int = DETAIL_MARGIN

    def rect_to_uv(self, xy):
        xy = np.asarray(xy, dtype=float)
        return (xy - self.margin + [self.x0, self.y0]) / self.full_size

    def rect_to_image(self, xy):
        return self.frame.to_image(self.rect_to_uv(xy))

    def image_to_rect(self, xy):
        return self.frame.to_norm(xy) * self.full_size - [self.x0, self.y0] + self.margin

    def uv_to_rect(self, uv):
        return np.asarray(uv, dtype=float) * self.full_size - [self.x0, self.y0] + self.margin

    def px_per_rect_px(self):
        c = self.frame.corners
        sides = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
        return float(np.mean(sides)) / self.full_size

    def in_core(self, x, y):
        """Half-open ownership means no pixel belongs to two tiles."""
        lo, hi = self.margin, self.margin + self.size
        return (x >= lo) & (x < hi) & (y >= lo) & (y < hi)


def iter_detail_tiles(img, frame):
    """Warp at 2x reference resolution and process tiles without extra copies.

    Only a single 3200px perspective warp is allocated. Tile halos come from
    *adjacent photographed pixels*, not reflected interior tile edges. The
    outermost halo uses the same reflected boundary as ordinary rectification.
    """
    big = rectify(img, frame, size=DETAIL_SIDE, margin=DETAIL_MARGIN)
    extent = REFERENCE_SIDE + 2 * DETAIL_MARGIN
    for y0 in range(0, DETAIL_SIDE, REFERENCE_SIDE):
        for x0 in range(0, DETAIL_SIDE, REFERENCE_SIDE):
            crop = big.image[y0:y0 + extent, x0:x0 + extent]
            if crop.shape[:2] != (extent, extent):
                raise RuntimeError("high-resolution tile exceeds warped frame")
            yield DetailTile(crop, frame, x0, y0)
