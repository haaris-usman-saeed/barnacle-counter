"""High-resolution tiling: coordinate mapping, exclusive tile ownership and deduplication."""
import unittest

import cv2
import numpy as np

from barnacles.frame import Frame
from barnacles.hires import DETAIL_MARGIN, DETAIL_SIDE, iter_detail_tiles, use_detail
from barnacles.ml import DuplicateIndex, select


class TestHighResolutionTiling(unittest.TestCase):
    @staticmethod
    def frame(n):
        corners = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], float)
        return Frame(corners, (n, n), "manual")

    def test_detail_gating_uses_frame_not_upload_dimensions(self):
        self.assertFalse(use_detail(self.frame(2000)))
        self.assertTrue(use_detail(self.frame(2800)))

    def test_four_tile_cores_cover_region_exactly_once(self):
        img = np.zeros((2800, 2800, 3), np.uint8)
        frame = self.frame(2800)
        tiles = iter_detail_tiles(img, frame)
        coverage = np.zeros((DETAIL_SIDE, DETAIL_SIDE), np.uint8)
        count = 0
        for tile in tiles:
            count += 1
            self.assertEqual(tile.image.shape[:2], (1600 + 2 * DETAIL_MARGIN,) * 2)
            y, x = np.mgrid[0:1600, 0:1600]
            self.assertTrue(tile.in_core(x + DETAIL_MARGIN, y + DETAIL_MARGIN).all())
            coverage[tile.y0:tile.y0 + 1600, tile.x0:tile.x0 + 1600] += 1
        self.assertEqual(count, 4)
        self.assertTrue((coverage == 1).all())

    def test_orig_and_detail_coordinates_roundtrip(self):
        img = np.zeros((2800, 2800, 3), np.uint8)
        frame = self.frame(2800)
        pts = np.array([[200.1, 250.3], [1350.0, 900.5], [2790.0, 1800.0]])
        for tile in iter_detail_tiles(img, frame):
            local = tile.image_to_rect(pts)
            np.testing.assert_allclose(tile.rect_to_image(local), pts, atol=0.0001)
            uv = tile.rect_to_uv(local)
            np.testing.assert_allclose(uv, frame.to_norm(pts), atol=0.0001)

    def test_global_duplicates_across_reference_and_tiles(self):
        xy = np.array([[10.0, 10], [10.3, 10.1], [50, 50], [500, 500]], float)
        radius = np.array([6, 6, 6, 6], float)
        scores = np.array([0.7, 0.9, 0.8, 0.4])
        ids = select(DuplicateIndex(xy, radius, frac=1.5), scores, 0.5)
        self.assertEqual(set(ids.tolist()), {1, 2})


if __name__ == "__main__":
    unittest.main()
