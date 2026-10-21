import json
import unittest

import cv2
import numpy as np

from tests.helpers import ROOT, DATA, synthetic_quadrat
from barnacles.frame import Frame, detect, frame_from_config, CORNER_NAMES


class TestCoordinateConversions(unittest.TestCase):
    def setUp(self):
        self.f = Frame([[100, 120], [900, 100], [920, 880], [90, 860]], (1000, 1000), "manual")

    def test_corners_map_to_unit_square(self):
        np.testing.assert_allclose(self.f.to_norm(self.f.corners), [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-9)

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        uv = rng.uniform(-0.2, 1.2, (500, 2))
        np.testing.assert_allclose(self.f.to_norm(self.f.to_image(uv)), uv, atol=1e-9)

    def test_resolution_invariance(self):
        xy = np.array([[300.5, 400.25], [850, 700], [101, 121]])
        for s in (0.5, 0.37, 2.0):
            g = self.f.scaled_to(1000 * s, 1000 * s)
            np.testing.assert_allclose(g.to_norm(xy * s), self.f.to_norm(xy), atol=1e-7)

    def test_inside_tests_agree(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform(0, 1000, (2000, 2))
        uv = self.f.to_norm(xy)
        inside_h = ((uv >= 0) & (uv <= 1)).all(1)
        sd = self.f.signed_distance_px(xy)
        clear = np.abs(sd) > 1e-6
        np.testing.assert_array_equal(inside_h[clear], self.f.contains(xy)[clear])


class TestFrameDetection(unittest.TestCase):
    def test_synthetic_axis_aligned(self):
        img, truth = synthetic_quadrat()
        f, _ = detect(img)
        self.assertLess(np.abs(f.corners - truth).max(), 2.0)
        self.assertFalse(f.quality["selection"]["used_seed"])

    def test_synthetic_perspective(self):
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [1199, 0], [1199, 899], [0, 899]]),
                                         np.float32([[30, 10], [1170, 40], [1150, 880], [10, 860]]))
        img, truth = synthetic_quadrat(seed=3, warp=Hm)
        f, _ = detect(img)
        self.assertLess(np.abs(f.corners - truth).max(), 2.5)

    def test_no_frame_raises(self):
        img = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            detect(img)

    def test_manual_config_bypasses_and_rescales(self):
        entry = {"corners_px": {"TL": [10, 10], "TR": [110, 10], "BR": [110, 110], "BL": [10, 110]},
                 "image_size": [200, 200], "source": "manual"}
        f = frame_from_config(entry, (400, 400))
        self.assertEqual(f.source, "manual")
        np.testing.assert_allclose(f.corners[0], [20, 20])
        np.testing.assert_allclose(f.to_norm([[220, 220]]), [[1, 1]], atol=1e-9)

    @unittest.skipUnless((DATA / "img1.png").exists(), "challenge photos not available")
    def test_real_photos_match_recorded_frames(self):
        cfg = json.loads((ROOT / "phase2_output" / "frame_config_detected.json").read_text())
        for name in ("img1.png", "img2.png"):
            img = cv2.imread(str(DATA / name), cv2.IMREAD_COLOR)
            f, _ = detect(img)
            ref = np.array([cfg[name]["corners_px"][n] for n in CORNER_NAMES])
            self.assertLess(np.abs(f.corners - ref).max(), 0.05, name)


if __name__ == "__main__":
    unittest.main()
