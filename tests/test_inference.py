"""New-upload pipeline: validation, frame handling, real inference, coordinates."""
import unittest

import cv2
import numpy as np

from tests.helpers import ROOT, DATA, HAVE_DATA, synthetic_quadrat
from barnacles.inference import (InputError, FrameError, Limits, decode_image, find_frame, choose_frame_and_mode, run, load_artifact,
                                 DEFAULT_ARTIFACT, header_size)
from barnacles.grid import GridSpec, assign

HAVE = HAVE_DATA and DEFAULT_ARTIFACT.exists()


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


class TestUploadValidation(unittest.TestCase):
    def test_rejects_bad_inputs(self):
        with self.assertRaises(InputError):
            decode_image(b"")
        with self.assertRaises(InputError) as e:
            decode_image(b"GIF89a not supported")
        self.assertEqual(e.exception.status, 415)
        with self.assertRaises(InputError) as e:
            decode_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        self.assertEqual(e.exception.status, 400)            # PNG signature but undecodable
        with self.assertRaises(InputError):
            decode_image(png(np.zeros((200, 200, 3), np.uint8)))   # too small
        with self.assertRaises(InputError) as e:
            decode_image(png(np.zeros((700, 700, 3), np.uint8)), Limits(max_bytes=10))
        self.assertEqual(e.exception.status, 413)

    def test_accepts_500_square_and_rejects_499(self):
        """Small cropped photos are valid uploads; frame detection is separate."""
        small = np.zeros((500, 500, 3), np.uint8)
        out, kind = decode_image(png(small))
        self.assertEqual(out.shape, (500, 500, 3))
        self.assertEqual(kind, "png")
        frame = find_frame(out, corners=[[0, 0], [499, 0], [499, 499], [0, 499]])
        self.assertEqual(frame.source, "manual")
        with self.assertRaises(InputError):
            decode_image(png(np.zeros((499, 500, 3), np.uint8)))

    def test_auto_mode_routes_small_unframed_photo_to_native(self):
        img = np.random.default_rng(222).integers(0, 255, (500, 500, 3), dtype=np.uint8)
        f, mode, reason = choose_frame_and_mode(img)
        self.assertEqual((mode, f.source), ("native", "crop"))
        self.assertIn("Automatically", reason)
        f2, mode2, _ = choose_frame_and_mode(img, mode="native")
        self.assertEqual((mode2, f2.source), ("native", "crop"))
        with self.assertRaises(InputError):
            choose_frame_and_mode(img, mode="unknown")

    def test_auto_does_not_silently_crop_large_unframed_photo(self):
        img = np.random.default_rng(223).integers(0, 255, (1400, 1400, 3), dtype=np.uint8)
        with self.assertRaises(FrameError):
            choose_frame_and_mode(img)

    def test_500_px_png_and_native_coordinate_mapping(self):
        from barnacles.rectify import native_rectify
        from barnacles.frame import Frame
        img = np.random.default_rng(10).integers(0, 255, (500, 500, 3), dtype=np.uint8)
        dec, _ = decode_image(png(img))
        self.assertEqual(dec.shape, img.shape)
        frame = Frame(np.array([[0,0], [499,0], [499,499], [0,499]], float), (500,500), "crop")
        rect = native_rectify(dec, frame)
        self.assertEqual(rect.size, 1600)
        self.assertEqual(rect.image.shape[:2], (676, 676))
        np.testing.assert_array_equal(rect.image[88:588, 88:588], img)
        pts = np.array([[10.,20.], [250.,250.], [499.,499.]])
        np.testing.assert_allclose(rect.rect_to_image(rect.image_to_rect(pts)), pts)
        np.testing.assert_allclose(rect.rect_to_uv(rect.image_to_rect(pts)), frame.to_norm(pts))
        with self.assertRaises(InputError):
            decode_image(png(np.zeros((499, 500, 3), np.uint8)))

    def test_rejects_other_formats_and_bombs(self):
        img = np.zeros((700, 900, 3), np.uint8)
        for ext in (".tiff", ".bmp"):
            with self.assertRaises(InputError) as e:
                decode_image(cv2.imencode(ext, img)[1].tobytes())
            self.assertEqual(e.exception.status, 415)
        # a PNG whose header claims 100000 x 100000 px is refused before decoding
        data = bytearray(png(img))
        data[16:20] = (100000).to_bytes(4, "big")
        data[20:24] = (100000).to_bytes(4, "big")
        self.assertEqual(header_size(bytes(data), "png"), (100000, 100000))
        with self.assertRaises(InputError) as e:
            decode_image(bytes(data))
        self.assertIn("MP", str(e.exception))

    def test_header_size_matches_decoder(self):
        img = np.random.default_rng(1).integers(0, 255, (701, 905, 3), dtype=np.uint8)
        self.assertEqual(header_size(png(img), "png"), (905, 701))
        self.assertEqual(header_size(cv2.imencode(".jpg", img)[1].tobytes(), "jpeg"), (905, 701))

    def test_accepts_png_and_jpeg(self):
        img = np.random.default_rng(0).integers(0, 255, (700, 900, 3), dtype=np.uint8)
        for ext in (".png", ".jpg"):
            out, kind = decode_image(cv2.imencode(ext, img)[1].tobytes())
            self.assertEqual(out.shape, img.shape)
            self.assertIn(kind, ("png", "jpeg"))

    def test_no_frame_raises_frame_error(self):
        img = np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8)
        with self.assertRaises(FrameError):
            find_frame(img)

    def test_manual_corners_validated(self):
        img = np.zeros((800, 1000, 3), np.uint8)
        with self.assertRaises(FrameError):
            find_frame(img, corners=[[0, 0], [2000, 0], [2000, 900], [0, 900]])   # outside the photo
        with self.assertRaises(FrameError):
            find_frame(img, corners=[[10, 10], [20, 10], [20, 20], [10, 20]])     # implausibly small
        f = find_frame(img, corners=[[100, 100], [900, 100], [900, 700], [100, 700]])
        self.assertEqual(f.source, "manual")


@unittest.skipUnless(DEFAULT_ARTIFACT.exists(), "model artifact not available")
class TestSyntheticInference(unittest.TestCase):
    """End-to-end inference on a synthetic quadrat photo: runs without the development data."""

    def test_synthetic_native_closeup(self):
        img = np.zeros((500, 500, 3), np.uint8)
        img[:] = (65, 65, 65)
        for y in range(48, 450, 30):
            for x in range(48, 450, 30):
                cv2.circle(img, (x,y), 10, (220,220,220), -1)
                cv2.circle(img, (x,y), 5, (85,85,85), -1)
        payload, records, timing = run(img, load_artifact(), mode="native", name="synthetic_crop")
        self.assertEqual(payload["processing_mode"], "native")
        self.assertEqual(payload["frame"]["source"], "crop")
        self.assertTrue(payload["frame"]["confirmed"])
        self.assertGreater(len(payload["c"]["x"]), 0)
        for c in records:
            self.assertTrue(0 <= c["x"] <= 499 and 0 <= c["y"] <= 499)
        with self.assertRaises(InputError):
            run(img, load_artifact(), mode="native", corners=[[0,0], [499,0], [499,499], [0,499]])

    def test_synthetic_photo(self):
        img, truth = synthetic_quadrat(seed=4, H=1200, W=1600)
        rng = np.random.default_rng(0)
        for _ in range(60):                                   # pale ring-shaped "barnacles" inside the frame
            x, y = rng.uniform(372, 828), rng.uniform(222, 678)          # inside the 500 px opening
            cv2.circle(img, (int(x), int(y)), 9, (220, 220, 220), -1)
            cv2.circle(img, (int(x), int(y)), 4, (90, 90, 90), -1)
        payload, records, t = run(img, load_artifact(), name="synthetic")
        self.assertLess(np.abs(np.array(payload["corners"]) - truth).max(), 3.0)
        self.assertEqual(payload["ann"], [])
        self.assertEqual(len({r["id"] for r in records}), len(records))
        for r in records:
            self.assertTrue(0 <= r["u"] <= 1 and 0 <= r["v"] <= 1)
        self.assertIsNone(payload["training_photo"])


@unittest.skipUnless(HAVE, "development photo or model artifact not available")
class TestRealInference(unittest.TestCase):
    """img2 processed as an ordinary upload: no masks or annotations are read."""

    @classmethod
    def setUpClass(cls):
        cls.art = load_artifact()
        full = cv2.imread(str(DATA / "img2.png"), cv2.IMREAD_COLOR)
        cls.full = full
        cls.half = cv2.resize(full, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img, _ = decode_image(png(cls.half))
        cls.payload, cls.records, _ = run(img, cls.art, name="half")
        cls.full_frame = find_frame(full)

    def test_payload_has_no_annotations(self):
        self.assertEqual(self.payload["ann"], [])
        self.assertIsNone(self.payload["cov"])
        self.assertFalse(self.payload["op"]["calibrated"])

    def test_detections_valid_and_unique(self):
        r = self.records
        self.assertGreater(len(r), 50)
        self.assertEqual(len({d["id"] for d in r}), len(r))
        h, w = self.half.shape[:2]
        for d in r:
            self.assertTrue(0 <= d["x"] < w and 0 <= d["y"] < h)
            self.assertTrue(0 <= d["u"] <= 1 and 0 <= d["v"] <= 1)
            self.assertEqual(d["review_status"], "unreviewed")
        uv = np.array([[d["u"], d["v"]] for d in r])
        for rows, cols in ((10, 10), (3, 7), (25, 25)):
            rr, cc, ins = assign(uv, GridSpec(rows, cols))
            self.assertTrue(ins.all())                         # every detection in exactly one cell
            self.assertEqual(np.bincount(rr * cols + cc, minlength=rows * cols).sum(), len(r))

    def test_calibration_sample_exact(self):
        self.assertEqual(len(self.payload["op"]["sample"]), 80)

    def test_coordinates_reconstruct_across_resolution(self):
        # frame found on the half-size upload, scaled back, matches the full-size frame
        c_half = np.array(self.payload["corners"]) * 2
        self.assertLess(np.abs(c_half - self.full_frame.corners).max(), 4.0)
        # detections in half-size pixels, scaled to full size, keep their (u, v)
        xy_full = np.array([[d["x"], d["y"]] for d in self.records]) * 2
        uv_full = self.full_frame.to_norm(xy_full)
        uv_half = np.array([[d["u"], d["v"]] for d in self.records])
        self.assertLess(np.abs(uv_full - uv_half).max(), 0.004)

    def test_training_photo_flag(self):
        from barnacles.fingerprint import recognize
        img, _ = decode_image(png(self.full))
        self.assertEqual(recognize(img)["level"], "exact")
        # the half-size upload is recognized as a likely copy, and is not presented as proof
        self.assertEqual(self.payload["training_photo"]["level"], "likely")
        self.assertFalse(self.payload["training_photo"]["definitive"])

if __name__ == "__main__":
    unittest.main()
