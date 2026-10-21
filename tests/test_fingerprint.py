"""Training-photo recognition: exact, likely (resized / re-compressed) and possible matches."""
import unittest

import cv2
import numpy as np

from tests.helpers import DATA, HAVE_DATA, synthetic_quadrat
from barnacles.fingerprint import fingerprint, recognize, load_refs


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


class TestSynthetic(unittest.TestCase):
    """Runs without the development photos."""

    def setUp(self):
        # smoothed so it has photo-like low-frequency structure (pixel noise makes pHash unstable,
        # which only ever downgrades a match from "likely" to "possible")
        self.a = cv2.GaussianBlur(synthetic_quadrat(seed=1)[0], (0, 0), 6)
        self.refs = {"a.png": fingerprint(self.a)}

    def test_levels(self):
        self.assertEqual(recognize(self.a, self.refs)["level"], "exact")
        self.assertTrue(recognize(self.a, self.refs)["definitive"])
        r = recognize(jpeg(cv2.resize(self.a, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), 70), self.refs)
        self.assertEqual(r["level"], "likely")
        self.assertFalse(r["definitive"])
        self.assertIn("not proof", r["message"])
        noise = np.random.default_rng(5).integers(0, 255, self.a.shape, dtype=np.uint8)
        self.assertIsNone(recognize(noise, self.refs))

    def test_no_refs(self):
        self.assertIsNone(recognize(self.a, {}))


@unittest.skipUnless(HAVE_DATA, "development photos not available (set BARNACLE_DATA)")
class TestDevelopmentPhotos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = load_refs()
        cls.imgs = {n: cv2.imread(str(DATA / n), cv2.IMREAD_COLOR) for n in ("img1.png", "img2.png")}

    def test_shipped_fingerprints_match_the_photos(self):
        for n, im in self.imgs.items():
            r = recognize(im, self.refs)
            self.assertEqual((r["level"], r["photo"]), ("exact", n))

    def test_resized_and_recompressed_copies_are_likely(self):
        for n, im in self.imgs.items():
            for t in (cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                      cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA),
                      jpeg(im, 90), jpeg(im, 40), np.clip(im * 0.8, 0, 255).astype(np.uint8)):
                r = recognize(t, self.refs)
                self.assertEqual((r["level"], r["photo"]), ("likely", n))

    def test_small_crop_is_at_most_possible_and_never_the_other_photo(self):
        for n, im in self.imgs.items():
            h, w = im.shape[:2]
            r = recognize(im[int(0.025 * h):int(0.975 * h), int(0.025 * w):int(0.975 * w)], self.refs)
            if r is not None:
                self.assertEqual(r["photo"], n)
                self.assertEqual(r["level"], "possible")

    def test_photos_do_not_match_each_other(self):
        r1 = recognize(self.imgs["img1.png"], {"img2.png": self.refs["img2.png"]})
        r2 = recognize(self.imgs["img2.png"], {"img1.png": self.refs["img1.png"]})
        self.assertIsNone(r1)
        self.assertIsNone(r2)


if __name__ == "__main__":
    unittest.main()
