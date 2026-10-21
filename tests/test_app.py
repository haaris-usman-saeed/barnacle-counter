"""HTTP layer: validation, frame failure with manual fallback, evaluation, health."""
import io
import json
import unittest

import cv2
import numpy as np

from tests.helpers import DATA, HAVE_DATA
from barnacles.inference import DEFAULT_ARTIFACT


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


@unittest.skipUnless(DEFAULT_ARTIFACT.exists(), "model artifact not available")
class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as appmod
        cls.c = appmod.app.test_client()

    def setUp(self):
        import app as appmod
        appmod._hits.clear()                      # rate limiter is per process; reset between tests

    def post(self, data, **form):
        return self.c.post("/api/process", data={"image": (io.BytesIO(data), "photo.png"), **form},
                           content_type="multipart/form-data")

    def test_index_and_health(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        h = self.c.get("/api/health").get_json()
        self.assertEqual(h["status"], "ok")
        self.assertIn("img1.png", h["trained_on"])

    def test_unsupported_and_corrupt(self):
        self.assertEqual(self.post(b"GIF89a....").status_code, 415)
        self.assertEqual(self.post(b"\x89PNG\r\n\x1a\nbroken").status_code, 400)

    def test_no_frame_returns_preview_for_manual_corners(self):
        noise = np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8)
        r = self.post(png(noise))
        self.assertEqual(r.status_code, 422)
        j = r.get_json()
        self.assertIn("corners", j["error"])
        self.assertTrue(j["preview"]["img"].startswith("data:image/jpeg"))

    def test_bad_corners_rejected(self):
        noise = np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8)
        self.assertEqual(self.post(png(noise), corners="[[1,2],[3,4]]").status_code, 400)

    def test_evaluate(self):
        body = {"reference_csv": "x,y\n10,10\n50,50\n90,90\n", "radius_px": 3,
                "points": {"model": [[10, 11], [50, 50], [200, 200]]}}
        j = self.c.post("/api/evaluate", json=body).get_json()
        self.assertEqual(j["model"]["n_matched"], 2)
        self.assertEqual(j["model"]["count_minus_annotations"], 0)

    def test_rate_limit(self):
        import app as appmod
        old = appmod.RATE_N
        appmod.RATE_N = 2
        appmod._hits.clear()
        try:
            codes = [self.post(b"GIF89a").status_code for _ in range(3)]
            self.assertEqual(codes[:2], [415, 415])
            self.assertEqual(codes[2], 429)
        finally:
            appmod.RATE_N = old
            appmod._hits.clear()

    def test_busy_server_returns_503(self):
        import app as appmod
        appmod._hits.clear()
        old = appmod.QUEUE_WAIT_S
        appmod.QUEUE_WAIT_S = 0.05
        appmod.JOBS.acquire()
        try:
            img = np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8)
            r = self.post(png(img))
            self.assertEqual(r.status_code, 503)
            self.assertIn("Retry-After", r.headers)
        finally:
            appmod.JOBS.release()
            appmod.QUEUE_WAIT_S = old

    def test_internal_errors_are_generic(self):
        import app as appmod
        appmod._hits.clear()
        real = appmod.run
        appmod.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret internal path /srv/x"))
        try:
            img = np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8)
            r = self.post(png(img))
            self.assertEqual(r.status_code, 500)
            j = r.get_json()
            self.assertNotIn("secret", j["error"])
            self.assertTrue(j["request_id"])
        finally:
            appmod.run = real

    def test_security_headers_and_404(self):
        r = self.c.get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(self.c.get("/healthz").get_json()["status"], "alive")

    @unittest.skipUnless(HAVE_DATA, "development photo not available")
    def test_real_upload_end_to_end(self):
        img = cv2.imread(str(DATA / "img2.png"), cv2.IMREAD_COLOR)
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        r = self.post(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:300])
        j = r.get_json()
        self.assertEqual(j["status"], "ok")
        self.assertGreater(len(j["detections"]), 50)
        self.assertEqual(j["payload"]["format"], "jpeg")
        # manual corners equal to the detected ones bypass detection and reproduce the result
        r2 = self.post(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(),
                       corners=json.dumps(j["payload"]["corners"]))
        j2 = r2.get_json()
        self.assertEqual(j2["payload"]["frame"]["source"], "manual")
        self.assertEqual(len(j2["detections"]), len(j["detections"]))


if __name__ == "__main__":
    unittest.main()
