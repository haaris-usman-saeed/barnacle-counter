"""Barnacle counting web application.

    python app.py                                   # development: http://127.0.0.1:8000
    gunicorn -c gunicorn.conf.py app:app            # deployment (binds 0.0.0.0:$PORT)

Endpoints
  GET  /               review interface in application mode
  POST /api/process    multipart field "image" (PNG or JPEG); optional "corners" =
                       JSON [[x,y] x4] (TL, TR, BR, BL in photo pixels) to bypass frame
                       detection; optional "k" = calibration labels to sample (10..400)
  POST /api/evaluate   JSON {"reference_csv", "points": {name: [[x,y],...]}, "radius_px"}
  GET  /api/health     readiness: model loaded, version, limits
  GET  /healthz        liveness (no model needed)

Resource protection (all configurable by environment variable):
  MAX_CONCURRENT_JOBS  photos processed at once per process (default 1); others wait up
                       to QUEUE_WAIT_S seconds (default 20) and then get 503 + Retry-After
  RATE_LIMIT_REQUESTS / RATE_LIMIT_WINDOW_S   uploads per client per window (default 12 per
                       600 s), answered with 429 + Retry-After. In-memory, per process.
  TRUST_PROXY_HEADERS=1  use the first X-Forwarded-For hop as the client address (only
                       behind a proxy you control)
Uploads are held in memory only (never written to disk by this code), decoded only
by OpenCV, and never logged; logs carry a request id, sizes, dimensions and timings.
"""
import os

# Must be set before OpenCV is imported anywhere: caps the pixels its decoders allocate.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(70_000_000))

import base64  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import defaultdict, deque  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, Response, g, jsonify, request  # noqa: E402

from barnacles.inference import (DEFAULT_ARTIFACT, FrameError, InputError, Limits, decode_image,  # noqa: E402
                                 load_artifact, run)
from barnacles.evaluation import evaluate  # noqa: E402
from barnacles.review_viewer import render  # noqa: E402

LIMITS = Limits()
MAX_EVAL_BYTES = 5 * 1024 * 1024
MAX_EVAL_POINTS = 200_000
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("barnacles")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = LIMITS.max_bytes + 1024 * 1024

# ---- model: loaded once per process at import (gunicorn --preload shares it)
_MODEL = {"artifact": None, "error": None}


def _load_model():
    try:
        _MODEL["artifact"] = load_artifact(os.environ.get("BARNACLE_MODEL", DEFAULT_ARTIFACT))
        log.info("model loaded: %s", _MODEL["artifact"]["variant"])
    except Exception as e:  # noqa: BLE001
        _MODEL["error"] = f"{type(e).__name__}"
        log.error("model failed to load: %s", e)


_load_model()

# ---- resource protection
JOBS = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_JOBS", 1)))
QUEUE_WAIT_S = float(os.environ.get("QUEUE_WAIT_S", 20))
RATE_N = int(os.environ.get("RATE_LIMIT_REQUESTS", 12))
RATE_WINDOW = float(os.environ.get("RATE_LIMIT_WINDOW_S", 600))
_hits = defaultdict(deque)
_hits_lock = threading.Lock()


def client_key():
    if os.environ.get("TRUST_PROXY_HEADERS") == "1" and request.headers.get("X-Forwarded-For"):
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limited():
    """Sliding-window limit on uploads per client. Returns seconds to wait, or 0."""
    now, key = time.monotonic(), client_key()
    with _hits_lock:
        q = _hits[key]
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_N:
            return int(RATE_WINDOW - (now - q[0])) + 1
        q.append(now)
        if len(_hits) > 10_000:                      # bound memory: drop idle clients
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > RATE_WINDOW][:5000]:
                _hits.pop(k, None)
    return 0


def error(msg, status, **extra):
    body = {"error": msg, "request_id": getattr(g, "rid", None), **extra}
    r = jsonify(body)
    r.status_code = status
    return r


@app.before_request
def _start():
    g.rid = uuid.uuid4().hex[:12]
    g.t0 = time.time()


@app.after_request
def _finish(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Request-Id"] = getattr(g, "rid", "")
    if request.path.startswith("/api/"):
        log.info("rid=%s %s %s status=%s bytes_in=%s ms=%d", getattr(g, "rid", "-"), request.method, request.path,
                 resp.status_code, request.content_length, 1000 * (time.time() - getattr(g, "t0", time.time())))
    return resp


def preview(img, max_px=1400):
    s = min(1.0, max_px / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"img": "data:image/jpeg;base64," + base64.b64encode(buf).decode(), "scale": s,
            "w": int(small.shape[1]), "h": int(small.shape[0])}


def safe_name(fn):
    """Display label only; never used as a path."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(fn or "upload").stem)[:40] or "upload"


@app.get("/")
def index():
    return Response(render([], app=True), mimetype="text/html")


@app.get("/healthz")
def healthz():
    return jsonify({"status": "alive"})


@app.get("/api/health")
def health():
    a = _MODEL["artifact"]
    if a is None:
        return error("Model not loaded.", 503)
    return jsonify({"status": "ok", "model_version": a["version"], "variant": a["variant"], "trained_on": a["trained_on"],
                    "default_threshold": a["default_threshold"], "created": a["created"],
                    "limits": {"max_mb": LIMITS.max_bytes / 1e6, "max_megapixels": LIMITS.max_pixels / 1e6,
                               "formats": list(LIMITS.formats), "max_concurrent_jobs": JOBS._initial_value,
                               "rate_limit": f"{RATE_N} per {RATE_WINDOW:.0f} s"}})


@app.post("/api/process")
def process():
    if _MODEL["artifact"] is None:
        return error("The model is not available on this server.", 503)
    wait = rate_limited()
    if wait:
        r = error(f"Too many uploads from this address. Try again in {wait} s.", 429)
        r.headers["Retry-After"] = str(wait)
        return r
    f = request.files.get("image")
    if f is None:
        return error('Send the photo as multipart form field "image".', 400)
    try:
        data = f.read(LIMITS.max_bytes + 1)
    finally:
        f.close()                                       # releases any spooled temporary file
    name = safe_name(f.filename)
    try:
        corners = request.form.get("corners")
        corners = json.loads(corners) if corners else None
        if corners is not None and not (len(corners) == 4 and all(
                len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in corners)):
            raise ValueError
    except (ValueError, TypeError):
        return error("corners must be four [x, y] number pairs: top-left, top-right, bottom-right, bottom-left.", 400)
    try:
        k = int(request.form.get("k") or 80)
        if not 10 <= k <= 400:
            raise ValueError
    except ValueError:
        return error("k must be an integer between 10 and 400.", 400)
    try:
        img, kind = decode_image(data, LIMITS)
    except InputError as e:
        return error(str(e), e.status)
    finally:
        del data
    log.info("rid=%s upload kind=%s size=%dx%d manual_corners=%s", g.rid, kind, img.shape[1], img.shape[0], corners is not None)
    if not JOBS.acquire(timeout=QUEUE_WAIT_S):
        r = error("The server is busy processing other photos. Try again shortly.", 503)
        r.headers["Retry-After"] = "30"
        return r
    try:
        payload, records, t = run(img, _MODEL["artifact"], corners=corners, k_calibration=k, name=name)
    except FrameError as e:
        return error(str(e), 422, preview=preview(img), image_size=[int(img.shape[1]), int(img.shape[0])])
    except MemoryError:
        log.error("rid=%s out of memory", g.rid)
        return error("The server ran out of memory processing this photo. Try a smaller photo.", 503)
    except Exception:  # noqa: BLE001  pipeline failure: logged in full, reported generically
        log.exception("rid=%s processing failed", g.rid)
        return error("The model could not process this photo. Quote the request id if you report it.", 500)
    finally:
        JOBS.release()
    log.info("rid=%s detections=%d timing=%s", g.rid, len(records), {k_: round(v, 1) for k_, v in t.items()})
    payload["format"] = kind
    return jsonify({"status": "ok", "payload": payload, "detections": records})


def parse_points(text):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("the CSV is empty")
    head = [h.strip().lower() for h in rows[0]]
    if "x" in head and "y" in head:
        ix, iy, body = head.index("x"), head.index("y"), rows[1:]
    else:
        ix, iy, body = 0, 1, rows
    pts = [[float(r[ix]), float(r[iy])] for r in body if len(r) > max(ix, iy) and r[ix].strip()]
    if not pts:
        raise ValueError("no x,y points found")
    if len(pts) > MAX_EVAL_POINTS:
        raise ValueError(f"more than {MAX_EVAL_POINTS} points")
    a = np.array(pts)
    if not np.isfinite(a).all():
        raise ValueError("non-numeric coordinates")
    return a


@app.post("/api/evaluate")
def evaluate_points():
    if (request.content_length or 0) > MAX_EVAL_BYTES:
        return error("The comparison request is too large (5 MB limit).", 413)
    body = request.get_json(silent=True) or {}
    try:
        ref = parse_points(body.get("reference_csv", ""))
        radius = float(body["radius_px"])
        if not (0 < radius < 1000):
            raise ValueError("radius_px out of range")
        sets = {k: np.asarray(v, float).reshape(-1, 2) for k, v in (body.get("points") or {}).items()}
        if any(len(v) > MAX_EVAL_POINTS for v in sets.values()) or len(sets) > 4:
            raise ValueError("too many points")
    except (ValueError, KeyError, TypeError) as e:
        return error(f"Could not read the comparison request: {e}", 400)
    out = {"n_reference": int(len(ref)), "radius_px": radius,
           "assumption": "precision is exact only if the verified points are complete; otherwise it is a lower bound"}
    for k, pts in sets.items():
        m, _ = evaluate(pts, ref, radius)
        out[k] = {kk: m[kk] for kk in ("n_detections", "n_matched", "n_missed_annotations", "n_unmatched_detections",
                                        "count_minus_annotations", "annotated_recall", "annotation_precision_lower_bound",
                                        "f1_vs_annotations", "match_dist_median_px")}
    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return error(f"The upload exceeds the {LIMITS.max_bytes / 1e6:.0f} MB limit.", 413)


@app.errorhandler(404)
def not_found(_):
    return error("Not found.", 404)


@app.errorhandler(500)
def internal(_):
    return error("Internal server error.", 500)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)), threaded=True)
