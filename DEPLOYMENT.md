# Deploying the barnacle app

## What runs in production

`app.py` (Flask) served by gunicorn with `gunicorn.conf.py`. The image or host needs only:
- `requirements.txt`;
- `barnacles/`;
- `models/barnacle_detector.pkl` and `models/training_fingerprints.json`;
- `app.py` and `gunicorn.conf.py`.

No data folder, training cache or annotations are needed.

## Run locally

```
pip install -r requirements.txt
python app.py                              # Flask development server, http://127.0.0.1:8000
gunicorn -c gunicorn.conf.py app:app       # production server, binds 0.0.0.0:$PORT (default 8000)
```

## Container

```
docker build -t barnacle-counter .
docker run --rm -p 8000:8000 -e PORT=8000 barnacle-counter
curl http://127.0.0.1:8000/api/health
```

The image:
- runs as a non-root user;
- copies only the files listed above (see `.dockerignore`);
- sets `OPENCV_IO_MAX_IMAGE_PIXELS`;
- has a Docker `HEALTHCHECK` on `/api/health`.

## Platforms that use a Procfile

`Procfile` runs `gunicorn -c gunicorn.conf.py app:app`, which binds to `0.0.0.0:$PORT` as these platforms require. Use `/healthz` as the liveness check and `/api/health` as the readiness check; readiness returns 503 if the model failed to load.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 8000 | Port to bind |
| `WEB_CONCURRENCY` | 1 | Gunicorn worker processes. Each holds the model and can process one photo, so raise only with memory to match. |
| `GUNICORN_THREADS` | 2 | Threads per worker, so health checks and page loads are served during processing |
| `GUNICORN_TIMEOUT` | 300 | Seconds before gunicorn kills a busy worker. Processing takes about 30 s on one CPU. |
| `MAX_CONCURRENT_JOBS` | 1 | Photos processed at once per worker; others wait |
| `QUEUE_WAIT_S` | 20 | How long a request waits for a free slot before getting 503 with Retry-After |
| `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_S` | 12, 600 | Uploads per client address per window; exceeding it gives 429 with Retry-After |
| `TRUST_PROXY_HEADERS` | unset | Set to 1 only behind a proxy you control, to rate-limit by X-Forwarded-For |
| `BARNACLE_MODEL` | `models/barnacle_detector.pkl` | Model artifact path |
| `LOG_LEVEL` | INFO | Log level |

## Automatic image-type routing

The default `mode=auto` tries to locate the green quadrat first. When no frame can be found in a photo at most 1200 px wide and high, it treats the whole photo as an experimental native-scale crop. If no frame is found in larger photos, it requests manual corners instead. The UI provides `auto`, `quadrat`, and `native` choices. Automatic selection is heuristic and does not certify a true quadrat or accurate detections.

## Upload safety

- **Formats:** only PNG and JPEG are accepted. They are identified by file signature, not by name or content type. TIFF, BMP and WebP were accepted earlier and are now refused.
- **Size limits:** the image dimensions are read from the file header before any decoding, and the upload is rejected if it is over 60 MP or under 500 px on the short side. Uploads are capped at 40 MB. Auto mode first attempts quadrat-frame detection; when the frame cannot be detected and the image is no larger than 1200 px per side, the entire crop is used at native pixel scale. Larger images without a detected frame require manual corners. Mode can be overridden; native mode is experimental and does not guarantee accuracy.
- **Corrupt files:** a corrupted or truncated file, or one whose header disagrees with its pixels, gets a 400 with a plain explanation.
- **Memory only:** uploads are held in memory and decoded only by OpenCV. This code never writes them to disk and never executes them. Werkzeug may spool a large request body to a temporary file; the app closes the upload stream immediately after reading, which deletes it.
- **Logging:** logs contain a request id, sizes, dimensions, timings and status, never image data.
- **Errors:** pipeline failures return a generic message with the request id; the full traceback goes only to the server log.
- **Headers:** every response carries `X-Content-Type-Options`, `X-Frame-Options` and `Referrer-Policy`.

## Verified in the development environment

These were checked in the development sandbox, not in a container:
- The Flask development server started.
- In a real browser (Playwright), a development photo was uploaded, the model ran, and the add, remove, accept, reject, undo, export and re-import steps all worked.
- Automated tests cover the HTTP layer: validation, oversized-header ("decompression bomb") rejection, rate limiting, the busy-server response, generic 500s, the no-frame response with preview, and an end-to-end upload.

## Not verified (deployment risks)

1. **Docker build and run.** Docker is not available in the development environment, so the Dockerfile has never been built. In particular, whether `libglib2.0-0` is needed by the OpenCV headless wheel is unconfirmed (it is installed defensively).
2. **Gunicorn.** It is not installed in the development environment, so `gunicorn.conf.py` and the `Procfile` command have not been run. The pinned `gunicorn==23.0.0` is an assumption.
3. **Platform request timeouts.** Processing takes about 30 to 45 s on one CPU (measured: 26 s for img2 and 42 s for img1, with a browser running on the same CPU). Some hosting routers cut requests at 30 s, so slow or shared CPUs could hit that limit. If they do, move processing to a background job with polling.
4. **Memory.** Processing img1 (3264 x 2448) peaked at about 1.1 GB resident memory for the Flask process (VmHWM) in the development environment. Choose an instance with at least 2 GB, and keep `WEB_CONCURRENCY=1` unless memory allows more.
5. **Rate limiting is per process and in memory.** It resets on restart and is not shared between workers or instances. Use the platform's or a proxy's limits for real abuse protection.
6. **No authentication.** Anyone who can reach the URL can upload. Put it behind the host's access controls if the data or compute matters.
7. **Pickled model.** `models/barnacle_detector.pkl` must be loaded with scikit-learn 1.8.0, which is pinned. Only load model files you produced yourself; pickles can execute code.
