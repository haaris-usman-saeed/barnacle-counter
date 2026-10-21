"""Gunicorn settings for the barnacle app (gunicorn -c gunicorn.conf.py app:app).

Inference is CPU-bound; processing a 3264 x 2448 photo peaked at about 1.1 GB
resident memory for the whole server process in the development environment
(score maps over 9 scales), so one worker per container is the default;
raise WEB_CONCURRENCY only with matching memory. Two threads let health checks
and the page load while a photo is processed; app.py's semaphore still allows
only MAX_CONCURRENT_JOBS photos at once.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"   # hosting platforms inject PORT
workers = int(os.environ.get("WEB_CONCURRENCY", 1))
threads = int(os.environ.get("GUNICORN_THREADS", 2))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 300))  # processing takes ~30 s on one CPU; allow slow hosts
graceful_timeout = 30
keepalive = 5
preload_app = True              # load the model once, before forking
max_requests = 200              # recycle workers to cap any slow memory growth
max_requests_jitter = 20
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
accesslog = "-"
errorlog = "-"
