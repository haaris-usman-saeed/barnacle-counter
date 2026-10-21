# Barnacle counting app. Local test (requires Docker):
#   docker build -t barnacle-counter .
#   docker run --rm -p 8000:8000 barnacle-counter      # then open http://127.0.0.1:8000
FROM python:3.12-slim

# libglib2.0-0 is a defensive addition for OpenCV wheels; headless wheels may not need it.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY barnacles/ barnacles/
COPY models/ models/
COPY app.py gunicorn.conf.py ./

RUN useradd --create-home --uid 10001 app && chown -R app /srv
USER app

ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    OPENCV_IO_MAX_IMAGE_PIXELS=70000000 \
    MAX_CONCURRENT_JOBS=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=4)"
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
