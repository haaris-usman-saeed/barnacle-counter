# Changelog

## Final submission pass

- **Training-photo recognition.** The exact pixel hash was replaced by three-level recognition (`barnacles/fingerprint.py`):
  - **exact:** SHA-256 match; definitive;
  - **likely:** perceptual hashes plus thumbnail correlation; catches resized and recompressed copies;
  - **possible:** looser similarity.

  Fingerprints ship in `models/training_fingerprints.json`, and the warning explains that training-photo predictions cannot show generalization. Tested on resized, recompressed, darkened and cropped versions of both photos, and the two photos never match each other.
- **Portable tests and scripts.** No developer-specific paths remain. Data comes from `data/` or `$BARNACLE_DATA`, and the experiment cache from `experiments/cache` or `$BARNACLE_EXPERIMENT_CACHE`. Data-dependent tests skip cleanly without the data. New synthetic fixtures cover fingerprints and a full end-to-end inference run, both without the dataset.
- **Dashboard.** The counts are now separate: selected detections, removed, accepted suggestions, added, reviewed, and still to review (detections and suggestions shown apart). It states the identity "corrected = selected - removed + accepted suggestions + added" and checks it on every update. The same breakdown is in exported sessions and in the Python replay, which asserts the identity. Count semantics are unchanged.
- **Upload hardening.**
  - Only PNG and JPEG are accepted (TIFF, BMP and WebP are now refused).
  - Image dimensions are checked from the header before decoding, and OpenCV's pixel cap is set.
  - Header/pixel mismatches are detected.
  - A semaphore limits concurrent jobs, returning 503 with Retry-After when busy, and per-client rate limiting returns 429.
  - Errors are generic, with a request id; logs contain no image data; security headers are added.
  - The model loads once per process, and the readiness endpoint reports 503 if it failed.
  - Evaluation requests have size and point limits.
- **Deployment configuration.**
  - New `gunicorn.conf.py`: binds `$PORT`, one worker, two threads, 300 s timeout, preloading, worker recycling.
  - Corrected `Procfile`.
  - `Dockerfile` now runs as non-root, has a health check, copies only runtime files, and has a `.dockerignore`.
  - Exact pins in `requirements.txt` (runtime) and a separate `requirements-dev.txt`.
  - `DEPLOYMENT.md` lists the settings and what is unverified.
- **README.** Restructured into the 11-step development story, with the C0 versus C7 table, the model-selection caveat, the review process and the deployment pointers.
- **Cleanup.** Removed bytecode caches, the old Phase 3 full-resolution outcome images (superseded by `experiments/figures`), the experiment cache, temporary files and unused screenshots.
