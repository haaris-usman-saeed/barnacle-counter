"""Deploy the existing Barnacle Counter Docker app to Modal.

One-time setup:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install modal
  modal setup
  modal deploy modal_app.py

The image is built from our existing Dockerfile; no training data is used.
Scale-to-zero with a single 2 GiB container to constrain demo compute usage.
"""

import subprocess
import modal

app = modal.App("barnacle-counter")
image = modal.Image.from_dockerfile("./Dockerfile", context_dir=".")

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=300,
)
@modal.web_server(8000, startup_timeout=180)
def serve():
    subprocess.Popen(
        ["gunicorn", "-c", "gunicorn.conf.py", "app:app"],
        cwd="/srv",
    )
