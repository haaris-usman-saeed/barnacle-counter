"""Write models/training_fingerprints.json from the development photos
(also done by train_model.py). Usage: python scripts/fingerprint_training_photos.py"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from barnacles.fingerprint import write_refs, REF_PATH  # noqa: E402
from barnacles.paths import data_dir  # noqa: E402

if __name__ == "__main__":
    write_refs({n: cv2.imread(str(data_dir() / n), cv2.IMREAD_COLOR) for n in ("img1.png", "img2.png")})
    print("wrote", REF_PATH)
