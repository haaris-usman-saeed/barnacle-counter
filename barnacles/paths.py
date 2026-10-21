"""Where the development data lives, without developer-specific paths.

The challenge photos and masks (img1.png, img2.png, mask1.png, mask2.png,
optionally masked_img*.png) are NOT bundled with the code. Put them in
`<repo>/data/`, or point BARNACLE_DATA at the folder that holds them.
Everything that needs them (research scripts, some tests) uses data_dir();
the web application never does.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return Path(os.environ.get("BARNACLE_DATA", REPO / "data")).expanduser()


def experiment_cache() -> Path:
    return Path(os.environ.get("BARNACLE_EXPERIMENT_CACHE", REPO / "experiments" / "cache")).expanduser()


def has_development_data() -> bool:
    d = data_dir()
    return all((d / f).exists() for f in ("img1.png", "img2.png"))
