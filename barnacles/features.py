"""Per-candidate features for the classifier.

All features are computed in the rectified frame on the contrast-normalized
grey image, and scaled by each candidate's own radius, so they do not
depend on camera resolution or on the absolute brightness of the photo.

Groups (dimension):
  radial   mean and std of intensity on 8 rings from 0.15 r to 2.4 r (16),
           plus angular contrast at the opening margin (1). Rotation-invariant.
  hog      HOG of a 32 x 32 patch covering +/- 2.4 r (324).
  lbp      uniform LBP histogram (P=8, R=1) in a window of side ~2.4 r (10).
  scores   ring, LoG and combined detector responses, and their within-image
           percentiles; radius relative to the image's median candidate (7).
  extra    (optional) wider HOG patch at +/- 4 r (324) and the radial
           gradient profile, mean and mean-absolute per ring (16).
  color    (ablation only) Lab a and b radial means, z-scored per image (16).
Colour is excluded from the main model on purpose: the two photos differ
completely in colour (white shells on grey rock versus brown shells on red
rock), so colour would mostly encode which photo a patch came from.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hog, local_binary_pattern

RADII = np.array([0.15, 0.45, 0.75, 1.0, 1.25, 1.6, 2.0, 2.4])
N_ANG = 16
PATCH = 32
HALF = 2.4


def _polar(img, x, y, r):
    ang = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
    rr = r[:, None, None] * RADII[None, :, None]
    px = x[:, None, None] + rr * np.cos(ang)[None, None, :]
    py = y[:, None, None] + rr * np.sin(ang)[None, None, :]
    v = ndi.map_coordinates(img, [py.ravel(), px.ravel()], order=1, mode="reflect")
    return v.reshape(len(x), len(RADII), N_ANG)


def _patches(img, x, y, r, batch=2000):
    g = (np.arange(PATCH) + 0.5) / PATCH * 2 - 1      # -1..1
    out = np.empty((len(x), PATCH, PATCH), np.float32)
    for s in range(0, len(x), batch):
        e = min(s + batch, len(x))
        h = (HALF * r[s:e])[:, None, None]
        py = y[s:e, None, None] + h * g[None, :, None]
        px = x[s:e, None, None] + h * g[None, None, :]
        py, px = np.broadcast_arrays(py, px)
        out[s:e] = ndi.map_coordinates(img, [py.ravel(), px.ravel()], order=1,
                                       mode="reflect").reshape(e - s, PATCH, PATCH)
    return out


def _pct(a):
    return (np.argsort(np.argsort(a)) + 0.5) / len(a)


def candidate_features(rect, cands, maps, with_color=False, extra=False):
    z = maps["z"].astype(np.float32)
    x, y, r = cands["x"], cands["y"], cands["radius"]
    feats, names = [], []

    pol = _polar(z, x, y, r)
    feats += [pol.mean(2), pol.std(2), (pol[:, 3].max(1) - pol[:, 3].min(1))[:, None]]
    names += [f"radial_mean_{v}" for v in RADII] + [f"radial_std_{v}" for v in RADII] + ["margin_contrast"]

    P = _patches(z, x, y, r)
    P = (P - P.mean((1, 2), keepdims=True)) / (P.std((1, 2), keepdims=True) + 1e-6)
    H = np.stack([hog(p, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      feature_vector=True) for p in P])
    feats.append(H)
    names += [f"hog_{i}" for i in range(H.shape[1])]

    zu8 = np.clip((z - z.min()) / (np.ptp(z) + 1e-6) * 255, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(zu8, 8, 1, "uniform").astype(int)
    lb = np.zeros((len(x), 10), np.float32)
    for lev in np.unique(cands["level"]):
        sel = cands["level"] == lev
        k = max(3, int(round(2.4 * maps["radii"][lev])) | 1)
        for code in range(10):
            m = cv2.boxFilter((lbp == code).astype(np.float32), -1, (k, k))
            lb[sel, code] = m[y[sel].astype(int), x[sel].astype(int)]
    feats.append(lb)
    names += [f"lbp_{i}" for i in range(10)]

    sc = np.c_[cands["ring_score"], cands["log_score"], cands["score"],
               _pct(cands["ring_score"]), _pct(cands["log_score"]), _pct(cands["score"]),
               r / np.median(r)]
    feats.append(sc)
    names += ["ring", "log", "combined", "ring_pct", "log_pct", "combined_pct", "rel_radius"]

    if extra:
        # multi-scale context: a second, wider HOG patch (+/- 4 r)
        P2 = _patches(z, x, y, r * (4.0 / HALF))
        P2 = (P2 - P2.mean((1, 2), keepdims=True)) / (P2.std((1, 2), keepdims=True) + 1e-6)
        H2 = np.stack([hog(p, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                           feature_vector=True) for p in P2])
        feats.append(H2)
        names += [f"hog_wide_{i}" for i in range(H2.shape[1])]
        # radial edge pattern: gradient component along the radius, per ring
        gy, gx = np.gradient(z)
        ang = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
        pgx, pgy = _polar(gx, x, y, r), _polar(gy, x, y, r)
        radial = pgx * np.cos(ang)[None, None, :] + pgy * np.sin(ang)[None, None, :]
        feats += [radial.mean(2), np.abs(radial).mean(2)]
        names += [f"radial_grad_{v}" for v in RADII] + [f"radial_absgrad_{v}" for v in RADII]
    if with_color:
        lab = cv2.cvtColor(rect.image, cv2.COLOR_BGR2LAB).astype(np.float32)
        for ch, nm in ((1, "a"), (2, "b")):
            c = (lab[..., ch] - lab[..., ch].mean()) / (lab[..., ch].std() + 1e-6)
            feats.append(_polar(c, x, y, r).mean(2))
            names += [f"color_{nm}_{v}" for v in RADII]
    return np.hstack(feats).astype(np.float32), names
