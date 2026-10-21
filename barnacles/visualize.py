"""Figures for checking the extraction by eye."""
import cv2
import numpy as np

# Okabe-Ito palette (colour-blind safe), BGR.
KIND_BGR = {"closed": (233, 180, 86), "merged": (0, 159, 230), "solid": (0, 94, 213),
            "open": (167, 121, 204), "fragment": (66, 228, 240)}
MARKER_BGR = (255, 0, 255)


def draw_centers(img, records, radius=4):
    """Small magenta dots with a 1 px dark rim; readable on white shells and dark rock."""
    out = img.copy()
    for r in records:
        c = (int(round(r["x"])), int(round(r["y"])))
        if r["counted"]:
            cv2.circle(out, c, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, c, radius, MARKER_BGR, -1, cv2.LINE_AA)
        else:
            cv2.circle(out, c, radius + 2, MARKER_BGR, 1, cv2.LINE_AA)
    return out


def _legend(canvas, items, origin=(12, 12), scale=0.7):
    x, y = origin
    h = 26 * len(items) + 12
    w = 12 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] for t, _ in items) + 40
    sub = canvas[y:y + h, x:x + w]
    sub[:] = (sub * 0.3).astype(np.uint8)
    for k, (text, col) in enumerate(items):
        yy = y + 22 + 26 * k
        cv2.rectangle(canvas, (x + 10, yy - 12), (x + 30, yy + 2), col, -1)
        cv2.putText(canvas, text, (x + 38, yy), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)


def draw_review(img, records, roi, voids, void_mask, region=None, pad=40):
    """ROI crop with extracted polygons by kind, centers, flagged items, and voids."""
    H, W = img.shape[:2]
    x0, y0 = max(roi[0] - pad, 0), max(roi[1] - pad, 0)
    x1, y1 = min(roi[2] + pad, W), min(roi[3] + pad, H)
    out = img[y0:y1, x0:x1].copy()
    vm, (vx, vy) = void_mask
    ov = out.copy()
    sub = ov[vy - y0:vy - y0 + vm.shape[0], vx - x0:vx - x0 + vm.shape[1]]
    sub[vm] = (0, 215, 255)
    out = cv2.addWeighted(ov, 0.35, out, 0.65, 0)
    for r in records:
        poly = (np.array(r["polygon"]) - [x0, y0]).astype(np.int32)
        cv2.polylines(out, [poly], True, KIND_BGR[r["kind"]], 2, cv2.LINE_AA)
    for r in records:
        c = (int(round(r["x"] - x0)), int(round(r["y"] - y0)))
        cv2.circle(out, c, 3, (0, 0, 0), -1)
        cv2.circle(out, c, 2, (255, 255, 255), -1)
        review_flags = [f for f in r["flags"] if f != "touches_frame"]
        if review_flags:
            bx, by, bw, bh = r["bbox_xywh"]
            cv2.rectangle(out, (bx - x0 - 4, by - y0 - 4), (bx - x0 + bw + 4, by - y0 + bh + 4),
                          (0, 0, 255), 2)
    for v in voids:
        c = (int(v["x"] - x0), int(v["y"] - y0))
        cv2.putText(out, f"V{v['void_id']}", c, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"V{v['void_id']}", c, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.rectangle(out, (roi[0] - x0, roi[1] - y0), (roi[2] - x0, roi[3] - y0), (255, 255, 255), 1)
    if region is not None:  # outline of the area where voids are evaluated
        cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c + [vx - x0, vy - y0] for c in cnts]
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)
    counts = {}
    for r in records:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    items = [(f"{k} ({counts[k]})", KIND_BGR[k]) for k in KIND_BGR if k in counts]
    items += [("flagged for review", (0, 0, 255)), (f"void: no annotation ({len(voids)})", (0, 215, 255))]
    _legend(out, items)
    return out


def gallery(img, stroke, records, title, cols=6, tile=220, ctx=1.6, max_n=24):
    """Tiles of photo | photo + raw stroke + extracted polygon, for human review."""
    tiles = []
    for r in records[:max_n]:
        bx, by, bw, bh = r["bbox_xywh"]
        half = int(max(bw, bh) * ctx / 2) + 10
        cx, cy = int(r["x"]), int(r["y"])
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x1, y1 = min(cx + half, img.shape[1]), min(cy + half, img.shape[0])
        ph = img[y0:y1, x0:x1]
        an = ph.copy()
        an[stroke[y0:y1, x0:x1]] = (200, 120, 60)
        poly = (np.array(r["polygon"]) - [x0, y0]).astype(np.int32)
        cv2.polylines(an, [poly], True, (0, 255, 255), 1)
        cv2.circle(an, (cx - x0, cy - y0), 2, (255, 0, 255), -1)
        pair = np.concatenate([ph, an], 1)
        pair = cv2.resize(pair, (2 * tile, tile), interpolation=cv2.INTER_NEAREST)
        lbl = f"#{r['id']} {r['kind']} a={r['area_px']}"
        cv2.putText(pair, lbl, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(pair, lbl, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(pair)
    if not tiles:
        return None
    blank = np.full_like(tiles[0], 40)
    while len(tiles) % cols:
        tiles.append(blank)
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1)
                           for i in range(0, len(tiles), cols)], 0)
    head = np.full((36, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return np.concatenate([head, grid], 0)


def void_gallery(img, records, voids, spacing, tile=300, cols=4, max_n=16):
    """Each void with annotation centers marked, so a human can judge it."""
    tiles = []
    for v in voids[:max_n]:
        half = int(max(v["bbox_xywh"][2], v["bbox_xywh"][3]) / 2 + spacing)
        half = max(half, int(2 * spacing))
        cx, cy = int(v["x"]), int(v["y"])
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x1, y1 = min(cx + half, img.shape[1]), min(cy + half, img.shape[0])
        t = img[y0:y1, x0:x1].copy()
        for r in records:
            if x0 <= r["x"] < x1 and y0 <= r["y"] < y1:
                c = (int(r["x"] - x0), int(r["y"] - y0))
                cv2.circle(t, c, 5, (0, 0, 0), -1)
                cv2.circle(t, c, 4, (255, 0, 255), -1)
        cv2.circle(t, (cx - x0, cy - y0), int(v["max_clearance_px"]), (0, 215, 255), 2)
        t = cv2.resize(t, (tile, tile), interpolation=cv2.INTER_AREA)
        lbl = f"V{v['void_id']} clr={v['clearance_over_spacing']:.1f}s" + (" frame" if v.get("frame_adjacent") else "")
        cv2.putText(t, lbl, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(t, lbl, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(t)
    if not tiles:
        return None
    blank = np.full_like(tiles[0], 40)
    while len(tiles) % cols:
        tiles.append(blank)
    return np.concatenate([np.concatenate(tiles[i:i + cols], 1)
                           for i in range(0, len(tiles), cols)], 0)


def stats_figure(per_image: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = ["#0072B2", "#E69F00"]
    for k, (name, d) in enumerate(per_image.items()):
        rec = [r for r in d["annotations"] if r["counted"]]
        diam = [r["equiv_diameter_px"] for r in rec]
        sw = [r["stroke_width_px"] for r in rec if r["stroke_width_px"] == r["stroke_width_px"]]
        nn = d["nn"]
        kw = dict(alpha=0.6, color=cols[k], label=f"{name} (n={len(rec)})", density=True)
        ax[0].hist(diam, bins=40, **kw)
        ax[1].hist(sw, bins=np.arange(0, 14, 0.5), **kw)
        ax[2].hist(nn, bins=40, **kw)
    ax[0].set(title="Annotated outline size", xlabel="equivalent diameter (px)", ylabel="density")
    ax[1].set(title="Pen stroke width per annotation", xlabel="stroke width (px)")
    ax[2].set(title="Center spacing", xlabel="nearest-neighbour distance (px)")
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def spot_check(img, records, spacing, n=2, seed=1, out_px=600):
    """Seeded random patches (6 x 6 spacings) with annotation centers, for a
    by-eye check of completeness and center placement. Not a measurement."""
    rng = np.random.default_rng(seed)
    recs = [r for r in records if r["counted"]]
    half = int(3 * spacing)
    tiles, meta = [], []
    for _ in range(n):
        r = recs[rng.integers(len(recs))]
        x0, y0 = int(r["x"]) - half, int(r["y"]) - half
        t = img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
        k = 0
        for q in recs:
            if x0 <= q["x"] < x0 + 2 * half and y0 <= q["y"] < y0 + 2 * half:
                c = (int(q["x"] - x0), int(q["y"] - y0))
                cv2.circle(t, c, 3, (0, 0, 0), -1)
                cv2.circle(t, c, 2, MARKER_BGR, -1)
                k += 1
        t = cv2.resize(t, (out_px, out_px), interpolation=cv2.INTER_CUBIC)
        lbl = f"patch x={x0} y={y0} {2 * half}px, annotated={k}"
        cv2.putText(t, lbl, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(t, lbl, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(t)
        meta.append({"x0": x0, "y0": y0, "size_px": 2 * half, "annotated": k})
    return np.concatenate(tiles, 1), meta
