"""Figures for Phase 3: match outcomes, PR curves, strata, failure galleries."""
import cv2
import numpy as np

# Okabe-Ito, BGR
MATCHED = (115, 158, 0)      # bluish green: detection matched to an annotation
UNMATCHED = (167, 121, 204)  # reddish purple: detection with no annotation nearby
MISSED = (0, 159, 230)       # orange: annotation with no detection
FRAME = (255, 255, 255)


def crop_box(frame, shape, pad=40):
    c = frame.corners
    x0, y0 = max(int(c[:, 0].min()) - pad, 0), max(int(c[:, 1].min()) - pad, 0)
    x1, y1 = min(int(c[:, 0].max()) + pad, shape[1]), min(int(c[:, 1].max()) + pad, shape[0])
    return x0, y0, x1, y1


def outcome_overlay(img, frame, det_xy, gt_xy, pairs, unmatched_det, missed_gt, title=None, scale=1.0):
    """Small markers only: matched = filled dot, unmatched detection = x,
    missed annotation = open ring. Nothing else is drawn on the photo."""
    x0, y0, x1, y1 = crop_box(frame, img.shape)
    out = img[y0:y1, x0:x1].copy()
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tf = lambda p: (int(round((p[0] - x0) * scale)), int(round((p[1] - y0) * scale)))
    q = np.array([tf(p) for p in frame.corners], np.int32)
    cv2.polylines(out, [q.reshape(-1, 1, 2)], True, FRAME, 1, cv2.LINE_AA)
    rr = max(2, int(round(4 * scale)))
    for j in missed_gt:
        cv2.circle(out, tf(gt_xy[j]), rr + 2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(out, tf(gt_xy[j]), rr + 2, MISSED, 2, cv2.LINE_AA)
    for i in unmatched_det:
        p = tf(det_xy[i])
        cv2.drawMarker(out, p, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 2 * rr + 3, 3, cv2.LINE_AA)
        cv2.drawMarker(out, p, UNMATCHED, cv2.MARKER_TILTED_CROSS, 2 * rr + 2, 1, cv2.LINE_AA)
    for i, _, _ in pairs:
        p = tf(det_xy[i])
        cv2.circle(out, p, rr, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p, rr - 1, MATCHED, -1, cv2.LINE_AA)
    if title:
        band = np.full((34, out.shape[1], 3), 255, np.uint8)
        cv2.putText(band, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1, cv2.LINE_AA)
        out = np.concatenate([band, out], 0)
    return out


def legend_strip(width):
    band = np.full((36, width, 3), 255, np.uint8)
    items = [("matched to an annotation", MATCHED, "dot"), ("detection with no annotation (not necessarily wrong)", UNMATCHED, "x"),
             ("annotation missed", MISSED, "ring")]
    x = 10
    for text, col, kind in items:
        c = (x + 8, 18)
        if kind == "dot":
            cv2.circle(band, c, 5, col, -1, cv2.LINE_AA)
        elif kind == "x":
            cv2.drawMarker(band, c, col, cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
        else:
            cv2.circle(band, c, 6, col, 2, cv2.LINE_AA)
        cv2.putText(band, text, (x + 22, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        x += 34 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    return band


def zoom_pair(img, frame, results, centers_uv, half_px, names):
    """For each zoom center, the same window for every method side by side."""
    rows = []
    for (u, v) in centers_uv:
        cx, cy = frame.to_image([[u, v]])[0]
        tiles = []
        for name, (det_xy, gt_xy, pairs, ud, mg) in zip(names, results):
            x0, y0 = int(cx - half_px), int(cy - half_px)
            sub = img[y0:y0 + 2 * half_px, x0:x0 + 2 * half_px].copy()
            def tf(p):
                return int(p[0] - x0), int(p[1] - y0)
            for j in mg:
                cv2.circle(sub, tf(gt_xy[j]), 7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(sub, tf(gt_xy[j]), 7, MISSED, 2, cv2.LINE_AA)
            for i in ud:
                cv2.drawMarker(sub, tf(det_xy[i]), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 13, 3, cv2.LINE_AA)
                cv2.drawMarker(sub, tf(det_xy[i]), UNMATCHED, cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
            for i, _, _ in pairs:
                cv2.circle(sub, tf(det_xy[i]), 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(sub, tf(det_xy[i]), 3, MATCHED, -1, cv2.LINE_AA)
            sub = cv2.resize(sub, (520, 520), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(sub, (0, 0), (len(name) * 11 + 12, 26), (255, 255, 255), -1)
            cv2.putText(sub, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
            tiles.append(sub)
        rows.append(np.concatenate(tiles, 1))
    grid = np.concatenate(rows, 0)
    return np.concatenate([grid, legend_strip(grid.shape[1])], 0)


def gallery(img, pts, labels, gt_xy=None, half=40, tile=150, cols=8):
    tiles = []
    for (x, y), lab in zip(pts, labels):
        x0, y0 = int(x) - half, int(y) - half
        t = img[max(y0, 0):y0 + 2 * half, max(x0, 0):x0 + 2 * half].copy()
        if t.shape[0] != 2 * half or t.shape[1] != 2 * half:
            t = cv2.copyMakeBorder(t, 0, 2 * half - t.shape[0], 0, 2 * half - t.shape[1], cv2.BORDER_CONSTANT)
        if gt_xy is not None:
            for g in gt_xy:
                if abs(g[0] - x) < half and abs(g[1] - y) < half:
                    cv2.circle(t, (int(g[0] - x0), int(g[1] - y0)), 2, (255, 0, 255), -1)
        cv2.circle(t, (half, half), 9, (0, 255, 255), 1, cv2.LINE_AA)
        t = cv2.resize(t, (tile, tile), interpolation=cv2.INTER_CUBIC)
        cv2.putText(t, lab, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
        cv2.putText(t, lab, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(t)
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 40))
    return np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)


def pr_figure(curves, points, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(curves), figsize=(6.2 * len(curves), 4.8), squeeze=False)
    cols = {"OpenCV baseline": "#E69F00", "ML classifier": "#0072B2"}
    for a, (title, cs) in zip(ax[0], curves.items()):
        for name, c in cs.items():
            r = [p["recall"] for p in c]
            pr = [p["precision_lb"] for p in c]
            a.plot(r, pr, "-", color=cols[name], label=name)
        for name, (rr, pp, kind) in points[title].items():
            a.plot(rr, pp, "o" if kind == "transferred" else "s", color=cols[name.split(" @")[0]],
                   mfc="white" if kind == "transferred" else cols[name.split(" @")[0]], ms=8,
                   label=f"{name}")
        a.set(xlim=(0, 1), ylim=(0, 1), xlabel="annotated recall",
              ylabel="annotation precision (lower bound)", title=title)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def strata_figure(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = ["sparse", "medium", "dense"]
    w = 0.2
    k = 0
    cols = ["#E69F00", "#0072B2", "#F0B860", "#56B4E9"]
    for (label, strata) in table.items():
        rec = [strata[n]["annotated_recall"] for n in names]
        un = [strata[n]["unmatched_detection_rate"] for n in names]
        ax[0].bar(np.arange(3) + (k - 1.5) * w, rec, w, label=label, color=cols[k % 4])
        ax[1].bar(np.arange(3) + (k - 1.5) * w, un, w, label=label, color=cols[k % 4])
        k += 1
    for a, t in zip(ax, ["Annotated recall by local annotation density",
                         "Share of detections with no annotation nearby"]):
        a.set_xticks(range(3))
        a.set_xticklabels(names)
        a.set_ylim(0, 1)
        a.set_title(t, fontsize=11)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
