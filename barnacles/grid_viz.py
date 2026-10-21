"""Figures for Phase 2: grid overlays, count summary, rectified view, frame check."""
import cv2
import numpy as np

from .grid import GridSpec

LINE_BGR = (255, 255, 255)
FRAME_BGR = (120, 230, 255)
DOT_BGR = (255, 0, 255)


def _blend_lines(canvas, segments, color, alpha, thickness=1):
    over = canvas.copy()
    for p, q in segments:
        cv2.line(over, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)),
                 color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(over, alpha, canvas, 1 - alpha, 0)


def grid_segments(frame, spec, shift=(0, 0)):
    s = np.array(shift, np.float64)
    segs = []
    for c in range(1, spec.cols):
        a, b = frame.to_image([[c / spec.cols, 0], [c / spec.cols, 1]])
        segs.append((a - s, b - s))
    for r in range(1, spec.rows):
        a, b = frame.to_image([[0, r / spec.rows], [1, r / spec.rows]])
        segs.append((a - s, b - s))
    return segs


def draw_grid_overlay(img, frame, spec, mapped, pad=60, band=44, dots=True, line_px=1, line_alpha=0.45):
    """Photo crop with subtle grid lines; row/column indices sit in a white
    margin band, never on the photo, so no label covers a barnacle."""
    H, W = img.shape[:2]
    c = frame.corners
    x0, y0 = max(int(c[:, 0].min()) - pad, 0), max(int(c[:, 1].min()) - pad, 0)
    x1, y1 = min(int(c[:, 0].max()) + pad, W), min(int(c[:, 1].max()) + pad, H)
    out = img[y0:y1, x0:x1].copy()
    out = _blend_lines(out, grid_segments(frame, spec, (x0, y0)), LINE_BGR, line_alpha, line_px)
    quad = frame.corners - [x0, y0]
    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    out = _blend_lines(out, edges, FRAME_BGR, 0.9, 2)
    if dots:
        for m in mapped:
            p = (int(round(m["x"] - x0)), int(round(m["y"] - y0)))
            cv2.circle(out, p, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, p, 3, DOT_BGR if m["inside"] else (0, 165, 255), -1, cv2.LINE_AA)
    if band == 0:
        return out
    canvas = np.full((out.shape[0] + 2 * band, out.shape[1] + 2 * band, 3), 255, np.uint8)
    canvas[band:-band, band:-band] = out
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.6
    for col in range(spec.cols):
        p = frame.to_image([[(col + 0.5) / spec.cols, 0]])[0] - [x0, y0] + [band, band]
        t = str(col)
        tw = cv2.getTextSize(t, font, fs, 1)[0][0]
        cv2.putText(canvas, t, (int(p[0] - tw / 2), band - 14), font, fs, (60, 60, 60), 1, cv2.LINE_AA)
    for row in range(spec.rows):
        p = frame.to_image([[0, (row + 0.5) / spec.rows]])[0] - [x0, y0] + [band, band]
        cv2.putText(canvas, str(row), (10, int(p[1] + 6)), font, fs, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def rectified_view(img, frame, spec, mapped, size=1200, margin=40):
    """Warp the frame to a square. If the mapping is right, the wire edges
    line up with the square and every dot sits on its operculum."""
    T = np.array([[size, 0, margin], [0, size, margin], [0, 0, 1]], np.float64)
    M = T @ frame.H
    out = cv2.warpPerspective(img, M, (size + 2 * margin, size + 2 * margin),
                              flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    segs = []
    for k in range(spec.cols + 1):
        x = margin + size * k / spec.cols
        segs.append(((x, margin), (x, margin + size)))
    for k in range(spec.rows + 1):
        y = margin + size * k / spec.rows
        segs.append(((margin, y), (margin + size, y)))
    out = _blend_lines(out, segs, LINE_BGR, 0.45, 1)
    for m in mapped:
        p = (int(round(margin + size * m["u"])), int(round(margin + size * m["v"])))
        cv2.circle(out, p, 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p, 2, DOT_BGR, -1, cv2.LINE_AA)
    return out


def frame_check(img, frame, debug, mapped, half=80, tile=300):
    """Corner and mid-side crops: yellow = detected opening edge, red = fitted
    frame quad. Plus each side's residual profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vis = img.copy()
    cv2.drawContours(vis, [debug["contour"].astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 255), 1)
    cv2.polylines(vis, [np.round(frame.corners).astype(np.int32).reshape(-1, 1, 2)], True,
                  (0, 0, 255), 1, cv2.LINE_AA)
    for m in mapped:
        cv2.circle(vis, (int(m["x"]), int(m["y"])), 3, DOT_BGR, -1)
    C = frame.corners
    spots = list(zip(("TL corner", "TR corner", "BR corner", "BL corner"), C))
    spots += list(zip(("top mid", "right mid", "bottom mid", "left mid"),
                      [(C[0] + C[1]) / 2, (C[1] + C[2]) / 2, (C[2] + C[3]) / 2, (C[3] + C[0]) / 2]))
    tiles = []
    for name, (x, y) in spots:
        x, y = int(x), int(y)
        t = vis[max(y - half, 0):y + half, max(x - half, 0):x + half]
        t = cv2.resize(t, (tile, tile), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(t, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(t)
    crops = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)

    fig, ax = plt.subplots(1, 4, figsize=(crops.shape[1] / 100, 2.4), dpi=100, sharey=True)
    ends = {"top": "TL to TR", "right": "TR to BR", "bottom": "BR to BL", "left": "BL to TL"}
    for a, (name, (pts, res, keep, pos)) in zip(ax, debug["side_points"].items()):
        o = np.argsort(pos)
        a.plot(pos[o], res[o], ".", ms=1.5, color="#0072B2")
        a.axhline(0, color="k", lw=0.6)
        a.axhline(2.5, color="#999", lw=0.5, ls=":")
        a.axhline(-2.5, color="#999", lw=0.5, ls=":")
        q = frame.quality["sides"][name]
        a.set_title(f"{name}: rms {q['rms_px']:.1f}px, bow {q['bow_px']:+.1f}px", fontsize=9)
        a.set_xlabel(f"fraction along side, {ends[name]}", fontsize=8)
        a.tick_params(labelsize=7)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    ax[0].set_ylabel("px into opening", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    prof = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1]
    plt.close(fig)
    prof = cv2.resize(prof, (crops.shape[1], int(prof.shape[0] * crops.shape[1] / prof.shape[1])))
    return np.concatenate([crops, prof], 0)


def summary_figure(img, frame, spec, cells, mapped, title, path):
    """Photo with grid | annotated count per cell | void fraction per cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Thicker lines here only because this panel is shown about 4x downscaled.
    ov = draw_grid_overlay(img, frame, spec, mapped, band=0, dots=False, line_px=4, line_alpha=0.6)
    counts = np.array([c["annotated_count"] for c in cells]).reshape(spec.rows, spec.cols)
    voids = np.array([c["void_fraction"] or 0 for c in cells]).reshape(spec.rows, spec.cols)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1, 1, 1]})
    ax[0].imshow(ov[..., ::-1])
    ax[0].set_axis_off()
    ax[0].set_title("Frame and grid on the photo", fontsize=11)
    im1 = ax[1].imshow(counts, cmap="Blues", vmin=0)
    ax[1].set_title("Annotated outlines per cell (not a barnacle count)", fontsize=11)
    im2 = ax[2].imshow(voids, cmap="Oranges", vmin=0, vmax=1)
    ax[2].set_title("Share of cell with no annotation nearby (void)", fontsize=11)
    fs = 9 if spec.n_cells <= 144 else 6
    for a, M, fmt in ((ax[1], counts, "{:d}"), (ax[2], voids, "{:.2f}")):
        a.set_xticks(range(spec.cols))
        a.set_yticks(range(spec.rows))
        a.tick_params(labelsize=7, length=0)
        a.set_xlabel("column", fontsize=9)
        a.set_ylabel("row", fontsize=9)
        if spec.n_cells <= 400:
            hi = M.max() if M.max() > 0 else 1
            for r in range(spec.rows):
                for c in range(spec.cols):
                    val = M[r, c]
                    a.text(c, r, fmt.format(int(val)) if fmt == "{:d}" else fmt.format(val),
                           ha="center", va="center", fontsize=fs,
                           color="white" if val > 0.6 * hi else "#222")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
