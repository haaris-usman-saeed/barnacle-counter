"""Self-contained HTML viewer: photo, frame, adjustable grid, selectable cells.

Review tool for Phase 2, not the application UI. The page recomputes cell
membership in the browser with the same rule as grid.assign, and checks its
result against the Python assignment at the default grid size.
"""
import base64
import json

import cv2
import numpy as np

from .grid import assign


def image_payload(name, img, frame, mapped, spec, max_px=1400, pad=60):
    H, W = img.shape[:2]
    c = frame.corners
    x0, y0 = max(int(c[:, 0].min()) - pad, 0), max(int(c[:, 1].min()) - pad, 0)
    x1, y1 = min(int(c[:, 0].max()) + pad, W), min(int(c[:, 1].max()) + pad, H)
    crop = img[y0:y1, x0:x1]
    s = min(1.0, max_px / max(crop.shape[:2]))
    disp = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 86])
    uv = np.array([[m["u"], m["v"]] for m in mapped])
    row, col, inside = assign(uv, spec)
    py_cell = np.where(inside, row * spec.cols + col, -1).tolist()
    return {
        "name": name,
        "img": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
        "crop": {"x0": x0, "y0": y0, "scale": s, "w": disp.shape[1], "h": disp.shape[0]},
        "H": frame.H.tolist(), "Hinv": frame.H_inv.tolist(),
        "corners": frame.corners.tolist(),
        "frameSource": frame.source,
        "centers": [[m["id"], m["x"], m["y"], m["u"], m["v"]] for m in mapped],
        "py": {"rows": spec.rows, "cols": spec.cols, "cell": py_cell},
    }


def write_viewer(payloads, path):
    html = TEMPLATE.replace("__DATA__", json.dumps(payloads))
    with open(path, "w") as f:
        f.write(html)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quadrat grid review</title>
<style>
:root {
  --bg: #e8ecea; --surface: #f6f8f7; --ink: #1d2a2c; --muted: #5b6b6d;
  --rule: #c5cfcc; --wire: #1e7470; --on-wire: #ffffff; --select: #e8b21c; --focus: #1e7470;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #172123; --surface: #1f2b2d; --ink: #e3ebe9; --muted: #97a9a7;
    --rule: #34474a; --wire: #5fb8b1; --on-wire: #102022; --select: #f2c230; --focus: #5fb8b1;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.45 "Avenir Next", "Segoe UI", system-ui, sans-serif; }
header { padding: 18px 24px 10px; display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 20px; }
h1 { margin: 0; font: 600 24px/1.2 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; }
header p { margin: 0; color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; padding: 0 24px 14px; }
.tabs { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.tabs button { border: 0; background: transparent; color: var(--ink); padding: 6px 14px; font: inherit; cursor: pointer; }
.tabs button[aria-pressed="true"] { background: var(--wire); color: var(--on-wire); }
label { display: inline-flex; gap: 6px; align-items: center; }
input[type=number] { width: 4.2em; padding: 4px 6px; font: inherit; color: var(--ink);
  background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; }
button.plain { font: inherit; padding: 5px 12px; border: 1px solid var(--rule); background: var(--surface);
  color: var(--ink); border-radius: 4px; cursor: pointer; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 0 22px; padding: 0 24px 24px; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
.stage { position: relative; }
canvas { max-width: 100%; max-height: calc(100vh - 140px); width: auto; height: auto; display: block;
  border-radius: 4px; cursor: crosshair; }
canvas:focus:not(:focus-visible) { outline: none; }
aside { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 16px 18px;
  align-self: start; position: sticky; top: 12px; max-height: calc(100vh - 24px); overflow: auto; }
aside h2 { font: 600 17px/1.3 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; margin: 0 0 8px; }
aside section + section { border-top: 1px solid var(--rule); margin-top: 14px; padding-top: 14px; }
dl { display: grid; grid-template-columns: auto 1fr; gap: 3px 14px; margin: 0; font-variant-numeric: tabular-nums; }
dt { color: var(--muted); } dd { margin: 0; text-align: right; }
.ok { color: #2f7d4a; } .bad { color: #b3261e; }
@media (prefers-color-scheme: dark) { .ok { color: #7fd09a; } .bad { color: #ff8a80; } }
.ids { font-variant-numeric: tabular-nums; font-size: 13px; color: var(--muted); max-height: 220px; overflow: auto;
  margin-top: 8px; line-height: 1.6; word-break: break-word; }
.note { color: var(--muted); font-size: 13px; }
.hint { color: var(--muted); font-size: 13px; margin: 6px 0 0; }
</style>
</head>
<body>
<header>
  <h1>Quadrat grid review</h1>
  <p>Annotated outlines mapped into the inner frame. Click a cell to inspect it; shift-click to add cells.</p>
</header>
<div class="controls">
  <div class="tabs" role="group" aria-label="Photo" id="tabs"></div>
  <label>Rows <input id="rows" type="number" min="1" max="60" step="1"></label>
  <label>Columns <input id="cols" type="number" min="1" max="60" step="1"></label>
  <label><input id="dots" type="checkbox" checked> Show centers</label>
  <label><input id="tint" type="checkbox"> Shade cells by count</label>
  <button class="plain" id="clear">Clear selection</button>
</div>
<main>
  <div class="stage"><canvas id="cv" tabindex="0" aria-label="Photo with frame grid; use arrow keys to move the selected cell"></canvas></div>
  <aside>
    <section>
      <h2>Frame totals</h2>
      <dl id="totals"></dl>
      <p class="hint" id="check"></p>
    </section>
    <section>
      <h2>Selection</h2>
      <div id="sel"><p class="note">No cell selected. Click a cell in the photo.</p></div>
    </section>
    <section>
      <p class="note">Counts are annotated outlines, not barnacles. The source masks miss some live barnacles
      and leave empty shells unmarked, so every number here is a lower bound for that cell.</p>
    </section>
  </aside>
</main>
<script>
const DATA = __DATA__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const el = id => document.getElementById(id);
let cur = 0, rows, cols, selected = new Set(), hover = -1, imgs = [];

function ap(M, x, y) {
  const w = M[2][0]*x + M[2][1]*y + M[2][2];
  return [(M[0][0]*x + M[0][1]*y + M[0][2]) / w, (M[1][0]*x + M[1][1]*y + M[1][2]) / w];
}
// Same rule as barnacles.grid.assign
function assign(u, v) {
  if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1)) return -1;
  const c = Math.min(Math.floor(u * cols), cols - 1), r = Math.min(Math.floor(v * rows), rows - 1);
  return r * cols + c;
}
function cellName(k) { const r = Math.floor(k / cols), c = k % cols;
  return 'r' + String(r).padStart(2, '0') + 'c' + String(c).padStart(2, '0'); }

let members = [], cellOf = [];
function recompute() {
  const d = DATA[cur];
  members = Array.from({length: rows * cols}, () => []);
  cellOf = d.centers.map(([id, x, y, u, v]) => assign(u, v));
  cellOf.forEach((k, i) => { if (k >= 0) members[k].push(d.centers[i][0]); });
}
function toDisp(x, y) { const c = DATA[cur].crop; return [(x - c.x0) * c.scale, (y - c.y0) * c.scale]; }
function uvDisp(u, v) { const p = ap(DATA[cur].Hinv, u, v); return toDisp(p[0], p[1]); }
function cellPath(k) {
  const r = Math.floor(k / cols), c = k % cols;
  const pts = [[c/cols, r/rows], [(c+1)/cols, r/rows], [(c+1)/cols, (r+1)/rows], [c/cols, (r+1)/rows]].map(p => uvDisp(p[0], p[1]));
  ctx.beginPath(); ctx.moveTo(...pts[0]); for (const p of pts.slice(1)) ctx.lineTo(...p); ctx.closePath();
}
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function draw() {
  const d = DATA[cur];
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(imgs[cur], 0, 0);
  if (el('tint').checked) {
    const mx = Math.max(1, ...members.map(m => m.length));
    for (let k = 0; k < rows * cols; k++) {
      cellPath(k); ctx.fillStyle = `rgba(30,116,112,${0.55 * members[k].length / mx})`; ctx.fill();
    }
  }
  ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,0.42)';
  ctx.beginPath();
  for (let c = 1; c < cols; c++) { const a = uvDisp(c/cols, 0), b = uvDisp(c/cols, 1); ctx.moveTo(...a); ctx.lineTo(...b); }
  for (let r = 1; r < rows; r++) { const a = uvDisp(0, r/rows), b = uvDisp(1, r/rows); ctx.moveTo(...a); ctx.lineTo(...b); }
  ctx.stroke();
  const q = d.corners.map(p => toDisp(p[0], p[1]));
  ctx.beginPath(); ctx.moveTo(...q[0]); q.slice(1).forEach(p => ctx.lineTo(...p)); ctx.closePath();
  ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,230,120,0.9)'; ctx.stroke();
  const sel = css('--select');
  for (const k of selected) {
    cellPath(k); ctx.fillStyle = 'rgba(232,178,28,0.16)'; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = sel; ctx.stroke();
  }
  if (hover >= 0 && !selected.has(hover)) { cellPath(hover); ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.stroke(); }
  if (el('dots').checked) {
    d.centers.forEach(([id, x, y], i) => {
      const p = toDisp(x, y), inSel = selected.has(cellOf[i]);
      ctx.beginPath(); ctx.arc(p[0], p[1], inSel ? 3.2 : 2.2, 0, 2 * Math.PI);
      ctx.fillStyle = cellOf[i] < 0 ? '#ff9f1c' : (inSel ? '#ffffff' : '#ff2bd6'); ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(0,0,0,0.7)'; ctx.stroke();
    });
  }
}

function panel() {
  const d = DATA[cur];
  const inside = cellOf.filter(k => k >= 0).length, out = cellOf.length - inside;
  const sum = members.reduce((a, m) => a + m.length, 0);
  el('totals').innerHTML =
    `<dt>Extracted centers</dt><dd>${cellOf.length}</dd>` +
    `<dt>Inside the frame</dt><dd>${inside}</dd>` +
    `<dt>Outside the frame</dt><dd>${out}</dd>` +
    `<dt>Sum over ${rows * cols} cells</dt><dd class="${sum === inside ? 'ok' : 'bad'}">${sum}</dd>` +
    `<dt>Frame corners</dt><dd>${d.frameSource === 'manual' ? 'manual' : 'detected'}</dd>`;
  let msg = '';
  if (rows === d.py.rows && cols === d.py.cols) {
    const same = cellOf.every((k, i) => k === d.py.cell[i]);
    msg = same ? `Cell assignment matches the Python output for all ${cellOf.length} centers.`
               : 'Cell assignment differs from the Python output. Re-export the viewer.';
  } else {
    msg = `Python reference is ${d.py.rows} x ${d.py.cols}; this grid is recomputed in the browser with the same rule.`;
  }
  el('check').textContent = msg;
  if (!selected.size) { el('sel').innerHTML = '<p class="note">No cell selected. Click a cell in the photo.</p>'; return; }
  const ks = [...selected].sort((a, b) => a - b);
  if (ks.length === 1) {
    const k = ks[0], r = Math.floor(k / cols), c = k % cols, ids = members[k];
    el('sel').innerHTML =
      `<dl><dt>Cell</dt><dd>${cellName(k)}</dd><dt>Row, column</dt><dd>${r}, ${c}</dd>` +
      `<dt>u range</dt><dd>${(c/cols).toFixed(3)} to ${((c+1)/cols).toFixed(3)}</dd>` +
      `<dt>v range</dt><dd>${(r/rows).toFixed(3)} to ${((r+1)/rows).toFixed(3)}</dd>` +
      `<dt>Annotated outlines</dt><dd>${ids.length}</dd></dl>` +
      `<div class="ids">${ids.length ? ids.join(', ') : 'No annotations in this cell.'}</div>`;
  } else {
    const tot = ks.reduce((a, k) => a + members[k].length, 0);
    el('sel').innerHTML =
      `<dl><dt>Cells selected</dt><dd>${ks.length}</dd><dt>Annotated outlines</dt><dd>${tot}</dd></dl>` +
      `<div class="ids">${ks.map(k => `${cellName(k)}: ${members[k].length}`).join('<br>')}</div>`;
  }
}

function cellAt(evt) {
  const rect = cv.getBoundingClientRect(), c = DATA[cur].crop;
  const dx = (evt.clientX - rect.left) * cv.width / rect.width, dy = (evt.clientY - rect.top) * cv.height / rect.height;
  const [u, v] = ap(DATA[cur].H, dx / c.scale + c.x0, dy / c.scale + c.y0);
  return assign(u, v);
}
cv.addEventListener('mousemove', e => { const k = cellAt(e); if (k !== hover) { hover = k; draw(); } });
cv.addEventListener('mouseleave', () => { hover = -1; draw(); });
cv.addEventListener('click', e => {
  const k = cellAt(e); if (k < 0) return;
  if (e.shiftKey || e.metaKey || e.ctrlKey) { selected.has(k) ? selected.delete(k) : selected.add(k); }
  else { selected = new Set(selected.size === 1 && selected.has(k) ? [] : [k]); }
  draw(); panel();
});
cv.addEventListener('keydown', e => {
  const moves = {ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0]};
  if (e.key === 'Escape') { selected.clear(); draw(); panel(); return; }
  if (!moves[e.key]) return;
  e.preventDefault();
  let k = selected.size ? [...selected][selected.size - 1] : 0;
  let r = Math.floor(k / cols) + moves[e.key][0], c = k % cols + moves[e.key][1];
  r = Math.max(0, Math.min(rows - 1, r)); c = Math.max(0, Math.min(cols - 1, c));
  selected = new Set([r * cols + c]); draw(); panel();
});
function setGrid() {
  const r = parseInt(el('rows').value, 10), c = parseInt(el('cols').value, 10);
  if (!(r >= 1 && r <= 60 && c >= 1 && c <= 60)) return;
  rows = r; cols = c; selected.clear(); hover = -1; recompute(); draw(); panel();
}
['rows', 'cols'].forEach(id => el(id).addEventListener('change', setGrid));
['dots', 'tint'].forEach(id => el(id).addEventListener('change', draw));
el('clear').addEventListener('click', () => { selected.clear(); draw(); panel(); });

function show(i) {
  cur = i;
  [...el('tabs').children].forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
  const c = DATA[i].crop; cv.width = c.w; cv.height = c.h;
  if (rows === undefined) { rows = DATA[i].py.rows; cols = DATA[i].py.cols; el('rows').value = rows; el('cols').value = cols; }
  selected.clear(); hover = -1; recompute(); draw(); panel();
}
DATA.forEach((d, i) => {
  const b = document.createElement('button'); b.textContent = d.name; b.addEventListener('click', () => show(i));
  el('tabs').appendChild(b);
  const im = new Image(); im.src = d.img; imgs.push(im);
});
let loaded = 0;
imgs.forEach(im => im.addEventListener('load', () => { if (++loaded === imgs.length) show(0); }));
</script>
</body>
</html>
"""
