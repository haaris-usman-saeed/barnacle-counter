"""Review interface: the Phase 2 grid viewer extended with detections,
review signals, corrections, undo, calibration, a results dashboard and
export. One template, two modes:

  static demo  (app=False) payloads embedded at build time (run_phase4.py)
  application  (app=True)  served by app.py; photos are uploaded, processed
               by the real model on the server, and reviewed here

All correction rules mirror barnacles.review, and the count estimator
mirrors barnacles.calibration, so an exported session replays identically
in Python (tests/test_review.py, tests/test_app.py).
"""
import json

from .grid_viewer import TEMPLATE as GRID_TEMPLATE

# Reuse the Phase 2 viewer's design tokens and layout CSS.
BASE_CSS = GRID_TEMPLATE[GRID_TEMPLATE.index("<style>") + 7:GRID_TEMPLATE.index("</style>")]


def render(payloads, app=False):
    return (TEMPLATE.replace("__CSS__", BASE_CSS).replace("__APP__", "true" if app else "false")
            .replace("__DATA__", json.dumps(payloads, separators=(",", ":"))))


def write_review_viewer(payloads, path, app=False):
    with open(path, "w") as f:
        f.write(render(payloads, app))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Barnacle count review</title>
<style>
__CSS__
main { grid-template-columns: minmax(0, 1fr) 380px; }
.tools { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.tools button { border: 0; background: transparent; color: var(--ink); padding: 6px 12px; font: inherit; cursor: pointer; }
.tools button[aria-pressed="true"] { background: var(--wire); color: var(--on-wire); }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
input[type=range] { width: 100%; }
.queue { list-style: none; margin: 8px 0 0; padding: 0; max-height: 240px; overflow: auto; border-top: 1px solid var(--rule); }
.queue li { padding: 6px 2px; border-bottom: 1px solid var(--rule); cursor: pointer; font-size: 13px; line-height: 1.35; }
.queue li[aria-selected="true"] { background: rgba(232,178,28,0.16); }
.queue .why { color: var(--muted); }
.inset { width: 100%; aspect-ratio: 1; background: #000; border-radius: 4px; display: block; margin-top: 8px; }
.btn { font: inherit; padding: 5px 10px; border: 1px solid var(--rule); background: var(--surface); color: var(--ink); border-radius: 4px; cursor: pointer; }
.btn.primary { background: var(--wire); color: var(--on-wire); border-color: var(--wire); }
.btn:disabled { opacity: 0.5; cursor: default; }
.small { font-size: 13px; color: var(--muted); }
.warn { border-left: 3px solid #e8b21c; padding: 6px 10px; background: rgba(232,178,28,0.12); border-radius: 3px; font-size: 14px; }
.err { border-left: 3px solid #b3261e; padding: 6px 10px; background: rgba(179,38,30,0.10); border-radius: 3px; font-size: 14px; }
#minimap { width: 100%; aspect-ratio: 1; display: block; margin-top: 8px; cursor: pointer; border-radius: 3px; }
.empty { padding: 60px 20px; text-align: center; color: var(--muted); border: 1px dashed var(--rule); border-radius: 6px; }
table.m { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
table.m td, table.m th { padding: 3px 4px; border-bottom: 1px solid var(--rule); text-align: right; }
table.m th:first-child, table.m td:first-child { text-align: left; }
</style>
</head>
<body>
<header>
  <h1>Barnacle count review</h1>
  <p id="sub">The model finds candidate barnacles; you confirm, remove or add them. Counts update as you go.</p>
</header>
<div class="controls">
  <div class="tabs" role="group" aria-label="Photo" id="tabs"></div>
  <div class="tools" role="group" aria-label="Tool" id="tools">
    <button data-tool="select" aria-pressed="true" title="S">Select cells</button>
    <button data-tool="mark" aria-pressed="false" title="M">Remove or restore</button>
    <button data-tool="add" aria-pressed="false" title="A">Add barnacle</button>
  </div>
  <button class="btn" id="undo" title="Ctrl+Z" disabled>Undo</button>
  <label>Rows <input id="rows" type="number" min="1" max="60"></label>
  <label>Columns <input id="cols" type="number" min="1" max="60"></label>
  <label><input id="dots" type="checkbox" checked> Detections</label>
  <label><input id="sugg" type="checkbox" checked> Suggestions</label>
  <label><input id="flags" type="checkbox"> Ring uncertain</label>
  <label id="annlab"><input id="ann" type="checkbox"> Reference annotations</label>
</div>
<main>
  <div class="stage"><div class="empty" id="empty" hidden>Upload a quadrat photo to start. The model runs on the server; nothing is precomputed.</div>
    <canvas id="cv" tabindex="0" aria-label="Photo with detections and grid"></canvas></div>
  <aside>
    <section id="upsec" hidden>
      <h2>Process a photo</h2>
      <input type="file" id="file" accept="image/png,image/jpeg">
      <label style="display:block;margin:10px 0"><input id="closeup" type="checkbox"> Experimental close-up mode (preserve original pixel scale; no green frame needed)</label>
      <div class="row"><button class="btn primary" id="proc">Process photo</button></div>
      <p class="small" id="upstatus">PNG or JPEG, up to 40 MB and 60 megapixels. Processing takes about 30 seconds.</p>
    </section>
    <section id="framesec" hidden>
      <h2>Frame</h2>
      <div id="framemsg"></div>
      <div class="row" id="framebtns">
        <button class="btn primary" id="fconfirm">Frame is correct</button>
        <button class="btn" id="fmanual">Set corners by hand</button>
      </div>
    </section>
    <section>
      <h2>Results</h2>
      <dl id="totals"></dl>
      <p class="small" id="defs"></p>
      <p class="hint" id="est"></p>
      <canvas id="minimap" width="300" height="300" aria-label="Corrected count per cell; outlined cells have the most items awaiting review"></canvas>
      <p class="small" id="hot"></p>
    </section>
    <section>
      <h2>Operating point</h2>
      <p class="small" id="opdesc"></p>
      <input id="thr" type="range" min="0" max="1000" step="1" aria-label="Detection threshold">
      <div class="row"><span class="small" id="thrval"></span><button class="btn" id="recal">Calibrate from my labels</button></div>
      <div id="calib" hidden>
        <canvas class="inset" id="calinset" width="240" height="240"></canvas>
        <p class="small" id="calprog"></p>
        <div class="row">
          <button class="btn primary" id="calyes">Opening</button>
          <button class="btn" id="calno">Not an opening</button>
          <button class="btn" id="calskip">Skip</button>
          <button class="btn" id="calback">Back</button>
        </div>
        <div class="row">
          <button class="btn" id="caldemo">Fill the rest from annotations (demo)</button>
          <button class="btn" id="calfinish">Finish</button>
        </div>
      </div>
      <div id="influence"></div>
    </section>
    <section>
      <h2>Selection</h2>
      <div id="sel"><p class="note">No cell selected. With the Select tool, click a cell; shift-click adds cells.</p></div>
    </section>
    <section>
      <h2>Review queue</h2>
      <p class="small" id="qdesc"></p>
      <canvas class="inset" id="inset" width="240" height="240" hidden></canvas>
      <div class="row" id="qbtns" hidden>
        <button class="btn primary" id="qyes">Accept (Y)</button>
        <button class="btn" id="qno">Reject (N)</button>
        <button class="btn" id="qnext">Next (J)</button>
      </div>
      <ul class="queue" id="queue"></ul>
    </section>
    <section id="accsec">
      <h2>Accuracy</h2>
      <div id="acc"></div>
      <div class="row" id="accup" hidden><label class="small">Compare with verified points (CSV with x,y in photo pixels)
        <input type="file" id="ref" accept=".csv,text/csv"></label></div>
    </section>
    <section>
      <h2>Export</h2>
      <div class="row">
        <button class="btn" id="exjson">Download session (JSON)</button>
        <button class="btn" id="excsv">Download points (CSV)</button>
      </div>
      <div class="row"><label class="small">Load session <input type="file" id="load" accept=".json,application/json"></label></div>
      <p class="note">The count is the model's detections plus your corrections. Model scores rank candidates; they are not probabilities until you calibrate on this photo.</p>
    </section>
  </aside>
</main>
<script>
const APP = __APP__;
const DATA = __DATA__;
const el = id => document.getElementById(id);
const cv = el('cv'), ctx = cv.getContext('2d');
const AMB = [0.25, 0.75], INFLUENCE_FLAG = 0.10;
let cur = -1, rows = 10, cols = 10, tool = 'select', selected = new Set(), hover = -1, imgs = [], focus = null, ST = [], V = null;
let corner = null;   // manual-corner mode: {points: [], preview: {img, scale, w, h} | null, file}
const files = [];    // uploaded File objects, for reprocessing with manual corners

const logit = s => { s = Math.min(Math.max(s, 1e-6), 1 - 1e-6); return Math.log(s / (1 - s)); };
const pc = (s, st) => 1 / (1 + Math.exp(-(st.a * logit(s) + st.b)));
function ap(M, x, y) { const w = M[2][0]*x + M[2][1]*y + M[2][2]; return [(M[0][0]*x + M[0][1]*y + M[0][2]) / w, (M[1][0]*x + M[1][1]*y + M[1][2]) / w]; }
function assign(u, v) {   // same rule as barnacles.grid.assign
  if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1)) return -1;
  return Math.min(Math.floor(v * rows), rows - 1) * cols + Math.min(Math.floor(u * cols), cols - 1);
}
const cellName = k => 'r' + String(Math.floor(k / cols)).padStart(2, '0') + 'c' + String(k % cols).padStart(2, '0');
function modelSet(d, thr) {     // same rule as barnacles.review.model_set
  const killed = new Uint8Array(d.n), keep = [];
  for (const i of d.order) { if (d.c.s[i] < thr) break; if (killed[i]) continue; keep.push(i); for (const j of d.nb[i]) killed[j] = 1; }
  return keep;
}
// ---- count estimator, mirrors barnacles.calibration.isotonic_count
function stratumSizes(d) {
  const S = d.op.sample, n = {}, w = {};
  S.forEach(s => { n[s[3]] = (n[s[3]] || 0) + 1; w[s[3]] = s[2]; });
  const out = {}; Object.keys(n).forEach(k => out[k] = Math.round(n[k] * w[k])); return out;
}
function isotonicCount(d, labels) {   // labels aligned with d.op.sample; null = skipped
  const sizes = stratumSizes(d), pos = {}, cnt = {};
  d.op.sample.forEach((s, k) => { if (labels[k] === null || labels[k] === undefined) return; pos[s[3]] = (pos[s[3]] || 0) + labels[k]; cnt[s[3]] = (cnt[s[3]] || 0) + 1; });
  const keys = Object.keys(cnt).map(Number).sort((a, b) => a - b);
  let blocks = keys.map(k => ({p: pos[k], n: cnt[k], s: [k]}));
  let i = 0;
  while (i < blocks.length - 1) {
    if (blocks[i].p / blocks[i].n > blocks[i + 1].p / blocks[i + 1].n) { const b = blocks.splice(i + 1, 1)[0]; blocks[i] = {p: blocks[i].p + b.p, n: blocks[i].n + b.n, s: blocks[i].s.concat(b.s)}; i = Math.max(i - 1, 0); }
    else i++;
  }
  let tot = 0; blocks.forEach(b => b.s.forEach(k => tot += (sizes[k] || 0) * b.p / b.n)); return tot;
}
function fitPlatt(d, labels) {        // weighted 1-D logistic regression (Newton), matches C=1e4
  const X = [], Y = [], W = [];
  d.op.sample.forEach((s, k) => { if (labels[k] !== null && labels[k] !== undefined) { X.push(logit(d.c.s[s[0]])); Y.push(labels[k]); W.push(s[2]); } });
  if (new Set(Y).size < 2) return null;
  let a = 1, b = 0;
  for (let it = 0; it < 100; it++) {
    let ga = -a / 1e4, gb = 0, haa = -1 / 1e4, hab = 0, hbb = -1e-9;
    X.forEach((x, k) => { const p = 1 / (1 + Math.exp(-(a * x + b))), w = W[k], r = Y[k] - p, q = p * (1 - p) * w;
      ga += w * r * x; gb += w * r; haa -= q * x * x; hab -= q * x; hbb -= q; });
    const det = haa * hbb - hab * hab; if (Math.abs(det) < 1e-12) break;
    const da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det; a -= da; b -= db; if (Math.abs(da) + Math.abs(db) < 1e-10) break;
  }
  return a > 0 ? [a, b] : null;
}
function thresholdForCount(d, n) {   // mirrors ml.select_count
  let lo = 0, hi = d.uniq.length - 1, thr = d.uniq[hi];
  if (n <= 0) return d.uniq[0] + 1e-9;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (modelSet(d, d.uniq[mid]).length >= n) { thr = d.uniq[mid]; hi = mid - 1; } else lo = mid + 1; }
  return thr;
}
// ---- payloads and state
function prep(d) {
  const n = d.c.s.length; d.n = n;
  d.order = Array.from({length: n}, (_, i) => i).sort((i, j) => (d.c.s[j] - d.c.s[i]) || (i - j));
  d.nb = Array.from({length: n}, () => []);
  for (let k = 0; k < d.pairs.length; k += 2) { d.nb[d.pairs[k]].push(d.pairs[k + 1]); d.nb[d.pairs[k + 1]].push(d.pairs[k]); }
  d.uniq = [...new Set(d.c.s)].sort((x, y) => y - x);
  d.idOf = new Map(d.c.id.map((id, i) => [id, i]));
  d.annUV = (d.ann || []).map(a => ap(d.H, a[0], a[1]));
  d.initialCount = modelSet(d, d.op.thr).length;
  d.hasLabels = d.op.sample.some(s => s[1] !== null);
}
function newState(d) {
  return {thr: d.op.thr, a: d.op.a, b: d.op.b, calibrated: d.op.calibrated !== false, opMethod: d.op.method, nHat: d.op.n_hat,
          overrides: new Map(), added: new Map(), actions: [], seq: 0, calib: null, calLabels: null, influence: [],
          frameConfirmed: d.frame ? d.frame.confirmed : true};
}
function addPayload(d, file) {
  prep(d); DATA.push(d); ST.push(newState(d)); files.push(file || null);
  const im = new Image(); im.src = d.img; imgs.push(im);
  const b = document.createElement('button'); b.textContent = d.file; const ix = DATA.length - 1;
  b.addEventListener('click', () => show(ix)); el('tabs').appendChild(b);
  return new Promise(res => im.addEventListener('load', () => res(ix)));
}
function replacePayload(ix, d) {
  prep(d); DATA[ix] = d; ST[ix] = newState(d);
  const im = new Image(); im.src = d.img; imgs[ix] = im;
  return new Promise(res => im.addEventListener('load', () => res(ix)));
}
function recompute() {
  const d = DATA[cur], st = ST[cur];
  const mset = modelSet(d, st.thr), inM = new Uint8Array(d.n); mset.forEach(i => inM[i] = 1);
  const counted = new Uint8Array(d.n);
  mset.forEach(i => { if (st.overrides.get(d.c.id[i]) !== 'remove') counted[i] = 1; });
  st.overrides.forEach((v, id) => { if (v === 'keep' && d.idOf.has(id)) counted[d.idOf.get(id)] = 1; });
  const p = d.c.s.map(s => pc(s, st));
  const covered = new Uint8Array(d.n);
  for (let i = 0; i < d.n; i++) if (counted[i]) for (const j of d.nb[i]) covered[j] = 1;
  // Queue (mirrors barnacles.review.review_queue): chance the item is a counting error,
  // 1 - p for counted model detections, p for suggestions; +0.05 per static flag.
  const items = [], nSugg = Math.ceil(0.25 * Math.max(mset.length, 1));
  const below = []; for (let i = 0; i < d.n; i++) if (!counted[i] && !covered[i] && d.c.s[i] < st.thr && st.overrides.get(d.c.id[i]) !== 'remove') below.push(i);
  below.sort((i, j) => (d.c.s[j] - d.c.s[i]) || (i - j));
  const sugg = new Uint8Array(d.n); below.slice(0, nSugg).forEach(i => sugg[i] = 1);
  for (let i = 0; i < d.n; i++) {
    const ov = st.overrides.get(d.c.id[i]);
    if (counted[i] && inM[i] && ov !== 'keep') {
      const why = [], nf = d.c.dense[i] + d.c.clar[i] + d.c.edge[i] + (d.c.cvp[i] < 0.5 ? 1 : 0);
      if (st.calibrated && p[i] >= AMB[0] && p[i] <= AMB[1]) why.push('ambiguous');
      if (d.nb[i].some(j => !counted[j] && p[j] >= AMB[0])) why.push('competing overlap');
      if (d.c.cvp[i] < 0.5) why.push('classical score low');
      if (d.c.dense[i]) why.push('crowded');
      if (d.c.clar[i]) why.push('low clarity');
      if (d.c.edge[i]) why.push('touches frame');
      // a detection awaits review if something specific flags it, or its (calibrated) score is below 0.5
      // the classical-score flag is weak on its own, so it adds a reason but never queues an item by itself
      if (why.some(w => w !== 'classical score low') || p[i] < 0.5) items.push({i, kind: 'detection', prio: 1 - p[i] + 0.05 * nf, why: why.length ? why : ['score below 0.5']});
    } else if (sugg[i]) items.push({i, kind: 'suggestion', prio: p[i], why: ['possible miss, just below threshold']});
  }
  items.sort((x, y) => (y.prio - x.prio) || (x.i - y.i));
  const pts = [];
  for (let i = 0; i < d.n; i++) if (counted[i]) pts.push([d.c.x[i], d.c.y[i], d.c.u[i], d.c.v[i],
    st.overrides.get(d.c.id[i]) === 'keep' ? (inM[i] ? 'reviewer_kept' : 'reviewer_accepted_suggestion') : 'model', d.c.id[i], d.c.s[i]]);
  st.added.forEach((a, id) => pts.push([a.x, a.y, a.u, a.v, 'reviewer_added', id, null]));
  const nC = rows * cols, cellC = new Array(nC).fill(0), cellM = new Array(nC).fill(0), cellA = new Array(nC).fill(0), cellU = new Array(nC).fill(0), cellQ = new Array(nC).fill(0);
  pts.forEach(q => { const k = assign(q[2], q[3]); if (k >= 0) cellC[k]++; });
  mset.forEach(i => { const k = assign(d.c.u[i], d.c.v[i]); if (k >= 0) { cellM[k]++; if (st.calibrated && p[i] >= AMB[0] && p[i] <= AMB[1]) cellU[k]++; } });
  items.forEach(it => { const k = assign(d.c.u[it.i], d.c.v[it.i]); if (k >= 0) cellQ[k]++; });
  d.annUV.forEach(q => { const k = assign(q[0], q[1]); if (k >= 0) cellA[k]++; });
  let cellCov = null;
  if (d.cov) {
    cellCov = new Array(nC).fill(0); const cellN = new Array(nC).fill(0), g = d.cov.n;
    for (let a = 0; a < g; a++) for (let b = 0; b < g; b++) { const k = assign((b + 0.5) / g, (a + 0.5) / g); cellN[k]++; if (d.cov.bits[a * g + b] === '0') cellCov[k]++; }
    cellCov = cellCov.map((c, k) => c / Math.max(cellN[k], 1));
  }
  V = {mset, inM, counted, p, items, pts, cellC, cellM, cellA, cellU, cellQ, cellCov};
}
// ---- drawing
function toDisp(x, y) { const c = corner && corner.preview ? {x0: 0, y0: 0, scale: corner.preview.scale} : DATA[cur].crop; return [(x - c.x0) * c.scale, (y - c.y0) * c.scale]; }
function dispToImg(dx, dy) { const c = corner && corner.preview ? {x0: 0, y0: 0, scale: corner.preview.scale} : DATA[cur].crop; return [dx / c.scale + c.x0, dy / c.scale + c.y0]; }
function uvDisp(u, v) { const q = ap(DATA[cur].Hinv, u, v); return toDisp(q[0], q[1]); }
function cellPath(k) {
  const r = Math.floor(k / cols), c = k % cols;
  const P = [[c/cols, r/rows], [(c+1)/cols, r/rows], [(c+1)/cols, (r+1)/rows], [c/cols, (r+1)/rows]].map(q => uvDisp(q[0], q[1]));
  ctx.beginPath(); ctx.moveTo(...P[0]); P.slice(1).forEach(q => ctx.lineTo(...q)); ctx.closePath();
}
function dot(x, y, r, fill) { const q = toDisp(x, y); ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(0,0,0,0.75)'; ctx.stroke(); }
function ring(x, y, r, col, w) { const q = toDisp(x, y); ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 2 * Math.PI); ctx.lineWidth = w || 1.5; ctx.strokeStyle = col; ctx.stroke(); }
function drawCorners() {
  const im = corner.preview ? corner.previewImg : imgs[cur];
  if (corner.preview) { cv.width = corner.preview.w; cv.height = corner.preview.h; }
  ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(im, 0, 0);
  corner.points.forEach((p, k) => { const q = toDisp(p[0], p[1]); ctx.beginPath(); ctx.arc(q[0], q[1], 6, 0, 2 * Math.PI); ctx.fillStyle = '#e8b21c'; ctx.fill();
    ctx.fillStyle = '#000'; ctx.font = '13px sans-serif'; ctx.fillText(['TL', 'TR', 'BR', 'BL'][k], q[0] + 8, q[1] - 8); });
  if (corner.points.length > 1) { ctx.beginPath(); corner.points.forEach((p, k) => { const q = toDisp(p[0], p[1]); k ? ctx.lineTo(...q) : ctx.moveTo(...q); }); ctx.strokeStyle = '#e8b21c'; ctx.lineWidth = 2; ctx.stroke(); }
}
function draw() {
  if (corner) return drawCorners();
  if (cur < 0) return;
  const d = DATA[cur], st = ST[cur];
  ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(imgs[cur], 0, 0);
  ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,0.38)'; ctx.beginPath();
  for (let c = 1; c < cols; c++) { ctx.moveTo(...uvDisp(c/cols, 0)); ctx.lineTo(...uvDisp(c/cols, 1)); }
  for (let r = 1; r < rows; r++) { ctx.moveTo(...uvDisp(0, r/rows)); ctx.lineTo(...uvDisp(1, r/rows)); }
  ctx.stroke();
  const q = d.corners.map(p => toDisp(p[0], p[1]));
  ctx.beginPath(); ctx.moveTo(...q[0]); q.slice(1).forEach(p => ctx.lineTo(...p)); ctx.closePath(); ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,230,120,0.9)'; ctx.stroke();
  for (const k of selected) { cellPath(k); ctx.fillStyle = 'rgba(232,178,28,0.14)'; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#e8b21c'; ctx.stroke(); }
  if (hover >= 0 && tool === 'select' && !selected.has(hover)) { cellPath(hover); ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.stroke(); }
  if (el('ann').checked) d.ann.forEach(a => { const p = toDisp(a[0], a[1]); ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1; ctx.strokeRect(p[0] - 3, p[1] - 3, 6, 6); });
  if (el('sugg').checked) V.items.filter(it => it.kind === 'suggestion').forEach(it => ring(d.c.x[it.i], d.c.y[it.i], 4.5, 'rgba(255,190,60,0.95)'));
  if (tool === 'mark') st.overrides.forEach((v, id) => { if (v === 'remove' && d.idOf.has(id)) { const i = d.idOf.get(id), p = toDisp(d.c.x[i], d.c.y[i]);
    ctx.strokeStyle = 'rgba(200,200,200,0.9)'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(p[0]-3, p[1]-3); ctx.lineTo(p[0]+3, p[1]+3); ctx.moveTo(p[0]+3, p[1]-3); ctx.lineTo(p[0]-3, p[1]+3); ctx.stroke(); } });
  if (el('dots').checked) {
    V.pts.forEach(t => dot(t[0], t[1], t[4] === 'model' ? 2.4 : 3, t[4] === 'model' ? '#ff2bd6' : (t[4] === 'reviewer_added' ? '#33d6ff' : '#22c08a')));
    if (el('flags').checked) V.items.filter(it => it.kind === 'detection' && V.p[it.i] <= AMB[1]).forEach(it => ring(d.c.x[it.i], d.c.y[it.i], 5, 'rgba(255,190,60,0.9)', 1));
  }
  if (focus) ring(focus.x, focus.y, 9, '#ffffff', 2);
}
function drawMinimap() {
  const m = el('minimap'), c = m.getContext('2d'), W = m.width, H = m.height, mx = Math.max(1, ...V.cellC), qmax = Math.max(...V.cellQ);
  c.clearRect(0, 0, W, H);
  const top = V.cellQ.map((q, k) => [q, k]).sort((a, b) => b[0] - a[0]).slice(0, 3).filter(t => t[0] > 0).map(t => t[1]);
  for (let k = 0; k < rows * cols; k++) {
    const r = Math.floor(k / cols), cc = k % cols, x = cc * W / cols, y = r * H / rows, t = V.cellC[k] / mx;
    c.fillStyle = `rgb(${Math.round(235 - 205 * t)},${Math.round(240 - 124 * t)},${Math.round(238 - 126 * t)})`; c.fillRect(x, y, W / cols - 1, H / rows - 1);
    if (top.includes(k)) { c.strokeStyle = '#e8b21c'; c.lineWidth = 3; c.strokeRect(x + 1.5, y + 1.5, W / cols - 4, H / rows - 4); }
    if (selected.has(k)) { c.strokeStyle = '#1d2a2c'; c.lineWidth = 2; c.strokeRect(x + 1, y + 1, W / cols - 3, H / rows - 3); }
  }
  el('hot').textContent = top.length ? 'Most items awaiting review: ' + top.map(k => `${cellName(k)} (${V.cellQ[k]})`).join(', ') + '. Darker cells have more barnacles.' : 'Nothing is waiting for review.';
}
// ---- panels
function fmt(n) { return Number.isFinite(n) ? Math.round(n).toLocaleString('en-US') : 'n/a'; }
function panel() {
  const d = DATA[cur], st = ST[cur];
  const B = breakdown(), nDet = V.items.filter(it => it.kind === 'detection').length, nSug = V.items.length - nDet;
  el('totals').innerHTML =
    `<dt title="Candidates the model counts at the current threshold, after duplicate suppression">Selected model detections</dt><dd>${fmt(B.selected)}</dd>` +
    `<dt title="Selected model detections you removed">Removed by you</dt><dd>${fmt(B.removed)}</dd>` +
    `<dt title="Candidates below the threshold that you accepted as barnacles">Accepted suggestions</dt><dd>${fmt(B.acceptedSuggestions)}</dd>` +
    `<dt title="Barnacles you clicked in that the model did not propose">Added by you</dt><dd>${fmt(B.added)}</dd>` +
    `<dt title="Selected minus removed, plus accepted suggestions, plus added">Corrected count</dt><dd><strong>${fmt(V.pts.length)}</strong></dd>` +
    `<dt title="Detections confirmed, detections removed, suggestions accepted, suggestions rejected">Reviewed items</dt><dd>${fmt(B.reviewed)}</dd>` +
    `<dt title="Selected detections flagged for a check, and suggestions not yet decided">Still to review</dt><dd>${fmt(nDet)} detections, ${fmt(nSug)} suggestions</dd>` +
    (d.ann.length ? `<dt>Reference annotations (incomplete)</dt><dd>${fmt(d.ann.length)}</dd>` : '');
  el('defs').textContent = `${fmt(B.selected)} selected - ${fmt(B.removed)} removed + ${fmt(B.acceptedSuggestions)} accepted suggestions + ${fmt(B.added)} added = ${fmt(V.pts.length)} corrected` +
    (B.selected - B.removed + B.acceptedSuggestions + B.added === V.pts.length ? '' : ' (MISMATCH: please report)') +
    `. Model count when the photo was opened: ${fmt(d.initialCount)}. Suggestions are extra candidates offered for checking; they are not counted unless you accept them.`;
  el('est').textContent = st.nHat != null ? `Count estimate from the calibration labels: ${fmt(st.nHat)}.` : 'No calibration yet: the count uses a default threshold from the training photos.';
  el('opdesc').textContent = `${st.opMethod}. Model: ${d.model.kind}, trained on ${d.model.trained_on}.`;
  el('thrval').textContent = st.calibrated ? `Score threshold ${st.thr.toFixed(3)}; calibrated estimate at threshold ${pc(st.thr, st).toFixed(2)}` : `Score threshold ${st.thr.toFixed(3)} (uncalibrated)`;
  el('thr').value = Math.round(1000 * Math.max(0, d.uniq.findIndex(s => s <= st.thr)) / Math.max(d.uniq.length - 1, 1));
  el('caldemo').hidden = !d.hasLabels;
  // frame
  el('framesec').hidden = !(d.frame);
  if (d.frame) {
    const w = d.frame.warnings || [];
    el('framemsg').innerHTML = (st.frameConfirmed ? `<p class="small">Frame ${d.frame.source === 'manual' ? 'set by hand' : 'confirmed'}.</p>` :
      `<p class="warn">Frame found automatically. Check that the yellow outline follows the inner edge of the wire before trusting the counts.</p>`) +
      (w.length ? `<p class="warn">${w.join('<br>')}</p>` : '') +
      (d.training_photo ? `<p class="warn">${d.training_photo.message}</p>` : '');
    el('fconfirm').hidden = st.frameConfirmed;
    el('fmanual').hidden = !APP;
  }
  // selection
  if (!selected.size) el('sel').innerHTML = '<p class="note">No cell selected. With the Select tool, click a cell; shift-click adds cells.</p>';
  else {
    const ks = [...selected], sum = a => ks.reduce((s, k) => s + a[k], 0), m = sum(V.cellM), unc = sum(V.cellU);
    const meanM = V.cellM.reduce((a, b) => a + b, 0) / V.cellM.length;
    el('sel').innerHTML = `<dl><dt>${ks.length === 1 ? 'Cell' : 'Cells'}</dt><dd>${ks.length === 1 ? cellName(ks[0]) : ks.length}</dd>` +
      `<dt>Model detections</dt><dd>${m}</dd><dt>Corrected count</dt><dd><strong>${sum(V.cellC)}</strong></dd>` +
      `<dt>Awaiting review</dt><dd>${sum(V.cellQ)}</dd>` +
      (st.calibrated ? `<dt>Uncertain share (detection confidence)</dt><dd>${m ? Math.round(100 * unc / m) + '%' : 'n/a'}</dd>` : '') +
      `<dt>Detection density vs photo average</dt><dd>${meanM ? (m / ks.length / meanM).toFixed(2) + 'x' : 'n/a'}</dd>` +
      (V.cellCov ? `<dt>Reference annotation coverage</dt><dd>${Math.round(100 * ks.reduce((s, k) => s + V.cellCov[k], 0) / ks.length)}%</dd>` +
                   `<dt>Reference annotations</dt><dd>${sum(V.cellA)}</dd>` : '') + '</dl>';
  }
  const inSel = it => !selected.size || selected.has(assign(d.c.u[it.i], d.c.v[it.i]));
  const q = V.items.filter(inSel);
  el('qdesc').textContent = `${q.length} items ${selected.size ? 'in the selection' : 'on this photo'}, ordered by ${st.calibrated ? 'the calibrated chance that each is a counting error' : 'model score (calibrate for better ordering)'}.`;
  el('queue').innerHTML = q.slice(0, 60).map(it =>
    `<li data-i="${it.i}" aria-selected="${focus && focus.i === it.i}">${it.kind === 'suggestion' ? 'Possible miss' : 'Detection'} ${d.c.id[it.i].split('_').pop()}, ` +
    `${st.calibrated ? 'est.' : 'score'} ${V.p[it.i].toFixed(2)}<br><span class="why">${it.why.join(', ')}</span></li>`).join('');
  [...el('queue').children].forEach(li => li.addEventListener('click', () => setFocus(+li.dataset.i)));
  el('undo').disabled = !st.actions.length;
  // accuracy
  el('accup').hidden = !APP;
  if (!st.accuracy) el('acc').innerHTML = d.ann.length ? `<p class="small">Reference annotations exist for this photo but are incomplete, so unmatched detections are not necessarily wrong. Detailed numbers: TECHNICAL_REPORT.md.</p>`
    : `<p class="small">True detection accuracy is unknown for this photo: there are no verified annotations.${APP ? ' Upload verified points to compare.' : ''}</p>`;
  else el('acc').innerHTML = st.accuracy;
  drawMinimap();
}
function breakdown() {   // mirrors barnacles.review.replay()["breakdown"]
  const d = DATA[cur], st = ST[cur], B = {selected: V.mset.length, confirmed: 0, removed: 0, acceptedSuggestions: 0, rejectedSuggestions: 0, added: st.added.size};
  st.overrides.forEach((v, id) => { if (!d.idOf.has(id)) return; const m = V.inM[d.idOf.get(id)];
    if (v === 'keep') m ? B.confirmed++ : B.acceptedSuggestions++; else m ? B.removed++ : B.rejectedSuggestions++; });
  B.reviewed = B.confirmed + B.removed + B.acceptedSuggestions + B.rejectedSuggestions;
  return B;
}
function setFocus(i) {
  const d = DATA[cur];
  focus = i == null ? null : {i, x: d.c.x[i], y: d.c.y[i]};
  el('inset').hidden = el('qbtns').hidden = !focus;
  if (focus) drawInset(el('inset'), focus.x, focus.y);
  draw(); panel();
}
function drawInset(canvas, x, y) {
  const c = canvas.getContext('2d'), p = toDisp(x, y), h = 30;
  c.imageSmoothingEnabled = true; c.drawImage(imgs[cur], p[0] - h, p[1] - h, 2 * h, 2 * h, 0, 0, canvas.width, canvas.height);
  c.beginPath(); c.arc(canvas.width / 2, canvas.height / 2, canvas.width * 0.12, 0, 2 * Math.PI); c.strokeStyle = '#fff'; c.lineWidth = 1.5; c.stroke();
}
// ---- actions (all recorded, replayable, undoable)
function act(a) { const st = ST[cur]; a.seq = ++st.seq; st.actions.push(a); applyAction(cur, a); refresh(); }
function applyAction(ix, a) {
  const st = ST[ix];
  if (a.action === 'keep') st.overrides.set(a.id, 'keep');
  else if (a.action === 'remove') { if (st.added.has(a.id)) st.added.delete(a.id); else st.overrides.set(a.id, 'remove'); }
  else if (a.action === 'restore') st.overrides.delete(a.id);
  else if (a.action === 'add') st.added.set(a.id, {x: a.x, y: a.y, u: a.u, v: a.v});
  else if (a.action === 'set_threshold') st.thr = a.threshold;
  else if (a.action === 'set_calibration') { st.a = a.a; st.b = a.b; st.opMethod = a.method; st.thr = a.threshold; st.calibrated = true; st.nHat = a.n_hat; st.calLabels = a.labels.map(l => l.label); }
  else if (a.action === 'confirm_frame') st.frameConfirmed = true;
}
function rebuild(ix) {
  const acts = ST[ix].actions.slice(), seq = ST[ix].seq, acc = ST[ix].accuracy;
  ST[ix] = newState(DATA[ix]); ST[ix].seq = seq; ST[ix].accuracy = acc;
  acts.forEach(a => { ST[ix].actions.push(a); applyAction(ix, a); });
}
function undo() { const st = ST[cur]; if (!st.actions.length) return; st.actions.pop(); rebuild(cur); el('influence').innerHTML = ''; refresh(); }
function refresh() { if (cur < 0) return; recompute(); draw(); panel(); }
function nearest(dx, dy, maxd) {
  const d = DATA[cur], st = ST[cur]; let best = null, bd = maxd * maxd;
  const sug = new Set(V.items.filter(it => it.kind === 'suggestion').map(it => it.i));
  const test = (x, y, ref) => { const p = toDisp(x, y), e = (p[0] - dx) ** 2 + (p[1] - dy) ** 2; if (e < bd) { bd = e; best = ref; } };
  for (let i = 0; i < d.n; i++) if (V.counted[i] || st.overrides.get(d.c.id[i]) === 'remove' || (el('sugg').checked && sug.has(i))) test(d.c.x[i], d.c.y[i], {type: 'cand', i});
  st.added.forEach((a, id) => test(a.x, a.y, {type: 'added', id}));
  return best;
}
function evtPos(e) { const rect = cv.getBoundingClientRect(); return [(e.clientX - rect.left) * cv.width / rect.width, (e.clientY - rect.top) * cv.height / rect.height]; }
cv.addEventListener('mousemove', e => { if (corner || cur < 0 || tool !== 'select') return; const [dx, dy] = evtPos(e), [x, y] = dispToImg(dx, dy), uv = ap(DATA[cur].H, x, y), k = assign(uv[0], uv[1]); if (k !== hover) { hover = k; draw(); } });
cv.addEventListener('click', e => {
  const [dx, dy] = evtPos(e);
  if (corner) { corner.points.push(dispToImg(dx, dy)); draw(); if (corner.points.length === 4) submitCorners(); return; }
  if (cur < 0) return;
  const d = DATA[cur], st = ST[cur], [x, y] = dispToImg(dx, dy), uv = ap(d.H, x, y);
  if (tool === 'select') {
    const k = assign(uv[0], uv[1]); if (k < 0) return;
    if (e.shiftKey || e.metaKey || e.ctrlKey) { selected.has(k) ? selected.delete(k) : selected.add(k); } else selected = new Set(selected.size === 1 && selected.has(k) ? [] : [k]);
    draw(); panel(); return;
  }
  if (tool === 'mark') {
    const t = nearest(dx, dy, 8); if (!t) return;
    if (t.type === 'added') return act({action: 'remove', id: t.id});
    const id = d.c.id[t.i], ov = st.overrides.get(id);
    if (ov === 'remove') act({action: 'restore', id}); else if (V.counted[t.i]) act({action: 'remove', id}); else act({action: 'keep', id});
    return;
  }
  if (tool === 'add') {
    if (assign(uv[0], uv[1]) < 0 || nearest(dx, dy, 4)) return;
    act({action: 'add', id: `${d.name}_add${String(st.seq + 1).padStart(4, '0')}`, x, y, u: uv[0], v: uv[1]});
  }
});
el('minimap').addEventListener('click', e => { if (cur < 0) return; const r = e.target.getBoundingClientRect(), c = Math.floor((e.clientX - r.left) / r.width * cols), rr = Math.floor((e.clientY - r.top) / r.height * rows); selected = new Set([rr * cols + c]); draw(); panel(); });
function setTool(t) { tool = t; [...el('tools').children].forEach(b => b.setAttribute('aria-pressed', String(b.dataset.tool === t))); cv.style.cursor = t === 'add' ? 'copy' : (t === 'mark' ? 'pointer' : 'crosshair'); draw(); }
[...el('tools').children].forEach(b => b.addEventListener('click', () => setTool(b.dataset.tool)));
function queueStep(dir) {
  const d = DATA[cur], inSel = it => !selected.size || selected.has(assign(d.c.u[it.i], d.c.v[it.i]));
  const q = V.items.filter(inSel); if (!q.length) return setFocus(null);
  let k = focus ? q.findIndex(it => it.i === focus.i) : -1; k = Math.max(0, Math.min(q.length - 1, k + dir)); setFocus(q[k].i);
}
function decide(keep) {
  if (!focus) return; const d = DATA[cur], id = d.c.id[focus.i], it = V.items.find(x => x.i === focus.i);
  if (keep) act({action: 'keep', id}); else if (it && it.kind === 'suggestion') act({action: 'remove', id}); else act({action: 'remove', id});
  queueStep(0);
}
el('qyes').addEventListener('click', () => decide(true)); el('qno').addEventListener('click', () => decide(false)); el('qnext').addEventListener('click', () => queueStep(1));
el('undo').addEventListener('click', undo);
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || cur < 0) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); return undo(); }
  const k = e.key.toLowerCase();
  if (k === 's') setTool('select'); else if (k === 'm') setTool('mark'); else if (k === 'a') setTool('add');
  else if (k === 'y') decide(true); else if (k === 'n') decide(false); else if (k === 'j') queueStep(1); else if (k === 'k') queueStep(-1);
  else if (k === 'escape') { selected.clear(); setFocus(null); }
});
el('thr').addEventListener('change', () => { const d = DATA[cur], k = Math.round(+el('thr').value / 1000 * (d.uniq.length - 1)); act({action: 'set_threshold', threshold: d.uniq[k]}); });
['rows', 'cols'].forEach(id => el(id).addEventListener('change', () => { const r = +el('rows').value, c = +el('cols').value; if (r >= 1 && r <= 60 && c >= 1 && c <= 60) { rows = r; cols = c; selected.clear(); refresh(); } }));
['dots', 'sugg', 'flags', 'ann'].forEach(id => el(id).addEventListener('change', refresh));
el('fconfirm').addEventListener('click', () => act({action: 'confirm_frame'}));
// ---- calibration: exactly the requested sample; isotonic count with an influence check
function startCal() { const st = ST[cur]; st.calib = {k: 0, labels: DATA[cur].op.sample.map(() => null)}; el('calib').hidden = false; el('influence').innerHTML = ''; showCal(); }
function showCal() {
  const d = DATA[cur], st = ST[cur], S = d.op.sample, k = st.calib.k;
  if (k >= S.length) { el('calprog').textContent = `All ${S.length} sample candidates seen. Press Finish.`; return; }
  drawInset(el('calinset'), d.c.x[S[k][0]], d.c.y[S[k][0]]);
  el('calprog').textContent = `Candidate ${k + 1} of ${S.length}: is the circled point the centre of a barnacle opening not already counted nearby?`;
}
function calAnswer(lab) { const st = ST[cur]; if (!st.calib) return; st.calib.labels[st.calib.k] = lab; st.calib.k++; showCal(); }
el('recal').addEventListener('click', startCal);
el('calyes').addEventListener('click', () => calAnswer(1)); el('calno').addEventListener('click', () => calAnswer(0)); el('calskip').addEventListener('click', () => calAnswer(null));
el('calback').addEventListener('click', () => { const st = ST[cur]; if (st.calib && st.calib.k > 0) { st.calib.k--; showCal(); } });
el('caldemo').addEventListener('click', () => { const st = ST[cur], S = DATA[cur].op.sample; for (let k = st.calib.k; k < S.length; k++) st.calib.labels[k] = S[k][1]; st.calib.k = S.length; showCal(); });
function applyCalibration(labels) {
  const d = DATA[cur], S = d.op.sample, ab = fitPlatt(d, labels);
  if (!ab) { el('calprog').textContent = 'Calibration needs at least one opening and one non-opening, with scores that separate them.'; return false; }
  const nHat = isotonicCount(d, labels), thr = thresholdForCount(d, Math.round(nHat));
  const used = labels.filter(l => l !== null).length;
  act({action: 'set_calibration', a: ab[0], b: ab[1], threshold: thr, n_hat: nHat, estimator: 'isotonic',
       method: `Calibrated from ${used} of your labels (count estimate ${Math.round(nHat)})`,
       labels: labels.map((l, k) => ({index: S[k][0], id: d.c.id[S[k][0]], label: l}))});
  // influence check: which single labels, if wrong, would move the count by more than 10%?
  const infl = [];
  labels.forEach((l, k) => { if (l === null) return; const l2 = labels.slice(); l2[k] = 1 - l; const dn = isotonicCount(d, l2) - nHat; if (Math.abs(dn) > INFLUENCE_FLAG * Math.max(nHat, 1)) infl.push([k, dn]); });
  showInfluence(labels, infl);
  return true;
}
function showInfluence(labels, infl) {
  const d = DATA[cur], S = d.op.sample;
  if (!infl.length) { el('influence').innerHTML = '<p class="small">No single label changes the count estimate by more than 10%.</p>'; return; }
  el('influence').innerHTML = `<p class="warn">${infl.length} label${infl.length > 1 ? 's' : ''} would each move the count by more than 10% if wrong. Please confirm:</p>` +
    infl.map(([k, dn]) => `<div class="row"><canvas class="inset" style="width:90px" width="90" height="90" data-k="${k}"></canvas>` +
      `<span class="small">Labelled ${labels[k] ? 'opening' : 'not an opening'}; if wrong, count ${dn > 0 ? '+' : ''}${Math.round(dn)}</span>` +
      `<button class="btn" data-ok="${k}">Correct</button><button class="btn" data-flip="${k}">Flip</button></div>`).join('');
  el('influence').querySelectorAll('canvas').forEach(c => { const i = S[+c.dataset.k][0]; drawInset(c, d.c.x[i], d.c.y[i]); });
  el('influence').querySelectorAll('[data-ok]').forEach(b => b.addEventListener('click', () => b.parentElement.remove()));
  el('influence').querySelectorAll('[data-flip]').forEach(b => b.addEventListener('click', () => { const l2 = labels.slice(), k = +b.dataset.flip; l2[k] = 1 - l2[k]; applyCalibration(l2); }));
}
el('calfinish').addEventListener('click', () => { const st = ST[cur]; if (applyCalibration(st.calib.labels)) { el('calib').hidden = true; st.calib = null; } });
// ---- export / import
function sessionObj() {
  const d = DATA[cur], st = ST[cur];
  const dets = V.pts.map(t => ({id: t[5], x: t[0], y: t[1], u: t[2], v: t[3], source: t[4], score: t[6],
    cell_id: assign(t[2], t[3]) >= 0 ? cellName(assign(t[2], t[3])) : null,
    review_status: t[4] === 'model' ? 'unreviewed' : (t[4] === 'reviewer_added' ? 'added' : 'accepted')}));
  dets.sort((p, q) => p.id < q.id ? -1 : 1);
  const perC = {}, perM = {}; V.cellC.forEach((n, k) => { if (n) perC[cellName(k)] = n; }); V.cellM.forEach((n, k) => { if (n) perM[cellName(k)] = n; });
  const B = breakdown();
  return {schema: 'barnacle-review/1', image: d.file, image_size: d.image_size || null, model: d.model, exported_at: new Date().toISOString(),
    frame: {corners_px: d.corners, source: d.frame ? d.frame.source : 'auto', confirmed: st.frameConfirmed},
    operating_point: {threshold: st.thr, platt_a_b: [st.a, st.b], calibrated: st.calibrated, method: st.opMethod, n_hat: st.nHat,
                      initial_threshold: d.op.thr, initial_platt_a_b: [d.op.a, d.op.b]},
    grid: {rows, cols}, actions: st.actions,
    result: {threshold: st.thr, initial_model_count: d.initialCount, model_count: V.mset.length, corrected_count: V.pts.length,
             breakdown: {selected_model: B.selected, confirmed: B.confirmed, removed: B.removed, accepted_suggestions: B.acceptedSuggestions,
                         rejected_suggestions: B.rejectedSuggestions, added: B.added, reviewed: B.reviewed,
                         still_to_review_detections: V.items.filter(it => it.kind === 'detection').length,
                         still_to_review_suggestions: V.items.filter(it => it.kind === 'suggestion').length},
             detections: dets, per_cell_corrected: perC, per_cell_model: perM}};
}
function download(name, text, type) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type})); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }
el('exjson').addEventListener('click', () => { if (cur >= 0) download(`${DATA[cur].name}_review_session.json`, JSON.stringify(sessionObj(), null, 1), 'application/json'); });
el('excsv').addEventListener('click', () => { if (cur < 0) return; const s = sessionObj(); download(`${DATA[cur].name}_corrected_points.csv`,
  'id,x,y,u,v,cell_id,source,review_status,score\n' + s.result.detections.map(q => [q.id, q.x.toFixed(2), q.y.toFixed(2), q.u.toFixed(6), q.v.toFixed(6), q.cell_id || '', q.source, q.review_status, q.score == null ? '' : q.score.toFixed(5)].join(',')).join('\n') + '\n', 'text/csv'); });
function loadSession(s) {
  const ix = DATA.findIndex(d => d.file === s.image);
  if (ix < 0) { alert(`This session is for ${s.image}. Open or process that photo first.`); return; }
  ST[ix].actions = s.actions.slice(); ST[ix].seq = Math.max(0, ...s.actions.map(a => a.seq || 0)); rebuild(ix);
  rows = s.grid.rows; cols = s.grid.cols; el('rows').value = rows; el('cols').value = cols; show(ix);
}
el('load').addEventListener('change', async e => { const f = e.target.files[0]; if (f) loadSession(JSON.parse(await f.text())); e.target.value = ''; });
// ---- upload and server processing (application mode)
async function postPhoto(file, corners) {
  const fd = new FormData(); fd.append('image', file);
  if (corners) fd.append('corners', JSON.stringify(corners));
  else if (el('closeup').checked) fd.append('mode', 'native');
  const r = await fetch('api/process', {method: 'POST', body: fd}); let j = null;
  try { j = await r.json(); } catch (_) { throw {error: `Server error (${r.status}).`}; }
  if (!r.ok) throw j; return j;
}
el('proc').addEventListener('click', async () => {
  const f = el('file').files[0]; if (!f) { el('upstatus').textContent = 'Choose a photo first.'; return; }
  el('proc').disabled = true; el('upstatus').textContent = `Processing ${f.name}: frame, candidates, features, classifier...`;
  try { const j = await postPhoto(f); const ix = await addPayload(j.payload, f); el('upstatus').textContent = `${f.name}: ${j.detections.length} detections in ${j.payload.timing_s ? Object.values(j.payload.timing_s).reduce((a, b) => a + b, 0).toFixed(0) : '?'} s.`; show(ix); }
  catch (err) {
    el('upstatus').innerHTML = `<span class="err">${err.error || 'Processing failed.'}</span>`;
    if (err.preview) startCorners(err.preview, f, -1);
  } finally { el('proc').disabled = false; }
});
function startCorners(preview, file, ix) {
  corner = {points: [], preview, file, ix};
  if (preview) { corner.previewImg = new Image(); corner.previewImg.src = preview.img; corner.previewImg.onload = () => draw(); }
  el('empty').hidden = true; cv.hidden = false;
  el('upstatus').innerHTML += '<br><span class="small">Click four corners around the area to count, in order: top-left, top-right, bottom-right, bottom-left. For a close-up without a green frame, use experimental close-up mode instead. Esc cancels.</span>';
  draw();
}
el('fmanual').addEventListener('click', () => {
  if (cur < 0) return; const f = files[cur];
  if (!f) { alert('Manual corners need the original upload; re-upload the photo.'); return; }
  if (ST[cur].actions.length && !confirm('Reprocessing with new corners discards the corrections on this photo. Continue?')) return;
  startCorners(null, f, cur);
});
async function submitCorners() {
  const c = corner; corner = null; el('upstatus').textContent = 'Reprocessing with your corners...';
  try { const j = await postPhoto(c.file, c.points);
    const ix = c.ix >= 0 ? await replacePayload(c.ix, j.payload) : await addPayload(j.payload, c.file);
    el('upstatus').textContent = `${c.file.name}: ${j.detections.length} detections with the frame you set.`; show(ix); }
  catch (err) { el('upstatus').innerHTML = `<span class="err">${err.error || 'Processing failed.'}</span>`; if (err.preview) startCorners(err.preview, c.file, c.ix); else if (cur >= 0) show(cur); }
}
document.addEventListener('keydown', e => { if (e.key === 'Escape' && corner) { corner = null; if (cur >= 0) show(cur); else { cv.hidden = true; el('empty').hidden = false; } } });
el('ref').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f || cur < 0) return;
  const d = DATA[cur], body = {reference_csv: await f.text(), points: {model: V.mset.map(i => [d.c.x[i], d.c.y[i]]), corrected: V.pts.map(t => [t[0], t[1]])},
    radius_px: 0.5 * 2 * (d.c.r.reduce((a, b) => a + b, 0) / d.c.r.length)};
  try { const r = await fetch('api/evaluate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}), j = await r.json();
    if (!r.ok) throw j;
    const row = (k, m) => `<tr><td>${k}</td><td>${m.n_detections}</td><td>${m.n_matched}</td><td>${m.n_missed_annotations}</td><td>${m.count_minus_annotations}</td><td>${m.annotated_recall.toFixed(2)}</td><td>${m.annotation_precision_lower_bound.toFixed(2)}</td></tr>`;
    ST[cur].accuracy = `<p class="small">Against ${j.n_reference} verified points, matched one-to-one within ${j.radius_px.toFixed(1)} px. Precision assumes your points are complete; if they are not, it is a lower bound.</p>` +
      `<table class="m"><tr><th></th><th>found</th><th>matched</th><th>missed</th><th>count diff</th><th>recall</th><th>precision</th></tr>${row('Model', j.model)}${row('Corrected', j.corrected)}</table>`;
    panel();
  } catch (err) { el('acc').innerHTML = `<p class="err">${err.error || 'Comparison failed.'}</p>`; }
  e.target.value = '';
});
// ---- boot
function show(i) {
  if (i < 0) return; corner = null; cur = i; [...el('tabs').children].forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
  el('empty').hidden = true; cv.hidden = false;
  const c = DATA[i].crop; cv.width = c.w; cv.height = c.h;
  el('rows').value = rows; el('cols').value = cols; el('annlab').hidden = !DATA[i].ann.length;
  selected.clear(); focus = null; el('inset').hidden = el('qbtns').hidden = true; el('calib').hidden = true; el('influence').innerHTML = ''; refresh();
}
el('upsec').hidden = !APP;
if (APP) el('sub').textContent = 'Upload a quadrat photo; the trained model runs on the server. Then confirm, remove or add detections. Counts update as you go.';
const boot = DATA.splice(0, DATA.length);
if (!boot.length) { cv.hidden = true; el('empty').hidden = false; }
Promise.all(boot.map(d => addPayload(d))).then(() => { if (DATA.length) show(0); });
window.__review = {sessionObj, act, undo, loadSession, state: () => ST[cur], V: () => V, data: () => DATA[cur], applyCalibration};
</script>
</body>
</html>
"""
