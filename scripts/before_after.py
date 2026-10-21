"""Before/after figures and the experiment table (from experiments/*.json and
the experiment cache). Before = C0 (the original Phase 3 model), after = C7.

Figures show both models at the SAME count, the number of annotations (an
oracle operating point, used only so the pictures compare which barnacles
each model ranks highest, not how well each guessed the count). The numbers
at a realistic operating point (80 simulated labels, seed 0) are written to
experiments/before_after.json alongside."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiments as rx  # noqa: E402
from barnacles.ml import DuplicateIndex, DUP_FRAC, label_candidates, select_count  # noqa: E402
from barnacles.evaluation import evaluate  # noqa: E402
from barnacles import calibration as cal  # noqa: E402
from barnacles import phase3_viz as pv  # noqa: E402
from barnacles.grid import GridSpec, assign  # noqa: E402

OUT = ROOT / "experiments" / "figures"


def detections(d, prefix, gen, n=None):
    c = d["sets"][gen]
    xy = rx.to_img(d, c)
    y, _ = label_candidates(xy, d["gt"], d["R"])
    s = np.load(rx.CACHE / f"scores_{prefix}_{d['name']}.npy")
    dup = DuplicateIndex(xy, c["radius"] * d["s"], DUP_FRAC)
    idx, w, strat = cal.stratified_sample(s, 80, np.random.default_rng(1000))
    sizes = np.bincount(strat, minlength=len(cal.STRATA_Q) - 1)
    if n is None:
        n = cal.isotonic_count((y == 1).astype(int)[idx], strat[idx], sizes)
    det, _ = select_count(dup, s, int(round(n)))
    m, r = evaluate(xy[det], d["gt"], d["R"])
    return xy[det], m, r


ALL = {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in rx.PHOTOS:
        d = rx.load(name)
        d["name"] = name
        img = cv2.imread(str(rx.DATA / name), cv2.IMREAD_COLOR)
        summary = {}
        for tag, pre, gen in (("before_C0", "C0", "ringlog"), ("after_C7", "C7", "hough")):
            summary[tag] = {"calibrated_seed0": detections(d, pre, gen)[1], "at_annotated_count": None}
        b_xy, b_m, b_r = detections(d, "C0", "ringlog", n=len(d["gt"]))
        a_xy, a_m, a_r = detections(d, "C7", "hough", n=len(d["gt"]))
        summary["before_C0"]["at_annotated_count"], summary["after_C7"]["at_annotated_count"] = b_m, a_m
        ALL[name] = summary
        for tag, xy, m, (pairs, ud, mg) in (("before_C0", b_xy, b_m, b_r), ("after_C7", a_xy, a_m, a_r)):
            ttl = f"{name}, {tag.replace('_', ' ')} at the annotated count: {m['n_matched']} of {m['n_annotations']} annotations matched, median offset {m['match_dist_median_px']:.1f} px"
            ov = pv.outcome_overlay(img, d["frame"], xy, d["gt"], pairs, ud, mg, title=ttl)
            cv2.imwrite(str(OUT / f"{Path(name).stem}_{tag}_outcomes.png"), np.concatenate([ov, pv.legend_strip(ov.shape[1])], 0))
        gt_uv = d["frame"].to_norm(d["gt"])
        rr, cc, _ = assign(gt_uv, GridSpec(10, 10))
        cnt = np.bincount(rr * 10 + cc, minlength=100)
        dense, sparse = int(np.argmax(cnt)), int(np.argmin(np.where(cnt > 0, cnt, 999)))
        cen = [((k % 10 + 0.5) / 10, (k // 10 + 0.5) / 10) for k in (dense, sparse)]
        side = np.mean([np.linalg.norm(d["frame"].corners[1] - d["frame"].corners[0]),
                        np.linalg.norm(d["frame"].corners[3] - d["frame"].corners[0])])
        z = pv.zoom_pair(img, d["frame"], [(b_xy, d["gt"], *b_r), (a_xy, d["gt"], *a_r)], cen, int(0.07 * side),
                         ["Before: C0 ring+LoG, base features", "After: C7 Hough, + wide HOG and gradients"])
        cv2.imwrite(str(OUT / f"{Path(name).stem}_before_after_zoom.png"), z)
        print(name, "at annotated count: before matched", b_m["n_matched"], "med", round(b_m["match_dist_median_px"], 1),
              "| after matched", a_m["n_matched"], "med", round(a_m["match_dist_median_px"], 1),
              "| seed0 calibrated F1 before", round(summary["before_C0"]["calibrated_seed0"]["f1_vs_annotations"], 3),
              "after", round(summary["after_C7"]["calibrated_seed0"]["f1_vs_annotations"], 3))

    json.dump(ALL, open(ROOT / "experiments" / "before_after.json", "w"), indent=1, default=float)
    # experiment table (markdown)
    C = json.load(open(ROOT / "experiments/classifiers.json"))
    lines = ["| Variant | AP img2 | AP img1 | Mean AP | Calibrated F1 img2 | Calibrated F1 img1 | Count img2 (397) | Count img1 (1,731) |",
             "|---|---|---|---|---|---|---|---|"]
    for k, v in C.items():
        a, b = v["test img2.png"], v["test img1.png"]
        f = lambda x: f"{x['calibrated_k80']['f1'][0]:.3f} ± {x['calibrated_k80']['f1'][1]:.3f}"
        n = lambda x: f"{x['calibrated_k80']['n_det'][0]:.0f} ± {x['calibrated_k80']['n_det'][1]:.0f}"
        lines.append(f"| {k} | {a['ap']:.3f} | {b['ap']:.3f} | {(a['ap'] + b['ap']) / 2:.3f} | {f(a)} | {f(b)} | {n(a)} | {n(b)} |")
    (ROOT / "experiments" / "classifier_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
