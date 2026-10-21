# DALI Barnacles: Phase 1, Annotation Extraction

Project overview and run instructions: `README.md`. Phase 2: `PHASE2.md`. Phases 2.1 to 5: `TECHNICAL_REPORT.md`.

Turns the blue contour masks into one record per annotated barnacle: a center `(x, y)`, a polygon, shape features, and review flags. It also audits the inputs and reports where the annotations are, and are not, complete.

No model, grid counting, or UI here.

## Run

```
pip install opencv-python numpy scipy scikit-image matplotlib
python run_phase1.py --data /path/to/uploads --out phase1_output
```

The run is deterministic, takes about 15 s per image, and has no network or GPU dependencies. It expects `imgN.png` + `maskN.png` pairs. If `masked_imgN.png` exists it is audited, never used.

## Code layout

| File | Role |
|---|---|
| `barnacles/io.py` | Load photos and masks, validate binary alpha and shared pixel grid, locate overlay crops (audit only) |
| `barnacles/extract.py` | Mask cleaning, interior detection, midline split, per-annotation features, classification |
| `barnacles/diagnostics.py` | Frame mask, annotated region, spacing, void detection, summary |
| `barnacles/export.py` | CSV, full JSON, COCO |
| `barnacles/visualize.py` | Center overlays, review figure, galleries, spot checks, stats plot |
| `run_phase1.py` | Orchestration and CLI |

## What the inputs actually contain

**Shared grid.** All four raw/mask files are 3264 x 2448 and share one pixel grid. Coordinates are raw-photo pixels: x is the column, y is the row, origin at the top-left pixel center.

**Masks are hard-edged RGBA.** Alpha is exactly 0 (white background) or 255 (stroke), with no anti-aliasing. `alpha > 0` is the stroke. The loader raises an error if a mask ever arrives anti-aliased.

**mask1 was drawn in two blues.**

| Stroke colour | Annotations | Vertical extent |
|---|---|---|
| RGB (39,57,191) | 1066 | y 470 to 1449 |
| RGB (59,76,160) | 665 | y 1264 to 1911 |

This is most likely two annotation sessions. The colour is kept per record as `stroke_color_index`. mask2 uses a single blue.

**Pen width varies.**
- mask1 is mostly 2 to 3 px. A thick-pen patch of about 6 px covers 271 annotations in the upper left.
- mask2 is about 6.3 px throughout, with ragged edges.

**The contours trace the operculum, not the whole shell.** They follow the inner plates, not the outer wall plates. Centers are therefore barnacle centers, but polygon area measures the opening, not the shell footprint.

**The overlays are exact crops, but their contours are shifted.**
- `masked_img1` is a pixel-exact crop of `img1` at (1124, 470); `masked_img2` is a pixel-exact crop of `img2` at (916, 194). Every unpainted pixel is identical to the photo.
- The painted contours are offset from the masks by (dx 0, dy +1) for img1 and (dx +1, dy -1) for img2, with IoU 1.0 at that shift.
- So the overlays are internally inconsistent by 1 px and should not be used for coordinates. At 18 to 30 px barnacle diameters this is immaterial.

## Method

1. **Specks.** Stroke blobs under 15 px are removed. These were 29 in mask1 and 124 in mask2.
   - All but one are single pixels about 2 px off a stroke (brush-edge debris).
   - The exception is a 4 px mark in mask2 at (3210, 1952), about 676 px from any annotation and outside the quadrat. It is reported as an artifact.

2. **Pinholes.** Enclosed background regions under 10 px are filled: 409 in mask1, 988 in mask2. The hole-size histogram justifies the cut. It is bimodal: 1 to 2 px pinholes, almost nothing from 3 to 12 px, then real interiors.

3. **Slivers.** Holes with an inscribed radius under 1.5 px are filled back into the stroke: 14 in mask1, 1 in mask2. These are left where a stroke overlaps its own start and are not barnacle interiors.

4. **Midline split.** Every stroke pixel is given the label of its nearest background region (a barnacle interior, or the outside), using a Euclidean distance transform with indices.
   - Interior k's polygon is the interior plus the inner half of its ring, so it follows the drawn centerline.
   - Touching rings are split on their shared stroke.
   - The center is the polygon's centroid (image moments).

5. **Rings with no interior.** These use the convex hull as geometry.
   - If the stroke fills at least 60% of the hull, the record is `solid`: the pen was so thick the middle closed up.
   - Otherwise it is `open`: the ring has a gap.
   - A `fragment` class exists for small open strokes; none occurred.

6. **Flags.** These mark records for review; they do not remove them. None occurred unless counts are shown in Results.
   - `size_outlier_large`: area above Q3 + 3 IQR.
   - `size_outlier_small`: area below 0.15 x median.
   - `near_duplicate`: centers closer than 0.35 x median diameter.
   - `encloses_other_stroke`: a nested ring.
   - `concave_merged_region`: a merged region with low solidity, a possible interstitial gap.
   - `touches_frame`: the polygon touches the quadrat wire, so the barnacle may be partial.

Parameters live in `ExtractionParams` and are written into `annotations.json`.

## Outputs

| File | Contents |
|---|---|
| `imgN_centers.csv` | One row per annotation: id, x, y, kind, flags, area, equivalent diameter, ellipse axes, solidity, stroke width, stroke colour, component id, bbox |
| `annotations.json` | Everything above plus polygons, parameters, removed specks, voids, overlay audit |
| `annotations_coco.json` | COCO instance segmentation (one category). Each image carries `roi_xyxy`. |
| `imgN_centers.png` | Full-resolution photo with 4 px magenta dots |
| `imgN_review.png` | Annotated region with polygons by kind, flagged boxes, voids (yellow), and the evaluated-region outline (white) |
| `review_galleries/` | Photo vs. stroke vs. polygon tiles for merged, solid/open, flagged, and frame-touching records; largest voids; a seeded sample of small voids; seeded spot-check patches |
| `stats.png` | Distributions of outline size, stroke width, and center spacing |
| `summary.json` | Per-image summary numbers |

## Results

| | img1 | img2 |
|---|---|---|
| **Annotated barnacles** | **1731** | **397** |
| closed / merged / solid / open | 1654 / 26 / 32 / 19 | 395 / 0 / 1 / 1 |
| Records excluded | 0 | 0 |
| Size flags | 5 small, 1 large | 1 large |
| Touching the frame | 38 | 9 |
| Median equivalent diameter (p5 to p95) | 17.9 px (9.9 to 28.2) | 30.2 px (19.5 to 43.5) |
| Median center spacing | 25.0 px | 48.0 px |

**Every special case was checked by eye in the galleries.**
- All 13 merged components are genuine pairs of touching rings, split correctly.
- The solid blobs are small barnacles in the thick-pen zone.
- The open rings are near-complete rings with small gaps.
- The small outliers are tiny barnacles.
- The two large outliers are drawn around genuinely large barnacles.

**Centers sit on the opercula in the spot-check patches.**

## Annotated barnacles are not ground-truth barnacles

The counts above are the number of drawn outlines. They are lower bounds on barnacles, for four reasons.

**1. Spatial scope.** Annotations exist only inside the central quadrat opening.
- img1: x 1125 to 2550, y 462 to 1923.
- img2: x 948 to 2579, y 232 to 1860.

The outer grid cells and everything beyond the frame are full of barnacles with zero annotations. That is out of scope rather than missed, but any training must treat it as **unlabelled, not background**. The COCO file carries `roi_xyxy` for this reason, but standard COCO tooling ignores it.

**2. Labelling policy.** Empty shells (dead barnacles with an open hole, no operculum) are consistently unannotated in both images. Barnacles seen side-on, with no plates visible, are mostly unannotated too. Whether these count as barnacles is a definitional decision for the challenge. It is not decided here.

**3. Misses inside scope.** A void is a spot farther than 0.9 x median spacing from every annotation center.
- **Why 0.9:** in tight packing no point is more than about 0.58 spacings from a center. A single skipped barnacle leaves a spot about 1.0 spacing from its neighbours.
- **Where it is evaluated:** inside the convex hull of centers, minus the frame and a half-spacing margin around it.

| | img1 | img2 |
|---|---|---|
| Voids | 102 | 20 |
| of which frame-adjacent | 17 | 8 |
| Share of evaluated region | 15% | 39% |

By-eye review of the galleries:
- **img1:**
  - The large interior voids are bare rock, old shell bases, empty shells, and the seaweed crack.
  - A random 16 of the small voids are mostly dark interstices or side-on shells. A few contain a plausible unannotated barnacle.
  - The spot-check patches show no clear misses.
- **img2:**
  - Several small voids (V5, V7, V8, V10, likely V11) contain live barnacles with visible plates and no contour. Spot-check patches show roughly 1 to 3 such misses per patch, alongside several empty shells.
  - img2 annotation is noticeably less complete than img1. This also biases img2's spacing statistic upward, which makes the void test less sensitive there, exactly where it matters most.
  - The pale, algae-rimmed shells inside the big bare-rock void are annotated. That was checked, because the yellow shading makes them look unannotated at low zoom.
- **Both images:** barnacles half under the wire are often unannotated, which is why frame-adjacent voids are reported separately.

**4. No recall number yet.** These are by-eye impressions on a handful of patches, not a measurement. A small, exhaustively hand-labelled patch set is the right way to get one.

## Limitations

**Extraction**
- A gapped ring touching a closed ring would be absorbed into its neighbour's component and silently lost. No case was seen, and there is no detector for it.
- A hole enclosed by three or more touching rings would be counted as a barnacle. No component here has more than two interiors, so none occurred.
- Polygon size depends on pen width, since the midline sits half a stroke outside the interior. Solid hulls include the full stroke, so their 33 areas run high. Interior inscribed radius is a pen-independent alternative for size work.
- Thresholds are justified by these two masks' histograms. Re-check them on any new mask; the loader enforces the binary-alpha assumption but nothing else.

**Diagnostics and geometry**
- Void detection uses spacing derived from the annotations themselves, so incomplete annotation weakens it.
- Bare rock and missed barnacles are indistinguishable without looking at the image; void review here is human judgement.
- The frame mask is a fixed teal hue band. It is clean on both photos, but it is not a general frame detector.
- The center is the operculum centroid. For tilted or crowded barnacles this can sit a few px off the shell centroid.
