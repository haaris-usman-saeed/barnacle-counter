# DALI Barnacles: Phase 2, Coordinate Mapping and Grid Aggregation

Phase 2 maps every Phase 1 annotation center into normalized coordinates inside the inner quadrat frame, then aggregates the centers into an adjustable grid. It reads `phase1_output/annotations.json` and does not re-run or modify extraction; all Phase 1 modules are byte-identical to the approved version.

No model, no final UI.

## Run

```
python run_phase1.py --out phase1_output          # unchanged, only if not already run
python run_phase2.py --phase1 phase1_output/annotations.json --out phase2_output \
    --rows 10 --cols 10 [--frame-config frame_config.json]
```

The run is deterministic and takes under a minute for both images.

## New modules

| File | Role |
|---|---|
| `barnacles/frame.py` | Inner frame detection, manual corner override, homography, resolution rescaling |
| `barnacles/grid.py` | Grid spec, the single assignment rule, cell records, verification, boundary self-test |
| `barnacles/grid_viz.py` | Grid overlay, rectified view, frame check, summary figure |
| `barnacles/grid_viewer.py` | Self-contained HTML review viewer with selectable cells and adjustable grid |
| `run_phase2.py` | Orchestration and CLI |

## Frame

### Definition

The frame is the opening bounded by the **inner edges** of the four wire bars around the central quadrat cell.

### Detection

1. **Wire mask.** Reuse the Phase 1 teal wire mask, then seal small gaps in it with a 31 px morphological close.
2. **Opening.** Take the non-wire region containing the median annotation center, and fill its holes.
3. **Sides.** Trace the opening's outline, split it into four sides at its extreme points, and drop 8% at each end so the corner fillets don't bias the fit.
4. **Line fits.** Fit each side with total least squares. The trim threshold shrinks from 40 px to 2.5 px, centred on the mode of the residuals, so weld nubs and bolts are rejected rather than averaged in.
5. **Corners.** The four corners are the intersections of adjacent fitted lines.

### Manual override

`frame_config_detected.json` holds the detected corners. To override:

1. Copy it and edit the corners.
2. Set `"source": "manual"`.
3. Pass the file back with `--frame-config`.

Manual corners are rescaled automatically if the image size differs from the `image_size` stored with them. The report then gives `manual_minus_auto_px` per corner.

I tested this path by pulling every corner 20 px inward. The counts and checks behaved correctly:

| | img1 | img2 |
|---|---|---|
| Centers inside | 1667 | 387 |
| Centers outside | 64 | 10 |
| Sum over cells | 1667 | 387 |

### Detected frames

| | img1 | img2 |
|---|---|---|
| TL, TR, BR, BL (px) | (1137.1, 513.1), (2537.1, 465.5), (2545.4, 1933.9), (1124.9, 1892.8) | (956.3, 227.4), (2581.6, 231.8), (2550.0, 1861.1), (964.0, 1858.4) |
| Corner angles | 92.5, 88.4, 88.0, 91.2 deg | 89.6, 89.0, 91.0, 90.4 deg |
| Side lengths T, R, B, L | 1401, 1468, 1421, 1380 px | 1625, 1630, 1586, 1631 px |
| Edge-fit RMS per side | 0.8 to 1.2 px | 0.9 to 1.3 px |
| Side bow (curvature) | at most 1.6 px | at most 1.4 px |
| Corner shift when detected on a half-size copy | at most 2.1 px | at most 1.7 px |
| Frame area covered by wire protrusions | 0.47% | 0.55% |

`imgN_frame_check.png` shows all four corners and side midpoints. The detected opening edge is in yellow, the fitted frame in red, and each side's residual profile is plotted below the crops. Both frames line up with the inner wire edge to within a couple of pixels everywhere I checked.

## Coordinates

**Mapping.** A homography H maps the corners TL, TR, BR, BL to (0,0), (1,0), (1,1), (0,1). u runs from left to right along the frame and v from top to bottom.

**Resolution independence.** (u, v) depends only on where a point sits relative to the corners. Scaling the image and the corners together leaves every (u, v) unchanged:
- At 0.5x and 2x, the change in (u, v) is exactly 0.
- At 0.37x, it is at most 8e-8, and every cell assignment is identical.

**Why a homography rather than a bounding box.** img1's frame is rotated about 2 degrees and keystoned (the right side is 88 px longer than the left). Normalizing by the corners' bounding box instead would move centers by up to 44 px. It would put **293 of 1731** img1 centers and **32 of 397** img2 centers in a different 10 x 10 cell.

## Grid and assignment

There is one assignment rule. It is implemented in `grid.assign` and mirrored line for line in the viewer.

```
inside  iff  0 <= u <= 1 and 0 <= v <= 1
col = min(floor(u * cols), cols - 1)
row = min(floor(v * rows), rows - 1)
```

**Cell intervals.** Each cell covers [k/n, (k+1)/n), except the last row and column, which also include their far edge. Every inside point therefore lands in exactly one cell, including points exactly on a grid line or on the frame edge.

**Cell shape.** Cells are uniform in (u, v), so each covers equal area of the frame plane. In the photo they appear as projected quadrilaterals.

**Adjustable size.** Any rows x cols from 1 x 1 upward works.

### Cell record

Each cell in `imgN_grid_RxC.json` has these fields:

| Field | Contents |
|---|---|
| `cell_id`, `row`, `col` | Cell identity, e.g. `r03c07` |
| `u_range`, `v_range` | Normalized bounds |
| `polygon_px` | The cell's four corners in the raw image |
| `area_px` | Pixel count of the cell |
| `annotated_count` | Number of annotation centers in the cell |
| `barnacle_ids` | IDs of those annotations |
| `occluded_fraction` | Share of the cell covered by wire protrusions |
| `void_fraction` | Share of the cell farther than 0.9 spacings from any annotation, using Phase 1's void rule applied to the whole frame |
| `verified_count` | `null`, reserved for human review |

The same table is also written as `imgN_cells_RxC.csv`. Per-center (u, v), signed distance to the frame edge, and cell are in `imgN_centers_normalized.csv`.

## Verification

Every check passes on both images. Any failure raises an error rather than writing output.

| Check | img1 | img2 |
|---|---|---|
| Extracted centers | 1731 | 397 |
| Inside the frame / outside | 1731 / 0 | 397 / 0 |
| Sum of 10 x 10 cell counts | **1731** | **397** |
| Every inside ID appears in exactly one cell | yes | yes |
| Homography inside test agrees with an independent polygon test | yes, 0 disagreements | yes, 0 disagreements |
| Closest center to the frame edge | 2.4 px | 9.0 px |
| Corner round-trip error | 5e-13 px | 1e-13 px |
| Pixel partition: cell areas sum to the quad area | ratio 1.000007 | ratio 1.000000 |

Further checks:
- **Grid sweep.** Grids of 1x1, 2x3, 4x4, 5x7, 10x10, 16x16, 25x20, 33x29 and 60x60 all sum to the inside count.
- **Boundary self-test.** Synthetic points on every grid line, corner and frame edge each land in exactly one cell, and points 1e-12 outside the frame land in none. This holds for every grid in the sweep.
- **Viewer agreement.** The viewer recomputes the assignment in the browser. At the default grid, it reports that it matches the Python output for every center.

## Visualizations

| File | Contents |
|---|---|
| `grid_viewer.html` | Both photos, frame outline, subtle grid lines, centers. Click a cell to see its ID, normalized bounds, annotated count and barnacle IDs; shift-click to combine cells; arrow keys move the selection. Rows and columns are editable live. Frame totals and the sum check update with every change. Counts appear only in the side panel, never on the photo. |
| `imgN_grid_overlay.png` | Full-resolution photo with subtle grid and centers. Row and column indices sit in a white margin, off the photo. |
| `imgN_grid_summary.png` | Photo with grid, annotated count per cell, void fraction per cell |
| `imgN_rectified.png` | The frame warped to a square with the grid and the mapped centers. Wire edges line up with the square, and dots sit on opercula, which confirms the mapping. |
| `imgN_frame_check.png` | Corner and mid-side fit crops plus residual profiles |

## Problems found: frame boundaries and alignment

**1. Weld nubs and bolts intrude into the frame.**
- On every side, at about 0.3 and 0.67 of the way along, the outer grid bars join the frame with a nub 26 to 41 px deep.
- img2 also has a bolt 58 px deep on the left side.
- Together they hide 0.47% (img1) and 0.55% (img2) of the frame. That is 23 and 24 cells at 10 x 10, up to 8.6% of a single cell.
- Barnacles under them cannot be counted by anyone; `occluded_fraction` records this per cell.

**2. The corners are rounded fillets.** The fitted corner is the intersection of the two straight edges, so it sits a few pixels outside the visible opening. This is the correct reference for a square frame, but the corner cells contain a sliver of wire.

**3. Edge sensitivity.**
- img1 has 7 centers within 5 px of the frame edge, 33 within 10 px and 64 within 20 px. For img2 the figures are 0, 2 and 10.
- Detection repeats to about 2 px, so boundary error can move at most a handful of img1 centers, and none currently sit within 2 px of the edge.
- A 20 px corner error would drop 64 img1 annotations (3.7%). If corners are entered by hand, they need to be placed carefully.

**4. Apparent edge versus physical edge.** The bars are raised cylinders, and the frame is defined by their inner edge as seen in the photo. Near the image edges the camera sees part of the bar's side, so the apparent edge can sit a few pixels from the physical inner edge. Barnacles partly under a bar are in the photo but ambiguous. These are the same records Phase 1 flagged `touches_frame` (38 in img1, 9 in img2).

**5. Lens distortion is negligible here.** Side bow is at most 1.6 px over about 1400 px, and edge residuals are about 1 px RMS. That is too small to justify modelling distortion for these two photos. A wider lens or an off-centre frame could change this, and `frame_check` will show it.

**6. Phase 1 annotation coverage does not reach the whole frame.**
- 9.5% of img2's frame lies outside the convex hull of its annotations (bottom-left corner), and 2.4% of img1's.
- 16 of img2's 100 cells have zero annotations. Some are bare rock; some are simply unannotated.
- `void_fraction` marks these cells, but it cannot tell bare rock from missed barnacles.

## Annotated counts are not barnacle counts

`annotated_count` is the number of Phase 1 outlines whose center falls in the cell. Everything Phase 1 found about incompleteness still applies, so every cell count is a lower bound:
- Annotation covers only the central opening.
- Empty and side-on shells are unannotated.
- img2 has misses of live barnacles.

The schema keeps this distinction explicit:
- `annotated_count` holds what the masks contain.
- `verified_count` is reserved, and stays `null` until human review fills it.
- Field names, the JSON `count_semantics` string, figure titles and the viewer all say "annotated outlines", never "barnacles".

## Limitations

- **Planar assumption.** The homography treats the frame as a flat plane. Equal cells are equal areas of that plane, not of the uneven rock surface beneath it.
- **Narrow frame detector.** Detection depends on the teal hue band and on the seed point (the median annotation center) lying inside the opening. A differently coloured frame, or an image with no annotations yet, needs manual corners or a seed.
- **Void fraction inherits Phase 1's caveats.** It uses spacing derived from the annotations themselves, so sparse labelling weakens it.
