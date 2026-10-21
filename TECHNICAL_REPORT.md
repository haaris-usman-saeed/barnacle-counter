# Technical report: ML-assisted barnacle counting

# Final submission pass: verification record

What was actually run in the development environment (1 CPU, Python 3.12, no Docker, no gunicorn):

| Check | Result |
|---|---|
| 1. Documented training workflow (`python train_model.py`) | Rerun from scratch. The new artifact gives identical predictions to the shipped one (5,000 random feature vectors), the same default threshold (0.310) and the same blocked-CV F1 (0.880, 0.664); training took 294 s. Fingerprints were regenerated, identical. |
| 1. Documented evaluation workflow | `scripts/run_experiments.py generators` was rerun from the relocated cache, and every candidate count and recall ceiling was reproduced exactly. The classifier, refinement and calibration stages were not rerun in this pass (about 15 min); their results are unchanged from Round 2. |
| 2. Start the application | `python app.py` (Flask development server). `/api/health` reported the model loaded. Gunicorn could not be tested: not installed. |
| 3 and 4. Upload a development photo; real inference | `img1.png` uploaded through the browser (Playwright) and processed by the model in 42.9 s, giving 1715 selected detections. The training-photo warning appeared at level "exact". Peak resident memory of the server process: about 1.1 GB. |
| 5. Frame and coordinates | Detected corners matched the Phase 2 frame configuration. Every detection's (u, v) matched the homography applied to its (x, y) to within 4.3e-06. Image size was preserved (3264 x 2448). |
| 6 and 7. Accept, remove, accept suggestion, reject suggestion, add, undo | Every step updated the corrected count and breakdown as expected (accept detection: count unchanged; remove: -1; accept suggestion: +1; reject suggestion: unchanged; add: +1; undo: reverts). The dashboard identity held after every step. One add click landed within 4 px of an existing marker and was correctly ignored. |
| 8. Export and re-import | Session JSON and points CSV were downloaded. The session was loaded into a fresh page after re-processing the photo; corrected count, breakdown, detections and per-cell counts were all identical. |
| 9. Output formats | Session keys: actions, exported_at, frame, grid, image, image_size, model, operating_point, result, schema; result keys: breakdown, corrected_count, detections, initial_model_count, model_count, per_cell_corrected, per_cell_model, threshold. The CSV header is `id,x,y,u,v,cell_id,source,review_status,score`, with one row per counted barnacle (1715 rows). |
| 10. Automated tests | 70 tests. Sandbox: 70 passed. Package without data: 58 passed, 12 skipped. Package with data: 69 passed, 1 skipped (needs the excluded Phase 3 cache). See `TEST_RESULTS.md`. |
| 11. Deployment configuration | Not tested: Docker and gunicorn are unavailable. See `DEPLOYMENT.md` for what remains unverified. |

## Training-photo recognition

There are three levels (`barnacles/fingerprint.py`): exact (SHA-256 of the decoded pixels), likely, and possible. The thresholds were set from measured distances:

| Comparison | pHash distance | Thumbnail correlation | Level reached |
|---|---|---|---|
| Resized to 0.5x and 0.2x, JPEG at q90 and q40, darkened to 0.8x | 0 to 2 | 1.00 | likely |
| 5% border crop, or 3 degree rotation (img2) | 6 to 10 | 0.76 to 0.83 | possible (img1's rotation: none) |
| 20% or 50% crop, mirror image | 16 to 30 | below 0.75 | none |
| img1 against img2 | 28 | 0.52 | none |
| Random noise | 32 or more | about 0 | none |

The bar for "likely" is that all four conditions hold: pHash distance at most 8, dHash distance at most 10, correlation at least 0.97, and aspect-ratio difference within 2%. Only "exact" is presented as definitive.

## Dashboard semantics

The counting semantics are unchanged. The displayed numbers are now defined separately and checked against each other:
- **Selected model detections:** candidates at or above the threshold, after duplicate suppression.
- **Removed by you:** selected detections you removed.
- **Accepted suggestions:** candidates below the threshold that you accepted.
- **Added by you:** barnacles you clicked in.
- **Reviewed items:** confirmed plus removed plus accepted suggestions plus rejected suggestions.
- **Still to review:** flagged detections and undecided suggestions, shown separately.
- **Corrected count:** selected minus removed plus accepted suggestions plus added.

"Suggestions" are extra candidates offered for checking, and are not counted unless accepted. This is why "still to review" could previously exceed the number of detections. The same breakdown appears in the exported session and in the Python replay, which asserts the identity.

---


# Round 2: detection improvements, upload application, calibration safeguards

This section supersedes the earlier sections where they disagree. Everything below is produced by `scripts/run_experiments.py`, `scripts/before_after.py` and `train_model.py`, and stored in `experiments/`.

## Protocol

- **Cross-photo throughout.** Each photo is scored by a model trained only on the other photo.
- **Metrics.**
  - AP is threshold-free.
  - "Calibrated" results use 80 simulated reviewer labels: exact budget, a count-matched threshold, and 10 random label draws. The estimator is Platt-sum in the classifier table and isotonic where stated.
  - Recall is annotated recall; precision is a lower bound.
- **Selection rule.** Declared before running: highest mean cross-photo AP, with differences under 0.01 going to the simpler variant.
- **Disclosure.** Variants C7 to C9 were added after C0 to C4 had run, once the generator stage showed Hough proposals are better centred. With two photos, any selection among variants is tuning on evaluation data.

## A. Candidate generation, measured separately from classification

Each cell gives the annotated recall ceiling at 0.5 / 0.25 x the median opening diameter, then the number of candidates, the median offset of matched candidates, and the ceiling in the densest third.

| Generator | img1 | img2 |
|---|---|---|
| Ring + LoG (previous) | 96.0% / 53.6%, 13,638, 4.2 px, dense 95.9% | 100.0% / 82.4%, 11,481, 4.8 px, dense 100.0% |
| Hough circles (frame-relative radius range) | 98.7% / 87.7%, 10,197, 2.9 px, dense 99.0% | 98.7% / 92.7%, 6,865, 3.5 px, dense 99.3% |
| Union | 99.7% / 70.2%, 17,247, 3.6 px, dense 99.9% | 100.0% / 93.2%, 14,853, 4.1 px, dense 100.0% |

The previous report's hint holds up. Hough proposals are far better centred: the recall ceiling at the tight radius rises from 54% to 88% on img1. They are also fewer, and they lose little at the standard radius (98.7% on both photos). The union has the highest standard-radius ceiling, but at the cost of more candidates.

Generation takes about 20 s per photo for any variant, dominated by the shared score maps.

## B. Classifiers

| Variant | AP img2 | AP img1 | Mean AP | Calibrated F1 img2 | Calibrated F1 img1 | Count img2 (397) | Count img1 (1,731) |
|---|---|---|---|---|---|---|---|
| C0 current: ring+LoG, base features, HGB | 0.310 | 0.553 | 0.432 | 0.335 ± 0.055 | 0.547 ± 0.014 | 268 ± 106 | 1516 ± 402 |
| C1 + per-photo feature standardization | 0.231 | 0.526 | 0.379 | 0.283 ± 0.028 | 0.519 ± 0.031 | 395 ± 241 | 1725 ± 730 |
| C2 + wide HOG and radial gradients | 0.331 | 0.574 | 0.453 | 0.387 ± 0.019 | 0.574 ± 0.015 | 507 ± 249 | 1852 ± 490 |
| C3 C2 + per-photo standardization | 0.277 | 0.566 | 0.422 | 0.346 ± 0.026 | 0.559 ± 0.021 | 444 ± 186 | 1441 ± 227 |
| C4 C1 features, random forest | 0.254 | 0.532 | 0.393 | 0.305 ± 0.040 | 0.540 ± 0.022 | 532 ± 326 | 1546 ± 361 |
| C5 C1 + contrast-jitter augmentation | 0.251 | 0.626 | 0.439 | 0.309 ± 0.019 | 0.593 ± 0.026 | 328 ± 152 | 1730 ± 624 |
| C6 C1 on union (ring+LoG + Hough) candidates | 0.295 | 0.638 | 0.466 | 0.316 ± 0.062 | 0.608 ± 0.020 | 431 ± 369 | 1473 ± 278 |
| C7 C2 features on Hough candidates | 0.404 | 0.696 | 0.550 | 0.358 ± 0.096 | 0.670 ± 0.034 | 495 ± 477 | 1907 ± 584 |
| C8 C2 features on union candidates | 0.388 | 0.656 | 0.522 | 0.397 ± 0.042 | 0.628 ± 0.031 | 288 ± 110 | 2024 ± 678 |
| C9 C2 + contrast-jitter augmentation | 0.380 | 0.611 | 0.496 | 0.420 ± 0.019 | 0.596 ± 0.018 | 545 ± 260 | 1540 ± 382 |

The bracketed counts in the last two column headers are the number of annotations.

- **C7 is selected** (mean AP 0.550). It is the deployed configuration.
- **What did not help:**
  - per-photo feature standardization (C1, C3);
  - a random forest (C4);
  - augmentation on top of standardization (C5).
- **Wider context helped modestly.** The wide HOG patch and radial gradient features (C2) improved on C0.
- **Augmentation on C2 (C9)** gave the steadiest calibrated F1 on img2, but lower AP.
- **C7's weak spot.** Its calibrated F1 on img2 has the largest spread of all variants (± 0.096).

At the annotated count (an oracle count, used only for figures; `experiments/before_after.json`):

| Photo | C0 matched | C7 matched | Median offset C0 / C7 |
|---|---|---|---|
| img1 (1,731 annotations) | 983 | 1,210 | 4.5 / 4.6 px |
| img2 (397 annotations) | 154 | 168 | 6.5 / 7.0 px |

Figures: `experiments/figures/*_before_after_zoom.png` and `*_outcomes.png`.

## C. Center refinement (C7, seed-0 calibrated detections)

| Test photo | Variant | Detections after re-suppression | F1 at 0.25 / 0.5 diameter | Median / p90 offset |
|---|---|---|---|---|
| img1.png | R0 none | 1739 of 1739 | 0.329 / 0.699 | 4.65 / 7.62 px |
| img1.png | R1 ring-map local max | 1739 of 1739 | 0.329 / 0.699 | 4.65 / 7.62 px |
| img1.png | R2 classifier local max | 1611 of 1739 | 0.363 / 0.699 | 4.33 / 7.59 px |
| img2.png | R0 none | 130 of 130 | 0.167 / 0.300 | 7.27 / 12.11 px |
| img2.png | R1 ring-map local max | 130 of 130 | 0.167 / 0.300 | 7.27 / 12.11 px |
| img2.png | R2 classifier local max | 130 of 130 | 0.167 / 0.304 | 7.02 / 12.64 px |

R1 (ring-map local maximum) only applies to ring + LoG candidates, so it is identical to R0 for Hough.

R2 moves each detection to the best-scoring of 8 offsets at ± 0.4 r:
- On img1 it slightly improves centring (median offset 4.65 to 4.33 px; F1 at the tight radius 0.33 to 0.36).
- But 128 detections then collapsed onto neighbours, and recall at the standard radius fell (0.700 to 0.675).
- On img2 it changed almost nothing.

**Not adopted**, because the brief explicitly warns against merging neighbours. R2 also costs about 14 s extra on img1.

## D. Calibration: sample budget, estimator stability, safeguard

**Bug fixed.** Requesting k labels used to sample (k // 9) x 9, so 80 became 72. Allocation is now exactly k, with the remainder given to the highest-score strata. This is tested for k in 9, 20, 40, 72, 80, 81 and 150, and for populations smaller than k.

**Why one label moved the count so much.** The count estimate is a stratum-weighted sum. The lowest-score stratum holds about half of all candidates but only about 9 labels, so one mistaken "opening" there adds roughly 5,000 / 9 openings.

**Safeguards tested** on C7, with 20 draws for accuracy and 10 draws for stability; each label in a draw was flipped in turn:
- **Isotonic estimate:** stratum positive rates are forced to be non-decreasing in score (pool adjacent violators).
- **Influence check:** any label whose flip moves the count by more than 10% is shown to the reviewer to confirm.

| Photo | Estimator | Count error (mean abs) | Calibrated F1 | Max change from one flipped label: mean / worst draw | Max change after confirming flagged labels |
|---|---|---|---|---|---|
| img1.png | Platt sum (old) | 27% | 0.666 ± 0.045 | 40% / 69% | 7.4% |
| img1.png | isotonic (new) | 25% | 0.658 ± 0.060 | 34% / 68% | 6.5% |
| img2.png | Platt sum (old) | 78% | 0.375 ± 0.075 | 295% / 1505% | 6.7% |
| img2.png | isotonic (new) | 53% | 0.364 ± 0.076 | 115% / 348% | 5.3% |

**Adopted: isotonic estimate plus influence confirmation** (both in `barnacles/calibration.py` and mirrored in the interface).
- The isotonic estimate cuts single-label sensitivity substantially on img2, and slightly on img1, while keeping F1.
- The confirmation step bounds the residual effect of one remaining mistake to about 5% to 7%. That figure assumes the reviewer catches the flagged label; I did not measure how many labels get flagged per draw.

Counts remain imprecise: with 80 labels the estimate is off by about 25% (img1) and 53% (img2) on average.

## E. Deployed model

`train_model.py` trains C7 on both photos together. Its default threshold (0.310) comes from 8-fold spatially blocked CV on the same two photos (F1 0.88 on img1 and 0.66 on img2, within-photo and therefore optimistic).

This model has no held-out evaluation of its own; the cross-photo C7 numbers above are the evidence. On a new photo the default threshold is uncalibrated, and the interface says so until the reviewer calibrates. Training takes about 4 minutes on one CPU.

## F. Upload application

**`barnacles/inference.py`** runs the same code path as training:
1. validation (magic-byte sniffing, 40 MB and 60 MP limits, at least 600 px on the short side);
2. frame detection on a copy capped at 3,300 px, with corners scaled back; or manual corners, validated;
3. rectification;
4. Hough candidates;
5. features, checked against the artifact's feature names;
6. classifier;
7. duplicate suppression;
8. an exact-k calibration sample;
9. review signals;
10. a payload in original photo pixels and (u, v).

No annotation or mask file is read.

**`app.py` (Flask)** returns:
- 400, 413 or 415 for unusable uploads;
- 422 with a preview when no frame is found, so the user can click the corners;
- 500 with the error message for pipeline failures, never fallback results.

Flask was used because FastAPI is not installed in this environment and there is no network access. `Procfile` and `Dockerfile` are included but were not tested here.

**Measured.** A full-resolution JPEG of img2 (3264 x 2448) processes in 26 to 27 s, giving 571 detections at the default threshold. This is in-sample, because the deployed model was trained on img2; re-encoding as JPEG means the pixel hash no longer matches, so the training-photo warning does not appear.

**Interface additions:**
- upload and manual-corner modes, frame confirmation, and undo;
- a dashboard (start, current and corrected counts; added, removed and accepted; reviewed and awaiting review), plus a per-cell minimap marking the cells with the most queued items;
- calibration with the influence check;
- comparison against a CSV of verified points (one-to-one matching, with the completeness assumption stated);
- "accuracy unknown" shown when no verified points exist.

Scores are labelled "score" until the reviewer calibrates, and "calibrated estimate" after.

**Browser test (Playwright).** Upload, frame confirmation, accept, remove, add, undo, calibration and export all worked. The exported session replays identically in Python.

## G. Tests

The full suite is listed in the README. New this round:
- end-to-end inference on a resized development photo, with no annotations read;
- coordinate reconstruction across resolution: corners within 4 px, (u, v) within 0.004;
- upload validation for each failure type, and graceful manual fallback;
- unique cell assignment at several grid sizes;
- an HTTP end-to-end upload, with manual corners reproducing the automatic result;
- exact calibration sample sizes and the isotonic safeguard;
- export and import round trip.

---

# Round 1 (original report)

Two photos, two incomplete annotation masks, one CPU, no deep-learning framework. Every number below comes from `phase3_output/phase3_report.json`, `phase4_output/phase4_report.json` or `phase2_output/frame_robustness.json`, produced by the scripts in this repository.

## Summary

- The frame, coordinate system and grid are solid. The frame is found from pixels alone, to within 2.6 px under nine image perturbations, and every grid count reconciles exactly.
- Classical detection does not work on these photos. The best hand-designed score reaches F1 0.2 to 0.4 against the annotations even when tuned on the same photo, and much less when transferred.
- The ML classifier, trained on one photo and applied to the other, ranks candidates 2.3x to 3.6x better than the classical score (average precision).
- **The weak point is choosing an operating point on a new photo.** A threshold learned on one photo lands in the wrong place on the other. Calibrating from 80 reviewer labels on the new photo gives F1 0.55 (img1) and 0.34 (img2) against annotations. The count from those 80 labels is uncertain by roughly 25% (img1) to 55% (img2), so review remains necessary.
- The review prototype's queue finds 1.2x to 1.8x more counting errors per item checked than random checking. That is useful, but modest.

This is a credible starting point, not a counting system that can be trusted without review.

## Data facts that shaped the design

- **What the annotations are.** They trace barnacle openings (opercula), not whole shells. img1 is almost completely annotated for live openings; img2 misses some live barnacles. Empty shells are unannotated in both, so the system learns that labelling policy.
- **Scale.** Median opening diameter is 18 px in img1 and 30 px in img2; median center spacing is 25 px and 48 px.
- **Appearance at annotated centers**, averaged over all annotations (`barnacles/candidate_detection.py`):
  - img1: a grey opening with a faint dark rim, inside a bright white shell.
  - img2: the opening's plates are the brightest spot, on a dark surround.

  So the two photos need different cues, which is why the candidate generator combines two responses.

## Phase 2.1: frame from pixels only

**Method.** The detector selects the largest closed wire cell that is nearly rectangular (rectangularity at least 0.9), roughly square (aspect ratio at most 1.3), and does not touch the image border. On both photos it is 8.7x (img1) and 9.6x (img2) larger than the next qualifying cell. Edge fitting and corner intersection are unchanged from Phase 2.

**Exposure fix.** The wire mask used for detection is gain-normalized before thresholding. Without this, a 0.6x darker img1 failed with a 17 px corner error. Phase 1's own mask is untouched.

**Robustness** (`scripts/frame_robustness.py`). Worst corner error over darker, brighter, blur, JPEG q50, half size, rotation by 4 and -7 degrees, and a crop: 1.7 px for img1 and 2.6 px for img2.

**Regression.** Corners are identical to Phase 2 to 0.00 px, and every 10 x 10 cell has the same members.

**Manual corners** bypass detection completely. Comparing them against detection is optional (`--compare-auto`).

## Phase 3A: classical candidates and baseline

All detection runs in the frame rectified to 1600 x 1600 px, so scales are fractions of the frame side and independent of camera resolution.

Methods tried (`scripts/explore_candidates.py`, `phase3_output/candidate_method_exploration.json`). Each method was run over the same radius range with the same peak finding. The table gives the annotated recall with all candidates kept, the number of candidates, and the best F1 after tuning a threshold on the same photo (an optimistic number, used only to choose the method):

| Method | img1 | img2 | Verdict |
|---|---|---|---|
| Bright-blob LoG | 37%, 2,846, F1 0.28 | 77%, 1,983, F1 0.37 | Fires on white shell walls in img1 |
| Dark-blob LoG | 13%, 2,103, F1 0.12 | 65%, 3,812, F1 0.17 | Wrong polarity |
| Hough circles | 99.8%, 16,204, F1 0.42 | 100%, 13,210, F1 0.22 | High ceiling; ranking good on img1 only |
| Ring filter only | 99.3%, 19,653, F1 0.23 | 100%, 19,141, F1 0.22 | Too many candidates |
| **Ring + bright LoG (chosen generator)** | **96.1%, 13,638**, F1 0.25 | **100%, 11,481**, F1 0.22 | Per-candidate scale, fewer candidates, about 18 s per photo |

Hough circles reach a slightly higher ceiling on img1 and rank better there. I chose ring + LoG because it gives a contrast-normalized response and scale for every candidate, with fewer candidates; Hough as a generator is untested (see Remaining work).

The ring filter measures centre-ring-surround contrast: a lighter centre enclosed by a dark rim. Median offset of matched candidates is 4.2 px (img1) and 4.8 px (img2).

**OpenCV baseline.** The combined ring + LoG score, thresholded, followed by duplicate suppression. On its own, this score separates openings from texture poorly in both photos (AP 0.13 to 0.16).

## Phase 3B: ML candidate classifier

**Features** (358 dimensions, all scaled by each candidate's own radius):
- radial intensity profile;
- HOG of a 32 x 32 patch;
- uniform LBP histogram;
- detector scores and their within-photo percentiles.

Colour was excluded a priori, because the photos differ completely in colour. An ablation confirmed this: adding colour lowered AP in both directions (0.310 to 0.190 on img2, 0.553 to 0.359 on img1).

**Labels.**

| Label | Rule |
|---|---|
| Positive | Candidate matched one-to-one to an annotation, within half the photo's median opening diameter |
| Negative | Unmatched candidate farther than that radius from every annotation |
| Ignored | Unmatched candidate inside the radius (it lost the one-to-one assignment to a closer candidate) |

| | Positive | Negative | Ignored |
|---|---|---|---|
| img1 | 1,661 | 10,132 | 1,845 |
| img2 | 397 | 10,150 | 934 |

**Label-quality audit.** I checked 96 random negatives by eye: 48 per photo, half near annotations and half in annotation voids. I found no clear live opening in img1 and 2 to 3 ambiguous ones in img2. Negative label noise is therefore small but not zero, and larger in img2.

**Models.** Logistic regression and histogram gradient boosting, both class-balanced. The model family and the score threshold are chosen by 2 x 2 spatially blocked cross-validation on the training photo, with a buffer so that overlapping patches never straddle a fold. Gradient boosting won in both directions. The test photo is never used for any choice that is then scored on it.

**Within-photo blocked CV.** This shows how separable each photo is internally, not how well a model generalizes:

| | AP | F1 |
|---|---|---|
| img1 | 0.76 | 0.72 |
| img2 | 0.63 | 0.66 |

## Phase 3C: cross-photo comparison

Each photo is scored by a model trained only on the other photo.

**Metric definitions.**
- Recall means annotated recall.
- Precision is a **lower bound**, because unmatched detections can be real unannotated barnacles.
- AP is the area under the curve of precision lower bound against annotated recall.

| | Test img2 (trained img1), 397 annotations | Test img1 (trained img2), 1,731 annotations |
|---|---|---|
| AP, OpenCV / ML | 0.132 / **0.310** | 0.155 / **0.553** |
| Threshold carried over from training photo: OpenCV | 6,971 detections, F1 0.107 | 903 detections, F1 0.023 |
| Threshold carried over: ML | 559 detections, F1 0.387 | 4,136 detections, F1 0.484 |
| Calibrated from 80 labels, 20 draws: OpenCV | F1 0.119 ± 0.068 | F1 0.019 ± 0.017 |
| Calibrated from 80 labels, 20 draws: ML | **F1 0.343 ± 0.057**, count 342 ± 185 | **F1 0.551 ± 0.021**, count 1,652 ± 410 |
| Label-free Otsu threshold: ML | 4,203 detections, F1 0.168 | 3,624 detections, F1 0.512 |
| Threshold tuned on the test photo itself (optimistic ceiling): OpenCV / ML | 0.224 / 0.390 | 0.282 / 0.567 |

The thresholds carried over from the training photo are unstable. An earlier run with a smaller duplicate radius put the img2 ML threshold at 51 detections and F1 0.156.

### Operating point on a new photo

`barnacles/calibration.py` implements few-shot review calibration in four steps:

1. Draw a stratified sample of candidates, oversampling the high-score tail.
2. The reviewer labels each one as a barnacle opening or not.
3. Fit a weighted 1-D logistic (Platt) model mapping score to probability.
4. Accept the top N detections, where N is the sum of the calibrated probabilities over all candidates.

The same sample also gives a design-unbiased (Horvitz-Thompson) count estimate with a standard error.

In the evaluation, the reviewer is simulated by the annotations. A plain "probability at least 0.5" rule accepted far too few detections (F1 0.15 to 0.40), so the count-matched rule is the one used throughout.

Count uncertainty after 80 labels, over 20 draws:

| Photo | Estimated count | Truth (candidate-level positives) |
|---|---|---|
| img1 | 1,652 ± 410 | 1,661 |
| img2 | 342 ± 185 | 397 |

The estimates are roughly unbiased but noisy.

### Recommended point: one deterministic draw of 80 labels

`*_calibrated_outcomes.png` and `*_comparison_zoom.png` show this draw.

| | img2 ML | img1 ML |
|---|---|---|
| Detections | 307 | 1,623 |
| F1 | 0.358 | 0.569 |
| Matching error, median / 90th percentile | 6.5 / 13.0 px | 4.5 / 7.2 px |
| Cell counts (10 x 10): mean absolute difference from annotated, correlation | 2.4, r 0.33 | 3.9, r 0.43 |

For comparison, the OpenCV baseline at the same calibration scores 12.5 (img1) and 3.7 (img2) on mean absolute cell difference.

### Dense versus sparse regions

Strata are terciles of local annotation density.

| | Annotated recall: sparse / medium / dense | Unmatched detection rate: sparse / medium / dense |
|---|---|---|
| ML on img1 | 0.61 / 0.58 / 0.49 | 0.50 / 0.42 / 0.30 |
| ML on img2 | 0.37 / 0.34 / 0.24 | 0.73 / 0.43 / 0.41 |
| OpenCV, both photos | at most 0.15 everywhere | |

Dense regions lose recall (touching barnacles are merged by duplicate suppression or rejected). Sparse regions carry more unmatched detections (rock texture, plus unannotated barnacles in img2).

### Matching-radius sensitivity

F1 at the recommended point, for radii of 0.25, 0.5, 0.75 and 1.0 x the median opening diameter (`radius_sensitivity.json`):

| | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| ML on img1 | 0.29 | 0.57 | 0.63 | 0.67 |
| ML on img2 | 0.22 | 0.36 | 0.40 | 0.44 |
| OpenCV on img1 | 0.02 | 0.10 | 0.32 | 0.57 |

At loose radii, dense packing lets even badly placed detections "match". That is why the reported radius is half a diameter.

### Failure patterns

Sources: `galleries/`, `unmatched_breakdown.json`, and a by-eye audit of the 32 top-scoring unmatched detections per photo.

1. **Off-centre detections on annotated barnacles.**
   - In img1, 508 of the 668 unmatched ML detections lie within one opening diameter of an annotation. Most sit on the white shell wall beside an annotated opening.
   - In img2, about 12 of the 32 audited lie on an annotated barnacle but are offset or are second detections.
   - This prompted raising the duplicate radius from 1.0 to 1.5 times the mean radius, on the geometric argument that two openings cannot be closer than the sum of their radii. The audit motivated the change, so it carries evaluation bias.
2. **Bare rock texture**, mostly in img2: about 6 of 32 audited.
3. **Barnacle-like structures with no annotation**, about 14 of 32 in img2. These are likely unannotated live barnacles or empty shells, so the precision lower bound understates real precision there.
4. **Misses.**
   - img1: 70 annotations have no candidate at all (a generator miss); 706 had a candidate that the classifier rejected or that was suppressed.
   - img2: every miss had a candidate (271 rejected).
   - Recall falls in dense patches.

### Computational cost (1 CPU)

| Step | Time |
|---|---|
| Frame detection and rectification | about 1.2 s per photo |
| Candidate generation (needed by both methods) | about 17.5 s per photo |
| Features | about 5 s per photo |
| ML inference and selection | 0.1 to 0.15 s per photo |
| Training: gradient boosting blocked CV | 50 to 59 s, one-off |
| Training: final fit | 3 to 6 s, one-off |

Training logistic regression's CV takes about 1 s. ML adds about 5 s per photo over the OpenCV baseline.

### Recommendation

Use **ML candidate classification with few-shot calibration on each new photo**.
- It ranks candidates far better than the classical score in both directions.
- Calibration fixes the operating-point problem that sinks carried-over thresholds.
- The classical score is useful only as the candidate generator.
- With only two photos, these numbers do not estimate performance on other photos, other species, or other lighting.

## Phase 4: review prototype

`phase4_output/review.html` extends the Phase 2 grid viewer. Scientists can:
- toggle detections, suggestions, uncertainty rings and reference annotations;
- select cells and see model, corrected and reference counts;
- work through a review queue with a zoomed inset, accepting with Y and removing with N;
- remove or restore detections, and add missed barnacles by clicking;
- move the threshold, or recalibrate by labelling the 80-candidate sample;
- export a session (JSON with an action log and results) or the corrected points (CSV), and load a session back.

**Separate panels for separate questions.** Detection confidence (uncertain share), detection density (relative to the photo average) and annotation coverage (share of a cell near any annotation, which only exists for the challenge photos) are shown separately.

**Replay.** Correction rules are mirrored in `barnacles/review.py`, so a session exported from the browser replays identically in Python. This is tested with sessions recorded from the prototype.

**JS versus Python calibration.** The browser recalibrates with Newton's method and Python with scikit-learn. Platt parameters agree to about 0.003, and the counts to within 1 detection (1,624 versus 1,623 on img1, 307 on both for img2).

### Review queue

The queue is ordered by the calibrated probability that an item is a counting error:
- 1 - p for a counted detection;
- p for a suggestion (a rejected candidate just below the threshold that is not a duplicate of a counted detection);
- plus 0.05 for each static flag: crowded, low clarity, touches frame, low classical score.

The flags are also shown as reasons.

**Simulated review** (annotations as the reviewer; errors found within the first N items checked):

| Photo | N = 25 | N = 50 | N = 100 | N = 200 |
|---|---|---|---|---|
| img1, queue / random | 16 / 8.8 | 25 / 18.0 | 45 / 38.2 | 95 / 75.0 |
| img2, queue / random | 19 / 13.0 | 42 / 26.1 | 77 / 53.1 | 134 / 106.5 |

My first ordering, a weighted sum of binary flags, was no better than random on img1 (39 versus 43.6 at 100 items). It was replaced after I saw that result, so the improvement above is partly tuned on these photos.

### Calibration sensitivity to reviewer error

In a UI test, one wrong "opening" answer on a low-score candidate moved img1's count from 1,623 to 2,464. Low-score strata carry large sampling weights, so mistakes there are expensive.

## Phase 5: tests

`python -m unittest discover -s tests -t .` runs 35 tests in about 6 s. They cover:
- coordinate conversions and resolution invariance;
- the inside test;
- frame detection on synthetic grids, including perspective;
- manual corners;
- the real photos against recorded corners;
- grid boundaries and reconciliation;
- optimal one-to-one matching;
- duplicate suppression and its parity with the review rule;
- review actions and count reconciliation;
- replay of browser-exported sessions;
- determinism of models, blocked CV and calibration samples;
- the saved models reproducing Phase 3 scores exactly;
- the calibration estimators.

pytest is not installed in this environment; the tests are plain `unittest` and also run under pytest.

## Limitations

**Data**
- Two photos, with different species or size classes, lighting and annotation completeness. Cross-photo results are two samples, not a performance estimate.
- The annotations are incomplete (img2) and skip empty shells, so recall and precision are measured against a policy, not against ground truth.

**Design choices made with both photos in view**
- The candidate generator's design and scale range.
- The duplicate radius change.
- The review-queue ordering.
- Matching radius and feature choices are justified on geometric grounds, but they were not frozen before seeing results.

**Operating point and counts**
- Counts from 80 labels are noisy: about ±25% (img1) to ±55% (img2).
- Calibration is sensitive to reviewer mistakes on low-score candidates.

**Detector**
- Dense clusters lose recall.
- About 4% of img1's annotations are never proposed by the generator.

**No deep learning.** PyTorch is not available here. With two photos, a CNN would also be hard to justify.

## Remaining work, ranked

1. **More annotated photos, exhaustively labelled in a few patches.** This is the only way to measure real precision and recall, and to estimate generalization.
2. **More robust calibration.** Cap stratum weights or use more strata, add a reviewer-error check (repeat a few labels), and report a count interval in the UI.
3. **Better dense-region recall.** Size-aware duplicate suppression (opening size estimated from accepted detections), and a second-stage check for touching openings.
4. **Better localization.** Refine each center to the operculum, for example with a local ring fit, to cut off-centre detections on shell walls.
5. **Try Hough circles as the candidate generator.** They reached a 99.8% recall ceiling on img1 versus 96.1% for the chosen generator.
6. **Use reviewer corrections as training data.** Exported sessions are already in a replayable format; retrain on them.
7. **Policy decision on empty shells**, then relabel or add a class.
8. **Product hardening of the prototype.** Undo, a per-cell review mode, multi-photo projects, and server-side session storage.
