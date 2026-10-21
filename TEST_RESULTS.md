# Test results

All runs used `python -m unittest discover -s tests -t .` on Python 3.12 and one CPU. The tests use only the standard library's `unittest`; pytest was not installed.

| Run | Where | Development data | Ran | Passed | Failed | Skipped |
|---|---|---|---|---|---|---|
| 1 | Development sandbox, with experiment and Phase 3 caches | yes | 70 | 70 | 0 | 0 |
| 2 | This package, as shipped (fresh-checkout conditions) | no (`BARNACLE_DATA` unset, `data/` empty) | 70 | 58 | 0 | 12 |
| 3 | This package, with `BARNACLE_DATA` pointing at the challenge photos | yes | 70 | 69 | 0 | 1 |

## Skipped in run 2 (no development data)

- `test_app.test_real_upload_end_to_end`: development photo not available
- `test_fingerprint.test_photos_do_not_match_each_other`: development photos not available (set BARNACLE_DATA)
- `test_fingerprint.test_resized_and_recompressed_copies_are_likely`: development photos not available (set BARNACLE_DATA)
- `test_fingerprint.test_shipped_fingerprints_match_the_photos`: development photos not available (set BARNACLE_DATA)
- `test_fingerprint.test_small_crop_is_at_most_possible_and_never_the_other_photo`: development photos not available (set BARNACLE_DATA)
- `test_frame.test_real_photos_match_recorded_frames`: challenge photos not available
- `test_inference.test_calibration_sample_exact`: development photo or model artifact not available
- `test_inference.test_coordinates_reconstruct_across_resolution`: development photo or model artifact not available
- `test_inference.test_detections_valid_and_unique`: development photo or model artifact not available
- `test_inference.test_payload_has_no_annotations`: development photo or model artifact not available
- `test_inference.test_training_photo_flag`: development photo or model artifact not available
- `test_reproducibility.test_saved_model_reproduces_phase3_scores`: phase3 outputs or cache not present (run run_phase3.py)

## Skipped in run 3

- `test_reproducibility.test_saved_model_reproduces_phase3_scores`: needs the Phase 3 feature cache, which is excluded from the package (regenerate with `python run_phase3.py`). It passed in run 1.

## Every test in run 3

**test_app** (10)

- test_internal_errors_are_generic: passed (it deliberately triggers a server error, so its log line was interleaved)
- test_bad_corners_rejected: passed
- test_busy_server_returns_503: passed
- test_evaluate: passed
- test_index_and_health: passed
- test_internal_errors_are_generic: passed
- test_rate_limit: passed
- test_real_upload_end_to_end: passed
- test_security_headers_and_404: passed
- test_unsupported_and_corrupt: passed

**test_calibration_safeguards** (6)

- test_session_round_trip: passed
- test_monotone_rates: passed
- test_single_low_stratum_error_is_damped: passed
- test_exact_budget: passed
- test_small_population: passed
- test_weights_sum_to_population: passed

**test_duplicates** (5)

- test_keeps_best_of_overlapping: passed
- test_pairs_roundtrip: passed
- test_review_rule_matches_ml_rule: passed
- test_select_count: passed
- test_separated_points_untouched: passed

**test_fingerprint** (6)

- test_photos_do_not_match_each_other: passed
- test_resized_and_recompressed_copies_are_likely: passed
- test_shipped_fingerprints_match_the_photos: passed
- test_small_crop_is_at_most_possible_and_never_the_other_photo: passed
- test_levels: passed
- test_no_refs: passed

**test_frame** (9)

- test_corners_map_to_unit_square: passed
- test_inside_tests_agree: passed
- test_resolution_invariance: passed
- test_round_trip: passed
- test_manual_config_bypasses_and_rescales: passed
- test_no_frame_raises: passed
- test_real_photos_match_recorded_frames: passed
- test_synthetic_axis_aligned: passed
- test_synthetic_perspective: passed

**test_grid** (5)

- test_boundaries: passed
- test_invalid_grid: passed
- test_outside: passed
- test_selftest_many_grids: passed
- test_sum_equals_inside_for_random_points: passed

**test_inference** (12)

- test_calibration_sample_exact: passed
- test_coordinates_reconstruct_across_resolution: passed
- test_detections_valid_and_unique: passed
- test_payload_has_no_annotations: passed
- test_training_photo_flag: passed
- test_synthetic_photo: passed
- test_accepts_png_and_jpeg: passed
- test_header_size_matches_decoder: passed
- test_manual_corners_validated: passed
- test_no_frame_raises_frame_error: passed
- test_rejects_bad_inputs: passed
- test_rejects_other_formats_and_bombs: passed

**test_matching** (5)

- test_empty: passed
- test_large_random_is_valid: passed
- test_metrics: passed
- test_one_to_one_and_radius: passed
- test_optimal_not_greedy: passed

**test_reproducibility** (6)

- test_horvitz_thompson_close_to_truth: passed
- test_platt_threshold_recovers_midpoint: passed
- test_blocked_cv_deterministic: passed
- test_calibration_sample_deterministic: passed
- test_models_deterministic: passed
- test_saved_model_reproduces_phase3_scores: skipped 'phase3 outputs or cache not present (run run_phase3.py)'

**test_review** (6)

- test_counts_reconcile: passed
- test_dashboard_breakdown: passed
- test_model_set: passed
- test_remove_keep_add_restore: passed
- test_threshold_action_and_overrides_persist: passed
- test_ui_session_replays_identically: passed

Total listed: 70. unittest summary: `Ran 70 tests ... OK (skipped=1)`. The log was produced with library warnings silenced (`-W ignore`, `OPENCV_LOG_LEVEL=SILENT`, `LOG_LEVEL=ERROR`).
