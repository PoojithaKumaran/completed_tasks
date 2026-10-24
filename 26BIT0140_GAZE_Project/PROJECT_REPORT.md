# GAZE Project â€” Development and Evaluation Report

## Architecture

The prototype uses MediaPipe Tasks FaceLandmarker with `face_landmarker.task`, reads iris landmarks 468 and 473 plus eye corners for v2, fits a two-output linear regression from the newest available calibration CSV, clamps predictions to a 1000 Ã— 650 screen, identifies one of nine 42 px targets, and requires a 3 second dwell. Completed selections append to `gaze_interactions.csv`; interrupted attempts append to `gaze_interruption_log.csv`. Cleaning and evaluation preserve raw input files.

## Changes in this handoff

- Added `gaze_core.py` for finite coordinate bounds, target identification, and independently testable dwell state.
- Updated `gaze_inteface.py` to use those helpers, validate calibration timestamps and full target coverage, reject non-finite gaze points, and pause dwell updates during blinks. Lost tracking or gaze away from targets resets an incomplete dwell and records interruption evidence.
- Calibration now stores the median of 15 valid frames per SPACE press, and live inference uses a seven-frame median filter to reduce brief frame noise. Blink time is excluded from dwell duration. Calibration target order shuffles each pass to reduce order/time confounding.
- Added a separate v2 calibration dataset with eye-relative iris features normalized by actual eye-corner landmarks. Live inference and the new evaluator share this feature extraction; the legacy calibration CSV and baseline evaluator remain intact.
- Anchored calibration evaluation, diagnostic, cleaning, and interaction report paths to the project directory.
- Made plotting headless-compatible and timestamp validation explicit in dataset cleaning.
- Added unit tests, this README, and this report. Kept `evaluate_calibration_v2.py` and raw datasets/logs in place.

## Actual evaluation results

Latest calibration session at the latest evaluation: **133 readings** across all targets (14â€“15 readings per target).

| Split | Mean position error | Median | 90th percentile | Within 42 px |
|---|---:|---:|---:|---:|
| Random stratified | 316.83 px | 361.36 px | 452.99 px | 8.82% |
| Global chronological | 303.61 px | 364.31 px | 427.30 px | 8.82% |
| Within-target chronological | 308.29 px | 367.12 px | 423.28 px | 8.33% |

The shuffled 133-reading legacy session remained inaccurate. The first physical v2 session contains 94 readings (9â€“12 per target). Its same-split results are:

| Split | Mean position error | Median | P90 | Within 42 px |
|---|---:|---:|---:|---:|
| Random stratified | 320.25 px | 356.59 px | 465.85 px | 8.33% |
| Global chronological | 317.96 px | 339.90 px | 476.23 px | 0.00% |
| Within-target chronological | 333.42 px | 332.21 px | 497.37 px | 0.00% |

Eye-relative features did not improve measured accuracy. Feature means remain very similar across target labels and within-target variation is substantial. This suggests these signals as collected do not reliably encode the intended target; the issue may be gaze protocol, setup, camera view, or the limits of the current features. Do not claim accurate gaze tracking based on a single successful selection.

Latest interaction evaluation found 80 raw interaction rows: 79 completed selections labeled successful and 244 interrupted dwell attempts across the included users. The combined logged-attempt success rate is **24.46%** (79/323); mean fixation time among completed selections is 3.019 s. This covers logged dwell attempts only; periods that never started are absent, and it is not a gaze-accuracy score. Cleaning retained all 80 rows, marked 11 missing/invalid blink labels `unknown`, and found no exact duplicate or inconsistent outcome labels.

## Verification actually run

- `python -m py_compile` on the main app, shared logic, cleaning, evaluation, baseline and diagnostics: passed.
- `python -m unittest discover -s tests -v`: 12 tests passed (CSV validation/latest-session selection, missing calibration, model output dimensions, bounds, targets, dwell/interruption/blink pause, feature extraction, log schemas, raw-data preservation, and evaluation reports).
- `python evaluate_calibration_v2.py`: passed on the last legacy calibration session (133 readings); the current metrics are shown above.
- `python evaluate_calibration_features_v2.py`: passed on 94 newly collected v2 readings; results above.
- `python diagnose_calibration.py`: passed; feature variation, target means and residual prediction CSVs produced.
- `python inspect_calibration.py`: passed; 110 usable records and 3 sessions, latest has all targets.
- `python dataset_check.py`: passed; 110 records, no missing or invalid rows.
- `python clean_dataset.py`: passed; raw interaction CSV unchanged, cleaned output produced.
- `python evaluate_dataset.py`: passed; summary CSV, two charts and participant report produced using a headless backend.
- `python webcam_test.py`: did not pass; OpenCV reported no camera device available. Live inference and physical collector use remain unverified.

## Demonstration status and next action

The analytical scripts and hardware-independent logic are runnable and tested. The user reports a successful nine-target live sequence. The interaction log contains 79 completed selections and 244 interrupted dwell attempts overall. The 94-reading eye-relative calibration evaluation still has high error, so reliable gaze accuracy is not demonstrated. Webcam availability was not verified by this execution environment; the user performed live interaction tests manually.
