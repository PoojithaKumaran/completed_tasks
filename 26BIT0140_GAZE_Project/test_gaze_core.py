import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from gaze_core import (
    DwellTracker, bound_screen_point, identify_target, load_latest_calibration,
    fit_gaze_model, RAW_FEATURE_COLUMNS, INTERACTION_COLUMNS,
    INTERRUPTION_COLUMNS,
)
from gaze_features import extract_gaze_features


class GazeCoreTests(unittest.TestCase):
    def test_coordinate_bounds_and_nonfinite_rejection(self):
        self.assertEqual(bound_screen_point(-20, 900), (0, 649))
        with self.assertRaises(ValueError):
            bound_screen_point(math.nan, 5)

    def test_target_identification(self):
        self.assertEqual(identify_target(150, 120), 1)
        self.assertEqual(identify_target(192, 120), 1)
        self.assertIsNone(identify_target(193, 120))

    def test_dwell_completion_and_interruption(self):
        tracker = DwellTracker(duration=3)
        self.assertEqual(tracker.update(2, 1.0), ("started", 0.0))
        self.assertEqual(tracker.update(2, 2.5), ("progress", 1.5))
        self.assertEqual(tracker.update(3, 2.6), ("started", 0.0))
        state, elapsed = tracker.update(None, 4.0)
        self.assertEqual(state, "interrupted")
        self.assertAlmostEqual(elapsed, 1.4)
        tracker.update(3, 5.0)
        self.assertEqual(tracker.update(3, 8.0)[0], "completed")

    def test_dwell_pause_excludes_blink_time(self):
        tracker = DwellTracker(duration=3)
        tracker.update(1, 10.0)
        tracker.update(1, 11.0)
        tracker.pause(11.0)
        self.assertEqual(tracker.elapsed(20.0), 1.0)
        tracker.resume(20.0)
        self.assertEqual(tracker.update(1, 22.0), ("completed", 3.0))

    def test_eye_relative_features_use_corner_geometry(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        landmarks[468] = SimpleNamespace(x=0.25, y=0.51)
        landmarks[33] = SimpleNamespace(x=0.20, y=0.50)
        landmarks[133] = SimpleNamespace(x=0.30, y=0.50)
        landmarks[473] = SimpleNamespace(x=0.75, y=0.49)
        landmarks[362] = SimpleNamespace(x=0.70, y=0.50)
        landmarks[263] = SimpleNamespace(x=0.80, y=0.50)
        result = extract_gaze_features(landmarks)
        self.assertEqual(result.shape, (8,))
        self.assertAlmostEqual(result[4], 0.5)
        self.assertAlmostEqual(result[5], 0.1)
        self.assertAlmostEqual(result[6], 0.5)
        self.assertAlmostEqual(result[7], -0.1)

    def test_missing_calibration_file_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_latest_calibration(Path(directory) / "missing.csv")

    def test_malformed_calibration_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            pd.DataFrame({"timestamp": [1], "target_id": [1]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_latest_calibration(path)

    def test_latest_session_selection_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.csv"
            rows = []
            for stamp, offset in ((1.0, 0.0), (2.0, 0.1), (100.0, 0.2), (101.0, float("nan"))):
                rows.append({
                    "timestamp": stamp, "target_id": 1, "target_x": 150, "target_y": 120,
                    **{column: offset for column in RAW_FEATURE_COLUMNS},
                })
            pd.DataFrame(rows).to_csv(path, index=False)
            X, Y, labels, session = load_latest_calibration(path)
            self.assertEqual(len(session), 1)
            self.assertEqual(X.shape, (1, 4))
            self.assertEqual(Y.shape, (1, 2))
            self.assertEqual(labels.tolist(), [1])

    def test_linear_model_prediction_dimensions(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        Y = np.array([[10, 20], [20, 20], [10, 30], [20, 30]], dtype=float)
        model = fit_gaze_model(X, Y)
        self.assertEqual(model.predict(X[:2]).shape, (2, 2))

    def test_interaction_log_schemas_are_explicit(self):
        self.assertEqual(len(INTERACTION_COLUMNS), 10)
        self.assertEqual(INTERACTION_COLUMNS[0], "timestamp")
        self.assertIn("selection_success", INTERACTION_COLUMNS)
        self.assertEqual(len(INTERRUPTION_COLUMNS), 6)
        self.assertEqual(INTERRUPTION_COLUMNS[-1], "event")

    def test_cleaning_writes_separate_file_and_preserves_raw(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            shutil.copy2(project / "clean_dataset.py", temp / "clean_dataset.py")
            raw = pd.DataFrame([{
                "timestamp": "2026-01-01T00:00:00", "user_id": "test_user",
                "gaze_x": 150, "gaze_y": 120, "target_under_gaze": 1,
                "expected_target": 1, "fixation_time": 3.0, "blink": "no",
                "selection": 1, "selection_success": "yes",
            }])
            raw_path = temp / "gaze_interactions.csv"
            raw.to_csv(raw_path, index=False)
            before = raw_path.read_bytes()
            subprocess.run([sys.executable, str(temp / "clean_dataset.py")],
                           cwd=temp, check=True, capture_output=True, text=True)
            self.assertEqual(raw_path.read_bytes(), before)
            self.assertTrue((temp / "gaze_interactions_cleaned.csv").is_file())

    def test_evaluation_scripts_generate_reports(self):
        project = Path(__file__).resolve().parents[1]
        for script in ("evaluate_calibration_v2.py", "evaluate_dataset.py"):
            result = subprocess.run(
                [sys.executable, str(project / script)], cwd=project,
                check=True, capture_output=True, text=True,
            )
            self.assertIn("successfully" if script == "evaluate_dataset.py" else "SIDE-BY-SIDE", result.stdout)
            if script == "evaluate_dataset.py":
                self.assertIn("Interrupted dwell attempts:", result.stdout)
                self.assertIn("Attempt success rate:", result.stdout)
        self.assertTrue((project / "calibration_evaluation_comparison.csv").is_file())
        self.assertTrue((project / "participant_evaluation_report.txt").is_file())


if __name__ == "__main__":
    unittest.main()
