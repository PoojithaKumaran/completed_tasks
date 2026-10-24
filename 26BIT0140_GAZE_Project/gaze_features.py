"""Consistent raw and eye-relative gaze features for calibration and inference."""
from __future__ import annotations

import numpy as np


def extract_gaze_features(landmarks) -> np.ndarray:
    """Return 8 features: raw iris x/y, then iris position relative to each eye.

    Relative horizontal position uses the eye-corner span. Relative vertical
    position uses the corner-line baseline and the same horizontal span, so
    head translation and camera distance have less direct influence.
    """
    if landmarks is None or len(landmarks) <= 477:
        raise ValueError("Face and iris landmarks are required")

    feature_values = [
        float(landmarks[468].x), float(landmarks[468].y),
        float(landmarks[473].x), float(landmarks[473].y),
    ]
    for iris_idx, corner_a, corner_b in ((468, 33, 133), (473, 362, 263)):
        iris = landmarks[iris_idx]
        a, b = landmarks[corner_a], landmarks[corner_b]
        span = abs(float(b.x) - float(a.x))
        if span <= 1e-6:
            raise ValueError("Eye-corner landmarks have invalid separation")
        low_x = min(float(a.x), float(b.x))
        baseline_y = (float(a.y) + float(b.y)) / 2.0
        feature_values.extend([
            (float(iris.x) - low_x) / span,
            (float(iris.y) - baseline_y) / span,
        ])

    features = np.asarray(feature_values, dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Gaze features contain non-finite values")
    return features
