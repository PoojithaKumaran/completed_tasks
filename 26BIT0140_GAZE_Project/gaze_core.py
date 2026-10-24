"""Hardware-independent helpers shared by gaze tracking and tests."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

WIDTH, HEIGHT = 1000, 650
TARGETS = (
    (150, 120), (500, 120), (850, 120),
    (150, 325), (500, 325), (850, 325),
    (150, 530), (500, 530), (850, 530),
)
CALIBRATION_BASE_COLUMNS = ("timestamp", "target_id", "target_x", "target_y")
RAW_FEATURE_COLUMNS = ("left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y")
RELATIVE_FEATURE_COLUMNS = (
    "left_eye_rel_x", "left_eye_rel_y", "right_eye_rel_x", "right_eye_rel_y"
)
INTERACTION_COLUMNS = (
    "timestamp", "user_id", "gaze_x", "gaze_y", "target_under_gaze",
    "expected_target", "fixation_time", "blink", "selection", "selection_success"
)
INTERRUPTION_COLUMNS = (
    "timestamp", "user_id", "expected_target", "target_under_gaze", "dwell_time", "event"
)


def load_latest_calibration(path, feature_columns=RAW_FEATURE_COLUMNS, gap_seconds=30.0):
    """Load finite readings from the latest timestamp session.

    Returns feature matrix, screen-point matrix, integer target labels, and the
    filtered session DataFrame. Never modifies the input file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    required = [*CALIBRATION_BASE_COLUMNS, *feature_columns]
    df = pd.read_csv(path)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Calibration CSV missing columns: {missing}")
    df = df[required].copy()
    df[required] = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    df = df[df.target_id.isin(range(1, 10))]
    df = df[df.target_x.between(0, WIDTH - 1) & df.target_y.between(0, HEIGHT - 1)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("Calibration CSV has no valid readings")
    session_starts = np.flatnonzero(df.timestamp.diff().gt(gap_seconds).fillna(False).to_numpy())
    start = int(session_starts[-1]) if session_starts.size else 0
    latest = df.iloc[start:].reset_index(drop=True)
    X = latest[list(feature_columns)].to_numpy(dtype=float)
    Y = latest[["target_x", "target_y"]].to_numpy(dtype=float)
    labels = latest.target_id.to_numpy(dtype=int)
    return X, Y, labels, latest


def fit_gaze_model(features, screen_points):
    """Fit the baseline explainable multi-output linear mapping."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(screen_points, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 2 or len(x) != len(y):
        raise ValueError("Expected feature matrix and matching N×2 screen points")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Model fitting requires at least two finite readings")
    return LinearRegression().fit(x, y)


def bound_screen_point(x: float, y: float) -> tuple[int, int]:
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Gaze coordinates must be finite")
    return (int(min(max(round(x), 0), WIDTH - 1)),
            int(min(max(round(y), 0), HEIGHT - 1)))


def identify_target(x: float, y: float, radius: float = 42) -> int | None:
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    for target_id, (tx, ty) in enumerate(TARGETS, start=1):
        if math.hypot(x - tx, y - ty) <= radius:
            return target_id
    return None


@dataclass
class DwellTracker:
    duration: float = 3.0
    target_id: int | None = None
    started_at: float | None = None
    carried_elapsed: float = 0.0
    paused_at: float | None = None

    def update(self, target_id: int | None, now: float) -> tuple[str, float]:
        """Return (started/progress/completed/interrupted/idle, elapsed)."""
        if target_id is None:
            elapsed = self.elapsed(now)
            state = "interrupted" if self.target_id is not None else "idle"
            self.reset()
            return state, elapsed
        if target_id != self.target_id or self.started_at is None:
            self.target_id, self.started_at = target_id, now
            self.carried_elapsed, self.paused_at = 0.0, None
            return "started", 0.0
        elapsed = self.elapsed(now)
        if elapsed >= self.duration:
            self.reset()
            return "completed", elapsed
        return "progress", elapsed

    def elapsed(self, now: float) -> float:
        if self.started_at is None:
            return 0.0
        if self.paused_at is not None:
            return self.carried_elapsed
        return self.carried_elapsed + max(0.0, now - self.started_at)

    def pause(self, now: float) -> None:
        if self.target_id is not None and self.started_at is not None and self.paused_at is None:
            self.carried_elapsed = self.elapsed(now)
            self.paused_at = now

    def resume(self, now: float) -> None:
        if self.paused_at is not None:
            self.started_at = now
            self.paused_at = None

    def reset(self) -> None:
        self.target_id, self.started_at = None, None
        self.carried_elapsed, self.paused_at = 0.0, None
