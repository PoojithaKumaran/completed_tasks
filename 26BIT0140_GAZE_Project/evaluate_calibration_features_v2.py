"""Evaluate raw plus eye-relative calibration features with baseline splits."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "calibration_data_v2.csv"
FEATURES = [
    "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
    "left_eye_rel_x", "left_eye_rel_y", "right_eye_rel_x", "right_eye_rel_y",
]
REQUIRED = ["timestamp", "target_id", "target_x", "target_y", *FEATURES]
if not INPUT.exists():
    raise FileNotFoundError("Run calibration.py to create calibration_data_v2.csv")

df = pd.read_csv(INPUT)
missing = [column for column in REQUIRED if column not in df]
if missing:
    raise ValueError(f"Missing columns in {INPUT.name}: {missing}")
df[REQUIRED] = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED)
df = df.sort_values("timestamp").reset_index(drop=True)
gaps = df["timestamp"].diff().gt(30).fillna(False).to_numpy()
starts = np.flatnonzero(gaps)
df = df.iloc[int(starts[-1]) if starts.size else 0:].reset_index(drop=True)
df = df[df.target_id.isin(range(1, 10))].reset_index(drop=True)
counts = df.target_id.value_counts()
print("Latest-session readings:", len(df))
print("Readings per target:\n", counts.sort_index().to_string())
if set(counts.index.astype(int)) != set(range(1, 10)) or counts.min() < 2:
    raise ValueError("Latest v2 session needs at least two valid samples per target.")

X = df[FEATURES].to_numpy(dtype=float)
Y = df[["target_x", "target_y"]].to_numpy(dtype=float)
labels = df.target_id.to_numpy(dtype=int)
all_indices = np.arange(len(df))
train_r, test_r = train_test_split(
    all_indices, test_size=0.25, random_state=42, stratify=labels
)
cut = int(len(df) * 0.75)
splits = [
    ("random_stratified", train_r, test_r),
    ("global_chronological", all_indices[:cut], all_indices[cut:]),
]
train_parts, test_parts = [], []
for target_id in range(1, 10):
    indices = np.flatnonzero(labels == target_id)
    split = min(max(int(len(indices) * 0.75), 1), len(indices) - 1)
    train_parts.extend(indices[:split])
    test_parts.extend(indices[split:])
splits.append(("within_target_chronological", np.array(sorted(train_parts)),
               np.array(sorted(test_parts))))

summaries, detail_frames = [], []
for name, train, test in splits:
    model = LinearRegression().fit(X[train], Y[train])
    prediction = model.predict(X[test])
    prediction[:, 0] = np.clip(prediction[:, 0], 0, 999)
    prediction[:, 1] = np.clip(prediction[:, 1], 0, 649)
    delta = prediction - Y[test]
    errors = np.linalg.norm(delta, axis=1)
    summaries.append({
        "method": name, "training_readings": len(train), "testing_readings": len(test),
        "mean_position_error": errors.mean(), "median_position_error": np.median(errors),
        "p90_position_error": np.percentile(errors, 90),
        "within_radius_percent": (errors <= 42).mean() * 100,
    })
    detail = pd.DataFrame({
        "target_id": labels[test], "actual_x": Y[test, 0], "actual_y": Y[test, 1],
        "predicted_x": prediction[:, 0], "predicted_y": prediction[:, 1],
        "error_x": delta[:, 0], "error_y": delta[:, 1], "position_error": errors,
    })
    detail.to_csv(BASE_DIR / f"feature_v2_predictions_{name}.csv", index=False)
    detail.insert(0, "split", name)
    detail_frames.append(detail)

summary = pd.DataFrame(summaries)
summary.to_csv(BASE_DIR / "calibration_feature_v2_comparison.csv", index=False)
print("\nSame-split v2 comparison:\n", summary.round(2).to_string(index=False))
print("\nSaved calibration_feature_v2_comparison.csv and per-split prediction files.")
