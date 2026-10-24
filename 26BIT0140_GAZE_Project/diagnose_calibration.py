
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Configuration
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "calibration_data.csv"
WIDTH, HEIGHT = 1000, 650
TEST_SIZE = 0.25
RANDOM_STATE = 42

FEATURES = [
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
]

REQUIRED = [
    "timestamp", "target_id", "target_x", "target_y",
    *FEATURES
]

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. Run this script from your GAZE Project folder."
    )

df = pd.read_csv(INPUT_FILE)

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for c in REQUIRED:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=REQUIRED).copy()
df = df.sort_values("timestamp").reset_index(drop=True)

# Match the baseline: select the latest session.
session_start = 0
for i in range(1, len(df)):
    if df.loc[i, "timestamp"] - df.loc[i - 1, "timestamp"] > 30:
        session_start = i

df = df.iloc[session_start:].copy()
df = df[
    df["target_id"].isin(range(1, 10))
    & df["target_x"].between(0, WIDTH - 1)
    & df["target_y"].between(0, HEIGHT - 1)
].copy()

df["target_id"] = df["target_id"].astype(int)
df = df.sort_values("timestamp").reset_index(drop=True)

counts = df["target_id"].value_counts().sort_index()

print("\n===== DATA VALIDATION =====")
print("Latest-session readings:", len(df))
print("\nReadings per target:")
print(counts.to_string())

if set(counts.index) != set(range(1, 10)):
    raise ValueError("The latest session must contain all 9 targets.")
if counts.min() < 2:
    raise ValueError("Each target needs at least 2 readings.")

# 1. Feature variation for each target
print("\n===== FEATURE VARIATION BY TARGET =====")
feature_stats = df.groupby("target_id")[FEATURES].agg(["mean", "std"])
print(feature_stats.round(5).to_string())
feature_stats.to_csv(BASE_DIR / "diagnostic_feature_variation.csv")

# 2. Compare target feature means
means = df.groupby("target_id")[FEATURES].mean()
print("\n===== TARGET MEAN FEATURES =====")
print(means.round(5).to_string())

print("\n===== NEAREST TARGET-MEAN CHECK =====")
# Diagnostic only: how often each reading is closest to its
# own target's average feature vector.
X = df[FEATURES].to_numpy(dtype=float)
labels = df["target_id"].to_numpy()
target_means = means.to_numpy()
target_ids = means.index.to_numpy()

distances = np.linalg.norm(
    X[:, None, :] - target_means[None, :, :], axis=2
)
nearest_ids = target_ids[np.argmin(distances, axis=1)]
print(
    "Readings closest to their own target mean: "
    f"{np.mean(nearest_ids == labels) * 100:.2f}%"
)
print("This is a diagnostic, not a screen-position accuracy score.")

# 3. Fit/evaluate model and inspect directional bias
Y = df[["target_x", "target_y"]].to_numpy(dtype=float)

def evaluate(name, train_idx, test_idx):
    model = LinearRegression()
    model.fit(X[train_idx], Y[train_idx])
    pred = model.predict(X[test_idx])

    pred[:, 0] = np.clip(pred[:, 0], 0, WIDTH - 1)
    pred[:, 1] = np.clip(pred[:, 1], 0, HEIGHT - 1)

    actual = Y[test_idx]
    delta = pred - actual
    errors = np.linalg.norm(delta, axis=1)

    print(f"\n===== {name} =====")
    print("Train:", len(train_idx), "| Test:", len(test_idx))
    print(f"Mean absolute X error: {np.mean(np.abs(delta[:, 0])):.2f}px")
    print(f"Mean absolute Y error: {np.mean(np.abs(delta[:, 1])):.2f}px")
    print(f"Mean position error: {errors.mean():.2f}px")
    print(f"Median position error: {np.median(errors):.2f}px")
    print(f"Within 42px: {np.mean(errors <= 42) * 100:.2f}%")
    print(f"Mean signed X error (right is positive): {delta[:, 0].mean():.2f}px")
    print(f"Mean signed Y error (down is positive): {delta[:, 1].mean():.2f}px")

    results = pd.DataFrame({
        "timestamp": df.iloc[test_idx]["timestamp"].to_numpy(),
        "target_id": labels[test_idx],
        "actual_x": actual[:, 0],
        "actual_y": actual[:, 1],
        "predicted_x": pred[:, 0],
        "predicted_y": pred[:, 1],
        "error_x": delta[:, 0],
        "error_y": delta[:, 1],
        "position_error": errors,
    })
    filename = "diagnostic_predictions_" + (
        name.lower().replace(" ", "_").replace("-", "_")
    ) + ".csv"
    results.to_csv(BASE_DIR / filename, index=False)
    print("Saved:", filename)

# Random stratified split
all_idx = np.arange(len(df))
train, test = train_test_split(
    all_idx, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=labels
)
evaluate("random_stratified", train, test)

# Global chronological split
cut = int(len(df) * (1 - TEST_SIZE))
evaluate(
    "global_chronological",
    np.arange(0, cut),
    np.arange(cut, len(df))
)

# Within-target chronological split
train_parts, test_parts = [], []
for target_id in range(1, 10):
    idx = df.index[df["target_id"] == target_id].to_numpy()
    idx = idx[np.argsort(df.loc[idx, "timestamp"].to_numpy())]
    cut = int(len(idx) * (1 - TEST_SIZE))
    cut = min(max(cut, 1), len(idx) - 1)
    train_parts.extend(idx[:cut])
    test_parts.extend(idx[cut:])

evaluate(
    "within_target_chronological",
    np.array(sorted(train_parts), dtype=int),
    np.array(sorted(test_parts), dtype=int)
)

print("\n===== FILES SAVED =====")
print("- diagnostic_feature_variation.csv")
print("- diagnostic_predictions_random_stratified.csv")
print("- diagnostic_predictions_global_chronological.csv")
print("- diagnostic_predictions_within_target_chronological.csv")
