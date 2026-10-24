import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "calibration_data.csv"
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650
TARGET_RADIUS = 42
TEST_SIZE = 0.25
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y"
]

REQUIRED_COLUMNS = [
    "timestamp", "target_id", "target_x", "target_y",
    *FEATURE_COLUMNS
]

# --------------------------------------------------
# 1. Load and validate calibration data
# --------------------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"{INPUT_FILE} not found.")

df = pd.read_csv(INPUT_FILE)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for column in REQUIRED_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=REQUIRED_COLUMNS).copy()

if df.empty:
    raise ValueError("No valid calibration records found.")

# Match the main application: select the latest session.
df = df.sort_values("timestamp").reset_index(drop=True)
timestamps = df["timestamp"].to_numpy()
session_start = 0

for i in range(1, len(df)):
    if timestamps[i] - timestamps[i - 1] > 30:
        session_start = i

df = df.iloc[session_start:].copy()
df = df[
    df["target_id"].isin(range(1, 10))
    & df["target_x"].between(0, SCREEN_WIDTH - 1)
    & df["target_y"].between(0, SCREEN_HEIGHT - 1)
].copy()

df["target_id"] = df["target_id"].astype(int)
df = df.sort_values("timestamp").reset_index(drop=True)

target_counts = df["target_id"].value_counts()

print("\n===== CALIBRATION VALIDATION =====")
print("Latest-session readings:", len(df))
print("\nReadings per target:")
print(target_counts.sort_index().to_string())

if set(target_counts.index) != set(range(1, 10)):
    raise ValueError("Latest session must contain all 9 targets.")

if target_counts.min() < 2:
    raise ValueError("Each target needs at least 2 readings for evaluation.")

X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
Y = df[["target_x", "target_y"]].to_numpy(dtype=float)
labels = df["target_id"].to_numpy(dtype=int)


def evaluate_split(name, train_indices, test_indices):
    """Fit a linear model and report held-out gaze-position errors."""
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    labels_test = labels[test_indices]

    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    predictions[:, 0] = np.clip(predictions[:, 0], 0, SCREEN_WIDTH - 1)
    predictions[:, 1] = np.clip(predictions[:, 1], 0, SCREEN_HEIGHT - 1)

    errors = np.linalg.norm(predictions - Y_test, axis=1)
    within_radius = errors <= TARGET_RADIUS

    summary = {
        "method": name,
        "training_readings": len(train_indices),
        "testing_readings": len(test_indices),
        "mean_absolute_x_error": mean_absolute_error(Y_test[:, 0], predictions[:, 0]),
        "mean_absolute_y_error": mean_absolute_error(Y_test[:, 1], predictions[:, 1]),
        "mean_position_error": errors.mean(),
        "median_position_error": np.median(errors),
        "p90_position_error": np.percentile(errors, 90),
        "within_radius_percent": within_radius.mean() * 100,
    }

    print(f"\n===== {name.upper()} =====")
    print("Training readings:", len(train_indices))
    print("Testing readings:", len(test_indices))
    print(f"Mean absolute X error: {summary['mean_absolute_x_error']:.2f} pixels")
    print(f"Mean absolute Y error: {summary['mean_absolute_y_error']:.2f} pixels")
    print(f"Mean screen-position error: {summary['mean_position_error']:.2f} pixels")
    print(f"Median screen-position error: {summary['median_position_error']:.2f} pixels")
    print(f"90th-percentile error: {summary['p90_position_error']:.2f} pixels")
    print(
        f"Predictions within {TARGET_RADIUS}px of intended target: "
        f"{summary['within_radius_percent']:.2f}%"
    )

    per_target = pd.DataFrame({
        "target_id": labels_test,
        "error_pixels": errors,
        "within_radius": within_radius
    }).groupby("target_id").agg(
        test_readings=("error_pixels", "size"),
        mean_error_pixels=("error_pixels", "mean"),
        median_error_pixels=("error_pixels", "median"),
        within_radius_percent=("within_radius", "mean")
    )
    per_target["within_radius_percent"] *= 100

    print("\nPer-target results:")
    print(per_target.round(2).to_string())

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    per_target.to_csv(BASE_DIR / f"calibration_target_results_{safe_name}.csv")
    return summary


# --------------------------------------------------
# 2. Random stratified split
# --------------------------------------------------

all_indices = np.arange(len(df))
train_random, test_random = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

random_summary = evaluate_split(
    "Random stratified test",
    train_random,
    test_random
)

# --------------------------------------------------
# 3. Global chronological split
# --------------------------------------------------

split_index = int(len(df) * (1 - TEST_SIZE))
train_chronological = np.arange(0, split_index)
test_chronological = np.arange(split_index, len(df))

chronological_summary = evaluate_split(
    "Global chronological test",
    train_chronological,
    test_chronological
)

# --------------------------------------------------
# 4. Target-balanced chronological split
#    For every target, train on its earlier readings
#    and test on its later readings.
# --------------------------------------------------

train_balanced = []
test_balanced = []

for target_id in range(1, 10):
    target_indices = df.index[df["target_id"] == target_id].to_numpy()
    target_indices = target_indices[
        np.argsort(df.loc[target_indices, "timestamp"].to_numpy())
    ]

    target_split = int(len(target_indices) * (1 - TEST_SIZE))
    target_split = min(max(target_split, 1), len(target_indices) - 1)

    train_balanced.extend(target_indices[:target_split])
    test_balanced.extend(target_indices[target_split:])

train_balanced = np.array(sorted(train_balanced), dtype=int)
test_balanced = np.array(sorted(test_balanced), dtype=int)

balanced_summary = evaluate_split(
    "Within-target chronological test",
    train_balanced,
    test_balanced
)

# --------------------------------------------------
# 5. Compare all evaluation methods
# --------------------------------------------------

comparison = pd.DataFrame([
    random_summary,
    chronological_summary,
    balanced_summary
])

comparison.to_csv(BASE_DIR / "calibration_evaluation_comparison.csv", index=False)

print("\n===== SIDE-BY-SIDE COMPARISON =====")
columns_to_show = [
    "method",
    "training_readings",
    "testing_readings",
    "mean_position_error",
    "median_position_error",
    "p90_position_error",
    "within_radius_percent"
]
print(comparison[columns_to_show].round(2).to_string(index=False))

print("\nSaved:")
print("- calibration_evaluation_comparison.csv")
print("- calibration_target_results_random_stratified_test.csv")
print("- calibration_target_results_global_chronological_test.csv")
print("- calibration_target_results_within_target_chronological_test.csv")

print(
    "\nInterpretation: the random split tests held-out readings mixed across "
    "the session. The global chronological split tests only the final part "
    "of the session and may omit some targets. The within-target chronological "
    "split tests later readings for every target, while training on earlier "
    "readings for each target. These are diagnostic tests within one session, "
    "not proof of accuracy for new users or independent sessions."
)
