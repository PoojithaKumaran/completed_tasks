import os
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. File configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "gaze_interactions.csv"
OUTPUT_FILE = BASE_DIR / "gaze_interactions_cleaned.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "user_id",
    "gaze_x",
    "gaze_y",
    "target_under_gaze",
    "expected_target",
    "fixation_time",
    "blink",
    "selection",
    "selection_success"
]

NUMERIC_COLUMNS = [
    "gaze_x",
    "gaze_y",
    "target_under_gaze",
    "expected_target",
    "fixation_time",
    "selection"
]


# --------------------------------------------------
# 2. Load and validate the input file
# --------------------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. "
        "Keep this script in your project folder."
    )

try:
    df = pd.read_csv(INPUT_FILE)
except (pd.errors.ParserError, UnicodeDecodeError) as error:
    raise ValueError(
        f"Could not read {INPUT_FILE}: {error}"
    ) from error

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

original_count = len(df)

# Keep the expected columns and preserve the original file.
df = df[REQUIRED_COLUMNS].copy()

print("\n===== GAZE DATASET CLEANING =====")
print("Original records:", original_count)


# --------------------------------------------------
# 3. Normalize text fields
# --------------------------------------------------

for column in ["timestamp", "user_id", "blink", "selection_success"]:
    df[column] = df[column].astype("string").str.strip()

df["blink"] = df["blink"].str.lower()
df["selection_success"] = (
    df["selection_success"].str.lower()
)


# --------------------------------------------------
# 4. Convert numeric fields
# --------------------------------------------------

for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

parsed_timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)


# --------------------------------------------------
# 5. Remove missing or invalid records
# --------------------------------------------------

essential_columns = [
    "timestamp",
    "user_id",
    "gaze_x",
    "gaze_y",
    "target_under_gaze",
    "expected_target",
    "fixation_time",
    "selection",
    "selection_success"
]

valid = df[essential_columns].notna().all(axis=1)
valid &= parsed_timestamps.notna()

# Required text fields must not be empty.
valid &= df["timestamp"].str.len().gt(0).fillna(False)
valid &= df["user_id"].str.len().gt(0).fillna(False)

# Coordinates must fall within the screen dimensions.
valid &= df["gaze_x"].between(0, 999)
valid &= df["gaze_y"].between(0, 649)

# Target IDs and selections must be whole numbers from 1 to 9.
for column in [
    "target_under_gaze",
    "expected_target",
    "selection"
]:
    valid &= df[column].isin(range(1, 10))

# Fixation time cannot be negative.
valid &= df["fixation_time"].ge(0)

# Outcomes must be explicitly labelled yes or no.
valid &= df["selection_success"].isin(["yes", "no"])

before_validation = len(df)
df = df.loc[valid].copy()

invalid_removed = before_validation - len(df)

print("Invalid or incomplete records removed:", invalid_removed)


# --------------------------------------------------
# 6. Handle blink labels
# --------------------------------------------------

# Blink information is useful, but an invalid blink label
# should not automatically discard an otherwise valid selection.

valid_blink_labels = ["yes", "no"]

invalid_blink = ~df["blink"].isin(valid_blink_labels)
blink_labels_repaired = int(invalid_blink.sum())

df.loc[invalid_blink, "blink"] = "unknown"

print("Missing/invalid blink labels marked unknown:",
      blink_labels_repaired)


# --------------------------------------------------
# 7. Remove exact duplicate records
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates().copy()

duplicates_removed = before_duplicates - len(df)

print("Exact duplicate records removed:", duplicates_removed)


# --------------------------------------------------
# 8. Check selection-label consistency
# --------------------------------------------------

# The main program labels a selection successful when
# the selected target matches the expected target.

consistent_labels = (
    df["selection_success"]
    == (df["selection"] == df["expected_target"]).map({
        True: "yes",
        False: "no"
    })
)

inconsistent_count = int((~consistent_labels).sum())

print("Records with inconsistent success labels:",
      inconsistent_count)

if inconsistent_count:
    print(
        "WARNING: Review these records before interpreting "
        "the evaluation results."
    )


# --------------------------------------------------
# 9. Save the cleaned dataset
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n===== CLEANING SUMMARY =====")
print("Original records:", original_count)
print("Final records:", len(df))
print("Total records removed:",
      original_count - len(df))
print("Cleaned dataset saved to:", OUTPUT_FILE)

if df.empty:
    print(
        "\nWARNING: No valid records remain. "
        "Check the input dataset before evaluation."
    )
else:
    print("Cleaning completed successfully.")
