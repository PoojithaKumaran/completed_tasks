import pandas as pd
import os

FILE = "calibration_data.csv"

if not os.path.exists(FILE):
    raise FileNotFoundError(f"{FILE} not found.")

df = pd.read_csv(FILE)

required = [
    "timestamp", "target_id", "target_x", "target_y",
    "left_iris_x", "left_iris_y",
    "right_iris_x", "right_iris_y"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["target_id"] = pd.to_numeric(df["target_id"], errors="coerce")

df = df.dropna(subset=["timestamp", "target_id"]).reset_index(drop=True)

print("\n===== CALIBRATION DATASET INSPECTION =====")
print("Total usable records:", len(df))
print("Unique target IDs:", sorted(df["target_id"].unique().tolist()))

if df.empty:
    raise ValueError("No usable calibration records found.")

# Match the session-splitting approach in the main program.
session_ids = (
    df["timestamp"].diff().gt(30).fillna(False).cumsum()
)

df["session"] = session_ids

print("\n===== SESSION SUMMARY =====")

for session_id, group in df.groupby("session"):
    targets = sorted(group["target_id"].unique().tolist())

    print(f"\nSession {session_id + 1}")
    print("  Records:", len(group))
    print("  Target IDs:", targets)
    print("  All 9 targets covered:", set(range(1, 10)).issubset(targets))

latest_session = df[df["session"] == df["session"].iloc[-1]]

print("\n===== LATEST SESSION =====")
print("Records:", len(latest_session))
print(
    "Targets covered:",
    sorted(latest_session["target_id"].unique().tolist())
)
print(
    "All 9 targets covered:",
    set(range(1, 10)).issubset(
        set(latest_session["target_id"].astype(int))
    )
)