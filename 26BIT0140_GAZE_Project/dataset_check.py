
import csv
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "calibration_data.csv")

with open(CSV_PATH, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("DATASET QUALITY REPORT")
print("-" * 35)
print("Total readings:", len(rows))

print("\nReadings per target:")
counts = Counter(row["target_id"] for row in rows)

for target_id in sorted(counts, key=int):
    print(f"Target {target_id}: {counts[target_id]} readings")

required_columns = [
    "timestamp", "target_id", "target_x", "target_y",
    "left_iris_x", "left_iris_y",
    "right_iris_x", "right_iris_y"
]

missing_count = 0
invalid_count = 0

for row in rows:
    if any(not row.get(col, "").strip() for col in required_columns):
        missing_count += 1
        continue

    try:
        for col in required_columns:
            float(row[col])
    except ValueError:
        invalid_count += 1

print("\nMissing-value rows:", missing_count)
print("Invalid-value rows:", invalid_count)

print("\nDataset check complete.")
