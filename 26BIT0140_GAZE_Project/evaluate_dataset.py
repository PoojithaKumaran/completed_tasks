
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "gaze_interactions_cleaned.csv"
INTERRUPTION_FILE = BASE_DIR / "gaze_interruption_log.csv"
REPORT_FILE = BASE_DIR / "participant_evaluation_report.txt"

# 1. Load the cleaned dataset
if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    raise SystemExit

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "user_id", "fixation_time", "selection_success"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    print("Missing required columns:", missing)
    raise SystemExit

# 2. Exclude deliberate software-test records
df["user_id"] = df["user_id"].astype(str).str.strip()
df = df[df["user_id"].str.lower() != "user_005"].copy()

if df.empty:
    print("No participant records available.")
    raise SystemExit

# 3. Clean evaluation fields
df["fixation_time"] = pd.to_numeric(
    df["fixation_time"], errors="coerce"
)

df["selection_success"] = (
    df["selection_success"]
    .astype(str).str.strip().str.lower()
)

df = df.dropna(subset=["user_id", "fixation_time"])
df = df[df["fixation_time"] >= 0]

if df.empty:
    print("No valid records available for evaluation.")
    raise SystemExit

# 4. Calculate completed outcomes and include logged interrupted dwells.
df["is_successful"] = df["selection_success"].eq("yes")
completed = df.groupby("user_id").agg(
    completed_selections=("is_successful", "size"),
    successful=("is_successful", "sum"),
    average_fixation_time_seconds=("fixation_time", "mean"),
)
completed["successful"] = completed["successful"].astype(int)

if INTERRUPTION_FILE.exists():
    interruptions = pd.read_csv(INTERRUPTION_FILE)
    required_interrupt = {"user_id", "event", "dwell_time"}
    missing_interrupt = required_interrupt - set(interruptions.columns)
    if missing_interrupt:
        raise ValueError(f"Interruption log missing columns: {sorted(missing_interrupt)}")
    interruptions["user_id"] = interruptions["user_id"].astype(str).str.strip()
    interruptions = interruptions[interruptions["user_id"].str.lower() != "user_005"]
    interruptions["dwell_time"] = pd.to_numeric(interruptions["dwell_time"], errors="coerce")
    interruptions = interruptions.dropna(subset=["user_id", "dwell_time", "event"])
    interruptions = interruptions[interruptions["dwell_time"] >= 0]
    interrupted = interruptions.groupby("user_id").size().rename("interrupted_dwells")
else:
    interrupted = pd.Series(dtype="int64", name="interrupted_dwells")

summary = completed.join(interrupted, how="outer").fillna(0)
for column in ("completed_selections", "successful", "interrupted_dwells"):
    summary[column] = summary[column].astype(int)
summary["unsuccessful_completed"] = summary["completed_selections"] - summary["successful"]
summary["total_logged_attempts"] = summary["completed_selections"] + summary["interrupted_dwells"]
summary["unsuccessful_attempts"] = summary["unsuccessful_completed"] + summary["interrupted_dwells"]
summary["attempt_success_rate_percent"] = np.where(
    summary["total_logged_attempts"] > 0,
    summary["successful"] / summary["total_logged_attempts"] * 100,
    0.0,
)

total_completed = int(summary["completed_selections"].sum())
successful = int(summary["successful"].sum())
interrupted_total = int(summary["interrupted_dwells"].sum())
total_attempts = int(summary["total_logged_attempts"].sum())
unsuccessful = total_attempts - successful
success_rate = successful / total_attempts * 100 if total_attempts else 0.0
average_time = df["fixation_time"].mean()

# 6. Display results
print("\n===== GAZE DATASET EVALUATION =====")
print("Participants:", summary.index.nunique())
print("Completed selections:", total_completed)
print("Recorded successful selections:", successful)
print("Interrupted dwell attempts:", interrupted_total)
print("Total logged attempts:", total_attempts)
print("Unsuccessful attempts (including interruptions):", unsuccessful)
print(f"Attempt success rate: {success_rate:.2f}%")
print(f"Average fixation time: {average_time:.3f} seconds")

print("\n===== RESULTS BY USER =====")
print(summary.round(2).to_string())

# 7. Save participant summary
summary.to_csv(BASE_DIR / "user_evaluation_summary.csv")
print("\nSaved: user_evaluation_summary.csv")

# 8. Generate success-rate chart
ax = summary["attempt_success_rate_percent"].plot(
    kind="bar",
    ylim=(0, 100),
    ylabel="Success rate (%)",
    title="Success Rate Including Interrupted Dwells"
)
ax.set_xlabel("Participant")
plt.tight_layout()
plt.savefig(BASE_DIR / "success_rate_by_user.png", dpi=150)
plt.close()

# 9. Generate average fixation-time chart
ax = summary["average_fixation_time_seconds"].replace(0, float("nan")).plot(
    kind="bar",
    ylabel="Time (seconds)",
    title="Average Fixation Time by User"
)
ax.set_xlabel("Participant")
plt.tight_layout()
plt.savefig(BASE_DIR / "average_time_by_user.png", dpi=150)
plt.close()

print("Saved: success_rate_by_user.png")
print("Saved: average_time_by_user.png")

# 10. Generate the text report
report = f"""
GAZE DATASET - PARTICIPANT EVALUATION REPORT
================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

OVERALL RESULTS
--------------------
Participants: {summary.index.nunique()}
Completed selections: {total_completed}
Recorded successful selections: {successful}
Interrupted dwell attempts: {interrupted_total}
Total logged attempts: {total_attempts}
Unsuccessful attempts including interruptions: {unsuccessful}
Attempt success rate: {success_rate:.2f}%
Average fixation time among completed selections: {average_time:.3f} seconds

RESULTS BY PARTICIPANT
-------------------------
{summary.round(2).to_string()}

INTERPRETATION
---------------
Attempt success rate uses completed selections plus each
valid logged interrupted dwell as the denominator. It is
still limited to attempts represented in these logs; gaze
periods that never started or were not logged are absent.
It does not independently measure gaze-tracking accuracy.

Average fixation time uses completed selections only.

The deliberate user_005 software test was excluded.

OUTPUT FILES
------------
user_evaluation_summary.csv
success_rate_by_user.png
average_time_by_user.png
participant_evaluation_report.txt
gaze_interruption_log.csv (input to interrupted-attempt counts)

END OF REPORT
"""

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write(report.strip() + "\n")

print("Saved:", REPORT_FILE)
print("\nAll evaluation outputs generated successfully.")
