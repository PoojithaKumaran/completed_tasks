"""Summarize held-out gaze predictions against each known target center."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SPLITS = {
    "random_stratified": BASE_DIR / "diagnostic_predictions_random_stratified.csv",
    "global_chronological": BASE_DIR / "diagnostic_predictions_global_chronological.csv",
    "within_target_chronological": BASE_DIR / "diagnostic_predictions_within_target_chronological.csv",
}

reports = []
for split_name, path in SPLITS.items():
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run diagnose_calibration.py first.")
    predictions = pd.read_csv(path)
    required = {
        "target_id", "actual_x", "actual_y", "predicted_x", "predicted_y",
        "error_x", "error_y", "position_error"
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    report = predictions.groupby("target_id").agg(
        held_out_samples=("position_error", "size"),
        actual_x=("actual_x", "mean"),
        actual_y=("actual_y", "mean"),
        predicted_x=("predicted_x", "mean"),
        predicted_y=("predicted_y", "mean"),
        bias_x=("error_x", "mean"),
        bias_y=("error_y", "mean"),
        mean_error_px=("position_error", "mean"),
        median_error_px=("position_error", "median"),
        p90_error_px=("position_error", lambda values: np.percentile(values, 90)),
    ).reset_index()
    report.insert(0, "split", split_name)
    for col in report.select_dtypes(include="number").columns:
        if col not in {"target_id", "held_out_samples"}:
            report[col] = report[col].round(2)
    reports.append(report)

combined = pd.concat(reports, ignore_index=True)
out = BASE_DIR / "calibration_target_check.csv"
combined.to_csv(out, index=False)
print(combined.to_string(index=False))
print(f"\nSaved target check: {out}")
print("Bias is predicted position minus labeled target center; values are pixels.")
