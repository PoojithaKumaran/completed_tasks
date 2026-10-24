"""
Batch 1 - Team 4: Dwell-Based Gaze Interaction
Student : Tanmay Gupta (26BIT0206)
Module  : Dwell Detection & Button Selection Evaluation (feeds Team 5 Dataset & Evaluation)

This script analyzes 'dwell_log.csv' (produced by dwell_gui.py and DwellLogger):
  1. Computes selection success rate (%) per dwell setting (3 s, 4 s, 5 s).
  2. Computes average selection / fixation time (s) per dwell setting.
  3. Exports an aggregated summary table to 'analysis_summary.csv'.
  4. Generates publication-ready evaluation bar charts saved to 'analysis_chart.png'.

Usage:
  python analyze_results.py                   # Analyze existing dwell_log.csv
  python analyze_results.py --input my.csv    # Analyze custom CSV
  python analyze_results.py --demo            # Generate demo data and analyze
"""
import argparse
import os
import sys
from typing import Tuple
import numpy as np

# Use Agg backend for headless execution (no DISPLAY needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Expected columns according to README_interface_contract.md:
# user_id, timestamp, gaze_x, gaze_y, target_button, fixation_time, blink, successful_selection, dwell_setting
EXPECTED_COLUMNS = [
    "user_id", "timestamp", "gaze_x", "gaze_y", "target_button",
    "fixation_time", "blink", "successful_selection", "dwell_setting"
]


def generate_synthetic_data(path: str = "dwell_log.csv", user_id: str = "U001_DEMO"):
    """Generate realistic evaluation trials across 3s, 4s, and 5s dwell settings."""
    print(f"Generating synthetic evaluation trials for demonstration in '{path}'...")
    buttons = ["WATER", "FOOD", "HELP", "CALL", "EMERGENCY", "MESSAGE", "ENTERTAIN", "SPEAK", "EXIT"]
    records = []
    base_t = 1700000000.0

    # Simulate 12 attempts per dwell setting:
    # 3s dwell: fast, higher inadvertent triggers / noise (~85% success)
    # 4s dwell: high balance of speed & intentionality (~92% success)
    # 5s dwell: very high accuracy (~98% success), longer time
    configs = [
        {"dwell": 3.0, "success_rate": 0.88, "n": 15, "noise_sd": 0.18},
        {"dwell": 4.0, "success_rate": 0.94, "n": 15, "noise_sd": 0.22},
        {"dwell": 5.0, "success_rate": 0.98, "n": 15, "noise_sd": 0.25},
    ]

    for cfg in configs:
        dwell = cfg["dwell"]
        for _ in range(cfg["n"]):
            base_t += float(np.random.uniform(5.0, 15.0))
            target = str(np.random.choice(buttons))
            is_success = bool(np.random.random() < cfg["success_rate"])

            if is_success:
                fixation = round(float(dwell + abs(np.random.normal(0.08, cfg["noise_sd"]))), 3)
                blink = int(np.random.random() < 0.2)
                gx = round(float(np.random.uniform(200, 800)), 1)
                gy = round(float(np.random.uniform(150, 600)), 1)
                records.append({
                    "user_id": user_id,
                    "timestamp": round(base_t, 3),
                    "gaze_x": gx,
                    "gaze_y": gy,
                    "target_button": target,
                    "fixation_time": fixation,
                    "blink": blink,
                    "successful_selection": 1,
                    "dwell_setting": dwell,
                })
            else:
                # Abandoned attempt before reaching threshold
                abandon_time = round(float(np.random.uniform(0.5, dwell * 0.7)), 3)
                blink = int(np.random.random() < 0.35)
                gx = round(float(np.random.uniform(200, 800)), 1)
                gy = round(float(np.random.uniform(150, 600)), 1)
                records.append({
                    "user_id": user_id,
                    "timestamp": round(base_t, 3),
                    "gaze_x": gx,
                    "gaze_y": gy,
                    "target_button": target,
                    "fixation_time": abandon_time,
                    "blink": blink,
                    "successful_selection": 0,
                    "dwell_setting": dwell,
                })

    df_demo = pd.DataFrame(records)
    df_demo.to_csv(path, index=False)
    print(f"Created '{path}' with {len(df_demo)} trial records.")
    return df_demo


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and validate columns."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise FileNotFoundError(f"Log file '{csv_path}' does not exist or is empty.")

    df = pd.read_csv(csv_path)
    # Check if header matches expected columns
    missing = [c for c in ["fixation_time", "successful_selection", "dwell_setting"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV file is missing essential evaluation columns: {missing}")

    # Ensure correct data types
    df["fixation_time"] = pd.to_numeric(df["fixation_time"], errors="coerce")
    df["successful_selection"] = pd.to_numeric(df["successful_selection"], errors="coerce").fillna(0).astype(int)
    df["dwell_setting"] = pd.to_numeric(df["dwell_setting"], errors="coerce")
    df = df.dropna(subset=["dwell_setting", "fixation_time"])

    # If the CSV contains raw 30ms samples (many rows with successful_selection==0 and tiny fixation times),
    # aggregate to attempts or filter to final rows
    return df


def analyze_dwell_performance(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute success rate and average selection time per dwell setting.
    Returns:
      (summary_df, successful_trials_df)
    """
    # Group by dwell setting
    settings = sorted(df["dwell_setting"].unique())
    summary_rows = []

    for s in settings:
        sub = df[df["dwell_setting"] == s]
        total_attempts = len(sub)
        successful = sub[sub["successful_selection"] == 1]
        success_count = len(successful)
        success_rate = (success_count / total_attempts * 100.0) if total_attempts > 0 else 0.0

        if success_count > 0:
            avg_time = successful["fixation_time"].mean()
            std_time = successful["fixation_time"].std()
            min_time = successful["fixation_time"].min()
            max_time = successful["fixation_time"].max()
        else:
            avg_time, std_time, min_time, max_time = 0.0, 0.0, 0.0, 0.0

        summary_rows.append({
            "dwell_setting_s": float(s),
            "total_attempts": int(total_attempts),
            "successful_selections": int(success_count),
            "abandoned_attempts": int(total_attempts - success_count),
            "success_rate_pct": round(success_rate, 2),
            "avg_selection_time_s": round(avg_time, 3),
            "std_selection_time_s": round(0.0 if np.isnan(std_time) else std_time, 3),
            "min_selection_time_s": round(min_time, 3),
            "max_selection_time_s": round(max_time, 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, df[df["successful_selection"] == 1]


def plot_results(summary_df: pd.DataFrame, output_png: str = "analysis_chart.png"):
    """Generate high-contrast bar charts for Success Rate and Selection Time."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=200)

    # Palette
    bar_color1 = "#3b82f6"  # Blue for success rate
    bar_color2 = "#10b981"  # Emerald green for selection time
    text_color = "#1f2937"
    line_color = "#f43f5e"

    x_labels = [f"{int(s) if s.is_integer() else s} s" for s in summary_df["dwell_setting_s"]]
    x_indices = np.arange(len(x_labels))
    width = 0.48

    # 1. Bar Chart: Success Rate
    bars1 = ax1.bar(x_indices, summary_df["success_rate_pct"], width=width, color=bar_color1, edgecolor="#1d4ed8", linewidth=1.5, zorder=3)
    ax1.set_title("Selection Success Rate by Dwell Setting", fontsize=13, fontweight="bold", pad=12, color=text_color)
    ax1.set_xlabel("Dwell Threshold Setting", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    # Annotate bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2.0,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=text_color,
        )

    # 2. Bar Chart: Average Selection Time
    bars2 = ax2.bar(
        x_indices,
        summary_df["avg_selection_time_s"],
        yerr=summary_df["std_selection_time_s"],
        width=width,
        color=bar_color2,
        edgecolor="#047857",
        linewidth=1.5,
        capsize=5,
        zorder=3,
    )
    ax2.set_title("Average Selection Time by Dwell Setting", fontsize=13, fontweight="bold", pad=12, color=text_color)
    ax2.set_xlabel("Dwell Threshold Setting", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Selection Time (seconds)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, fontsize=10, fontweight="bold")
    max_t = max(summary_df["avg_selection_time_s"].max() * 1.25, 6.0)
    ax2.set_ylim(0, max_t)
    ax2.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    # Annotate bars
    for bar, val in zip(bars2, summary_df["avg_selection_time_s"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.15,
            f"{val:.2f} s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=text_color,
        )

    plt.suptitle("Gaze Dwell Interaction Evaluation — Team 4 (Tanmay Gupta, 26BIT0206)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation chart to '{output_png}'.")


def main():
    parser = argparse.ArgumentParser(description="Analyze Dwell Selection Gaze Evaluation Log (Team 4)")
    parser.add_argument("--input", "-i", default="dwell_log.csv", help="Input CSV path (default: dwell_log.csv)")
    parser.add_argument("--out-csv", default="analysis_summary.csv", help="Output summary table CSV")
    parser.add_argument("--out-png", default="analysis_chart.png", help="Output bar chart image PNG")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic trial dataset and analyze")
    args = parser.parse_args()

    # If demo requested or input file does not exist, generate demo data
    if args.demo or not os.path.exists(args.input):
        if not os.path.exists(args.input):
            print(f"'{args.input}' not found. Initializing with demo evaluation data...")
        generate_synthetic_data(args.input)

    try:
        df = load_and_validate_data(args.input)
    except Exception as err:
        print(f"Error reading dataset: {err}", file=sys.stderr)
        sys.exit(1)

    summary_df, _ = analyze_dwell_performance(df)

    # Print clean table for viva & presentation
    print("\n" + "=" * 80)
    print("DWELL INTERACTION EVALUATION RESULTS (Team 4: 26BIT0206)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Save summary table
    summary_df.to_csv(args.out_csv, index=False)
    print(f"Saved summary table to '{args.out_csv}'.")

    # Generate and save chart
    plot_results(summary_df, args.out_png)
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
