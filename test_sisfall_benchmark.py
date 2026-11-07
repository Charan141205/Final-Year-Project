#!/usr/bin/env python3
"""
SisFall Benchmark Evaluation Script for Smart Insole Fall Detector.

Evaluates the Orientation-Invariant Motion Engine on the SisFall public benchmark:
- 15 Fall Types (F01 to F15) across 38 subjects (including 15 elderly adults).
- 19 ADL Types (D01 to D19) including walking, running, jumping, stairs, stumbling.

Converts SisFall ADXL345 (accel) and ITG3200 (gyro) from 200 Hz to 50 Hz,
computes 3D vector norms (SVM, GM, Jerk), and evaluates classification accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# SisFall Sensor Scaling Factors
# ADXL345: +/- 16g range, 13-bit resolution (4 mg/LSB) -> m/s^2
ADXL_SCALE = (32.0 / 8192.0) * 9.80665

# ITG3200: +/- 2000 deg/s range, 14.375 LSB/(deg/s) -> rad/s
ITG_SCALE = (1.0 / 14.375) * (np.pi / 180.0)

# 50 Hz configuration
TARGET_FREQ_HZ = 50
SOURCE_FREQ_HZ = 200
DOWNSAMPLE_STEP = SOURCE_FREQ_HZ // TARGET_FREQ_HZ # 4
WINDOW_SAMPLES = 50 # 1.0 second at 50 Hz
STEP_SAMPLES = 25   # 50% overlap (0.5 second)


def parse_sisfall_file(file_path: Path) -> pd.DataFrame | None:
    """Read a raw SisFall ASCII text file and return clean 50 Hz IMU signals."""
    if file_path.stat().st_size < 1000:
        # Skip empty or macOS metadata AppleDouble resource forks
        return None

    try:
        # SisFall files contain comma-delimited integers (lines often end with semicolon/comma)
        raw_lines = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                cleaned = line.strip().rstrip(";,").strip()
                if cleaned:
                    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                    if len(parts) >= 6:
                        raw_lines.append([float(x) for x in parts[:6]])

        if not raw_lines:
            return None

        arr = np.array(raw_lines, dtype=np.float32)

        # Columns: ADXL345 (Ax, Ay, Az), ITG3200 (Gx, Gy, Gz)
        ax = arr[:, 0] * ADXL_SCALE
        ay = arr[:, 1] * ADXL_SCALE
        az = arr[:, 2] * ADXL_SCALE

        gx = arr[:, 3] * ITG_SCALE
        gy = arr[:, 4] * ITG_SCALE
        gz = arr[:, 5] * ITG_SCALE

        # Downsample from 200 Hz to 50 Hz
        df = pd.DataFrame({
            "ax": ax[::DOWNSAMPLE_STEP],
            "ay": ay[::DOWNSAMPLE_STEP],
            "az": az[::DOWNSAMPLE_STEP],
            "gx": gx[::DOWNSAMPLE_STEP],
            "gy": gy[::DOWNSAMPLE_STEP],
            "gz": gz[::DOWNSAMPLE_STEP],
        })
        return df

    except Exception as e:
        return None


def evaluate_trial(df_50hz: pd.DataFrame) -> bool:
    """Evaluate sliding windows with orientation-invariant thresholds and stillness check."""
    ax = df_50hz["ax"].to_numpy()
    ay = df_50hz["ay"].to_numpy()
    az = df_50hz["az"].to_numpy()
    gx = df_50hz["gx"].to_numpy()
    gy = df_50hz["gy"].to_numpy()
    gz = df_50hz["gz"].to_numpy()

    # Instantaneous 3D Vector Norms
    svm = np.sqrt(ax**2 + ay**2 + az**2)
    gm = np.sqrt(gx**2 + gy**2 + gz**2)

    total_samples = len(svm)
    if total_samples < WINDOW_SAMPLES:
        return False

    window_triggers = []

    # Slide 50-sample windows with 50% overlap
    for w_start in range(0, total_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        w_end = w_start + WINDOW_SAMPLES

        w_svm = svm[w_start:w_end]
        w_gm = gm[w_start:w_end]

        svm_max = np.max(w_svm)
        svm_min = np.min(w_svm)
        gm_max = np.max(w_gm)
        jerk_max = np.max(np.abs(np.diff(w_svm))) / 0.02 if len(w_svm) > 1 else 0.0

        # Orientation-Invariant Trigger: Impact peak + Pre-fall drop + Tumbling rotation
        is_candidate = (
            (svm_max > 22.0) and          # Impact > 2.24 g
            (svm_min < 6.5) and           # Free-fall descent drop
            (gm_max > 1.20) and           # Rotational tumbling speed
            (jerk_max > 150.0)            # High dynamic rise rate
        )
        window_triggers.append((is_candidate, w_end))

    # Multi-Stage Temporal Confirmation: Impact followed by post-impact stillness
    for i, (is_impact, end_sample) in enumerate(window_triggers):
        if is_impact:
            # Check stillness in subsequent window (victim lying still on mattress)
            post_start = end_sample
            post_end = min(post_start + WINDOW_SAMPLES, total_samples)

            if post_end - post_start >= 25:
                post_svm = svm[post_start:post_end]
                post_gm = gm[post_start:post_end]

                post_svm_std = np.std(post_svm)
                post_gm_mean = np.mean(post_gm)

                # Stillness condition: low acceleration variance and low rotation
                if post_svm_std < 1.4 and post_gm_mean < 1.0:
                    return True # Confirmed fall!

    return False


def run_sisfall_benchmark(dataset_root: Path):
    """Scan and evaluate all SisFall subjects and trials."""
    dataset_root = Path(dataset_root)
    print(f"\nScanning SisFall dataset directory: {dataset_root}")

    # Discover all txt files
    all_files = list(dataset_root.rglob("*.txt"))
    # Filter out AppleDouble metadata files
    valid_files = [f for f in all_files if not f.name.startswith("._") and f.stat().st_size > 1000]

    print(f"Total files discovered: {len(all_files):,}")
    print(f"Valid data files (>1 KB, non-metadata): {len(valid_files):,}")

    if len(valid_files) == 0:
        print("\n[!] NOTICE: All discovered files are AppleDouble resource forks (size ~211 bytes).")
        print("    The actual SisFall raw sensor data files have not been extracted into this folder.")
        print("    Please see the verification report for extraction and theoretical analysis.")
        return

    y_true = []
    y_pred = []
    trial_records = []

    print("\nProcessing SisFall trials...")
    for idx, path in enumerate(valid_files, 1):
        filename = path.name
        # SisFall naming convention: Dxx_SAxx_Rxx (ADL) or Fxx_SAxx_Rxx (Fall)
        is_fall = 1 if filename.startswith("F") else 0

        df_50hz = parse_sisfall_file(path)
        if df_50hz is None:
            continue

        predicted = 1 if evaluate_trial(df_50hz) else 0

        y_true.append(is_fall)
        y_pred.append(predicted)
        trial_records.append({
            "file": filename,
            "ground_truth": "fall" if is_fall else "adl",
            "prediction": "fall" if predicted else "adl",
            "correct": is_fall == predicted
        })

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(valid_files)} trials...")

    # Compute Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / len(y_true)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n================ SISFALL BENCHMARK RESULTS ================")
    print(f"Total Trials Evaluated: {len(y_true):,}")
    print(f"Overall Accuracy:        {accuracy * 100:.2f}%")
    print(f"Fall Sensitivity/Recall: {recall * 100:.2f}% (TP={tp}, FN={fn})")
    print(f"ADL Specificity:         {(1 - fpr) * 100:.2f}% (TN={tn}, FP={fp})")
    print(f"Precision:               {precision * 100:.2f}%")
    print(f"F1-Score:                {f1 * 100:.2f}%")
    print(f"False Positive Rate:     {fpr * 100:.2f}%")
    print("===========================================================")

    # Save summary CSV
    res_df = pd.DataFrame(trial_records)
    res_df.to_csv("sisfall_benchmark_results.csv", index=False)
    print("Saved trial-level results to sisfall_benchmark_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SisFall benchmark.")
    parser.add_argument("--data-dir", type=str, default=r"C:\Users\lagis\Downloads\SisFall_dataset")
    args = parser.parse_args()

    run_sisfall_benchmark(Path(args.data_dir))
