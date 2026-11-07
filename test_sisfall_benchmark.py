#!/usr/bin/env python3
"""
Full Empirical SisFall Benchmark Evaluation for Smart Fall Detection System.

Evaluates all 38 subjects (SA01-SA23 young adults, SE01-SE15 elderly adults):
- 15 Fall Types (F01 to F15)
- 19 ADL Activities (D01 to D19)

Applies:
1. 200 Hz to 50 Hz downsampling.
2. 3D Vector Norms (SVM, GM, Jerk).
3. 4-Phase Fall State Machine:
   - Free-fall drop (SVM < 6.8 m/s^2)
   - Impact shock peak (SVM > 22.0 m/s^2)
   - Rotational tumbling (GM > 1.2 rad/s)
   - Post-impact stillness (SVM_std < 1.4 m/s^2, GM_mean < 1.0 rad/s)
   - Post-impact tilt angle change (theta > 45 deg)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# SisFall Conversion Factors
# ADXL345: +/- 16g, 13-bit -> m/s^2
ADXL_SCALE = (32.0 / 8192.0) * 9.80665

# ITG3200: +/- 2000 deg/s, 16-bit -> rad/s
ITG_SCALE = (4000.0 / 65536.0) * (np.pi / 180.0)

DOWNSAMPLE_STEP = 4  # 200 Hz -> 50 Hz
WINDOW_SAMPLES = 50  # 1.0 s nominal window
STEP_SAMPLES = 25    # 50% overlap (0.5 s)


def parse_sisfall_file(file_path: Path) -> pd.DataFrame | None:
    """Parse raw SisFall text file and return 50 Hz data."""
    try:
        raw_lines = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                cleaned = line.strip().rstrip(";,")
                if cleaned:
                    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                    if len(parts) >= 6:
                        raw_lines.append([float(x) for x in parts[:6]])

        if len(raw_lines) < 100:
            return None

        arr = np.array(raw_lines, dtype=np.float32)

        # Columns: ADXL345 (Ax, Ay, Az), ITG3200 (Gx, Gy, Gz)
        ax = arr[:, 0] * ADXL_SCALE
        ay = arr[:, 1] * ADXL_SCALE
        az = arr[:, 2] * ADXL_SCALE

        gx = arr[:, 3] * ITG_SCALE
        gy = arr[:, 4] * ITG_SCALE
        gz = arr[:, 5] * ITG_SCALE

        # Downsample 200 Hz -> 50 Hz
        return pd.DataFrame({
            "ax": ax[::DOWNSAMPLE_STEP],
            "ay": ay[::DOWNSAMPLE_STEP],
            "az": az[::DOWNSAMPLE_STEP],
            "gx": gx[::DOWNSAMPLE_STEP],
            "gy": gy[::DOWNSAMPLE_STEP],
            "gz": gz[::DOWNSAMPLE_STEP],
        })

    except Exception:
        return None


def evaluate_trial(df_50hz: pd.DataFrame) -> bool:
    """Evaluate sliding windows with 4-phase fall detection and tilt verification."""
    ax = df_50hz["ax"].to_numpy()
    ay = df_50hz["ay"].to_numpy()
    az = df_50hz["az"].to_numpy()
    gx = df_50hz["gx"].to_numpy()
    gy = df_50hz["gy"].to_numpy()
    gz = df_50hz["gz"].to_numpy()

    # 3D Vector Magnitudes
    svm = np.sqrt(ax**2 + ay**2 + az**2)
    gm = np.sqrt(gx**2 + gy**2 + gz**2)

    total_samples = len(svm)
    if total_samples < WINDOW_SAMPLES:
        return False

    # Compute upright standing reference vector from initial 1 second
    cal_len = min(50, total_samples // 4)
    v_ref = np.array([np.mean(ax[:cal_len]), np.mean(ay[:cal_len]), np.mean(az[:cal_len])])
    ref_norm = np.linalg.norm(v_ref)
    if ref_norm > 0.001:
        v_ref /= ref_norm
    else:
        v_ref = np.array([0.0, -1.0, 0.0])

    # Slide 50-sample windows with 50% overlap
    for w_start in range(0, total_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        w_end = w_start + WINDOW_SAMPLES

        w_svm = svm[w_start:w_end]
        w_gm = gm[w_start:w_end]

        svm_max = np.max(w_svm)
        svm_min = np.min(w_svm)
        gm_max = np.max(w_gm)

        # Stage A: Impact shock peak + Free-fall drop + Tumbling rotation
        if (svm_max > 22.0) and (svm_min < 6.8) and (gm_max > 1.20):
            # Stage B: Post-impact stillness in the subsequent 1.0 - 2.0 seconds
            post_start = w_end
            post_end = min(post_start + 75, total_samples) # 1.5 s window

            if post_end - post_start >= 25:
                post_svm = svm[post_start:post_end]
                post_gm = gm[post_start:post_end]

                post_svm_std = np.std(post_svm)
                post_gm_mean = np.mean(post_gm)

                # Stillness condition: low acceleration variance and low rotation
                if post_svm_std < 1.4 and post_gm_mean < 1.0:
                    # Stage C: Tilt Angle Change (Trunk Orientation)
                    v_post = np.array([np.mean(ax[post_start:post_end]),
                                       np.mean(ay[post_start:post_end]),
                                       np.mean(az[post_start:post_end])])
                    post_norm = np.linalg.norm(v_post)
                    if post_norm > 0.001:
                        v_post /= post_norm
                        cos_theta = np.clip(np.dot(v_ref, v_post), -1.0, 1.0)
                        theta_deg = np.degrees(np.arccos(cos_theta))
                    else:
                        theta_deg = 0.0

                    # Confirmed Fall: Orientation tilted significantly from upright (> 40 degrees)
                    if theta_deg > 40.0:
                        return True

    return False


def run_benchmark(dataset_root: Path):
    dataset_root = Path(dataset_root)
    print(f"Scanning SisFall directory: {dataset_root}")

    # Gather all text files across SA and SE folders (ignoring Readme.txt)
    all_files = sorted([f for f in dataset_root.rglob("*.txt") if "_" in f.name and not f.name.startswith("._") and f.stat().st_size > 1000])
    total_files = len(all_files)
    print(f"Discovered {total_files:,} valid raw trial files across 38 subjects.\n")

    if total_files == 0:
        print("No valid data files found.")
        return

    y_true = []
    y_pred = []
    category_results = {}
    subject_results = {}

    start_time = time.time()
    total_samples_processed = 0

    print("Evaluating trials across all 15 Fall categories and 19 ADL categories...")
    for idx, fpath in enumerate(all_files, 1):
        filename = fpath.name
        is_fall = 1 if filename.startswith("F") else 0
        cat_code = filename.split("_")[0]
        subj_code = filename.split("_")[1]

        df = parse_sisfall_file(fpath)
        if df is None:
            continue

        total_samples_processed += len(df)
        pred = 1 if evaluate_trial(df) else 0

        y_true.append(is_fall)
        y_pred.append(pred)

        # Track per-category
        if cat_code not in category_results:
            category_results[cat_code] = {"total": 0, "correct": 0, "type": "Fall" if is_fall else "ADL"}
        category_results[cat_code]["total"] += 1
        if is_fall == pred:
            category_results[cat_code]["correct"] += 1

        # Track per-subject
        if subj_code not in subject_results:
            subject_results[subj_code] = {"total": 0, "correct": 0, "is_elderly": subj_code.startswith("SE")}
        subject_results[subj_code]["total"] += 1
        if is_fall == pred:
            subject_results[subj_code]["correct"] += 1

        if idx % 500 == 0 or idx == total_files:
            elapsed = time.time() - start_time
            print(f"  Processed {idx:,}/{total_files:,} files ({idx/total_files*100:.1f}%) in {elapsed:.1f}s...")

    elapsed_total = time.time() - start_time
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / len(y_true)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "="*60)
    print("        SISFALL BENCHMARK EMPIRICAL TEST RESULTS")
    print("="*60)
    print(f"Total Trials Tested:       {len(y_true):,}")
    print(f"Total Time Taken:          {elapsed_total:.2f} seconds")
    print(f"Throughput:                {len(y_true)/elapsed_total:.1f} trials/second")
    print(f"Average Inference Latency: {(elapsed_total / len(y_true))*1000:.3f} ms per full trial")
    print("-" * 60)
    print(f"OVERALL ACCURACY:          {accuracy * 100:.2f}%")
    print(f"FALL SENSITIVITY (RECALL): {recall * 100:.2f}% ({tp}/{tp+fn} falls detected)")
    print(f"ADL SPECIFICITY:           {specificity * 100:.2f}% ({tn}/{tn+fp} ADLs rejected)")
    print(f"FALL PRECISION:            {precision * 100:.2f}%")
    print(f"F1-SCORE:                  {f1 * 100:.2f}%")
    print("-" * 60)
    print(f"Confusion Matrix: True Positives (TP) = {tp} | False Negatives (FN) = {fn}")
    print(f"                  True Negatives (TN) = {tn} | False Positives (FP) = {fp}")
    print("="*60)

    # Save summary dataframe
    cat_df = pd.DataFrame([
        {"Category": k, "Type": v["type"], "Trials": v["total"], "Correct": v["correct"], "Accuracy": v["correct"]/v["total"]}
        for k, v in sorted(category_results.items())
    ])
    cat_df.to_csv("sisfall_category_performance.csv", index=False)

    subj_df = pd.DataFrame([
        {"Subject": k, "Group": "Elderly (SE)" if v["is_elderly"] else "Young (SA)", "Trials": v["total"], "Correct": v["correct"], "Accuracy": v["correct"]/v["total"]}
        for k, v in sorted(subject_results.items())
    ])
    subj_df.to_csv("sisfall_subject_performance.csv", index=False)
    print("\nDetailed breakdown exported to sisfall_category_performance.csv and sisfall_subject_performance.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=r"c:\Users\lagis\Downloads\SisFall\SisFall_dataset")
    args = parser.parse_args()
    run_benchmark(Path(args.dir))
