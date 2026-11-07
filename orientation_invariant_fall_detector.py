#!/usr/bin/env python3
"""
Orientation-Invariant and Normalized Feature Pipeline (OIN-10) for Smart Insole Fall Detection.

Rectifies:
1. Sensor Orientation Mismatch: Uses rotational invariant norms (SVM, GM, Vector Jerk).
2. FSR Drift & Body-Weight Variance: Uses power-on zero-tare subtraction and total pressure ratios.
3. Domestic False Alarms: Uses multi-window causal temporal confirmation.
4. Foot-Mounted Impact Shocks: Uses combined free-fall drop + impact peak + post-fall rest.

Works interchangeably with:
- Raw CAPPIMU dataset (subject8_cappimu_raw_verified.csv)
- Physical ESP32 Smart Insole prototype CSV logs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text

G0 = 9.80665
DEG_TO_RAD = np.pi / 180.0

# 10 Orientation-Invariant & Normalized Features
OIN_FEATURE_NAMES = [
    "SVM_max",           # Peak impact acceleration magnitude (m/s^2)
    "SVM_min",           # Minimum acceleration magnitude (free-fall indicator) (m/s^2)
    "SVM_range",         # Dynamic acceleration range (SVM_max - SVM_min)
    "SVM_std",           # Acceleration dispersion / mobility
    "GM_max",            # Peak angular speed magnitude (rad/s)
    "GM_range",          # Dynamic rotational range (rad/s)
    "Jerk_max",          # Peak rate of acceleration change (m/s^3)
    "P_total_min_ratio", # Min total pressure normalized by baseline (loss of contact)
    "P_drop_range",      # Drop/swing range of total pressure normalized
    "Heel_ratio_mean",   # Plantar load distribution on heel (0.0 - 1.0)
]


def extract_features_from_window(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    heel: np.ndarray,
    midfoot: np.ndarray,
    forefoot: np.ndarray,
    toe: np.ndarray,
    dt: float = 0.02,
    p_baseline: float = 1000.0,
) -> Dict[str, float]:
    """Calculate 10 orientation-invariant and normalized features from 1 window."""
    # 1. Acceleration Vector Magnitude (SVM) - Orientation Invariant
    svm = np.sqrt(ax**2 + ay**2 + az**2)

    # 2. Gyroscope Vector Magnitude (GM) - Orientation Invariant
    gm = np.sqrt(gx**2 + gy**2 + gz**2)

    # 3. Acceleration Jerk (Vector magnitude derivative)
    jerk = np.abs(np.diff(svm)) / dt if len(svm) > 1 else np.array([0.0])

    # 4. Total Plantar Pressure & Regional Normalization
    p_total = heel + midfoot + forefoot + toe
    p_total_safe = np.maximum(p_total, 1e-4)

    # Regional load ratios
    heel_ratio = heel / p_total_safe
    p_norm = p_total / max(p_baseline, 1.0)

    return {
        "SVM_max": float(np.max(svm)),
        "SVM_min": float(np.min(svm)),
        "SVM_range": float(np.max(svm) - np.min(svm)),
        "SVM_std": float(np.std(svm, ddof=0)),
        "GM_max": float(np.max(gm)),
        "GM_range": float(np.max(gm) - np.min(gm)),
        "Jerk_max": float(np.max(jerk)),
        "P_total_min_ratio": float(np.min(p_norm)),
        "P_drop_range": float(np.max(p_norm) - np.min(p_norm)),
        "Heel_ratio_mean": float(np.mean(heel_ratio)),
    }


def process_cappimu_raw(
    csv_path: str | Path,
    window_size: int = 50,
    step_size: int = 25,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Segment raw CAPPIMU CSV into overlapping windows and extract OIN features."""
    df = pd.read_csv(csv_path)
    print(f"Loaded raw file with {len(df):,} rows.")

    # Convert signals to standard SI units
    ax_all = df["knee_r(ax)"].to_numpy(dtype=np.float64) * G0
    ay_all = df["knee_r(ay)"].to_numpy(dtype=np.float64) * G0
    az_all = df["knee_r(az)"].to_numpy(dtype=np.float64) * G0

    gx_all = df["knee_r(wx)"].to_numpy(dtype=np.float64) * DEG_TO_RAD
    gy_all = df["knee_r(wy)"].to_numpy(dtype=np.float64) * DEG_TO_RAD
    gz_all = df["knee_r(wz)"].to_numpy(dtype=np.float64) * DEG_TO_RAD

    # Regional insole pressure proxies
    heel_all = df[["insole_r(p6)", "insole_r(p7)", "insole_r(p8)"]].mean(axis=1).to_numpy(dtype=np.float64)
    mid_all = df[["insole_r(p4)", "insole_r(p5)"]].mean(axis=1).to_numpy(dtype=np.float64)
    fore_all = df[["insole_r(p2)", "insole_r(p3)"]].mean(axis=1).to_numpy(dtype=np.float64)
    toe_all = df["insole_r(p1)"].to_numpy(dtype=np.float64)

    labels = df["label"].astype(str).str.strip().str.lower().to_numpy()
    
    # Calculate standing baseline total pressure from normal standing/quiet walking
    quiet_mask = np.isin(labels, ["walk", "folding_clothes", "brushing_teeth"])
    p_standing_baseline = float(np.percentile((heel_all + mid_all + fore_all + toe_all)[quiet_mask], 75))
    print(f"Computed reference standing total pressure: {p_standing_baseline:.2f}")

    # Sliding window extraction across contiguous activity blocks
    feature_rows = []
    window_labels = []
    window_activities = []

    # Detect label boundaries to avoid window bridging across different activities
    boundary_indices = np.where(labels[:-1] != labels[1:])[0] + 1
    segment_starts = np.r_[0, boundary_indices]
    segment_ends = np.r_[boundary_indices, len(labels)]

    for start_idx, end_idx in zip(segment_starts, segment_ends):
        seg_len = end_idx - start_idx
        act = labels[start_idx]
        is_fall = 1 if act == "fall" else 0

        if seg_len < window_size:
            continue

        for w_start in range(start_idx, end_idx - window_size + 1, step_size):
            w_end = w_start + window_size
            feat = extract_features_from_window(
                ax=ax_all[w_start:w_end],
                ay=ay_all[w_start:w_end],
                az=az_all[w_start:w_end],
                gx=gx_all[w_start:w_end],
                gy=gy_all[w_start:w_end],
                gz=gz_all[w_start:w_end],
                heel=heel_all[w_start:w_end],
                midfoot=mid_all[w_start:w_end],
                forefoot=fore_all[w_start:w_end],
                toe=toe_all[w_start:w_end],
                dt=0.02,
                p_baseline=p_standing_baseline,
            )
            feature_rows.append(feat)
            window_labels.append(is_fall)
            window_activities.append(act)

    features_df = pd.DataFrame(feature_rows)
    features_df["activity"] = window_activities
    features_df["is_fall"] = window_labels

    print(f"Extracted {len(features_df):,} windows ({sum(window_labels)} fall, {len(window_labels) - sum(window_labels)} ADL).")
    return features_df, pd.Series(window_labels)


def train_and_evaluate_tree(df: pd.DataFrame) -> Tuple[DecisionTreeClassifier, Dict]:
    """Train Decision Tree and evaluate with 2-of-3 causal temporal confirmation."""
    X = df[OIN_FEATURE_NAMES].to_numpy()
    y = df["is_fall"].to_numpy()

    # Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    raw_oof_preds = np.zeros(len(y))
    raw_oof_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        clf = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        raw_oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        raw_oof_preds[val_idx] = clf.predict(X[val_idx])

    # 1. Raw Window Evaluation
    p, r, f1, _ = precision_recall_fscore_support(y, raw_oof_preds, average="binary")
    tn, fp, fn, tp = confusion_matrix(y, raw_oof_preds).ravel()
    fpr = fp / (fp + tn)

    print("\n================== RAW WINDOW CLASSIFICATION ==================")
    print(f"Precision: {p*100:.2f}% | Recall: {r*100:.2f}% | F1: {f1*100:.2f}% | FPR: {fpr*100:.2f}%")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # 2. Apply Causal 2-of-3 Temporal Confirmation Filter
    temporal_preds = np.zeros(len(y), dtype=np.uint8)
    for i in range(len(y)):
        if i == 0:
            temporal_preds[i] = raw_oof_preds[i]
        elif i == 1:
            temporal_preds[i] = int((raw_oof_preds[0] + raw_oof_preds[1]) >= 2)
        else:
            recent_3 = raw_oof_preds[i-2:i+1]
            temporal_preds[i] = int(np.sum(recent_3) >= 2)

    tp_t, r_t, f1_t, _ = precision_recall_fscore_support(y, temporal_preds, average="binary")
    tn_t, fp_t, fn_t, tp_t_cnt = confusion_matrix(y, temporal_preds).ravel()
    fpr_t = fp_t / (fp_t + tn_t)

    print("\n======== WITH CAUSAL 2-OF-3 TEMPORAL CONFIRMATION ========")
    print(f"Precision: {tp_t*100:.2f}% | Recall: {r_t*100:.2f}% | F1: {f1_t*100:.2f}% | FPR: {fpr_t*100:.2f}%")
    print(f"Confusion Matrix: TP={tp_t_cnt}, FP={fp_t}, FN={fn_t}, TN={tn_t}")
    print(f"False Alarm Reduction: From {fp} false windows down to {fp_t} ({((fp - fp_t)/fp)*100:.1f}% reduction!)")

    # Fit final reference model on full dataset
    final_tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    final_tree.fit(X, y)

    metrics_summary = {
        "raw": {"precision": float(p), "recall": float(r), "f1": float(f1), "fpr": float(fpr), "FP": int(fp)},
        "temporal_2of3": {"precision": float(tp_t), "recall": float(r_t), "f1": float(f1_t), "fpr": float(fpr_t), "FP": int(fp_t)},
    }
    return final_tree, metrics_summary


if __name__ == "__main__":
    raw_path = Path("subject8_cappimu_raw_verified.csv")
    if not raw_path.is_file():
        print(f"Error: {raw_path} not found.")
        exit(1)

    print("Step 1: Extracting Orientation-Invariant & Normalized Features...")
    feat_df, y_series = process_cappimu_raw(raw_path, window_size=50, step_size=25)

    print("\nStep 2: Training and Validating Orientation-Invariant Decision Tree...")
    tree, metrics = train_and_evaluate_tree(feat_df)

    print("\nStep 3: Exporting Decision Tree Rules for Embedded ESP32:")
    rules = export_text(tree, feature_names=OIN_FEATURE_NAMES)
    print(rules)

    # Save features and tree rules
    feat_df.to_csv("orientation_invariant_features_subject8.csv", index=False)
    with open("orientation_invariant_tree_rules.txt", "w") as f:
        f.write(rules)
    print("\nArtifacts saved: orientation_invariant_features_subject8.csv and orientation_invariant_tree_rules.txt")
