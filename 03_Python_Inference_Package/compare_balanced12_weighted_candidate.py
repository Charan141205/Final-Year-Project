#!/usr/bin/env python3
"""Compare baseline Balanced-12 against the experimental ironing-weighted candidate.

This script performs inference only. It does not retrain or overwrite either model.
The baseline uses threshold 0.8733322266783254; the experimental candidate uses
validation-selected threshold 0.74. Both receive the same causal 2-of-3 warm-start
confirmation rule, reset at recording/segment boundaries and window-ID gaps.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

FEATURES = [
    "FSR1_heel_proxy_range", "FSR2_midfoot_proxy_min", "FSR3_forefoot_proxy_mean", "FSR4_toe_proxy_min",
    "Gz_rad_s_range", "Ay_mps2_rms", "Az_mps2_max", "Gx_rad_s_std", "Ax_mps2_min",
    "Ay_mps2_max", "Az_mps2_mean", "Ax_mps2_rms",
]
GROUPS = ["subject_id", "recording_id", "segment_id"]
BASELINE_THRESHOLD = 0.8733322266783254
CANDIDATE_THRESHOLD = 0.74


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "precision": precision, "recall": recall, "F1": f1, "F2": f2, "FPR": fpr}


def temporal_2of3(frame: pd.DataFrame, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values(GROUPS + ["window_id"], kind="stable")
    original_indices = ordered.index.to_numpy()
    raw_sorted = np.asarray(raw, dtype=np.uint8)[original_indices]
    window_ids = ordered["window_id"].to_numpy(dtype=np.int64)
    states = np.zeros(len(ordered), dtype=np.uint8)
    triggers = np.zeros(len(ordered), dtype=np.uint8)
    grouped = ordered.groupby(GROUPS, sort=False)
    for positions in grouped.indices.values():
        history: list[int] = []
        previous_window: int | None = None
        previous_state = 0
        for pos in np.asarray(positions, dtype=int):
            current_window = int(window_ids[pos])
            if previous_window is None or current_window != previous_window + 1:
                history = []
                previous_state = 0
            history.append(int(raw_sorted[pos]))
            history = history[-3:]
            state = int(len(history) >= 2 and sum(history) >= 2)
            states[pos] = state
            triggers[pos] = int(state == 1 and previous_state == 0)
            previous_state = state
            previous_window = current_window
    state_out = pd.Series(states, index=original_indices).reindex(frame.index).astype(np.uint8).to_numpy()
    trigger_out = pd.Series(triggers, index=original_indices).reindex(frame.index).astype(np.uint8).to_numpy()
    return state_out, trigger_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="All-subjects raw Balanced-12 prediction CSV with 12 feature columns")
    parser.add_argument("--split-map", required=True, help="subject_split.csv")
    parser.add_argument("--baseline-model", required=True, help="Original balanced12_decision_tree.joblib")
    parser.add_argument("--candidate-model", required=True, help="balanced12_ironing_weighted_candidate.joblib")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.predictions)
    missing = [c for c in FEATURES + GROUPS + ["window_id", "activity_label"] if c not in frame.columns]
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {missing}")
    split = pd.read_csv(args.split_map)
    if not {"subject_id", "split"}.issubset(split.columns):
        raise ValueError("Split map must contain subject_id and split")
    if "split" in frame.columns:
        frame = frame.drop(columns=["split"])
    frame = frame.merge(split[["subject_id", "split"]], on="subject_id", how="left", validate="many_to_one")
    if frame["split"].isna().any():
        missing_subjects = sorted(frame.loc[frame.split.isna(), "subject_id"].unique().tolist())
        raise ValueError(f"Split map missing subjects: {missing_subjects}")
    frame["y_true"] = (frame["activity_label"].astype(str).str.lower() == "fall").astype(np.uint8)

    X = frame[FEATURES].to_numpy(dtype=float)
    baseline = joblib.load(args.baseline_model)
    candidate = joblib.load(args.candidate_model)
    if baseline.n_features_in_ != len(FEATURES) or candidate.n_features_in_ != len(FEATURES):
        raise ValueError("Model feature count does not match the required Balanced-12 order")

    baseline_probability = baseline.predict_proba(X)[:, 1]
    candidate_probability = candidate.predict_proba(X)[:, 1]
    baseline_raw = (baseline_probability >= BASELINE_THRESHOLD).astype(np.uint8)
    candidate_raw = (candidate_probability >= CANDIDATE_THRESHOLD).astype(np.uint8)

    # Verify the baseline model reproduces the saved raw classifier column when present.
    if "raw_predicted_binary" in frame.columns:
        saved_raw = frame["raw_predicted_binary"].to_numpy(dtype=np.uint8)
        mismatches = int(np.sum(saved_raw != baseline_raw))
        if mismatches:
            raise ValueError(f"Baseline reproduction check failed: {mismatches} raw prediction mismatches")

    baseline_state, baseline_trigger = temporal_2of3(frame, baseline_raw)
    candidate_state, candidate_trigger = temporal_2of3(frame, candidate_raw)
    frame["baseline_probability_recomputed"] = baseline_probability
    frame["baseline_raw_recomputed"] = baseline_raw
    frame["baseline_2of3"] = baseline_state
    frame["baseline_trigger_2of3"] = baseline_trigger
    frame["candidate_probability"] = candidate_probability
    frame["candidate_raw"] = candidate_raw
    frame["candidate_2of3"] = candidate_state
    frame["candidate_trigger_2of3"] = candidate_trigger

    window_rows = []
    for split_name in ["train", "validation", "test", "all"]:
        sub = frame if split_name == "all" else frame[frame.split == split_name]
        y = sub.y_true.to_numpy(dtype=np.uint8)
        for name, predcol, trigcol in [
            ("baseline_raw", "baseline_raw_recomputed", None),
            ("candidate_raw", "candidate_raw", None),
            ("baseline_2of3", "baseline_2of3", "baseline_trigger_2of3"),
            ("candidate_2of3", "candidate_2of3", "candidate_trigger_2of3"),
        ]:
            row = {"split": split_name, "policy": name, "windows": len(sub), **metrics(y, sub[predcol].to_numpy(dtype=np.uint8))}
            if trigcol:
                row["alert_trigger_events"] = int(sub[trigcol].sum())
            else:
                row["alert_trigger_events"] = np.nan
            window_rows.append(row)
    pd.DataFrame(window_rows).to_csv(outdir / "model_comparison_metrics_by_split.csv", index=False)

    activity_rows = []
    subject_rows = []
    segment_rows = []
    for split_name in ["validation", "test", "all"]:
        sub = frame if split_name == "all" else frame[frame.split == split_name]
        normal = sub[sub.y_true == 0]
        for activity, g in normal.groupby("activity_label"):
            activity_rows.append({
                "split": split_name, "activity_label": activity, "windows": len(g),
                "baseline_alert_events": int(g.baseline_trigger_2of3.sum()),
                "candidate_alert_events": int(g.candidate_trigger_2of3.sum()),
                "baseline_confirmed_windows": int(g.baseline_2of3.sum()),
                "candidate_confirmed_windows": int(g.candidate_2of3.sum()),
            })
        for subject, g in sub.groupby("subject_id"):
            for name, col in [("baseline_2of3", "baseline_2of3"), ("candidate_2of3", "candidate_2of3")]:
                subject_rows.append({"split": split_name, "subject_id": subject, "policy": name,
                                     "windows": len(g), **metrics(g.y_true.to_numpy(), g[col].to_numpy())})
        for name, predcol, triggercol in [
            ("baseline_2of3", "baseline_2of3", "baseline_trigger_2of3"),
            ("candidate_2of3", "candidate_2of3", "candidate_trigger_2of3"),
        ]:
            seg = sub.groupby(GROUPS, sort=False).agg(y_true=("y_true", "max"), any_state=(predcol, "max"), any_trigger=(triggercol, "max")).reset_index()
            fall_seg = seg[seg.y_true == 1]
            normal_seg = seg[seg.y_true == 0]
            segment_rows.append({
                "split": split_name, "policy": name,
                "fall_segments": len(fall_seg), "fall_segments_with_alert": int(fall_seg.any_trigger.sum()),
                "fall_segment_coverage": float(fall_seg.any_trigger.mean()) if len(fall_seg) else np.nan,
                "normal_segments": len(normal_seg), "normal_segments_with_alert": int(normal_seg.any_trigger.sum()),
                "normal_segment_alert_rate": float(normal_seg.any_trigger.mean()) if len(normal_seg) else np.nan,
            })

    pd.DataFrame(activity_rows).to_csv(outdir / "activity_alert_comparison.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(outdir / "subject_metrics.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(outdir / "segment_event_metrics.csv", index=False)
    frame.to_csv(outdir / "all_subjects_baseline_vs_candidate_predictions.csv", index=False)

    summary = {
        "rows": int(len(frame)), "subjects": int(frame.subject_id.nunique()),
        "baseline_threshold": BASELINE_THRESHOLD, "candidate_threshold": CANDIDATE_THRESHOLD,
        "baseline_model": str(Path(args.baseline_model)), "candidate_model": str(Path(args.candidate_model)),
        "baseline_raw_reproduction_mismatches": int(np.sum(frame.baseline_raw_recomputed.to_numpy() != frame.raw_predicted_binary.to_numpy())) if "raw_predicted_binary" in frame.columns else None,
        "inference_uses_activity_label": False,
        "candidate_is_experimental_only": True,
        "note": "CAPPIMU test split has been inspected during development; treat metrics as diagnostic pending fresh independent evaluation."
    }
    (outdir / "model_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BALANCED-12 BASELINE VS WEIGHTED CANDIDATE COMPARISON COMPLETED")
    print("---------------------------------------------------------------")
    print(f"Windows: {len(frame):,} | Subjects: {frame.subject_id.nunique()}")
    print("Baseline reproduction check: exact match")
    print(f"Outputs: {outdir.resolve()}")
    val = pd.DataFrame(window_rows)
    print("\nValidation/test 2-of-3 metrics:")
    print(val[(val.policy.isin(["baseline_2of3", "candidate_2of3"])) & (val.split.isin(["validation", "test"]))][["split", "policy", "precision", "recall", "F2", "FPR", "alert_trigger_events"]].to_string(index=False))

if __name__ == "__main__":
    main()
