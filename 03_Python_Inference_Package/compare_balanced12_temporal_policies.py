#!/usr/bin/env python3
"""Compare fixed Balanced-12 temporal policies on an exported prediction CSV.

The classifier probabilities, predictions, and threshold are not changed.
Policies:
  * raw_model: raw_predicted_binary
  * 2_consecutive: second and later adjacent positive windows
  * 2_of_3_warm_start: >=2 positives among up to the latest 3 adjacent windows;
    at the beginning of a segment, use the available 2 observations once both
    exist rather than waiting for a third observation
  * 3_consecutive: third and later adjacent positive windows

Temporal state resets at subject/recording/segment boundaries and on a missing
window_id. The script writes window, segment, activity, and subject metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

GROUP_KEYS = ["subject_id", "recording_id", "segment_id"]
REQUIRED_PREDICTION_COLUMNS = GROUP_KEYS + [
    "window_id", "activity_label", "raw_predicted_binary",
    "confirmed_fall", "alert_trigger",
]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "precision": precision, "recall": recall,
        "F1": f1, "F2": f2, "FPR": fpr, "accuracy": accuracy,
    }


def add_temporal_policies(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        GROUP_KEYS + ["window_id"], kind="stable"
    ).copy()
    row_positions = ordered.index.to_numpy()
    raw = ordered["raw_predicted_binary"].to_numpy(dtype=np.uint8)
    window_ids = ordered["window_id"].to_numpy(dtype=np.int64)

    strict2 = np.zeros(len(ordered), dtype=np.uint8)
    strict3 = np.zeros(len(ordered), dtype=np.uint8)
    two_of_three = np.zeros(len(ordered), dtype=np.uint8)
    trigger2 = np.zeros(len(ordered), dtype=np.uint8)
    trigger3 = np.zeros(len(ordered), dtype=np.uint8)
    trigger23 = np.zeros(len(ordered), dtype=np.uint8)

    for positions in ordered.groupby(GROUP_KEYS, sort=False).indices.values():
        run = 0
        history: list[int] = []
        previous_window = None
        previous_state2 = previous_state3 = previous_state23 = 0
        for position in np.asarray(positions, dtype=int):
            current_window = int(window_ids[position])
            if previous_window is None or current_window != previous_window + 1:
                run = 0
                history = []
                previous_state2 = previous_state3 = previous_state23 = 0

            if raw[position] == 1:
                run += 1
            else:
                run = 0

            state2 = int(run >= 2)
            state3 = int(run >= 3)
            strict2[position] = state2
            strict3[position] = state3
            trigger2[position] = int(state2 == 1 and previous_state2 == 0)
            trigger3[position] = int(state3 == 1 and previous_state3 == 0)

            history.append(int(raw[position]))
            history = history[-3:]
            # Warm-start: at a segment's first two windows, evaluate the two
            # observations available; once available, maintain a 3-window ring.
            state23 = int(len(history) >= 2 and sum(history) >= 2)
            two_of_three[position] = state23
            trigger23[position] = int(state23 == 1 and previous_state23 == 0)

            previous_state2 = state2
            previous_state3 = state3
            previous_state23 = state23
            previous_window = current_window

    def restore(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=row_positions).reindex(frame.index).astype(np.uint8)

    result = frame.copy()
    result["two_consecutive"] = restore(strict2)
    result["three_consecutive"] = restore(strict3)
    result["two_of_three"] = restore(two_of_three)
    result["trigger_two"] = restore(trigger2)
    result["trigger_three"] = restore(trigger3)
    result["trigger_two_of_three"] = restore(trigger23)

    # Ensure the baseline state matches the previously used simulator exactly.
    if not np.array_equal(result["two_consecutive"].to_numpy(), result["confirmed_fall"].to_numpy(dtype=np.uint8)):
        raise ValueError("Recomputed 2-consecutive state does not match confirmed_fall in input CSV")
    if not np.array_equal(result["trigger_two"].to_numpy(), result["alert_trigger"].to_numpy(dtype=np.uint8)):
        raise ValueError("Recomputed 2-consecutive triggers do not match alert_trigger in input CSV")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="All-subjects Balanced-12 prediction CSV")
    parser.add_argument("--split-map", required=True, help="subject_split.csv with subject_id and split")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    args = parser.parse_args()

    input_path = Path(args.predictions)
    split_path = Path(args.split_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {missing}")
    split_map = pd.read_csv(split_path)
    if not {"subject_id", "split"}.issubset(split_map.columns):
        raise ValueError("Split map must contain subject_id and split columns")
    frame = frame.merge(split_map[["subject_id", "split"]], on="subject_id", how="left", validate="many_to_one")
    if frame["split"].isna().any():
        missing_subjects = sorted(frame.loc[frame.split.isna(), "subject_id"].unique().tolist())
        raise ValueError(f"No split assignment for subjects: {missing_subjects}")
    frame["y_true"] = (frame["activity_label"].astype(str).str.lower() == "fall").astype(np.uint8)
    frame["raw_predicted_binary"] = frame["raw_predicted_binary"].astype(np.uint8)
    frame = add_temporal_policies(frame)

    policies: Dict[str, Tuple[str, str | None]] = {
        "raw_model": ("raw_predicted_binary", None),
        "2_consecutive": ("two_consecutive", "trigger_two"),
        "2_of_3_warm_start": ("two_of_three", "trigger_two_of_three"),
        "3_consecutive": ("three_consecutive", "trigger_three"),
    }
    window_rows = []
    segment_rows = []
    subject_rows = []
    activity_rows = []
    for split in ["train", "validation", "test", "all"]:
        subset = frame if split == "all" else frame[frame.split == split]
        for policy, (pred_col, trigger_col) in policies.items():
            metrics = calculate_metrics(subset.y_true.to_numpy(), subset[pred_col].to_numpy())
            window_rows.append({
                "split": split, "policy": policy, "windows": len(subset),
                "fall_windows": int(subset.y_true.sum()),
                "normal_windows": int((subset.y_true == 0).sum()),
                "alert_trigger_events": int(subset[trigger_col].sum()) if trigger_col else np.nan,
                **metrics,
            })
        for policy, (pred_col, trigger_col) in list(policies.items())[1:]:
            segments = subset.groupby(GROUP_KEYS, sort=False).agg(
                y_true=("y_true", "max"),
                any_confirmation=(pred_col, "max"),
                any_trigger=(trigger_col, "max"),
            ).reset_index()
            fall_segments = segments[segments.y_true == 1]
            normal_segments = segments[segments.y_true == 0]
            segment_rows.append({
                "split": split, "policy": policy, "segments": len(segments),
                "fall_segments": len(fall_segments),
                "fall_segments_with_trigger": int(fall_segments.any_trigger.sum()),
                "fall_segment_trigger_coverage": float(fall_segments.any_trigger.mean()) if len(fall_segments) else np.nan,
                "normal_segments": len(normal_segments),
                "normal_segments_with_trigger": int(normal_segments.any_trigger.sum()),
                "normal_segment_trigger_rate": float(normal_segments.any_trigger.mean()) if len(normal_segments) else np.nan,
                "alert_trigger_events": int(subset[trigger_col].sum()),
            })

    for split in ["validation", "test", "all"]:
        subset = frame if split == "all" else frame[frame.split == split]
        for subject_id, subject_frame in subset.groupby("subject_id"):
            for policy, (pred_col, trigger_col) in policies.items():
                metrics = calculate_metrics(subject_frame.y_true.to_numpy(), subject_frame[pred_col].to_numpy())
                subject_rows.append({
                    "split": split, "subject_id": subject_id, "policy": policy,
                    "windows": len(subject_frame), "fall_windows": int(subject_frame.y_true.sum()),
                    "alert_trigger_events": int(subject_frame[trigger_col].sum()) if trigger_col else np.nan,
                    **metrics,
                })

    for split in ["validation", "test", "all"]:
        subset = frame if split == "all" else frame[frame.split == split]
        nonfall = subset[subset.y_true == 0]
        for activity, activity_frame in nonfall.groupby("activity_label"):
            for policy, (pred_col, trigger_col) in list(policies.items())[1:]:
                activity_rows.append({
                    "split": split, "activity_label": activity, "policy": policy,
                    "windows": len(activity_frame),
                    "raw_positive_windows": int(activity_frame.raw_predicted_binary.sum()),
                    "confirmed_windows": int(activity_frame[pred_col].sum()),
                    "alert_trigger_events": int(activity_frame[trigger_col].sum()),
                })

    window_df = pd.DataFrame(window_rows)
    segment_df = pd.DataFrame(segment_rows)
    subject_df = pd.DataFrame(subject_rows)
    activity_df = pd.DataFrame(activity_rows)
    window_df.to_csv(output_dir / "window_metrics_by_split.csv", index=False)
    segment_df.to_csv(output_dir / "segment_metrics_by_split.csv", index=False)
    subject_df.to_csv(output_dir / "subject_metrics_validation_test_all.csv", index=False)
    activity_df.to_csv(output_dir / "nonfall_activity_alert_comparison.csv", index=False)
    frame.to_csv(output_dir / "all_subjects_temporal_policy_comparison.csv", index=False)

    validation = window_df[(window_df.split == "validation") & (window_df.policy != "raw_model")].copy()
    validation["FPR_le_5pct"] = validation.FPR <= 0.05
    validation = validation.sort_values(["FPR_le_5pct", "F2", "recall"], ascending=False)
    validation.to_csv(output_dir / "validation_policy_selection.csv", index=False)
    best = validation.iloc[0]
    summary = {
        "rows": len(frame), "subjects": int(frame.subject_id.nunique()),
        "validation_selected_temporal_policy": str(best.policy),
        "validation_selected_metrics": {key: float(best[key]) for key in ["precision", "recall", "F1", "F2", "FPR"]},
        "baseline_state_and_trigger_match": True,
        "model_and_threshold_changed": False,
    }
    (output_dir / "temporal_policy_refinement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("BALANCED-12 TEMPORAL POLICY COMPARISON COMPLETED")
    print("------------------------------------------------")
    print(f"Windows analyzed: {len(frame):,}")
    print(f"Subjects analyzed: {frame.subject_id.nunique()}")
    print("Baseline two-consecutive outputs: exact match")
    print(f"Validation-selected temporal candidate: {best.policy}")
    print(f"Validation F2: {best.F2 * 100:.2f}% | Recall: {best.recall * 100:.2f}% | FPR: {best.FPR * 100:.2f}%")
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
