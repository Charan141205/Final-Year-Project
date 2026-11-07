#!/usr/bin/env python3
"""Compare segment-reset and gap-aware Balanced-12 temporal policies.

This script does NOT retrain a classifier or change probabilities/thresholds.
It operates on the existing all-subject prediction CSV and subject split map.

A temporal link is allowed only when:
- it is the next exported window_id in the same subject + recording_id,
- start_time_ms - previous end_time_ms is non-negative and <= max_gap_ms.

Thus, it may bridge an inner feature-segment boundary, but never crosses a
recording/activity boundary or a missing window_id. The 100 ms setting should
reproduce the current segment-reset policy because inner segment boundaries
are created by gaps >100 ms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_GAPS = [100, 125, 150, 175, 200, 250, 300, 400, 500]
GROUP = ["subject_id", "recording_id"]
REQUIRED = [
    "subject_id", "recording_id", "segment_id", "window_id", "activity_label",
    "start_time_ms", "end_time_ms", "raw_predicted_binary", "confirmed_fall", "alert_trigger",
]


def score_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float | int]:
    y = np.asarray(y, dtype=np.uint8)
    pred = np.asarray(pred, dtype=np.uint8)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "precision": precision, "recall": recall, "F1": f1,
            "F2": f2, "FPR": fpr, "accuracy": accuracy}


def prepare_frame(pred_path: Path, split_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(pred_path)
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {missing}")
    split_map = pd.read_csv(split_path)
    if not {"subject_id", "split"}.issubset(split_map.columns):
        raise ValueError("Split map must contain subject_id and split columns")
    frame = frame.merge(split_map[["subject_id", "split"]], on="subject_id", how="left", validate="many_to_one")
    if frame["split"].isna().any():
        missing_subjects = sorted(frame.loc[frame.split.isna(), "subject_id"].unique().tolist())
        raise ValueError(f"Subjects missing from split map: {missing_subjects}")
    frame["y_true"] = (frame["activity_label"].astype(str).str.lower() == "fall").astype(np.uint8)
    frame["raw"] = frame["raw_predicted_binary"].astype(np.uint8)
    frame["window_id"] = pd.to_numeric(frame["window_id"], errors="raise").astype(np.int64)
    for col in ["start_time_ms", "end_time_ms"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    frame = frame.sort_values(["subject_id", "recording_id", "window_id"], kind="stable").reset_index(drop=True)
    return frame


def simulate(frame: pd.DataFrame, max_gap_ms: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return causal confirmed-window state and first-confirmation trigger."""
    if mode not in {"two_consecutive", "two_of_three"}:
        raise ValueError(f"Unknown mode: {mode}")
    state = np.zeros(len(frame), dtype=np.uint8)
    trigger = np.zeros(len(frame), dtype=np.uint8)

    for _, positions in frame.groupby(GROUP, sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        history: List[int] = []
        run = 0
        prev_wid = None
        prev_end = None
        prev_state = 0

        for pos in positions:
            wid = int(frame.at[pos, "window_id"])
            start = float(frame.at[pos, "start_time_ms"])
            end = float(frame.at[pos, "end_time_ms"])
            raw = int(frame.at[pos, "raw"])

            linked = False
            if prev_wid is not None and prev_end is not None:
                gap = start - prev_end
                linked = (wid == prev_wid + 1) and (0.0 <= gap <= float(max_gap_ms))
            if not linked:
                history = []
                run = 0
                prev_state = 0

            if mode == "two_consecutive":
                run = run + 1 if raw else 0
                current_state = int(run >= 2)
            else:
                history.append(raw)
                history = history[-3:]
                # Warm-start: once two observations are available, evaluate them;
                # subsequently use the latest three adjacent windows.
                current_state = int(len(history) >= 2 and sum(history) >= 2)

            state[pos] = current_state
            trigger[pos] = int(current_state == 1 and prev_state == 0)
            prev_state = current_state
            prev_wid = wid
            prev_end = end

    return state, trigger


def segment_summary(frame: pd.DataFrame, state: np.ndarray, trigger: np.ndarray) -> Dict[str, int | float]:
    tmp = frame[["subject_id", "recording_id", "segment_id", "y_true"]].copy()
    tmp["state"] = state
    tmp["trigger"] = trigger
    fall = tmp[tmp.y_true == 1].groupby(["subject_id", "recording_id", "segment_id"])["state"].max()
    normal = tmp[tmp.y_true == 0].groupby(["subject_id", "recording_id", "segment_id"])["trigger"].max()
    return {
        "fall_segments_alerted": int(fall.sum()),
        "fall_segments_total": int(len(fall)),
        "fall_segment_coverage": float(fall.mean()) if len(fall) else 0.0,
        "normal_segments_with_alert": int(normal.sum()),
        "normal_segments_total": int(len(normal)),
        "normal_segment_alert_rate": float(normal.mean()) if len(normal) else 0.0,
    }


def evaluate(frame: pd.DataFrame, state: np.ndarray, trigger: np.ndarray, policy: str, gap: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_rows = []
    subject_rows = []
    segment_rows = []
    activity_rows = []
    for split_name, idx in frame.groupby("split", sort=False).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        g = frame.iloc[idx]
        st = state[idx]
        tr = trigger[idx]
        met = score_metrics(g.y_true.to_numpy(), st)
        seg = segment_summary(g, st, tr)
        window_rows.append({"policy": policy, "max_interwindow_gap_ms": gap, "split": split_name,
                            **met, "alert_trigger_events": int(tr.sum()), **seg})

    d = frame.copy()
    d["confirmed"] = state
    d["alert_trigger_new"] = trigger
    for (split_name, subject), g in d.groupby(["split", "subject_id"], sort=False):
        met = score_metrics(g.y_true.to_numpy(), g.confirmed.to_numpy())
        seg = segment_summary(g, g.confirmed.to_numpy(), g.alert_trigger_new.to_numpy())
        subject_rows.append({"policy": policy, "max_interwindow_gap_ms": gap, "split": split_name,
                             "subject_id": subject, **met,
                             "alert_trigger_events": int(g.alert_trigger_new.sum()), **seg})

    # Segment-level table for this policy and split.
    seg_df = d.groupby(["split", "subject_id", "recording_id", "segment_id", "activity_label"], sort=False).agg(
        windows=("window_id", "size"), fall_label=("y_true", "max"),
        confirmed_windows=("confirmed", "sum"), alert_triggers=("alert_trigger_new", "sum")
    ).reset_index()
    seg_df.insert(0, "policy", policy)
    seg_df.insert(1, "max_interwindow_gap_ms", gap)
    segment_rows.append(seg_df)

    act_df = d.groupby(["split", "activity_label"], sort=False).agg(
        windows=("window_id", "size"), fall_windows=("y_true", "sum"),
        confirmed_windows=("confirmed", "sum"), alert_triggers=("alert_trigger_new", "sum")
    ).reset_index()
    act_df.insert(0, "policy", policy)
    act_df.insert(1, "max_interwindow_gap_ms", gap)
    activity_rows.append(act_df)

    return (pd.DataFrame(window_rows), pd.DataFrame(subject_rows),
            pd.concat(segment_rows, ignore_index=True), pd.concat(activity_rows, ignore_index=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split-map", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gaps-ms", nargs="+", type=int, default=DEFAULT_GAPS)
    ap.add_argument("--fpr-limit", type=float, default=0.01,
                    help="Validation FPR constraint for candidate selection (default 0.01)")
    args = ap.parse_args()

    frame = prepare_frame(Path(args.predictions), Path(args.split_map))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Baseline reproducibility checks.
    base2, base2tr = simulate(frame, 100, "two_consecutive")
    if not np.array_equal(base2, frame.confirmed_fall.astype(np.uint8).to_numpy()):
        raise ValueError("100 ms two-consecutive state does not match stored confirmed_fall")
    if not np.array_equal(base2tr, frame.alert_trigger.astype(np.uint8).to_numpy()):
        raise ValueError("100 ms two-consecutive triggers do not match stored alert_trigger")

    sweep_window, sweep_subject, sweep_segment, sweep_activity = [], [], [], []
    for gap in sorted(set(args.gaps_ms)):
        if gap < 100 or gap > 500:
            raise ValueError("Gap values must be between 100 and 500 ms")
        for mode, policy in [("two_consecutive", "gap_aware_2_consecutive"),
                             ("two_of_three", "gap_aware_2_of_3_warm_start")]:
            st, tr = simulate(frame, gap, mode)
            a, b, c, d = evaluate(frame, st, tr, policy, gap)
            sweep_window.append(a); sweep_subject.append(b); sweep_segment.append(c); sweep_activity.append(d)

    win = pd.concat(sweep_window, ignore_index=True)
    sub = pd.concat(sweep_subject, ignore_index=True)
    seg = pd.concat(sweep_segment, ignore_index=True)
    act = pd.concat(sweep_activity, ignore_index=True)

    # Select the gap-aware 2-of-3 threshold using validation only.
    val = win[(win.split == "validation") & (win.policy == "gap_aware_2_of_3_warm_start")].copy()
    feasible = val[val.FPR <= args.fpr_limit]
    if feasible.empty:
        selected = val.sort_values(["FPR", "F2", "max_interwindow_gap_ms"], ascending=[True, False, True]).iloc[0]
        selection_reason = "No validation candidate met FPR limit; selected lowest-FPR candidate, then F2."
    else:
        selected = feasible.sort_values(["F2", "max_interwindow_gap_ms"], ascending=[False, True]).iloc[0]
        selection_reason = "Highest validation F2 among candidates meeting the validation FPR limit; smaller gap wins ties."

    selection = pd.DataFrame([{
        "selected_policy": selected.policy,
        "selected_max_interwindow_gap_ms": int(selected.max_interwindow_gap_ms),
        "validation_F2": float(selected.F2),
        "validation_precision": float(selected.precision),
        "validation_recall": float(selected.recall),
        "validation_FPR": float(selected.FPR),
        "validation_alert_trigger_events": int(selected.alert_trigger_events),
        "FPR_limit": float(args.fpr_limit),
        "selection_reason": selection_reason,
        "test_used_for_selection": False,
    }])

    # Include raw and original policies in summary by recomputing original states.
    raw_state = frame.raw.to_numpy(dtype=np.uint8)
    raw_trig = np.zeros(len(frame), dtype=np.uint8)
    original2_state = frame.confirmed_fall.to_numpy(dtype=np.uint8)
    original2_trig = frame.alert_trigger.to_numpy(dtype=np.uint8)
    original23_state, original23_trig = simulate(frame, 100, "two_of_three")
    baseline_rows = []
    for policy, st, tr in [
        ("raw_model", raw_state, raw_trig),
        ("segment_reset_2_consecutive", original2_state, original2_trig),
        ("segment_reset_2_of_3_warm_start", original23_state, original23_trig),
    ]:
        a, b, c, d = evaluate(frame, st, tr, policy, 100)
        baseline_rows.append(a)
    baseline = pd.concat(baseline_rows, ignore_index=True)

    win.to_csv(outdir / "gap_policy_metrics_by_split.csv", index=False)
    sub.to_csv(outdir / "gap_policy_subject_metrics.csv", index=False)
    seg.to_csv(outdir / "gap_policy_segment_metrics.csv", index=False)
    act.to_csv(outdir / "gap_policy_activity_metrics.csv", index=False)
    selection.to_csv(outdir / "validation_selected_gap_policy.csv", index=False)
    baseline.to_csv(outdir / "raw_and_segment_reset_baselines.csv", index=False)

    # Export the selected candidate's per-window state for reproducibility.
    chosen_gap = int(selected.max_interwindow_gap_ms)
    chosen_state, chosen_trigger = simulate(frame, chosen_gap, "two_of_three")
    export_cols = [c for c in frame.columns if c not in {"raw_positive", "prev_end", "prev_raw", "prev_segment", "interwindow_gap", "segment_changed"}]
    export = frame[export_cols].copy()
    export["gap_aware_2_of_3_confirmed"] = chosen_state
    export["gap_aware_2_of_3_alert_trigger"] = chosen_trigger
    export["gap_aware_max_interwindow_gap_ms"] = chosen_gap
    export.to_csv(outdir / "selected_gap_aware_2_of_3_predictions.csv", index=False)

    summary = {
        "windows_analyzed": int(len(frame)),
        "subjects_analyzed": int(frame.subject_id.nunique()),
        "split_counts": frame.groupby("split").subject_id.nunique().astype(int).to_dict(),
        "baseline_two_consecutive_exact_match": True,
        "gap_thresholds_tested_ms": sorted(set(args.gaps_ms)),
        "validation_fpr_limit": float(args.fpr_limit),
        "selected_policy": str(selected.policy),
        "selected_max_interwindow_gap_ms": chosen_gap,
        "validation_F2": float(selected.F2),
        "validation_recall": float(selected.recall),
        "validation_FPR": float(selected.FPR),
        "test_not_used_for_selection": True,
        "note": "Gap-aware links require same subject/recording, consecutive window_id, non-negative inter-window gap <= selected max. Recording/activity boundaries and missing window IDs always reset state."
    }
    (outdir / "gap_aware_temporal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    test_rows = win[(win.split == "test") & (win.policy == selected.policy)].sort_values("max_interwindow_gap_ms")
    print("BALANCED-12 GAP-AWARE TEMPORAL COMPARISON COMPLETED")
    print("--------------------------------------------------")
    print(f"Windows analyzed:                 {len(frame):,}")
    print(f"Subjects analyzed:                 {frame.subject_id.nunique()}")
    print("Baseline two-consecutive match:    exact")
    print(f"Validation FPR limit:              {args.fpr_limit:.2%}")
    print(f"Selected policy:                   {selected.policy}")
    print(f"Selected max inter-window gap:     {chosen_gap} ms")
    print(f"Validation F2 / Recall / FPR:      {selected.F2:.2%} / {selected.recall:.2%} / {selected.FPR:.2%}")
    print("\nHeld-out test metrics for selected policy (diagnostic only):")
    print(test_rows[test_rows.max_interwindow_gap_ms == chosen_gap][[
        "precision", "recall", "F2", "FPR", "alert_trigger_events",
        "fall_segment_coverage", "normal_segment_alert_rate"
    ]].to_string(index=False))
    print(f"Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
