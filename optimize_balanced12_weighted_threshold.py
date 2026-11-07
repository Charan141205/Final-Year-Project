#!/usr/bin/env python3
"""Select a threshold for the ironing-weighted Balanced-12 model using TRAIN-only grouped OOF.

The Decision Tree is never trained on validation/test participants. The candidate
threshold is selected from 5-fold StratifiedGroupKFold out-of-fold probabilities
on training participants only, maximizing 2-of-3 temporal F2 subject to an OOF
false-positive-rate cap. The final weighted tree is then refit on all training
participants and evaluated on validation/test for diagnostic reporting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "FSR1_heel_proxy_range", "FSR2_midfoot_proxy_min", "FSR3_forefoot_proxy_mean", "FSR4_toe_proxy_min",
    "Gz_rad_s_range", "Ay_mps2_rms", "Az_mps2_max", "Gx_rad_s_std", "Ax_mps2_min", "Ay_mps2_max",
    "Az_mps2_mean", "Ax_mps2_rms",
]
GROUP_KEYS = ["subject_id", "recording_id", "segment_id"]
TREE_PARAMS = dict(criterion="gini", max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42)
DEFAULT_WEIGHT = 3.0
DEFAULT_FPR_CAP = 0.01
DEFAULT_CURRENT_CANDIDATE_THRESHOLD = 0.740000
DEFAULT_BASELINE_THRESHOLD = 0.8733322266783254


def metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.uint8)
    p = np.asarray(y_pred, dtype=np.uint8)
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp, "precision": precision,
            "recall": recall, "F1": f1, "F2": f2, "FPR": fpr, "accuracy": accuracy}


def contiguous_runs(frame: pd.DataFrame):
    """Positional indexes of contiguous windows, resetting at each gap/boundary."""
    ordered = frame.sort_values(GROUP_KEYS + ["window_id"], kind="stable")
    positions_by_label = pd.Series(np.arange(len(frame)), index=frame.index)
    positions = positions_by_label.loc[ordered.index].to_numpy(dtype=int)
    window_ids = ordered["window_id"].to_numpy(dtype=int)
    keys = ordered[GROUP_KEYS].astype(str).agg("|".join, axis=1).to_numpy()
    breaks = np.r_[True, (keys[1:] != keys[:-1]) | (window_ids[1:] != window_ids[:-1] + 1)]
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], len(positions)]
    return [positions[start:end] for start, end in zip(starts, ends)]


def temporal_2of3(probabilities, threshold, runs, row_count):
    raw = (np.asarray(probabilities) >= float(threshold)).astype(np.uint8)
    state = np.zeros(row_count, dtype=np.uint8)
    trigger = np.zeros(row_count, dtype=np.uint8)
    for positions in runs:
        values = raw[positions]
        local_state = np.zeros(len(positions), dtype=np.uint8)
        # Warm start: once two samples exist, evaluate those two. Afterwards use a 3-window history.
        if len(positions) >= 2:
            local_state[1] = int(values[0] + values[1] >= 2)
        if len(positions) >= 3:
            sums = values[:-2].astype(np.int16) + values[1:-1].astype(np.int16) + values[2:].astype(np.int16)
            local_state[2:] = (sums >= 2).astype(np.uint8)
        local_trigger = local_state.copy()
        if len(local_trigger) > 1:
            local_trigger[1:] = ((local_state[1:] == 1) & (local_state[:-1] == 0)).astype(np.uint8)
        state[positions] = local_state
        trigger[positions] = local_trigger
    return raw, state, trigger


def grouped_oof_predictions(train, weight, n_splits=5):
    x = train[FEATURES].to_numpy(dtype=float)
    y = (train["binary_label"].astype(str).str.lower() == "fall_related").astype(np.uint8).to_numpy()
    groups = train["subject_id"].astype(str).to_numpy()
    activities = train["activity_label"].astype(str).str.lower().to_numpy()
    weighted_oof = np.full(len(train), np.nan, dtype=float)
    baseline_oof = np.full(len(train), np.nan, dtype=float)
    fold_ids = np.full(len(train), -1, dtype=int)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_rows = []
    for fold, (train_idx, heldout_idx) in enumerate(cv.split(x, y, groups), start=1):
        baseline = DecisionTreeClassifier(**TREE_PARAMS)
        baseline.fit(x[train_idx], y[train_idx])
        baseline_oof[heldout_idx] = baseline.predict_proba(x[heldout_idx])[:, 1]

        sample_weight = np.ones(len(train_idx), dtype=float)
        fold_activity = activities[train_idx]
        fold_label = y[train_idx]
        weighted_mask = (fold_activity == "ironing") & (fold_label == 0)
        sample_weight[weighted_mask] = weight
        candidate = DecisionTreeClassifier(**TREE_PARAMS)
        candidate.fit(x[train_idx], y[train_idx], sample_weight=sample_weight)
        weighted_oof[heldout_idx] = candidate.predict_proba(x[heldout_idx])[:, 1]
        fold_ids[heldout_idx] = fold
        fold_rows.append({
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "heldout_rows": int(len(heldout_idx)),
            "heldout_subjects": ";".join(sorted(np.unique(groups[heldout_idx]))),
            "weighted_ironing_normal_training_rows": int(weighted_mask.sum()),
            "baseline_nodes": int(baseline.tree_.node_count),
            "weighted_candidate_nodes": int(candidate.tree_.node_count),
        })
        print(f"Fold {fold}/{n_splits}: train={len(train_idx):,}; held-out={len(heldout_idx):,}; "
              f"subjects={len(np.unique(groups[heldout_idx]))}; nodes={candidate.tree_.node_count}")
    if np.isnan(weighted_oof).any() or np.isnan(baseline_oof).any() or (fold_ids < 0).any():
        raise RuntimeError("Grouped OOF predictions are incomplete")
    return baseline_oof, weighted_oof, fold_ids, y, pd.DataFrame(fold_rows)


def sweep_thresholds(frame, probabilities, y, fpr_cap, current_threshold, outdir, label):
    runs = contiguous_runs(frame)
    thresholds = np.unique(probabilities)
    rows = []
    normal_ironing = (frame["activity_label"].astype(str).str.lower().to_numpy() == "ironing") & (y == 0)
    for threshold in thresholds:
        raw, state, trigger = temporal_2of3(probabilities, float(threshold), runs, len(frame))
        score = metrics(y, state)
        raw_score = metrics(y, raw)
        rows.append({
            "threshold": float(threshold), **score,
            "raw_precision": raw_score["precision"], "raw_recall": raw_score["recall"],
            "raw_F2": raw_score["F2"], "raw_FPR": raw_score["FPR"],
            "alert_trigger_events": int(trigger.sum()),
            "ironing_raw_FP_windows": int(raw[normal_ironing].sum()),
            "ironing_confirmed_FP_windows": int(state[normal_ironing].sum()),
            "ironing_alert_triggers": int(trigger[normal_ironing].sum()),
        })
    sweep = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    sweep.to_csv(outdir / f"{label}_threshold_sweep.csv", index=False)
    eligible = sweep[sweep.FPR <= fpr_cap + 1e-12].copy()
    if eligible.empty:
        raise RuntimeError(f"No {label} threshold satisfies FPR <= {fpr_cap:.2%}")
    selected = eligible.sort_values(["F2", "recall", "precision", "threshold"],
                                    ascending=[False, False, False, False]).iloc[0]
    return float(selected.threshold), selected.to_dict(), sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", required=True, help="balanced12_training_data.csv with train/validation/test rows")
    parser.add_argument("--baseline-model", required=True, help="Original balanced12_decision_tree.joblib")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--fpr-cap", type=float, default=DEFAULT_FPR_CAP)
    parser.add_argument("--current-candidate-threshold", type=float, default=DEFAULT_CURRENT_CANDIDATE_THRESHOLD)
    parser.add_argument("--baseline-threshold", type=float, default=DEFAULT_BASELINE_THRESHOLD)
    args = parser.parse_args()
    if not 0 < args.fpr_cap < 1:
        raise ValueError("--fpr-cap must be between 0 and 1, e.g. 0.01 for 1%")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.feature_table)
    required = ["split", "subject_id", "recording_id", "segment_id", "window_id", "activity_label", "binary_label"] + FEATURES
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")
    train = data[data.split == "train"].copy().reset_index(drop=True)
    holdout = data[data.split.isin(["validation", "test"])].copy().reset_index(drop=True)
    print(f"Training data: {len(train):,} windows / {train.subject_id.nunique()} participants")
    print(f"Validation + test: {len(holdout):,} windows")

    baseline_oof, weighted_oof, fold_ids, y_train, fold_summary = grouped_oof_predictions(train, args.weight)
    fold_summary.to_csv(outdir / "grouped_cv_fold_summary.csv", index=False)
    oof_frame = train[["subject_id", "recording_id", "segment_id", "window_id", "activity_label"]].copy()
    oof_frame["y_true"] = y_train
    oof_frame["fold"] = fold_ids
    oof_frame["baseline_oof_probability"] = baseline_oof
    oof_frame["weighted_oof_probability"] = weighted_oof
    oof_frame.to_csv(outdir / "training_grouped_oof_predictions.csv", index=False)
    runs_train = contiguous_runs(train)

    selected_threshold, selected_row, candidate_sweep = sweep_thresholds(
        train, weighted_oof, y_train, args.fpr_cap, args.current_candidate_threshold, outdir, "weighted_candidate_oof"
    )
    # OOF comparison with the original model at its established threshold.
    base_raw, base_state, base_trigger = temporal_2of3(baseline_oof, args.baseline_threshold, runs_train, len(train))
    candidate_current_raw, candidate_current_state, candidate_current_trigger = temporal_2of3(
        weighted_oof, args.current_candidate_threshold, runs_train, len(train)
    )
    candidate_selected_raw, candidate_selected_state, candidate_selected_trigger = temporal_2of3(
        weighted_oof, selected_threshold, runs_train, len(train)
    )
    oof_compare_rows = []
    for name, raw, state, trigger, threshold in [
        ("baseline_fixed_threshold", base_raw, base_state, base_trigger, args.baseline_threshold),
        ("weighted_candidate_current_0.74", candidate_current_raw, candidate_current_state, candidate_current_trigger, args.current_candidate_threshold),
        ("weighted_candidate_oof_selected", candidate_selected_raw, candidate_selected_state, candidate_selected_trigger, selected_threshold),
    ]:
        oof_compare_rows.append({"policy": name, "threshold": threshold, **metrics(y_train, state),
                                 "raw_F2": metrics(y_train, raw)["F2"], "raw_FPR": metrics(y_train, raw)["FPR"],
                                 "alert_trigger_events": int(trigger.sum())})
    pd.DataFrame(oof_compare_rows).to_csv(outdir / "grouped_oof_model_comparison.csv", index=False)

    # Final candidate tree: fit training participants only, with selected hard-negative weight.
    x_train = train[FEATURES].to_numpy(dtype=float)
    activities = train.activity_label.astype(str).str.lower().to_numpy()
    sample_weight = np.ones(len(train), dtype=float)
    sample_weight[(activities == "ironing") & (y_train == 0)] = args.weight
    final_model = DecisionTreeClassifier(**TREE_PARAMS)
    final_model.fit(x_train, y_train, sample_weight=sample_weight)
    model_path = outdir / "balanced12_ironing_weighted_oof_threshold.joblib"
    joblib.dump(final_model, model_path)

    baseline_model = joblib.load(args.baseline_model)
    if baseline_model.n_features_in_ != len(FEATURES):
        raise ValueError("Baseline model feature count does not match Balanced-12")

    metric_rows, activity_rows, subject_rows, segment_rows, prediction_rows = [], [], [], [], []
    for split_name in ["validation", "test"]:
        sub = holdout[holdout.split == split_name].copy().reset_index(drop=True)
        y = (sub.binary_label.astype(str).str.lower() == "fall_related").astype(np.uint8).to_numpy()
        runs = contiguous_runs(sub)
        x = sub[FEATURES].to_numpy(dtype=float)
        base_prob = baseline_model.predict_proba(x)[:, 1]
        weighted_prob = final_model.predict_proba(x)[:, 1]
        policies = [
            ("baseline_original_model", base_prob, args.baseline_threshold),
            ("weighted_candidate_current_0.74", weighted_prob, args.current_candidate_threshold),
            ("weighted_candidate_oof_selected", weighted_prob, selected_threshold),
        ]
        for policy, probabilities, threshold in policies:
            raw, state, trigger = temporal_2of3(probabilities, threshold, runs, len(sub))
            score = metrics(y, state)
            raw_score = metrics(y, raw)
            metric_rows.append({"split": split_name, "policy": policy, "threshold": float(threshold), **score,
                                "raw_F2": raw_score["F2"], "raw_FPR": raw_score["FPR"],
                                "raw_positive_windows": int(raw.sum()), "confirmed_windows": int(state.sum()),
                                "alert_trigger_events": int(trigger.sum())})
            normal_mask = y == 0
            for activity, indices in sub.loc[normal_mask].groupby("activity_label", sort=False).groups.items():
                idx = np.asarray(list(indices), dtype=int)
                activity_rows.append({"split": split_name, "policy": policy, "activity": activity,
                                      "normal_windows": len(idx), "raw_FP_windows": int(raw[idx].sum()),
                                      "confirmed_FP_windows": int(state[idx].sum()),
                                      "alert_trigger_events": int(trigger[idx].sum())})
            for subject, indices in sub.groupby("subject_id", sort=False).groups.items():
                idx = np.asarray(list(indices), dtype=int)
                subject_rows.append({"split": split_name, "policy": policy, "subject_id": subject,
                                     "windows": len(idx), **metrics(y[idx], state[idx]),
                                     "alert_trigger_events": int(trigger[idx].sum())})
            tmp = sub[GROUP_KEYS].copy()
            tmp["y_true"] = y
            tmp["trigger"] = trigger
            segments = tmp.groupby(GROUP_KEYS, sort=False).agg(y_true=("y_true", "max"), trigger=("trigger", "max")).reset_index()
            fall_segments = segments[segments.y_true == 1]
            normal_segments = segments[segments.y_true == 0]
            segment_rows.append({"split": split_name, "policy": policy,
                                 "fall_segments": len(fall_segments),
                                 "fall_segments_with_alert": int(fall_segments.trigger.sum()),
                                 "fall_segment_coverage": float(fall_segments.trigger.mean()) if len(fall_segments) else np.nan,
                                 "normal_segments": len(normal_segments),
                                 "normal_segments_with_alert": int(normal_segments.trigger.sum()),
                                 "normal_segment_alert_rate": float(normal_segments.trigger.mean()) if len(normal_segments) else np.nan})
            if policy == "weighted_candidate_oof_selected":
                pred = sub[["subject_id", "recording_id", "segment_id", "window_id", "activity_label", "binary_label"]].copy()
                pred["y_true"] = y
                pred["candidate_probability"] = weighted_prob
                pred["candidate_raw"] = raw
                pred["candidate_2of3"] = state
                pred["candidate_alert_trigger"] = trigger
                pred["threshold"] = float(threshold)
                pred["split"] = split_name
                prediction_rows.append(pred)

    pd.DataFrame(metric_rows).to_csv(outdir / "validation_test_policy_metrics.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(outdir / "validation_test_activity_metrics.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(outdir / "validation_test_subject_metrics.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(outdir / "validation_test_segment_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(outdir / "validation_test_selected_candidate_predictions.csv", index=False)

    manifest = {
        "candidate": "Ironing-weighted Balanced-12 DecisionTreeClassifier",
        "features_in_order": FEATURES,
        "tree_parameters": TREE_PARAMS,
        "normal_ironing_training_sample_weight": float(args.weight),
        "threshold_selected": float(selected_threshold),
        "threshold_selection": f"5-fold StratifiedGroupKFold OOF predictions on training participants only; maximize pooled 2-of-3 F2 subject to OOF FPR <= {args.fpr_cap:.4f}",
        "fpr_cap": float(args.fpr_cap),
        "current_candidate_threshold_for_reference": float(args.current_candidate_threshold),
        "baseline_threshold": float(args.baseline_threshold),
        "validation_used_to_select_threshold": False,
        "test_used_to_select_threshold": False,
        "temporal_policy": "2-of-3 warm-start; state resets at subject/recording/segment boundaries and window-ID gaps",
        "selected_oof_metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in selected_row.items()},
        "warning": "CAPPIMU labels are activity/window-level, not precise fall-onset labels. Test metrics are diagnostic only; this test split was inspected in previous development."
    }
    (outdir / "threshold_selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nTRAIN-ONLY GROUPED OOF THRESHOLD SELECTION COMPLETED")
    print("---------------------------------------------------")
    print(f"Selected threshold: {selected_threshold:.9f}")
    print(f"OOF F2: {selected_row['F2']:.5f} | OOF recall: {selected_row['recall']:.5f} | OOF FPR: {selected_row['FPR']:.5f}")
    print(f"FPR cap: {args.fpr_cap:.2%}; candidate normal-ironing weight: {args.weight:g}")
    print("\nGrouped OOF model comparison:")
    print(pd.DataFrame(oof_compare_rows)[["policy", "threshold", "precision", "recall", "F2", "FPR", "FP", "FN", "TP"]].to_string(index=False))
    print("\nValidation/test metrics:")
    result = pd.DataFrame(metric_rows)
    print(result[["split", "policy", "threshold", "precision", "recall", "F2", "FPR", "FP", "FN", "TP", "alert_trigger_events"]].to_string(index=False))
    print(f"\nModel saved: {model_path}")
    print(f"Outputs: {outdir.resolve()}")

if __name__ == "__main__":
    main()
