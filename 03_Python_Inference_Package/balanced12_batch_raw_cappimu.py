"""Batch runner for raw CAPPIMU Balanced-12 processing.

Place this file beside balanced12_full_pipeline.py and its supporting modules.
The raw-data root must contain participant folders such as:
    CAPPIMU/subject_1/joint/insole_knee_r.csv
    CAPPIMU/subject_2/joint/insole_knee_r.csv

Each participant is processed independently so temporal confirmation never
bridges from one participant to another. The trained model, feature order,
threshold and temporal rule are loaded from the existing pipeline unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from balanced12_full_pipeline import (  # noqa: E402
    FEATURE_ORDER,
    raw_cappimu_to_feature_windows,
    run_pipeline,
    write_outputs,
)

SUBJECT_RE = re.compile(r"^subject_(\d+)$", re.IGNORECASE)


def subject_id_from_path(path: Path) -> str | None:
    for parent in (path.parent, *path.parents):
        match = SUBJECT_RE.match(parent.name)
        if match:
            return f"subject_{int(match.group(1))}"
    return None


def discover_recordings(raw_root: Path, expected_subjects: int) -> tuple[dict[str, Path], dict[str, str]]:
    found: dict[str, list[Path]] = {}
    unassigned: list[str] = []
    for path in raw_root.rglob("insole_knee_r.csv"):
        sid = subject_id_from_path(path)
        if sid is None:
            unassigned.append(str(path))
            continue
        found.setdefault(sid, []).append(path)

    recordings: dict[str, Path] = {}
    issues: dict[str, str] = {}
    for number in range(1, expected_subjects + 1):
        sid = f"subject_{number}"
        paths = sorted(found.get(sid, []))
        if len(paths) == 1:
            recordings[sid] = paths[0]
        elif len(paths) == 0:
            issues[sid] = "missing_recording"
        else:
            issues[sid] = "multiple_matching_files: " + "; ".join(str(p) for p in paths)
    if unassigned:
        issues["unassigned_files"] = "Files not inside a subject_N folder: " + "; ".join(unassigned)
    for sid in found:
        match = SUBJECT_RE.match(sid)
        if match and int(match.group(1)) > expected_subjects:
            issues[sid] = f"outside_expected_subject_range_1_to_{expected_subjects}"
    return recordings, issues


def compare_to_reference(raw_features: pd.DataFrame, reference: pd.DataFrame,
                          subject_id: str) -> dict[str, Any]:
    keys = ["subject_id", "recording_id", "segment_id", "window_id"]
    subject_ref = reference.loc[reference["subject_id"].astype(str) == subject_id].copy()
    if subject_ref.empty:
        return {
            "reference_windows": 0, "matched_windows": 0,
            "raw_only_windows": int(len(raw_features)), "reference_only_windows": 0,
            "max_abs_feature_diff": None, "feature_match": False,
            "verification_note": "No matching subject rows found in reference feature table",
        }

    for frame_name, frame in (("raw", raw_features), ("reference", subject_ref)):
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate sequence keys found in {frame_name} features for {subject_id}")

    merged = raw_features[keys + FEATURE_ORDER].merge(
        subject_ref[keys + FEATURE_ORDER], on=keys, how="outer", suffixes=("_raw", "_ref"),
        indicator=True,
    )
    matched = merged.loc[merged["_merge"] == "both"]
    if len(matched):
        raw_values = matched[[f"{c}_raw" for c in FEATURE_ORDER]].to_numpy(dtype=np.float64)
        ref_values = matched[[f"{c}_ref" for c in FEATURE_ORDER]].to_numpy(dtype=np.float64)
        diffs = np.abs(raw_values - ref_values)
        max_diff = float(np.max(diffs))
        close = bool(np.allclose(raw_values, ref_values, rtol=1e-7, atol=1e-7))
    else:
        max_diff = None
        close = False

    raw_only = int((merged["_merge"] == "left_only").sum())
    ref_only = int((merged["_merge"] == "right_only").sum())
    return {
        "reference_windows": int(len(subject_ref)),
        "matched_windows": int(len(matched)),
        "raw_only_windows": raw_only,
        "reference_only_windows": ref_only,
        "max_abs_feature_diff": max_diff,
        "feature_match": bool(raw_only == 0 and ref_only == 0 and close),
        "verification_note": "Matched by subject/recording/segment/window keys; tolerance rtol=1e-7, atol=1e-7",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch process all available CAPPIMU subject_*/joint/insole_knee_r.csv files."
    )
    parser.add_argument("--raw-root", required=True,
                        help="Root folder containing subject_1, subject_2, ... participant folders")
    parser.add_argument("--output-dir", default="04_Integrated_Pipeline_Output\\All_Subjects_Raw")
    parser.add_argument("--reference-feature-table", default=None,
                        help="Optional balanced12_training_data.csv used to verify extracted features")
    parser.add_argument("--expected-subjects", type=int, default=30)
    parser.add_argument("--min-consecutive", type=int, default=2)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    if not raw_root.is_dir():
        parser.error(f"--raw-root is not a directory: {raw_root}")
    if args.expected_subjects < 1:
        parser.error("--expected-subjects must be at least 1")

    reference = None
    if args.reference_feature_table:
        ref_path = Path(args.reference_feature_table).expanduser().resolve()
        if not ref_path.is_file():
            parser.error(f"Reference feature table not found: {ref_path}")
        reference = pd.read_csv(ref_path)
        missing = [c for c in ["subject_id", "recording_id", "segment_id", "window_id", *FEATURE_ORDER]
                   if c not in reference.columns]
        if missing:
            parser.error(f"Reference feature table is missing columns: {missing}")

    recordings, discovery_issues = discover_recordings(raw_root, args.expected_subjects)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_alerts: list[pd.DataFrame] = []

    for number in range(1, args.expected_subjects + 1):
        sid = f"subject_{number}"
        source = recordings.get(sid)
        if source is None:
            rows.append({
                "subject_id": sid, "status": "MISSING_OR_AMBIGUOUS",
                "source_file": "", "error": discovery_issues.get(sid, "not_found"),
                "raw_rows": 0, "windows_processed": 0, "raw_positive_windows": 0,
                "pending_positive_windows": 0, "confirmed_fall_windows": 0,
                "alert_trigger_events": 0, "reference_windows": None,
                "matched_windows": None, "raw_only_windows": None,
                "reference_only_windows": None, "max_abs_feature_diff": None,
                "feature_match": None,
            })
            print(f"[{number:02d}/{args.expected_subjects:02d}] {sid}: missing or ambiguous recording")
            continue

        print(f"[{number:02d}/{args.expected_subjects:02d}] Processing {sid}: {source}")
        try:
            # Feature extraction happens once; retain the feature frame for comparison.
            raw_frame = raw_cappimu_to_feature_windows(source, sid)
            raw_rows = int(sum(1 for _ in open(source, "rb")) - 1)
            result = run_pipeline(raw_frame, min_consecutive=args.min_consecutive)
            subject_out = out_root / sid
            with contextlib.redirect_stdout(io.StringIO()):
                write_outputs(result, subject_out, "raw-cappimu", args.min_consecutive)

            comparison: dict[str, Any] = {}
            if reference is not None:
                comparison = compare_to_reference(raw_frame, reference, sid)

            row = {
                "subject_id": sid, "status": "PROCESSED",
                "source_file": str(source), "error": "",
                "raw_rows": raw_rows,
                "windows_processed": int(len(result)),
                "raw_positive_windows": int(result["raw_predicted_binary"].sum()),
                "pending_positive_windows": int(result["pending_positive"].sum()),
                "confirmed_fall_windows": int(result["confirmed_fall"].sum()),
                "alert_trigger_events": int(result["alert_trigger"].sum()),
                "reference_windows": comparison.get("reference_windows"),
                "matched_windows": comparison.get("matched_windows"),
                "raw_only_windows": comparison.get("raw_only_windows"),
                "reference_only_windows": comparison.get("reference_only_windows"),
                "max_abs_feature_diff": comparison.get("max_abs_feature_diff"),
                "feature_match": comparison.get("feature_match"),
            }
            rows.append(row)
            all_predictions.append(result)
            alerts = result.loc[result["alert_trigger"] == 1].copy()
            all_alerts.append(alerts)
            comparison_status = (
                f", feature_match={comparison['feature_match']}"
                if reference is not None else ""
            )
            print(
                f"    windows={len(result):,}, raw+={row['raw_positive_windows']:,}, "
                f"confirmed={row['confirmed_fall_windows']:,}, alerts={row['alert_trigger_events']:,}"
                f"{comparison_status}"
            )
        except Exception as exc:  # Continue processing remaining subjects.
            rows.append({
                "subject_id": sid, "status": "ERROR", "source_file": str(source),
                "error": f"{type(exc).__name__}: {exc}", "raw_rows": 0,
                "windows_processed": 0, "raw_positive_windows": 0,
                "pending_positive_windows": 0, "confirmed_fall_windows": 0,
                "alert_trigger_events": 0, "reference_windows": None,
                "matched_windows": None, "raw_only_windows": None,
                "reference_only_windows": None, "max_abs_feature_diff": None,
                "feature_match": None,
            })
            print(f"    ERROR: {type(exc).__name__}: {exc}")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_root / "batch_subject_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            out_root / "all_subjects_balanced12_predictions.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(out_root / "all_subjects_balanced12_predictions.csv", index=False)
    if all_alerts:
        pd.concat(all_alerts, ignore_index=True).to_csv(
            out_root / "all_subjects_balanced12_alert_events.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(out_root / "all_subjects_balanced12_alert_events.csv", index=False)

    processed = [r for r in rows if r["status"] == "PROCESSED"]
    successful_matches = sum(1 for r in processed if r.get("feature_match") is True)
    summary = {
        "pipeline": "Balanced-12 raw CAPPIMU batch processing",
        "raw_root": str(raw_root),
        "output_dir": str(out_root),
        "expected_subjects": int(args.expected_subjects),
        "processed_subjects": len(processed),
        "missing_or_ambiguous_subjects": sum(1 for r in rows if r["status"] == "MISSING_OR_AMBIGUOUS"),
        "error_subjects": sum(1 for r in rows if r["status"] == "ERROR"),
        "total_windows_processed": int(sum(r["windows_processed"] for r in processed)),
        "total_raw_positive_windows": int(sum(r["raw_positive_windows"] for r in processed)),
        "total_pending_positive_windows": int(sum(r["pending_positive_windows"] for r in processed)),
        "total_confirmed_fall_windows": int(sum(r["confirmed_fall_windows"] for r in processed)),
        "total_alert_trigger_events": int(sum(r["alert_trigger_events"] for r in processed)),
        "subjects_matching_reference_features": successful_matches if reference is not None else None,
        "reference_feature_table_used": str(ref_path) if reference is not None else None,
        "per_subject_summary_csv": str(summary_csv),
        "limitations": [
            "Offline processing of existing CAPPIMU recordings only; no new ESP32 data is collected.",
            "CAPPIMU pressure channels are regional insole proxies, not actual shoe FSR measurements.",
            "CAPPIMU IMU is knee-mounted, not shoe-mounted.",
            "Dataset labels are activity/window-level, not precise fall-onset annotations.",
            "Confirmed alerts are model outputs, not verified real-world falls.",
            "Temporal confirmation is reset independently for each participant recording.",
        ],
    }
    (out_root / "BATCH_RAW_CAPPIMU_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nBALANCED-12 RAW CAPPIMU BATCH COMPLETED")
    print("--------------------------------------")
    print(f"Expected subjects:                  {args.expected_subjects}")
    print(f"Processed subjects:                 {summary['processed_subjects']}")
    print(f"Missing/ambiguous subjects:         {summary['missing_or_ambiguous_subjects']}")
    print(f"Subjects with errors:               {summary['error_subjects']}")
    print(f"Total windows processed:             {summary['total_windows_processed']:,}")
    print(f"Total alert trigger events:          {summary['total_alert_trigger_events']:,}")
    if reference is not None:
        print(f"Subjects matching reference table:  {successful_matches}/{len(processed)}")
    print(f"Per-subject summary:                 {summary_csv}")
    print(f"Output folder:                       {out_root}")
    return 0 if processed else 2


if __name__ == "__main__":
    raise SystemExit(main())
