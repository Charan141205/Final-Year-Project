#!/usr/bin/env python3
"""
CAPPIMU timestamp and orientation-quality audit.

Scans subject_*/joint/insole_knee_r.csv without changing source files.
Separates duplicate timestamps (dt == 0) from backward timestamps (dt < 0),
and reports positive gaps above a configurable raw-unit threshold.

The script does NOT assume timestamp units and does NOT rotate, interpolate,
resample, delete, or modify any source data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["time", "knee_r(ex)", "knee_r(ey)", "knee_r(ez)"]


def finite_quantiles(values: np.ndarray, qs=(0.01, 0.50, 0.95, 0.99)) -> dict[str, float | None]:
    arr = values[np.isfinite(values)]
    if arr.size == 0:
        return {f"q{int(q*100):02d}": None for q in qs}
    quantiles = np.quantile(arr, qs)
    return {f"q{int(q*100):02d}": float(v) for q, v in zip(qs, quantiles)}


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def examples_for_subject(subject: str, t: np.ndarray, dt: np.ndarray,
                         gap_threshold: float, max_each: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Largest backward steps first.
    neg = np.flatnonzero(np.isfinite(dt) & (dt < 0))
    if neg.size:
        order = neg[np.argsort(dt[neg])[:max_each]]
        for idx in order:
            out.append({
                "subject_id": subject,
                "event_type": "backward_timestamp",
                "previous_row_0based": int(idx),
                "current_row_0based": int(idx + 1),
                "previous_time_raw": float(t[idx]) if np.isfinite(t[idx]) else None,
                "current_time_raw": float(t[idx + 1]) if np.isfinite(t[idx + 1]) else None,
                "delta_time_raw": float(dt[idx]),
            })

    # Largest positive gaps first.
    gap = np.flatnonzero(np.isfinite(dt) & (dt > gap_threshold))
    if gap.size:
        order = gap[np.argsort(dt[gap])[::-1][:max_each]]
        for idx in order:
            out.append({
                "subject_id": subject,
                "event_type": "positive_gap_over_threshold",
                "previous_row_0based": int(idx),
                "current_row_0based": int(idx + 1),
                "previous_time_raw": float(t[idx]) if np.isfinite(t[idx]) else None,
                "current_time_raw": float(t[idx + 1]) if np.isfinite(t[idx + 1]) else None,
                "delta_time_raw": float(dt[idx]),
            })

    # First few exact duplicates, to make the duplicate pattern inspectable.
    dup = np.flatnonzero(np.isfinite(dt) & (dt == 0))
    for idx in dup[:max_each]:
        out.append({
            "subject_id": subject,
            "event_type": "duplicate_timestamp",
            "previous_row_0based": int(idx),
            "current_row_0based": int(idx + 1),
            "previous_time_raw": float(t[idx]),
            "current_time_raw": float(t[idx + 1]),
            "delta_time_raw": 0.0,
        })
    return out


def scan_subject(path: Path, subject: str, gap_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header = pd.read_csv(path, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    n_rows = int(len(df))
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    ex = pd.to_numeric(df["knee_r(ex)"], errors="coerce").to_numpy(dtype=float)
    ey = pd.to_numeric(df["knee_r(ey)"], errors="coerce").to_numpy(dtype=float)
    ez = pd.to_numeric(df["knee_r(ez)"], errors="coerce").to_numpy(dtype=float)

    dt = np.diff(t)
    finite_pair = np.isfinite(t[:-1]) & np.isfinite(t[1:])
    valid_dt = dt[finite_pair]

    equal = int(np.sum(finite_pair & (dt == 0)))
    backward = int(np.sum(finite_pair & (dt < 0)))
    positive = int(np.sum(finite_pair & (dt > 0)))
    gaps = dt[finite_pair & (dt > gap_threshold)]
    positives = dt[finite_pair & (dt > 0)]

    # This reconstructs the earlier segment-boundary rule as a diagnostic:
    # a negative step or a positive step larger than the chosen threshold.
    breaks = int(np.sum(finite_pair & ((dt < 0) | (dt > gap_threshold))))

    # A suspicious joint Euler signature observed in the earlier audit:
    # ey outside provisional +/-90 and ez very near 114.3731689453125.
    # This is reported as a pattern, not declared to be a vendor sentinel.
    joint_pattern = (
        np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez)
        & (np.abs(ey) > 90.0)
        & (np.abs(ez - 114.3731689453125) < 1e-5)
    )

    row = {
        "subject_id": subject,
        "file_path": str(path),
        "rows": n_rows,
        "time_missing_or_nonfinite_rows": int(np.sum(~np.isfinite(t))),
        "adjacent_pairs": max(n_rows - 1, 0),
        "adjacent_pairs_with_finite_time": int(np.sum(finite_pair)),
        "duplicate_timestamp_steps_dt_eq_0": equal,
        "backward_timestamp_steps_dt_lt_0": backward,
        "positive_timestamp_steps": positive,
        "nonincreasing_steps_total": equal + backward,
        "nonincreasing_percent_of_finite_pairs": (
            100.0 * (equal + backward) / np.sum(finite_pair) if np.sum(finite_pair) else None
        ),
        "positive_gaps_over_threshold": int(gaps.size),
        "gap_threshold_raw_units": float(gap_threshold),
        "segment_breaks_if_dt_lt_0_or_gt_threshold": breaks,
        "positive_dt_min_raw_units": float(np.min(positives)) if positives.size else None,
        "positive_dt_median_raw_units": float(np.median(positives)) if positives.size else None,
        "positive_dt_q99_raw_units": float(np.quantile(positives, 0.99)) if positives.size else None,
        "positive_dt_max_raw_units": float(np.max(positives)) if positives.size else None,
        "backward_dt_min_raw_units": float(np.min(dt[finite_pair & (dt < 0)])) if backward else None,
        "backward_dt_median_raw_units": float(np.median(dt[finite_pair & (dt < 0)])) if backward else None,
        "longest_consecutive_duplicate_step_run": longest_true_run(finite_pair & (dt == 0)),
        "ex_nonfinite_rows": int(np.sum(~np.isfinite(ex))),
        "ey_nonfinite_rows": int(np.sum(~np.isfinite(ey))),
        "ez_nonfinite_rows": int(np.sum(~np.isfinite(ez))),
        "ey_outside_provisional_pm90": int(np.sum(np.isfinite(ey) & (np.abs(ey) > 90.0))),
        "ey_outside_pm90_and_ez_near_114_373169": int(np.sum(joint_pattern)),
        "ez_exact_suspect_value_count": int(np.sum(np.isfinite(ez) & (np.abs(ez - 114.3731689453125) < 1e-5))),
    }
    # Include descriptive positive-step quantiles.
    row.update({f"positive_dt_{k}_raw_units": v for k, v in finite_quantiles(positives).items()})
    return row, examples_for_subject(subject, t, dt, gap_threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CAPPIMU timestamp continuity and orientation anomalies.")
    parser.add_argument("--raw-root", required=True, help="Folder containing subject_1, subject_2, ...")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--expected-subjects", type=int, default=30)
    parser.add_argument("--gap-threshold", type=float, default=500.0,
                        help="Positive timestamp gap threshold in raw timestamp units (default: 500; units are not assumed).")
    args = parser.parse_args()

    root = Path(args.raw_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for i in range(1, args.expected_subjects + 1):
        subject = f"subject_{i}"
        path = root / subject / "joint" / "insole_knee_r.csv"
        if not path.is_file():
            inventory.append({"subject_id": subject, "status": "missing", "rows": 0, "error": "file not found"})
            print(f"[{i:02d}/{args.expected_subjects:02d}] {subject}: MISSING")
            continue
        try:
            summary, ex_rows = scan_subject(path, subject, args.gap_threshold)
            summaries.append(summary)
            examples.extend(ex_rows)
            inventory.append({"subject_id": subject, "status": "processed", "rows": summary["rows"], "error": ""})
            print(
                f"[{i:02d}/{args.expected_subjects:02d}] {subject}: "
                f"rows={summary['rows']:,}, duplicate={summary['duplicate_timestamp_steps_dt_eq_0']:,}, "
                f"backward={summary['backward_timestamp_steps_dt_lt_0']:,}, "
                f"gaps>{args.gap_threshold:g}={summary['positive_gaps_over_threshold']:,}"
            )
        except Exception as exc:
            inventory.append({"subject_id": subject, "status": "error", "rows": 0, "error": str(exc)})
            print(f"[{i:02d}/{args.expected_subjects:02d}] {subject}: ERROR: {exc}")

    summary_df = pd.DataFrame(summaries)
    inventory_df = pd.DataFrame(inventory)
    examples_df = pd.DataFrame(examples)

    summary_path = outdir / "timestamp_orientation_quality_by_subject.csv"
    inventory_path = outdir / "timestamp_audit_inventory.csv"
    examples_path = outdir / "timestamp_event_examples.csv"
    json_path = outdir / "timestamp_orientation_audit_summary.json"
    readme_path = outdir / "README_FIRST.txt"

    summary_df.to_csv(summary_path, index=False)
    inventory_df.to_csv(inventory_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    total_rows = int(summary_df["rows"].sum()) if not summary_df.empty else 0
    total_pairs = int(summary_df["adjacent_pairs_with_finite_time"].sum()) if not summary_df.empty else 0
    total_dup = int(summary_df["duplicate_timestamp_steps_dt_eq_0"].sum()) if not summary_df.empty else 0
    total_back = int(summary_df["backward_timestamp_steps_dt_lt_0"].sum()) if not summary_df.empty else 0
    total_gaps = int(summary_df["positive_gaps_over_threshold"].sum()) if not summary_df.empty else 0
    total_breaks = int(summary_df["segment_breaks_if_dt_lt_0_or_gt_threshold"].sum()) if not summary_df.empty else 0

    result = {
        "raw_root": str(root),
        "expected_subjects": int(args.expected_subjects),
        "processed_subjects": int((inventory_df["status"] == "processed").sum()) if not inventory_df.empty else 0,
        "missing_subjects": int((inventory_df["status"] == "missing").sum()) if not inventory_df.empty else 0,
        "error_subjects": int((inventory_df["status"] == "error").sum()) if not inventory_df.empty else 0,
        "rows_scanned": total_rows,
        "finite_adjacent_timestamp_pairs": total_pairs,
        "duplicate_timestamp_steps_dt_eq_0": total_dup,
        "backward_timestamp_steps_dt_lt_0": total_back,
        "nonincreasing_timestamp_steps_total": total_dup + total_back,
        "nonincreasing_percent_of_finite_pairs": 100.0 * (total_dup + total_back) / total_pairs if total_pairs else None,
        "positive_gaps_over_threshold": total_gaps,
        "gap_threshold_raw_units": float(args.gap_threshold),
        "segment_breaks_if_dt_lt_0_or_gt_threshold": total_breaks,
        "orientation_rows_ey_outside_provisional_pm90": int(summary_df["ey_outside_provisional_pm90"].sum()) if not summary_df.empty else 0,
        "orientation_rows_matching_joint_ey_ez_pattern": int(summary_df["ey_outside_pm90_and_ez_near_114_373169"].sum()) if not summary_df.empty else 0,
        "warning": "Timestamp units are not assumed. This diagnostic distinguishes exact duplicates from backwards steps. It does not change source files or establish Euler axis mapping/rotation convention.",
        "outputs": {
            "subject_summary": summary_path.name,
            "inventory": inventory_path.name,
            "event_examples": examples_path.name,
        },
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    readme_path.write_text(
        "CAPPIMU TIMESTAMP + ORIENTATION QUALITY AUDIT\n"
        "==============================================\n\n"
        "This program separates dt == 0 duplicate timestamps from dt < 0 backwards timestamps.\n"
        "It also counts positive gaps above the selected threshold (default 500 raw units).\n"
        "Timestamp units are not assumed. No source data is changed.\n\n"
        "Files:\n"
        "- timestamp_orientation_quality_by_subject.csv: detailed per-subject counts and intervals\n"
        "- timestamp_audit_inventory.csv: processed/missing/error inventory\n"
        "- timestamp_event_examples.csv: largest backward steps, largest positive gaps, and first duplicate examples\n"
        "- timestamp_orientation_audit_summary.json: aggregate totals\n\n"
        "The 500 threshold is expressed in raw timestamp units, not automatically milliseconds.\n",
        encoding="utf-8",
    )

    print("\nCAPPIMU TIMESTAMP + ORIENTATION QUALITY AUDIT COMPLETED")
    print("-------------------------------------------------------")
    print(f"Processed subjects:                 {result['processed_subjects']}")
    print(f"Missing/error subjects:             {result['missing_subjects'] + result['error_subjects']}")
    print(f"Rows scanned:                       {total_rows:,}")
    print(f"Duplicate timestamp steps (dt=0):   {total_dup:,}")
    print(f"Backward timestamp steps (dt<0):    {total_back:,}")
    print(f"Non-increasing steps total:          {total_dup + total_back:,} ({result['nonincreasing_percent_of_finite_pairs']:.3f}%)" if total_pairs else "Non-increasing steps total: n/a")
    print(f"Positive gaps > {args.gap_threshold:g} raw units:         {total_gaps:,}")
    print(f"Potential segment boundaries:        {total_breaks:,}")
    print(f"ey outside provisional +/-90:        {result['orientation_rows_ey_outside_provisional_pm90']}")
    print(f"Outputs: {outdir.resolve()}")
    return 0 if result["processed_subjects"] == args.expected_subjects and result["missing_subjects"] == 0 and result["error_subjects"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
