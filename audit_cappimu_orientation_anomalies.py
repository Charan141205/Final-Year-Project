#!/usr/bin/env python3
"""Read-only audit of CAPPIMU Euler-angle outliers and local timestamp continuity.

This script does NOT assume the ex/ey/ez-to-roll/pitch/yaw mapping is confirmed,
does NOT rotate IMU vectors, and does NOT modify source files or models.
It provisionally checks ex/ez against +/-180 degrees and ey against +/-90 degrees.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ANGLE_COLS = ["knee_r(ex)", "knee_r(ey)", "knee_r(ez)"]
IMU_COLS = [
    "knee_r(ax)", "knee_r(ay)", "knee_r(az)",
    "knee_r(wx)", "knee_r(wy)", "knee_r(wz)",
]
REQUIRED = ["time", *ANGLE_COLS, *IMU_COLS]


def wrapped_delta(values: np.ndarray) -> np.ndarray:
    d = np.diff(values)
    return (d + 180.0) % 360.0 - 180.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", required=True, help="Folder containing subject_1, subject_2, ...")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--expected-subjects", type=int, default=30)
    ap.add_argument("--context-rows", type=int, default=5, help="Rows before/after each ey outlier")
    ap.add_argument("--provisional-gap-threshold", type=float, default=500.0,
                    help="Flag timestamp differences above this many raw timestamp units")
    args = ap.parse_args()

    root = Path(args.raw_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory, summaries, contexts = [], [], []
    total_rows = 0
    missing = errors = 0

    for sid in range(1, args.expected_subjects + 1):
        subject = f"subject_{sid}"
        path = root / subject / "joint" / "insole_knee_r.csv"
        print(f"[{sid:02d}/{args.expected_subjects:02d}] {subject}: {path}")
        if not path.is_file():
            missing += 1
            inventory.append({"subject_id": subject, "status": "missing", "rows_scanned": 0, "file_path": str(path)})
            continue
        try:
            # Read only the required columns; preserve original row order.
            frame = pd.read_csv(path, usecols=lambda c: c in REQUIRED, low_memory=False)
            absent = [c for c in REQUIRED if c not in frame.columns]
            if absent:
                raise ValueError(f"Missing required columns: {absent}")
            for c in REQUIRED:
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
            n = len(frame)
            total_rows += n
            inventory.append({"subject_id": subject, "status": "processed", "rows_scanned": n, "file_path": str(path)})

            ey = frame["knee_r(ey)"].to_numpy(dtype=float)
            ex = frame["knee_r(ex)"].to_numpy(dtype=float)
            ez = frame["knee_r(ez)"].to_numpy(dtype=float)
            t = frame["time"].to_numpy(dtype=float)
            ey_bad = np.isfinite(ey) & ((ey < -90.0) | (ey > 90.0))

            row = {"subject_id": subject, "rows": n}
            for col, arr, bound in [("ex", ex, 180.0), ("ey", ey, 90.0), ("ez", ez, 180.0)]:
                finite = np.isfinite(arr)
                vals = arr[finite]
                row[f"{col}_missing_or_nonfinite"] = int((~finite).sum())
                row[f"{col}_min"] = float(vals.min()) if vals.size else np.nan
                row[f"{col}_max"] = float(vals.max()) if vals.size else np.nan
                row[f"{col}_outside_provisional_range"] = int((np.abs(vals) > bound).sum())
                if vals.size > 1:
                    raw_d = np.diff(vals)
                    wd = wrapped_delta(vals)
                    row[f"{col}_raw_jump_gt180"] = int((np.abs(raw_d) > 180.0).sum())
                    row[f"{col}_wrapped_step_gt30"] = int((np.abs(wd) > 30.0).sum())
                    row[f"{col}_wrapped_step_gt90"] = int((np.abs(wd) > 90.0).sum())
                    row[f"{col}_wrapped_abs_step_q99_deg"] = float(np.quantile(np.abs(wd), .99))
                else:
                    row[f"{col}_raw_jump_gt180"] = 0
                    row[f"{col}_wrapped_step_gt30"] = 0
                    row[f"{col}_wrapped_step_gt90"] = 0
                    row[f"{col}_wrapped_abs_step_q99_deg"] = np.nan

            finite_t = np.isfinite(t)
            dt = np.diff(t[finite_t]) if finite_t.sum() > 1 else np.array([])
            row["time_missing_or_nonfinite"] = int((~finite_t).sum())
            row["time_duplicate_or_nonincreasing_steps"] = int((dt <= 0).sum())
            row["time_positive_step_q01_raw_units"] = float(np.quantile(dt[dt > 0], .01)) if np.any(dt > 0) else np.nan
            row["time_positive_step_median_raw_units"] = float(np.median(dt[dt > 0])) if np.any(dt > 0) else np.nan
            row["time_positive_step_q99_raw_units"] = float(np.quantile(dt[dt > 0], .99)) if np.any(dt > 0) else np.nan
            row["time_gap_gt_threshold_count"] = int((dt > args.provisional_gap_threshold).sum())
            row["ey_outlier_count"] = int(ey_bad.sum())

            # Inspect a local neighborhood around each out-of-range ey sample.
            bad_idx = np.flatnonzero(ey_bad)
            for event_no, idx in enumerate(bad_idx, start=1):
                lo = max(0, idx - args.context_rows)
                hi = min(n, idx + args.context_rows + 1)
                for j in range(lo, hi):
                    rec = {"subject_id": subject, "event_no": event_no, "outlier_row_index_0based": int(idx),
                           "row_index_0based": int(j), "relative_to_outlier": int(j - idx),
                           "is_outlier_row": bool(j == idx)}
                    for c in REQUIRED:
                        rec[c] = frame.iloc[j][c]
                    contexts.append(rec)

            summaries.append(row)
            print(f"    rows={n:,}, ey_outliers={int(ey_bad.sum())}, time_nonincreasing={row['time_duplicate_or_nonincreasing_steps']:,}")
        except Exception as exc:
            errors += 1
            inventory.append({"subject_id": subject, "status": "error", "rows_scanned": 0, "file_path": str(path), "error": str(exc)})
            print(f"    ERROR: {exc}", file=sys.stderr)

    inv = pd.DataFrame(inventory)
    sm = pd.DataFrame(summaries)
    cx = pd.DataFrame(contexts)
    inv.to_csv(out / "orientation_anomaly_inventory.csv", index=False)
    sm.to_csv(out / "orientation_anomaly_subject_summary.csv", index=False)
    cx.to_csv(out / "ey_outlier_local_context.csv", index=False)

    ey_total = int(sm["ey_outlier_count"].sum()) if not sm.empty else 0
    summary = {
        "raw_root": str(root), "expected_subjects": args.expected_subjects,
        "processed_subjects": int((inv.status == "processed").sum()),
        "missing_subjects": missing, "error_subjects": errors,
        "rows_scanned": int(total_rows), "ey_outlier_count": ey_total,
        "ey_outlier_percent_of_rows": (100.0 * ey_total / total_rows) if total_rows else None,
        "provisional_range_assumptions": {"knee_r(ex)": "+/-180 degrees", "knee_r(ey)": "+/-90 degrees", "knee_r(ez)": "+/-180 degrees"},
        "timestamp_gap_threshold_raw_units": args.provisional_gap_threshold,
        "warning": "This is a range, local-context and timestamp diagnostic only. It does not confirm Euler-axis mapping, units, rotation convention, reference frame, or justify rotating IMU vectors."
    }
    (out / "orientation_anomaly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README_FIRST.txt").write_text(
        "CAPPIMU ORIENTATION ANOMALY AUDIT\n"
        "=================================\n\n"
        "Files:\n"
        "- orientation_anomaly_subject_summary.csv: per-subject angle and timestamp statistics\n"
        "- ey_outlier_local_context.csv: +/- context rows around every ey value outside provisional +/-90 deg\n"
        "- orientation_anomaly_inventory.csv: processed/missing/error inventory\n"
        "- orientation_anomaly_summary.json: run totals and assumptions\n\n"
        "The +/-180, +/-90, +/-180 degree bounds are provisional. This script does not establish axis mapping or rotation convention and does not rotate any vector. Timestamp differences are reported in raw timestamp units, not assumed milliseconds.\n",
        encoding="utf-8")

    print("\nCAPPIMU ORIENTATION ANOMALY AUDIT COMPLETED")
    print("------------------------------------------")
    print(f"Processed subjects: {int((inv.status == 'processed').sum())}")
    print(f"Missing subjects:   {missing}")
    print(f"Error subjects:     {errors}")
    print(f"Rows scanned:       {total_rows:,}")
    print(f"ey outlier samples: {ey_total} ({summary['ey_outlier_percent_of_rows']:.8f}% of rows if rows exist)" if total_rows else "ey outlier samples: 0")
    print(f"Outputs:            {out}")
    print("No source file or trained model was changed.")
    return 0 if missing == 0 and errors == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
