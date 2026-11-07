#!/usr/bin/env python3
"""Read-only Euler-angle sanity scan for CAPPIMU joint recordings.

Assumed provisional axis mapping for diagnostic purposes only:
  knee_r(ex) -> X angle / roll, nominal range +/-180 degrees
  knee_r(ey) -> Y angle / pitch, nominal range +/-90 degrees
  knee_r(ez) -> Z angle / yaw, nominal range +/-180 degrees

This script does NOT rotate, normalize, rewrite, or modify any input data.
The mapping must be confirmed against the exact CAPPIMU export/device configuration
before using the angles for coordinate transformations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ANGLE_LIMITS = {
    "knee_r(ex)": 180.0,
    "knee_r(ey)": 90.0,
    "knee_r(ez)": 180.0,
}
REQUIRED = ["time", *ANGLE_LIMITS.keys(), "knee_r(ax)", "knee_r(ay)", "knee_r(az)",
            "knee_r(wx)", "knee_r(wy)", "knee_r(wz)"]


def locate_file(root: Path, subject: int) -> Path | None:
    candidates = [
        root / f"subject_{subject}" / "joint" / "insole_knee_r.csv",
        root / f"subject_{subject}" / "joint" / "insole_knee_r.csv.gz",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def wrapped_delta(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def scan_subject(path: Path, subject_id: str, chunksize: int, sample_every: int,
                 max_examples: int) -> tuple[dict, list[dict]]:
    usecols = list(ANGLE_LIMITS) + ["time", "knee_r(ax)", "knee_r(ay)", "knee_r(az)",
                                    "knee_r(wx)", "knee_r(wy)", "knee_r(wz)"]
    counts = {c: 0 for c in usecols}
    finite = {c: 0 for c in usecols}
    mins = {c: math.inf for c in usecols}
    maxs = {c: -math.inf for c in usecols}
    sums = {c: 0.0 for c in usecols}
    sums2 = {c: 0.0 for c in usecols}
    out_count = {c: 0 for c in ANGLE_LIMITS}
    angle_samples: Dict[str, list[np.ndarray]] = {c: [] for c in ANGLE_LIMITS}
    prev_angle = {c: None for c in ANGLE_LIMITS}
    prev_time = None
    dt_samples: list[np.ndarray] = []
    raw_jump_count = {c: 0 for c in ANGLE_LIMITS}
    wrapped_jump_count_30 = {c: 0 for c in ANGLE_LIMITS}
    delta_samples: Dict[str, list[np.ndarray]] = {c: [] for c in ANGLE_LIMITS}
    examples: list[dict] = []
    row_offset = 0

    reader = pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize)
    for chunk in reader:
        # Numeric conversion is explicit; source files remain untouched.
        for c in usecols:
            if c not in chunk.columns:
                chunk[c] = np.nan
            x = pd.to_numeric(chunk[c], errors="coerce").to_numpy(dtype=float)
            n = len(x)
            counts[c] += n
            mask = np.isfinite(x)
            xx = x[mask]
            finite[c] += len(xx)
            if len(xx):
                mins[c] = min(mins[c], float(np.min(xx)))
                maxs[c] = max(maxs[c], float(np.max(xx)))
                sums[c] += float(np.sum(xx, dtype=np.float64))
                sums2[c] += float(np.sum(xx * xx, dtype=np.float64))
            if c in ANGLE_LIMITS:
                lim = ANGLE_LIMITS[c]
                out_count[c] += int(np.count_nonzero(mask & (np.abs(x) > lim + 1e-6)))
                # Deterministic sparse sample for quantiles; no random sampling.
                idx = np.arange(n) + row_offset
                take = mask & (idx % sample_every == 0)
                if np.any(take):
                    angle_samples[c].append(x[take])
                # Compute wrapped angular increments, preserving chunk boundary.
                good_indices = np.flatnonzero(mask)
                if len(good_indices):
                    vals = x[good_indices]
                    if prev_angle[c] is not None:
                        diffs = np.diff(np.r_[prev_angle[c], vals])
                        if len(diffs):
                            raw_jump_count[c] += int(np.count_nonzero(np.abs(diffs) > 180.0))
                            wd = np.abs(wrapped_delta(diffs))
                            wrapped_jump_count_30[c] += int(np.count_nonzero(wd > 30.0))
                            delta_samples[c].append(wd[::max(1, len(wd)//2000)])
                    prev_angle[c] = float(vals[-1])
        # Timestamp interval summaries, preserving chunk boundary.
        if "time" in chunk.columns:
            t = pd.to_numeric(chunk["time"], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(t)
            tv = t[m]
            if len(tv):
                dif = np.diff(np.r_[prev_time, tv]) if prev_time is not None else np.diff(tv)
                if len(dif):
                    dt_samples.append(dif[::max(1, len(dif)//2000)])
                prev_time = float(tv[-1])
        # Record a small number of pitch-range exceptions for inspection.
        if "knee_r(ey)" in chunk.columns and len(examples) < max_examples:
            ey = pd.to_numeric(chunk["knee_r(ey)"], errors="coerce").to_numpy(dtype=float)
            idxs = np.flatnonzero(np.isfinite(ey) & (np.abs(ey) > 90.0))
            for ix in idxs[:max_examples - len(examples)]:
                rec = {"subject_id": subject_id, "row_index_0based": row_offset + int(ix)}
                for c in ["time", "knee_r(ex)", "knee_r(ey)", "knee_r(ez)",
                          "knee_r(ax)", "knee_r(ay)", "knee_r(az)",
                          "knee_r(wx)", "knee_r(wy)", "knee_r(wz)"]:
                    if c in chunk.columns:
                        rec[c] = pd.to_numeric(pd.Series([chunk.iloc[ix][c]]), errors="coerce").iloc[0]
                examples.append(rec)
        row_offset += len(chunk)

    out = {"subject_id": subject_id, "file_path": str(path), "rows_scanned": row_offset}
    for c in ANGLE_LIMITS:
        n = counts[c]
        nf = finite[c]
        smp = np.concatenate(angle_samples[c]) if angle_samples[c] else np.array([], dtype=float)
        ds = np.concatenate(delta_samples[c]) if delta_samples[c] else np.array([], dtype=float)
        out[f"{c}_finite"] = nf
        out[f"{c}_missing_or_non_numeric"] = n - nf
        out[f"{c}_outside_nominal_range_count"] = out_count[c]
        out[f"{c}_outside_nominal_range_percent"] = (100.0 * out_count[c] / nf) if nf else np.nan
        out[f"{c}_min"] = mins[c] if nf else np.nan
        out[f"{c}_max"] = maxs[c] if nf else np.nan
        out[f"{c}_sample_q01"] = float(np.quantile(smp, .01)) if len(smp) else np.nan
        out[f"{c}_sample_median"] = float(np.quantile(smp, .50)) if len(smp) else np.nan
        out[f"{c}_sample_q99"] = float(np.quantile(smp, .99)) if len(smp) else np.nan
        out[f"{c}_raw_wrap_jump_count_gt180"] = raw_jump_count[c]
        out[f"{c}_wrapped_step_gt30deg_count"] = wrapped_jump_count_30[c]
        out[f"{c}_wrapped_abs_step_q99_deg"] = float(np.quantile(ds, .99)) if len(ds) else np.nan
    if dt_samples:
        dt = np.concatenate(dt_samples)
        out["time_step_median_raw_units"] = float(np.median(dt))
        out["time_step_q01_raw_units"] = float(np.quantile(dt, .01))
        out["time_step_q99_raw_units"] = float(np.quantile(dt, .99))
    else:
        out["time_step_median_raw_units"] = np.nan
        out["time_step_q01_raw_units"] = np.nan
        out["time_step_q99_raw_units"] = np.nan
    return out, examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate CAPPIMU Euler-angle channel ranges and continuity (read-only).")
    ap.add_argument("--raw-root", required=True, help="Folder containing subject_1 ... subject_30")
    ap.add_argument("--output-dir", required=True, help="Output folder")
    ap.add_argument("--expected-subjects", type=int, default=30)
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--sample-every", type=int, default=25, help="Use every Nth row for quantile estimates")
    ap.add_argument("--max-examples-per-subject", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.raw_root).expanduser()
    outdir = Path(args.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    summaries, inventory, examples = [], [], []
    for sid in range(1, args.expected_subjects + 1):
        f = locate_file(root, sid)
        name = f"subject_{sid}"
        if f is None:
            inventory.append({"subject_id": name, "status": "missing", "file_path": ""})
            print(f"[{sid:02d}/{args.expected_subjects:02d}] {name}: MISSING")
            continue
        try:
            row, ex = scan_subject(f, name, args.chunksize, args.sample_every, args.max_examples_per_subject)
            summaries.append(row)
            examples.extend(ex)
            inventory.append({"subject_id": name, "status": "processed", "file_path": str(f),
                              "rows_scanned": row["rows_scanned"]})
            pitch_out = row.get("knee_r(ey)_outside_nominal_range_percent", float("nan"))
            print(f"[{sid:02d}/{args.expected_subjects:02d}] {name}: rows={row['rows_scanned']:,}, "
                  f"ey_outside_+/-90={pitch_out:.5f}%")
        except Exception as e:
            inventory.append({"subject_id": name, "status": "error", "file_path": str(f), "error": repr(e)})
            print(f"[{sid:02d}/{args.expected_subjects:02d}] {name}: ERROR {e}")

    pd.DataFrame(summaries).to_csv(outdir / "euler_angle_validation_by_subject.csv", index=False)
    pd.DataFrame(inventory).to_csv(outdir / "euler_angle_validation_inventory.csv", index=False)
    pd.DataFrame(examples).to_csv(outdir / "ey_out_of_range_examples.csv", index=False)
    total_rows = int(sum(x.get("rows_scanned", 0) for x in summaries))
    total_missing = sum(x.get("status") == "missing" for x in inventory)
    total_errors = sum(x.get("status") == "error" for x in inventory)
    overall = {
        "raw_root": str(root),
        "expected_subjects": args.expected_subjects,
        "processed_subjects": len(summaries),
        "missing_subjects": int(total_missing),
        "error_subjects": int(total_errors),
        "rows_scanned": total_rows,
        "provisional_mapping_for_range_check": {"knee_r(ex)": "+/-180 deg", "knee_r(ey)": "+/-90 deg", "knee_r(ez)": "+/-180 deg"},
        "important_note": "This is a diagnostic range/continuity check only. The nominal range mapping is provisional (X/roll, Y/pitch, Z/yaw) based on the BWT901CL datasheet. Do not rotate vectors until CAPPIMU export mapping and device configuration are confirmed."
    }
    (outdir / "euler_angle_validation_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (outdir / "README_FIRST.txt").write_text(
        "CAPPIMU EULER ANGLE VALIDATION\n"
        "==============================\n\n"
        "Read-only scan of ex/ey/ez angle ranges, missing values, wrapped step sizes, and timestamps.\n"
        "The +/-180, +/-90, +/-180 degree checks assume ex=X/roll, ey=Y/pitch, ez=Z/yaw provisionally.\n"
        "The manufacturer datasheet specifies BWT901CL angle output in degrees and Z-Y-X order,\n"
        "but the exact CAPPIMU column-to-angle export mapping must be confirmed before rotation.\n"
        "No input recording or trained model is modified.\n",
        encoding="utf-8"
    )
    print("\nCAPPIMU EULER ANGLE VALIDATION COMPLETED")
    print("---------------------------------------")
    print(f"Processed subjects: {len(summaries)}")
    print(f"Missing subjects:   {total_missing}")
    print(f"Error subjects:     {total_errors}")
    print(f"Rows scanned:       {total_rows:,}")
    print(f"Outputs:            {outdir}")

if __name__ == "__main__":
    main()
