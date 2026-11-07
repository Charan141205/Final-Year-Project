#!/usr/bin/env python3
"""Scan CAPPIMU knee-IMU Euler/acceleration/gyroscope fields without changing data.

Example:
  python check_cappimu_orientation_fields.py \
      --raw-root "C:\\Users\\lagis\\Downloads\\data\\data" \
      --output-dir ".\\orientation_channel_check" \
      --expected-subjects 30

The script reads only the requested numeric channels in chunks. It does not
rotate axes, normalize features, train a model, or modify source CSV files.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

CHANNELS = [
    "knee_r(ex)", "knee_r(ey)", "knee_r(ez)",
    "knee_r(ax)", "knee_r(ay)", "knee_r(az)",
    "knee_r(wx)", "knee_r(wy)", "knee_r(wz)",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True, help="Folder containing subject_1, subject_2, ...")
    p.add_argument("--output-dir", required=True, help="Folder for the orientation scan outputs")
    p.add_argument("--expected-subjects", type=int, default=30)
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--sample-per-chunk", type=int, default=2_000,
                   help="Random sample per chunk used only for approximate quantiles")
    p.add_argument("--max-subjects", type=int, default=None,
                   help="Optional small smoke test; omit to scan all participants")
    return p.parse_args()


def numeric_summary(values: List[np.ndarray], n: int, missing: int,
                    total_sum: float, total_sumsq: float,
                    minimum: float, maximum: float) -> Dict[str, float]:
    sample = np.concatenate(values) if values else np.array([], dtype=float)
    sample = sample[np.isfinite(sample)]
    mean = total_sum / n if n else float("nan")
    variance = max(0.0, total_sumsq / n - mean * mean) if n else float("nan")
    row = {
        "finite_values": int(n),
        "missing_or_non_numeric": int(missing),
        "missing_percent": 100.0 * missing / (n + missing) if (n + missing) else float("nan"),
        "mean": mean,
        "std_population": math.sqrt(variance) if n else float("nan"),
        "min": minimum if n else float("nan"),
        "max": maximum if n else float("nan"),
        "sample_q01": float(np.quantile(sample, 0.01)) if len(sample) else float("nan"),
        "sample_median": float(np.quantile(sample, 0.50)) if len(sample) else float("nan"),
        "sample_q99": float(np.quantile(sample, 0.99)) if len(sample) else float("nan"),
        "quantile_sample_n": int(len(sample)),
    }
    return row


def scan_file(subject_dir: Path, csv_path: Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    header = pd.read_csv(csv_path, nrows=0)
    missing_cols = [c for c in CHANNELS if c not in header.columns]
    if missing_cols:
        return [], {
            "subject_id": subject_dir.name,
            "file_path": str(csv_path),
            "status": "missing_required_orientation_or_imu_columns",
            "missing_columns": ";".join(missing_cols),
            "rows_scanned": 0,
        }

    stats = {
        c: {"n": 0, "missing": 0, "sum": 0.0, "sumsq": 0.0,
            "min": float("inf"), "max": float("-inf"), "samples": []}
        for c in CHANNELS
    }
    rows_scanned = 0
    rng = np.random.default_rng(20261106)

    for chunk in pd.read_csv(csv_path, usecols=CHANNELS, chunksize=args.chunksize):
        rows_scanned += len(chunk)
        for col in CHANNELS:
            vals = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            st = stats[col]
            st["n"] += int(finite.size)
            st["missing"] += int(vals.size - finite.size)
            if finite.size:
                st["sum"] += float(finite.sum())
                st["sumsq"] += float(np.dot(finite, finite))
                st["min"] = min(st["min"], float(finite.min()))
                st["max"] = max(st["max"], float(finite.max()))
                take = min(args.sample_per_chunk, finite.size)
                if take:
                    idx = rng.choice(finite.size, size=take, replace=False)
                    st["samples"].append(finite[idx])

    rows = []
    for col in CHANNELS:
        st = stats[col]
        summary = numeric_summary(
            st["samples"], st["n"], st["missing"], st["sum"], st["sumsq"], st["min"], st["max"]
        )
        rows.append({
            "subject_id": subject_dir.name,
            "channel": col,
            **summary,
        })

    inventory = {
        "subject_id": subject_dir.name,
        "file_path": str(csv_path),
        "status": "processed",
        "rows_scanned": int(rows_scanned),
        "columns_in_source": int(len(header.columns)),
        "file_size_bytes": int(csv_path.stat().st_size),
        "missing_columns": "",
    }
    return rows, inventory


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists() or not raw_root.is_dir():
        print(f"ERROR: raw-root folder does not exist: {raw_root}", file=sys.stderr)
        return 2

    subject_dirs = sorted(
        [p for p in raw_root.glob("subject_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 10**9,
    )
    if args.max_subjects is not None:
        subject_dirs = subject_dirs[:args.max_subjects]

    if not subject_dirs:
        print(f"ERROR: no subject_* folders found under {raw_root}", file=sys.stderr)
        return 2

    all_rows: list[dict] = []
    inventory: list[dict] = []
    for i, subject_dir in enumerate(subject_dirs, start=1):
        csv_path = subject_dir / "joint" / "insole_knee_r.csv"
        print(f"[{i:02d}/{len(subject_dirs):02d}] {subject_dir.name}: {csv_path}")
        if not csv_path.exists():
            inventory.append({
                "subject_id": subject_dir.name,
                "file_path": str(csv_path),
                "status": "file_not_found",
                "rows_scanned": 0,
                "columns_in_source": 0,
                "file_size_bytes": 0,
                "missing_columns": "",
            })
            continue
        try:
            rows, inv = scan_file(subject_dir, csv_path, args)
            all_rows.extend(rows)
            inventory.append(inv)
            print(f"    status={inv['status']}, rows={inv['rows_scanned']:,}")
        except Exception as exc:
            inventory.append({
                "subject_id": subject_dir.name,
                "file_path": str(csv_path),
                "status": f"error: {type(exc).__name__}: {exc}",
                "rows_scanned": 0,
                "columns_in_source": 0,
                "file_size_bytes": int(csv_path.stat().st_size) if csv_path.exists() else 0,
                "missing_columns": "",
            })
            print(f"    ERROR: {type(exc).__name__}: {exc}")

    summary_df = pd.DataFrame(all_rows)
    inventory_df = pd.DataFrame(inventory)
    summary_df.to_csv(out_dir / "orientation_channel_summary.csv", index=False)
    inventory_df.to_csv(out_dir / "orientation_file_inventory.csv", index=False)

    processed = int((inventory_df.status == "processed").sum())
    failed = int(len(inventory_df) - processed)
    result = {
        "raw_root": str(raw_root),
        "expected_subjects": int(args.expected_subjects),
        "subject_folders_found": int(len(subject_dirs)),
        "processed_subjects": processed,
        "missing_or_failed_subjects": failed,
        "total_rows_scanned": int(inventory_df.rows_scanned.sum()),
        "channels_scanned": CHANNELS,
        "outputs": ["orientation_channel_summary.csv", "orientation_file_inventory.csv"],
        "important_note": (
            "Euler-like fields ex/ey/ez were summarized as stored. Their units, rotation order, "
            "axis convention, and reference frame must be confirmed from authoritative dataset/device "
            "documentation before using them to rotate acceleration or gyroscope vectors. No transformation was applied."
        ),
    }
    (out_dir / "orientation_scan_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "README_FIRST.txt").write_text(
        "CAPPIMU ORIENTATION CHANNEL SCAN\n"
        "================================\n\n"
        "This is a read-only inventory of knee_r(ex/ey/ez), accelerometer and gyroscope channels.\n"
        "It does not transform data, normalize features, retrain a model, or modify raw files.\n\n"
        "Do not assume the Euler fields are degrees or use them for a coordinate rotation until\n"
        "their units, Euler order, axis convention and reference frame are verified in official\n"
        "dataset/device documentation.\n",
        encoding="utf-8",
    )

    print("\nCAPPIMU ORIENTATION CHANNEL SCAN COMPLETED")
    print("------------------------------------------")
    print(f"Subject folders found:      {len(subject_dirs)}")
    print(f"Processed subjects:         {processed}")
    print(f"Missing/failed subjects:    {failed}")
    print(f"Rows scanned:               {int(inventory_df.rows_scanned.sum()):,}")
    print(f"Outputs:                    {out_dir}")
    print("No source file or model was changed.")
    return 0 if processed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
