#!/usr/bin/env python3
"""Audit timing inside CAPPIMU's existing 50-row feature windows.

Read-only. Reproduces the current integrated raw pipeline's boundaries:
  1) recording boundary: label change, dt < 0, or dt > 500 ms
  2) within each recording, feature-segment boundary: label change,
     dt < 0, or dt > 100 ms
  3) retain dt == 0 rows; make non-overlapping complete 50-row windows.

CAPPIMU documents its timestamp in milliseconds and sampling rate as 50 Hz.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

WINDOW_SIZE = 50
RECORDING_GAP_MS = 500.0
SEGMENT_GAP_MS = 100.0
EXPECTED_WINDOWS = 110_949


def group_ids(labels: pd.Series, times: pd.Series, gap_ms: float) -> pd.Series:
    clean = labels.astype("string").fillna("unknown").str.strip().str.lower()
    ts = pd.to_numeric(times, errors="coerce")
    dt = ts.diff()
    boundary = clean.ne(clean.shift(1)) | ((dt < 0) | (dt > gap_ms)).fillna(False)
    if len(boundary):
        boundary.iloc[0] = True
    return boundary.cumsum().astype(int)


def percentile(values: np.ndarray, pct: float) -> float | None:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, pct)) if a.size else None


def audit_subject(sid: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.read_csv(path, low_memory=False)
    for col in ("time", "label"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    times = pd.to_numeric(df["time"], errors="coerce")
    labels = df["label"].astype("string").fillna("unknown").str.strip().str.lower()
    dt = times.diff()
    finite_pairs = times.notna() & times.shift(1).notna()
    dup_steps = int(((dt == 0) & finite_pairs).sum())
    back_steps = int(((dt < 0) & finite_pairs).sum())
    gap100_500 = int(((dt > SEGMENT_GAP_MS) & (dt <= RECORDING_GAP_MS) & finite_pairs).sum())
    gap500 = int(((dt > RECORDING_GAP_MS) & finite_pairs).sum())
    label_changes = int((labels != labels.shift(1)).iloc[1:].sum())

    # Exactly mirror raw_cappimu_to_feature_windows() recording and segment grouping.
    recording_ids = group_ids(labels, times, RECORDING_GAP_MS)
    window_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    complete_windows = 0
    tail_rows = 0
    windows_with_dup = 0
    dup_steps_in_windows = 0
    spans: list[float] = []
    unique_counts: list[int] = []
    duplicate_counts: list[int] = []
    seg_count = 0

    # Preserve original row ordering; no sort, de-duplication, interpolation or filtering.
    for block_number, block_key in enumerate(pd.unique(recording_ids), start=1):
        block_idx = np.flatnonzero(recording_ids.to_numpy() == block_key)
        block = df.iloc[block_idx].reset_index(drop=True)
        block_times = pd.to_numeric(block["time"], errors="coerce")
        block_labels = block["label"].astype("string").fillna("unknown").str.strip().str.lower()
        inner_ids = group_ids(block_labels, block_times, SEGMENT_GAP_MS)
        activity = str(block_labels.iloc[0]) if len(block_labels) else "unknown"
        recording_id = f"{sid}_block{block_number:02d}_{activity}"
        for inner_key in pd.unique(inner_ids):
            local_idx = np.flatnonzero(inner_ids.to_numpy() == inner_key)
            seg = block.iloc[local_idx]
            seg_t = pd.to_numeric(seg["time"], errors="coerce").to_numpy(dtype=float)
            seg_label = str(seg["label"].iloc[0]) if len(seg) else "unknown"
            n = len(seg)
            nwin, tail = divmod(n, WINDOW_SIZE)
            tail_rows += tail
            seg_count += 1
            segment_id = seg_count
            segment_rows.append({
                "subject_id": sid,
                "recording_id": recording_id,
                "segment_id": segment_id,
                "label": seg_label,
                "start_source_row_0based": int(block_idx[local_idx[0]]),
                "end_source_row_0based": int(block_idx[local_idx[-1]]),
                "rows": n,
                "complete_50row_windows": nwin,
                "tail_rows_not_windowed": tail,
            })
            for w in range(nwin):
                lo, hi = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
                wt = seg_t[lo:hi]
                wd = np.diff(wt)
                finite = np.isfinite(wt[:-1]) & np.isfinite(wt[1:])
                dup = int(np.sum(finite & (wd == 0)))
                backward = int(np.sum(finite & (wd < 0)))
                pos = wd[finite & (wd > 0)]
                fw = wt[np.isfinite(wt)]
                elapsed = float(fw[-1] - fw[0]) if fw.size >= 2 else np.nan
                uniq = int(pd.Series(wt).nunique(dropna=True))
                complete_windows += 1
                windows_with_dup += int(dup > 0)
                dup_steps_in_windows += dup
                duplicate_counts.append(dup)
                unique_counts.append(uniq)
                if np.isfinite(elapsed):
                    spans.append(elapsed)
                source_start = int(block_idx[local_idx[lo]])
                source_end = int(block_idx[local_idx[hi - 1]])
                window_rows.append({
                    "subject_id": sid,
                    "recording_id": recording_id,
                    "segment_id": segment_id,
                    "window_in_segment": w,
                    "label": seg_label,
                    "start_source_row_0based": source_start,
                    "end_source_row_0based": source_end,
                    "start_timestamp_ms": float(wt[0]) if np.isfinite(wt[0]) else np.nan,
                    "end_timestamp_ms": float(wt[-1]) if np.isfinite(wt[-1]) else np.nan,
                    "elapsed_first_to_last_ms": elapsed,
                    "unique_timestamp_values": uniq,
                    "adjacent_duplicate_steps": dup,
                    "backward_steps_inside_window": backward,
                    "positive_dt_median_ms": float(np.median(pos)) if pos.size else np.nan,
                    "positive_dt_p05_ms": percentile(pos, 5),
                    "positive_dt_p95_ms": percentile(pos, 95),
                })

    summary = {
        "subject_id": sid,
        "status": "processed",
        "rows": int(len(df)),
        "label_changes": label_changes,
        "duplicate_timestamp_steps_dt_eq_0": dup_steps,
        "backward_timestamp_steps_dt_lt_0": back_steps,
        "positive_gaps_100_to_500ms": gap100_500,
        "positive_gaps_over_500ms": gap500,
        "recording_blocks_500ms_rule": int(recording_ids.nunique()),
        "feature_segments_100ms_rule": int(seg_count),
        "complete_50row_windows": int(complete_windows),
        "tail_rows_not_in_complete_windows": int(tail_rows),
        "windows_with_at_least_one_duplicate_step": int(windows_with_dup),
        "duplicate_adjacent_steps_inside_complete_windows": int(dup_steps_in_windows),
        "window_elapsed_ms_p05": percentile(np.asarray(spans), 5),
        "window_elapsed_ms_median": percentile(np.asarray(spans), 50),
        "window_elapsed_ms_p95": percentile(np.asarray(spans), 95),
        "window_elapsed_ms_min": float(np.min(spans)) if spans else None,
        "window_elapsed_ms_max": float(np.max(spans)) if spans else None,
        "window_unique_timestamps_median": percentile(np.asarray(unique_counts), 50),
        "window_duplicate_steps_median": percentile(np.asarray(duplicate_counts), 50),
    }
    return summary, window_rows, segment_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expected-subjects", type=int, default=30)
    args = ap.parse_args()
    root, out = Path(args.raw_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries, windows, segments, errors = [], [], [], []
    for i in range(1, args.expected_subjects + 1):
        sid = f"subject_{i}"
        path = root / sid / "joint" / "insole_knee_r.csv"
        if not path.is_file():
            errors.append({"subject_id": sid, "status": "missing", "error": str(path)})
            print(f"[{i:02d}/{args.expected_subjects:02d}] {sid}: MISSING")
            continue
        try:
            s, w, seg = audit_subject(sid, path)
            summaries.append(s)
            windows.extend(w)
            segments.extend(seg)
            print(f"[{i:02d}/{args.expected_subjects:02d}] {sid}: rows={s['rows']:,}, blocks={s['recording_blocks_500ms_rule']}, segments={s['feature_segments_100ms_rule']}, windows={s['complete_50row_windows']:,}, median_span={s['window_elapsed_ms_median']:.1f} ms, windows_with_duplicates={s['windows_with_at_least_one_duplicate_step']:,}")
        except Exception as exc:
            errors.append({"subject_id": sid, "status": "error", "error": repr(exc)})
            print(f"[{i:02d}/{args.expected_subjects:02d}] {sid}: ERROR: {exc}")

    sdf, wdf, segdf = pd.DataFrame(summaries), pd.DataFrame(windows), pd.DataFrame(segments)
    pd.DataFrame(errors, columns=["subject_id", "status", "error"]).to_csv(out / "window_timing_missing_error_inventory.csv", index=False)
    sdf.to_csv(out / "window_timing_subject_summary.csv", index=False)
    wdf.to_csv(out / "window_timing_all_50row_windows.csv", index=False)
    segdf.to_csv(out / "window_timing_segment_inventory.csv", index=False)
    total_windows = int(sdf["complete_50row_windows"].sum()) if len(sdf) else 0
    summary = {
        "expected_subjects": args.expected_subjects,
        "processed_subjects": int(len(summaries)),
        "missing_or_error_subjects": int(len(errors)),
        "rows_scanned": int(sdf["rows"].sum()) if len(sdf) else 0,
        "recording_gap_boundary_ms": RECORDING_GAP_MS,
        "inner_feature_segment_gap_boundary_ms": SEGMENT_GAP_MS,
        "window_size_rows": WINDOW_SIZE,
        "total_complete_50row_windows": total_windows,
        "historical_reference_windows": EXPECTED_WINDOWS,
        "matches_historical_reference_window_count": bool(total_windows == EXPECTED_WINDOWS),
        "windows_with_at_least_one_duplicate_step": int(wdf["adjacent_duplicate_steps"].gt(0).sum()) if len(wdf) else 0,
        "duplicate_adjacent_steps_inside_complete_windows": int(wdf["adjacent_duplicate_steps"].sum()) if len(wdf) else 0,
        "global_window_elapsed_ms_p05": percentile(wdf["elapsed_first_to_last_ms"].to_numpy(dtype=float), 5) if len(wdf) else None,
        "global_window_elapsed_ms_median": percentile(wdf["elapsed_first_to_last_ms"].to_numpy(dtype=float), 50) if len(wdf) else None,
        "global_window_elapsed_ms_p95": percentile(wdf["elapsed_first_to_last_ms"].to_numpy(dtype=float), 95) if len(wdf) else None,
        "note": "Read-only. Mirrors balanced12_full_pipeline.py: split recording at label change/dt<0/dt>500ms; split feature segments at label change/dt<0/dt>100ms; preserve dt=0; form non-overlapping 50-row windows. Timestamp units are milliseconds per CAPPIMU README.",
    }
    (out / "window_timing_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README_FIRST.txt").write_text(
        "CAPPIMU 50-ROW WINDOW TIMING AUDIT (MATCHES CURRENT PIPELINE)\n"
        "=============================================================\n\n"
        "The script mirrors the project's integrated raw pipeline segmentation:\n"
        "- recording split: label change, dt < 0, or dt > 500 ms\n"
        "- inner feature-segment split: label change, dt < 0, or dt > 100 ms\n"
        "- dt == 0 rows are preserved\n"
        "- complete non-overlapping 50-row windows are measured\n\n"
        "Files: subject summary, all-window timing CSV, segment inventory, missing/error inventory, JSON summary.\n"
        "Check that total_complete_50row_windows equals 110,949. No source CSV or model is modified.\n",
        encoding="utf-8"
    )
    print("\nCAPPIMU 50-ROW WINDOW TIMING AUDIT COMPLETED")
    print("-------------------------------------------")
    print(f"Processed subjects:        {len(summaries)}")
    print(f"Missing/error subjects:    {len(errors)}")
    print(f"Rows scanned:              {int(summary['rows_scanned']):,}")
    print(f"Complete 50-row windows:   {total_windows:,}")
    print(f"Matches 110,949 reference: {summary['matches_historical_reference_window_count']}")
    print(f"Windows with duplicates:   {summary['windows_with_at_least_one_duplicate_step']:,}")
    med = summary["global_window_elapsed_ms_median"]
    print(f"Global median window span: {med:.2f} ms" if med is not None else "Global median window span: n/a")
    print(f"Outputs: {out}")
    print("No source file or trained model was changed.")
    return 0 if not errors else 2

if __name__ == "__main__":
    raise SystemExit(main())
