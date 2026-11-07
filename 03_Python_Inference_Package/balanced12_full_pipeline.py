"""Integrated Balanced-12 Python pipeline.

Supported input modes
---------------------
1) feature-table: consume an existing CAPPIMU 50-sample-window feature table
   containing the 12 Balanced-12 features and sequence metadata. This is the
   recommended first run with the supplied package.
2) raw-cappimu: consume one participant's original insole_knee_r.csv file,
   segment it into contiguous recordings and 50-row windows, calculate the
   12 features, then run inference and causal temporal confirmation.

The model and operating threshold are loaded from the adjacent manifest.
No model retraining or threshold adjustment is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from balanced12_feature_calculator import (
    FEATURE_ORDER,
    calculate_balanced12_from_raw_window,
)
from balanced12_temporal_confirmation import Balanced12TemporalSimulator

HERE = Path(__file__).resolve().parent
DEFAULT_MIN_CONSECUTIVE = 2
RAW_RECORDING_GAP_MS = 500.0
RAW_SEGMENT_GAP_MS = 100.0
WINDOW_SIZE = 50


def _contiguous_group_ids(labels: pd.Series, timestamps: pd.Series,
                          gap_threshold_ms: float) -> pd.Series:
    """Match label/backward-clock/gap segmentation while preserving row order."""
    clean = labels.fillna("unknown").astype(str).str.strip().str.lower()
    ts = pd.to_numeric(timestamps, errors="coerce")
    dt = ts.diff()
    changed = clean.ne(clean.shift(1)) | ((dt < 0) | (dt > gap_threshold_ms)).fillna(False)
    if len(changed):
        changed.iloc[0] = True
    return changed.cumsum().astype(int)


def raw_cappimu_to_feature_windows(raw_csv: Path, subject_id: str) -> pd.DataFrame:
    """Create feature windows from one original CAPPIMU participant CSV.

    Recording boundaries: label change, backwards timestamp, or >500 ms gap.
    Within each recording, segment boundaries: label change, backwards
    timestamp, or >100 ms gap. Complete non-overlapping 50-row windows are
    calculated; incomplete tails are ignored.

    The >100 ms inner segmentation was checked against the stored subject_8,
    recording 1 example (175/175 windows, all 12 features; maximum absolute
    difference ~9.2e-9). For other participants, compare derived row counts and
    sample outputs with the supplied feature table before treating this raw
    route as fully reproduced.
    """
    raw_csv = Path(raw_csv)
    if not raw_csv.is_file():
        raise FileNotFoundError(f"Raw CAPPIMU CSV not found: {raw_csv}")

    raw = pd.read_csv(raw_csv)
    required = [
        "time", "label",
        "insole_r(p1)", "insole_r(p2)", "insole_r(p3)", "insole_r(p4)",
        "insole_r(p5)", "insole_r(p6)", "insole_r(p7)", "insole_r(p8)",
        "knee_r(ax)", "knee_r(ay)", "knee_r(az)",
        "knee_r(wx)", "knee_r(wy)", "knee_r(wz)",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Raw file is missing required CAPPIMU columns: {missing}")
    if raw.empty:
        raise ValueError("Raw CAPPIMU file contains no rows")

    # Keep source row order. The CAPPIMU processing workflow does not reorder rows.
    raw = raw.reset_index(drop=True)
    raw["time"] = pd.to_numeric(raw["time"], errors="coerce")
    raw["label"] = raw["label"].astype("string").str.strip().str.lower()

    recording_ids = _contiguous_group_ids(raw["label"], raw["time"], RAW_RECORDING_GAP_MS)
    records: list[dict] = []

    for block_index, block_key in enumerate(pd.unique(recording_ids), start=1):
        block = raw.loc[recording_ids == block_key].reset_index(drop=True)
        activity = str(block["label"].iloc[0]) if pd.notna(block["label"].iloc[0]) else "unknown"
        recording_id = f"{subject_id}_block{block_index:02d}_{activity}"

        # The original feature table can contain multiple segments per recording.
        # Segment numbering includes short segments that may not yield a full window.
        segment_ids = _contiguous_group_ids(block["label"], block["time"], RAW_SEGMENT_GAP_MS)
        window_id = 0

        output_segment_id = 0
        for segment_key in pd.unique(segment_ids):
            segment = block.loc[segment_ids == segment_key].reset_index(drop=True)
            complete_windows = len(segment) // WINDOW_SIZE
            if complete_windows == 0:
                continue
            # The stored feature table numbers only segments that yield at least
            # one complete feature window; short empty segments do not consume an ID.
            output_segment_id += 1
            for start in range(0, len(segment) - WINDOW_SIZE + 1, WINDOW_SIZE):
                window = segment.iloc[start:start + WINDOW_SIZE].copy()
                # Fail loudly on a corrupt/incomplete numeric window instead of
                # silently fabricating a prediction.
                features = calculate_balanced12_from_raw_window(window)
                t = pd.to_numeric(window["time"], errors="coerce").to_numpy(dtype=np.float64)
                if not np.isfinite(t).all():
                    raise ValueError(
                        f"Non-finite timestamps in {recording_id}, segment {segment_key}, "
                        f"window starting at local row {start}"
                    )
                diffs = np.diff(t)
                window_id += 1
                row = {
                    "subject_id": subject_id,
                    "recording_id": recording_id,
                    "segment_id": int(output_segment_id),
                    "window_id": int(window_id),
                    "activity_label": activity,
                    "window_samples": WINDOW_SIZE,
                    "start_time_ms": float(t[0]),
                    "end_time_ms": float(t[-1]),
                    "elapsed_time_ms": float(t[-1] - t[0]),
                    "duplicate_timestamp_steps": int(np.sum(diffs == 0)),
                    "timestamp_gap_steps": int(np.sum(diffs > RAW_SEGMENT_GAP_MS)),
                }
                row.update(features)
                records.append(row)

    if not records:
        raise ValueError(
            "No complete 50-row windows were created. Check the raw file and sampling data."
        )
    return pd.DataFrame.from_records(records)


def _validate_feature_table(frame: pd.DataFrame) -> None:
    metadata = ["subject_id", "recording_id", "segment_id", "window_id"]
    missing_meta = [c for c in metadata if c not in frame.columns]
    missing_feat = [c for c in FEATURE_ORDER if c not in frame.columns]
    if missing_meta or missing_feat:
        raise ValueError(
            f"Input feature table is missing metadata {missing_meta} "
            f"or model features {missing_feat}"
        )
    try:
        values = frame[FEATURE_ORDER].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("The 12 model features must all be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("The feature table contains NaN or infinite model values")


def run_pipeline(frame: pd.DataFrame, min_consecutive: int = DEFAULT_MIN_CONSECUTIVE) -> pd.DataFrame:
    """Run fixed Balanced-12 inference followed by causal temporal confirmation."""
    _validate_feature_table(frame)
    simulator = Balanced12TemporalSimulator(min_consecutive=min_consecutive)
    return simulator.predict_dataframe(frame)


def write_outputs(result: pd.DataFrame, output_dir: Path, mode: str,
                  min_consecutive: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "balanced12_full_pipeline_predictions.csv"
    alerts_path = output_dir / "balanced12_alert_events.csv"
    summary_path = output_dir / "balanced12_full_pipeline_summary.json"

    result.to_csv(predictions_path, index=False)
    result.loc[result["alert_trigger"] == 1].to_csv(alerts_path, index=False)

    manifest_path = HERE / "trained_models" / "balanced12_decision_tree_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "pipeline": "Balanced-12 inference + causal temporal confirmation",
        "input_mode": mode,
        "windows_processed": int(len(result)),
        "raw_positive_windows": int(result["raw_predicted_binary"].sum()),
        "pending_positive_windows": int(result["pending_positive"].sum()),
        "confirmed_fall_windows": int(result["confirmed_fall"].sum()),
        "alert_trigger_events": int(result["alert_trigger"].sum()),
        "temporal_rule": f"{min_consecutive}_consecutive_positive_windows",
        "threshold": float(manifest["selected_operating_threshold"]),
        "limitations": [
            "This is an offline Python simulation using CAPPIMU data, not live footwear sensing.",
            "CAPPIMU pressure inputs are regional insole proxies, not actual shoe FSR measurements.",
            "CAPPIMU IMU is knee-mounted, not shoe-mounted.",
            "Labels are activity/window-level, not precise fall-onset annotations.",
            "A confirmed alert is a model output, not a verified real-world fall.",
        ],
        "prediction_csv": str(predictions_path),
        "alert_events_csv": str(alerts_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBALANCED-12 FULL PIPELINE COMPLETED")
    print("----------------------------------")
    print(f"Input mode:                 {mode}")
    print(f"Windows processed:          {len(result):,}")
    print(f"Raw positive windows:       {int(result['raw_predicted_binary'].sum()):,}")
    print(f"Pending positive windows:   {int(result['pending_positive'].sum()):,}")
    print(f"Confirmed fall windows:    {int(result['confirmed_fall'].sum()):,}")
    print(f"Alert trigger events:       {int(result['alert_trigger'].sum()):,}")
    print(f"Predictions saved to:       {predictions_path}")
    print(f"Alert events saved to:      {alerts_path}")
    print(f"Summary saved to:           {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated Balanced-12 feature-window inference and temporal confirmation"
    )
    parser.add_argument(
        "--mode", choices=["feature-table", "raw-cappimu"], default="feature-table",
        help="Use an existing feature-window CSV or a raw CAPPIMU participant CSV.",
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output-dir", default="balanced12_full_pipeline_output")
    parser.add_argument("--split", choices=["train", "validation", "test"], default=None,
                        help="Optional split filter for feature-table mode only")
    parser.add_argument("--subject-id", default=None,
                        help="Required for raw-cappimu mode, e.g. subject_8")
    parser.add_argument("--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.mode == "feature-table":
        frame = pd.read_csv(input_path)
        if args.split:
            if "split" not in frame.columns:
                raise ValueError("--split was supplied, but the input has no 'split' column")
            frame = frame.loc[frame["split"] == args.split].copy()
            if frame.empty:
                raise ValueError(f"No rows found for split '{args.split}'")
    else:
        if not args.subject_id:
            raise ValueError("--subject-id is required for --mode raw-cappimu")
        if args.split:
            raise ValueError("--split applies only to --mode feature-table")
        frame = raw_cappimu_to_feature_windows(input_path, args.subject_id)
        print(f"Created {len(frame):,} non-overlapping {WINDOW_SIZE}-sample feature windows")

    result = run_pipeline(frame, min_consecutive=args.min_consecutive)
    write_outputs(result, Path(args.output_dir), args.mode, args.min_consecutive)


if __name__ == "__main__":
    main()
