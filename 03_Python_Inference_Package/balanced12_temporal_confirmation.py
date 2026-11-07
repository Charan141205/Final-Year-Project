"""
Balanced-12 causal temporal confirmation simulator (Python only).

The first positive window is PENDING. An alert is confirmed only when
`min_consecutive` adjacent windows in the same subject/recording/segment
are positive. The first confirmed alert is emitted on the Nth positive
window, not retroactively on the first one.

This is an offline simulator over precomputed feature-window CSV data.
It is not a real-time sensor interface and is not validated for real-world
fall-onset detection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "trained_models" / "balanced12_decision_tree.joblib"
MANIFEST_PATH = HERE / "trained_models" / "balanced12_decision_tree_manifest.json"
DEFAULT_MIN_CONSECUTIVE = 2


class Balanced12TemporalSimulator:
    def __init__(self, model_path=MODEL_PATH, manifest_path=MANIFEST_PATH,
                 min_consecutive: int = DEFAULT_MIN_CONSECUTIVE):
        if min_consecutive < 1:
            raise ValueError("min_consecutive must be >= 1")
        self.model = joblib.load(model_path)
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.features = list(self.manifest["feature_order"])
        self.threshold = float(self.manifest["selected_operating_threshold"])
        self.min_consecutive = int(min_consecutive)

        if len(self.features) != 12 or self.model.n_features_in_ != 12:
            raise ValueError("Expected Balanced-12 model with 12 features")
        if list(self.model.classes_) != [0, 1]:
            raise ValueError("Expected class 0=normal_activity, class 1=fall_related")

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        required_meta = ["subject_id", "recording_id", "segment_id", "window_id"]
        missing_meta = [c for c in required_meta if c not in frame.columns]
        missing_feat = [c for c in self.features if c not in frame.columns]
        if missing_meta or missing_feat:
            raise ValueError(
                f"Missing metadata columns: {missing_meta}; missing model features: {missing_feat}"
            )

        X = frame[self.features].to_numpy(dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 12 or not np.isfinite(X).all():
            raise ValueError("Feature matrix must be finite numeric data with 12 columns")

        result = frame.copy()
        probabilities = self.model.predict_proba(X)[:, 1]
        raw_pred = (probabilities >= self.threshold).astype(np.uint8)
        result["fall_probability"] = probabilities
        result["raw_predicted_binary"] = raw_pred
        result["raw_predicted_label"] = np.where(raw_pred == 1, "fall_related", "normal_activity")

        # Sort for temporal processing, but restore original input order at the end.
        result["_original_order"] = np.arange(len(result))
        result = result.sort_values(
            ["subject_id", "recording_id", "segment_id", "window_id", "_original_order"],
            kind="stable"
        ).copy()

        counts = np.zeros(len(result), dtype=np.int32)
        pending = np.zeros(len(result), dtype=np.uint8)
        confirmed = np.zeros(len(result), dtype=np.uint8)
        trigger = np.zeros(len(result), dtype=np.uint8)

        # DataFrame index labels are preserved after sort; use positional indexes.
        group_positions = result.groupby(
            ["subject_id", "recording_id", "segment_id"], sort=False
        ).indices

        raw_sorted = result["raw_predicted_binary"].to_numpy(dtype=np.uint8)
        window_ids = result["window_id"].to_numpy(dtype=np.int64)

        for positions in group_positions.values():
            positions = np.asarray(positions, dtype=int)
            run = 0
            previous_window = None
            for pos in positions:
                wid = int(window_ids[pos])
                # Do not bridge a missing window ID.
                if previous_window is None or wid != previous_window + 1:
                    run = 0
                if raw_sorted[pos] == 1:
                    run += 1
                    counts[pos] = run
                    if run < self.min_consecutive:
                        pending[pos] = 1
                    else:
                        confirmed[pos] = 1
                        if run == self.min_consecutive:
                            trigger[pos] = 1
                else:
                    run = 0
                previous_window = wid

        result["consecutive_positive_windows"] = counts
        result["pending_positive"] = pending
        result["confirmed_fall"] = confirmed
        result["alert_trigger"] = trigger
        result["temporal_rule"] = f"{self.min_consecutive}_consecutive_positive_windows"
        result = result.sort_values("_original_order", kind="stable").drop(columns=["_original_order"])
        return result


def main():
    parser = argparse.ArgumentParser(description="Balanced-12 temporal confirmation simulation")
    parser.add_argument("--input", required=True, help="CSV containing feature windows and sequence metadata")
    parser.add_argument("--output", default="balanced12_temporal_predictions.csv")
    parser.add_argument("--split", default=None, choices=["train", "validation", "test"],
                        help="Optional split filter; for development use validation")
    parser.add_argument("--min-consecutive", type=int, default=2)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.split and "split" in frame.columns:
        frame = frame[frame["split"] == args.split].copy()
    simulator = Balanced12TemporalSimulator(min_consecutive=args.min_consecutive)
    output = simulator.predict_dataframe(frame)
    output.to_csv(args.output, index=False)

    print(f"Processed windows: {len(output):,}")
    print(f"Raw positive windows: {int(output.raw_predicted_binary.sum()):,}")
    print(f"Confirmed windows: {int(output.confirmed_fall.sum()):,}")
    print(f"Alert trigger events: {int(output.alert_trigger.sum()):,}")
    print(f"Pending positive windows: {int(output.pending_positive.sum()):,}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
