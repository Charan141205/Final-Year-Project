"""
Balanced-12 Python inference pipeline for the CAPPIMU fall-related classifier.

Input:
  One row per window, containing the 12 selected features.

Output:
  fall_probability, predicted_label, predicted_binary

Important:
  - Feature order is loaded from the adjacent JSON manifest.
  - This pipeline performs model inference only. It does not calculate
    features from raw IMU/pressure samples.
  - Dataset-trained CAPPIMU model; not yet validated on physical footwear.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence, Union

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL = Path(__file__).with_name("trained_models") / "balanced12_decision_tree.joblib"
DEFAULT_MANIFEST = Path(__file__).with_name("trained_models") / "balanced12_decision_tree_manifest.json"


class Balanced12Inference:
    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL,
                 manifest_path: Union[str, Path] = DEFAULT_MANIFEST):
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.model = joblib.load(self.model_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        self.feature_order = list(self.manifest["feature_order"])
        self.threshold = float(self.manifest["selected_operating_threshold"])

        if len(self.feature_order) != 12:
            raise ValueError(f"Expected 12 features, manifest contains {len(self.feature_order)}")
        if int(self.model.n_features_in_) != len(self.feature_order):
            raise ValueError("Model feature count does not match manifest")
        if list(self.model.classes_) != [0, 1]:
            raise ValueError(f"Expected classes [0, 1], got {list(self.model.classes_)}")
        if self.model.get_depth() != 6 or self.model.tree_.node_count != 81:
            raise ValueError("Loaded model structure does not match the expected Balanced-12 model")

    def _validate_matrix(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_order):
            raise ValueError(
                f"Expected input shape (n, {len(self.feature_order)}), got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("Input contains NaN or infinite values")
        return arr

    def predict_one(self, values: Union[Mapping[str, float], Sequence[float]]) -> dict:
        """Predict one window. Mapping inputs are assembled in manifest order."""
        if isinstance(values, Mapping):
            missing = [name for name in self.feature_order if name not in values]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            row = [values[name] for name in self.feature_order]
        else:
            row = values

        X = self._validate_matrix(row)
        probability = float(self.model.predict_proba(X)[0, 1])
        is_fall = probability >= self.threshold
        return {
            "fall_probability": probability,
            "threshold": self.threshold,
            "predicted_binary": int(is_fall),
            "predicted_label": "fall_related" if is_fall else "normal_activity",
        }

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict all rows in a DataFrame, selecting columns in exact model order."""
        missing = [name for name in self.feature_order if name not in frame.columns]
        if missing:
            raise ValueError(f"Input CSV/DataFrame is missing features: {missing}")

        X = self._validate_matrix(frame[self.feature_order].to_numpy())
        probabilities = self.model.predict_proba(X)[:, 1]
        predicted = (probabilities >= self.threshold).astype(np.uint8)

        result = frame.copy()
        result["fall_probability"] = probabilities
        result["predicted_binary"] = predicted
        result["predicted_label"] = np.where(predicted == 1, "fall_related", "normal_activity")
        return result

    def predict_csv(self, input_csv: Union[str, Path],
                    output_csv: Union[str, Path]) -> pd.DataFrame:
        source = Path(input_csv)
        destination = Path(output_csv)
        frame = pd.read_csv(source)
        result = self.predict_dataframe(frame)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        return result


def main():
    parser = argparse.ArgumentParser(description="Balanced-12 CAPPIMU inference")
    parser.add_argument("--input", required=True, help="CSV containing the 12 selected features")
    parser.add_argument("--output", default="balanced12_predictions.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    predictor = Balanced12Inference()
    result = predictor.predict_csv(args.input, args.output)
    print(f"Processed {len(result):,} windows")
    print(f"Fall-related predictions: {(result['predicted_binary'] == 1).sum():,}")
    print(f"Normal predictions: {(result['predicted_binary'] == 0).sum():,}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
