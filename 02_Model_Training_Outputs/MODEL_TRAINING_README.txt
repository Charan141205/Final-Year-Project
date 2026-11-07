CAPPIMU MODEL TRAINING OUTPUTS
==============================

This package contains freshly retrained Decision Tree models using the
previously extracted CAPPIMU feature datasets and the existing participant-wise
train/validation/test splits.

Models:
  1. balanced12_decision_tree.joblib
     - 12 features
     - max_depth=6
     - min_samples_leaf=50
     - 81 nodes, 41 leaves
     - fall_probability threshold = 0.8733322266783254

  2. low_arithmetic12_decision_tree.joblib
     - 12 features
     - max_depth=4
     - min_samples_leaf=200
     - 25 nodes, 13 leaves
     - fall_probability threshold = 0.7202987044355778

Both models were retrained using only the train split. Thresholds are retained
from the previously selected validation F2 operating points. The test split
was not used for model fitting or threshold selection.

Retraining check:
  The retrained tree structures exactly match their previously saved bundle
  counterparts (feature index arrays, child arrays and split thresholds).

Label mapping:
  class 0 = normal_activity
  class 1 = fall_related

Recommended current reference:
  Balanced-12, using threshold 0.8733322266783254.
  Low-arithmetic-12 remains a smaller alternative with fewer nodes.

Reports:
  - retrained_model_metrics.csv
  - retrained_classification_reports.csv
  - retrained_validation_threshold_sweep.csv
  - retraining_reproduction_checks.csv
  - training_configuration.json

Important limitations:
  - CAPPIMU pressure data are regional proxies, not actual footwear FSR readings.
  - CAPPIMU IMU is knee-mounted, not shoe-mounted.
  - The classifier predicts fall_related vs normal_activity at window/activity
    level; it does not precisely detect fall onset.
  - These results do not establish accuracy on the physical ESP32 prototype.
  - The CAPPIMU test split was previously inspected during exploratory work;
    treat test metrics as diagnostic rather than untouched final evidence.
