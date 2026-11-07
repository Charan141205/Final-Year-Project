CAPPIMU EMBEDDED MODEL TRAINING DATA — EXTRACTED SUBSET
======================================================

Source:
  cappimu_window_features.csv (processed CAPPIMU window-feature dataset)
  Hardware Feature Selection Bundle:
    recommended_balanced12_depth6_feature_order.csv
    recommended_lowcost12_depth4_feature_order.csv

Purpose:
  These files contain only the metadata, labels, and features needed to
  reproduce/compare the two existing embedded-oriented Decision Tree candidates.
  No new ESP32 measurements were collected or added.

Files:
  balanced12_training_data.csv
    12 selected features in the exact order used by the Balanced-12 model.
    Candidate: depth 6, 81 nodes, 41 leaves.
    Validation-selected probability threshold: 0.8733322267.

  low_arithmetic12_training_data.csv
    12 selected features in the exact order used by the Low-arithmetic-12 model.
    Candidate: depth 4, 25 nodes, 13 leaves.
    Validation-selected probability threshold: 0.7202987044.

Column layout:
  First columns are identifiers, split metadata, activity label, binary target,
  and window timing/quality metadata. The remaining 12 columns are model inputs,
  in exact feature order. Keep subject_id and split for participant-wise evaluation;
  do not randomly split rows.

Split counts:
binary_label  fall_related  normal_activity
split                                      
test                   815            17602
train                 3365            74486
validation             647            14034

Total windows: 110,949
Unique subjects: 30
Unique recordings: 632

Important interpretation:
  - binary_label is the existing CAPPIMU binary target. Inspect its values before
    training; it distinguishes fall_related and normal_activity.
  - These are existing CAPPIMU features, not actual ESP32 FSR402/MPU6500 readings.
  - Pressure features are regional proxies; the IMU was knee-mounted.
  - CAPPIMU fall labels are activity/window-level, not precise fall-onset labels.
  - The same test subjects were previously inspected during exploratory model
    comparisons. Treat test results as diagnostic, not untouched final evidence.
  - No feature scaling or transformations have been applied during extraction.
  - Feature definitions and units are inherited from the existing processing
    pipeline; do not silently rename or reorder model inputs.
