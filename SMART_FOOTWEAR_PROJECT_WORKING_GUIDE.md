# Smart Footwear Project — Working Guide & Handoff

**Purpose:** Keep one reliable record of what has already been done, what files exist, what decisions are currently active, and what to do next. Use this guide to avoid repeating work.

## 1. Project direction and current working mode

- Project: ESP32-based smart footwear for emergency/fall-related detection.
- Current priority: **Python dataset/model work first**. Do not move to C++/Arduino deployment or VLSI unless explicitly decided later.
- Dataset policy currently chosen by the user: **use the existing CAPPIMU datasets only**. Do not start collecting a new dataset from the ESP32 unless the user changes this decision.
- Current model reference: **Balanced-12 Decision Tree**.
- Alternative model to retain for comparison: **Low-arithmetic-12 Decision Tree**.

## 2. Dataset and preprocessing already completed

### Source dataset
- CAPPIMU public dataset.
- 30 participants.
- Existing window-feature dataset: `cappimu_window_features.csv`.
- 110,949 windows total.
- Participant-wise splits are already assigned and must be preserved:
  - Train: 77,851 windows
  - Validation: 14,681 windows
  - Test: 18,417 windows
- Binary target labels:
  - `normal_activity`
  - `fall_related`

### CAPPIMU domain limitations
These limitations must be stated whenever discussing real-world footwear performance:
- Pressure inputs are four regional proxies derived from the right insole pressure points, not direct readings from the project's four FSR sensors.
- IMU data is from a right-knee-mounted BWT901CL, not the shoe-mounted MPU6500.
- The binary target is activity/window-level `fall_related` versus `normal_activity`; it is not an exact fall-onset detector.
- Results on CAPPIMU do not prove accuracy on the physical footwear prototype.

### Available preprocessing files
- `process_cappimu.py`
- `CAPPIMU_processing_README.txt`
- `cappimu_processing_bundle.zip`
- `cappimu_window_features.csv`

The preprocessing code documents the four regional pressure proxy mapping:
- Heel: mean of P6, P7, P8
- Midfoot: mean of P4, P5
- Forefoot: mean of P2, P3
- Toe: P1

It also documents conversion of source acceleration assumed in g to m/s² and gyroscope assumed in degrees/s to rad/s. It does not, by itself, establish the full formula for every window-level feature used by the selected 12-feature classifier.

## 3. Hardware Feature Selection Bundle work already completed

The bundle has been inspected. Important candidates:

| Candidate | Features | Depth | Nodes | Leaves |
|---|---:|---:|---:|---:|
| Balanced-12 | 12 | 6 | 81 | 41 |
| Low-arithmetic-12 | 12 | 4 | 25 | 13 |
| Top-10 multimodal | 10 | 8 | 183 | 92 |

Do not repeat the original feature-selection experiment unless a specific reason is identified. Preserve the earlier candidates and use them as baselines.

### Balanced-12 feature order (exact)
1. `FSR1_heel_proxy_range`
2. `FSR2_midfoot_proxy_min`
3. `FSR3_forefoot_proxy_mean`
4. `FSR4_toe_proxy_min`
5. `Gz_rad_s_range`
6. `Ay_mps2_rms`
7. `Az_mps2_max`
8. `Gx_rad_s_std`
9. `Ax_mps2_min`
10. `Ay_mps2_max`
11. `Az_mps2_mean`
12. `Ax_mps2_rms`

### Low-arithmetic-12 feature order (exact)
1. `FSR1_heel_proxy_range`
2. `FSR2_midfoot_proxy_min`
3. `FSR3_forefoot_proxy_mean`
4. `FSR4_toe_proxy_min`
5. `Gz_rad_s_range`
6. `Az_mps2_max`
7. `Ax_mps2_mean`
8. `Ay_mps2_max`
9. `Az_mps2_mean`
10. `Ax_mps2_min`
11. `Gz_rad_s_mean`
12. `Ax_mps2_range`

## 4. Models already trained and saved

Both candidates were freshly retrained using only the existing training split, with original hyperparameters and random state. The retrained tree structures exactly matched the earlier saved models (feature-index arrays, child links, and split thresholds).

### Balanced-12 (current reference)
- 12 features
- max depth: 6
- nodes: 81
- leaves: 41
- min_samples_leaf: 50
- class_weight: balanced
- random_state: 42
- class mapping: 0 = normal_activity, 1 = fall_related
- selected probability threshold: **0.8733322266783254**

Validation at selected threshold:
- Precision (fall): 55.21%
- Recall (fall): 65.53%
- F1 (fall): 59.93%
- F2 (fall): 63.17%
- False-positive rate: 2.45%

Diagnostic test at same threshold:
- Precision (fall): 47.15%
- Recall (fall): 57.79%
- F1 (fall): 51.93%
- False-positive rate: 3.00%

### Low-arithmetic-12 (smaller alternative)
- 12 features
- max depth: 4
- nodes: 25
- leaves: 13
- min_samples_leaf: 200
- class_weight: balanced
- random_state: 42
- class mapping: 0 = normal_activity, 1 = fall_related
- selected probability threshold: **0.7202987044355778**

Validation:
- Precision (fall): 53.05%
- Recall (fall): 59.20%
- F1 (fall): 55.95%
- F2 (fall): 57.86%
- False-positive rate: 2.42%

Diagnostic test:
- Precision (fall): 55.91%
- Recall (fall): 44.66%
- F1 (fall): 49.66%
- False-positive rate: 1.63%

Test set was already viewed in previous exploratory comparisons. Treat test metrics as **diagnostic**, not as an untouched final estimate. Do not use the test split to tune thresholds.

## 5. Fixed-point work already performed (do not repeat without a reason)

The Balanced-12 tree was compared against a fixed-point representation by rounding both feature values and split thresholds.

| Fractional bits | Scale | Total mismatches | Test mismatches |
|---:|---:|---:|---:|
| 4 | 16 | 375 | 49 |
| 6 | 64 | 56 | 6 |
| 8 | 256 | 19 | 0 |
| 10 | 1,024 | 3 | 0 |
| 12 | 4,096 | 2 | 0 |
| 14 | 16,384 | 1 | 0 |
| 16 | 65,536 | 0 | 0 |

- Mismatch means fixed-point tree output differed from the floating-point tree output, not necessarily from the ground-truth label.
- Q12 was identified as a practical signed 32-bit candidate on the observed data range, but **this is not yet a validated embedded implementation**.
- User explicitly asked to skip further investigation of an extreme feature outlier and continue with model work. Do not return to outlier investigation unless the user asks.
- No C++/Arduino or Verilog implementation is currently requested. Continue with Python.

## 6. Python inference pipeline already created

Files:
- `balanced12_inference.py`
- `trained_models/balanced12_decision_tree.joblib`
- `trained_models/balanced12_decision_tree_manifest.json`
- `BALANCED12_PYTHON_INFERENCE_CONTRACT.txt`
- `balanced12_python_inference_verification.csv`
- `balanced12_python_inference_examples.csv`
- `balanced12_validation_sample_input.csv`
- `balanced12_validation_sample_expected_output.csv`

Inference logic:
1. Load model and manifest.
2. Require the exact 12 features in the manifest.
3. Reject missing, NaN, infinite, or incorrectly shaped inputs.
4. Calculate `predict_proba`.
5. Extract class-1 probability (`fall_related`).
6. Apply threshold 0.8733322266783254.
7. Return `fall_related` or `normal_activity`.

Verification:
- Wrapper probabilities exactly matched the direct Scikit-learn model probabilities on train, validation, and test rows.
- Wrapper predictions exactly matched the direct Scikit-learn thresholded predictions on all 110,949 rows.
- This verifies the inference wrapper, not physical sensor compatibility or ground-truth correctness.

### Interpreting the user's shared output sample
The user shared 10 rows:
- All 10 were predicted `normal_activity` (`predicted_binary = 0`).
- First five windows received probability 0.002423516.
- Next five received probability 0.365995174.
- Identical probabilities across different rows are expected for a Decision Tree because samples reaching the same leaf get the same leaf probability.
- Both probabilities are below 0.873332, so both groups are classified as normal.
- This alone does not establish correctness; actual labels are needed to assess those ten predictions.

## 7. The exact next work — Python only

### Next Stage A: Establish the 12-feature calculation contract
Before changing the model or deploying anywhere:
1. Locate the exact code or notebook that generated `cappimu_window_features.csv`.
2. Document the mathematical definition of each of the 12 selected features.
3. Record source signal, units, window length, sample rate/timestamp handling, and any filtering/resampling.
4. Verify whether source windows use 50 samples and how irregular timestamps/gaps are handled.
5. Do not guess formulas from feature names alone.

**Important current gap:** `process_cappimu.py` creates the master row-level pressure/IMU dataset and documents source conversions and regional proxy mapping. It does not fully document the window-feature generator for all 12 model features. The script/notebook used to create `cappimu_window_features.csv` is the most useful next file to locate/share.

### Next Stage B: Reproduce feature rows
- Implement the documented feature calculations in Python.
- Compare calculated feature values against existing `cappimu_window_features.csv` for selected recordings/windows.
- Investigate mismatches in formula/units/window handling only as needed.
- Keep the saved feature order unchanged.

### Next Stage C: Prediction evaluation and error analysis
- Join predictions to their actual labels.
- Examine false negatives and false positives on validation.
- Keep threshold selection based on validation only.
- Do not optimize thresholds using test rows.
- Keep the Balanced-12 model as reference; compare Low-arithmetic-12 only when relevant.

### Later, only after Python is stable
If the user chooses to proceed:
- ESP32 integration would require proving that the actual ESP32 feature calculations produce inputs compatible with the trained features. CAPPIMU proxies and knee-mounted IMU values are not automatically interchangeable with actual shoe-mounted FSR/MPU6500 features.
- C++/Arduino deployment is a later stage; do not introduce it during current Python work.
- VLSI/RTL is deferred and out of scope for the current stage.

## 8. What the user should share in future turns

### For ordinary continuation in the same conversation
- Do not ask the user to upload the CAPPIMU dataset, extraction ZIP, models, or inference package again if they are already accessible in the current session.
- Ask for only the specific missing artifact needed for the next task.

### If continuing in a new conversation
Start by sharing:
1. This `SMART_FOOTWEAR_PROJECT_WORKING_GUIDE.md`.
2. `balanced12_python_inference_package.zip` if inference code/model is needed.
3. `cappimu_embedded_training_data.zip` if extracted feature datasets/reports are needed.
4. `cappimu_model_training_outputs.zip` if both trained candidate models and training reports are needed.

For the *next specific task*, the most useful additional file is the script/notebook that generated `cappimu_window_features.csv`, if available. Do not re-upload the 129 MB feature dataset unless requested or unavailable in the new session.

### Avoid sending repeatedly
Do not repeatedly upload:
- the hardware feature-selection bundle,
- all prediction CSVs,
- the full raw CAPPIMU dataset,
- ESP32 sensor logs,
unless a task specifically requires them.

## 9. Reusable project continuation prompt

Copy and paste this at the start of a new conversation:

> Continue my ESP32-based smart footwear emergency-detection project using the attached Smart Footwear Project Working Guide. We are currently working in Python only and using the existing CAPPIMU datasets; do not collect new ESP32 data, do not move to C++/Arduino or VLSI yet, and do not repeat completed experiments. Balanced-12 is the current reference model with 12 inputs and threshold 0.8733322266783254. The Python inference wrapper already matches the saved Scikit-learn model on all 110,949 rows. First check the guide and attached files, identify the current pending task, and ask only for the one missing artifact if required. Preserve all documented feature orders, split assignments, thresholds, and limitations.

---
**Working rule:** Before proposing an experiment, check this guide and existing artifacts. Reuse completed work; do not repeat work unless there is a clear technical reason.


## 10. Balanced-12 feature formulas reconstructed and verified

The original window-feature generation script was not found. A raw `insole_knee_r.csv` file already available in the workspace was matched by timestamps/activity to `subject_8` in `cappimu_window_features.csv`.

The 12 Balanced-12 feature formulas were reconstructed from:
- `process_cappimu.py` pressure-region mappings and SI unit conversions.
- The existing feature names and saved 50-sample window metadata.
- Direct calculation from the available raw subject_8 recording.

Verification performed:
- Recording: `subject_8_block01_cut`, segment 1.
- 34 consecutive complete windows, each containing 50 raw rows.
- For all 12 selected features, computed values matched the saved CAPPIMU feature rows.
- Maximum absolute difference across 408 comparisons: about 8.80e-9.
- `Gx_rad_s_std` matches population standard deviation (`ddof=0`).
- RMS is `sqrt(mean(x**2))`.
- The first segment has 1,723 rows: 34 complete 50-row windows and 23 tail rows.

New files:
- `BALANCED12_FEATURE_CALCULATION_SPEC.md`
- `balanced12_feature_calculator.py`
- `balanced12_feature_formula_validation_subject8.csv`
- `balanced12_feature_formula_verification_summary.csv`

Important boundary:
- The raw-to-feature calculation formulas are validated for the inspected subject_8 segment.
- The original full-window segmentation/invalid-row handling program is still unavailable, so do not claim that every segmentation edge case across all participants was independently reconstructed.
- The existing 110,949-row feature table remains the source for model training and evaluation. No need to regenerate it.

Next Python task:
- Use the existing saved model and feature dataset to perform focused validation error analysis (false negatives and false positives), without retraining or changing the selected threshold unless a validation-only experiment is specifically planned.


## 11. Balanced-12 validation error analysis completed

The selected Balanced-12 model was evaluated on validation participants only at threshold 0.8733322266783254.

Validation confusion matrix:
- TN 13,690
- FP 344
- FN 223
- TP 424

By activity:
- Highest false-positive counts: ironing (99), sweep (82), wc (48), mop (40), wash_window (29).
- All validation false negatives are from activity_label `fall`.
- Fall recall was 65.53% (424/647).

By validation subject:
- subject_25: 72 false negatives / 173 fall windows; 101 true positives.
- subject_29: 23 false negatives / 137 fall windows; 114 true positives.
- subject_5: 57 false negatives / 162 fall windows; 105 true positives.
- subject_8: 71 false negatives / 175 fall windows; 104 true positives.

The validation error analysis is diagnostic. Do not automatically change the model or threshold from these counts alone. The CAPPIMU target remains window/activity-level and the test split must not be used for tuning.

New files:
- `validation_error_analysis/BALANCED12_VALIDATION_ERROR_ANALYSIS.txt`
- `validation_error_analysis/validation_errors_by_activity.csv`
- `validation_error_analysis/validation_errors_by_subject.csv`
- `validation_error_analysis/validation_misclassified_windows.csv`

Next Python task:
- Trace validation false-positive and false-negative windows through the Balanced-12 decision tree and inspect their selected feature values. Keep the current model and threshold unchanged as the baseline.


## 12. Balanced-12 validation decision-path trace completed

Validation misclassifications were traced through the fitted 81-node tree at threshold 0.8733322266783254.

- 344 false positives occur in 5 leaves.
- Two leaves (68 and 51) account for 313/344 false positives (90.99%).
- Leaf 68 score = 0.897214; 218 false positives, strongly represented by normal `sweep` windows and other activities.
- Leaf 51 score = 0.884954; 95 false positives, predominantly normal `ironing` windows.
- Both leaves also contain true fall-labelled validation windows, so simply raising the threshold would remove some true positives too.
- 223 false negatives are spread across 17 leaves; the largest are leaf 70 (80), leaf 71 (46), leaf 67 (35), leaf 64 (15), and leaf 60 (14).

Probability interpretation:
- The model uses `class_weight='balanced'`; Scikit-learn leaf probabilities are class-weight-adjusted training scores. Do not interpret `fall_probability` as a calibrated real-world fall probability.

New files:
- `validation_error_analysis/BALANCED12_DECISION_PATH_ANALYSIS.txt`
- `validation_error_analysis/validation_error_leaf_paths.csv`
- `validation_error_analysis/validation_misclassification_paths.csv`
- `validation_error_analysis/error_path_feature_frequency.csv`

Next step:
- Conduct a controlled Python-only refinement experiment without modifying the frozen Balanced-12 baseline.
- Train candidates on training participants only; predefine candidate model settings; compare on validation participants using F2, recall, FPR and node count.
- Do not use test participants for tuning. Because validation has been used in prior selection, avoid extensive repeated tuning and state that validation is a development split.


## 13. Controlled grouped-CV refinement completed

A predefined 15-configuration search was run on the 21 training subjects only:
- max_depth [4,5,6,7,8]
- min_samples_leaf [50,100,200]
- 5-fold StratifiedGroupKFold, grouped by subject_id
- Threshold selected from out-of-fold training probabilities by maximizing F2 subject to FPR <= 5%
- CAPPIMU test set was not used for model or threshold selection

Top grouped-CV candidate:
- max_depth 8, min_samples_leaf 100
- OOF threshold 0.818754
- OOF F2 0.6672; recall 0.7421; FPR 0.0370
- final full-training tree has 163 nodes and 82 leaves

One-time validation comparison:
- Current Balanced-12 at legacy threshold 0.873332: F2 0.6317; recall 0.6553; precision 0.5521; FPR 0.0245; 81 nodes.
- Refined depth-8/min_leaf-100 at OOF threshold 0.818754: F2 0.6315; recall 0.6739; precision 0.5046; FPR 0.0305; 163 nodes.

Decision: do NOT promote the refined candidate. Validation F2 is essentially unchanged, while the model is about twice as large and has higher FPR. Keep the original Balanced-12 as the official reference. Retain `balanced12_refined_depth8_leaf100.joblib` only as an experiment.

New report:
- `BALANCED12_CONTROLLED_REFINEMENT_REPORT.txt`
- `grouped_cv_candidate_comparison.csv`
- `grouped_cv_fold_subjects.csv`
- `refined_candidate_validation_comparison.csv`
- `refined_candidate_validation_predictions.csv`
- `refined_candidate_feature_importance.csv`

Next step:
- Perform offline temporal consistency analysis of validation predictions by recording (consecutive positive-window run lengths, isolated false positives, fall-window run detection). Treat windows as approximate 1-second windows, noting timestamp irregularity and that CAPPIMU labels are activity-level rather than event-onset labels.
- Do not begin another broad hyperparameter sweep or use the test set for tuning.


## 14. Offline temporal confirmation analysis completed

Using the frozen Balanced-12 model and threshold 0.8733322266783254, validation predictions were grouped into consecutive runs within each subject/recording/segment.

Approximate start-to-start spacing between adjacent windows:
- Median 995.3 ms
- 5th-95th percentile 911.1-1037.8 ms
- 99.9846% of adjacent window gaps were between 500 and 1500 ms

Validation comparison:
- Original model output: precision 55.21%, recall 65.53%, F2 63.17%, FPR 2.45%, FP 344, FN 223.
- Require >=2 consecutive positive windows: precision 86.06%, recall 60.12%, F2 63.98%, FPR 0.45%, FP 63, FN 258.
- Require >=3 consecutive positive windows: precision 95.90%, recall 54.25%, F2 59.41%, FPR 0.11%, FP 15, FN 296.

The >=2-window rule removes many isolated normal false alarms, but it also removes 35 true-positive fall windows on this validation split. It is an offline candidate policy only. It must not be treated as a proven emergency alarm rule because CAPPIMU labels are activity/window-level, not onset-level, and timestamps are irregular.

New files:
- `validation_error_analysis/BALANCED12_TEMPORAL_CONFIRMATION_ANALYSIS.txt`
- `temporal_filter_validation_comparison.csv`
- `validation_error_analysis/temporal_filter_validation_predictions.csv`
- `validation_error_analysis/positive_prediction_runs.csv`
- `validation_error_analysis/temporal_filter_subject_metrics.csv`
- `validation_error_analysis/temporal_false_positive_runs_by_activity.csv`
- `validation_error_analysis/temporal_filter_activity_comparison.csv`

Current decision:
- Keep original Balanced-12 model and threshold unchanged.
- Carry >=2 consecutive positive windows as a Python simulation candidate only.
- No threshold/model change and no ESP32 deployment decision is made.


## 15. Causal temporal confirmation simulator created

A Python-only simulator was created for a causal rule: the first positive window is pending; a confirmation/alert trigger is emitted only when the second adjacent positive window arrives within the same subject/recording/segment. The counter resets after a negative prediction or a window-ID gap. No runs cross segment boundaries.

Causal validation results at the frozen Balanced-12 threshold:
- TN 14,000; FP 34; FN 354; TP 293.
- Precision 89.60%; recall 45.29%; F1 60.16%; F2 50.26%; FPR 0.242%.
- This is a per-window metric and includes the one-window confirmation delay.

Segment-level activity-label analysis:
- 49/61 fall-labelled activity segments received at least one alert (80.33%).
- 12/61 fall-labelled activity segments received no alert.
- 26/1,657 normal activity segments had at least one false alarm (1.57%).
- 96 trigger pulses occurred during fall-labelled activity segments; 29 during normal activity segments.

Critical distinction:
- Segment detection means an alert occurred somewhere inside a segment labelled `fall`; it is not precise fall-onset detection.
- The retrospective run filter and causal real-time simulator are different. The causal simulator does not retroactively confirm the first positive window.
- The temporal simulator has no cooldown, cancellation, GPS/GSM or caregiver notification behavior.
- The classifier, feature order, model parameters and threshold remain unchanged.
- No physical ESP32 deployment or field accuracy is established.

New files:
- `balanced12_temporal_confirmation.py`
- `validation_temporal_confirmation_predictions_causal.csv`
- `validation_temporal_segment_event_summary.csv`
- `validation_temporal_causal_metrics.json`
- `validation_error_analysis/BALANCED12_CAUSAL_TEMPORAL_SIMULATION_REPORT.txt`

Current decision:
- Continue using the original Balanced-12 as reference.
- Keep causal 2-window confirmation as a Python-only candidate, not final alarm policy.
- Next Python design task: specify a complete event state machine (pending, confirmed, alert-active, reset/rearm) and evaluate it offline. Avoid adding SMS/GPS actions until the logic is defined and validated.
