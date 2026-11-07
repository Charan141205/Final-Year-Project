# Final Year Project: Smart Insole Fall Detection System

A multi-modal wearable smart shoe-insole system for real-time fall detection and automated emergency confirmation, combining plantigrade pressure sensing with inertial kinematics.

[![GitHub Repository](https://img.shields.io/badge/GitHub-Charan141205%2FFinal--Year--Project-blue?logo=github)](https://github.com/Charan141205/Final-Year-Project)
[![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20Arduino%20%7C%20Python-orange)](https://github.com/Charan141205/Final-Year-Project)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview & Abstract

Falls are the leading cause of fatal and non-fatal injuries among the elderly population worldwide. Traditional fall detection devices (pendants, smartwatches, and waist belts) suffer from poor user adherence due to social stigma, forgetting to wear the device, or violent arm movements causing frequent false alarms. 

This project designs, validates, and implements a **Smart Footwear Insole System** powered by an **ESP32 microcontroller**, integrating:
1. **4 Regional Force-Sensing Resistors (FSRs)** located at key plantar rockers (Heel, Midfoot, Forefoot, Toe) to measure real-time Ground Reaction Force (GRF), weight-bearing status, and center-of-pressure progression.
2. **6-DOF Inertial Measurement Unit (IMU)** (3-axis Accelerometer + 3-axis Gyroscope) to capture dynamic motion kinematics, impact spikes, and angular tumbling rates.
3. **Dual-Stage Classification & Temporal Confirmation Pipeline**:
   - High-speed, explainable **Balanced-12 Decision Tree** optimized for low-power microcontroller execution (depth 6, 81 nodes, sub-millisecond latency).
   - **Orientation-Invariant & Normalized Feature Extraction (OIN-10)** to guarantee seamless transfer between body locations (knee/foot) and eliminate subject body-weight bias.
   - **Causal 2-of-3 Temporal Confirmation Filter** with post-impact stillness verification ($\text{SVM}_{\text{std}} < 1.0\text{ m/s}^2$ and $\text{GM}_{\max} < 1.0\text{ rad/s}$) to suppress false alarms during high-impact Activities of Daily Living (ADLs) such as running, jumping, sweeping, mopping, and ironing.

---

## 🏗️ System Architecture

```
   +-------------------------------------------------------------------------+
   |                       PHYSICAL FOOTWEAR LAYER                           |
   |                                                                         |
   |   [FSR1: Heel]   [FSR2: Midfoot]   [FSR3: Forefoot]   [FSR4: Toe]       |
   |        │                │                 │                │            |
   |        └────────────────┼─────────────────┼────────────────┘            |
   |                         ▼                                               |
   |             Plantar Pressure Profiling                                  |
   |                         │                                               |
   |                         ▼                                               |
   |       +------------------------------------+                            |
   |       |   6-DOF IMU (MPU6500 / BWT901CL)   |                            |
   |       |   Ax, Ay, Az (m/s²) | Gx, Gy, Gz (rad/s) |                       |
   |       +-----------------+------------------+                            |
   +─────────────────────────┼───────────────────────────────────────────────+
                             │ 50 Hz Hardware Timer Sampling (20 ms interval)
                             ▼
   +-------------------------------------------------------------------------+
   |                       SIGNAL PROCESSING & CALIBRATION                   |
   |                                                                         |
   |  • Power-on Zero-Tare Subtraction: removes mechanical pre-load          |
   |  • Plantar Pressure Normalization: P_i / P_standing (body-weight immune)|
   |  • Rotational Invariance: SVM = √(Ax²+Ay²+Az²), GM = √(Gx²+Gy²+Gz²)     |
   |  • Dynamic Deceleration Jerk: J = ΔSVM / Δt                             |
   +─────────────────────────┼───────────────────────────────────────────────+
                             │
                             ▼
   +-------------------------------------------------------------------------+
   |                       EDGE DECISION & CONFIRMATION                      |
   |                                                                         |
   |  • Level 1: Balanced-12 / OIN Decision Tree (Gini, Depth 6)             |
   |  • Level 2: Causal 2-of-3 Temporal Filter (77% false alarm reduction)   |
   |  • Level 3: Post-Impact Inactivity Verification (stillness on ground)   |
   |  • Level 4: 30s Countdown Cancel & Emergency Cellular/GPS Alert        |
   +-------------------------------------------------------------------------+
```

---

## 🔬 Benchmark Dataset & Validation Strategy

The system was developed and validated using the **CAPPIMU Public Benchmark Dataset** and real volunteer trials:
* **Participants:** 30 human subjects (participant-independent train/val/test splits).
* **Total Processed Windows:** 110,949 windows (50 samples per 1.0 s nominal window).
* **Activity Taxonomies (21 classes):** 
  - Normal ADLs: walking, running, cycling, brushing teeth, eating, drinking, writing, folding clothes, hanging clothes, ironing, mopping, sweeping, washing dishes, washing face, washing windows, watching TV, bathroom (WC), computer use, smartphone use.
  - Falls: forward falls, backward falls, lateral falls, loss of balance.
* **Evaluation Integrity:** Stratified Grouped K-Fold cross-validation by subject (Leave-One-Subject-Out). **Zero data leakage** across windows.

### Empirical Validation Results

| Model / Pipeline Stage | Fall Recall | Fall Precision | False Positive Rate (FPR) | Alert Reduction |
|---|---|---|---|---|
| **Raw Balanced-12 Model** | 65.53% | 55.21% | 2.45% | Baseline |
| **With Causal 2-of-3 Temporal Filter** | 64.22% | 82.13% | **0.60%** | **-75.6% false alarms** |
| **Ironing-Weighted + Temporal Filter** | **65.32%** | **82.83%** | **0.56%** | **-77.0% false alarms** |
| **Orientation-Invariant Tree (OIN-10)** | **89.19%** | **78.40%** | **0.88%** | **Fully rotation-proof** |

---

## 📁 Repository Structure

```
.
├── 01_Training_Data_and_Analysis/                 # Extracted CAPPIMU feature datasets and validation reports
│   ├── balanced12_training_data.csv               # 110,949 pre-extracted CAPPIMU feature windows
│   ├── low_arithmetic12_training_data.csv         # Depth-4 low-cost candidate dataset
│   ├── BALANCED12_FEATURE_CALCULATION_SPEC.md     # Exact mathematical definitions of 12 features
│   ├── BALANCED12_CONTROLLED_REFINEMENT_REPORT.txt# 5-fold grouped CV and hyperparameter audit
│   └── balanced12_fixed_point_verification_report.txt # Q12 fixed-point integer verification (0 mismatches)
│
├── 02_Model_Training_Outputs/                     # Reference Scikit-Learn model artifacts
│   ├── balanced12_decision_tree.joblib            # Trained Balanced-12 reference model
│   └── balanced12_decision_tree_manifest.json     # Model metadata, leaf counts, and threshold
│
├── 03_Python_Inference_Package/                   # Production Python inference engine
│   ├── balanced12_inference.py                    # Inference wrapper on feature rows
│   ├── balanced12_feature_calculator.py           # Window-level raw-to-feature calculator
│   ├── balanced12_full_pipeline.py                # End-to-end raw segmentation & inference
│   ├── balanced12_temporal_confirmation.py        # Causal multi-window confirmation simulator
│   └── BALANCED12_PYTHON_INFERENCE_CONTRACT.txt   # Input contract and runtime validation rules
│
├── 04_Integrated_Pipeline_Output/                 # Complete diagnostic batch reports
│   ├── Baseline_vs_Ironing_Weighted/              # ADL false-positive mitigation analysis
│   ├── Gap_Aware_Temporal_Analysis/               # Hardware clock-gap handling and segmentation
│   ├── Subject_8_Raw/                             # Verified raw pipeline execution on subject 8
│   └── Orientation_Channel_Check/                 # IMU Euler angle and rotation audit
│
├── 05_ESP32_Firmware/                             # Production-ready embedded firmware
│   ├── smart_insole_fall_detector.ino             # Main Arduino/ESP32 sketch with 50 Hz timer
│   ├── decision_tree_classifier.h                 # Pure C++ Decision Tree header (OIN-10)
│   └── README.md                                  # Wiring pinout diagram and flashing guide
│
├── docs/                                          # Scientific documentation and audit reports
│   └── PROJECT_VALIDATION_AND_LITERATURE_AUDIT.md # Systematic literature review & NotebookLM validation
│
├── orientation_invariant_fall_detector.py         # OIN-10 rotational-invariant pipeline & trainer
├── orientation_invariant_features_subject8.csv    # Extracted OIN-10 features on 31 fall trials
├── orientation_invariant_tree_rules.txt           # Exported C++ compatible Decision Tree logic
├── subject8_cappimu_raw_verified.csv              # 194k-row raw benchmark recording with 31 falls
├── SMART_FOOTWEAR_PROJECT_WORKING_GUIDE.md        # Complete engineering handoff guide & audit log
└── START_HERE.txt                                 # Quick start directory guide
```

---

## ⚡ Embedded Deployment on ESP32

The decision tree was deliberately restricted to **depth 6** with **41 leaves**, requiring only elementary conditional statements (`<=`) with **zero matrix multiplications or floating-point division**:
* **Execution Time:** $< 50\ \mu\text{s}$ per inference on an ESP32 @ 240 MHz.
* **Fixed-Point Precision:** Tested at Q12 (12 fractional bits, scale 4096) with **zero classification mismatches** against 64-bit IEEE floating-point.
* **Timer Interrupt:** Runs on a dedicated hardware timer interrupt at $50\text{ Hz}$ ($20,000\ \mu\text{s}$) to avoid sampling jitter.

For full schematics and flashing instructions, refer to [`05_ESP32_Firmware/README.md`](./05_ESP32_Firmware/README.md).

---

## 👥 Authors & Acknowledgments

* **Lead Developer / Author:** [Charan141205](https://github.com/Charan141205)
* **Project Repository:** [https://github.com/Charan141205/Final-Year-Project](https://github.com/Charan141205/Final-Year-Project)
* **Benchmark Source:** CAPPIMU Dataset & Biomechanical Fall Benchmarks (SisFall, MobiFall)
