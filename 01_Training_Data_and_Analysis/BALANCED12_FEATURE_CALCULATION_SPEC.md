# Balanced-12 Feature Calculation Specification

## Scope and provenance

This specification documents the 12 input features used by the current Balanced-12 CAPPIMU Decision Tree.
The source row-level mapping and SI conversions are documented in `process_cappimu.py`.
The window-feature generation script was not found. The core feature formulas were reconstructed and checked directly against the available raw `insole_knee_r.csv` for subject_8, recording `subject_8_block01_cut`, segment 1.

## Source signals and units

| Signal | Source column(s) | Transformation / unit |
|---|---|---|
| Heel pressure proxy | `insole_r(p6)`, `insole_r(p7)`, `insole_r(p8)` | Per-sample arithmetic mean; source pressure units retained |
| Midfoot pressure proxy | `insole_r(p4)`, `insole_r(p5)` | Per-sample arithmetic mean; source pressure units retained |
| Forefoot pressure proxy | `insole_r(p2)`, `insole_r(p3)` | Per-sample arithmetic mean; source pressure units retained |
| Toe pressure proxy | `insole_r(p1)` | Direct source pressure value; source pressure units retained |
| Acceleration X/Y/Z | `knee_r(ax/ay/az)` | Assumed source unit g; multiply by 9.80665 to obtain m/s² |
| Gyroscope X/Y/Z | `knee_r(wx/wy/wz)` | Assumed source unit degrees/s; multiply by π/180 to obtain rad/s |

These are CAPPIMU insole pressure proxies and a right-knee BWT901CL IMU, not measurements from the project's shoe FSRs and shoe-mounted MPU6500.

## Window definition

- Each saved feature row records `window_samples = 50`.
- The CAPPIMU source documentation reports a nominal sampling frequency of 50 Hz.
- In the verified first segment, windows are consecutive, non-overlapping chunks of 50 source rows.
- For `subject_8_block01_cut`, segment 1 contains 1,723 rows, produces 34 complete windows (1,700 rows), and leaves 23 tail rows.
- Actual window time span is derived from source timestamps and is not guaranteed to be exactly 1,000 ms because source timestamps are irregular.
- This validation confirms the window size and feature formulas for the inspected segment. The complete original segmentation/invalid-row handling script was not found, so do not assume all edge-case segmentation rules have been reconstructed.

## Mathematical definitions

Let a window contain N=50 samples. Let μ(.) denote the per-sample regional pressure proxy, and let a_x, a_y, a_z and g_x, g_z denote the converted SI-unit IMU sequences. `std` is population standard deviation (ddof=0); `RMS(x) = sqrt((1/N) * Σ x_i²)`.

| Model index | Feature | Definition |
|---:|---|---|
| 0 | `FSR1_heel_proxy_range` | max(heel proxy) − min(heel proxy), where heel proxy = mean(P6,P7,P8) at each sample |
| 1 | `FSR2_midfoot_proxy_min` | min(midfoot proxy), where midfoot proxy = mean(P4,P5) at each sample |
| 2 | `FSR3_forefoot_proxy_mean` | mean(forefoot proxy), where forefoot proxy = mean(P2,P3) at each sample |
| 3 | `FSR4_toe_proxy_min` | min(P1) over the 50 samples |
| 4 | `Gz_rad_s_range` | max(g_z) − min(g_z), after degrees/s to rad/s conversion |
| 5 | `Ay_mps2_rms` | sqrt(mean(a_y²)), after g to m/s² conversion |
| 6 | `Az_mps2_max` | max(a_z), after g to m/s² conversion |
| 7 | `Gx_rad_s_std` | population standard deviation of g_x, after degrees/s to rad/s conversion |
| 8 | `Ax_mps2_min` | min(a_x), after g to m/s² conversion |
| 9 | `Ay_mps2_max` | max(a_y), after g to m/s² conversion |
| 10 | `Az_mps2_mean` | mean(a_z), after g to m/s² conversion |
| 11 | `Ax_mps2_rms` | sqrt(mean(a_x²)), after g to m/s² conversion |

## Verification against saved dataset

- Raw source file: `insole_knee_r.csv` (timestamps and activity labels match subject_8).
- Recording/segment checked: `subject_8_block01_cut`, segment 1.
- Complete windows checked: 34.
- All 12 features were recalculated from each 50-row raw window and compared to the saved feature table.
- Maximum absolute numeric difference across all 34 × 12 comparisons: 8.79634864774e-09.
- Differences are at floating-point rounding precision.
- Population standard deviation (ddof=0) matches the saved values; sample standard deviation (ddof=1) does not.

## Current limitations

- Only the available subject_8 raw file was available for direct raw-to-feature verification in this workspace.
- The saved 110,949-row feature table remains the training source for all participants.
- The full original window-generation/segmentation implementation was not found; the feature formulas and 50-sample window size were verified for the inspected segment, but all segmentation edge cases were not independently reconstructed.
- These feature names and units must be preserved exactly for the current model. They are not automatically interchangeable with ESP32 feature calculations.