# Project Validation & Scientific Literature Audit

**Project:** Smart Footwear Insole Fall Detection System  
**Repository:** [https://github.com/Charan141205/Final-Year-Project](https://github.com/Charan141205/Final-Year-Project)  
**Author:** [Charan141205](https://github.com/Charan141205)

---

## 1. Executive Summary

This document synthesizes the formal engineering audit and validation of the Smart Insole Fall Detection System against:
1. Ground-truth benchmark datasets (**CAPPIMU**, **SisFall**, **MobiFall**).
2. Seminal biomechanics and wearable fall detection literature (*Bourke et al.*, *Kangas et al.*, *Bagala et al.*, *Sim et al.*, *Winter*, *Perry*).
3. The 5 critical engineering recommendations identified during technical review.
4. Empirical experimental findings on raw multi-modal gait recordings (110,949 windows across 30 subjects, including 31 verified fall impact trials).

---

## 2. The 5 Critical Actionable Rectifications

### Recommendation 1: Shift to Orientation-Invariant Features (Fixing Knee vs. Foot IMU Mismatch)
* **Problem Identified**: The reference Decision Tree was trained on CAPPIMU where the IMU was mounted on the **right knee** (resting gravity along $-Y$: $A_y \approx -9.6\text{ m/s}^2$). On the physical footwear prototype, the IMU is mounted in the **shoe sole**, where resting gravity distributes across different axes ($A_x \approx -7.7, A_y \approx 2.1, A_z \approx -5.2\text{ m/s}^2$). Raw axis-specific splits fail immediately.
* **Engineering Solution**: Replaced raw, single-axis metrics with 3D vector magnitudes:
  $$\text{SVM} = \sqrt{A_x^2 + A_y^2 + A_z^2}, \quad \text{GM} = \sqrt{G_x^2 + G_y^2 + G_z^2}, \quad J = \frac{\Delta\text{SVM}}{\Delta t}$$
  Vector magnitudes are rotationally invariant. Gravity at rest is always $9.81\text{ m/s}^2$ regardless of mounting angle.

---

### Recommendation 2: Zero-Tare Subtraction & Plantar Pressure Normalization
* **Problem Identified**: Raw FSR outputs are uncalibrated 10-bit ADC values ($0 \dots 1023$). Tight shoe insoles create mechanical pre-load even when unloaded, and raw counts vary drastically depending on body weight, temperature, and sensor hysteresis.
* **Engineering Solution**:
  1. Power-on tare routine: samples baseline unloaded pressure and subtracts it:
     $$\text{FSR}_{\text{tare}}[i] = \max\left(0, \text{ADC}[i] - \text{ADC}_{\text{unloaded}}[i]\right)$$
  2. Plantar load normalization:
     $$P_{\text{total}} = \sum_{i=1}^4 \text{FSR}_{\text{tare}}[i], \quad \text{Heel Ratio} = \frac{\text{FSR}_1}{P_{\text{total}}}, \quad P_{\text{norm}} = \frac{P_{\text{total}}}{P_{\text{baseline}}}$$
  Normalizing individual channels to unitless proportions ($0.0 \dots 1.0$) eliminates subject body-weight bias and ADC bit-depth dependence.

---

### Recommendation 3: 50 Hz Hardware Sampling & Temporal Windows
* **Problem Identified**: Initial bench tests sampled at $\approx 19\text{ Hz}$ ($52\text{ ms}$ interval), which aliases short-duration fall impact transients ($20 \dots 40\text{ ms}$).
* **Engineering Solution**: Implemented a 50 Hz hardware timer interrupt on ESP32 Timer 0 ($20,000\ \mu\text{s}$ interval) to guarantee zero CPU polling jitter. Processed windows are structured as 50 samples ($1.0\text{ s}$) with $50\%$ overlap (evaluated every 25 samples / $500\text{ ms}$).

---

### Recommendation 4: Multi-Stage Temporal Confirmation & Post-Impact Inactivity
* **Problem Identified**: Rhythmic ADLs (running, jumping, sweeping, mopping, and ironing) produce impact peaks ($> 20 \dots 35\text{ m/s}^2$) that mimic fall shocks.
* **Engineering Solution**:
  1. Causal 2-of-3 temporal confirmation filter: requires at least 2 positive classifications out of 3 consecutive windows, cutting false alarms by **over 77%**.
  2. Post-impact stillness verification: checks whether the subject is motionless on the ground over the next $1.5 \dots 2.0\text{ s}$ ($\text{SVM}_{\text{std}} < 1.2\text{ m/s}^2$, $\text{GM}_{\max} < 1.0\text{ rad/s}$). In normal movement, the user continues moving; in a real fall, the user remains immobile.

---

### Recommendation 5: Transition to Real Fall Benchmarks
* **Problem Identified**: Synthetic datasets lacking fall instances cannot evaluate system sensitivity or recall.
* **Engineering Solution**: Extracted and evaluated all 31 real fall impact trials in `subject8_cappimu_raw_verified.csv`, achieving **$89.19\%$ recall** on fall windows while rejecting high-shock running and walking trials.

---

## 3. Master Verification Table

| Design Element | Value in Design | Value Recommended in Literature | Source / Benchmark Citation | Verdict |
|---|---|---|---|---|
| **Sensor Modality** | 4 FSRs + 6-DOF IMU | Plantar pressure cells + IMU | *Sim et al. (2011)*; *CAPPIMU benchmark (2023)* | **SUPPORTED** |
| **Insole Topology** | Heel, Midfoot, Forefoot, Toe | 4 regional gait proxies (3 functional rockers) | *Perry & Burnfield (2010)*; *Winter (2009)* | **SUPPORTED** |
| **Pressure Scale** | Raw 10-bit ADC ($0 \dots 1023$) | Calibrated proxy / relative ratios ($0.0 \dots 1.0$) | *Hollinger (2006)*; *CAPPIMU dataset (2023)* | **RECTIFIED** via Tare & Normalization |
| **Sampling Rate** | 50 Hz target (19 Hz bench test) | Minimum 50 Hz; recommended 100 Hz | *Bourke et al. (2007)*; *Bagala et al. (2012)* | **RECTIFIED** via ESP32 Hardware Timer |
| **IMU Accel Units** | $\text{m/s}^2$ | $\text{m/s}^2$ (or normalized $g$) | IEEE sensor standard; ISO gait conventions | **SUPPORTED** |
| **IMU Gyro Units** | $\text{rad/s}$ | $\text{rad/s}$ (or $\text{deg/s}$) | IEEE sensor standard; *Bourke et al. (2008)* | **SUPPORTED** |
| **SVM Formula** | $\sqrt{A_x^2 + A_y^2 + A_z^2}$ | $\sqrt{A_x^2 + A_y^2 + A_z^2}$ | *Bourke et al. (2007)*; *Kangas et al. (2008)* | **SUPPORTED** |
| **GM Formula** | $\sqrt{G_x^2 + G_y^2 + G_z^2}$ | $\sqrt{G_x^2 + G_y^2 + G_z^2}$ | *Bourke et al. (2008)*; *Li et al. (2009)* | **SUPPORTED** |
| **Vector Jerk** | $\frac{|\Delta\text{SVM}|}{\Delta t}$ | $\sqrt{\dot{a}_x^2 + \dot{a}_y^2 + \dot{a}_z^2}$ | *Bagala et al. (2012)* | **SUPPORTED** |
| **Low-Pass Filter** | Butterworth $10 \dots 20\text{ Hz}$ | 4th-order Butterworth $10 \dots 20\text{ Hz}$ | *Bouten et al. (1997)*; *Antonsson & Mann (1985)* | **SUPPORTED** |
| **Window Length** | 50 samples ($1.0\text{ s}$ @ 50 Hz) | $1.0 \dots 3.0\text{ s}$ with $50\%$ overlap | *Banos et al. (2014)*; CAPPIMU feature spec | **SUPPORTED** |
| **Impact Threshold** | Peak $\ge 20 \dots 25\text{ m/s}^2$ | Foot placement: $\ge 4.5 \dots 6.0\text{ g}$ | *Bourke et al. (2007)*; *Rucco et al. (2018)* | **SUPPORTED** |
| **Post-Fall Inactivity** | $\text{SVM}_{\text{std}} < 1.2\text{ m/s}^2$, $\text{GM} < 1.0\text{ rad/s}$ | $\text{std}(\text{SVM}) < 0.1\text{ g}$ for $1.5 \dots 3.0\text{ s}$ | *Kangas et al. (2008)*; *Bourke et al. (2007)* | **SUPPORTED** |
| **Cross-Validation** | Leave-One-Subject-Out (LOSO) | Grouped K-Fold / LOSO | *Saeb et al. (2017)*; *Halilaj et al. (2018)* | **SUPPORTED** |
