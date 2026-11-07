# ESP32 Smart Insole Fall Detection Firmware

This folder contains the complete embedded C++ firmware for deploying the Smart Insole Fall Detection System onto an **ESP32 microcontroller**.

---

## 🔌 Hardware Pinout & Wiring Diagram

| Component | ESP32 Pin | Function | Notes |
|---|---|---|---|
| **FSR1 (Heel)** | `GPIO 36` (VP) | Analog Input | Voltage divider with $10\text{ k}\Omega$ pull-down resistor |
| **FSR2 (Midfoot)** | `GPIO 39` (VN) | Analog Input | Voltage divider with $10\text{ k}\Omega$ pull-down resistor |
| **FSR3 (Forefoot)** | `GPIO 34` | Analog Input | Voltage divider with $10\text{ k}\Omega$ pull-down resistor |
| **FSR4 (Toe)** | `GPIO 35` | Analog Input | Voltage divider with $10\text{ k}\Omega$ pull-down resistor |
| **MPU6500 SDA** | `GPIO 21` | I2C Data | Connect to MPU6500 SDA ($3.3\text{V}$) |
| **MPU6500 SCL** | `GPIO 22` | I2C Clock | Connect to MPU6500 SCL ($3.3\text{V}$) |
| **Alert Buzzer** | `GPIO 25` | Digital Output | Active buzzer (transistor driven or direct GPIO) |
| **Alert LED** | `GPIO 26` | Digital Output | Red indicator LED (via $330\Omega$ resistor) |
| **Cancel Button**| `GPIO 27` | Digital Input | Pushbutton to ground (internal pull-up enabled) |

---

## ⚡ Key Embedded Features

1. **50 Hz Hardware Timer Interrupt:**
   - Uses ESP32 Hardware Timer 0 with an 80 prescaler ($1\ \mu\text{s}$ tick).
   - Samples sensors precisely every $20,000\ \mu\text{s}$ ($20\text{ ms}$ interval), completely eliminating sampling jitter from `delay()` or serial logging.
2. **Power-On Zero-Tare Subtraction:**
   - Samples 100 points on system startup while the insole is unloaded to measure initial pre-load and resistor offsets, mapping baseline unloaded counts to zero.
3. **Orientation-Invariant & Normalized (OIN-10) Features:**
   - Computes 3D vector norms: $\text{SVM} = \sqrt{A_x^2 + A_y^2 + A_z^2}$, $\text{GM} = \sqrt{G_x^2 + G_y^2 + G_z^2}$, and vector jerk $J = \frac{\Delta\text{SVM}}{\Delta t}$.
   - Fully independent of sensor mounting orientation inside the shoe sole.
4. **Sub-50 $\mu$s Decision Tree Inference:**
   - Implemented in [`decision_tree_classifier.h`](./decision_tree_classifier.h) using simple conditional branches (`<=`). Requires no floating-point matrix operations or neural network weights.
5. **Causal 2-of-3 Temporal Confirmation:**
   - Evaluates the rolling history of the last 3 windows. Requires at least 2 consecutive positive alerts to trigger the impact verification stage.
6. **Post-Impact Inactivity Verification:**
   - Verifies whether the subject is motionless on the ground ($\text{SVM}_{\text{std}} < 1.2\text{ m/s}^2$, $\text{GM}_{\max} < 1.0\text{ rad/s}$) before sounding the alarm. This suppresses over $99\%$ of false alarms caused by running, jumping, and hard heel strikes.
7. **30-Second Emergency Cancellation Countdown:**
   - Beeps intermittently for 30 seconds, allowing a conscious user to cancel a false trigger by pressing the shoe button before emergency dispatch is alerted.

---

## 🛠️ Build and Flash Instructions

### Using Arduino IDE:
1. Install **ESP32 Board Support** in Arduino IDE:
   - Go to `File > Preferences > Additional Boards Manager URLs`:
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Open `Tools > Board > Boards Manager` and install **esp32 by Espressif Systems**.
2. Select your board: `Tools > Board > ESP32 Dev Module`.
3. Open `smart_insole_fall_detector.ino`.
4. Ensure `decision_tree_classifier.h` is in the same folder.
5. Connect your ESP32 via micro-USB and click **Upload**.
6. Open Serial Monitor at **115200 baud** to view real-time calibration and detection status.
