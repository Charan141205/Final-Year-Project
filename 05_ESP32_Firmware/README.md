# ESP32 Smart Insole Fall Detection Firmware (v2.0)

This folder contains the complete, fine-tuned embedded C++ firmware for deploying the Smart Insole Fall Detection System onto an **ESP32 microcontroller**.

---

## 🎛️ Operational User Presets

The firmware includes 3 pre-calibrated operational modes in [`smart_insole_fall_detector.ino`](./smart_insole_fall_detector.ino) (`#define USER_PROFILE_PRESET`):

| Preset | Mode | Stillness Window | Shock Threshold | Target Demographic / Use Case |
|---|---|---|---|---|
| **1** | `HIGH_SENSITIVITY` | $1.0\text{ s}$ | $16.0\text{ m/s}^2$ ($1.63\text{ g}$) | Sedentary / High-risk elderly; captures slow, damped falls |
| **2** | `BALANCED_DEFAULT` *(Recommended)* | $1.5\text{ s}$ | $20.0\text{ m/s}^2$ ($2.04\text{ g}$) | Standard daily living with $<1\%$ false alarm rate |
| **3** | `ACTIVE_LIFESTYLE` | $2.5\text{ s}$ | $25.0\text{ m/s}^2$ ($2.55\text{ g}$) | Highly active users, joggers; maximum false alarm suppression |

---

## ⚡ Dual-Path Triggering Mechanism

To ensure both violent falls and slow/damped falls are caught without raising false alarms:
1. **Path 1 (High-Shock Violent Falls):**
   - Triggered when temporal decision tree detects fall pattern AND acceleration peak $\text{SVM}_{\max} > \text{Threshold}$ or vector jerk $J > 200\text{ m/s}^3$.
2. **Path 2 (Soft / Damped Falls — e.g., F07 fainting against table, F10 chair slide):**
   - Triggered when moderate shock ($\text{SVM}_{\max} > 14.0\text{ m/s}^2$) is accompanied by severe plantar pressure collapse ($P_{\text{total}} < 20\%$ standing baseline and $P_{\text{drop\_range}} > 0.40$).

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

## 🛠️ Build and Flash Instructions

### Using Arduino IDE:
1. Install **ESP32 Board Support** in Arduino IDE:
   - Go to `File > Preferences > Additional Boards Manager URLs`:
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Open `Tools > Board > Boards Manager` and install **esp32 by Espressif Systems**.
2. Select your board: `Tools > Board > ESP32 Dev Module`.
3. Open `smart_insole_fall_detector.ino`.
4. Ensure `decision_tree_classifier.h` is in the same folder.
5. Connect your ESP32 via USB and click **Upload**.
6. Open Serial Monitor at **115200 baud** to view real-time calibration and detection status.
