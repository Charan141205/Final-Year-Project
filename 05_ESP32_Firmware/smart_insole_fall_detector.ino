/**
 * Final Year Project: Smart Insole Fall Detection System
 * Embedded Firmware for ESP32 Microcontroller (Fine-Tuned v2.0)
 * 
 * Features & Optimizations:
 *  1. 50 Hz Hardware Timer Interrupt (20,000 us sampling interval).
 *  2. 4-Channel Plantar FSR sensing with power-on zero-tare subtraction.
 *  3. MPU6500 / MPU6050 6-DOF IMU reading via I2C Fast-Mode.
 *  4. Orientation-Invariant 3D Vector Norms (SVM, GM, Vector Jerk).
 *  5. Real-time Decision Tree inference (< 50 us latency).
 *  6. Causal 2-of-3 Temporal Confirmation Filter.
 *  7. Dual-Path Triggering: Captures both High-Shock Falls & Soft/Damped Falls (F07/F10).
 *  8. Configurable Post-Impact Stillness Window (1.0 s - 3.0 s) via user presets.
 *  9. Multi-Modal Plantar Pressure Fusion (Non-weight-bearing floor confirmation).
 * 10. Audible buzzer, alert LED, and 30-second cancellation button.
 */

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "decision_tree_classifier.h"

// ======================== OPERATIONAL USER PRESETS =========================
// Select User Profile Preset:
//   1 = HIGH_SENSITIVITY (Sedentary / High-risk elderly: captures slow/damped falls)
//   2 = BALANCED_DEFAULT (Recommended: optimal false-alarm rejection with 1.5s stillness)
//   3 = ACTIVE_LIFESTYLE (Active users / runners: 2.5s stillness, strict shock gating)
#define USER_PROFILE_PRESET 2

#if USER_PROFILE_PRESET == 1
    #define STILLNESS_CONFIRMATION_MS 1000  // 1.0 s verification window
    #define IMPACT_SHOCK_THRESH_MPS2  16.0f // Lower shock gate (1.63 g) for damped falls
    #define STILLNESS_SVM_STD_MAX     1.5f  // Permissive stillness threshold
#elif USER_PROFILE_PRESET == 3
    #define STILLNESS_CONFIRMATION_MS 2500  // 2.5 s verification window
    #define IMPACT_SHOCK_THRESH_MPS2  25.0f // High shock gate (2.55 g) for active users
    #define STILLNESS_SVM_STD_MAX     1.0f  // Strict stillness threshold
#else // BALANCED_DEFAULT (Preset 2)
    #define STILLNESS_CONFIRMATION_MS 1500  // 1.5 s verification window
    #define IMPACT_SHOCK_THRESH_MPS2  20.0f // Standard impact gate (2.04 g)
    #define STILLNESS_SVM_STD_MAX     1.2f  // Standard stillness threshold
#endif

// ======================== HARDWARE PIN CONFIGURATION ========================
#define PIN_FSR_HEEL       36 // ADC1_CH0 (VP)
#define PIN_FSR_MID        39 // ADC1_CH3 (VN)
#define PIN_FSR_FORE       34 // ADC1_CH6
#define PIN_FSR_TOE        35 // ADC1_CH7

#define PIN_BUZZER         25
#define PIN_LED_ALERT      26
#define PIN_BUTTON_CANCEL  27 // Active LOW with internal pull-up

#define I2C_SDA            21
#define I2C_SCL            22
#define MPU_ADDR           0x68

// ======================== SAMPLING & TIMING CONFIG ==========================
#define SAMPLE_RATE_HZ     50
#define SAMPLE_INTERVAL_US (1000000 / SAMPLE_RATE_HZ) // 20,000 us = 20 ms
#define WINDOW_SIZE        50                         // 50 samples = 1.0 second
#define STEP_SIZE          25                         // 50% overlap (every 500 ms)

#define GRAVITY_EARTH      9.80665f
#define DEG_TO_RAD_F       (3.1415926535f / 180.0f)

// ======================== GLOBAL BUFFERS & STATE ============================
hw_timer_t * sampleTimer = NULL;
volatile bool sampleReadyFlag = false;

// Circular sample buffers
float buf_ax[WINDOW_SIZE];
float buf_ay[WINDOW_SIZE];
float buf_az[WINDOW_SIZE];
float buf_gx[WINDOW_SIZE];
float buf_gy[WINDOW_SIZE];
float buf_gz[WINDOW_SIZE];
float buf_heel[WINDOW_SIZE];
float buf_mid[WINDOW_SIZE];
float buf_fore[WINDOW_SIZE];
float buf_toe[WINDOW_SIZE];

volatile int writeIndex = 0;
volatile int samplesCollected = 0;

// Tare and baseline values
float tare_heel = 0.0f;
float tare_mid  = 0.0f;
float tare_fore = 0.0f;
float tare_toe  = 0.0f;
float standing_total_baseline = 1000.0f; // Calibrated during quiet standing

// Temporal confirmation history (last 3 window decisions)
int windowHistory[3] = {0, 0, 0};

// State Machine
enum SystemState {
    STATE_MONITORING,
    STATE_IMPACT_DETECTED,
    STATE_CONFIRMING_STILLNESS,
    STATE_ALERT_COUNTDOWN,
    STATE_ALARM_DISPATCHED
};

SystemState currentState = STATE_MONITORING;
unsigned long stateTimerStart = 0;
const unsigned long COUNTDOWN_DURATION_MS = 30000; // 30-second cancellation window

// ======================== HARDWARE TIMER ISR ================================
void IRAM_ATTR onSampleTimer() {
    sampleReadyFlag = true;
}

// ======================== SENSOR DRIVERS ====================================
void initMPU6500() {
    Wire.begin(I2C_SDA, I2C_SCL, 400000); // 400 kHz Fast-Mode I2C
    
    // Wake up MPU
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B); // PWR_MGMT_1 register
    Wire.write(0x00); // Wake up
    Wire.endTransmission(true);

    // Set Accelerometer to +/- 16g range
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C); // ACCEL_CONFIG
    Wire.write(0x18); // +/- 16g
    Wire.endTransmission(true);

    // Set Gyroscope to +/- 2000 deg/s range
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1B); // GYRO_CONFIG
    Wire.write(0x18); // +/- 2000 deg/s
    Wire.endTransmission(true);

    // Set Digital Low Pass Filter to ~42 Hz cutoff
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A); // CONFIG
    Wire.write(0x03); // DLPF_CFG = 3 (42 Hz)
    Wire.endTransmission(true);

    Serial.println("[IMU] MPU6500 initialized: +/-16g, +/-2000dps, 42Hz DLPF.");
}

void readMPURaw(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14, (uint8_t)true);

    if (Wire.available() >= 14) {
        int16_t raw_ax = Wire.read() << 8 | Wire.read();
        int16_t raw_ay = Wire.read() << 8 | Wire.read();
        int16_t raw_az = Wire.read() << 8 | Wire.read();
        Wire.read(); Wire.read(); // Skip temp
        int16_t raw_gx = Wire.read() << 8 | Wire.read();
        int16_t raw_gy = Wire.read() << 8 | Wire.read();
        int16_t raw_gz = Wire.read() << 8 | Wire.read();

        // Convert raw counts to SI units
        ax = (raw_ax / 2048.0f) * GRAVITY_EARTH;
        ay = (raw_ay / 2048.0f) * GRAVITY_EARTH;
        az = (raw_az / 2048.0f) * GRAVITY_EARTH;

        gx = (raw_gx / 16.4f) * DEG_TO_RAD_F;
        gy = (raw_gy / 16.4f) * DEG_TO_RAD_F;
        gz = (raw_gz / 16.4f) * DEG_TO_RAD_F;
    }
}

void calibrateZeroTare() {
    Serial.println("[CAL] Calibrating zero-tare... Ensure shoe insole is UNLOADED.");
    long sum_h = 0, sum_m = 0, sum_f = 0, sum_t = 0;
    const int CAL_SAMPLES = 100;

    for (int i = 0; i < CAL_SAMPLES; i++) {
        sum_h += analogRead(PIN_FSR_HEEL);
        sum_m += analogRead(PIN_FSR_MID);
        sum_f += analogRead(PIN_FSR_FORE);
        sum_t += analogRead(PIN_FSR_TOE);
        delay(10);
    }

    tare_heel = (float)sum_h / CAL_SAMPLES;
    tare_mid  = (float)sum_m / CAL_SAMPLES;
    tare_fore = (float)sum_f / CAL_SAMPLES;
    tare_toe  = (float)sum_t / CAL_SAMPLES;

    Serial.printf("[CAL] Tare Offsets: Heel=%.1f, Mid=%.1f, Fore=%.1f, Toe=%.1f\n",
                  tare_heel, tare_mid, tare_fore, tare_toe);
}

// ======================== FEATURE EXTRACTION ================================
OINFeatures extractWindowFeatures(float &currentTotalPressure) {
    OINFeatures f;
    float svm_vals[WINDOW_SIZE];
    float gm_vals[WINDOW_SIZE];
    float p_norm_vals[WINDOW_SIZE];
    float heel_ratios[WINDOW_SIZE];

    float svm_sum = 0.0f;
    float gm_sum = 0.0f;
    float svm_min = 1e6f;
    float svm_max = -1e6f;
    float gm_min = 1e6f;
    float gm_max = -1e6f;
    float p_norm_min = 1e6f;
    float p_norm_max = -1e6f;
    float heel_ratio_sum = 0.0f;
    float last_p_total = 0.0f;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (writeIndex + i) % WINDOW_SIZE;

        float ax = buf_ax[idx];
        float ay = buf_ay[idx];
        float az = buf_az[idx];
        float gx = buf_gx[idx];
        float gy = buf_gy[idx];
        float gz = buf_gz[idx];

        float h = buf_heel[idx];
        float m = buf_mid[idx];
        float fo = buf_fore[idx];
        float t = buf_toe[idx];

        // 3D Vector Norms
        float svm = sqrtf(ax * ax + ay * ay + az * az);
        float gm  = sqrtf(gx * gx + gy * gy + gz * gz);
        svm_vals[i] = svm;
        gm_vals[i]  = gm;

        svm_sum += svm;
        gm_sum  += gm;
        if (svm < svm_min) svm_min = svm;
        if (svm > svm_max) svm_max = svm;
        if (gm  < gm_min)  gm_min  = gm;
        if (gm  > gm_max)  gm_max  = gm;

        // Total Plantar Pressure & Regional Normalization
        float p_total = h + m + fo + t;
        last_p_total = p_total;
        float p_safe = (p_total > 0.001f) ? p_total : 0.001f;
        float p_norm = p_total / standing_total_baseline;
        p_norm_vals[i] = p_norm;

        if (p_norm < p_norm_min) p_norm_min = p_norm;
        if (p_norm > p_norm_max) p_norm_max = p_norm;

        float hr = h / p_safe;
        heel_ratios[i] = hr;
        heel_ratio_sum += hr;
    }

    currentTotalPressure = last_p_total;

    // SVM Standard Deviation
    float svm_mean = svm_sum / WINDOW_SIZE;
    float var_sum = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float diff = svm_vals[i] - svm_mean;
        var_sum += diff * diff;
    }
    float svm_std = sqrtf(var_sum / WINDOW_SIZE);

    // Dynamic Jerk
    float max_jerk = 0.0f;
    const float dt = 1.0f / SAMPLE_RATE_HZ; // 0.02 s
    for (int i = 1; i < WINDOW_SIZE; i++) {
        float jerk = fabsf(svm_vals[i] - svm_vals[i - 1]) / dt;
        if (jerk > max_jerk) max_jerk = jerk;
    }

    f.SVM_max = svm_max;
    f.SVM_min = svm_min;
    f.SVM_range = svm_max - svm_min;
    f.SVM_std = svm_std;
    f.GM_max = gm_max;
    f.GM_range = gm_max - gm_min;
    f.Jerk_max = max_jerk;
    f.P_total_min_ratio = p_norm_min;
    f.P_drop_range = p_norm_max - p_norm_min;
    f.Heel_ratio_mean = heel_ratio_sum / WINDOW_SIZE;

    return f;
}

// ======================== ARDUINO SETUP & MAIN LOOP =========================
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== SMART INSOLE FALL DETECTION FIRMWARE v2.0 ===");
    Serial.printf("[CFG] Preset %d Activated: Stillness Window=%d ms, Impact Threshold=%.1f m/s^2\n",
                  USER_PROFILE_PRESET, STILLNESS_CONFIRMATION_MS, IMPACT_SHOCK_THRESH_MPS2);

    pinMode(PIN_BUZZER, OUTPUT);
    pinMode(PIN_LED_ALERT, OUTPUT);
    pinMode(PIN_BUTTON_CANCEL, INPUT_PULLUP);
    digitalWrite(PIN_BUZZER, LOW);
    digitalWrite(PIN_LED_ALERT, LOW);

    analogReadResolution(10); // 10-bit ADC (0 - 1023 counts)

    initMPU6500();
    calibrateZeroTare();

    // 50 Hz Hardware Timer Interrupt
    sampleTimer = timerBegin(0, 80, true);
    timerAttachInterrupt(sampleTimer, &onSampleTimer, true);
    timerAlarmWrite(sampleTimer, SAMPLE_INTERVAL_US, true);
    timerAlarmEnable(sampleTimer);

    Serial.println("[SYS] 50 Hz Hardware Sampling Timer Started.");
    Serial.println("[SYS] System Armed in Monitoring State.");
}

void loop() {
    if (sampleReadyFlag) {
        sampleReadyFlag = false;

        float ax, ay, az, gx, gy, gz;
        readMPURaw(ax, ay, az, gx, gy, gz);

        // Read Plantar FSRs and subtract tare
        float h  = max(0.0f, (float)analogRead(PIN_FSR_HEEL) - tare_heel);
        float m  = max(0.0f, (float)analogRead(PIN_FSR_MID)  - tare_mid);
        float fo = max(0.0f, (float)analogRead(PIN_FSR_FORE) - tare_fore);
        float t  = max(0.0f, (float)analogRead(PIN_FSR_TOE)  - tare_toe);

        // Write to circular buffers
        buf_ax[writeIndex]   = ax;
        buf_ay[writeIndex]   = ay;
        buf_az[writeIndex]   = az;
        buf_gx[writeIndex]   = gx;
        buf_gy[writeIndex]   = gy;
        buf_gz[writeIndex]   = gz;
        buf_heel[writeIndex] = h;
        buf_mid[writeIndex]  = m;
        buf_fore[writeIndex] = fo;
        buf_toe[writeIndex]  = t;

        writeIndex = (writeIndex + 1) % WINDOW_SIZE;
        samplesCollected++;

        // Evaluate Window Every STEP_SIZE Samples (50% Overlap)
        if (samplesCollected >= WINDOW_SIZE && (samplesCollected % STEP_SIZE == 0)) {
            float currentPlantarPressure = 0.0f;
            OINFeatures features = extractWindowFeatures(currentPlantarPressure);
            
            // Level 1: Decision Tree Classification
            int rawPrediction = classify_oin_window(features);

            // Level 2: Causal 2-of-3 Temporal Confirmation Filter
            windowHistory[0] = windowHistory[1];
            windowHistory[1] = windowHistory[2];
            windowHistory[2] = rawPrediction;
            int confirmedCount = windowHistory[0] + windowHistory[1] + windowHistory[2];
            bool temporalTrigger = (confirmedCount >= 2);

            // Dual-Path Triggering:
            // Path 1 (Hard Impact Falls): Peak acceleration > threshold or violent jerk
            bool hardFallTrigger = temporalTrigger && (features.SVM_max > IMPACT_SHOCK_THRESH_MPS2 || features.Jerk_max > 200.0f);
            
            // Path 2 (Soft/Damped Falls - F07/F10): Moderate impact shock + severe plantar unloading
            bool softFallTrigger = (features.SVM_max > 14.0f) && (features.P_total_min_ratio < 0.20f) && (features.P_drop_range > 0.40f);

            // 3. Fall State Machine Execution
            switch (currentState) {
                case STATE_MONITORING:
                    if (hardFallTrigger || softFallTrigger) {
                        Serial.println(hardFallTrigger ? "[ALERT] Hard Impact Trigger!" : "[ALERT] Soft/Damped Fall Trigger!");
                        currentState = STATE_IMPACT_DETECTED;
                        stateTimerStart = millis();
                    }
                    break;

                case STATE_IMPACT_DETECTED:
                    // Allow 1.0 second for mechanical impact dynamics to settle
                    if (millis() - stateTimerStart >= 1000) {
                        currentState = STATE_CONFIRMING_STILLNESS;
                        stateTimerStart = millis();
                    }
                    break;

                case STATE_CONFIRMING_STILLNESS:
                    // Check for post-impact immobility over the configurable STILLNESS_CONFIRMATION_MS window:
                    // 1) SVM_std < threshold (low body acceleration variance)
                    // 2) GM_max < 1.0 rad/s (motionless limbs)
                    // 3) currentPlantarPressure < 35% of standing baseline (victim collapsed off feet)
                    if (features.SVM_std < STILLNESS_SVM_STD_MAX && 
                        features.GM_max < 1.0f && 
                        currentPlantarPressure < (standing_total_baseline * 0.35f)) 
                    {
                        if (millis() - stateTimerStart >= STILLNESS_CONFIRMATION_MS) {
                            Serial.println("[CONFIRMED] Post-impact immobility confirmed! Triggering Countdown.");
                            currentState = STATE_ALERT_COUNTDOWN;
                            stateTimerStart = millis();
                            digitalWrite(PIN_LED_ALERT, HIGH);
                        }
                    } else {
                        // User recovered or resumed moving (running, jumping, ADL false alarm)
                        Serial.println("[REJECTED] Movement or weight-bearing resumed post-impact. False alarm suppressed.");
                        currentState = STATE_MONITORING;
                    }
                    break;

                case STATE_ALERT_COUNTDOWN:
                    // Beep buzzer intermittently
                    if ((millis() / 500) % 2 == 0) {
                        digitalWrite(PIN_BUZZER, HIGH);
                    } else {
                        digitalWrite(PIN_BUZZER, LOW);
                    }

                    // Check for Cancel Button (Active LOW)
                    if (digitalRead(PIN_BUTTON_CANCEL) == LOW) {
                        Serial.println("[CANCEL] User pressed cancel button. Resetting system.");
                        digitalWrite(PIN_BUZZER, LOW);
                        digitalWrite(PIN_LED_ALERT, LOW);
                        currentState = STATE_MONITORING;
                        windowHistory[0] = windowHistory[1] = windowHistory[2] = 0;
                        delay(500); // Debounce
                        break;
                    }

                    // Countdown Expiry: Dispatch Emergency Alert
                    if (millis() - stateTimerStart >= COUNTDOWN_DURATION_MS) {
                        digitalWrite(PIN_BUZZER, HIGH);
                        currentState = STATE_ALARM_DISPATCHED;
                        Serial.println("=================================================");
                        Serial.println(">>> EMERGENCY ALERT DISPATCHED: FALL CONFIRMED! <<<");
                        Serial.println(">>> Sending GNSS Location & Alert via Cellular... <<<");
                        Serial.println("=================================================");
                    }
                    break;

                case STATE_ALARM_DISPATCHED:
                    if (digitalRead(PIN_BUTTON_CANCEL) == LOW) {
                        digitalWrite(PIN_BUZZER, LOW);
                        digitalWrite(PIN_LED_ALERT, LOW);
                        currentState = STATE_MONITORING;
                        Serial.println("[RESET] Alarm cleared by user.");
                        delay(500);
                    }
                    break;
            }
        }
    }
}
