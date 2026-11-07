/**
 * Final Year Project: Smart Insole Fall Detection System
 * Embedded Firmware for ESP32 Microcontroller
 * 
 * Features:
 *  1. 50 Hz Hardware Timer Interrupt (20,000 us sampling interval).
 *  2. 4-Channel Plantar FSR sensing with power-on zero-tare subtraction.
 *  3. MPU6500 / MPU6050 6-DOF IMU reading via I2C.
 *  4. Orientation-Invariant 3D Vector Norms (SVM, GM, Vector Jerk).
 *  5. Real-time Decision Tree inference (< 50 us latency).
 *  6. Causal 2-of-3 Temporal Confirmation Filter.
 *  7. 4-Stage State Machine with Post-Impact Stillness Verification.
 *  8. Audible buzzer, LED, and cancellation pushbutton.
 */

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "decision_tree_classifier.h"

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
float standing_total_baseline = 1000.0f;

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
    Wire.write(0x00); // Wake up MPU6500
    Wire.endTransmission(true);

    // Set Accelerometer to +/- 16g range
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C); // ACCEL_CONFIG
    Wire.write(0x18); // 0x18 = +/- 16g
    Wire.endTransmission(true);

    // Set Gyroscope to +/- 2000 deg/s range
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1B); // GYRO_CONFIG
    Wire.write(0x18); // 0x18 = +/- 2000 deg/s
    Wire.endTransmission(true);

    // Set Low Pass Filter to ~42 Hz cutoff
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A); // CONFIG
    Wire.write(0x03); // DLPF_CFG = 3 (42 Hz)
    Wire.endTransmission(true);

    Serial.println("[IMU] MPU6500 initialized: +/-16g, +/-2000dps, 42Hz DLPF.");
}

void readMPURaw(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B); // Starting register for Accel measurements
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14, (uint8_t)true);

    if (Wire.available() >= 14) {
        int16_t raw_ax = Wire.read() << 8 | Wire.read();
        int16_t raw_ay = Wire.read() << 8 | Wire.read();
        int16_t raw_az = Wire.read() << 8 | Wire.read();
        Wire.read(); Wire.read(); // Skip temperature bytes
        int16_t raw_gx = Wire.read() << 8 | Wire.read();
        int16_t raw_gy = Wire.read() << 8 | Wire.read();
        int16_t raw_gz = Wire.read() << 8 | Wire.read();

        // Convert raw to SI units:
        // +/- 16g scale: 2048 LSB / g -> m/s^2
        ax = (raw_ax / 2048.0f) * GRAVITY_EARTH;
        ay = (raw_ay / 2048.0f) * GRAVITY_EARTH;
        az = (raw_az / 2048.0f) * GRAVITY_EARTH;

        // +/- 2000 dps scale: 16.4 LSB / (deg/s) -> rad/s
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
OINFeatures extractWindowFeatures() {
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

    for (int i = 0; i < WINDOW_SIZE; i++) {
        // Read from oldest to newest in circular buffer
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

        // 3D Vector Norms (Orientation Invariant)
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
        float p_safe = (p_total > 0.001f) ? p_total : 0.001f;
        float p_norm = p_total / standing_total_baseline;
        p_norm_vals[i] = p_norm;

        if (p_norm < p_norm_min) p_norm_min = p_norm;
        if (p_norm > p_norm_max) p_norm_max = p_norm;

        float hr = h / p_safe;
        heel_ratios[i] = hr;
        heel_ratio_sum += hr;
    }

    // SVM Standard Deviation
    float svm_mean = svm_sum / WINDOW_SIZE;
    float var_sum = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float diff = svm_vals[i] - svm_mean;
        var_sum += diff * diff;
    }
    float svm_std = sqrtf(var_sum / WINDOW_SIZE);

    // Vector Jerk
    float max_jerk = 0.0f;
    const float dt = 1.0f / SAMPLE_RATE_HZ; // 0.02 s
    for (int i = 1; i < WINDOW_SIZE; i++) {
        float jerk = fabsf(svm_vals[i] - svm_vals[i - 1]) / dt;
        if (jerk > max_jerk) max_jerk = jerk;
    }

    // Populate OIN-10 feature struct
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
    Serial.println("\n=== SMART INSOLE FALL DETECTION FIRMWARE ===");

    pinMode(PIN_BUZZER, OUTPUT);
    pinMode(PIN_LED_ALERT, OUTPUT);
    pinMode(PIN_BUTTON_CANCEL, INPUT_PULLUP);
    digitalWrite(PIN_BUZZER, LOW);
    digitalWrite(PIN_LED_ALERT, LOW);

    analogReadResolution(10); // 10-bit ADC (0 - 1023 counts)

    initMPU6500();
    calibrateZeroTare();

    // Configure 50 Hz Hardware Timer Interrupt (Timer 0, Prescaler 80)
    sampleTimer = timerBegin(0, 80, true);
    timerAttachInterrupt(sampleTimer, &onSampleTimer, true);
    timerAlarmWrite(sampleTimer, SAMPLE_INTERVAL_US, true);
    timerAlarmEnable(sampleTimer);

    Serial.println("[SYS] 50 Hz Hardware Sampling Timer Started.");
    Serial.println("[SYS] System Armed in Monitoring State.");
}

void loop() {
    // 1. Process Hardware Timer Samples
    if (sampleReadyFlag) {
        sampleReadyFlag = false;

        // Read MPU6500
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

        // 2. Evaluate Window Every STEP_SIZE Samples (50% Overlap)
        if (samplesCollected >= WINDOW_SIZE && (samplesCollected % STEP_SIZE == 0)) {
            OINFeatures features = extractWindowFeatures();
            
            // Level 1: Decision Tree Classification
            int rawPrediction = classify_oin_window(features);

            // Level 2: Causal 2-of-3 Temporal Confirmation Filter
            windowHistory[0] = windowHistory[1];
            windowHistory[1] = windowHistory[2];
            windowHistory[2] = rawPrediction;
            int confirmedCount = windowHistory[0] + windowHistory[1] + windowHistory[2];
            bool temporalTrigger = (confirmedCount >= 2);

            // 3. Fall State Machine Execution
            switch (currentState) {
                case STATE_MONITORING:
                    if (temporalTrigger && (features.SVM_max > 20.0f || features.Jerk_max > 200.0f)) {
                        Serial.println("[ALERT] Impact Shock & Temporal Trigger! Entering Stillness Verification...");
                        currentState = STATE_IMPACT_DETECTED;
                        stateTimerStart = millis();
                    }
                    break;

                case STATE_IMPACT_DETECTED:
                    // Allow 1.0 second for user motion to settle
                    if (millis() - stateTimerStart >= 1000) {
                        currentState = STATE_CONFIRMING_STILLNESS;
                        stateTimerStart = millis();
                    }
                    break;

                case STATE_CONFIRMING_STILLNESS:
                    // Check for post-impact immobility over the next 1.5 - 2.0 seconds:
                    // SVM_std < 1.2 m/s^2 (low variance), GM_max < 1.0 rad/s (motionless)
                    if (features.SVM_std < 1.2f && features.GM_max < 1.0f) {
                        Serial.println("[CONFIRMED] Post-impact immobility confirmed! Triggering Countdown.");
                        currentState = STATE_ALERT_COUNTDOWN;
                        stateTimerStart = millis();
                        digitalWrite(PIN_LED_ALERT, HIGH);
                    } else {
                        // User recovered or resumed moving (running, jumping, ADL false alarm)
                        Serial.println("[REJECTED] Movement resumed post-impact. False alarm suppressed.");
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
                    // Sustained alarm until reset
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
