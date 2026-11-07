#ifndef DECISION_TREE_CLASSIFIER_H
#define DECISION_TREE_CLASSIFIER_H

/**
 * Orientation-Invariant Decision Tree Classifier for Smart Insole Fall Detection.
 * 
 * Automatically generated from training on 21 activity classes (including 31 fall trials).
 * Operates strictly on 3D vector norms and normalized pressure metrics (OIN-10).
 * 
 * Features required:
 *   0: GM_max             (Peak angular speed magnitude, rad/s)
 *   1: SVM_max            (Peak acceleration magnitude, m/s^2)
 *   2: SVM_min            (Minimum acceleration magnitude, m/s^2)
 *   3: SVM_range          (SVM_max - SVM_min, m/s^2)
 *   4: SVM_std            (Standard deviation of SVM, m/s^2)
 *   5: GM_range           (GM_max - GM_min, rad/s)
 *   6: Jerk_max           (Peak rate of acceleration change, m/s^3)
 *   7: P_total_min_ratio  (Min total pressure normalized by baseline)
 *   8: P_drop_range       (Max - Min normalized total pressure)
 *   9: Heel_ratio_mean    (Average load distribution on heel: 0.0 - 1.0)
 * 
 * Returns:
 *   0: Normal activity (no fall)
 *   1: Fall-related event candidate
 */

struct OINFeatures {
    float GM_max;
    float SVM_max;
    float SVM_min;
    float SVM_range;
    float SVM_std;
    float GM_range;
    float Jerk_max;
    float P_total_min_ratio;
    float P_drop_range;
    float Heel_ratio_mean;
};

inline int classify_oin_window(const OINFeatures& f) {
    if (f.GM_max <= 1.24f) {
        if (f.SVM_max <= 10.03f) {
            return 0; // Normal quiet activity
        } else {
            if (f.SVM_min <= 5.08f) {
                if (f.SVM_std <= 1.00f) {
                    if (f.Jerk_max <= 249.19f) {
                        return 0;
                    } else {
                        return 1; // High jerk transition with low baseline
                    }
                } else {
                    return 0;
                }
            } else {
                if (f.SVM_range <= 1.19f) {
                    if (f.SVM_max <= 10.42f) {
                        return 0;
                    } else {
                        return 1;
                    }
                } else {
                    return 0;
                }
            }
        }
    } else { // GM_max > 1.24 rad/s (tumbling / rapid movement)
        if (f.P_drop_range <= 0.55f) {
            if (f.SVM_max <= 11.07f) {
                return 1;
            } else {
                if (f.P_drop_range <= 0.12f) {
                    return 1;
                } else {
                    if (f.SVM_min <= 9.03f) {
                        return 1;
                    } else {
                        return 0;
                    }
                }
            }
        } else { // P_drop_range > 0.55 (significant plantar pressure swing)
            if (f.SVM_min <= 5.93f) {
                if (f.GM_range <= 5.81f) {
                    if (f.GM_max <= 1.90f) {
                        return 0;
                    } else {
                        return 1;
                    }
                } else {
                    if (f.Heel_ratio_mean <= 0.01f) {
                        return 1; // Plantar off-ground (feet horizontal)
                    } else {
                        return 0;
                    }
                }
            } else {
                if (f.SVM_range <= 7.80f) {
                    if (f.Heel_ratio_mean <= 0.0001f) {
                        return 0;
                    } else {
                        return 1;
                    }
                } else {
                    if (f.P_total_min_ratio <= 0.0001f) {
                        return 0;
                    } else {
                        return 1;
                    }
                }
            }
        }
    }
}

#endif // DECISION_TREE_CLASSIFIER_H
