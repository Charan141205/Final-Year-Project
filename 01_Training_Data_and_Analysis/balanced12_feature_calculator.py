"""
Balanced-12 feature calculator for raw CAPPIMU 50-sample windows.

This module calculates the 12 selected Balanced-12 features from one
50-row window of the original CAPPIMU `insole_knee_r.csv` data.

It does not perform recording segmentation or create windows from a complete
participant file. It expects the caller to provide one complete 50-row window.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

G0 = 9.80665
DEG_TO_RAD = np.pi / 180.0

FEATURE_ORDER = [
    "FSR1_heel_proxy_range",
    "FSR2_midfoot_proxy_min",
    "FSR3_forefoot_proxy_mean",
    "FSR4_toe_proxy_min",
    "Gz_rad_s_range",
    "Ay_mps2_rms",
    "Az_mps2_max",
    "Gx_rad_s_std",
    "Ax_mps2_min",
    "Ay_mps2_max",
    "Az_mps2_mean",
    "Ax_mps2_rms",
]

def calculate_balanced12_from_raw_window(window: pd.DataFrame) -> dict[str, float]:
    """Return exactly 12 features in the model's required feature order."""
    required = [
        "insole_r(p1)", "insole_r(p2)", "insole_r(p3)",
        "insole_r(p4)", "insole_r(p5)", "insole_r(p6)",
        "insole_r(p7)", "insole_r(p8)",
        "knee_r(ax)", "knee_r(ay)", "knee_r(az)",
        "knee_r(wx)", "knee_r(wy)", "knee_r(wz)",
    ]
    missing = [c for c in required if c not in window.columns]
    if missing:
        raise ValueError(f"Missing raw CAPPIMU columns: {missing}")
    if len(window) != 50:
        raise ValueError(f"Expected exactly 50 samples, got {len(window)}")

    data = window[required].apply(pd.to_numeric, errors="coerce")
    values = data.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Window contains NaN or infinite source values")

    # Convert eight raw insole sensors into four regional pressure proxies.
    heel = data[["insole_r(p6)", "insole_r(p7)", "insole_r(p8)"]].mean(axis=1).to_numpy()
    midfoot = data[["insole_r(p4)", "insole_r(p5)"]].mean(axis=1).to_numpy()
    forefoot = data[["insole_r(p2)", "insole_r(p3)"]].mean(axis=1).to_numpy()
    toe = data["insole_r(p1)"].to_numpy()

    # Source accelerometer is in g; convert to metres/second squared.
    ax = data["knee_r(ax)"].to_numpy() * G0
    ay = data["knee_r(ay)"].to_numpy() * G0
    az = data["knee_r(az)"].to_numpy() * G0

    # Source gyroscope is in degrees/second; convert to radians/second.
    gx = data["knee_r(wx)"].to_numpy() * DEG_TO_RAD
    gz = data["knee_r(wz)"].to_numpy() * DEG_TO_RAD

    features = {
        "FSR1_heel_proxy_range": float(np.max(heel) - np.min(heel)),
        "FSR2_midfoot_proxy_min": float(np.min(midfoot)),
        "FSR3_forefoot_proxy_mean": float(np.mean(forefoot)),
        "FSR4_toe_proxy_min": float(np.min(toe)),
        "Gz_rad_s_range": float(np.max(gz) - np.min(gz)),
        "Ay_mps2_rms": float(np.sqrt(np.mean(ay ** 2))),
        "Az_mps2_max": float(np.max(az)),
        "Gx_rad_s_std": float(np.std(gx, ddof=0)),
        "Ax_mps2_min": float(np.min(ax)),
        "Ay_mps2_max": float(np.max(ay)),
        "Az_mps2_mean": float(np.mean(az)),
        "Ax_mps2_rms": float(np.sqrt(np.mean(ax ** 2))),
    }
    return {name: features[name] for name in FEATURE_ORDER}
