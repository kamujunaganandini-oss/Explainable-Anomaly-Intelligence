import numpy as np
from scipy.stats import chi2

import pandas as pd
from datetime import timedelta


def clip(x, low=0.0, high=1.0):
    return max(low, min(x, high))


def compute_risk_zone_score(values, bounds):
    """
    values: array-like of feature values in the window
    bounds: dict with expected / caution / critical thresholds
    """
    zone_scores = []

    for v in values:
        # Lower-bound features (inventory, fulfillment, etc.)
        if "expected_min" in bounds:
            if v >= bounds["expected_min"]:
                zone_scores.append(0.0)
            elif v >= bounds["caution_min"]:
                zone_scores.append(0.5)
            else:
                zone_scores.append(1.0)

        # Upper-bound features (lead_time, backorders, etc.)
        else:
            if v <= bounds["expected_max"]:
                zone_scores.append(0.0)
            elif v <= bounds["caution_max"]:
                zone_scores.append(0.5)
            else:
                zone_scores.append(1.0)

    return np.mean(zone_scores), max(zone_scores)


def run_stage1_v2(
    df,
    time_column,
    feature_list,
    feature_bounds,
    window_days=14,
    alpha=0.01,
    weights=None
):
    """
    CADEN Stage 1 v2
    Governed, windowed, risk-aware anomaly assessment
    """

    if weights is None:
        weights = {
            "t2": 0.35,
            "persistence": 0.25,
            "risk_zone": 0.25,
        }

    # -----------------------------
    # STEP 0 — Select window
    # -----------------------------
    '''
    # TEMPORARY TEST OVERRIDE
    window_df = df[
        (df[time_column] >= "2025-12-10") &
        (df[time_column] <= "2025-12-23")
    ]
    window_start = "2025-12-10"
    window_end = "2025-12-23"'''
    
    
    # Ensure date column is datetime  - main code block
    df[time_column] = pd.to_datetime(df[time_column])

    end_date = df[time_column].max()
    start_date = end_date - timedelta(days=window_days)

    window_df = df[(df[time_column] >= start_date) &(df[time_column] <= end_date)]
    window_start = start_date
    window_end = end_date

    if window_df.empty or len(window_df) < 2:
        return {
            "anomaly_level": "none",
            "risk_score": 0.0,
            "decision_gate": "stop",
            "reason": "Insufficient data in analysis window",
            "window_start": window_start,
            "window_end": window_end,
            "t2_summary": {
                "max": 0.0,
                "mean": 0.0,
                "exceedance_count": 0
            }
        }


    # -----------------------------
    # STEP 1 — Multivariate matrix
    # -----------------------------
    X = window_df[feature_list].values

    # -----------------------------
    # STEP 2 — Hotelling T²
    # -----------------------------
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)

    # Numerical safety
    Sigma += 1e-6 * np.eye(Sigma.shape[0])
    Sigma_inv = np.linalg.inv(Sigma)

    T2 = []
    for x in X:
        diff = x - mu
        T2.append(diff.T @ Sigma_inv @ diff)

    T2 = np.array(T2)

    # -----------------------------
    # STEP 3 — Chi-square threshold
    # -----------------------------
    p = X.shape[1]
    threshold = chi2.ppf(1 - alpha, p)

    # -----------------------------
    # STEP 4 — T² summary
    # -----------------------------
    T2_mean = float(np.mean(T2))
    T2_max = float(np.max(T2))
    exceedance_count = int(np.sum(T2 > threshold))

    # -----------------------------
    # STEP 5 — T² score (Signal A)
    # -----------------------------
    T2_score = clip(
        0.4 * (T2_mean / threshold) +
        0.6 * (T2_max / threshold)
    )

    # -----------------------------
    # STEP 6 — Persistence score (Signal B)
    # -----------------------------
    persistence_score = clip(exceedance_count / 5)

    # -----------------------------
    # STEP 7 — Risk zone score (Signal C)
    # -----------------------------
    zone_scores = []
    risk_zone_hits = {}

    for feature, bounds in feature_bounds.items():
        values = window_df[feature].values
        mean_score, max_score = compute_risk_zone_score(values, bounds)
        zone_scores.append(mean_score)

        if max_score == 1.0:
            risk_zone_hits[feature] = "critical"
        elif max_score == 0.5:
            risk_zone_hits[feature] = "caution"
        else:
            risk_zone_hits[feature] = "expected"

    risk_zone_score = clip(float(np.mean(zone_scores))) if zone_scores else 0.0

    critical_count = sum(1 for v in risk_zone_hits.values() if v == "critical")

   

    # -----------------------------
    # STEP 9 — Final risk score
    # -----------------------------
    risk_score = clip(
        weights["t2"] * T2_score +
        weights["persistence"] * persistence_score +
        weights["risk_zone"] * risk_zone_score
    )
    # Risk override: many critical KPIs
    risk_override = (risk_zone_score >= 0.6 and critical_count >= 2)

    # -----------------------------
    # STEP 10 — Anomaly level
    # -----------------------------
    if risk_override:
        anomaly_level = "strong"
    elif risk_score < 0.20:
        anomaly_level = "none"
    elif risk_score < 0.40:
        anomaly_level = "marginal"
    elif risk_score < 0.65:
        anomaly_level = "moderate"
    else:
        anomaly_level = "strong"

    # -----------------------------
    # STEP 11 — Decision gate
    # -----------------------------
    decision_gate = "proceed" if anomaly_level in ["moderate", "strong"] else "stop"

    # -----------------------------
    # STEP 12 — Output
    # -----------------------------
    print("T2 score:", T2_score)
    print("Persistence score:", persistence_score)
    print("Risk zone score:", risk_zone_score)
    return {
        "window_start": window_start,
        "window_end": window_end,

        "anomaly_level": anomaly_level,
        "risk_score": round(risk_score, 3),

        "t2_summary": {
            "mean": round(T2_mean, 3),
            "max": round(T2_max, 3),
            "threshold": round(float(threshold), 3),
            "exceedance_count": exceedance_count
        },

        "risk_zone_hits": risk_zone_hits,
        "decision_gate": decision_gate
    }
