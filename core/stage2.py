import numpy as np

def run_stage2_v2(state_df, stage1_output, features, baseline_days):
    """
    Stage 2 v2 (Refactored)
    Extracts directional signals for Bayesian likelihoods
    """

    window_start = stage1_output["window_start"]
    window_end = stage1_output["window_end"]

    # Analysis window
    window_df = state_df[
        (state_df["time"] >= window_start) &
        (state_df["time"] <= window_end)
    ]

    # Baseline window
    baseline_df = state_df[
        (state_df["time"] < window_start)
    ].tail(baseline_days)

    signals = {}

    for feature in features:
        window_mean = window_df[feature].mean()
        baseline_mean = baseline_df[feature].mean()
        baseline_std = baseline_df[feature].std()

        if baseline_std == 0 or np.isnan(baseline_std):
            continue

        delta = window_mean - baseline_mean
        z = delta / baseline_std

        # Direction
        if abs(z) < 0.5:
            direction = "flat"
        elif z > 0:
            direction = "up"
        else:
            direction = "down"

        # Strength (bounded)
        strength = min(abs(z) / 3.0, 1.0)

        signals[feature] = {
            "direction": direction,
            "strength": round(strength, 3)
        }

    return {
        "signals": signals
    }
