# stages/stage_2/relational.py

import numpy as np

def extract_relational_context(anomaly_row, df, feature_cols):
    context = {}

    for feature in feature_cols:
        mean = df[feature].mean()
        std = df[feature].std()

        z = (getattr(anomaly_row, feature) - mean) / std if std > 0 else 0

        context[feature] = {
            "z_score": round(z, 2),
            "direction": "up" if z > 0 else "down" if z < 0 else "flat"
        }

    return {
        "feature_deviations": context,
        "pattern_break": True  # v1 heuristic: anomaly implies deviation
    }
