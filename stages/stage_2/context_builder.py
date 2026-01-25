# stages/stage_2/context_builder.py

from .temporal import extract_temporal_context
from .operational import extract_operational_context
from .relational import extract_relational_context

def build_context(anomaly_row, df, feature_cols, config):
    """
    Build enriched context for a single anomaly.
    Stage 2 v1: anomaly + temporal only.
    """

    context = {
        "anomaly": {
            "date": anomaly_row.date,
            "t2_score": anomaly_row.T2_score,
            "features": {
                k: getattr(anomaly_row, k)
                for k in anomaly_row._fields
                if k not in ("Index", "date", "T2_score", "is_anomaly")
            }
        },
        "temporal": extract_temporal_context(anomaly_row),
        "operational": extract_operational_context(config),
        "relational": extract_relational_context(anomaly_row, df, feature_cols)
    }

    return context
