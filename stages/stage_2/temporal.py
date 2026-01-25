# stages/stage_2/temporal.py

import pandas as pd

def extract_temporal_context(anomaly_row):
    date = pd.to_datetime(anomaly_row.date)

    return {
        "day_of_week": date.day_name(),
        "is_month_end": date.is_month_end,
        "is_quarter_end": date.is_quarter_end
    }
