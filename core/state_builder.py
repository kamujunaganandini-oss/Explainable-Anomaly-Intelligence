import pandas as pd


SUPPORTED_AGGREGATIONS = {
    "sum": "sum",
    "mean": "mean",
    "max": "max",
    "min": "min",
    "count": "count",
    "nunique": "nunique"
}


def build_system_state(
    df_raw: pd.DataFrame,
    date_column: str,
    time_unit: str,
    aggregation_config: dict,
    date_format: str | None = None) -> pd.DataFrame:
    """
    Build time-indexed system state from raw event data.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw input data
    date_column : str
        Column representing business time
    time_unit : str
        One of ['day', 'week', 'hour']
    aggregation_config : dict
        Mapping of feature -> aggregation function
    date_format : str, optional
        Explicit datetime format if needed

    Returns
    -------
    pd.DataFrame
        Aggregated system state (one row per time unit)
    """

    df = df_raw.copy()

    # -----------------------------
    # STEP 0 — Validate inputs
    # -----------------------------
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in dataframe")

    for feature, agg in aggregation_config.items():
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe")
        if agg not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Aggregation '{agg}' not supported. "
                f"Supported: {list(SUPPORTED_AGGREGATIONS.keys())}"
            )

    # -----------------------------
    # STEP 1 — Parse datetime
    # -----------------------------
    if date_format:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])

    # -----------------------------
    # STEP 2 — Create time key
    # -----------------------------
    if time_unit == "day":
        df["time_key"] = df[date_column].dt.floor("D")
    elif time_unit == "week":
        df["time_key"] = df[date_column].dt.to_period("W").apply(lambda r: r.start_time)
    elif time_unit == "hour":
        df["time_key"] = df[date_column].dt.floor("H")
    else:
        raise ValueError("time_unit must be one of ['day', 'week', 'hour']")

    # -----------------------------
    # STEP 3 — Build aggregation dict
    # -----------------------------
    agg_dict = {
        feature: SUPPORTED_AGGREGATIONS[agg]
        for feature, agg in aggregation_config.items()
    }

    # -----------------------------
    # STEP 4 — Aggregate
    # -----------------------------
    state_df = (
        df
        .groupby("time_key", as_index=False)
        .agg(agg_dict)
        .rename(columns={"time_key": "time"})
        .sort_values("time")
        .reset_index(drop=True)
    )

    return state_df
