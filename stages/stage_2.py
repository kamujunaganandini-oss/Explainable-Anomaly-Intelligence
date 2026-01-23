def enrich_anomaly_context(anomaly_row, metadata_sources):
    """
    Retrieves temporal, operational, and relational context for an anomaly.
    
    Parameters:
    - anomaly_row: Row from anomaly detection output
    - metadata_sources: Dict of contextual data sources (inventory, suppliers, calendar)
    
    Returns:
    - Dict with enriched context
    """
    context = {
        'date': anomaly_row['date'],
        'anomaly_magnitude': anomaly_row['T2_score'],
        'temporal': get_temporal_context(anomaly_row['date'], metadata_sources['calendar']),
        'operational': get_operational_context(anomaly_row, metadata_sources['suppliers']),
        'relational': get_relational_context(anomaly_row, metadata_sources['cross_metrics'])
    }
    return context