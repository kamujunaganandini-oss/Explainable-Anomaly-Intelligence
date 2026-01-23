import numpy as np
import pandas as pd
from scipy.stats import chi2

def detect_multivariate_anomalies(df, features, alpha=0.01):
    """
    Detects anomalies using Hotelling's T² statistic.
    
    Parameters:
    - df: DataFrame with time-series data
    - features: List of column names to monitor
    - alpha: Significance level
    
    Returns:
    - DataFrame with anomaly flags and T² scores
    """
    X = df[features].values
    mu = np.mean(X, axis=0)
    S_inv = np.linalg.inv(np.cov(X.T))
    
    T2_scores = []
    for i in range(len(X)):
        diff = X[i] - mu
        T2 = diff @ S_inv @ diff.T
        T2_scores.append(T2)
    
    critical_value = chi2.ppf(1 - alpha, df=len(features))
    df['T2_score'] = T2_scores
    df['is_anomaly'] = df['T2_score'] > critical_value
    
    return df