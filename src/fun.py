import pandas as pd
import numpy as np

def get_percentile_filtered(df, col='total_nights', percentiles=[0.95, 0.98, 0.99]):
    filtered_data = {}
    for p in percentiles:
        cutoff = df[col].quantile(p)
        print(f"{int(p*100)}% of bookings have {col} ≤ {cutoff:.0f}")
        filtered_data[p] = df[df[col] <= cutoff][col]
    return filtered_data


def compute_survival_function(series):
    """
    Compute PMF and Survival Function (SF ≥ N) for a pandas Series of counts.

    Parameters
    ----------
    series : pd.Series
        Numeric data, e.g., total_nights.

    Returns
    -------
    pmf : pd.Series
        Probability Mass Function (normalized counts).
    sf : pd.Series
        Survival Function (≥ N).
    """
    counts = series.value_counts().sort_index()       # 每个数值出现的次数
    pmf = counts / counts.sum()                        # 转为概率
    sf = 1 - pmf.cumsum() + pmf                        # SF ≥ N
    return pmf, sf