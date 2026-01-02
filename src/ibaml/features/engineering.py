from __future__ import annotations
from typing import Dict, List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def zscore(s: pd.Series, w: int) -> pd.Series:
    """Compute rolling z-score normalization."""
    mu = s.rolling(window=w, min_periods=1).mean()
    sd = s.rolling(window=w, min_periods=1).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def apply_minmax_scaler(X_group: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """Scale group features using MinMaxScaler fitted on training indices."""
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    scaler.fit(X_group.iloc[train_idx])
    Xs = scaler.transform(X_group).astype("float32", copy=False)
    return pd.DataFrame(Xs, index=X_group.index, columns=X_group.columns, dtype="float32")


def build_group_features(
    df_factors: pd.DataFrame, 
    factors_map: Dict[str, List[str]], 
    zscore_windows: List[int]
) -> Dict[str, pd.DataFrame]:
    """Build z-score normalized features for each factor group.
    
    For each factor in each group, creates z-score normalized versions
    using the specified rolling windows.
    
    Args:
        df_factors: DataFrame containing raw factor data
        factors_map: Dict mapping group names to lists of factor column names
        zscore_windows: List of rolling window sizes (e.g., [6, 12] for 6 and 12 months)
        
    Returns:
        Dict mapping group names to DataFrames of z-score normalized features
        Column naming: {factor}_{window}m_zs
    """
    out = {}
    for group, cols in factors_map.items():
        frames = {}
        for c in cols:
            if c not in df_factors.columns:
                continue
            v = df_factors[c].astype(float)
            for w in zscore_windows:
                frames[f"{c}_{w}m_zs"] = zscore(v, int(w))
        out[group] = pd.DataFrame(frames, index=df_factors.index, dtype="float32")
    return out


def build_endogenous_features(df_targets: pd.DataFrame, target: str, zscore_windows: List[int]) -> pd.DataFrame:
    """Build endogenous features for a target using its own historical returns.
    
    Creates:
    - {target}_raw: Raw target returns
    - {target}_{window}m_zs: Z-score normalized returns for each window
    
    Args:
        df_targets: DataFrame containing target returns
        target: target name (column in df_targets)
        zscore_windows: List of rolling window sizes for z-score normalization
        
    Returns:
        DataFrame with endogenous features indexed by the same dates as df_targets
    """
    frames = {}
    if target in df_targets.columns:
        v = df_targets[target].astype(float)
        frames[f"{target}_raw"] = v
        for w in zscore_windows:
            frames[f"{target}_{w}m_zs"] = zscore(v, int(w))
    return pd.DataFrame(frames, index=df_targets.index, dtype="float32")


def build_exogenous_features(
    df_factors: pd.DataFrame,
    factors_map: Dict[str, List[str]],
    zscore_windows: List[int],
) -> pd.DataFrame:
    """Build exogenous factor features (all groups combined).
    
    Creates z-score normalized features for all factor groups and concatenates them.
    This is used in benchmarking where all exogenous features are treated uniformly
    without group-specific polynomial aggregation.
    
    Args:
        df_factors: Raw factors DataFrame
        factors_map: Dict mapping group names to factor column names
        zscore_windows: Rolling windows for z-score normalization
        
    Returns:
        DataFrame with all z-score normalized exogenous features concatenated
    """
    group_frames = build_group_features(df_factors, factors_map, zscore_windows)
    
    # Concatenate all groups
    all_frames = []
    for group_name, df in group_frames.items():
        if df.shape[1] > 0:
            all_frames.append(df)
    
    if not all_frames:
        return pd.DataFrame(index=df_factors.index)
    
    return pd.concat(all_frames, axis=1)

