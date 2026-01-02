from __future__ import annotations
import numpy as np


def rmse(y_true, y_pred) -> float:
    """Calculate Root Mean Squared Error between true and predicted values."""
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def sign_accuracy(y_true, y_pred) -> float:
    """Calculate Sign Accuracy between true and predicted values."""
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    s_true = np.sign(y_true); s_pred = np.sign(y_pred)
    mask = ~np.isnan(s_true) & ~np.isnan(s_pred)
    if mask.sum() == 0: return float("nan")
    return float(np.mean(s_true[mask] == s_pred[mask]))


def sharpe_ratio(returns, eps: float = 1e-12) -> float:
    """Calculate annualized Sharpe Ratio from returns."""
    r = np.asarray(returns, dtype=np.float64)
    mu = np.nanmean(r); sd = np.nanstd(r)
    if sd < eps: return 0.0
    return float((mu / sd) * np.sqrt(12.0))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.
    Handles edge cases: empty arrays, zero peaks, flat returns.
    """
    x = np.asarray(equity_curve, dtype=np.float64)
    if x.size == 0: return 0.0
    
    if np.all(x == 0): return 0.0
    
    peak = np.maximum.accumulate(x)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = (x - peak) / np.where(peak != 0, peak, 1.0)
    
    dd = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
    
    return float(np.min(dd))

