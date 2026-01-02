from __future__ import annotations
import pandas as pd
import numpy as np


def apply_date_filters(df, starting_date=None, end_date=None):
    """
    Filter a DataFrame or Series by starting and/or end date using the index.
    Args:
        df: pd.DataFrame or pd.Series with DatetimeIndex
        starting_date: str or pd.Timestamp or None
        end_date: str or pd.Timestamp or None
    Returns:
        Filtered DataFrame or Series
    """
    if starting_date is not None:
        df = df[df.index >= pd.to_datetime(starting_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df


def _read_csv(path: str, date_column: str, parse_dates: bool) -> pd.DataFrame:
    """Read CSV with date parsing and index setting."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python")

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in CSV: {path}")

    if parse_dates:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format='mixed', errors='coerce')
        except Exception:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df[pd.notnull(df[date_column])]

    df = df.set_index(date_column)
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[pd.notnull(df.index)]
        except Exception:
            pass
    df = df.sort_index()
    return df


def _align(df: pd.DataFrame, freq: str, agg: str) -> pd.DataFrame:
    """Align DataFrame to specified frequency with aggregation."""
    if not freq:
        return df
    if str(freq).upper() == "M":
        freq = "ME"
    if agg == "last": return df.resample(freq).last()
    if agg == "mean": return df.resample(freq).mean()
    if agg == "sum":  return df.resample(freq).sum()
    return df


def read_raw_pair(factors_path: str, targets_path: str, date_column: str, parse_dates: bool, align_freq: str, align_agg: str, align_how: str):
    """
    Read and align factors and targets datasets.
    
    :param factors_path: factors CSV file path
    :type factors_path: str
    :param targets_path: targets CSV file path
    :type targets_path: str
    :param date_column: Date column name
    :type date_column: str
    :param parse_dates: Whether to parse dates
    :type parse_dates: bool
    :param align_freq: Alignment frequency
    :type align_freq: str
    :param align_agg: Alignment aggregation method
    :type align_agg: str
    :param align_how: Alignment method ('inner', 'left', 'right', 'end')
    :type align_how: str

    :return: Aligned factors and targets DataFrames
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    df_f = _read_csv(factors_path, date_column, parse_dates)
    df_t = _read_csv(targets_path, date_column, parse_dates)
    df_f = _align(df_f, align_freq, align_agg)
    df_t = _align(df_t, align_freq, align_agg)
    if align_how in ("inner","end"):
        idx = df_f.index.intersection(df_t.index)
    elif align_how == "left":
        idx = df_f.index
    elif align_how == "right":
        idx = df_t.index
    else:
        raise ValueError("align_how must be 'inner','left','right','end'")
    return df_f.loc[idx], df_t.loc[idx]


def cumulative_shifted_returns(df: pd.DataFrame, horizon: int, shift: int) -> pd.DataFrame:
    """Compute cumulative returns over a horizon with a shift."""
    h = int(horizon); s = int(shift)
    base = (1.0 + df.astype(float)).clip(lower=1e-9)
    log1p = np.log(base)
    roll = log1p.rolling(window=h, min_periods=h).sum()
    compounded = np.exp(roll) - 1.0
    return compounded.shift(-s)


def normalize_to_month_end(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Force timestamps to month-end to ensure consistent joins and rolling windows."""
    return index.to_period('M').to_timestamp('M')


def compute_holdout_indices(
    y: pd.Series | pd.DataFrame,
    eval_size: int,
    require_consecutive: bool = True,
    fallback_to_recent_valid: bool = True,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Compute train and holdout indices based on evaluation size and validity.

    :param y: Target series or DataFrame
    :type y: pd.Series | pd.DataFrame
    :param eval_size: Number of periods for holdout
    :type eval_size: int
    :param require_consecutive: Whether to require consecutive periods in holdout
    :type require_consecutive: bool
    :param fallback_to_recent_valid: Whether to fallback to most recent valid periods if not enough consecutive
    :type fallback_to_recent_valid: bool
    
    :return: Tuple of train indices and holdout indices
    :rtype: tuple[pd.DatetimeIndex, pd.DatetimeIndex]
    """
    if isinstance(y, pd.DataFrame):
        valid_mask = pd.notnull(y).any(axis=1)
        idx_all = pd.DatetimeIndex(y.index)
    else:
        valid_mask = pd.notnull(y)
        idx_all = pd.DatetimeIndex(y.index)

    idx_valid = idx_all[valid_mask.values]
    if len(idx_valid) == 0:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([])

    idx_valid = normalize_to_month_end(pd.DatetimeIndex(idx_valid))

    eval_size = int(max(1, eval_size))

    def _consecutive_tail(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if len(idx) == 0:
            return idx
        periods = idx.to_period("M")
        tail = [periods[-1]]
        for p in reversed(periods[:-1]):
            if tail[-1] - p == 1:
                tail.append(p)
            else:
                break
        tail = list(reversed(tail))
        return pd.DatetimeIndex([p.to_timestamp("M") for p in tail])

    if require_consecutive:
        tail = _consecutive_tail(idx_valid)
        if len(tail) >= eval_size:
            ho_idx = tail[-eval_size:]
        else:
            if fallback_to_recent_valid:
                ho_idx = idx_valid[-min(eval_size, len(idx_valid)):]
            else:
                ho_idx = tail
    else:
        ho_idx = idx_valid[-min(eval_size, len(idx_valid)):]

    train_idx = idx_all[idx_all < ho_idx[0]]
    return pd.DatetimeIndex(train_idx), pd.DatetimeIndex(ho_idx)

