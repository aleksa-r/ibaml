from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple


def mask_from_indices(G: int, indices: Iterable[int]) -> np.ndarray:
    """Create boolean mask of length G with True at specified indices."""
    m = np.zeros(G, dtype=bool)
    for i in indices:
        if 0 <= i < G: m[i] = True
    return m


def compute_gbp_product(X: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Compute product of selected and complement features for GBP."""
    X = X.astype("float32", copy=False)
    sel = mask_bool.astype(bool)
    not_sel = ~sel
    pos = X[:, sel] if sel.any() else None
    neg = 1.0 - X[:, not_sel] if not_sel.any() else None
    prod_pos = np.prod(pos, axis=1) if pos is not None and pos.shape[1] > 0 else np.ones(X.shape[0], dtype="float32")
    prod_neg = np.prod(neg, axis=1) if neg is not None and neg.shape[1] > 0 else np.ones(X.shape[0], dtype="float32")
    return (prod_pos * prod_neg).astype("float32")

def precompute_log_base_ratio(X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute log base and log ratio for GBP calculations."""
    X = X_scaled.astype("float32", copy=False)
    X = np.clip(X, 0.01, 0.99)
    one_minus = 1.0 - X
    log_base = np.sum(np.log(one_minus, dtype=np.float32), axis=1).astype("float32")
    log_r = (np.log(X, dtype=np.float32) - np.log(one_minus, dtype=np.float32)).astype("float32")
    return log_base, log_r

def gbp_from_mask_logs(log_base: np.ndarray, log_r: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Compute GBP values from precomputed logs and mask."""
    if mask_bool.ndim == 1:
        s = mask_bool.astype(bool)
        out = log_base + np.sum(log_r[:, s], axis=1)
        return np.exp(out, dtype=np.float32).astype("float32")
    else:
        S = mask_bool.astype("float32", copy=False).T
        out = log_base[:, None] + log_r @ S
        return np.exp(out, dtype=np.float32).astype("float32")

