from __future__ import annotations
from typing import Iterable, List, Set
import itertools
import numpy as np

from ibaml.eval.metrics import rmse, sign_accuracy
from ibaml.iba.polynomials import gbp_from_mask_logs
from ibaml.losses.objective import make_xgb_objective, total_loss
from ibaml.utils.core_predict import xgb_core_predict


def mask_expression(all_cols: List[str], selected_cols: Iterable[str]) -> str:
    """
    Generate IBA formula expression with proper operators:
    - ⊗ for multiplication (between features)
    - (1 - x) for negation
    """
    sel: Set[str] = set(selected_cols)
    pos = [c for c in all_cols if c in sel]
    neg = [c for c in all_cols if c not in sel]
    parts: List[str] = []
    if pos: parts.append(" ⊗ ".join(pos))
    if neg: parts.append(" ⊗ ".join([f"(1 - {c})" for c in neg]))
    return " ⊗ ".join([p for p in parts if p]) if parts else "1"


def iter_nonempty_masks(n: int, max_size: Optional[int] = None):
    """Generate all non-empty combinations of feature indices up to max_size."""
    if n <= 0:
        return
    sizes = range(1, n + 1) if max_size is None else range(1, min(n, max_size) + 1)
    for k in sizes:
        for comb in itertools.combinations(range(n), k):
            yield comb


def eval_mask_over_folds_logs(y, fold_indices, per_fold_logs, selected_mask, xgb_params, paper_params):
    """Evaluate a given feature mask over all CV folds using precomputed logs."""
    losses, rmses, saccs = [], [], []
    obj = make_xgb_objective(paper_params)
    for (tr_idx, va_idx), (log_base, log_r) in zip(fold_indices, per_fold_logs):
        gbp = gbp_from_mask_logs(log_base, log_r, selected_mask)
        Xtr = gbp[tr_idx].reshape(-1, 1).astype("float32", copy=False)
        Xva = gbp[va_idx].reshape(-1, 1).astype("float32", copy=False)
        ytr = y[tr_idx].astype("float32", copy=False); yva = y[va_idx].astype("float32", copy=False)
        yhat = xgb_core_predict(Xtr, ytr, Xva, obj_fn=obj, xgb_params={"n_jobs": 1, **xgb_params})
        losses.append(total_loss(yva, yhat, paper_params))
        rmses.append(rmse(yva, yhat))
        saccs.append(sign_accuracy(yva, yhat))
    return float(np.mean(losses)), float(np.mean(rmses)), float(np.mean(saccs)), {"losses": losses, "rmses": rmses, "signacc": saccs}

