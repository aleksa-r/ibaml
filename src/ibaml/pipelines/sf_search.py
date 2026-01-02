from __future__ import annotations
from ibaml.features.engineering import apply_minmax_scaler
from ibaml.utils.expressions import iter_nonempty_masks
from ibaml.utils.expressions import eval_mask_over_folds_logs
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..validation.expanding_cv import ExpandingWindowSplit, adjust_split_ratios
from ..iba.polynomials import precompute_log_base_ratio


import logging
logger = logging.getLogger(__name__)


def single_factor_search(
    target,
    y_train,
    group_frames,
    M: int,
    Q: int,
    N: int,
    K: int,
    xgb_params,
    paper_params,
    top_k=1,
    limit_mask_size=None,
    n_jobs_groups=1,
    n_jobs_masks=1,
    progress=True,
    ):
    """Perform single-factor feature search with expanding window CV.
    Evaluates all non-empty feature subsets (masks) for each factor group using an XGBoost model
    trained and validated on expanding window splits. Returns top-K masks per group sorted by mean loss.
    Speeds up mask evaluations using joblib parallelism."""
    
    T = len(y_train)
    splitter = ExpandingWindowSplit(M=M, Q=Q, N=N, K=K, T=T)
    fold_indices = list(iter(splitter))
    # Anchor last validation window to end-of-training if needed
    if len(fold_indices) > 0:
        last_tr, last_va = fold_indices[-1]
        if len(last_va) > 0 and last_va[-1] != T - 1:
            va_start = max(0, T - Q)
            va_idx = np.arange(va_start, T, dtype=int)
            tr_end = va_start
            tr_start = max(0, tr_end - M)
            tr_idx = np.arange(tr_start, tr_end, dtype=int)
            # Append anchored fold if different from last
            if not (np.array_equal(last_tr, tr_idx) and np.array_equal(last_va, va_idx)):
                fold_indices.append((tr_idx, va_idx))

    items = list(group_frames.items())
    group_iter = tqdm(items, desc=f"SF[{target}] groups", disable=not progress) if n_jobs_groups == 1 else items

    def _run_group(item):
        """Run single-factor search for a single group."""
        g, Xg = item
        per_fold_scaled = []
        for (tr, _) in fold_indices:
            per_fold_scaled.append(apply_minmax_scaler(Xg, tr))
        per_fold_logs = [precompute_log_base_ratio(Xs.values) for Xs in per_fold_scaled]
        G = Xg.shape[1]
        masks = list(iter_nonempty_masks(G, limit_mask_size))
        mask_iter = tqdm(masks, desc=f"SF[{target}:{g}] masks", leave=False, disable=not progress) if n_jobs_masks == 1 else masks
        evals = []
        if n_jobs_masks and n_jobs_masks != 1:
            evals = Parallel(n_jobs=n_jobs_masks, backend="loky", prefer="processes")(delayed(eval_mask_over_folds_logs)(
                y_train, fold_indices, per_fold_logs,
                selected_mask=np.eye(G, dtype=bool)[list(mask)].any(axis=0),
                xgb_params=xgb_params, paper_params=paper_params
            ) for mask in masks)
        else:
            for mask in mask_iter:
                evals.append(eval_mask_over_folds_logs(
                    y_train, fold_indices, per_fold_logs,
                    selected_mask=np.eye(G, dtype=bool)[list(mask)].any(axis=0),
                    xgb_params=xgb_params, paper_params=paper_params
                ))
        rows = []
        for mask, (m_loss, m_rmse, m_sacc, per) in zip(masks, evals):
            rows.append({
                "columns": list(Xg.columns[list(mask)]),
                "mask_indices": list(mask),
                "mean_loss": m_loss,
                "mean_rmse": m_rmse,
                "mean_sign_acc": m_sacc,
                "folds": len(fold_indices),
                "per_fold": per,
            })
        rows.sort(key=lambda r: (r["mean_loss"], -r["mean_sign_acc"]))
        return g, rows[:top_k]

    if n_jobs_groups and n_jobs_groups != 1:
        res = Parallel(n_jobs=n_jobs_groups, backend="loky", prefer="processes")(delayed(_run_group)(it) for it in items)
    else:
        res = []
        for it in group_iter:
            res.append(_run_group(it))
    # Prepare fold indices info for artifact
    fold_indices_info = []
    for i, (tr_idx, va_idx) in enumerate(fold_indices):
        fold_info = {"fold": i}
        if len(tr_idx) > 0:
            fold_info["train_range"] = (int(tr_idx[0]), int(tr_idx[-1]))
        else:
            fold_info["train_range"] = None
        if len(va_idx) > 0:
            fold_info["valid_range"] = (int(va_idx[0]), int(va_idx[-1]))
        else:
            fold_info["valid_range"] = None
        fold_indices_info.append(fold_info)

    return {"results": dict(res), "cv_fold_indices": fold_indices_info}

