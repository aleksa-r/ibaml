from __future__ import annotations
from typing import Dict, List, Tuple
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import xgboost as xgb
from tqdm.auto import tqdm

from ..validation.expanding_cv import ExpandingWindowSplit
from ..iba.polynomials import precompute_log_base_ratio, gbp_from_mask_logs
from ..losses.objective import ObjectiveFunctionParameters, make_xgb_objective, total_loss
from ..eval.metrics import rmse, sign_accuracy
from ..validation.expanding_cv import adjust_split_ratios
from ..features.engineering import apply_minmax_scaler
from ..utils.core_predict import xgb_core_predict


def multifactor_search(
    target: str,
    y_train: np.ndarray,
    group_frames: Dict[str, pd.DataFrame],
    per_group_topk: Dict[str, List[Dict]],
    M: int,
    Q: int,
    N: int,
    K: int,
    xgb_params: Dict,
    paper_params: ObjectiveFunctionParameters,
    allow_empty_groups=True,
    min_groups_in_combo=1,
    max_groups_in_combo=None,
    n_jobs_combos=1,
    progress=True):
    T = len(y_train)
    """Perform multifactor feature combination search with expanding window CV.
    Performs an exhaustive search over all combinations of top-K features per group. Combinations
    are evaluated using an XGBoost model trained and validated on expanding window splits. 
    Combinations can be constrained by minimum and maximum number of groups included.
    """
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
            if not (np.array_equal(last_tr, tr_idx) and np.array_equal(last_va, va_idx)):
                fold_indices.append((tr_idx, va_idx))

    per_group_logs = {g: [] for g in group_frames}
    for g, Xg in group_frames.items():
        per_fold_scaled = [apply_minmax_scaler(Xg, tr) for (tr, _) in fold_indices]
        per_group_logs[g] = [precompute_log_base_ratio(Xs.values) for Xs in per_fold_scaled]

    gbp_cache = {}
    for g, rows in per_group_topk.items():
        mats = []
        Xg_cols = list(group_frames[g].columns)
        for item in rows:
            sel = np.array([c in set(item["columns"]) for c in Xg_cols], dtype=bool)
            folds_gbp = [gbp_from_mask_logs(log_base, log_r, sel) for (log_base, log_r) in per_group_logs[g]]
            mats.append(folds_gbp)
        gbp_cache[g] = mats

    groups_order = list(per_group_topk.keys())
    choices = []
    for g in groups_order:
        idxs = list(range(len(per_group_topk[g])))
        if allow_empty_groups:
            idxs = [-1] + idxs
        choices.append(idxs)
    raw_product = list(itertools.product(*choices))
    Gtot = len(groups_order)
    max_ok = Gtot if (max_groups_in_combo is None or max_groups_in_combo > Gtot) else max_groups_in_combo

    combos = []
    for c in raw_product:
        used = sum(1 for ix in c if ix != -1)
        if used == 0: continue
        if used < min_groups_in_combo: continue
        if used > max_ok: continue
        combos.append(c)

    combo_iter = tqdm(combos, desc=f"MF[{target}] combos", disable=not progress) if n_jobs_combos == 1 else combos

    def _eval_combo(combo):
        losses, rmses, saccs = [], [], []
        obj = make_xgb_objective(paper_params)
        for fold_pos, (tr, va) in enumerate(fold_indices):
            cols = []
            for gpos, idx in enumerate(combo):
                if idx == -1: continue
                gbp_vec = gbp_cache[groups_order[gpos]][idx][fold_pos]
                cols.append(gbp_vec.reshape(-1, 1).astype("float32", copy=False))
            X_full = np.hstack(cols) if cols else np.zeros((len(y_train), 0), dtype="float32")
            Xtr, Xva = X_full[tr], X_full[va]
            ytr, yva = y_train[tr].astype("float32", copy=False), y_train[va].astype("float32", copy=False)
            yhat = xgb_core_predict(Xtr, ytr, Xva, obj_fn=obj, xgb_params={"n_jobs": 1, **xgb_params})
            losses.append(total_loss(yva, yhat, paper_params))
            rmses.append(rmse(yva, yhat))
            saccs.append(sign_accuracy(yva, yhat))
        return float(np.mean(losses)), float(np.mean(rmses)), float(np.mean(saccs)), {"losses": losses, "rmses": rmses, "signacc": saccs}

    if n_jobs_combos and n_jobs_combos != 1:
        evals = Parallel(n_jobs=n_jobs_combos, backend="loky", prefer="processes")(delayed(_eval_combo)(c) for c in combos)
    else:
        evals = []
        for c in combo_iter:
            evals.append(_eval_combo(c))

    leaderboard = []
    for combo, (m_loss, m_rmse, m_sacc, per) in zip(combos, evals):
        used = [(groups_order[i], idx) for i, idx in enumerate(combo) if idx != -1]
        groups_used = [g for g, _ in used]
        columns_used = [per_group_topk[g][idx]["columns"] for g, idx in used]
        masks_used = [per_group_topk[g][idx]["mask_indices"] for g, idx in used]
        leaderboard.append({
            "groups": groups_used,
            "columns": columns_used,
            "masks": masks_used,
            "mean_loss": m_loss, "mean_rmse": m_rmse, "mean_sign_acc": m_sacc,
            "folds": len(fold_indices),
            "per_fold": per,
        })
    leaderboard.sort(key=lambda r: (r["mean_loss"], -r["mean_sign_acc"]))
    best = leaderboard[0] if leaderboard else None
    
    # Extract feature importances from best model for visualization
    if best:
        try:
            combo = combos[0]  # Best combo is first in sorted leaderboard
            cols = []
            col_names = []
            for gpos, idx in enumerate(combo):
                if idx == -1: continue
                g = groups_order[gpos]
                col_info = per_group_topk[g][idx]
                col_names.extend(col_info["columns"])
                for fold_pos, (tr, va) in enumerate(fold_indices):
                    gbp_vec = gbp_cache[g][idx][fold_pos]
                    cols.append(gbp_vec.reshape(-1, 1).astype("float32", copy=False))
                    break  # Just use first fold for importance extraction
            
            if cols:
                X_best = np.hstack(cols).astype("float32", copy=False)
                y_best = y_train.astype("float32", copy=False)
                dtr_best = xgb.DMatrix(X_best, label=y_best)
                obj = make_xgb_objective(paper_params)
                booster_best = xgb.train(
                    {**{"verbosity": 0}, **(xgb_params or {})},
                    dtr_best, num_boost_round=500, obj=obj
                )
                importance_dict = booster_best.get_score(importance_type='gain')
                best["feature_importances"] = importance_dict if importance_dict else {}
            else:
                best["feature_importances"] = {}
        except Exception as e:
            best["feature_importances"] = {}
    
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

    return best, leaderboard, fold_indices_info

