from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config.schemas import Config
from ..data.prep import compute_holdout_indices
from ..features.engineering import build_endogenous_features, apply_minmax_scaler
from ..losses.objective import (
    ObjectiveFunctionParameters, make_xgb_objective, initialize_objective_params,
    total_loss
)
from ..eval.metrics import rmse, sign_accuracy, sharpe_ratio, max_drawdown
from .sf_search import single_factor_search, adjust_split_ratios
from .mf_search import multifactor_search

logger = logging.getLogger(__name__)


@dataclass
class TargetResult:
    """Result container for a single target's pipeline run.
    
    Attributes:
        target: Name of the target column
        skipped: Whether processing was skipped
        skip_reason: Reason for skipping (if skipped)
        single_factor: Single-factor search results
        leaderboard: Multi-factor leaderboard
        best_combo: Best multi-factor combination
        feature_importance: XGBoost feature importance (gain-based)
        shap_values: SHAP values for feature attribution
        shap_feature_names: Feature names for SHAP
        shap_X: Feature matrix used for SHAP
        train_size: Number of training samples
        holdout_size: Number of holdout samples
        delta: δ threshold for objective function
        gamma: γ threshold for objective function
        delta_gamma_source: Source of δ/γ values
        splits_requested: Requested CV splits
        splits_actual: Actual CV splits (after clamping)
        holdout: Holdout evaluation results
        hyperopt: Hyperparameter optimization results
    """
    target: str
    skipped: bool = False
    skip_reason: Optional[str] = None
    single_factor: Optional[Dict] = None
    leaderboard: Optional[List] = None
    best_combo: Optional[Dict] = None
    cv_fold_indices_single_factor: Optional[List[Dict[str, Any]]] = None
    cv_fold_indices_multifactor: Optional[List[Dict[str, Any]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    shap_values: Optional[List] = None
    shap_feature_names: Optional[List[str]] = None
    shap_X: Optional[List] = None
    train_size: int = 0
    holdout_size: int = 0
    delta: float = 0.0
    gamma: float = 0.0
    delta_gamma_source: str = "unknown"
    splits_requested: Dict[str, Any] = field(default_factory=dict)
    splits_actual: Dict[str, Any] = field(default_factory=dict)
    holdout: Optional[Dict] = None
    hyperopt: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        if self.skipped:
            return {
                "target": self.target,
                "target": self.target,  # Backward compatibility
                "skipped": True,
                "reason": self.skip_reason,
                "single_factor": self.single_factor,
                "leaderboard": self.leaderboard,
            }
        
        result = {
            "target": self.target,
            "target": self.target,  # Backward compatibility
            "single_factor": self.single_factor,
            "leaderboard": self.leaderboard,
            "best_combo": self.best_combo,
            "feature_importance": self.feature_importance,
            "shap_values": self.shap_values,
            "shap_feature_names": self.shap_feature_names,
            "shap_X": self.shap_X,
            "train_size": self.train_size,
            "holdout_size": self.holdout_size,
            "delta": self.delta,
            "gamma": self.gamma,
            "delta_gamma_source": self.delta_gamma_source,
            "splits_requested": self.splits_requested,
            "splits_actual": self.splits_actual,
            "holdout": self.holdout,
            "hyperopt": self.hyperopt,
        }
        if self.cv_fold_indices_single_factor is not None:
            result["cv_fold_indices_single_factor"] = self.cv_fold_indices_single_factor
        if self.cv_fold_indices_multifactor is not None:
            result["cv_fold_indices_multifactor"] = self.cv_fold_indices_multifactor
        return result


def run_target_pipeline(
    target: str,
    y_all: pd.Series,
    df_targets: pd.DataFrame,
    group_frames_all: Dict[str, pd.DataFrame],
    cfg: Config,
    xgb_params: Dict[str, Any],
    num_boost_round: int,
    quantile_delta: float,
    quantile_gamma: float,
    config_dict: Dict[str, Any],
    outdir: str = "artifacts",
    models_subdir: str = "models",
    top_k: int = 1,
    limit_mask_size: Optional[int] = None,
    allow_empty_groups: bool = True,
    min_groups_in_combo: int = 1,
    max_groups_in_combo: Optional[int] = None,
    n_jobs_groups: int = 1,
    n_jobs_masks: int = 1,
    n_jobs_combos: int = 1,
) -> TargetResult:
    """Run complete pipeline for a single target.
    
    Args:
        target: Name of the target column
        y_all: Full target series
        df_targets: Full targets DataFrame (for endogenous features)
        group_frames_all: Pre-built factor group DataFrames
        cfg: Configuration object
        xgb_params: XGBoost parameters
        num_boost_round: Number of boosting rounds
        quantile_delta: Quantile for δ threshold
        quantile_gamma: Quantile for γ threshold
        config_dict: Configuration as dictionary
        outdir: Output directory
        models_subdir: Subdirectory for models
        top_k: Number of top single-factor models
        limit_mask_size: Maximum mask size
        allow_empty_groups: Allow empty groups in combos
        min_groups_in_combo: Minimum groups in combo
        max_groups_in_combo: Maximum groups in combo
        n_jobs_groups: Parallel jobs for groups
        n_jobs_masks: Parallel jobs for masks
        n_jobs_combos: Parallel jobs for combos
        
    Returns:
        TargetResult containing all pipeline outputs
    """
    # Build endogenous features for this target
    endg = build_endogenous_features(df_targets, target, cfg.dataset.zscore_windows)
    group_frames = {"ENDG": endg, **group_frames_all}
    
    # Clean dataset of NaNs
    mask_valid = np.isfinite(y_all.values)
    if endg.shape[1] > 0:
        mask_valid &= np.isfinite(endg.values).all(axis=1)
    idx_clean = y_all.index[mask_valid]
    
    # Compute holdout indices
    eval_size = int(getattr(cfg.evaluation, 'size', 18))
    y_clean = y_all.loc[idx_clean]
    tr_idx, ho_idx_eff = compute_holdout_indices(
        y_clean,
        eval_size=eval_size,
        require_consecutive=True,
        fallback_to_recent_valid=True,
    )
    
    if len(ho_idx_eff) < eval_size:
        logger.warning(
            f"[{target}] Holdout has {len(ho_idx_eff)} obs (requested {eval_size}). "
            "Check data gaps in the last window."
        )
    
    if len(ho_idx_eff) == 0:
        return TargetResult(target=target, skipped=True, skip_reason="no valid holdout window")
    
    logger.info(
        f"[{target}] Train period: {str(tr_idx[0].date())}..{str(tr_idx[-1].date())} ({len(tr_idx)} obs)"
    )
    logger.info(
        f"[{target}] Holdout period: {str(ho_idx_eff[0].date())}..{str(ho_idx_eff[-1].date())} ({len(ho_idx_eff)} obs)"
    )
    
    y_train = y_all.loc[tr_idx].dropna().values
    if len(y_train) == 0:
        return TargetResult(target=target, skipped=True, skip_reason="no training labels")
    
    # Compute CV splits dynamically to ensure final fold validation ends at holdout start
    # This guarantees the most recent pre-holdout data is used for final validation
    M, Q, N, K = adjust_split_ratios(
        len(y_train),
        cfg.splits.init_tran_size,
        cfg.splits.folds_val_set_size,
        cfg.splits.folds_steps_size,
        cfg.splits.num_of_folds,
    )
    
    # Verify final fold ends at training data end
    final_val_end = M + (K - 1) * N + Q
    logger.info(
        f"[{target}] Dynamic CV: M={M}, N={N}, Q={Q}, K={K} | "
        f"T_cv={len(y_train)}, final_val_end={final_val_end}"
    )
    
    splits_requested = {
        "init_tran_size": float(cfg.splits.init_tran_size),
        "folds_val_set_size": int(cfg.splits.folds_val_set_size),
        "folds_steps_size": int(cfg.splits.folds_steps_size),
        "num_of_folds": int(cfg.splits.num_of_folds),
    }
    splits_actual = {"M": int(M), "Q": int(Q), "N": int(N), "K": int(K)}
    
    # Create target-specific delta/gamma
    paper_params = initialize_objective_params(
        y_train=y_train,
        config=config_dict,
        target=target,
        lambda1=cfg.objective.lambda1,
        lambda2=cfg.objective.lambda2,
        lambda3=cfg.objective.lambda3,
        quantile_delta=quantile_delta,
        quantile_gamma=quantile_gamma,
    )
    
    logger.info(
        f"[{target}] Using δ={paper_params.delta:.6f}, γ={paper_params.gamma:.6f} "
        f"(quantiles: {paper_params.computed_from_quantiles})"
    )
    
    # Single-factor search
    sf_result = single_factor_search(
        target, y_train, group_frames, M, Q, N, K,
        xgb_params, paper_params,
        top_k=top_k,
        limit_mask_size=limit_mask_size,
        n_jobs_groups=n_jobs_groups,
        n_jobs_masks=n_jobs_masks,
        progress=(n_jobs_groups == 1 and n_jobs_masks == 1),
    )
    sf_topk = sf_result["results"]
    cv_fold_indices_single_factor = sf_result["cv_fold_indices"]

    # Multi-factor search
    best, leaderboard, cv_fold_indices_multifactor = multifactor_search(
        target, y_train, group_frames, sf_topk, M, Q, N, K,
        xgb_params, paper_params,
        allow_empty_groups=allow_empty_groups,
        min_groups_in_combo=min_groups_in_combo,
        max_groups_in_combo=max_groups_in_combo,
        n_jobs_combos=n_jobs_combos,
        progress=(n_jobs_combos == 1),
    )
    
    if not best or not best.get("groups"):
        return TargetResult(
            target=target,
            skipped=True,
            skip_reason="no valid multifactor combo",
            single_factor=sf_topk,
            leaderboard=leaderboard,
        )
    
    # Build feature matrix from best combo
    X_cols, X_mat = [], []
    for g, cols in zip(best.get("groups", []), best.get("columns", [])):
        if not cols:
            continue
        Xg = group_frames[g]
        # Use apply_minmax_scaler (fits on training indices) to avoid leakage
        # Need positional train indices relative to Xg
        tr_pos = np.where(np.isin(Xg.index.values, tr_idx.values))[0]
        Xs_df = apply_minmax_scaler(Xg[cols], tr_pos)
        Xs = Xs_df.values
        X_mat.append(np.prod(Xs, axis=1, dtype=np.float32).reshape(-1, 1))
        X_cols.append(f"{g}_gbp")
    
    X_full = np.hstack(X_mat).astype("float32") if X_mat else np.zeros((len(y_all), 0), dtype="float32")
    
    # Recompute holdout based on feature validity
    mask_x_ok = np.isfinite(X_full).all(axis=1) if X_full.shape[1] > 0 else np.ones(len(y_all), dtype=bool)
    mask_y_ok = np.isfinite(y_all.values)
    idx_valid_for_combo = y_all.index[mask_x_ok & mask_y_ok]
    y_valid_for_combo = y_all.loc[idx_valid_for_combo]
    
    tr_idx, ho_idx_eff = compute_holdout_indices(
        y_valid_for_combo,
        eval_size=eval_size,
        require_consecutive=True,
        fallback_to_recent_valid=True,
    )
    
    if len(ho_idx_eff) < eval_size:
        logger.warning(
            f"[{target}] Effective holdout has {len(ho_idx_eff)} obs after feature filtering."
        )
    
    if len(ho_idx_eff) == 0:
        return TargetResult(
            target=target,
            skipped=True,
            skip_reason="no valid holdout window after combo feature filtering",
        )
    
    # Extract training and holdout data
    tr_pos = np.where(np.isin(y_all.index.values, tr_idx.values))[0]
    ho_pos = np.where(np.isin(y_all.index.values, ho_idx_eff.values))[0]
    Xtr, Xho = X_full[tr_pos], X_full[ho_pos]
    ytr = y_all.iloc[tr_pos].values.astype("float32")
    yho = y_all.iloc[ho_pos].values.astype("float32")
    
    # Filter NaN from training data
    valid_tr_mask = np.isfinite(ytr) & np.isfinite(Xtr).all(axis=1)
    if not np.all(valid_tr_mask):
        Xtr = Xtr[valid_tr_mask]
        ytr = ytr[valid_tr_mask]
    
    if len(ytr) == 0:
        return TargetResult(
            target=target,
            skipped=True,
            skip_reason="no valid training labels after NaN filtering",
        )
    
    # Hyperparameter optimization (optional)
    hyperopt_result = None
    effective_xgb_params = xgb_params.copy()
    effective_num_boost_round = num_boost_round
    
    if hasattr(cfg, 'hyperopt') and cfg.hyperopt.enabled:
        def _paper_params_factory(y_tr_arr: np.ndarray):
            return initialize_objective_params(
                y_train=y_tr_arr,
                config=config_dict,
                target=target,
                lambda1=cfg.objective.lambda1,
                lambda2=cfg.objective.lambda2,
                lambda3=cfg.objective.lambda3,
                quantile_delta=cfg.objective.quantile_delta if hasattr(cfg.objective, 'quantile_delta') else None,
                quantile_gamma=cfg.objective.quantile_gamma if hasattr(cfg.objective, 'quantile_gamma') else None,
            )

        hyperopt_result = _run_hyperopt(
            target, Xtr, ytr, cfg, xgb_params, _paper_params_factory, X_cols
        )
        if hyperopt_result:
            for k, v in hyperopt_result.get("best_params", {}).items():
                if k == "n_estimators":
                    effective_num_boost_round = int(v)
                else:
                    effective_xgb_params[k] = v
    
    # Train final model and predict
    obj = make_xgb_objective(paper_params)
    eval_cfg = getattr(cfg, 'evaluation', None)
    do_retrain = bool(eval_cfg and getattr(eval_cfg, 'retrain', False))
    retrain_steps = int(getattr(eval_cfg, 'steps', 0)) if eval_cfg else 0
    
    booster, yhat_ho = _train_and_predict(
        Xtr, ytr, Xho, yho, X_cols,
        effective_xgb_params, effective_num_boost_round, obj,
        do_retrain, retrain_steps
    )
    
    # Filter holdout by validity
    valid_ho_mask = np.isfinite(yho) & np.isfinite(Xho).all(axis=1)
    yho = yho[valid_ho_mask]
    yhat_ho = yhat_ho[valid_ho_mask]
    ho_idx_filtered = ho_idx_eff[valid_ho_mask]
    
    # Compute metrics
    realized, predicted = yho, yhat_ho
    gate_kind = getattr(cfg, 'simulation', None).gate if getattr(cfg, 'simulation', None) else 'zero'
    thr = float(paper_params.delta) if gate_kind == 'delta' else 0.0
    signal = (predicted >= thr).astype("float32")
    simulated_returns = signal * realized
    
    realized_cum = np.cumprod(1.0 + realized) - 1.0
    predicted_cum = np.cumprod(1.0 + predicted) - 1.0
    simulated = np.cumprod(1.0 + simulated_returns) - 1.0
    
    # Handle NaN filtering in holdout
    ho_valid = np.isfinite(realized) & np.isfinite(predicted)
    if not np.all(ho_valid):
        n_drop = int(np.sum(~ho_valid))
        logger.warning(f"[{target}] Dropping {n_drop} invalid holdout rows")
        ho_idx_filtered = ho_idx_filtered[ho_valid]
        realized = realized[ho_valid]
        predicted = predicted[ho_valid]
        simulated_returns = signal[ho_valid] * realized
        realized_cum = np.cumprod(1.0 + realized) - 1.0
        predicted_cum = np.cumprod(1.0 + predicted) - 1.0
        simulated = np.cumprod(1.0 + simulated_returns) - 1.0
    
    holdout_loss = total_loss(realized, predicted, paper_params)
    metrics = {
        "rmse": float(rmse(realized, predicted)),
        "sign_acc": float(sign_accuracy(realized, predicted)),
        "sharpe_realized": float(sharpe_ratio(realized)),
        "mdd_realized": float(max_drawdown(realized_cum)),
        "sharpe_pred": float(sharpe_ratio(predicted)),
        "mdd_pred": float(max_drawdown(predicted_cum)),
        "sharpe_sim": float(sharpe_ratio(simulated_returns) if len(simulated_returns) > 0 else 0.0),
        "mdd_sim": float(max_drawdown(simulated)),
        "holdout_loss": float(holdout_loss),
    }
    
    # Feature importance
    fi_raw = booster.get_score(importance_type='gain')
    fi_map = {}
    for k, v in fi_raw.items():
        name = k
        if k.startswith("f") and X_cols:
            try:
                idx = int(k[1:])
                name = X_cols[idx]
            except (ValueError, IndexError):
                pass
        fi_map[name] = float(v)
    
    # SHAP values
    shap_values, shap_X = _compute_shap(target, booster, Xtr, X_cols)
    
    # Save model
    mdir = os.path.join(outdir, models_subdir)
    os.makedirs(mdir, exist_ok=True)
    booster.save_model(os.path.join(mdir, f"{target}_best.json"))
    
    return TargetResult(
        target=target,
        single_factor=sf_topk,
        leaderboard=leaderboard,
        best_combo=best,
        feature_importance=fi_map,
        shap_values=shap_values.tolist() if shap_values is not None else None,
        shap_feature_names=X_cols,
        shap_X=shap_X.tolist() if shap_X is not None else None,
        train_size=len(tr_idx),
        holdout_size=len(ho_idx_filtered),
        delta=float(paper_params.delta),
        gamma=float(paper_params.gamma),
        delta_gamma_source="quantiles" if paper_params.computed_from_quantiles else "default",
        splits_requested=splits_requested,
        splits_actual=splits_actual,
        holdout={
            "index": [str(x) for x in ho_idx_filtered],
            "realized": realized.tolist(),
            "predicted": predicted.tolist(),
            "simulated": simulated.tolist(),
            "metrics": metrics,
            "evaluation": {"retrain": do_retrain, "steps": int(retrain_steps)},
        },
        hyperopt=hyperopt_result,
        cv_fold_indices_single_factor=cv_fold_indices_single_factor,
        cv_fold_indices_multifactor=cv_fold_indices_multifactor,
    )


def _run_hyperopt(
    target: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    cfg: Config,
    xgb_params: Dict,
    paper_params_factory: Optional[Callable[[np.ndarray], ObjectiveFunctionParameters]],
    feature_names: List[str],
) -> Optional[Dict]:
    """Run hyperparameter optimization if enabled."""
    try:
        from ..training.hyperopt import (
            optimize_xgb_params as run_hyperopt,
            create_time_series_cv_splits,
        )

        logger.info(f"[{target}] Running hyperopt ({cfg.hyperopt.n_trials} trials)")

        # Convert search space
        search_space = {}
        for k, v in cfg.hyperopt.search_space.items():
            if hasattr(v, 'model_dump'):
                search_space[k] = v.model_dump()
            else:
                search_space[k] = dict(v) if hasattr(v, '__iter__') else v

        # Precompute per-fold scaled feature matrices and serve them from a cache
        fold_indices = create_time_series_cv_splits(
            n_samples=len(ytr),
            n_splits=cfg.hyperopt.cv_folds,
            min_train_ratio=getattr(cfg.hyperopt, 'min_train_ratio', 0.5),
        )

        fold_cache = {}
        # Xtr is a numpy array of precomputed IBA features (already scaled when built
        # via `apply_minmax_scaler` during feature construction). For hyperopt we
        # use identity preprocessing (no additional per-fold scaling) to avoid
        # altering IBA feature magnitudes; this preserves CV integrity without
        # unnecessary transforms.
        for tr_idx, va_idx in fold_indices:
            if getattr(cfg.hyperopt, 'scale_iba_features', True):
                df = pd.DataFrame(Xtr)
                Xs_df = apply_minmax_scaler(df, tr_idx)
                X_tr = Xs_df.iloc[tr_idx].values.astype("float32", copy=False)
                X_va = Xs_df.iloc[va_idx].values.astype("float32", copy=False)
            else:
                X_tr = Xtr[tr_idx].astype("float32", copy=False)
                X_va = Xtr[va_idx].astype("float32", copy=False)
            fold_cache[(tuple(tr_idx), tuple(va_idx))] = (X_tr, X_va, None)

        def _cached_scaler_preprocess(X: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray):
            return fold_cache.get((tuple(train_idx), tuple(val_idx)), (None, None, None))

        result = run_hyperopt(
            X_train=Xtr,
            y_train=ytr,
            cv_splitter=fold_indices,
            cv_folds=cfg.hyperopt.cv_folds,
            min_train_ratio=getattr(cfg.hyperopt, 'min_train_ratio', 0.5),
            objective_fn=total_loss,
            paper_params=None,
            paper_params_factory=paper_params_factory,
            make_obj_fn=make_xgb_objective,
            preprocess_fn=_cached_scaler_preprocess,
            search_space=search_space,
            n_trials=cfg.hyperopt.n_trials,
            timeout=cfg.hyperopt.timeout,
            n_jobs=cfg.hyperopt.n_jobs,
            sampler=cfg.hyperopt.sampler,
            pruner=cfg.hyperopt.pruner,
            base_params=xgb_params,
            feature_names=feature_names,
        )
        
        logger.info(f"[{target}] Hyperopt complete: best_loss={result.best_value:.6f}")
        
        return {
            "enabled": True,
            "best_params": result.best_params,
            "best_value": float(result.best_value),
            "n_trials": result.n_trials,
        }
        
    except Exception as e:
        logger.warning(f"[{target}] Hyperopt failed: {e}. Using default params.")
        return None


def _train_and_predict(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xho: np.ndarray,
    yho: np.ndarray,
    feature_names: List[str],
    xgb_params: Dict,
    num_boost_round: int,
    obj,
    do_retrain: bool,
    retrain_steps: int,
) -> Tuple[xgb.Booster, np.ndarray]:
    """Train XGBoost model and make holdout predictions."""
    # When using custom objective, XGBoost defaults base_score=0.5 which is far
    # from typical return values (~0.01). This causes tree building to fail
    # because all splits appear unprofitable. Set base_score=0 to fix this.
    train_params = {**xgb_params}
    if obj is not None and 'base_score' not in train_params:
        train_params['base_score'] = 0.0
    
    if not do_retrain:
        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_names if feature_names else None)
        dho = xgb.DMatrix(Xho, feature_names=feature_names if feature_names else None)
        booster = xgb.train(train_params, dtr, num_boost_round=num_boost_round, obj=obj)
        yhat_ho = booster.predict(dho).astype("float32")
    else:
        # Rolling retrain
        yhat_list = []
        booster = None
        for i in range(len(Xho)):
            need_train = (booster is None) or (retrain_steps == 0) or (i % retrain_steps == 0)
            if need_train:
                if i > 0:
                    Xtr_cat = np.vstack([Xtr, Xho[:i]])
                    ytr_cat = np.concatenate([ytr, yho[:i]])
                else:
                    Xtr_cat, ytr_cat = Xtr, ytr
                
                valid = np.isfinite(ytr_cat)
                Xtr_exp = Xtr_cat[valid] if not np.all(valid) else Xtr_cat
                ytr_exp = ytr_cat[valid] if not np.all(valid) else ytr_cat
                
                dtr = xgb.DMatrix(Xtr_exp, label=ytr_exp, feature_names=feature_names if feature_names else None)
                booster = xgb.train(train_params, dtr, num_boost_round=num_boost_round, obj=obj)
            
            dcur = xgb.DMatrix(Xho[i:i+1], feature_names=feature_names if feature_names else None)
            yhat_i = booster.predict(dcur).astype("float32")
            yhat_list.append(float(yhat_i[0]))
        
        yhat_ho = np.array(yhat_list, dtype="float32")
    
    return booster, yhat_ho


def _compute_shap(
    target: str,
    booster: xgb.Booster,
    Xtr: np.ndarray,
    feature_names: List[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute SHAP values for feature importance."""
    try:
        import shap
        explainer = shap.TreeExplainer(booster)
        X_shap = Xtr if len(Xtr) <= 500 else Xtr[np.random.choice(len(Xtr), 500, replace=False)]
        shap_values = explainer.shap_values(X_shap)
        logger.debug(f"SHAP computed for {target}: shape={shap_values.shape}")
        return shap_values, X_shap
    except ImportError:
        logger.warning("SHAP library not installed - skipping SHAP analysis")
        return None, None
    except Exception as e:
        logger.warning(f"SHAP computation failed for {target}: {e}")
        return None, None
