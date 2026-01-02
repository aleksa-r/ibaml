from __future__ import annotations
import os
import json
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import xgboost as xgb

from ..config.schemas import load_config, Config
from ..data.prep import (
    read_raw_pair, cumulative_shifted_returns,
    normalize_to_month_end, apply_date_filters, compute_holdout_indices
)
from ..features.engineering import build_endogenous_features, build_exogenous_features
from ..losses.objective import (
    ObjectiveFunctionParameters, make_xgb_objective, initialize_objective_params,
    total_loss, loss_components, get_quantile_params_from_config
)
from ..eval.metrics import rmse, sign_accuracy, sharpe_ratio, max_drawdown
from ..validation.expanding_cv import ExpandingWindowSplit, adjust_split_ratios
from ..utils.core_predict import xgb_core_predict
from ..reporting.figures import save_holdout_figure
from ..training.hyperopt import (
    create_time_series_cv_splits,
    optimize_xgb_params,
    create_hyperopt_report,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


@dataclass
class PCABenchmarkResult:
    """Result container for PCA benchmark pipeline run.
    """
    target: str
    skipped: bool = False
    skip_reason: Optional[str] = None
    n_components: int = 6
    explained_variance_ratio: Optional[List[float]] = None
    train_size: int = 0
    holdout_size: int = 0
    delta: float = 0.0
    gamma: float = 0.0
    delta_gamma_source: str = "unknown"
    cv_loss: float = 0.0
    cv_metrics: Optional[Dict] = None
    holdout: Optional[Dict] = None
    feature_names: Optional[List[str]] = None
    hyperopt: Optional[Dict] = None
    cv_fold_indices: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        if self.skipped:
            return {
                "target": self.target,
                "target": self.target,
                "skipped": True,
                "reason": self.skip_reason,
            }
        result = {
            "target": self.target,
            "target": self.target,
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio,
            "train_size": self.train_size,
            "holdout_size": self.holdout_size,
            "delta": self.delta,
            "gamma": self.gamma,
            "delta_gamma_source": self.delta_gamma_source,
            "cv_loss": self.cv_loss,
            "cv_metrics": self.cv_metrics,
            "holdout": self.holdout,
            "feature_names": self.feature_names,
            "hyperopt": self.hyperopt,
        }
        if self.cv_fold_indices is not None:
            result["cv_fold_indices"] = self.cv_fold_indices
        return result


def apply_pca_transform(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 6,
) -> Tuple[np.ndarray, np.ndarray, PCA, np.ndarray]:
    """Apply PCA transformation to features.
    
    Fits PCA on training data only, then transforms both train and test.
    Uses standard PCA (equivalent to KernelPCA with linear kernel).
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        n_components: Number of PCA components to retain       
    Returns:
        Tuple of (X_train_pca, X_test_pca, pca_model, explained_variance_ratio)
    """
    # Adjust n_components if needed
    n_components_eff = min(n_components, X_train.shape[1], X_train.shape[0])
    
    pca = PCA(n_components=n_components_eff)
    
    # Fit on training data only
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    return (
        X_train_pca.astype(np.float32),
        X_test_pca.astype(np.float32),
        pca,
        pca.explained_variance_ratio_,
    )


def evaluate_pca_cv(
    X: np.ndarray,
    y: np.ndarray,
    M: int,
    Q: int,
    N: int,
    K: int,
    xgb_params: Dict,
    custom_objective_params: ObjectiveFunctionParameters,
    n_components: int = 6,
) -> Tuple[float, Dict, List[float]]:
    """Evaluate PCA model using expanding window CV.
    
    For each fold:
    1. Fit PCA on training portion only
    2. Transform both train and validation
    3. Train XGBoost with custom objective
    4. Evaluate on validation set
    
    Args:
        X: Full feature matrix (before PCA)
        y: Target array
        splits: (M, Q, N, K) CV split parameters
        xgb_params: XGBoost parameters
        paper_params: Custom objective parameters
        n_components: Number of PCA components
        
    Returns:
        Tuple of (mean_loss, metrics_dict, explained_variance_ratios)
    """
    T = len(y)
    splitter = ExpandingWindowSplit(M=M, Q=Q, N=N, K=K, T=T)
    fold_indices = list(iter(splitter))

    # Collect fold indices for artifact
    fold_indices_info = []

    losses, rmses, saccs = [], [], []
    all_var_ratios = []
    obj_fn = make_xgb_objective(custom_objective_params)

    for i, (tr_idx, va_idx) in enumerate(fold_indices):
        # Save indices for artifact (as lists for JSON compatibility)
        fold_indices_info.append({
            "fold": i,
            "train_indices": tr_idx.tolist(),
            "valid_indices": va_idx.tolist(),
        })

        # Fit PCA on training fold only
        X_tr_pca, X_va_pca, pca, var_ratio = apply_pca_transform(
            X[tr_idx], X[va_idx], n_components
        )
        all_var_ratios.append(var_ratio.tolist())

        y_tr = y[tr_idx].astype(np.float32)
        y_va = y[va_idx].astype(np.float32)

        # Train and predict
        pc_names = [f"PC{i+1}" for i in range(X_tr_pca.shape[1])]
        _, y_pred = xgb_core_predict(X_tr_pca, y_tr, X_va_pca, obj_fn, xgb_params, return_booster=True, feature_names=pc_names)

        # Compute metrics
        losses.append(total_loss(y_va, y_pred, custom_objective_params))
        rmses.append(rmse(y_va, y_pred))
        saccs.append(sign_accuracy(y_va, y_pred))

    metrics = {
        "mean_loss": float(np.mean(losses)),
        "std_loss": float(np.std(losses)),
        "mean_rmse": float(np.mean(rmses)),
        "mean_sign_acc": float(np.mean(saccs)),
        "fold_losses": losses,
    }

    # Average explained variance across folds
    avg_var_ratio = np.mean(all_var_ratios, axis=0).tolist() if all_var_ratios else []

    return float(np.mean(losses)), metrics, avg_var_ratio, fold_indices_info


def _run_pca_hyperopt(
    target: str,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
    xgb_params: Dict,
    paper_params_factory: Optional[Callable[[np.ndarray], ObjectiveFunctionParameters]],
    n_components: int,
) -> Optional[Dict]:
    """Run hyperparameter optimization for PCA benchmark if enabled.
    
    Uses Optuna to find optimal XGBoost parameters with PCA features.
    Each trial fits PCA + XGBoost using time-series CV.
    
    Args:
        target: target name (for logging)
        X_train_raw: Raw training features (before PCA)
        y_train: Training targets
        cfg: Configuration with hyperopt settings
        xgb_params: Base XGBoost parameters
        paper_params: Custom objective parameters
        n_components: Number of PCA components
        
    Returns:
        Dict with best_params, best_value, n_trials if successful, None otherwise
    """
    # Delegate to shared optimizer (handles Optuna import internally)
    try:
        # Convert search space entries to plain dicts if Pydantic models
        search_space = {}
        for k, v in cfg.hyperopt.search_space.items():
            if hasattr(v, 'model_dump'):
                search_space[k] = v.model_dump()
            else:
                search_space[k] = dict(v) if hasattr(v, '__iter__') else v

        # Precompute per-fold PCA transforms and provide a cached preprocess_fn
        from ..training.hyperopt import create_time_series_cv_splits

        fold_indices = create_time_series_cv_splits(
            n_samples=len(y_train),
            n_splits=cfg.hyperopt.cv_folds,
            min_train_ratio=getattr(cfg.hyperopt, 'min_train_ratio', 0.5),
        )

        fold_cache = {}
        for i, (tr_idx, va_idx) in enumerate(fold_indices):
            X_tr_pca, X_va_pca, _, _ = apply_pca_transform(
                X_train_raw[tr_idx], X_train_raw[va_idx], n_components
            )
            pc_names = [f"PC{j+1}" for j in range(X_tr_pca.shape[1])]
            fold_cache[(tuple(tr_idx), tuple(va_idx))] = (
                X_tr_pca.astype(np.float32), X_va_pca.astype(np.float32), pc_names
            )

        def _cached_pca_preprocess(X_full: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray):
            key = (tuple(train_idx), tuple(val_idx))
            return fold_cache.get(key, (None, None, None))

        result = optimize_xgb_params(
            X_train=X_train_raw,
            y_train=y_train,
            cv_splitter=fold_indices,
            cv_folds=cfg.hyperopt.cv_folds,
            min_train_ratio=getattr(cfg.hyperopt, 'min_train_ratio', 0.5),
            objective_fn=total_loss,
            paper_params=None,
            paper_params_factory=paper_params_factory,
            make_obj_fn=make_xgb_objective,
            preprocess_fn=_cached_pca_preprocess,
            search_space=search_space,
            n_trials=cfg.hyperopt.n_trials,
            timeout=cfg.hyperopt.timeout,
            n_jobs=cfg.hyperopt.n_jobs,
            sampler=cfg.hyperopt.sampler,
            pruner=cfg.hyperopt.pruner,
            base_params=xgb_params,
            feature_names=None,
            seed=42,
        )
    except ImportError:
        logger.warning(f"[{target}] Optuna not installed - skipping hyperopt")
        return None

    if result is None:
        return None

    report = create_hyperopt_report(result)
    return {
        "enabled": True,
        "best_params": result.best_params,
        "best_value": float(result.best_value),
        "n_trials": int(result.n_trials),
        "report": report,
    }


def run_pca_benchmark_target(
    target: str,
    y_all: pd.Series,
    df_targets: pd.DataFrame,
    X_exogenous: pd.DataFrame,
    cfg: Config,
    xgb_params: Dict[str, Any],
    quantile_delta: float,
    quantile_gamma: float,
    config_dict: Dict[str, Any],
    n_components: int = 6,
    outdir: str = "benchmarking_artifacts",
    enable_hyperopt: bool = False,
) -> PCABenchmarkResult:
    """Run PCA benchmark pipeline for a single target.
    
    Pipeline steps:
    1. Build endogenous features for this target
    2. Combine with exogenous features
    3. Split into train/holdout
    4. Run expanding window CV with PCA + XGBoost
    5. (Optional) Run hyperparameter optimization
    6. Evaluate on holdout set
    
    Args:
        target: target name
        y_all: Full target series
        df_targets: Full targets DataFrame (for endogenous features)
        X_exogenous: Pre-built exogenous features DataFrame
        cfg: Configuration object
        xgb_params: XGBoost parameters
        quantile_delta: Quantile for δ threshold
        quantile_gamma: Quantile for γ threshold
        config_dict: Configuration as dictionary
        n_components: Number of PCA components
        outdir: Output directory
        
    Returns:
        PCABenchmarkResult with all outputs
    """
    logger.info(f"[{target}] Starting PCA benchmark pipeline")
    
    # Build endogenous features for this target
    endg = build_endogenous_features(df_targets, target, cfg.dataset.zscore_windows)
    
    # Combine endogenous + exogenous features
    X_combined = pd.concat([endg, X_exogenous], axis=1)
    feature_names = X_combined.columns.tolist()
    
    logger.info(f"[{target}] Feature matrix: {X_combined.shape[1]} features")
    

    # Clean dataset of NaNs (harmonized with IBA: only target and endogenous features)
    mask_valid = np.isfinite(y_all.values)
    if endg.shape[1] > 0:
        mask_valid &= np.isfinite(endg.values).all(axis=1)
    idx_clean = y_all.index[mask_valid]
    
    # Compute holdout indices
    eval_size = int(getattr(cfg.evaluation, 'size', 36))
    y_clean = y_all.loc[idx_clean]
    X_clean = X_combined.loc[idx_clean]
    
    tr_idx, ho_idx = compute_holdout_indices(
        y_clean,
        eval_size=eval_size,
        require_consecutive=True,
        fallback_to_recent_valid=True,
    )
    
    if len(ho_idx) == 0:
        return PCABenchmarkResult(
            target=target, skipped=True, skip_reason="no valid holdout window"
        )
    
    logger.info(
        f"[{target}] Train: {tr_idx[0].date()}..{tr_idx[-1].date()} ({len(tr_idx)} obs) | "
        f"Holdout: {ho_idx[0].date()}..{ho_idx[-1].date()} ({len(ho_idx)} obs)"
    )
    

    # Extract train/holdout data (do NOT drop rows for exogenous NaNs; instead, impute exogenous NaNs with zero)
    X_train_df = X_combined.loc[tr_idx].copy()
    y_train = y_all.loc[tr_idx].values.astype(np.float32)
    X_holdout_df = X_combined.loc[ho_idx].copy()
    y_holdout = y_all.loc[ho_idx].values.astype(np.float32)

    # Impute exogenous NaNs with zero (only for exogenous columns)
    exog_cols = [col for col in X_exogenous.columns if col in X_train_df.columns]
    X_train_df[exog_cols] = X_train_df[exog_cols].fillna(0.0)
    X_holdout_df[exog_cols] = X_holdout_df[exog_cols].fillna(0.0)

    X_train_raw = X_train_df.values.astype(np.float32)
    X_holdout_raw = X_holdout_df.values.astype(np.float32)
    
    if len(y_train) == 0:
        return PCABenchmarkResult(
            target=target, skipped=True, skip_reason="no training labels"
        )
    
    # Compute CV splits using the configured initial-train fraction and fold params
    M, Q, N, K = adjust_split_ratios(
        len(y_train),
        cfg.splits.init_tran_size,
        cfg.splits.folds_val_set_size,
        cfg.splits.folds_steps_size,
        cfg.splits.num_of_folds,
    )
    logger.info(f"[{target}] CV splits: M={M}, Q={Q}, N={N}, K={K}")
    
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
        f"[{target}] Using δ={paper_params.delta:.6f}, γ={paper_params.gamma:.6f}"
    )
    
    # Run CV evaluation
    cv_loss, cv_metrics, explained_var, cv_fold_indices = evaluate_pca_cv(
        X_train_raw, y_train, M, Q, N, K, xgb_params, paper_params, n_components
    )
    
    logger.info(f"[{target}] CV loss: {cv_loss:.6f}")
    
    # Hyperparameter optimization (optional)
    hyperopt_result = None
    effective_xgb_params = xgb_params.copy()
    
    if enable_hyperopt and hasattr(cfg, 'hyperopt') and cfg.hyperopt.enabled:
        # Build a factory to compute paper objective params (δ/γ) from training folds
        def _paper_params_factory(y_tr_arr: np.ndarray):
            return initialize_objective_params(
                y_train=y_tr_arr,
                config=config_dict,
                target=target,
                lambda1=cfg.objective.lambda1,
                lambda2=cfg.objective.lambda2,
                lambda3=cfg.objective.lambda3,
                quantile_delta=quantile_delta,
                quantile_gamma=quantile_gamma,
            )

        hyperopt_result = _run_pca_hyperopt(
            target, X_train_raw, y_train, cfg, xgb_params, _paper_params_factory, n_components
        )
        if hyperopt_result:
            for k, v in hyperopt_result.get("best_params", {}).items():
                if k == "n_estimators":
                    effective_xgb_params["n_estimators"] = int(v)
                elif k == "min_split_loss":
                    # Map back to gamma for XGBoost
                    effective_xgb_params["gamma"] = v
                else:
                    effective_xgb_params[k] = v
            logger.info(f"[{target}] Using hyperopt params: {hyperopt_result.get('best_params', {})}")
    
    # Final model training on full training set with PCA
    X_train_pca, X_holdout_pca, pca_model, final_var_ratio = apply_pca_transform(
        X_train_raw, X_holdout_raw, n_components
    )
    
    logger.info(
        f"[{target}] PCA: {X_train_pca.shape[1]} components, "
        f"explained variance: {sum(final_var_ratio):.2%}"
    )
    
    # Train final model with rolling retrain if enabled
    obj_fn = make_xgb_objective(paper_params)
    eval_cfg = getattr(cfg, 'evaluation', None)
    do_retrain = bool(eval_cfg and getattr(eval_cfg, 'retrain', False))
    retrain_steps = int(getattr(eval_cfg, 'steps', 0)) if eval_cfg else 0
    
    pc_names = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
    
    if not do_retrain:
        booster, y_pred = xgb_core_predict(
            X_train_pca, y_train, X_holdout_pca, obj_fn, effective_xgb_params, return_booster=True, feature_names=pc_names
        )
    else:
        # Rolling retrain on holdout - predict one step at a time
        y_pred_list = []
        booster = None
        
        # Track current state for rolling window
        # Each time we retrain, we re-fit PCA on expanded training data
        # and transform remaining holdout data accordingly
        remaining_holdout_pca = None
        last_retrain_idx = 0
        
        for i in range(len(X_holdout_pca)):
            need_train = (booster is None) or (retrain_steps > 0 and i % retrain_steps == 0)
            
            if need_train:
                if i > 0:
                    # Expand training set with holdout observations seen so far
                    X_expanded_raw = np.vstack([X_train_raw, X_holdout_raw[:i]])
                    y_expanded = np.concatenate([y_train, y_holdout[:i]])
                    
                    # Re-fit PCA on expanded data, transform remaining holdout
                    X_expanded_pca, remaining_holdout_pca, _, _ = apply_pca_transform(
                        X_expanded_raw, X_holdout_raw[i:], n_components
                    )
                    
                    booster, _ = xgb_core_predict(
                        X_expanded_pca, y_expanded, remaining_holdout_pca, obj_fn, effective_xgb_params, return_booster=True, feature_names=pc_names
                    )
                else:
                    # Initial training - use pre-computed PCA
                    remaining_holdout_pca = X_holdout_pca.copy()
                    booster, _ = xgb_core_predict(
                        X_train_pca, y_train, remaining_holdout_pca, obj_fn, effective_xgb_params, return_booster=True, feature_names=pc_names
                    )
                
                last_retrain_idx = i
            
            # Predict current observation
            # Index relative to last retrain point
            rel_idx = i - last_retrain_idx
            dcur = xgb.DMatrix(
                remaining_holdout_pca[rel_idx:rel_idx+1].astype(np.float32), 
                feature_names=pc_names
            )
            y_pred_i = booster.predict(dcur)
            y_pred_list.append(float(y_pred_i[0]))
        
        y_pred = np.array(y_pred_list, dtype=np.float32)
    
    # Compute holdout metrics
    realized = y_holdout
    predicted = y_pred
    
    gate_kind = getattr(cfg, 'simulation', None).gate if getattr(cfg, 'simulation', None) else 'zero'
    thr = float(paper_params.delta) if gate_kind == 'delta' else 0.0
    signal = (predicted >= thr).astype(np.float32)
    simulated_returns = signal * realized
    
    realized_cum = np.cumprod(1.0 + realized) - 1.0
    predicted_cum = np.cumprod(1.0 + predicted) - 1.0
    simulated_cum = np.cumprod(1.0 + simulated_returns) - 1.0
    
    holdout_loss = total_loss(realized, predicted, paper_params)
    loss_comps = loss_components(realized, predicted, paper_params)
    
    metrics = {
        "rmse": float(rmse(realized, predicted)),
        "sign_acc": float(sign_accuracy(realized, predicted)),
        "sharpe_realized": float(sharpe_ratio(realized)),
        "mdd_realized": float(max_drawdown(realized_cum)),
        "sharpe_pred": float(sharpe_ratio(predicted)),
        "mdd_pred": float(max_drawdown(predicted_cum)),
        "sharpe_sim": float(sharpe_ratio(simulated_returns)),
        "mdd_sim": float(max_drawdown(simulated_cum)),
        "holdout_loss": float(holdout_loss),
        "loss_components": loss_comps,
    }
    
    logger.info(
        f"[{target}] Holdout: loss={holdout_loss:.6f}, RMSE={metrics['rmse']:.6f}, "
        f"SignAcc={metrics['sign_acc']:.2%}"
    )
    
    # Save model
    models_dir = os.path.join(outdir, "models")
    os.makedirs(models_dir, exist_ok=True)
    booster.save_model(os.path.join(models_dir, f"{target}_pca_best.json"))
    
    return PCABenchmarkResult(
        target=target,
        n_components=X_train_pca.shape[1],
        explained_variance_ratio=final_var_ratio.tolist(),
        train_size=len(tr_idx),
        holdout_size=len(ho_idx),
        delta=float(paper_params.delta),
        gamma=float(paper_params.gamma),
        delta_gamma_source="quantiles" if paper_params.computed_from_quantiles else "default",
        cv_loss=cv_loss,
        cv_metrics=cv_metrics,
        holdout={
            "index": [str(x) for x in ho_idx],
            "realized": realized.tolist(),
            "predicted": predicted.tolist(),
            "simulated": simulated_cum.tolist(),
            "simulated_returns": simulated_returns.tolist(),
            "metrics": metrics,
            "evaluation": {"retrain": do_retrain, "steps": int(retrain_steps)},
        },
        feature_names=feature_names,
        hyperopt=hyperopt_result,
        cv_fold_indices=cv_fold_indices,
    )


def run_pca_benchmark(
    cfg: Config,
    config_dict: Dict[str, Any],
    outdir: str = "benchmarking_artifacts",
    n_components: int = 6,
    enable_hyperopt: bool = False,
) -> List[PCABenchmarkResult]:
    """Run PCA benchmark for all target.
    
    Args:
        cfg: Configuration object
        config_dict: Configuration as dictionary
        outdir: Output directory
        n_components: Number of PCA components
        enable_hyperopt: Enable hyperparameter optimization (uses config settings)
        
    Returns:
        List of PCABenchmarkResult for each target
    """
    logger.info("=" * 60)
    logger.info("Starting PCA Benchmark Pipeline")
    if enable_hyperopt:
        logger.info("Hyperparameter optimization: ENABLED")
    logger.info("=" * 60)
    
    # Read and prepare data
    df_factors, df_targets = read_raw_pair(
        cfg.dataset.factors_path,
        cfg.dataset.targets_path,
        cfg.dataset.date_column,
        cfg.dataset.parse_dates,
        align_freq=cfg.dataset.align_freq,
        align_agg=cfg.dataset.align_agg,
        align_how=cfg.dataset.align_how,
    )
    
    # Normalize to month-end
    df_factors.index = normalize_to_month_end(pd.DatetimeIndex(df_factors.index))
    df_targets.index = normalize_to_month_end(pd.DatetimeIndex(df_targets.index))
    
    # Build cumulative shifted returns
    cum_returns = cumulative_shifted_returns(
        df_targets[cfg.dataset.target_columns],
        cfg.dataset.cumulative_horizon,
        cfg.dataset.forecast_shift,
    )
    cum_returns.index = normalize_to_month_end(pd.DatetimeIndex(cum_returns.index))
    
    # Apply date filters if specified
    starting_date = getattr(cfg.splits, 'starting_date', None)
    end_date = getattr(cfg.splits, 'end_date', None)
    cum_returns = apply_date_filters(cum_returns, starting_date, end_date)
    df_factors = apply_date_filters(df_factors, starting_date, end_date)
    df_targets = apply_date_filters(df_targets, starting_date, end_date)
    
    # Build factors map from config
    factors_map = {k: list(v) for k, v in cfg.factors.items()}
    
    # Build exogenous features (all factor groups, z-score normalized)
    X_exogenous = build_exogenous_features(df_factors, factors_map, cfg.dataset.zscore_windows)
    
    logger.info(f"Exogenous features: {X_exogenous.shape[1]} columns")
    logger.info(f"Targets: {cfg.dataset.target_columns}")
    
    # Get XGBoost params
    xgb_params = dict(cfg.xgb.params) if hasattr(cfg.xgb, 'params') else {}
    
    # Get quantile params
    quantile_delta, quantile_gamma = get_quantile_params_from_config(config_dict)
    
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "reports"), exist_ok=True)
    
    # Run benchmark for each target
    results = []
    for target in cfg.dataset.target_columns:
        if target not in cum_returns.columns:
            logger.warning(f"[{target}] Not found in cumulative returns, skipping")
            continue
        
        y_all = cum_returns[target]
        
        result = run_pca_benchmark_target(
            target=target,
            y_all=y_all,
            df_targets=df_targets,
            X_exogenous=X_exogenous,
            cfg=cfg,
            xgb_params=xgb_params,
            quantile_delta=quantile_delta,
            quantile_gamma=quantile_gamma,
            config_dict=config_dict,
            n_components=n_components,
            outdir=outdir,
            enable_hyperopt=enable_hyperopt,
        )
        results.append(result)
    
    return results


def write_benchmark_results(
    results: List[PCABenchmarkResult],
    outdir: str,
    timestamp: str,
    n_components: int = 6,
) -> None:
    """Write benchmark results to files aligned with IBA reporting format.
    
    Creates:
    - reports/run_pca_benchmark_{timestamp}.json: Full results
    - reports/Table1_DeltaGamma.csv: Delta/Gamma thresholds per target
    - reports/Table2_Performance.csv: Performance metrics comparison
    - reports/Table3_ErrorAnalysis.csv: CV and holdout error metrics
    - reports/AnalyticalReport_{timestamp}.html: Full HTML report
    - data/holdout_{target}.csv: Holdout predictions for each target
    """
    reports_dir = os.path.join(outdir, "reports")
    data_dir = os.path.join(outdir, "data")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Full JSON results
    json_path = os.path.join(reports_dir, f"run_pca_benchmark_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logger.info(f"Saved full results to {json_path}")
    
    # Collect data for tables
    targets, deltas, gammas = [], [], []
    realized_returns, predicted_returns, simulated_returns = [], [], []
    realized_sharpe, predicted_sharpe, simulated_sharpe = [], [], []
    realized_mdd, predicted_mdd, simulated_mdd = [], [], []
    cv_loss, cv_mse, cv_sign_acc = [], [], []
    holdout_loss, holdout_mse, holdout_sign_acc = [], [], []
    
    for r in results:
        if r.skipped:
            continue
        
        targets.append(r.target)
        deltas.append(r.delta)
        gammas.append(r.gamma)
        
        ho = r.holdout or {}
        m = ho.get("metrics", {})
        
        # Compute cumulative returns from holdout arrays
        realized_arr = np.array(ho.get("realized", []))
        predicted_arr = np.array(ho.get("predicted", []))
        simulated_arr = np.array(ho.get("simulated", []))
        
        if len(realized_arr) > 0:
            realized_returns.append(float(np.cumprod(1.0 + realized_arr)[-1] - 1.0))
        else:
            realized_returns.append(np.nan)
        
        if len(predicted_arr) > 0:
            predicted_returns.append(float(np.cumprod(1.0 + predicted_arr)[-1] - 1.0))
        else:
            predicted_returns.append(np.nan)
        
        if len(simulated_arr) > 0:
            simulated_returns.append(float(simulated_arr[-1]))
        else:
            simulated_returns.append(np.nan)
        
        realized_sharpe.append(m.get("sharpe_realized", np.nan))
        predicted_sharpe.append(m.get("sharpe_pred", np.nan))
        simulated_sharpe.append(m.get("sharpe_sim", np.nan))
        realized_mdd.append(m.get("mdd_realized", np.nan))
        predicted_mdd.append(m.get("mdd_pred", np.nan))
        simulated_mdd.append(m.get("mdd_sim", np.nan))
        
        # CV metrics
        cv_metrics = r.cv_metrics or {}
        cv_loss.append(r.cv_loss)
        cv_rmse = cv_metrics.get("mean_rmse", np.nan)
        cv_mse.append(cv_rmse ** 2 if not np.isnan(cv_rmse) else np.nan)
        cv_sign_acc.append(cv_metrics.get("mean_sign_acc", np.nan))
        
        # Holdout metrics
        holdout_loss.append(m.get("holdout_loss", np.nan))
        ho_rmse = m.get("rmse", np.nan)
        holdout_mse.append(ho_rmse ** 2 if not np.isnan(ho_rmse) else np.nan)
        holdout_sign_acc.append(m.get("sign_acc", np.nan))
    
    # Table 1: Delta/Gamma Thresholds
    table1 = pd.DataFrame({
        "Target": targets,
        "Delta (δ)": deltas,
        "Gamma (γ)": gammas,
    })
    table1_path = os.path.join(reports_dir, "Table1_DeltaGamma.csv")
    table1.to_csv(table1_path, index=False)
    logger.info(f"Saved Table1 to {table1_path}")
    
    # Table 2: Performance Metrics
    table2 = pd.DataFrame({
        "Target": targets,
        "Realized Total Return": realized_returns,
        "Predicted Total Return": predicted_returns,
        "Simulated Total Return": simulated_returns,
        "Realized Sharpe": realized_sharpe,
        "Predicted Sharpe": predicted_sharpe,
        "Simulated Sharpe": simulated_sharpe,
        "Realized MDD": realized_mdd,
        "Predicted MDD": predicted_mdd,
        "Simulated MDD": simulated_mdd,
    })
    table2_path = os.path.join(reports_dir, "Table2_Performance.csv")
    table2.to_csv(table2_path, index=False)
    logger.info(f"Saved Table2 to {table2_path}")
    
    # Table 3: Error Analysis
    table3 = pd.DataFrame({
        "Target": targets,
        "CV Loss": cv_loss,
        "CV MSE": cv_mse,
        "CV Sign Accuracy": cv_sign_acc,
        "Holdout Loss": holdout_loss,
        "Holdout MSE": holdout_mse,
        "Holdout Sign Accuracy": holdout_sign_acc,
    })
    table3_path = os.path.join(reports_dir, "Table3_ErrorAnalysis.csv")
    table3.to_csv(table3_path, index=False)
    logger.info(f"Saved Table3 to {table3_path}")
    
    # Generate figures
    assets_dir = os.path.join(reports_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    fig_paths = {}
    
    for result in results:
        if result.skipped:
            continue
        ho_dict = dict(result.holdout)
        if "simulated_returns" in ho_dict:
            sim_ret = np.array(ho_dict["simulated_returns"], dtype=float)
            ho_dict["simulated"] = (np.cumprod(1.0 + sim_ret) - 1.0).tolist()
        fig_path = save_holdout_figure(assets_dir, result.target, ho_dict)
        if fig_path:
            fig_paths[result.target] = fig_path
            logger.info(f"Generated figure for {result.target}")
    
    # Holdout CSVs per target
    for result in results:
        if result.skipped or result.holdout is None:
            continue
        
        ho = result.holdout
        realized_arr = np.array(ho["realized"])
        predicted_arr = np.array(ho["predicted"])
        simulated_ret = np.array(ho.get("simulated_returns", []))
        
        df_ho = pd.DataFrame({
            "idx": ho["index"],
            "realized": ho["realized"],
            "predicted": ho["predicted"],
            "simulated": simulated_ret.tolist() if len(simulated_ret) > 0 else ho.get("simulated", []),
            "realized_cum": np.cumsum(realized_arr).tolist(),
            "predicted_cum": np.cumsum(predicted_arr).tolist(),
        })
        csv_path = os.path.join(data_dir, f"holdout_{result.target}.csv")
        df_ho.to_csv(csv_path, index=False)
    
    # Generate HTML Report
    html_path = _write_pca_html_report(
        results, reports_dir, timestamp, n_components,
        table1_path, table2_path, table3_path, fig_paths
    )
    logger.info(f"Saved HTML report to {html_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PCA BENCHMARK RESULTS")
    print("=" * 70)
    print("\nTable 3: Error Analysis")
    print(table3.to_string(index=False))
    print("\nTable 2: Performance (Simulated)")
    print(table2[["Target", "Simulated Total Return", "Simulated Sharpe", "Simulated MDD"]].to_string(index=False))
    print("=" * 70)


def _write_pca_html_report(
    results: List[PCABenchmarkResult],
    outdir: str,
    timestamp: str,
    n_components: int,
    table1_path: str,
    table2_path: str,
    table3_path: str,
    fig_paths: Optional[Dict[str, str]] = None,
) -> str:
    """Generate HTML analytical report for PCA benchmark."""
    
    if fig_paths is None:
        fig_paths = {}
    
    css = """
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 20px;
        background: #f5f5f5;
        color: #333;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2e7d32;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 10px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 13px;
    }
    th {
        background: #2e7d32;
        color: white;
        padding: 12px;
        text-align: left;
    }
    td {
        padding: 10px 12px;
        border-bottom: 1px solid #ddd;
    }
    tr:hover {
        background: #f9f9f9;
    }
    .card {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 15px;
        margin: 15px 0;
        background: #fafafa;
    }
    .formula {
        background: #e8f5e9;
        padding: 10px;
        border-left: 4px solid #2e7d32;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    .metric {
        display: inline-block;
        background: #e8f5e9;
        padding: 8px 12px;
        margin: 5px;
        border-radius: 4px;
        font-size: 12px;
    }
    .pca-info {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 15px 0;
    }
    """
    
    html = ["<!DOCTYPE html>", "<html>", "<head>"]
    html.append("<meta charset='utf-8'>")
    html.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append("<title>PCA Benchmark Report</title>")
    html.append(f"<style>{css}</style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<div class='container'>")
    
    # Header
    html.append("<h1>PCA Benchmark Report</h1>")
    html.append(f"<p><strong>Generated</strong>: {timestamp}</p>")
    html.append("<p><strong>Method</strong>: Principal Component Analysis (PCA) Feature Reduction + XGBoost</p>")
    
    # Executive Summary
    html.append("<h2>Executive Summary</h2>")
    html.append("<p>PCA-based benchmark comparison for hedge fund target forecasting.</p>")
    html.append("<ul>")
    html.append(f"<li><strong>PCA Components</strong>: {n_components}</li>")
    html.append("<li><strong>Targets</strong>: 8 (CTA, CA, DS, ELS, EMN, EDMS, FIA, GM)</li>")
    html.append("<li><strong>Objective</strong>: Same custom loss function as IBA (λ₁·SE + λ₂·ME⁻ + λ₃·ME⁺ + MSE)</li>")
    html.append("<li><strong>CV Scheme</strong>: Expanding window cross-validation (same as IBA)</li>")
    html.append("</ul>")
    
    # PCA Methodology
    html.append("<div class='pca-info'>")
    html.append("<h3>PCA Methodology</h3>")
    html.append("<p>Unlike IBA which uses interpretable Boolean polynomials, PCA reduces dimensionality by:</p>")
    html.append("<ol>")
    html.append("<li>Combining exogenous factors (IR, B, V, AP, TF groups) with endogenous features</li>")
    html.append(f"<li>Applying PCA to extract {n_components} principal components</li>")
    html.append("<li>Training XGBoost on PC scores with the same custom objective function</li>")
    html.append("</ol>")
    html.append("</div>")
    
    # Table 1: Delta/Gamma
    html.append("<h2>Table 1: Delta/Gamma Thresholds</h2>")
    df1 = pd.read_csv(table1_path)
    html.append(_df_to_html_table(df1))
    
    # Table 2: Performance
    html.append("<h2>Table 2: Performance Metrics</h2>")
    df2 = pd.read_csv(table2_path)
    html.append(_df_to_html_table(df2, fmt=".4f"))
    
    # Table 3: Error Analysis
    html.append("<h2>Table 3: Error Analysis</h2>")
    df3 = pd.read_csv(table3_path)
    html.append(_df_to_html_table(df3, fmt=".6f"))
    
    # Per-target details
    html.append("<h2>PCA Models per target</h2>")
    
    for r in results:
        if r.skipped:
            continue
        
        ho = r.holdout or {}
        m = ho.get("metrics", {})
        
        # Compute cumulative return
        simulated_arr = np.array(ho.get("simulated", []))
        cum_return = float(simulated_arr[-1]) if len(simulated_arr) > 0 else np.nan
        
        html.append(f"<div class='card'>")
        html.append(f"<h3>{r.target} Target</h3>")
        
        # PCA Info
        explained_var = sum(r.explained_variance_ratio) if r.explained_variance_ratio else 0
        html.append(f"<div class='formula'>")
        html.append(f"PCA Model: {r.n_components} components explaining {explained_var:.1%} of variance")
        html.append("</div>")
        
        # Performance table
        html.append("<h4>Performance Summary</h4>")
        html.append("<table style='font-size: 12px;'>")
        html.append("<tr><th>Metric</th><th>Cross-Validation</th><th>Holdout</th></tr>")
        
        cv_m = r.cv_metrics or {}
        cv_rmse = cv_m.get("mean_rmse", np.nan)
        ho_rmse = m.get("rmse", np.nan)
        
        html.append(f"<tr><td>Loss</td><td>{r.cv_loss:.6f}</td><td>{m.get('holdout_loss', np.nan):.6f}</td></tr>")
        html.append(f"<tr><td>MSE</td><td>{cv_rmse**2:.6f}</td><td>{ho_rmse**2:.6f}</td></tr>")
        html.append(f"<tr><td>Sign Accuracy</td><td>{cv_m.get('mean_sign_acc', np.nan):.2%}</td><td>{m.get('sign_acc', np.nan):.2%}</td></tr>")
        html.append(f"<tr><td>Cumulative Return</td><td>-</td><td>{cum_return:.2%}</td></tr>")
        html.append(f"<tr><td>Sharpe Ratio</td><td>-</td><td>{m.get('sharpe_sim', np.nan):.4f}</td></tr>")
        html.append(f"<tr><td>Max Drawdown</td><td>-</td><td>{m.get('mdd_sim', np.nan):.2%}</td></tr>")
        html.append("</table>")
        
        html.append(f"<p><strong>Thresholds:</strong> δ = {r.delta:.6f}, γ = {r.gamma:.6f}</p>")
        
        # Holdout figure
        if r.target in fig_paths:
            html.append("<h4>Holdout Performance</h4>")
            html.append("<p><em>Realized, Predicted, and Simulated cumulative returns over test period</em></p>")
            html.append(f"<img src='assets/{r.target}_holdout.png' alt='{r.target} holdout' style='max-width: 800px;'/>")
        
        html.append("</div>")
    
    html.append("</div>")  # container
    html.append("</body>")
    html.append("</html>")
    
    html_path = os.path.join(outdir, f"AnalyticalReport_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    
    return html_path


def _df_to_html_table(df: pd.DataFrame, fmt: str = None) -> str:
    """Convert DataFrame to HTML table string."""
    lines = ["<table>", "<tr>"]
    for col in df.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr>")
    
    for _, row in df.iterrows():
        lines.append("<tr>")
        for i, val in enumerate(row):
            if fmt and isinstance(val, (int, float)) and i > 0 and not np.isnan(val):
                lines.append(f"<td>{val:{fmt}}</td>")
            else:
                lines.append(f"<td>{val}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def main():
    """CLI entry point for PCA benchmark."""
    parser = argparse.ArgumentParser(
        description="Run PCA benchmark for IBAML comparison"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="benchmarking_artifacts",
        help="Output directory (default: benchmarking_artifacts)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=6,
        help="Number of PCA components (default: 6)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Enable hyperparameter optimization (default: disabled)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Load config
    cfg = load_config(args.config)
    
    # Load raw config dict for objective parameters
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # Generate timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    # Run benchmark
    results = run_pca_benchmark(
        cfg=cfg,
        config_dict=config_dict,
        outdir=args.outdir,
        n_components=args.n_components,
        enable_hyperopt=args.hyperopt,
    )
    
    # Write results
    write_benchmark_results(results, args.outdir, timestamp, args.n_components)
    
    logger.info("PCA benchmark complete!")


if __name__ == "__main__":
    main()
