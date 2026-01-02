from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Lazy import Optuna to avoid hard dependency
_optuna = None


def _get_optuna():
    """Lazy import Optuna.
    
    Raises:
        ImportError: If Optuna is not installed.
    """
    global _optuna
    if _optuna is None:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            _optuna = optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install it with: pip install optuna"
            )
    return _optuna


@dataclass
class HyperoptResult:
    """Result container for hyperparameter optimization.
    
    Attributes:
        best_params: Best hyperparameters found
        best_value: Best objective value (CV loss)
        n_trials: Number of trials completed
        study: Optuna study object (for analysis)
        all_trials: List of all trial results
    """
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study: Any  # optuna.Study
    all_trials: List[Dict[str, Any]] = field(default_factory=list)


def _create_sampler(sampler_type: str, seed: Optional[int] = None):
    """Create Optuna sampler based on config."""
    optuna = _get_optuna()
    
    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    elif sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_type == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def _create_pruner(pruner_type: str):
    """Create Optuna pruner based on config."""
    optuna = _get_optuna()
    
    if pruner_type == "median":
        return optuna.pruners.MedianPruner()
    elif pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner()
    elif pruner_type == "none":
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner type: {pruner_type}")


def _suggest_param(trial, name: str, space: Dict[str, Any]) -> Any:
    """Suggest a parameter value based on search space definition.
    
    Args:
        trial: Optuna trial
        name: Parameter name
        space: Search space definition (from HyperoptSearchSpace)
        
    Returns:
        Suggested parameter value
    """
    space_type = space.get("type", "float")
    
    if space_type == "int":
        step = space.get("step")
        # Optuna requires step to be a positive int or None
        if step is None or step <= 0:
            return trial.suggest_int(name, int(space["low"]), int(space["high"]))
        else:
            return trial.suggest_int(name, int(space["low"]), int(space["high"]), step=int(step))
    
    elif space_type == "float":
        return trial.suggest_float(name, space["low"], space["high"])
    
    elif space_type == "log_float":
        return trial.suggest_float(name, space["low"], space["high"], log=True)
    
    elif space_type == "categorical":
        return trial.suggest_categorical(name, space["choices"])
    
    else:
        raise ValueError(f"Unknown search space type: {space_type}")


def create_time_series_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train_ratio: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create time-series aware train/validation splits (expanding window).
    
    This is critical for financial time series to prevent data leakage.
    Training always uses past data, validation uses future data.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of CV folds
        min_train_ratio: Minimum fraction of data for initial training (default 0.5)
        
    Returns:
        List of (train_indices, val_indices) tuples
        
    Example with n_samples=200, n_splits=5, min_train_ratio=0.5:
        Fold 0: Train [0:100], Val [100:120]
        Fold 1: Train [0:120], Val [120:140]
        Fold 2: Train [0:140], Val [140:160]
        Fold 3: Train [0:160], Val [160:180]
        Fold 4: Train [0:180], Val [180:200]
    """
    min_train_size = int(n_samples * min_train_ratio)
    
    # Ensure we have enough data
    if n_samples < min_train_size + n_splits:
        n_splits = max(1, n_samples - min_train_size)
        logger.warning(f"Reduced hyperopt CV splits to {n_splits} due to data size")
    
    # Calculate step size - how much training grows each fold
    available_for_expansion = n_samples - min_train_size
    step_size = max(1, available_for_expansion // (n_splits + 1))
    val_size = step_size  # Validation window equals step size
    
    splits = []
    for i in range(n_splits):
        # Training window expands with each fold
        train_end = min_train_size + i * step_size
        val_start = train_end
        val_end = min(val_start + val_size, n_samples)
        
        if val_start >= n_samples or val_end <= val_start:
            break
            
        train_idx = np.arange(0, train_end, dtype=int)
        val_idx = np.arange(val_start, val_end, dtype=int)
        
        splits.append((train_idx, val_idx))
    
    return splits


def optimize_xgb_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splitter=None,  # ExpandingWindowSplit or None to create internally
    cv_folds: int = 5,  # Used if cv_splitter is None
    min_train_ratio: float = 0.5,
    objective_fn: Callable = None,  # paper_total_loss function
    paper_params: Any = None,  # PaperObjParams
    paper_params_factory: Optional[Callable[[np.ndarray], Any]] = None,
    make_obj_fn: Callable = None,  # make_xgb_objective factory
    preprocess_fn: Optional[Callable] = None,  # Optional per-fold preprocessing
    search_space: Dict[str, Dict[str, Any]] = None,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    n_jobs: int = 1,
    sampler: str = "tpe",
    pruner: str = "median",
    base_params: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
    seed: Optional[int] = 42,
) -> HyperoptResult:
    """Run Optuna hyperparameter optimization for XGBoost.
    
    Uses time-series aware cross-validation (expanding window) to prevent
    data leakage, which is critical for financial forecasting.
    
    Args:
        X_train: Training features (must be in chronological order)
        y_train: Training labels (must be in chronological order)
        cv_splitter: Optional ExpandingWindowSplit instance. If None, creates
                     time-series CV splits internally using cv_folds.
        cv_folds: Number of CV folds (used if cv_splitter is None)
        objective_fn: Loss function for evaluation (paper_total_loss)
        paper_params: PaperObjParams with delta/gamma thresholds
        make_obj_fn: Factory to create XGBoost objective function
        search_space: Parameter search spaces (from HyperoptCfg)
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds
        n_jobs: Parallel trials
        sampler: Sampler type ('tpe', 'random', 'cmaes')
        pruner: Pruner type ('median', 'hyperband', 'none')
        base_params: Base XGBoost parameters to merge with suggested
        feature_names: Feature names for DMatrix
        seed: Random seed for reproducibility
        
    Returns:
        HyperoptResult with best parameters and study
    """
    import xgboost as xgb
    optuna = _get_optuna()
    
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    base_params = dict(base_params or {})
    search_space = search_space or {}
    
    # Create time-series CV splits
    if cv_splitter is not None:
        fold_indices = list(cv_splitter)
    else:
        # Create internal time-series CV splits
        fold_indices = create_time_series_cv_splits(
            n_samples=len(y_train),
            n_splits=cv_folds,
            min_train_ratio=min_train_ratio,
        )
    
    n_folds = len(fold_indices)
    
    if n_folds == 0:
        raise ValueError("CV splitter produced no folds")
    
    logger.info(
        f"Starting hyperparameter optimization: {n_trials} trials, "
        f"{n_folds} time-series CV folds, sampler={sampler}, pruner={pruner}"
    )
    
    # Log fold structure
    for i, (tr_idx, va_idx) in enumerate(fold_indices):
        logger.debug(f"  Fold {i+1}: train=[0:{len(tr_idx)}], val=[{len(tr_idx)}:{len(tr_idx)+len(va_idx)}]")
    
    def objective(trial) -> float:
        """Optuna objective function with time-series CV."""
        # Suggest hyperparameters
        params = dict(base_params)
        for param_name, space in search_space.items():
            # Convert Pydantic model to dict if needed
            if hasattr(space, 'model_dump'):
                space = space.model_dump()
            params[param_name] = _suggest_param(trial, param_name, space)
        
        # Handle parameter name mapping:
        # - 'min_split_loss' in search_space -> 'gamma' for XGBoost
        # This avoids confusion with gamma's γ threshold parameter
        if "min_split_loss" in params:
            params["gamma"] = params.pop("min_split_loss")
        
        # Extract n_estimators
        num_boost_round = int(params.pop("n_estimators", 500))
        params.setdefault("verbosity", 0)
        params.setdefault("objective", "reg:squarederror")
        
        # Note: if `paper_params_factory` is provided, we'll compute fold-specific
        # paper params (δ/γ and related thresholds) from the training fold to
        # avoid leakage. The XGBoost objective will be created per-fold when
        # a factory is used.
        
        # Time-series cross-validation
        fold_losses = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            # Optionally preprocess per-fold (e.g., fit PCA on train and transform val)
            if preprocess_fn is not None:
                X_tr, X_va, fnames = preprocess_fn(X_train, train_idx, val_idx)
                if fnames is not None and feature_names is None:
                    current_feature_names = fnames
                else:
                    current_feature_names = feature_names
            else:
                X_tr = X_train[train_idx]
                X_va = X_train[val_idx]
                current_feature_names = feature_names

            y_tr = y_train[train_idx]
            y_va = y_train[val_idx]
            
            # Filter NaN
            tr_mask = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
            X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
            
            va_mask = np.isfinite(y_va) & np.isfinite(X_va).all(axis=1)
            X_va, y_va = X_va[va_mask], y_va[va_mask]
            
            if len(y_tr) == 0 or len(y_va) == 0:
                continue
            
            # Compute fold-specific paper params (δ/γ) if requested
            fold_paper_params = None
            if paper_params_factory is not None:
                try:
                    fold_paper_params = paper_params_factory(y_tr)
                except Exception as e:
                    logger.warning(f"Failed computing paper params for fold {fold_idx}: {e}")
                    return float('inf')
            else:
                fold_paper_params = paper_params

            # Create per-fold objective if needed
            xgb_obj_fold = None
            if make_obj_fn is not None and fold_paper_params is not None:
                xgb_obj_fold = make_obj_fn(fold_paper_params)
                params.setdefault("base_score", 0.0)

            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=current_feature_names)
            dval = xgb.DMatrix(X_va, feature_names=current_feature_names)

            try:
                booster = xgb.train(
                    params, dtrain, num_boost_round=num_boost_round, obj=xgb_obj_fold
                )
                y_pred = booster.predict(dval)

                # Compute loss using fold-specific paper params if available
                if objective_fn is not None and fold_paper_params is not None:
                    loss = objective_fn(y_va, y_pred, fold_paper_params)
                else:
                    loss = float(np.mean((y_va - y_pred) ** 2))
                fold_losses.append(loss)

            except Exception as e:
                logger.warning(f"Trial {trial.number} fold {fold_idx} failed: {e}")
                return float('inf')
            
            # Report intermediate value for pruning
            if fold_losses:
                trial.report(np.mean(fold_losses), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        if not fold_losses:
            return float('inf')
        
        return float(np.mean(fold_losses))
    
    # Create study
    sampler_obj = _create_sampler(sampler, seed)
    pruner_obj = _create_pruner(pruner)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler_obj,
        pruner=pruner_obj,
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    
    # Collect results
    all_trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            all_trials.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
            })
    
    logger.info(
        f"Optimization complete: best_value={study.best_value:.6f}, "
        f"n_trials={len(study.trials)}"
    )
    logger.info(f"Best parameters: {study.best_params}")
    
    return HyperoptResult(
        best_params=study.best_params,
        best_value=study.best_value,
        n_trials=len(study.trials),
        study=study,
        all_trials=all_trials,
    )


def get_optuna_importance(study) -> Dict[str, float]:
    """Get parameter importance from Optuna study.
    
    Args:
        study: Optuna study object
        
    Returns:
        Dictionary mapping parameter names to importance scores
    """
    optuna = _get_optuna()
    
    try:
        importance = optuna.importance.get_param_importances(study)
        return dict(importance)
    except Exception as e:
        logger.warning(f"Could not compute parameter importance: {e}")
        return {}


def create_hyperopt_report(result: HyperoptResult) -> Dict[str, Any]:
    """Create a summary report of hyperparameter optimization.
    
    Args:
        result: HyperoptResult from optimize_xgb_params
        
    Returns:
        Dictionary with optimization summary
    """
    report = {
        "best_params": result.best_params,
        "best_value": result.best_value,
        "n_trials": result.n_trials,
        "n_complete": len(result.all_trials),
    }
    
    # Parameter importance
    importance = get_optuna_importance(result.study)
    if importance:
        report["param_importance"] = importance
    
    # Top 5 trials
    sorted_trials = sorted(result.all_trials, key=lambda t: t["value"])
    report["top_5_trials"] = sorted_trials[:5]
    
    return report
