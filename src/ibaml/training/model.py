from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


@dataclass
class XGBTrainResult:
    """Result container for XGBoost training.
    
    Attributes:
        booster: Trained XGBoost booster
        predictions: Predictions on validation/test set
        feature_importance: Feature importance scores (gain-based)
        train_loss: Training loss (if computed)
        n_iterations: Number of boosting rounds used
    """
    booster: xgb.Booster
    predictions: np.ndarray
    feature_importance: Dict[str, float]
    train_loss: Optional[float] = None
    n_iterations: int


def prepare_xgb_params(
    params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
) -> Tuple[Dict[str, Any], int]:
    """Prepare XGBoost parameters for training.
    
    Extracts n_estimators and normalizes parameter names.
    
    Args:
        params: Raw XGBoost parameters
        n_jobs: Number of parallel threads
        
    Returns:
        (normalized_params, num_boost_round)
    """
    params = dict(params or {})
    
    # Extract n_estimators (not a native xgb.train param)
    num_boost_round = int(params.pop("n_estimators", 500))
    
    # Normalize thread settings
    n_jobs = int(params.pop("n_jobs", n_jobs))
    params.setdefault("nthread", n_jobs)
    
    # Set defaults
    params.setdefault("verbosity", 0)
    params.setdefault("objective", "reg:squarederror")
    
    return params, num_boost_round


def train_xgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 500,
    obj_fn: Optional[Callable] = None,
    feature_names: Optional[List[str]] = None,
    early_stopping_rounds: Optional[int] = None,
) -> XGBTrainResult:
    """Train XGBoost model with unified interface.
    
    Args:
        X_train: Training features (2D array)
        y_train: Training labels (1D array)
        X_val: Validation features (optional, for predictions)
        y_val: Validation labels (optional, for early stopping)
        params: XGBoost parameters
        num_boost_round: Number of boosting rounds
        obj_fn: Custom objective function (gradient, hessian)
        feature_names: Names for features
        early_stopping_rounds: Early stopping patience (requires X_val, y_val)
        
    Returns:
        XGBTrainResult with trained model and predictions
    """
    # Ensure correct dtypes
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    
    # Filter NaN from training data
    valid_mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    if not np.all(valid_mask):
        n_dropped = int(np.sum(~valid_mask))
        logger.debug(f"Dropping {n_dropped} samples with NaN in training data")
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
    
    if len(y_train) == 0:
        raise ValueError("No valid training samples after NaN filtering")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    
    # Prepare evaluation set
    evals = [(dtrain, 'train')]
    if X_val is not None and y_val is not None:
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        evals.append((dval, 'valid'))
    
    # Prepare params
    params = dict(params or {})
    params.setdefault("verbosity", 0)
    params.setdefault("objective", "reg:squarederror")
    
    # Set base_score=0 when using custom objective
    # XGBoost defaults to 0.5 which is far from typical return values (~0.01)
    # causing tree building to fail as all splits appear unprofitable
    if obj_fn is not None and 'base_score' not in params:
        params['base_score'] = 0.0
    
    # Train
    callbacks = []
    if early_stopping_rounds and len(evals) > 1:
        callbacks.append(xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name='rmse',
            save_best=True
        ))
    
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals if len(evals) > 1 else None,
        obj=obj_fn,
        callbacks=callbacks if callbacks else None,
        verbose_eval=False,
    )
    
    # Get predictions
    if X_val is not None:
        dval = xgb.DMatrix(X_val, feature_names=feature_names)
        predictions = booster.predict(dval).astype(np.float32)
    else:
        predictions = booster.predict(dtrain).astype(np.float32)
    
    # Extract feature importance
    importance_raw = booster.get_score(importance_type='gain')
    importance = {}
    for k, v in importance_raw.items():
        name = k
        # Map fN to actual feature name if available
        if k.startswith("f") and feature_names:
            try:
                idx = int(k[1:])
                if idx < len(feature_names):
                    name = feature_names[idx]
            except ValueError:
                pass
        importance[name] = float(v)
    
    return XGBTrainResult(
        booster=booster,
        predictions=predictions,
        feature_importance=importance,
        n_iterations=booster.num_boosted_rounds(),
    )


def predict_with_model(
    booster: xgb.Booster,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Make predictions with trained XGBoost model.
    
    Args:
        booster: Trained XGBoost booster
        X: Features to predict on
        feature_names: Feature names (must match training)
        
    Returns:
        Predictions as numpy array
    """
    X = np.asarray(X, dtype=np.float32)
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    return booster.predict(dmat).astype(np.float32)


def cv_train_predict(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: Dict[str, Any],
    num_boost_round: int,
    obj_fn: Optional[Callable] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, float]:
    """Train on CV fold and return validation predictions.
    
    This is a convenience wrapper for CV evaluation.
    
    Args:
        X: Full feature matrix
        y: Full label vector
        train_idx: Training indices
        val_idx: Validation indices
        params: XGBoost parameters
        num_boost_round: Number of boosting rounds
        obj_fn: Custom objective function
        feature_names: Feature names
        
    Returns:
        (val_predictions, val_loss_placeholder)
    """
    X_tr = X[train_idx]
    y_tr = y[train_idx]
    X_va = X[val_idx]
    y_va = y[val_idx]
    
    result = train_xgb_model(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        params=params,
        num_boost_round=num_boost_round,
        obj_fn=obj_fn,
        feature_names=feature_names,
    )
    
    return result.predictions, 0.0  # Loss computed externally


class RollingRetrainPredictor:
    """Predictor with optional rolling retrain during evaluation.
    
    Implements the rolling retrain functionality from evaluation config:
    - If retrain=False: Train once on training data, predict all holdout
    - If retrain=True: Retrain at specified cadence using expanding window
    
    Attributes:
        booster: Current XGBoost booster
        params: XGBoost parameters
        num_boost_round: Boosting rounds
        obj_fn: Custom objective function
        feature_names: Feature names
        retrain_cadence: Steps between retraining (0 = every step)
    """
    
    def __init__(
        self,
        params: Dict[str, Any],
        num_boost_round: int = 500,
        obj_fn: Optional[Callable] = None,
        feature_names: Optional[List[str]] = None,
        retrain: bool = False,
        retrain_cadence: int = 0,
    ):
        self.params = params
        self.num_boost_round = num_boost_round
        self.obj_fn = obj_fn
        self.feature_names = feature_names
        self.retrain = retrain
        self.retrain_cadence = retrain_cadence
        self.booster: Optional[xgb.Booster] = None
        self._step = 0
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Initial training on base training set."""
        result = train_xgb_model(
            X_train=X_train,
            y_train=y_train,
            params=self.params,
            num_boost_round=self.num_boost_round,
            obj_fn=self.obj_fn,
            feature_names=self.feature_names,
        )
        self.booster = result.booster
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
    
    def predict_step(
        self,
        X_new: np.ndarray,
        y_realized: Optional[float] = None,
    ) -> float:
        """Predict for single step, optionally retraining.
        
        Args:
            X_new: Features for new observation (1D or 2D)
            y_realized: Realized value from previous step (for expanding training)
            
        Returns:
            Prediction for the step
        """
        if self.booster is None:
            raise RuntimeError("Must call fit() before predict_step()")
        
        X_new = np.atleast_2d(np.asarray(X_new, dtype=np.float32))
        
        # Check if we need to retrain
        needs_retrain = (
            self.retrain and
            self._step > 0 and
            (self.retrain_cadence == 0 or self._step % self.retrain_cadence == 0)
        )
        
        if needs_retrain and y_realized is not None:
            # Expand training set with previous observation
            X_prev = self._last_X if hasattr(self, '_last_X') else None
            if X_prev is not None:
                self._X_train = np.vstack([self._X_train, X_prev])
                self._y_train = np.append(self._y_train, y_realized)
                
                # Retrain
                result = train_xgb_model(
                    X_train=self._X_train,
                    y_train=self._y_train,
                    params=self.params,
                    num_boost_round=self.num_boost_round,
                    obj_fn=self.obj_fn,
                    feature_names=self.feature_names,
                )
                self.booster = result.booster
                logger.debug(f"Retrained at step {self._step} with {len(self._y_train)} samples")
        
        # Make prediction
        prediction = predict_with_model(
            self.booster, X_new, self.feature_names
        )[0]
        
        # Save for potential next retrain
        self._last_X = X_new
        self._step += 1
        
        return float(prediction)
    
    def predict_holdout(
        self,
        X_holdout: np.ndarray,
        y_holdout: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict full holdout set with optional rolling retrain.
        
        Args:
            X_holdout: Holdout features
            y_holdout: Holdout labels (needed for rolling retrain)
            
        Returns:
            Predictions for all holdout observations
        """
        if not self.retrain:
            # Simple case: predict all at once
            return predict_with_model(
                self.booster, X_holdout, self.feature_names
            )
        
        # Rolling retrain case
        predictions = []
        for i in range(len(X_holdout)):
            y_prev = y_holdout[i-1] if i > 0 and y_holdout is not None else None
            pred = self.predict_step(X_holdout[i:i+1], y_prev)
            predictions.append(pred)
        
        return np.array(predictions, dtype=np.float32)

