from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import logging
import warnings
import numpy as np


logger = logging.getLogger(__name__)


def get_quantile_params_from_config(config: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
    """
    Extract quantile_delta and quantile_gamma from config.
        
    param config: Configuration dictionary
    return: Tuple (quantile_delta, quantile_gamma)
    """
    # Default values
    quantile_delta = 0.5
    quantile_gamma = 0.5
    
    if not config:
        return quantile_delta, quantile_gamma
    
    if "objective" in config and config["objective"]:
        obj_cfg = config["objective"]
        if isinstance(obj_cfg, dict):
            if "quantile_delta" in obj_cfg:
                quantile_delta = float(obj_cfg["quantile_delta"])
            if "quantile_gamma" in obj_cfg:
                quantile_gamma = float(obj_cfg["quantile_gamma"])
            return quantile_delta, quantile_gamma
    
    return quantile_delta, quantile_gamma


def compute_quantile_thresholds(
    y_train: np.ndarray,
    quantile_delta: float,
    quantile_gamma: float
) -> Tuple[float, float]:
    """
    Compute delta and gamma thresholds from training labels using quantiles.
    
    param y_train: 1D array of training labels (returns)
    param quantile_delta: Quantile for delta threshold (e.g. 0.5 for median)
    param quantile_gamma: Quantile for gamma threshold (e.g. 0.5 for median)
    return: Tuple (delta, gamma)
        
    Raises:
        ValueError: If y_train is empty after NaN filtering.
    """
    y_train = np.asarray(y_train, dtype=np.float32)
    
    # Filter NaN values
    y_clean = y_train[~np.isnan(y_train)]
    
    if len(y_clean) == 0:
        raise ValueError(
            f"Cannot compute quantile thresholds: all {len(y_train)} training labels are NaN"
        )
    
    delta = float(np.quantile(y_clean, quantile_delta))
    gamma = float(np.quantile(y_clean, quantile_gamma))
    
    logger.debug(
        f"Computed delta/gamma from {len(y_clean)} training samples: "
        f"δ=Q_{quantile_delta:.2f}={delta:.6f}, γ=Q_{quantile_gamma:.2f}={gamma:.6f}"
    )
    
    return delta, gamma


@dataclass
class ObjectiveFunctionParameters:
    """Parameters for the custom objective function."""
    lambda1: float
    lambda2: float
    lambda3: float
    delta: float
    gamma: float
    target_name: Optional[str] = None
    computed_from_quantiles: bool = False


def initialize_objective_params(
    y_train: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    quantile_delta: float,
    quantile_gamma: float,
    config: Optional[dict] = None,
    target: str = "default",
) -> ObjectiveFunctionParameters:
    """ Initialize objective parameters for training objective function."""
    delta = 0.5
    gamma = 0.5
    computed_from_quantiles = False
    
    # Check 1: Config-provided override (new schema: objective.target_overrides)
    if config:
        target_overrides = None
        
        # New schema: objective.target_overrides
        if "objective" in config and config["objective"]:
            obj_cfg = config["objective"]
            if isinstance(obj_cfg, dict) and "target_overrides" in obj_cfg:
                target_overrides = obj_cfg["target_overrides"]
        
        if target_overrides and target in target_overrides:
            override = target_overrides[target]
            if "delta" in override:
                delta = float(override["delta"])
            if "gamma" in override:
                gamma = float(override["gamma"])
            logger.info(
                f"[{target}] Using config override: δ={delta:.6f}, γ={gamma:.6f}"
            )
            return ObjectiveFunctionParameters(
                lambda1=lambda1,
                lambda2=lambda2,
                lambda3=lambda3,
                delta=delta,
                gamma=gamma,
                target_name=target,
                computed_from_quantiles=False,
            )
    
    # Check 2: Auto-compute from training labels
    try:
        y_train_clean = np.asarray(y_train, dtype=np.float32)
        if len(y_train_clean[~np.isnan(y_train_clean)]) > 0:
            delta, gamma = compute_quantile_thresholds(
                y_train_clean, quantile_delta, quantile_gamma
            )
            logger.info(
                f"[{target}] Auto-computed delta/gamma from quantiles: "
                f"δ=Q_{quantile_delta}={delta:.6f}, γ=Q_{quantile_gamma}={gamma:.6f}"
            )
            computed_from_quantiles = True
        else:
            raise ValueError("No valid training labels")
    except (ValueError, IndexError) as e:
        # Fallback to defaults
        logger.warning(
            f"[{target}] Could not compute delta/gamma ({str(e)}). "
            f"Using hardcoded defaults: δ={delta:.6f}, γ={gamma:.6f}. "
            f"Recommended: provide training data or set objective.target_overrides in config."
        )
    
    return ObjectiveFunctionParameters(
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        delta=delta,
        gamma=gamma,
        target_name=target,
        computed_from_quantiles=computed_from_quantiles,
    )


def _weights_for_y(y: np.ndarray, p: ObjectiveFunctionParameters) -> np.ndarray:
    """
    Compute loss weights for each sample based on 4-component decomposition.
    
    The loss decomposes into 4 components:
    1. Base MSE: all samples get weight λ₁ (baseline magnitude error)
    2. Positive Magnitude Error (ME_pos): y ≥ γ gets additional +λ₂
    3. Negative Magnitude Error (ME_neg): y ≤ δ gets additional +λ₃
    4. Sign Error (SE): implicitly captured via direction in weighted squared error
    
    Final weight for sample i:
    w_i = λ₁ + λ₂·[y_i ≥ γ] + λ₃·[y_i ≤ δ]
    
    where [·] is Iverson bracket (1 if true, 0 if false).
    
    param y: 1D array of true labels (returns)
    param p: ObjectiveFunctionParameters with λ₁, λ₂, λ₃, δ, γ
    
    return: 1D array of weights w_i
    """
    w = np.full_like(y, fill_value=p.lambda1, dtype=np.float32)
    y = y.astype(np.float32, copy=False)
    # Add lambda2 for large positive returns (y >= gamma)
    # Add lambda3 for large negative returns (y <= delta)
    w = w + (y >= p.gamma) * np.float32(p.lambda2) + (y <= p.delta) * np.float32(p.lambda3)
    return w


def loss_components(y_true: np.ndarray, y_pred: np.ndarray, p: ObjectiveFunctionParameters) -> dict:
    """Compute the paper's *exact* loss components.
    The total loss is decomposed into 4 components as follows:

      TotalLoss = λ1·SE + λ2·MENegative + λ3·MEPositive + MSE

    where:
      SE = (1/n) Σ 1[ sign(y_i) != sign(ŷ_i) ]
      MENegative = (1/n) Σ (ŷ_i - y_i)^2 · 1[y_i <= δ]
      MEPositive = (1/n) Σ (ŷ_i - y_i)^2 · 1[y_i >= γ]
      MSE = (1/n) Σ (ŷ_i - y_i)^2
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"n": 0, "se": np.nan, "me_neg": np.nan, "me_pos": np.nan, "mse": np.nan, "total": np.nan}

    yt = y_true[mask]
    yp = y_pred[mask]
    err2 = (yp - yt) ** 2

    # SE: sign mismatch. Treat 0 as 0 sign (np.sign(0)=0) to match indicator definition.
    se = np.mean((np.sign(yt) != np.sign(yp)).astype(np.float32))

    me_neg = np.mean(err2 * (yt <= np.float32(p.delta)).astype(np.float32))
    me_pos = np.mean(err2 * (yt >= np.float32(p.gamma)).astype(np.float32))
    mse = np.mean(err2)

    total = float(np.float32(p.lambda1) * se + np.float32(p.lambda2) * me_neg + np.float32(p.lambda3) * me_pos + mse)
    return {
        "n": int(len(yt)),
        "se": float(se),
        "me_neg": float(me_neg),
        "me_pos": float(me_pos),
        "mse": float(mse),
        "total": total,
    }


def total_loss(y_true: np.ndarray, y_pred: np.ndarray, p: ObjectiveFunctionParameters) -> float:
    """Total loss value only."""
    return float(loss_components(y_true, y_pred, p)["total"])


def make_xgb_objective(p: ObjectiveFunctionParameters):
    """
    Create XGBoost-compatible objective function with gradients and Hessians.
    param p: ObjectiveFunctionParameters
    return: Function (preds, dtrain) -> (grad, hess)
    """
    def _obj(preds, dtrain):
        y = dtrain.get_label().astype(np.float32, copy=False)
        preds = preds.astype(np.float32, copy=False)
        w = _weights_for_y(y, p)
        grad = 2.0 * w * (preds - y)
        hess = 2.0 * w
        return grad, hess
    return _obj

