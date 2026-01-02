"""Custom objective functions for XGBoost training.

This module implements the paper's custom loss function (Section 4.3):
    TotalLoss = λ1·SE + λ2·MENegative + λ3·MEPositive + MSE
"""
from .objective import (
    ObjectiveFunctionParameters,
    compute_quantile_thresholds,
    initialize_objective_params,
    get_quantile_params_from_config,
    loss_components,
    total_loss,
    make_xgb_objective,
)

__all__ = [
    "ObjectiveFunctionParameters",
    "compute_quantile_thresholds",
    "initialize_objective_params",
    "get_quantile_params_from_config",
    "loss_components",
    "total_loss",
    "make_xgb_objective",
]