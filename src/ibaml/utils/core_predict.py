import numpy as np
import xgboost as xgb
from typing import Dict, Tuple, Union, Optional, List


def xgb_core_predict(
    Xtr: np.ndarray, 
    ytr: np.ndarray, 
    Xva: np.ndarray, 
    obj_fn, 
    xgb_params: Dict,
    return_booster: bool = False,
    feature_names: Optional[List[str]] = None,
) -> Union[np.ndarray, Tuple[xgb.Booster, np.ndarray]]:
    """Perform XGBoost prediction with given training and validation data.
    
    Args:
        Xtr: Training features
        ytr: Training labels
        Xva: Validation features
        obj_fn: Custom objective function (None for default)
        xgb_params: XGBoost parameters dict
        return_booster: If True, return (booster, predictions) tuple
        feature_names: Optional feature names for DMatrix
        
    Returns:
        predictions array if return_booster=False, else (booster, predictions) tuple
    """
    params = dict(xgb_params or {})
    num_boost_round = int(params.pop("n_estimators", 500))
    n_jobs = int(params.pop("n_jobs", 1))
    params.setdefault("nthread", n_jobs)
    params.setdefault("verbosity", 0)
    params.setdefault("objective", "reg:squarederror")
    # Set base_score=0 when using custom objective
    # XGBoost defaults to 0.5 which is far from typical return values (~0.01)
    # causing tree building to fail as all splits appear unprofitable
    if obj_fn is not None and 'base_score' not in params:
        params['base_score'] = 0.0
    
    dtr = xgb.DMatrix(
        Xtr.astype("float32", copy=False), 
        label=ytr.astype("float32", copy=False),
        feature_names=feature_names,
    )
    dva = xgb.DMatrix(Xva.astype("float32", copy=False), feature_names=feature_names)
    booster = xgb.train(params, dtr, num_boost_round=num_boost_round, obj=obj_fn)
    predictions = booster.predict(dva).astype("float32", copy=False)
    
    if return_booster:
        return booster, predictions
    return predictions

