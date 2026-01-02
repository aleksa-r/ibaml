from __future__ import annotations
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, field_validator, model_validator
import yaml


class DatasetCfg(BaseModel):
    """Dataset configuration for loading and preprocessing factor/target data.
    
    Attributes:
        factors_path: Path to CSV containing exogenous factor data
        targets_path: Path to CSV containing Target return data
        date_column: Name of date column in both CSVs
        target_columns: List of target names to analyze
        parse_dates: Whether to parse date column as datetime
        align_freq: Frequency for resampling ('M' for monthly)
        align_agg: Aggregation method for resampling
        align_how: How to align factor and target indices
        zscore_windows: Rolling window sizes for z-score normalization
        cumulative_horizon: Number of periods for cumulative returns
        forecast_shift: Number of periods to shift target for forecasting
    """
    factors_path: str
    targets_path: str
    date_column: str
    target_columns: List[str]
    parse_dates: bool
    align_freq: str
    align_agg: Literal["last", "mean", "sum"]
    align_how: Literal["inner", "left", "right", "end"]
    zscore_windows: List[int]
    cumulative_horizon: int
    forecast_shift: int


class SplitsCfg(BaseModel):
    """Cross-validation split configuration (EWCV-EFS parameters).

    New naming and semantics:
      - `init_tran_size` (float in (0,1)): fraction of available CV data used
        as the initial training window when computing dynamic EWCV splits.
        Example: 0.6 means the algorithm will attempt to set the initial
        training window to ~60% of the CV sample when computing M/N/K.
      - `folds_val_set_size` (int): validation window size per fold (previously `Q`).
      - `folds_steps_size` (int): step size between folds (previously `N`).
      - `num_of_folds` (int): number of CV folds (previously `K`).

    Backwards compatibility note: callers that used absolute `M` should now
    pass either an absolute integer or set `init_tran_size` in the config.

    Attributes:
        init_tran_size: Initial training window fraction (0 < fraction < 1).
        folds_val_set_size: Validation window size per fold (months).
        folds_steps_size: Step size for expanding training window between folds (months).
        num_of_folds: Number of CV folds.
        allow_dynamic_adjustment: If True, allows automatic reduction of folds
                                   or window sizes when data is insufficient.
    """
    init_tran_size: float = 0.6
    folds_val_set_size: int
    folds_steps_size: int
    num_of_folds: int
    allow_dynamic_adjustment: bool

    # No legacy migration: configuration must use new split keys explicitly.

    @field_validator('folds_val_set_size', 'folds_steps_size', 'num_of_folds')
    @classmethod
    def positive_int(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @field_validator('init_tran_size')
    @classmethod
    def valid_ratio(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"init_tran_size must be in (0,1), got {v}")
        return v

    # No legacy attribute properties: use `init_tran_size`, `folds_val_set_size`,
    # `folds_steps_size`, and `num_of_folds` directly.


class ObjectiveCfg(BaseModel):
    """Custom objective function configuration.
    
    The loss function is composed of four components:
        Total Loss = λ1·SE + λ2·ME_negative + λ3·ME_positive + MSE
    
    Where:
        SE = Sign Error (fraction of predictions with wrong sign)
        ME_negative = Magnitude error for y <= δ (large negative returns)
        ME_positive = Magnitude error for y >= γ (large positive returns)
        MSE = Mean Squared Error (baseline)
    
    Attributes:
        kind: Objective type ('paper' for the custom objective)
        lambda1: Weight for sign error component
        lambda2: Weight for negative magnitude error component  
        lambda3: Weight for positive magnitude error component
        quantile_delta: Quantile level for δ threshold (default 0.3 = Q1)
        quantile_gamma: Quantile level for γ threshold (default 0.7 = Q3)
        target_overrides: Optional dict mapping target -> {delta, gamma} for fixed values
    """
    kind: Literal["paper", "mse"]
    lambda1: float
    lambda2: float
    lambda3: float
    quantile_delta: float
    quantile_gamma: float
    # Optional target-specific overrides for delta/gamma
    target_overrides: Optional[Dict[str, Dict[str, float]]]
    
    @field_validator('lambda1', 'lambda2', 'lambda3')
    @classmethod
    def valid_weight(cls, v: float, info) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"{info.field_name} must be in [0, 1], got {v}")
        return v
    
    @field_validator('quantile_delta', 'quantile_gamma')
    @classmethod
    def valid_quantile(cls, v: float, info) -> float:
        if v <= 0 or v >= 1:
            raise ValueError(f"{info.field_name} must be in (0, 1), got {v}")
        return v


class XGBParamsCfg(BaseModel):
    """XGBoost model parameters.
    
    Attributes:
        params: Dictionary of XGBoost parameters passed to xgb.train()
    """
    params: Dict[str, Any]


class SimulationCfg(BaseModel):
    """Trading simulation configuration for backtesting.
    
    Attributes:
        enabled: Whether to run trading simulation
        gate: Signal gating target ('zero' = trade if pred >= 0, 'delta' = trade if pred >= δ)
        slippage_bps: Slippage in basis points
        fees_bps: Trading fees in basis points
    """
    enabled: bool
    gate: Literal["zero", "delta"]
    slippage_bps: float
    fees_bps: float


class EvaluationCfg(BaseModel):
    """Holdout evaluation configuration.
    
    The holdout set is extracted FIRST from the end of the data series.
    The remaining data is used for training and cross-validation.
    
    Attributes:
        size: Number of observations in holdout set (extracted from end of series)
        retrain: Whether to use rolling retrain during holdout evaluation
        steps: Retrain cadence (0 = retrain every step when retrain=True)
    """
    size: int
    retrain: bool
    steps: int
    
    @field_validator('size')
    @classmethod
    def positive_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"evaluation.size must be positive, got {v}")
        return v


class SearchParamsCfg(BaseModel):
    """Feature search pipeline parameters for SF and MF search.
    
    Attributes:
        top_k: Number of top feature masks to keep per group in single-factor search
        limit_mask_size: Maximum number of features in a mask (None = unlimited)
        min_groups_in_combo: Minimum groups in multi-factor combination
        max_groups_in_combo: Maximum groups in multi-factor combination (None = all)
        allow_empty_groups: Allow combinations with some groups having no features
    """
    top_k: int
    limit_mask_size: Optional[int]
    min_groups_in_combo: int
    max_groups_in_combo: Optional[int]
    allow_empty_groups: bool


class HyperoptSearchSpace(BaseModel):
    """Search space definition for a single hyperparameter.
    
    Attributes:
        type: Distribution type ('int', 'float', 'log_float', 'categorical')
        low: Lower bound for numeric types
        high: Upper bound for numeric types
        choices: List of choices for categorical type
        step: Step size for int type (optional)
    """
    type: Literal["int", "float", "log_float", "categorical"]
    low: Optional[float]
    high: Optional[float]
    choices: Optional[List[Any]]
    step: Optional[int]
    
    @model_validator(mode='after')
    def validate_space(self):
        if self.type in ("int", "float", "log_float"):
            if self.low is None or self.high is None:
                raise ValueError(f"type '{self.type}' requires 'low' and 'high' bounds")
            if self.low >= self.high:
                raise ValueError(f"'low' ({self.low}) must be less than 'high' ({self.high})")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError("type 'categorical' requires non-empty 'choices' list")
        return self


class HyperoptCfg(BaseModel):
    """Hyperparameter optimization configuration using Optuna.
    
    When enabled, runs Bayesian optimization to find optimal XGBoost parameters
    for the best multi-factor model. Only runs on the final MF model, not during
    the exhaustive feature search (which uses default parameters for speed).
    
    Attributes:
        enabled: Whether to enable hyperparameter optimization
        n_trials: Number of Optuna trials to run
        timeout: Maximum time in seconds (None = no limit)
        n_jobs: Number of parallel jobs for Optuna (-1 = all cores)
        sampler: Optuna sampler type ('tpe', 'random', 'cmaes')
        pruner: Optuna pruner type ('median', 'hyperband', 'none')
        search_space: Dictionary mapping parameter names to search space definitions
        cv_folds: Number of CV folds for hyperopt evaluation (uses same EWCV logic)
    """
    enabled: bool
    n_trials: int
    timeout: Optional[int]
    n_jobs: int
    sampler: Literal["tpe", "random", "cmaes"]
    pruner: Literal["median", "hyperband", "none"]
    cv_folds: int
    search_space: Dict[str, HyperoptSearchSpace]
    # Minimum fraction of data to use for the initial training window
    # when creating time-series CV splits for hyperopt (0 < ratio < 1)
    min_train_ratio: float = 0.5
    # Whether to apply per-fold MinMax scaling for IBA features during hyperopt
    scale_iba_features: bool = True

    @field_validator('min_train_ratio')
    @classmethod
    def valid_min_train_ratio(cls, v: float) -> float:
        if v <= 0.0 or v >= 1.0:
            raise ValueError(f"min_train_ratio must be in (0,1), got {v}")
        return v


class Config(BaseModel):
    """Root configuration model for IBAML pipeline."""
    dataset: DatasetCfg
    factors: Dict[str, List[str]]
    splits: SplitsCfg
    objective: ObjectiveCfg
    search: Optional[SearchParamsCfg]
    xgb: Optional[XGBParamsCfg]
    simulation: Optional[SimulationCfg]
    evaluation: Optional[EvaluationCfg]
    hyperopt: Optional[HyperoptCfg]
    
    @field_validator('factors')
    @classmethod
    def validate_factors(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        if not v:
            raise ValueError("factors dict cannot be empty")
        for group, cols in v.items():
            if not cols:
                raise ValueError(f"Factor group '{group}' has no columns")
        return v


def load_config(path: str) -> Config:
    """Load and validate configuration from YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Config object with full validation
        
    Raises:
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If fields fail schema validation
        FileNotFoundError: If config file doesn't exist
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    if raw is None:
        raise ValueError(f"Empty configuration file: {path}")
    
    if 'dataset' in raw and isinstance(raw.get('dataset'), dict):
        if 'dataset' in raw['dataset']:
            # Nested dataset - flatten it
            raw['dataset'] = raw['dataset']['dataset']
    
    return Config.model_validate(raw)

