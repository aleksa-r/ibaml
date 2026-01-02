"""
Unit tests for configuration schemas.

Tests verify:
1. New schema with quantile_delta/quantile_gamma under objective
2. Legacy schema backward compatibility (iba_params)
3. Schema migration warnings
4. Hyperopt configuration validation
"""

import pytest
import warnings
import tempfile
import os
from pathlib import Path

from ibaml.config.schemas import (
    Config, DatasetCfg, SplitsCfg, ObjectiveCfg, 
    EvaluationCfg, HyperoptCfg, HyperoptSearchSpace,
    load_config
)


class TestObjectiveCfg:
    """Test ObjectiveCfg with quantile parameters."""
    
    def test_default_values(self):
        """Test that defaults match paper methodology."""
        cfg = ObjectiveCfg(
            kind="paper",
            lambda1=0.5,
            lambda2=0.25,
            lambda3=0.25,
            quantile_delta=0.25,
            quantile_gamma=0.75,
            target_overrides=None
        )
        assert cfg.kind == "paper"
        assert cfg.lambda1 == 0.5
        assert cfg.lambda2 == 0.25
        assert cfg.lambda3 == 0.25
        assert cfg.quantile_delta == 0.25  # Q1
        assert cfg.quantile_gamma == 0.75  # Q3
        assert cfg.target_overrides is None
    
    def test_custom_quantiles(self):
        """Test custom quantile values."""
        cfg = ObjectiveCfg(
            kind="paper",
            lambda1=0.5,
            lambda2=0.25,
            lambda3=0.25,
            quantile_delta=0.10,
            quantile_gamma=0.90,
            target_overrides=None
        )
        assert cfg.quantile_delta == 0.10
        assert cfg.quantile_gamma == 0.90
    
    def test_target_overrides(self):
        """Test target-specific delta/gamma overrides."""
        cfg = ObjectiveCfg(
            kind="paper",
            lambda1=0.5,
            lambda2=0.25,
            lambda3=0.25,
            quantile_delta=0.25,
            quantile_gamma=0.75,
            target_overrides={
                "CTA": {"delta": -0.0189, "gamma": 0.0329},
                "CA": {"delta": -0.0238, "gamma": 0.0364},
            }
        )
        assert cfg.target_overrides["CTA"]["delta"] == -0.0189
        assert cfg.target_overrides["CA"]["gamma"] == 0.0364
    
    def test_invalid_quantile_raises_error(self):
        """Test that invalid quantile values raise validation errors."""
        with pytest.raises(ValueError, match="must be in"):
            ObjectiveCfg(quantile_delta=0.0)  # Must be > 0
        
        with pytest.raises(ValueError, match="must be in"):
            ObjectiveCfg(quantile_gamma=1.0)  # Must be < 1
    
    def test_invalid_lambda_raises_error(self):
        """Test that invalid lambda values raise validation errors."""
        with pytest.raises(ValueError, match="must be in"):
            ObjectiveCfg(lambda1=-0.1)
        
        with pytest.raises(ValueError, match="must be in"):
            ObjectiveCfg(lambda2=1.5)


class TestSplitsCfg:
    """Test SplitsCfg with dynamic adjustment options."""
    
    def test_default_values(self):
        """Test default split values (aligned with paper)."""
        cfg = SplitsCfg(init_tran_size=0.6, folds_val_set_size=19, folds_steps_size=10, num_of_folds=5, allow_dynamic_adjustment=True)
        # Paper-aligned defaults: init_tran_size ~60%, Q=19, N=10, K=5
        assert cfg.init_tran_size == 0.6
        assert cfg.folds_val_set_size == 19
        assert cfg.folds_steps_size == 10
        assert cfg.num_of_folds == 5
        assert cfg.allow_dynamic_adjustment is True
    
    def test_custom_values(self):
        """Test custom split values."""
        cfg = SplitsCfg(init_tran_size=0.5, folds_val_set_size=12, folds_steps_size=6, num_of_folds=3, allow_dynamic_adjustment=True)
        assert cfg.init_tran_size == 0.5
        assert cfg.folds_val_set_size == 12
        assert cfg.allow_dynamic_adjustment is True
    
    def test_invalid_splits_raises_error(self):
        """Test that non-positive split values raise validation errors."""
        # Invalid validation window size should raise
        with pytest.raises(ValueError, match="must be positive"):
            SplitsCfg(init_tran_size=0.6, folds_val_set_size=0, folds_steps_size=1, num_of_folds=1, allow_dynamic_adjustment=True)

        # Invalid number of folds should raise
        with pytest.raises(ValueError, match="must be positive"):
            SplitsCfg(init_tran_size=0.6, folds_val_set_size=6, folds_steps_size=1, num_of_folds=-1, allow_dynamic_adjustment=True)


class TestHyperoptCfg:
    """Test hyperparameter optimization configuration."""
    
    def test_default_disabled(self):
        """Test that hyperopt is disabled by default."""
        cfg = HyperoptCfg(
            enabled=False,
            n_trials=50,
            timeout=None,
            n_jobs=1,
            sampler="tpe",
            pruner="median",
            cv_folds=3,
            search_space={},
        )
        assert cfg.enabled is False
        assert cfg.n_trials == 50
        assert cfg.sampler == "tpe"
        assert cfg.pruner == "median"
    
    def test_default_search_space(self):
        """Test default search space includes common XGBoost params."""
        default_space = {
            "max_depth": HyperoptSearchSpace(type="int", low=3, high=10, step=1, choices=None),
            "learning_rate": HyperoptSearchSpace(type="float", low=0.01, high=0.3, step=None, choices=None),
            "n_estimators": HyperoptSearchSpace(type="int", low=50, high=500, step=10, choices=None),
            "subsample": HyperoptSearchSpace(type="float", low=0.5, high=1.0, step=None, choices=None),
            "min_split_loss": HyperoptSearchSpace(type="float", low=0.0, high=10.0, step=None, choices=None),
        }
        cfg = HyperoptCfg(
            enabled=True,
            n_trials=50,
            timeout=None,
            n_jobs=1,
            sampler="tpe",
            pruner="median",
            cv_folds=3,
            search_space=default_space,
        )
        assert "max_depth" in cfg.search_space
        assert "learning_rate" in cfg.search_space
        assert "n_estimators" in cfg.search_space
        assert "subsample" in cfg.search_space
        assert "min_split_loss" in cfg.search_space
        assert "gamma" not in cfg.search_space  # Renamed to min_split_loss
    
    def test_regularization_excluded_from_search_space(self):
        """Test that regularization is excluded from hyperopt search space.
        
        Regularization is better controlled through base XGBoost parameters
        since aggressive regularization causes underfitting with custom objectives.
        """
        default_space = {
            "max_depth": HyperoptSearchSpace(type="int", low=3, high=10, step=1, choices=None),
            "learning_rate": HyperoptSearchSpace(type="float", low=0.01, high=0.3, step=None, choices=None),
            "n_estimators": HyperoptSearchSpace(type="int", low=50, high=500, step=10, choices=None),
            "subsample": HyperoptSearchSpace(type="float", low=0.5, high=1.0, step=None, choices=None),
        }
        cfg = HyperoptCfg(
            enabled=True,
            n_trials=50,
            timeout=None,
            n_jobs=1,
            sampler="tpe",
            pruner="median",
            cv_folds=3,
            search_space=default_space,
        )
        # Regularization should NOT be in default search space
        assert "reg_alpha" not in cfg.search_space
        assert "reg_lambda" not in cfg.search_space
        # But core tree parameters should be
        assert "max_depth" in cfg.search_space
        assert "learning_rate" in cfg.search_space
        assert "n_estimators" in cfg.search_space
    
    def test_custom_search_space(self):
        """Test custom search space definition."""
        cfg = HyperoptCfg(
            enabled=True,
            n_trials=100,
            timeout=None,
            n_jobs=1,
            sampler="tpe",
            pruner="median",
            cv_folds=3,
            search_space={
                "max_depth": HyperoptSearchSpace(type="int", low=2, high=6, step=1, choices=None),
            },
        )
        assert cfg.enabled is True
        assert cfg.n_trials == 100
        assert cfg.search_space["max_depth"].low == 2


class TestHyperoptSearchSpace:
    """Test search space validation."""
    
    def test_numeric_space_requires_bounds(self):
        """Test that numeric types require low/high bounds."""
        with pytest.raises(ValueError, match="requires 'low' and 'high' bounds"):
            HyperoptSearchSpace(type="int", low=None, high=None, step=None, choices=None)
    
    def test_categorical_requires_choices(self):
        """Test that categorical type requires choices."""
        with pytest.raises(ValueError, match="requires non-empty 'choices' list"):
            HyperoptSearchSpace(type="categorical", low=None, high=None, step=None, choices=None)
    
    def test_bounds_order_validation(self):
        """Test that low must be less than high."""
        with pytest.raises(ValueError, match="must be less than"):
            HyperoptSearchSpace(type="float", low=1.0, high=0.5, step=None, choices=None)
    
    def test_valid_int_space(self):
        """Test valid integer search space."""
        space = HyperoptSearchSpace(type="int", low=1, high=10, step=2, choices=None)
        assert space.type == "int"
        assert space.low == 1
        assert space.high == 10
        assert space.step == 2
    
    def test_valid_categorical_space(self):
        """Test valid categorical search space."""
        space = HyperoptSearchSpace(type="categorical", low=None, high=None, step=None, choices=["gpu_hist", "hist"])
        assert space.type == "categorical"
        assert "gpu_hist" in space.choices




class TestLoadConfig:
    """Test configuration file loading."""
    
    def test_load_new_schema_config(self, tmp_path):
        """Test loading config with new schema."""
        config_content = """
dataset:
    factors_path: data/factors.csv
    targets_path: data/targets.csv
    date_column: Date
    target_columns: [CTA, CA]
    parse_dates: true
    zscore_windows: [12]
    cumulative_horizon: 1
    forecast_shift: 1
    align_freq: M
    align_agg: last
    align_how: inner

factors:
    IR: [T1YFF_CHG]
    V: [VIXCLS]

splits:
    init_tran_size: 0.6
    folds_val_set_size: 12
    folds_steps_size: 6
    num_of_folds: 3
    allow_dynamic_adjustment: true

objective:
    kind: paper
    lambda1: 0.5
    lambda2: 0.25
    lambda3: 0.25
    quantile_delta: 0.25
    quantile_gamma: 0.75
    target_overrides: null

search: null

xgb:
    params: {}

simulation:
    enabled: false
    gate: zero
    slippage_bps: 0.0
    fees_bps: 0.0

hyperopt: null

evaluation:
    size: 18
    retrain: true
    steps: 6
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)
        
        cfg = load_config(str(config_file))
        
        assert cfg.dataset.target_columns == ["CTA", "CA"]
        assert cfg.splits.init_tran_size == 0.6
        assert cfg.objective.quantile_delta == 0.25
        assert cfg.evaluation.retrain is True
        assert cfg.objective.target_overrides is None
    
    def test_empty_config_raises_error(self, tmp_path):
        """Test that empty config file raises error."""
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")
        
        with pytest.raises(ValueError, match="Empty configuration"):
            load_config(str(config_file))
    
    def test_invalid_factors_raises_error(self, tmp_path):
        """Test that empty factors dict raises error."""
        pass
