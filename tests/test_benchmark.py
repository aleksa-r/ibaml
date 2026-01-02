"""
Unit tests for the PCA benchmark module.

Tests verify:
1. PCA transformation works correctly
2. Benchmark result data structure
3. Integration with existing pipeline components
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import benchmark components
from ibaml.benchmarks.pca import (
    PCABenchmarkResult,
    build_exogenous_features,
    apply_pca_transform,
)


class TestPCABenchmarkResult:
    """Test PCABenchmarkResult data class."""
    
    def test_default_values(self):
        """Test default values for result container."""
        result = PCABenchmarkResult(target="CTA")
        
        assert result.target == "CTA"
        assert result.skipped is False
        assert result.n_components == 6
        assert result.cv_loss == 0.0
    
    def test_skipped_result(self):
        """Test skipped result to_dict output."""
        result = PCABenchmarkResult(
            target="CTA",
            skipped=True,
            skip_reason="insufficient data"
        )
        
        d = result.to_dict()
        assert d["target"] == "CTA"
        assert d["skipped"] is True
        assert d["reason"] == "insufficient data"
    
    def test_full_result_to_dict(self):
        """Test full result serialization."""
        result = PCABenchmarkResult(
            target="CTA",
            n_components=6,
            explained_variance_ratio=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05],
            train_size=200,
            holdout_size=18,
            delta=-0.01,
            gamma=0.03,
            cv_loss=0.15,
        )
        
        d = result.to_dict()
        assert d["target"] == "CTA"
        assert d["n_components"] == 6
        assert len(d["explained_variance_ratio"]) == 6
        assert d["train_size"] == 200
        assert d["holdout_size"] == 18


class TestApplyPCATransform:
    """Test PCA transformation function."""
    
    def test_basic_pca_transform(self):
        """Test basic PCA transformation."""
        np.random.seed(42)
        X_train = np.random.randn(100, 20).astype(np.float32)
        X_test = np.random.randn(18, 20).astype(np.float32)
        
        X_train_pca, X_test_pca, pca, var_ratio = apply_pca_transform(
            X_train, X_test, n_components=6
        )
        
        # Check shapes
        assert X_train_pca.shape == (100, 6)
        assert X_test_pca.shape == (18, 6)
        
        # Check variance ratio
        assert len(var_ratio) == 6
        assert sum(var_ratio) > 0
        assert all(v >= 0 for v in var_ratio)
    
    def test_pca_with_fewer_features(self):
        """Test PCA when features < requested components."""
        np.random.seed(42)
        X_train = np.random.randn(100, 4).astype(np.float32)  # Only 4 features
        X_test = np.random.randn(18, 4).astype(np.float32)
        
        X_train_pca, X_test_pca, pca, var_ratio = apply_pca_transform(
            X_train, X_test, n_components=6  # Request 6, but only 4 available
        )
        
        # Should get at most 4 components
        assert X_train_pca.shape[1] <= 4
        assert X_test_pca.shape[1] <= 4
    
    def test_pca_output_dtype(self):
        """Test that PCA output is float32."""
        np.random.seed(42)
        X_train = np.random.randn(50, 10).astype(np.float64)  # Input as float64
        X_test = np.random.randn(10, 10).astype(np.float64)
        
        X_train_pca, X_test_pca, _, _ = apply_pca_transform(
            X_train, X_test, n_components=5
        )
        
        # Output should be float32
        assert X_train_pca.dtype == np.float32
        assert X_test_pca.dtype == np.float32


class TestBuildExogenousFeatures:
    """Test exogenous feature building."""
    
    def test_empty_factors_map(self):
        """Test with empty factors map."""
        dates = pd.date_range("2020-01-31", periods=36, freq="ME")
        df_factors = pd.DataFrame(
            np.random.randn(36, 4),
            index=dates,
            columns=["A", "B", "C", "D"]
        )
        
        # Empty map should return empty DataFrame
        result = build_exogenous_features(df_factors, {}, [6, 12])
        assert result.shape[1] == 0
    
    def test_single_group(self):
        """Test with single factor group."""
        dates = pd.date_range("2020-01-31", periods=36, freq="ME")
        df_factors = pd.DataFrame(
            np.random.randn(36, 2),
            index=dates,
            columns=["F1", "F2"]
        )
        
        factors_map = {"G1": ["F1", "F2"]}
        result = build_exogenous_features(df_factors, factors_map, [6, 12])
        
        # Should have z-score columns for each window
        assert result.shape[0] == 36
        assert result.shape[1] >= 2  # At least 2 features


class TestBenchmarkIntegration:
    """Integration tests for benchmark module."""
    
    def test_benchmark_module_imports(self):
        """Test that all benchmark components can be imported."""
        from ibaml.benchmarks.pca import (
            PCABenchmarkResult,
            build_exogenous_features,
            apply_pca_transform,
            run_pca_benchmark,
            write_benchmark_results,
            main,
        )
        
        # All imports should succeed
        assert PCABenchmarkResult is not None
        assert build_exogenous_features is not None
        assert apply_pca_transform is not None
        assert run_pca_benchmark is not None
        assert write_benchmark_results is not None
        assert main is not None
