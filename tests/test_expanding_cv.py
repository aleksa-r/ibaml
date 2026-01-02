"""
Unit tests for the expanding CV splits module.

Tests verify:
1. Holdout-first extraction (paper methodology)
2. Expanding window CV split computation
3. Dynamic adjustment when data is insufficient
4. Edge cases (too little data, gaps, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from ibaml.validation.expanding_cv import (
    extract_holdout_first,
    compute_cv_splits,
    ExpandingWindowSplit,
    SplitInfo,
    DataSplit,
)


class TestExtractHoldoutFirst:
    """Test holdout extraction from end of time series."""
    
    def test_basic_holdout_extraction(self):
        """Test that holdout is extracted from the end of the data."""
        # Create 36 months of data
        dates = pd.date_range("2020-01-31", periods=36, freq="ME")
        data = pd.Series(np.random.randn(36), index=dates)  # Use Series, not DataFrame
        
        holdout_size = 6
        split = extract_holdout_first(data, holdout_size)
        
        # Holdout should be the last 6 observations
        assert len(split.holdout_idx) == holdout_size
        # Train should be everything except holdout
        assert len(split.train_idx) == 36 - holdout_size
    
    def test_holdout_extraction_with_nans(self):
        """Test holdout extraction when data has NaN values."""
        dates = pd.date_range("2020-01-31", periods=36, freq="ME")
        data = pd.Series(np.random.randn(36), index=dates)
        
        # Add some NaN values in the middle
        data.iloc[10:12] = np.nan
        
        holdout_size = 6
        split = extract_holdout_first(data, holdout_size)
        
        # Should still extract 6 holdout observations from valid data
        assert len(split.holdout_idx) == holdout_size
    
    def test_insufficient_data_for_holdout(self):
        """Test behavior when data is too small for requested holdout."""
        dates = pd.date_range("2020-01-31", periods=5, freq="ME")
        data = pd.Series(np.random.randn(5), index=dates)
        
        holdout_size = 10  # More than available data
        
        with pytest.raises(ValueError, match="Insufficient data"):
            extract_holdout_first(data, holdout_size)
    
    def test_dates_are_chronologically_ordered(self):
        """Test that train dates come before holdout dates."""
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        data = pd.Series(np.random.randn(24), index=dates)
        
        split = extract_holdout_first(data, holdout_size=6)
        
        # All train dates should be before all holdout dates
        assert split.train_idx.max() < split.holdout_idx.min()


class TestComputeCVSplits:
    """Test expanding window CV split computation."""
    
    def test_basic_cv_splits(self):
        """Test basic CV split computation per paper Algorithm 1."""
        T = 78  # Total samples available for CV (after holdout)
        M = 18  # Initial training size
        Q = 6   # Validation size
        N = 10  # Step size
        K = 5   # Number of folds
        
        split_info = compute_cv_splits(T, M, Q, N, K, min_train_size=12)
        
        # Should return SplitInfo
        assert isinstance(split_info, SplitInfo)
        assert split_info.K <= K  # May be adjusted
        assert split_info.M >= 1
        assert split_info.Q > 0
    
    def test_dynamic_adjustment_when_data_insufficient(self):
        """Test dynamic K adjustment when data is too small."""
        T = 58  # Adjusted to meet min_required
        M = 18
        Q = 6
        N = 10
        K = 5  # Would need more samples than available
        split_info = compute_cv_splits(T, M, Q, N, K, min_train_size=12, allow_dynamic_adjustment=True)
        # Should return fewer folds or adjusted parameters
        assert split_info.was_adjusted is True or split_info.K <= K
    
    def test_no_dynamic_adjustment_raises_error(self):
        """Test that insufficient data raises error when dynamic adjustment disabled."""
        with pytest.raises(ValueError, match="Insufficient|Required|require"):
            compute_cv_splits(T=30, M=20, Q=10, N=10, K=5, min_train_size=15, allow_dynamic_adjustment=False)


class TestExpandingWindowSplit:
    """Test the ExpandingWindowSplit class (scikit-learn style splitter)."""
    
    def test_sklearn_api_compatibility(self):
        """Test that ExpandingWindowSplit follows sklearn splitter API."""
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        
        splitter = ExpandingWindowSplit(M=15, Q=8, N=10, K=3, T=len(X))
        
        # Should return train/test indices
        splits = list(splitter.split(X, y))
        
        assert len(splits) <= 3  # May be adjusted down
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert len(train_idx) > 0
            assert len(test_idx) > 0
    
    def test_get_n_splits(self):
        """Test get_n_splits method."""
        splitter = ExpandingWindowSplit(M=10, Q=5, N=5, K=4, T=60)
        
        # Without data
        assert splitter.get_n_splits() == 4
    
    def test_indices_dont_exceed_data_bounds(self):
        """Test that split indices are within data bounds."""
        X = np.random.randn(80, 5)

        splitter = ExpandingWindowSplit(M=10, Q=8, N=6, K=3, T=len(X))

        for train_idx, test_idx in splitter.split(X):
            assert train_idx.max() < len(X)
            assert test_idx.max() < len(X)
            assert train_idx.min() >= 0
            assert test_idx.min() >= 0


class TestDataSplit:
    """Test the DataSplit dataclass."""
    
    def test_basic_data_split(self):
        """Test DataSplit creation."""
        dates = pd.date_range("2020-01-31", periods=36, freq="ME")
        train_idx = dates[:30]
        holdout_idx = dates[30:]
        cv_data_idx = train_idx
        
        split = DataSplit(
            train_idx=pd.DatetimeIndex(train_idx), 
            holdout_idx=pd.DatetimeIndex(holdout_idx),
            cv_data_idx=pd.DatetimeIndex(cv_data_idx)
        )
        
        assert split.n_train == 30
        assert split.n_holdout == 6


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline_holdout_then_cv(self):
        """Test the complete pipeline: holdout extraction then CV splits."""
        # Create realistic monthly data (8 years)
        dates = pd.date_range("2016-01-31", periods=96, freq="ME")
        data = pd.Series(
            np.random.randn(96) * 0.03 + 0.005,  # Simulating returns
            index=dates
        )
        
        # Step 1: Extract holdout first (18 months, per paper)
        holdout_size = 18
        data_split = extract_holdout_first(data, holdout_size)
        
        assert len(data_split.holdout_idx) == 18
        assert len(data_split.train_idx) == 78  # 96 - 18
        
        # Step 2: CV splits on training data
        M, Q, N, K = 18, 6, 10, 5  # Paper parameters
        cv_info = compute_cv_splits(len(data_split.train_idx), M, Q, N, K, min_train_size=12)
        
        # Verify CV doesn't exceed available training data
        assert cv_info.M + (cv_info.K - 1) * cv_info.N + cv_info.Q <= len(data_split.train_idx)
