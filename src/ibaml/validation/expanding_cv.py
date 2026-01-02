from __future__ import annotations
from typing import Tuple, List, Iterator, NamedTuple, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SplitInfo(NamedTuple):
    """Information about computed CV splits.
    
    Attributes:
        M: Effective initial training window size
        Q: Effective validation window size
        N: Effective step size
        K: Effective number of folds
        T_available: Total samples available for CV (after holdout extraction)
        was_adjusted: Whether dynamic adjustment was applied
        adjustments: List of adjustments made (if any)
    """
    M: int
    Q: int
    N: int
    K: int
    T_available: int
    was_adjusted: bool
    adjustments: List[str]


@dataclass
class DataSplit:
    """Container for train/validation/holdout split information.
    
    Attributes:
        train_idx: Indices for training data (DatetimeIndex or integer array)
        holdout_idx: Indices for holdout evaluation
        cv_data_idx: Indices available for cross-validation (train portion only)
        n_train: Number of training samples
        n_holdout: Number of holdout samples
        n_cv: Number of CV samples
    """
    train_idx: pd.DatetimeIndex
    holdout_idx: pd.DatetimeIndex
    cv_data_idx: pd.DatetimeIndex
    
    @property
    def n_train(self) -> int:
        return len(self.train_idx)
    
    @property
    def n_holdout(self) -> int:
        return len(self.holdout_idx)
    
    @property
    def n_cv(self) -> int:
        return len(self.cv_data_idx)


def extract_holdout_first(
    y: pd.Series,
    holdout_size: int,
    require_consecutive: bool = True,
) -> DataSplit:
    """Extract holdout set FIRST from the end of the series.
    
    Order of operations:
    1. Extract holdout from the END of the series
    2. Use remaining data for training and cross-validation
    
    Args:
        y: Target series with DatetimeIndex
        holdout_size: Number of observations for holdout
        require_consecutive: If True, ensure holdout is consecutive months
        
    Returns:
        DataSplit with train, holdout, and cv indices        
    Raises:
        ValueError: If not enough data for holdout
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        y = y.copy()
        y.index = pd.to_datetime(y.index)
    
    # Filter to valid (non-NaN) observations
    valid_mask = pd.notnull(y)
    idx_valid = y.index[valid_mask]
    
    if len(idx_valid) < holdout_size + 1:  # Need at least 1 training sample
        raise ValueError(
            f"Insufficient data: {len(idx_valid)} valid observations, "
            f"need at least {holdout_size + 1} (holdout_size + 1 for training)"
        )
    
    if require_consecutive:
        # Find longest consecutive tail of monthly data
        # Convert to periods for easier comparison
        periods = idx_valid.to_period("M")
        
        # Start from the last period and work backwards
        consecutive_count = 1
        for i in range(len(periods) - 1, 0, -1):
            # Check if previous period is exactly one month before
            if (periods[i] - periods[i-1]).n == 1:
                consecutive_count += 1
            else:
                break
        
        if consecutive_count < holdout_size:
            logger.warning(
                f"Longest consecutive tail ({consecutive_count}) is shorter than "
                f"requested holdout_size ({holdout_size}). Using available tail."
            )
            holdout_size = consecutive_count
        
        # Take the last holdout_size from the valid indices
        holdout_idx = idx_valid[-holdout_size:]
    else:
        # Simply take last holdout_size observations
        holdout_idx = idx_valid[-holdout_size:]
    
    # Training is everything before holdout
    train_idx = idx_valid[:-holdout_size] if holdout_size > 0 else idx_valid
    
    # CV data is the same as train for now (CV will further split this)
    cv_data_idx = train_idx
    
    logger.info(
        f"Data split: train={len(train_idx)}, holdout={len(holdout_idx)}, "
        f"holdout_range=[{holdout_idx[0].date()}, {holdout_idx[-1].date()}]"
    )
    
    return DataSplit(
        train_idx=train_idx,
        holdout_idx=holdout_idx,
        cv_data_idx=cv_data_idx,
    )


def compute_cv_splits(
    T: int,
    M: int,
    Q: int,
    N: int,
    K: int,
    min_train_size: int | None = None,
    allow_dynamic_adjustment: bool = True,
) -> SplitInfo:
    """Compute effective CV split parameters with optional dynamic adjustment.
    
    The EWCV formula: For fold i (0-indexed):
        - Training: [0, M + i*N)
        - Validation: [M + i*N, M + i*N + Q)
    
    Last fold must satisfy: M + (K-1)*N + Q <= T
    
    Args:
        T: Total samples available for CV (after holdout extraction)
        M: Requested initial training window size
        Q: Requested validation window size
        N: Requested step size
        K: Requested number of folds
        min_train_size: Minimum allowed training size
        allow_dynamic_adjustment: Whether to adjust splits when data is insufficient
        
    Returns:
        SplitInfo with effective parameters and adjustment details       
    Raises:
        ValueError: If splits cannot be satisfied even with adjustments
    """
    adjustments = []
    was_adjusted = False
    M_eff, Q_eff, N_eff, K_eff = M, Q, N, K
    
    # Compute default min_train_size if not provided (use 50% of available data)
    if min_train_size is None:
        min_train_size = max(1, int(T * 0.5))

    # Minimum required: min_train_size + Q (one fold with training + validation)
    min_required = min_train_size + Q_eff

    if T < min_required:
        raise ValueError(
            f"Insufficient data for CV: T={T} < min_required={min_required} "
            f"(min_train_size={min_train_size} + Q={Q_eff})"
        )
    
    # Check if requested splits fit
    required_for_full = M_eff + (K_eff - 1) * N_eff + Q_eff
    
    if required_for_full <= T:
        # Everything fits, no adjustment needed
        return SplitInfo(
            M=M_eff, Q=Q_eff, N=N_eff, K=K_eff,
            T_available=T, was_adjusted=False, adjustments=[]
        )
    
    if not allow_dynamic_adjustment:
        raise ValueError(
            f"Requested CV splits require {required_for_full} samples but only {T} available. "
            f"Set allow_dynamic_adjustment=True to enable automatic adjustment."
        )
    
    # Dynamic adjustment target:
    # 1. First, try reducing K (fewer folds)
    # 2. Then, reduce M towards min_train_size
    # 3. Finally, reduce N (smaller steps between folds)
    
    # target 1: Reduce K
    # For K folds: M + (K-1)*N + Q <= T
    # K <= (T - M - Q) / N + 1
    K_max = max(1, int((T - M_eff - Q_eff) / max(N_eff, 1)) + 1)
    if K_max < K_eff:
        adjustments.append(f"K: {K_eff} -> {K_max} (reduced folds to fit data)")
        K_eff = K_max
        was_adjusted = True
    
    # Check if adjustment worked
    required = M_eff + (K_eff - 1) * N_eff + Q_eff
    if required <= T:
        return SplitInfo(
            M=M_eff, Q=Q_eff, N=N_eff, K=K_eff,
            T_available=T, was_adjusted=was_adjusted, adjustments=adjustments
        )
    
    # target 2: Reduce M towards min_train_size
    # M + (K-1)*N + Q <= T
    # M <= T - (K-1)*N - Q
    M_max = T - (K_eff - 1) * N_eff - Q_eff
    if M_max < min_train_size:
        # Even with min_train_size, we need to reduce N or K further
        M_eff = min_train_size
    else:
        M_eff = M_max
    
    if M_eff < M:
        adjustments.append(f"M: {M} -> {M_eff} (reduced initial training window)")
        was_adjusted = True
    
    # Check again
    required = M_eff + (K_eff - 1) * N_eff + Q_eff
    if required <= T:
        return SplitInfo(
            M=M_eff, Q=Q_eff, N=N_eff, K=K_eff,
            T_available=T, was_adjusted=was_adjusted, adjustments=adjustments
        )
    
    # target 3: Reduce N (step size)
    # M + (K-1)*N + Q <= T
    # N <= (T - M - Q) / (K - 1)
    if K_eff > 1:
        N_max = max(1, int((T - M_eff - Q_eff) / (K_eff - 1)))
        if N_max < N_eff:
            adjustments.append(f"N: {N_eff} -> {N_max} (reduced step size)")
            N_eff = N_max
            was_adjusted = True
    
    # Final check
    required = M_eff + (K_eff - 1) * N_eff + Q_eff
    if required <= T:
        return SplitInfo(
            M=M_eff, Q=Q_eff, N=N_eff, K=K_eff,
            T_available=T, was_adjusted=was_adjusted, adjustments=adjustments
        )
    
    # Last resort: reduce K to 1 fold
    K_eff = 1
    N_eff = 1  # N doesn't matter with K=1
    adjustments.append("K: reduced to 1 fold as last resort")
    was_adjusted = True
    
    required = M_eff + Q_eff
    if required > T:
        raise ValueError(
            f"Cannot fit even 1 CV fold: need M={M_eff} + Q={Q_eff} = {required}, "
            f"but only T={T} available. Reduce holdout size or min_train_size."
        )
    
    return SplitInfo(
        M=M_eff, Q=Q_eff, N=N_eff, K=K_eff,
        T_available=T, was_adjusted=was_adjusted, adjustments=adjustments
    )


def compute_dynamic_cv_params(
    T_cv: int,
    Q: int,
    K: int,
    min_train_ratio: float,
    min_train_abs: int,
    requested_N: int | None = None,
) -> Dict[str, int]:
    """
    Compute CV parameters dynamically so final fold validation ends at T_cv.
    
    This ensures:
    1. Final fold's validation window = [T_cv - Q, T_cv)
    2. Training expands across K folds
    3. All available CV data is used efficiently
    
    The key insight is that we FIX Q (validation size) and K (number of folds),
    then COMPUTE M (initial training size) and N (step size) to fill the space.
    
    Formula derivation:
        - Final fold (i=K-1) validation ends at: M + (K-1)*N + Q = T_cv
        - Therefore: M + (K-1)*N = T_cv - Q
        - We want training to expand reasonably, so we choose N based on
          how much expansion we want between first and last fold.
    
    Args:
        T_cv: Total observations available for CV (= total - holdout)
        Q: Validation window size per fold (fixed, typically 18)
        K: Number of CV folds (fixed, typically 5)
        min_train_ratio: Minimum ratio of final_train_size for initial train
        min_train_abs: Absolute minimum training size
        
    Returns:
        Dictionary with computed parameters:
        - M: Initial training window size
        - N: Step size between folds  
        - Q: Validation size (unchanged)
        - K: Number of folds (may be reduced if insufficient data)
        - final_train_size: Training size for last fold
        - final_val_end: Where final validation ends (should = T_cv)
    """
    if T_cv <= Q:
        raise ValueError(f"T_cv ({T_cv}) must be > Q ({Q})")
    
    # Space available for training in the final fold
    # Final fold: train=[0, M+(K-1)*N), val=[M+(K-1)*N, M+(K-1)*N+Q)
    # We want val to end at T_cv, so: M + (K-1)*N + Q = T_cv
    # Therefore: M + (K-1)*N = T_cv - Q (this is final_train_size)
    final_train_size = T_cv - Q
    
    if final_train_size < min_train_abs:
        raise ValueError(
            f"Not enough data for CV: final_train_size={final_train_size} < min={min_train_abs}. "
            f"T_cv={T_cv}, Q={Q}"
        )
    
    # Reduce K if needed
    K_eff = K
    while K_eff > 1:
        # Check if we can fit K_eff folds with reasonable step size
        # M + (K_eff-1)*N = final_train_size
        # With N >= 1 and M >= min_train_abs:
        # min_train_abs + (K_eff-1)*1 <= final_train_size
        if min_train_abs + (K_eff - 1) <= final_train_size:
            break
        K_eff -= 1
        logger.warning(f"Reduced K from {K} to {K_eff} due to insufficient data")
    
    if K_eff == 1:
        # Single fold: M = final_train_size, N = 0
        return {
            "M": final_train_size,
            "N": 0,
            "Q": Q,
            "K": 1,
            "final_train_size": final_train_size,
            "final_val_end": T_cv,
        }
    
    # Compute M and N for K_eff folds
    # We want M to be at least min_train_ratio of final_train_size
    M_target = max(min_train_abs, int(final_train_size * min_train_ratio))

    # If a requested_N is provided, prefer it when feasible (closer to user intent)
    if requested_N is not None and requested_N > 0:
        # Maximum feasible N given M_target: (final_train_size - M_target) // (K_eff - 1)
        max_N_allowed = (final_train_size - M_target) // (K_eff - 1)
        if max_N_allowed < 1:
            # Not feasible to keep M_target; fall back to smallest N=1 and adjust M
            N_eff = 1
            M_eff = final_train_size - (K_eff - 1)
            if M_eff < min_train_abs:
                # Try reducing K to satisfy min_train_abs
                K_eff = final_train_size - min_train_abs + 1
                if K_eff < 1:
                    K_eff = 1
                M_eff = final_train_size - (K_eff - 1) if K_eff > 1 else final_train_size
                N_eff = 1 if K_eff > 1 else 0
        else:
            # Use the requested N but don't exceed feasibility
            N_eff = min(requested_N, max_N_allowed)
            M_eff = final_train_size - (K_eff - 1) * N_eff
            # If resulting M_eff is below min, reduce N to increase M_eff
            if M_eff < min_train_abs:
                # Reduce N until M_eff >= min_train_abs or N becomes 1
                while N_eff > 1 and final_train_size - (K_eff - 1) * N_eff < min_train_abs:
                    N_eff -= 1
                M_eff = final_train_size - (K_eff - 1) * N_eff
                if M_eff < min_train_abs:
                    # As a last resort, set N=1 and recompute
                    N_eff = 1
                    M_eff = final_train_size - (K_eff - 1)
    else:
        # No requested_N: compute N to meet M_target evenly across folds
        N_computed = (final_train_size - M_target) / (K_eff - 1)
        if N_computed < 1:
            # Step size too small, reduce M
            N_eff = 1
            M_eff = final_train_size - (K_eff - 1)
            if M_eff < min_train_abs:
                # Still not enough, reduce K further
                K_eff = final_train_size - min_train_abs + 1
                if K_eff < 1:
                    K_eff = 1
                M_eff = final_train_size - (K_eff - 1) if K_eff > 1 else final_train_size
                N_eff = 1 if K_eff > 1 else 0
        else:
            N_eff = int(N_computed)
            M_eff = final_train_size - (K_eff - 1) * N_eff
    
    # Verify
    computed_final = M_eff + (K_eff - 1) * N_eff
    computed_val_end = computed_final + Q
    
    if computed_val_end != T_cv:
        # Adjust M to make it exact
        M_eff = T_cv - Q - (K_eff - 1) * N_eff
        computed_final = M_eff + (K_eff - 1) * N_eff
        computed_val_end = computed_final + Q
    
    logger.info(
        f"Dynamic CV params: M={M_eff}, N={N_eff}, Q={Q}, K={K_eff} | "
        f"T_cv={T_cv}, final_train=[0:{computed_final}), final_val=[{computed_final}:{computed_val_end})"
    )
    
    return {
        "M": M_eff,
        "N": N_eff,
        "Q": Q,
        "K": K_eff,
        "final_train_size": computed_final,
        "final_val_end": computed_val_end,
    }


class ExpandingWindowSplit:
    """Expanding window cross-validation iterator.
    
    Implements Algorithm 1 from the paper: EWCV-EFS
    
    For fold i (0 to K-1):
        - Training window: [0, M + i*N)
        - Validation window: [M + i*N, M + i*N + Q)
    
    The training window expands with each fold while the validation
    window slides forward by N steps.
    
    Attributes:
        M: Initial training window size
        Q: Validation window size
        N: Step size between folds
        K: Number of folds
        T: Total samples available
        split_info: Computed split information (if dynamic adjustment was used)
    """
    
    def __init__(
        self,
        M: int,
        Q: int,
        N: int,
        K: int,
        T: int,
        min_train_size: int | None = None,
        allow_dynamic_adjustment: bool = True,
    ):
        """Initialize expanding window splitter.
        
        Args:
            M: Initial training window size
            Q: Validation window size
            N: Step size between folds
            K: Number of folds
            T: Total samples available
            min_train_size: Minimum training size for dynamic adjustment
            allow_dynamic_adjustment: Whether to adjust splits automatically
        """
        self.split_info = compute_cv_splits(
            T=T, M=M, Q=Q, N=N, K=K,
            min_train_size=min_train_size,
            allow_dynamic_adjustment=allow_dynamic_adjustment,
        )
        
        self.M = self.split_info.M
        self.Q = self.split_info.Q
        self.N = self.split_info.N
        self.K = self.split_info.K
        self.T = T
        
        if self.split_info.was_adjusted:
            logger.warning(
                f"CV splits adjusted for T={T}: " + 
                ", ".join(self.split_info.adjustments)
            )
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over CV folds.
        
        Yields:
            (train_indices, validation_indices) as numpy arrays
        """
        for i in range(self.K):
            train_end = self.M + i * self.N
            valid_start = train_end
            valid_end = valid_start + self.Q
            
            # Safety check
            if valid_end > self.T:
                break
            
            train_idx = np.arange(0, train_end, dtype=int)
            valid_idx = np.arange(valid_start, valid_end, dtype=int)
            
            if len(train_idx) == 0 or len(valid_idx) == 0:
                continue
            
            yield train_idx, valid_idx
    
    def __len__(self) -> int:
        """Return number of folds."""
        return self.K
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations in the cross-validator.
        
        Scikit-learn compatible method.
        
        Args:
            X: Ignored, exists for sklearn API compatibility
            y: Ignored, exists for sklearn API compatibility  
            groups: Ignored, exists for sklearn API compatibility
            
        Returns:
            Number of folds
        """
        return self.K
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.
        
        Scikit-learn compatible method.
        
        Args:
            X: Array-like of shape (n_samples, n_features). Training data.
            y: Array-like of shape (n_samples,). Target variable (ignored).
            groups: Group labels for samples (ignored).
            
        Yields:
            train_idx: The training set indices for that split.
            test_idx: The testing set indices for that split.
        """
        for train_idx, test_idx in self:
            yield train_idx, test_idx
    
    def get_fold_info(self) -> List[dict]:
        """Get detailed information about each fold.
        
        Returns:
            List of dicts with fold details (train_size, valid_size, train_range, valid_range)
        """
        folds = []
        for i, (train_idx, valid_idx) in enumerate(self):
            folds.append({
                "fold": i,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
                "train_range": (int(train_idx[0]), int(train_idx[-1])),
                "valid_range": (int(valid_idx[0]), int(valid_idx[-1])),
            })
        return folds


def create_cv_splitter(
    n_samples: int,
    M: int,
    Q: int,
    N: int,
    K: int,
    min_train_size: int | None = None,
    allow_dynamic_adjustment: bool = True,
) -> ExpandingWindowSplit:
    """Factory function to create CV splitter with validation.
    
    This is the recommended way to create an ExpandingWindowSplit instance
    as it provides better error messages and logging.
    
    Args:
        n_samples: Number of samples available for CV
        M: Initial training window size
        Q: Validation window size
        N: Step size between folds
        K: Number of folds
        min_train_size: Minimum training size
        allow_dynamic_adjustment: Whether to adjust splits automatically
        
    Returns:
        Configured ExpandingWindowSplit instance
    """
    return ExpandingWindowSplit(
        M=M, Q=Q, N=N, K=K, T=n_samples,
        min_train_size=min_train_size,
        allow_dynamic_adjustment=allow_dynamic_adjustment,
    )


def adjust_split_ratios(
    T: int,
    init_tran_size: float,
    folds_val_set_size: int,
    folds_steps_size: int,
    num_of_folds: int,
    use_dynamic: bool = True,
) -> Tuple[int, int, int, int]:
    """Clamp CV splits to fit available data (new naming semantics).

    Parameters renamed for clarity:
      - init_tran_size: initial training window (fraction 0-1 or absolute int)
      - folds_val_set_size: validation window size Q
      - folds_steps_size: step size N between folds
      - num_of_folds: number of folds K

    If `init_tran_size` is a float in (0,1), it will be interpreted as a
    fraction of available CV data and used when computing dynamic M/N values.

    Returns (M_eff, Q_eff, N_eff, K_eff) as integers.
    """
    if T <= 0:
        raise ValueError("Empty training target series after preprocessing/holdout split.")
    Q_eff = int(folds_val_set_size)
    K_eff = int(num_of_folds)

    # Support init_tran_size being a fraction (0<M<1) indicating initial training ratio.
    if use_dynamic:
        try:
            if isinstance(init_tran_size, float) and 0.0 < init_tran_size < 1.0:
                params = compute_dynamic_cv_params(
                    T_cv=T,
                    Q=Q_eff,
                    K=K_eff,
                    min_train_ratio=float(init_tran_size),
                    min_train_abs=50,
                    requested_N=int(folds_steps_size) if folds_steps_size is not None else None,
                )
            else:
                # If an absolute integer was provided for init_tran_size,
                # we still allow dynamic computation using default ratio.
                params = compute_dynamic_cv_params(
                    T_cv=T,
                    Q=Q_eff,
                    K=K_eff,
                    min_train_ratio=0.5,
                    min_train_abs=50,
                    requested_N=int(folds_steps_size) if folds_steps_size is not None else None,
                )

            M_eff = params["M"]
            N_eff = params["N"]
            Q_eff = params["Q"]
            K_eff = params["K"]

            logger.debug(
                f"Dynamic CV: T={T}, M={M_eff}, N={N_eff}, Q={Q_eff}, K={K_eff} | "
                f"final_val_end={params['final_val_end']}"
            )
            return M_eff, Q_eff, N_eff, K_eff
        except ValueError as e:
            logger.warning(f"Dynamic CV computation failed: {e}. Falling back to static.")

    # Fallback: static clamping (original behavior)
    # Interpret fractional init_tran_size as absolute when falling back
    if isinstance(init_tran_size, float) and 0.0 < init_tran_size < 1.0:
        M_eff = max(1, int(init_tran_size * T))
    else:
        try:
            M_eff = int(init_tran_size)
        except Exception:
            M_eff = max(30, T // 2)

    N_eff = int(folds_steps_size)

    if M_eff + Q_eff > T:
        M_eff = max(min(M_eff, T - max(Q_eff, 1)), max(30, T // 2))
        if M_eff + Q_eff > T:
            Q_eff = max(1, T - M_eff)

    max_i = (T - (M_eff + Q_eff)) // max(N_eff, 1)
    K_max = max_i + 1
    if K_eff > K_max:
        K_eff = K_max

    if K_eff <= 0 or M_eff + Q_eff > T:
        raise ValueError(f"Not enough samples with T={T}, M={M_eff}, Q={Q_eff}.")

    return M_eff, Q_eff, N_eff, K_eff

