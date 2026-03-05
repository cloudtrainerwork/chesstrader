"""
Test walk-forward optimization framework

Tests rolling window optimization, parameter validation, and out-of-sample testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backtesting.optimization.walk_forward import WalkForwardOptimizer
from src.backtesting.optimization.parameter_grid import ParameterGrid
from src.backtesting.core.engine import BacktestEngine


class TestWalkForwardOptimizer:
    """Test WalkForwardOptimizer functionality"""

    def test_initialization(self):
        """Test WalkForwardOptimizer initialization with default parameters"""
        optimizer = WalkForwardOptimizer()

        assert optimizer.train_window == 252  # Default 252 trading days
        assert optimizer.test_window == 63    # Default 63 trading days (quarter)
        assert optimizer.step_size == 63     # Default step size matches test window
        assert optimizer.backtest_engine is not None

    def test_initialization_with_custom_params(self):
        """Test WalkForwardOptimizer with custom window parameters"""
        optimizer = WalkForwardOptimizer(train_window=126, test_window=31, step_size=31)

        assert optimizer.train_window == 126
        assert optimizer.test_window == 31
        assert optimizer.step_size == 31

    def test_optimization_windows(self):
        """Test walk-forward optimization generates correct training and testing windows"""
        # Create sample data spanning 1 year
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        # Create parameter grid
        param_grid = ParameterGrid({
            'lookback': [10, 20],
            'threshold': [0.2, 0.3]
        })

        optimizer = WalkForwardOptimizer(train_window=126, test_window=63, step_size=63)

        # Test optimization
        results = optimizer.optimize(start_date, end_date, param_grid, {})

        # Verify results structure
        assert isinstance(results, pd.DataFrame)
        assert 'period_start' in results.columns
        assert 'period_end' in results.columns
        assert 'train_return' in results.columns
        assert 'test_return' in results.columns
        assert 'best_params' in results.columns
        assert len(results) > 0  # Should have at least one optimization period


class TestParameterGrid:
    """Test ParameterGrid functionality"""

    def test_discrete_parameter_grid(self):
        """Test parameter grid generation with discrete parameters"""
        grid = ParameterGrid({
            'lookback': [10, 20, 30],
            'threshold': [0.1, 0.2, 0.3]
        })

        # Test grid search generation
        combinations = list(grid.grid_search())
        assert len(combinations) == 9  # 3 * 3 = 9 combinations
        assert {'lookback': 10, 'threshold': 0.1} in combinations
        assert {'lookback': 30, 'threshold': 0.3} in combinations

        # Test size calculation
        assert grid.size() == 9

    def test_continuous_parameter_grid(self):
        """Test parameter grid generation with continuous parameters"""
        grid = ParameterGrid({
            'param1': {'type': 'continuous', 'min': 0.0, 'max': 1.0, 'steps': 5},
            'param2': [5, 10, 15]
        })

        # Test grid search generation
        combinations = list(grid.grid_search())
        assert len(combinations) == 15  # 5 * 3 = 15 combinations

        # Test size calculation
        assert grid.size() == 15

        # Check parameter ranges
        ranges = grid.get_param_ranges()
        assert ranges['param1']['type'] == 'continuous'
        assert ranges['param2']['type'] == 'discrete'


class TestWalkForwardIntegration:
    """Test walk-forward optimization integration with backtesting engine"""

    def test_complete_optimization_pipeline(self):
        """Test end-to-end walk-forward optimization pipeline"""
        # Create test data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 8, 31)  # 8 months of data

        # Create parameter grid
        param_grid = ParameterGrid({
            'lookback': [10, 20],
            'threshold': [0.2, 0.3]
        })

        optimizer = WalkForwardOptimizer(train_window=126, test_window=63, step_size=63)

        # Run optimization
        results = optimizer.optimize(start_date, end_date, param_grid, {})

        # Verify results structure
        assert isinstance(results, pd.DataFrame)
        assert 'period_start' in results.columns
        assert 'period_end' in results.columns
        assert 'train_return' in results.columns
        assert 'test_return' in results.columns
        assert 'best_params' in results.columns

        # Verify we have optimization periods
        assert len(results) >= 1

        # Verify performance columns have valid data
        for idx, row in results.iterrows():
            assert isinstance(row['best_params'], dict)
            assert 'lookback' in row['best_params']
            assert 'threshold' in row['best_params']
            assert isinstance(row['train_return'], (int, float))
            assert isinstance(row['test_return'], (int, float))