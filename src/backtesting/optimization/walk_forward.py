"""
Walk-forward optimization for strategy parameter validation

Implements rolling window optimization to validate strategy parameters
on new data and prevent over-fitting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..core.engine import BacktestEngine
from .parameter_grid import ParameterGrid


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework

    Validates strategy parameters using rolling training and testing windows
    to ensure robustness on out-of-sample data.
    """

    def __init__(self, train_window: int = 252, test_window: int = 63, step_size: Optional[int] = None):
        """
        Initialize walk-forward optimizer

        Args:
            train_window: Training window size in days (default 252 trading days)
            test_window: Testing window size in days (default 63 trading days)
            step_size: Step size for rolling windows (default equals test_window)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size or test_window

        # Create backtesting engine for optimization
        self.backtest_engine = BacktestEngine()

        # Configure logging
        self.logger = logging.getLogger(__name__)

    def optimize(self, start_date: datetime, end_date: datetime,
                 parameter_grid: ParameterGrid, strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Run walk-forward optimization over date range

        Args:
            start_date: Start date for optimization
            end_date: End date for optimization
            parameter_grid: Parameter grid for optimization
            strategy_config: Base strategy configuration

        Returns:
            DataFrame with period results including best parameters and performance
        """
        results = []
        current_date = start_date + timedelta(days=self.train_window)

        while current_date + timedelta(days=self.test_window) <= end_date:
            # Define training and testing windows
            train_start = current_date - timedelta(days=self.train_window)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=self.test_window)

            self.logger.info(f"Optimizing period: {train_start} to {train_end}")

            # Optimize parameters on training data
            best_params, train_performance = self._optimize_params(
                train_start, train_end, parameter_grid, strategy_config
            )

            # Test on out-of-sample data
            test_performance = self._backtest_period(
                test_start, test_end, best_params, strategy_config
            )

            # Record period results
            period_result = {
                'period_start': test_start,
                'period_end': test_end,
                'train_start': train_start,
                'train_end': train_end,
                'best_params': best_params,
                'train_return': train_performance.get('returns', 0.0),
                'train_sharpe': train_performance.get('sharpe_ratio', 0.0),
                'train_drawdown': train_performance.get('max_drawdown', 0.0),
                'test_return': test_performance.get('returns', 0.0),
                'test_sharpe': test_performance.get('sharpe_ratio', 0.0),
                'test_drawdown': test_performance.get('max_drawdown', 0.0)
            }

            results.append(period_result)

            # Move to next window
            current_date += timedelta(days=self.step_size)

        return pd.DataFrame(results)

    def _optimize_params(self, start_date: datetime, end_date: datetime,
                        parameter_grid: ParameterGrid, strategy_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Optimize parameters on training data using grid search

        Args:
            start_date: Training start date
            end_date: Training end date
            parameter_grid: Parameter combinations to test
            strategy_config: Base strategy configuration

        Returns:
            Tuple of (best_parameters, best_performance)
        """
        best_performance = {'sharpe_ratio': -np.inf}
        best_params = {}

        # Grid search over parameter combinations
        for params in parameter_grid.grid_search():
            # Merge parameters with base config
            test_config = {**strategy_config, **params}

            # Run backtest with these parameters
            performance = self._backtest_period(start_date, end_date, params, strategy_config)

            # Track best performance based on Sharpe ratio
            if performance.get('sharpe_ratio', -np.inf) > best_performance.get('sharpe_ratio', -np.inf):
                best_performance = performance
                best_params = params

        self.logger.info(f"Best params: {best_params}, Sharpe: {best_performance.get('sharpe_ratio', 0):.3f}")

        return best_params, best_performance

    def _backtest_period(self, start_date: datetime, end_date: datetime,
                        params: Dict[str, Any], strategy_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Run backtest for a specific period with given parameters

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            params: Strategy parameters
            strategy_config: Strategy configuration

        Returns:
            Performance metrics dictionary
        """
        # For now, return mock performance metrics
        # In full implementation, this would run actual backtest

        # Calculate some randomized but realistic metrics
        duration_days = (end_date - start_date).days
        base_return = 0.08 * (duration_days / 252)  # 8% annualized base return

        # Add some parameter-dependent variation
        param_factor = 1.0
        if 'lookback' in params:
            param_factor *= (1.0 + params['lookback'] / 1000)  # Slight variation based on lookback
        if 'threshold' in params:
            param_factor *= (1.0 + params['threshold'])  # Slight variation based on threshold

        returns = base_return * param_factor + np.random.normal(0, 0.02)  # Add noise
        sharpe_ratio = returns / 0.15 if returns > 0 else -1.0  # Rough Sharpe calculation
        max_drawdown = -abs(np.random.normal(0.05, 0.02))  # Random drawdown

        return {
            'returns': returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }