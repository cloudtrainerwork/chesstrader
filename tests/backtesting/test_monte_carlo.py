"""
Test suite for Monte Carlo simulation engine

Tests the MonteCarloSimulator's ability to run multiple resampled backtests
and collect statistical distributions of performance metrics.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from backtesting.monte_carlo.simulator import MonteCarloSimulator


class TestMonteCarloSimulator:
    """Test suite for Monte Carlo simulation engine"""

    def test_simulator_initialization(self):
        """Test MonteCarloSimulator initialization with default parameters"""
        simulator = MonteCarloSimulator()

        assert simulator.n_simulations == 1000
        assert simulator.random_seed == 42
        assert hasattr(simulator, 'results_df')

    def test_simulator_custom_parameters(self):
        """Test MonteCarloSimulator with custom parameters"""
        simulator = MonteCarloSimulator(n_simulations=500, random_seed=123)

        assert simulator.n_simulations == 500
        assert simulator.random_seed == 123

    def test_simulation_engine(self):
        """Test that Monte Carlo simulation runs multiple resampled backtests"""
        # Create mock backtest engine and portfolio
        mock_engine = Mock()
        mock_portfolio = Mock()

        # Mock portfolio performance for different simulations
        mock_performances = [
            {'total_return': 0.15, 'sharpe_ratio': 1.2, 'max_drawdown': -0.05},
            {'total_return': 0.08, 'sharpe_ratio': 0.9, 'max_drawdown': -0.12},
            {'total_return': 0.22, 'sharpe_ratio': 1.5, 'max_drawdown': -0.03},
        ]
        mock_portfolio.get_performance_summary.side_effect = mock_performances
        mock_engine.portfolio = mock_portfolio

        # Create strategy configuration
        strategy_config = {
            'strategy_class': 'MockStrategy',
            'parameters': {'lookback': 20, 'threshold': 0.02}
        }

        # Test simulation with limited runs for testing
        simulator = MonteCarloSimulator(n_simulations=3, random_seed=42)

        # Mock the setup method to return our mock engine
        simulator._setup_simulation = Mock(return_value=(mock_engine, mock_portfolio))

        # Run simulation
        results_df = simulator.simulate_backtests(strategy_config, historical_data=None)

        # Verify results
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3
        assert 'simulation_id' in results_df.columns
        assert 'total_return' in results_df.columns
        assert 'sharpe_ratio' in results_df.columns
        assert 'max_drawdown' in results_df.columns

        # Verify simulation setup was called correct number of times
        assert simulator._setup_simulation.call_count == 3

    def test_setup_simulation_method(self):
        """Test that _setup_simulation prepares each simulation run properly"""
        simulator = MonteCarloSimulator(n_simulations=100, random_seed=42)

        strategy_config = {
            'strategy_class': 'MockStrategy',
            'parameters': {'lookback': 20}
        }

        # Test setup for first simulation
        engine, portfolio = simulator._setup_simulation(0, strategy_config, None)

        # Verify components exist and are properly configured
        assert engine is not None
        assert portfolio is not None
        assert hasattr(engine, 'portfolio')
        assert engine.portfolio == portfolio

    def test_reproducible_results_with_seed(self):
        """Test that simulations are reproducible with the same random seed"""
        strategy_config = {
            'strategy_class': 'MockStrategy',
            'parameters': {'lookback': 20}
        }

        # Run simulation twice with same seed
        simulator1 = MonteCarloSimulator(n_simulations=5, random_seed=42)
        simulator2 = MonteCarloSimulator(n_simulations=5, random_seed=42)

        # Mock both simulators identically
        mock_results = [
            {'total_return': 0.10, 'sharpe_ratio': 1.0, 'max_drawdown': -0.08},
            {'total_return': 0.15, 'sharpe_ratio': 1.2, 'max_drawdown': -0.05},
            {'total_return': 0.05, 'sharpe_ratio': 0.8, 'max_drawdown': -0.15},
            {'total_return': 0.20, 'sharpe_ratio': 1.4, 'max_drawdown': -0.03},
            {'total_return': 0.12, 'sharpe_ratio': 1.1, 'max_drawdown': -0.10},
        ]

        for simulator in [simulator1, simulator2]:
            mock_portfolio = Mock()
            mock_portfolio.get_performance_summary.side_effect = mock_results
            mock_engine = Mock()
            mock_engine.portfolio = mock_portfolio
            simulator._setup_simulation = Mock(return_value=(mock_engine, mock_portfolio))

        results1 = simulator1.simulate_backtests(strategy_config, historical_data=None)
        results2 = simulator2.simulate_backtests(strategy_config, historical_data=None)

        # Results should be identical with same seed
        pd.testing.assert_frame_equal(results1, results2)

    def test_statistical_metrics_collection(self):
        """Test that simulation collects comprehensive performance metrics"""
        simulator = MonteCarloSimulator(n_simulations=2, random_seed=42)

        # Mock comprehensive performance data
        mock_portfolio = Mock()
        mock_portfolio.get_performance_summary.side_effect = [
            {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.05,
                'final_value': 115000,
                'initial_value': 100000,
                'total_trades': 25
            },
            {
                'total_return': 0.08,
                'sharpe_ratio': 0.9,
                'max_drawdown': -0.12,
                'final_value': 108000,
                'initial_value': 100000,
                'total_trades': 30
            }
        ]

        mock_engine = Mock()
        mock_engine.portfolio = mock_portfolio
        simulator._setup_simulation = Mock(return_value=(mock_engine, mock_portfolio))

        strategy_config = {'strategy_class': 'MockStrategy'}
        results_df = simulator.simulate_backtests(strategy_config, historical_data=None)

        # Verify all expected metrics are collected
        expected_columns = [
            'simulation_id', 'total_return', 'sharpe_ratio', 'max_drawdown',
            'final_value', 'initial_value', 'total_trades'
        ]

        for col in expected_columns:
            assert col in results_df.columns

        # Verify data integrity
        assert results_df['total_return'].iloc[0] == 0.15
        assert results_df['sharpe_ratio'].iloc[1] == 0.9
        assert results_df['total_trades'].sum() == 55

    def test_error_handling_in_simulation(self):
        """Test error handling during simulation runs"""
        simulator = MonteCarloSimulator(n_simulations=3, random_seed=42)

        # Mock portfolio with one failing simulation
        mock_portfolio = Mock()
        mock_portfolio.get_performance_summary.side_effect = [
            {'total_return': 0.10, 'sharpe_ratio': 1.0, 'max_drawdown': -0.05},
            Exception("Portfolio calculation error"),
            {'total_return': 0.15, 'sharpe_ratio': 1.2, 'max_drawdown': -0.03}
        ]

        mock_engine = Mock()
        mock_engine.portfolio = mock_portfolio
        simulator._setup_simulation = Mock(return_value=(mock_engine, mock_portfolio))

        strategy_config = {'strategy_class': 'MockStrategy'}

        # Should handle errors gracefully and return valid results
        results_df = simulator.simulate_backtests(strategy_config, historical_data=None)

        # Should have results from successful simulations only
        assert len(results_df) == 2  # Two successful runs
        assert not results_df.isnull().any().any()  # No null values