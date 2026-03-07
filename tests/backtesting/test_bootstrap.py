"""
Test suite for Bootstrap resampling system

Tests the BootstrapResampler's ability to resample trade sequences
and test strategy robustness against sequence dependency risk.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from backtesting.monte_carlo.bootstrap import BootstrapResampler


class TestBootstrapResampler:
    """Test suite for Bootstrap resampling system"""

    def test_resampler_initialization(self):
        """Test BootstrapResampler initialization with default parameters"""
        resampler = BootstrapResampler()

        assert resampler.n_bootstrap_samples == 1000
        assert resampler.random_seed == 42
        assert hasattr(resampler, 'bootstrap_results')

    def test_resampler_custom_parameters(self):
        """Test BootstrapResampler with custom parameters"""
        resampler = BootstrapResampler(n_bootstrap_samples=500, random_seed=123)

        assert resampler.n_bootstrap_samples == 500
        assert resampler.random_seed == 123

    def test_trade_resampling(self):
        """Test bootstrap resampling of trade sequences"""
        resampler = BootstrapResampler(n_bootstrap_samples=3, random_seed=42)

        # Create sample trade returns
        trade_returns = [0.02, -0.01, 0.05, -0.03, 0.08, 0.01, -0.02, 0.04]

        # Resample trade sequence
        bootstrap_results = resampler.resample_trade_sequence(trade_returns)

        # Verify results structure
        assert isinstance(bootstrap_results, pd.DataFrame)
        assert len(bootstrap_results) == 3  # Number of bootstrap samples
        assert 'sample_id' in bootstrap_results.columns
        assert 'cumulative_return' in bootstrap_results.columns
        assert 'max_drawdown' in bootstrap_results.columns
        assert 'final_return' in bootstrap_results.columns

        # Verify each sample has same number of trades
        for idx, row in bootstrap_results.iterrows():
            assert row['trade_count'] == len(trade_returns)

    def test_performance_metrics_calculation(self):
        """Test calculation of performance metrics from resampled sequences"""
        resampler = BootstrapResampler(n_bootstrap_samples=1, random_seed=42)

        # Known trade sequence for predictable results
        trade_returns = [0.10, -0.05, 0.15, -0.08, 0.12]

        # Test the calculation directly
        cumulative_returns, max_drawdown, final_return = resampler._calculate_performance_metrics(trade_returns)

        # Verify calculations
        expected_cumulative = np.cumprod(1 + np.array(trade_returns)) - 1
        expected_final = expected_cumulative[-1]

        np.testing.assert_array_almost_equal(cumulative_returns, expected_cumulative)
        assert abs(final_return - expected_final) < 1e-6
        assert max_drawdown <= 0  # Drawdown should be negative or zero

    def test_simulate_trade_sequences(self):
        """Test simulation of multiple bootstrap trade sequences"""
        resampler = BootstrapResampler(n_bootstrap_samples=5, random_seed=42)

        # Sample trade data
        trade_returns = [0.02, -0.01, 0.05, -0.03, 0.08]
        initial_capital = 100000

        # Run simulation
        simulation_results = resampler.simulate_trade_sequences(trade_returns, initial_capital)

        # Verify results
        assert isinstance(simulation_results, dict)
        assert 'bootstrap_stats' in simulation_results
        assert 'performance_distribution' in simulation_results
        assert 'risk_metrics' in simulation_results

        # Check bootstrap stats
        bootstrap_df = simulation_results['bootstrap_stats']
        assert len(bootstrap_df) == 5
        assert all(col in bootstrap_df.columns for col in ['final_return', 'max_drawdown', 'final_value'])

        # Check performance distribution
        perf_dist = simulation_results['performance_distribution']
        assert 'mean_return' in perf_dist
        assert 'std_return' in perf_dist
        assert 'mean_drawdown' in perf_dist

    def test_sequence_dependency_analysis(self):
        """Test that resampling actually changes sequence order"""
        resampler = BootstrapResampler(n_bootstrap_samples=10, random_seed=42)

        # Create trade sequence with clear pattern
        trade_returns = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05]

        # Mock the resample method to verify it's using random sampling
        with patch('numpy.random.choice') as mock_choice:
            # Setup mock to return a different sequence
            mock_choice.return_value = np.array([0.05, 0.05, -0.05, -0.05, 0.05, -0.05])

            resampled_sequence = resampler._resample_with_replacement(trade_returns)

            # Verify numpy.random.choice was called with correct parameters
            mock_choice.assert_called_once()
            call_args = mock_choice.call_args
            assert len(call_args[0][0]) == len(trade_returns)  # Original sequence length
            assert call_args[1]['replace'] == True  # With replacement
            assert call_args[1]['size'] == len(trade_returns)  # Same size

    def test_edge_cases_handling(self):
        """Test handling of edge cases in trade data"""
        resampler = BootstrapResampler(n_bootstrap_samples=2, random_seed=42)

        # Test empty trade sequence
        empty_results = resampler.resample_trade_sequence([])
        assert len(empty_results) == 0

        # Test single trade
        single_trade = [0.05]
        single_results = resampler.resample_trade_sequence(single_trade)
        assert len(single_results) == 2
        assert all(single_results['final_return'] == 0.05)

        # Test all zero returns
        zero_returns = [0.0, 0.0, 0.0]
        zero_results = resampler.resample_trade_sequence(zero_returns)
        assert all(zero_results['final_return'] == 0.0)
        assert all(zero_results['max_drawdown'] == 0.0)

    def test_reproducible_bootstrap_sampling(self):
        """Test that bootstrap sampling is reproducible with same seed"""
        trade_returns = [0.05, -0.02, 0.08, -0.04, 0.03, -0.01]

        # Create two resamplers with same seed
        resampler1 = BootstrapResampler(n_bootstrap_samples=3, random_seed=42)
        resampler2 = BootstrapResampler(n_bootstrap_samples=3, random_seed=42)

        # Should produce identical results
        results1 = resampler1.resample_trade_sequence(trade_returns)
        results2 = resampler2.resample_trade_sequence(trade_returns)

        pd.testing.assert_frame_equal(results1, results2)

    def test_statistical_robustness_validation(self):
        """Test that resampling provides statistical insights about sequence dependency"""
        resampler = BootstrapResampler(n_bootstrap_samples=100, random_seed=42)

        # Create trade sequence with negative sequence dependency
        # (good trades followed by bad ones)
        trade_returns = [0.05, 0.04, 0.03, -0.05, -0.04, -0.03] * 3

        simulation_results = resampler.simulate_trade_sequences(trade_returns, 100000)

        # Verify we get statistical distribution information
        perf_dist = simulation_results['performance_distribution']

        assert 'confidence_intervals' in perf_dist
        assert '95%' in perf_dist['confidence_intervals']
        assert 'lower' in perf_dist['confidence_intervals']['95%']
        assert 'upper' in perf_dist['confidence_intervals']['95%']

        # Risk metrics should include sequence dependency insights
        risk_metrics = simulation_results['risk_metrics']
        assert 'sequence_dependency_risk' in risk_metrics
        assert 'var_95' in risk_metrics  # Value at Risk
        assert 'cvar_95' in risk_metrics  # Conditional Value at Risk