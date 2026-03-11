"""
Test suite for Statistical Analysis system

Tests the StatisticalAnalyzer's ability to calculate confidence intervals,
test statistical significance, and provide comprehensive risk assessment.
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

from backtesting.monte_carlo.analysis import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test suite for Statistical Analysis system"""

    def test_analyzer_initialization(self):
        """Test StatisticalAnalyzer initialization"""
        analyzer = StatisticalAnalyzer()

        assert hasattr(analyzer, 'confidence_levels')
        assert hasattr(analyzer, 'results')
        assert analyzer.confidence_levels == [90, 95, 99]

    def test_confidence_intervals(self):
        """Test calculation of confidence intervals for Monte Carlo results"""
        analyzer = StatisticalAnalyzer()

        # Create sample Monte Carlo results
        monte_carlo_results = pd.DataFrame({
            'simulation_id': range(100),
            'total_return': np.random.normal(0.12, 0.05, 100),
            'sharpe_ratio': np.random.normal(1.2, 0.3, 100),
            'max_drawdown': np.random.normal(-0.08, 0.03, 100)
        })

        # Calculate confidence intervals
        confidence_intervals = analyzer.calculate_confidence_intervals(monte_carlo_results)

        # Verify structure
        assert isinstance(confidence_intervals, dict)
        assert 'total_return' in confidence_intervals
        assert 'sharpe_ratio' in confidence_intervals
        assert 'max_drawdown' in confidence_intervals

        # Check each metric has all confidence levels
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            assert '90%' in confidence_intervals[metric]
            assert '95%' in confidence_intervals[metric]
            assert '99%' in confidence_intervals[metric]

            # Each level should have lower and upper bounds
            for level in ['90%', '95%', '99%']:
                assert 'lower' in confidence_intervals[metric][level]
                assert 'upper' in confidence_intervals[metric][level]
                assert 'mean' in confidence_intervals[metric][level]

    def test_statistical_significance_testing(self):
        """Test statistical significance comparison against benchmarks"""
        analyzer = StatisticalAnalyzer()

        # Strategy results
        strategy_returns = np.array([0.12, 0.08, 0.15, 0.10, 0.14, 0.09, 0.13])

        # Benchmark results (slightly lower performance)
        benchmark_returns = np.array([0.08, 0.06, 0.10, 0.07, 0.09, 0.05, 0.08])

        # Test significance
        significance_results = analyzer.statistical_significance_test(
            strategy_returns, benchmark_returns, test_type='t_test'
        )

        # Verify results structure
        assert isinstance(significance_results, dict)
        assert 'test_statistic' in significance_results
        assert 'p_value' in significance_results
        assert 'significant' in significance_results
        assert 'confidence_level' in significance_results
        assert 'test_type' in significance_results

        # Test different significance test types
        wilcoxon_results = analyzer.statistical_significance_test(
            strategy_returns, benchmark_returns, test_type='wilcoxon'
        )
        assert wilcoxon_results['test_type'] == 'wilcoxon'

    def test_risk_analysis(self):
        """Test comprehensive risk analysis with tail statistics"""
        analyzer = StatisticalAnalyzer()

        # Create sample distribution with some extreme values
        np.random.seed(42)
        returns_distribution = np.concatenate([
            np.random.normal(0.10, 0.03, 900),  # Most returns
            np.random.normal(-0.15, 0.05, 100)  # Some bad tail outcomes
        ])

        # Perform risk analysis
        risk_metrics = analyzer.risk_analysis(returns_distribution)

        # Verify comprehensive risk metrics
        assert isinstance(risk_metrics, dict)

        # Value at Risk metrics
        assert 'var_95' in risk_metrics
        assert 'var_99' in risk_metrics
        assert 'var_99.9' in risk_metrics

        # Conditional Value at Risk
        assert 'cvar_95' in risk_metrics
        assert 'cvar_99' in risk_metrics

        # Tail statistics
        assert 'tail_ratio' in risk_metrics
        assert 'skewness' in risk_metrics
        assert 'kurtosis' in risk_metrics
        assert 'worst_case' in risk_metrics
        assert 'best_case' in risk_metrics

        # Probability metrics
        assert 'prob_loss' in risk_metrics
        assert 'prob_large_loss' in risk_metrics

        # VaR should be negative (losses)
        assert risk_metrics['var_95'] <= 0
        assert risk_metrics['var_99'] <= risk_metrics['var_95']  # 99% VaR worse than 95%

    def test_format_results_for_reporting(self):
        """Test formatting of statistical results for clean reporting"""
        analyzer = StatisticalAnalyzer()

        # Sample raw results
        raw_results = {
            'confidence_intervals': {
                'total_return': {
                    '95%': {'lower': 0.08, 'upper': 0.16, 'mean': 0.12}
                }
            },
            'risk_metrics': {
                'var_95': -0.05,
                'cvar_95': -0.08,
                'skewness': -0.25
            }
        }

        # Format results
        formatted = analyzer._format_results(raw_results)

        # Verify formatting
        assert isinstance(formatted, dict)
        assert 'summary' in formatted
        assert 'detailed_metrics' in formatted
        assert 'interpretation' in formatted

        # Check that numbers are properly formatted
        summary = formatted['summary']
        assert isinstance(summary, str)
        assert 'confidence interval' in summary.lower()

    def test_monte_carlo_analysis_integration(self):
        """Test complete analysis of Monte Carlo simulation results"""
        analyzer = StatisticalAnalyzer()

        # Mock Monte Carlo results
        monte_carlo_df = pd.DataFrame({
            'simulation_id': range(50),
            'total_return': np.random.normal(0.12, 0.04, 50),
            'sharpe_ratio': np.random.normal(1.1, 0.2, 50),
            'max_drawdown': np.random.normal(-0.06, 0.02, 50),
            'final_value': np.random.normal(112000, 4000, 50)
        })

        # Mock benchmark for comparison
        benchmark_returns = np.random.normal(0.08, 0.03, 50)

        # Perform complete analysis
        complete_analysis = analyzer.analyze_monte_carlo_results(
            monte_carlo_df, benchmark_returns=benchmark_returns
        )

        # Verify comprehensive analysis structure
        assert isinstance(complete_analysis, dict)
        assert 'confidence_intervals' in complete_analysis
        assert 'risk_analysis' in complete_analysis
        assert 'significance_testing' in complete_analysis
        assert 'performance_summary' in complete_analysis

        # Check significance testing results
        significance = complete_analysis['significance_testing']
        assert 'vs_benchmark' in significance
        assert 'test_type' in significance['vs_benchmark']

    def test_tail_behavior_analysis(self):
        """Test analysis of tail behavior and extreme scenarios"""
        analyzer = StatisticalAnalyzer()

        # Create distribution with fat tails
        returns = np.concatenate([
            np.random.normal(0.08, 0.02, 800),   # Normal returns
            np.random.normal(-0.20, 0.05, 100), # Left tail
            np.random.normal(0.25, 0.05, 100)   # Right tail
        ])

        # Analyze tail behavior
        tail_analysis = analyzer.analyze_tail_behavior(returns)

        # Verify tail analysis
        assert isinstance(tail_analysis, dict)
        assert 'left_tail' in tail_analysis
        assert 'right_tail' in tail_analysis
        assert 'tail_dependency' in tail_analysis

        # Left tail analysis (losses)
        left_tail = tail_analysis['left_tail']
        assert 'extreme_scenarios' in left_tail
        assert 'frequency' in left_tail
        assert 'severity' in left_tail

        # Right tail analysis (gains)
        right_tail = tail_analysis['right_tail']
        assert 'extreme_scenarios' in right_tail
        assert 'frequency' in right_tail

    def test_error_handling_edge_cases(self):
        """Test error handling with edge cases and invalid data"""
        analyzer = StatisticalAnalyzer()

        # Test with empty data
        empty_results = analyzer.calculate_confidence_intervals(pd.DataFrame())
        assert isinstance(empty_results, dict)

        # Test with single data point
        single_point_df = pd.DataFrame({
            'total_return': [0.12],
            'sharpe_ratio': [1.2]
        })
        single_results = analyzer.calculate_confidence_intervals(single_point_df)
        assert isinstance(single_results, dict)

        # Test with all identical values
        identical_df = pd.DataFrame({
            'total_return': [0.12] * 10,
            'sharpe_ratio': [1.2] * 10
        })
        identical_results = analyzer.calculate_confidence_intervals(identical_df)
        assert isinstance(identical_results, dict)

        # Test significance testing with insufficient data
        small_sample1 = np.array([0.12])
        small_sample2 = np.array([0.08])

        # Should handle gracefully without crashing
        small_significance = analyzer.statistical_significance_test(
            small_sample1, small_sample2
        )
        assert isinstance(small_significance, dict)
        assert 'error' in small_significance or 'p_value' in small_significance