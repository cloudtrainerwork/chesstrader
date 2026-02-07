"""
Test PPO Evaluation System.

Tests for comprehensive agent evaluation including performance metrics,
statistical validation, and benchmark comparisons.
"""

import unittest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from src.training.ppo.evaluation import (
    AgentEvaluator,
    PerformanceMetrics,
    EvaluationResults
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample returns data
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)  # One year of daily returns
        self.positive_returns = np.abs(np.random.normal(0.002, 0.01, 100))
        self.negative_returns = -np.abs(np.random.normal(0.002, 0.01, 100))

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Test with positive returns
        sharpe = PerformanceMetrics.sharpe_ratio(self.positive_returns)
        self.assertGreater(sharpe, 0)
        self.assertIsInstance(sharpe, float)

        # Test with negative returns
        sharpe_neg = PerformanceMetrics.sharpe_ratio(self.negative_returns)
        self.assertLess(sharpe_neg, 0)

        # Test with zero volatility
        constant_returns = np.full(100, 0.01)
        sharpe_const = PerformanceMetrics.sharpe_ratio(constant_returns)
        self.assertEqual(sharpe_const, 0.0)

        # Test with empty array
        sharpe_empty = PerformanceMetrics.sharpe_ratio(np.array([]))
        self.assertEqual(sharpe_empty, 0.0)

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        sortino = PerformanceMetrics.sortino_ratio(self.returns)
        self.assertIsInstance(sortino, float)

        # Sortino should be higher than Sharpe for same data (typically)
        sharpe = PerformanceMetrics.sharpe_ratio(self.returns)
        # Note: This is not always true, but often is for normal market data

        # Test with only positive returns (should be infinity)
        sortino_pos = PerformanceMetrics.sortino_ratio(self.positive_returns)
        self.assertEqual(sortino_pos, float('inf'))

        # Test edge cases
        sortino_empty = PerformanceMetrics.sortino_ratio(np.array([]))
        self.assertEqual(sortino_empty, 0.0)

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calmar = PerformanceMetrics.calmar_ratio(self.returns)
        self.assertIsInstance(calmar, float)

        # Test with positive trend
        trending_returns = np.linspace(-0.01, 0.02, 100)
        calmar_trend = PerformanceMetrics.calmar_ratio(trending_returns)
        self.assertGreater(calmar_trend, 0)

        # Test edge cases
        calmar_empty = PerformanceMetrics.calmar_ratio(np.array([]))
        self.assertEqual(calmar_empty, 0.0)

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create returns with known drawdown
        test_returns = np.array([0.1, -0.05, -0.1, 0.05, 0.15, -0.08])
        max_dd = PerformanceMetrics.max_drawdown(test_returns)

        self.assertGreater(max_dd, 0)  # Should be positive
        self.assertIsInstance(max_dd, float)

        # Test with only positive returns
        max_dd_pos = PerformanceMetrics.max_drawdown(self.positive_returns)
        self.assertGreaterEqual(max_dd_pos, 0)

        # Test edge cases
        max_dd_empty = PerformanceMetrics.max_drawdown(np.array([]))
        self.assertEqual(max_dd_empty, 0.0)

    def test_annualized_return(self):
        """Test annualized return calculation."""
        ann_return = PerformanceMetrics.annualized_return(self.returns)
        self.assertIsInstance(ann_return, float)

        # Test with known returns
        # 1% daily return should give approximately (1.01)^252 - 1
        daily_1pct = np.full(252, 0.01)
        ann_return_1pct = PerformanceMetrics.annualized_return(daily_1pct)
        expected = (1.01 ** 252) - 1
        self.assertAlmostEqual(ann_return_1pct, expected, places=3)

        # Test edge cases
        ann_return_empty = PerformanceMetrics.annualized_return(np.array([]))
        self.assertEqual(ann_return_empty, 0.0)

    def test_value_at_risk(self):
        """Test Value at Risk calculation."""
        var_95 = PerformanceMetrics.value_at_risk(self.returns, 0.95)
        var_99 = PerformanceMetrics.value_at_risk(self.returns, 0.99)

        self.assertIsInstance(var_95, float)
        self.assertIsInstance(var_99, float)

        # VaR 99% should be more extreme than VaR 95%
        self.assertLessEqual(var_99, var_95)  # More negative

        # Test edge cases
        var_empty = PerformanceMetrics.value_at_risk(np.array([]), 0.95)
        self.assertEqual(var_empty, 0.0)

    def test_expected_shortfall(self):
        """Test Expected Shortfall calculation."""
        es_95 = PerformanceMetrics.expected_shortfall(self.returns, 0.95)
        var_95 = PerformanceMetrics.value_at_risk(self.returns, 0.95)

        self.assertIsInstance(es_95, float)

        # Expected shortfall should be more extreme than VaR
        self.assertLessEqual(es_95, var_95)

        # Test edge cases
        es_empty = PerformanceMetrics.expected_shortfall(np.array([]), 0.95)
        self.assertEqual(es_empty, 0.0)

    def test_profit_factor(self):
        """Test profit factor calculation."""
        # Create mixed returns
        mixed_returns = np.array([0.1, -0.05, 0.08, -0.03, 0.12, -0.07])
        pf = PerformanceMetrics.profit_factor(mixed_returns)

        self.assertGreater(pf, 0)
        self.assertIsInstance(pf, float)

        # Test with only wins (should be infinity)
        pf_wins = PerformanceMetrics.profit_factor(self.positive_returns)
        self.assertEqual(pf_wins, float('inf'))

        # Test with only losses
        pf_losses = PerformanceMetrics.profit_factor(self.negative_returns)
        self.assertEqual(pf_losses, 0.0)

        # Test edge cases
        pf_empty = PerformanceMetrics.profit_factor(np.array([]))
        self.assertEqual(pf_empty, 0.0)

    def test_win_rate(self):
        """Test win rate calculation."""
        # Create known win/loss pattern
        test_returns = np.array([0.1, -0.05, 0.08, -0.03, 0.12])  # 3 wins, 2 losses
        win_rate = PerformanceMetrics.win_rate(test_returns)

        self.assertEqual(win_rate, 0.6)  # 3/5 = 0.6

        # Test with all wins
        win_rate_all = PerformanceMetrics.win_rate(self.positive_returns)
        self.assertEqual(win_rate_all, 1.0)

        # Test with all losses
        win_rate_none = PerformanceMetrics.win_rate(self.negative_returns)
        self.assertEqual(win_rate_none, 0.0)

        # Test edge cases
        win_rate_empty = PerformanceMetrics.win_rate(np.array([]))
        self.assertEqual(win_rate_empty, 0.0)


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, returns_sequence=None):
        self.returns_sequence = returns_sequence or [0.01, -0.005, 0.015, -0.01, 0.02]
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.random.randn(10)

    def step(self, action):
        if self.step_count >= len(self.returns_sequence):
            return np.random.randn(10), 0, True, {}

        reward = self.returns_sequence[self.step_count]
        self.step_count += 1
        done = self.step_count >= len(self.returns_sequence)

        return np.random.randn(10), reward, done, {}


class MockPolicyNetwork:
    """Mock policy network for testing."""

    def __init__(self):
        pass

    def __call__(self, obs):
        batch_size = obs.shape[0]
        actions = torch.randint(0, 3, (batch_size,))
        log_probs = torch.randn(batch_size)
        values = torch.randn(batch_size, 1)
        return actions, log_probs, values, {}

    def eval(self):
        pass

    def get_deterministic_action(self, obs):
        batch_size = obs.shape[0]
        return torch.zeros(batch_size, dtype=torch.long)


class TestEvaluationResults(unittest.TestCase):
    """Test EvaluationResults dataclass."""

    def test_evaluation_results_creation(self):
        """Test creating EvaluationResults."""
        results = EvaluationResults(
            episode_returns=[0.1, 0.05, -0.02],
            episode_lengths=[100, 150, 80],
            episode_sharpe_ratios=[0.5, 0.3, -0.1],
            total_return=0.05,
            annualized_return=0.12,
            sharpe_ratio=0.4,
            sortino_ratio=0.5,
            calmar_ratio=0.8,
            max_drawdown=0.03,
            win_rate=0.67,
            profit_factor=1.5,
            average_episode_length=110.0,
            volatility=0.15,
            downside_deviation=0.1,
            var_95=-0.02,
            var_99=-0.04,
            expected_shortfall_95=-0.025
        )

        self.assertEqual(results.total_return, 0.05)
        self.assertEqual(results.sharpe_ratio, 0.4)
        self.assertEqual(len(results.episode_returns), 3)

    def test_evaluation_results_to_dict(self):
        """Test converting EvaluationResults to dictionary."""
        results = EvaluationResults(
            episode_returns=[0.1, 0.05],
            episode_lengths=[100, 150],
            episode_sharpe_ratios=[0.5, 0.3],
            total_return=0.075,
            annualized_return=0.18,
            sharpe_ratio=0.6,
            sortino_ratio=0.7,
            calmar_ratio=1.2,
            max_drawdown=0.02,
            win_rate=1.0,
            profit_factor=2.0,
            average_episode_length=125.0,
            volatility=0.12,
            downside_deviation=0.05,
            var_95=-0.01,
            var_99=-0.02,
            expected_shortfall_95=-0.015
        )

        results_dict = results.to_dict()

        self.assertIsInstance(results_dict, dict)
        self.assertEqual(results_dict['total_return'], 0.075)
        self.assertEqual(results_dict['episode_returns'], [0.1, 0.05])

    def test_evaluation_results_summary(self):
        """Test evaluation results summary."""
        results = EvaluationResults(
            episode_returns=[0.1, 0.05],
            episode_lengths=[100, 150],
            episode_sharpe_ratios=[0.5, 0.3],
            total_return=0.075,
            annualized_return=0.18,
            sharpe_ratio=0.6,
            sortino_ratio=0.7,
            calmar_ratio=1.2,
            max_drawdown=0.02,
            win_rate=1.0,
            profit_factor=2.0,
            average_episode_length=125.0,
            volatility=0.12,
            downside_deviation=0.05,
            var_95=-0.01,
            var_99=-0.02,
            expected_shortfall_95=-0.015,
            n_episodes=2
        )

        summary = results.summary()

        self.assertIsInstance(summary, str)
        self.assertIn('Total Return', summary)
        self.assertIn('Sharpe Ratio', summary)
        self.assertIn('7.50%', summary)  # Should format return as percentage


class TestAgentEvaluator(unittest.TestCase):
    """Test AgentEvaluator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_factory = lambda: MockEnvironment()
        self.evaluator = AgentEvaluator(
            env_factory=self.env_factory,
            n_eval_episodes=5,
            parallel_evaluation=False  # Use sequential for testing
        )

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.n_eval_episodes, 5)
        self.assertEqual(self.evaluator.confidence_level, 0.95)
        self.assertFalse(self.evaluator.parallel_evaluation)

    def test_evaluate_agent(self):
        """Test agent evaluation."""
        policy = MockPolicyNetwork()

        results = self.evaluator.evaluate_agent(
            policy_network=policy,
            deterministic=True,
            save_results=False
        )

        self.assertIsInstance(results, EvaluationResults)
        self.assertEqual(results.n_episodes, 5)
        self.assertGreater(results.total_evaluation_time, 0)
        self.assertIsInstance(results.total_return, float)
        self.assertIsInstance(results.sharpe_ratio, float)

    def test_evaluate_agent_with_save(self):
        """Test agent evaluation with result saving."""
        policy = MockPolicyNetwork()

        results = self.evaluator.evaluate_agent(
            policy_network=policy,
            deterministic=True,
            save_results=True,
            results_dir=self.temp_dir
        )

        self.assertIsInstance(results, EvaluationResults)

        # Check that files were saved
        results_dir = Path(self.temp_dir)
        json_files = list(results_dir.glob('evaluation_results_*.json'))
        report_files = list(results_dir.glob('evaluation_report_*.txt'))

        self.assertGreater(len(json_files), 0)
        self.assertGreater(len(report_files), 0)

    def test_run_single_episode(self):
        """Test running a single episode."""
        policy = MockPolicyNetwork()

        episode_data = self.evaluator._run_single_episode(
            policy_network=policy,
            deterministic=True,
            render=False,
            episode_id=0
        )

        self.assertIsInstance(episode_data, dict)
        self.assertIn('episode_return', episode_data)
        self.assertIn('episode_length', episode_data)
        self.assertIn('episode_rewards', episode_data)
        self.assertIn('episode_sharpe', episode_data)

    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        # Create mock episode data
        episode_data = []
        for i in range(3):
            episode_data.append({
                'episode_id': i,
                'episode_return': 0.1 * (i + 1),
                'episode_length': 100 + i * 10,
                'episode_rewards': [0.01] * (100 + i * 10),
                'episode_sharpe': 0.2 + i * 0.1,
                'episode_max_drawdown': 0.01 + i * 0.005
            })

        results = self.evaluator._calculate_comprehensive_metrics(episode_data)

        self.assertIsInstance(results, EvaluationResults)
        self.assertEqual(len(results.episode_returns), 3)
        self.assertGreater(results.total_return, 0)
        self.assertGreater(results.win_rate, 0)

    def test_calculate_confidence_intervals(self):
        """Test confidence interval calculation."""
        # Create mock episode data
        episode_data = []
        for i in range(10):
            episode_data.append({
                'episode_return': 0.05 + np.random.normal(0, 0.02)
            })

        confidence_intervals = self.evaluator._calculate_confidence_intervals(episode_data)

        self.assertIsInstance(confidence_intervals, dict)
        self.assertIn('total_return', confidence_intervals)
        self.assertIn('sharpe_ratio', confidence_intervals)

        # Check interval format
        for metric, (lower, upper) in confidence_intervals.items():
            self.assertIsInstance(lower, float)
            self.assertIsInstance(upper, float)
            self.assertLessEqual(lower, upper)

    def test_compare_to_benchmark(self):
        """Test benchmark comparison."""
        # Create agent results
        agent_results = EvaluationResults(
            episode_returns=[0.1, 0.08, 0.12],
            episode_lengths=[100, 110, 90],
            episode_sharpe_ratios=[0.5, 0.4, 0.6],
            total_return=0.10,
            annualized_return=0.25,
            sharpe_ratio=0.5,
            sortino_ratio=0.6,
            calmar_ratio=1.0,
            max_drawdown=0.05,
            win_rate=1.0,
            profit_factor=2.0,
            average_episode_length=100.0,
            volatility=0.15,
            downside_deviation=0.1,
            var_95=-0.02,
            var_99=-0.04,
            expected_shortfall_95=-0.025
        )

        # Create benchmark results
        benchmark_results = EvaluationResults(
            episode_returns=[0.05, 0.06, 0.04],
            episode_lengths=[100, 100, 100],
            episode_sharpe_ratios=[0.3, 0.3, 0.2],
            total_return=0.05,
            annualized_return=0.12,
            sharpe_ratio=0.3,
            sortino_ratio=0.35,
            calmar_ratio=0.6,
            max_drawdown=0.08,
            win_rate=1.0,
            profit_factor=1.5,
            average_episode_length=100.0,
            volatility=0.18,
            downside_deviation=0.12,
            var_95=-0.03,
            var_99=-0.05,
            expected_shortfall_95=-0.035
        )

        comparison = self.evaluator.compare_to_benchmark(agent_results, benchmark_results)

        self.assertIsInstance(comparison, dict)
        self.assertIn('performance_comparison', comparison)
        self.assertIn('statistical_tests', comparison)
        self.assertIn('summary', comparison)

        # Check performance comparison
        perf_comp = comparison['performance_comparison']
        self.assertIn('total_return', perf_comp)
        self.assertGreater(perf_comp['total_return']['difference'], 0)  # Agent should be better

    def test_generate_evaluation_report(self):
        """Test evaluation report generation."""
        results = EvaluationResults(
            episode_returns=[0.1, 0.08, 0.12],
            episode_lengths=[100, 110, 90],
            episode_sharpe_ratios=[0.5, 0.4, 0.6],
            total_return=0.10,
            annualized_return=0.25,
            sharpe_ratio=0.5,
            sortino_ratio=0.6,
            calmar_ratio=1.0,
            max_drawdown=0.05,
            win_rate=1.0,
            profit_factor=2.0,
            average_episode_length=100.0,
            volatility=0.15,
            downside_deviation=0.1,
            var_95=-0.02,
            var_99=-0.04,
            expected_shortfall_95=-0.025,
            n_episodes=3,
            total_trades=10,
            winning_trades=8,
            losing_trades=2,
            average_win=0.125,
            average_loss=-0.05,
            confidence_intervals={'total_return': (0.08, 0.12)}
        )

        report = self.evaluator.generate_evaluation_report(results)

        self.assertIsInstance(report, str)
        self.assertIn('Trading Agent Evaluation Report', report)
        self.assertIn('10.00%', report)  # Total return
        self.assertIn('0.500', report)   # Sharpe ratio
        self.assertIn('100.0%', report)  # Win rate

    def test_validate_statistical_significance(self):
        """Test statistical significance validation."""
        results = EvaluationResults(
            episode_returns=[0.01] * 50,  # 50 episodes with small positive returns
            episode_lengths=[100] * 50,
            episode_sharpe_ratios=[0.1] * 50,
            total_return=0.01,
            annualized_return=0.025,
            sharpe_ratio=1.5,  # High Sharpe ratio
            sortino_ratio=1.8,
            calmar_ratio=2.0,
            max_drawdown=0.01,  # Low drawdown
            win_rate=1.0,
            profit_factor=5.0,
            average_episode_length=100.0,
            volatility=0.05,
            downside_deviation=0.02,
            var_95=-0.005,
            var_99=-0.01,
            expected_shortfall_95=-0.006,
            n_episodes=50
        )

        validation = self.evaluator.validate_statistical_significance(results)

        self.assertIsInstance(validation, dict)
        self.assertIn('sample_size_adequacy', validation)
        self.assertIn('distribution_tests', validation)
        self.assertIn('performance_significance', validation)
        self.assertIn('recommendations', validation)

        # Check sample size assessment
        sample_adequacy = validation['sample_size_adequacy']
        self.assertTrue(sample_adequacy['adequate_for_basic_stats'])
        self.assertTrue(sample_adequacy['adequate_for_robust_inference'])

    def test_parallel_evaluation(self):
        """Test parallel evaluation."""
        evaluator_parallel = AgentEvaluator(
            env_factory=self.env_factory,
            n_eval_episodes=4,
            parallel_evaluation=True,
            max_workers=2
        )

        policy = MockPolicyNetwork()

        results = evaluator_parallel.evaluate_agent(
            policy_network=policy,
            deterministic=True,
            save_results=False
        )

        self.assertIsInstance(results, EvaluationResults)
        self.assertEqual(results.n_episodes, 4)

    def test_evaluator_with_different_returns(self):
        """Test evaluator with different return patterns."""
        # Test with losing strategy
        losing_env_factory = lambda: MockEnvironment([-0.01, -0.02, -0.015])
        losing_evaluator = AgentEvaluator(losing_env_factory, n_eval_episodes=3)

        policy = MockPolicyNetwork()
        losing_results = losing_evaluator.evaluate_agent(policy, save_results=False)

        self.assertLess(losing_results.total_return, 0)
        self.assertEqual(losing_results.win_rate, 0.0)

        # Test with mixed strategy
        mixed_env_factory = lambda: MockEnvironment([0.02, -0.01, 0.015, -0.005, 0.01])
        mixed_evaluator = AgentEvaluator(mixed_env_factory, n_eval_episodes=3)

        mixed_results = mixed_evaluator.evaluate_agent(policy, save_results=False)
        self.assertGreater(mixed_results.win_rate, 0)
        self.assertLess(mixed_results.win_rate, 1.0)


class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for evaluation system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Create realistic return sequence
        np.random.seed(42)
        daily_returns = np.random.normal(0.0008, 0.015, 252)  # Realistic daily returns
        env_factory = lambda: MockEnvironment(daily_returns.tolist())

        evaluator = AgentEvaluator(
            env_factory=env_factory,
            n_eval_episodes=20,
            bootstrap_samples=500,
            parallel_evaluation=False
        )

        policy = MockPolicyNetwork()

        # Run evaluation
        results = evaluator.evaluate_agent(
            policy_network=policy,
            deterministic=True,
            save_results=True,
            results_dir=self.temp_dir
        )

        # Verify comprehensive results
        self.assertEqual(results.n_episodes, 20)
        self.assertIsInstance(results.sharpe_ratio, float)
        self.assertIsInstance(results.max_drawdown, float)
        self.assertGreater(results.total_evaluation_time, 0)

        # Verify confidence intervals
        self.assertGreater(len(results.confidence_intervals), 0)

        # Verify files saved
        results_dir = Path(self.temp_dir)
        json_files = list(results_dir.glob('*.json'))
        txt_files = list(results_dir.glob('*.txt'))

        self.assertGreater(len(json_files), 0)
        self.assertGreater(len(txt_files), 0)

        # Test report generation
        report = evaluator.generate_evaluation_report(results)
        self.assertIn('Trading Agent Evaluation Report', report)

        # Test statistical validation
        validation = evaluator.validate_statistical_significance(results)
        self.assertIn('sample_size_adequacy', validation)


if __name__ == '__main__':
    unittest.main()