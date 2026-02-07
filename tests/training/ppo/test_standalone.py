"""
Standalone tests for new PPO components without external dependencies.
"""

import unittest
import tempfile
import shutil
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Test our new components
from src.training.ppo.evaluation import PerformanceMetrics, EvaluationResults
from src.training.ppo.checkpoints import CheckpointManager, CheckpointInfo, TrainerState


class TestPerformanceMetricsStandalone(unittest.TestCase):
    """Test performance metrics without external dependencies."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 100)

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = PerformanceMetrics.sharpe_ratio(self.returns)
        self.assertIsInstance(sharpe, float)

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        max_dd = PerformanceMetrics.max_drawdown(self.returns)
        self.assertIsInstance(max_dd, float)
        self.assertGreaterEqual(max_dd, 0)

    def test_win_rate(self):
        """Test win rate calculation."""
        win_rate = PerformanceMetrics.win_rate(self.returns)
        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0)
        self.assertLessEqual(win_rate, 1)


class TestEvaluationResultsStandalone(unittest.TestCase):
    """Test EvaluationResults without dependencies."""

    def test_evaluation_results_creation(self):
        """Test creating evaluation results."""
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

        self.assertEqual(results.total_return, 0.075)
        self.assertEqual(len(results.episode_returns), 2)

    def test_to_dict(self):
        """Test converting results to dictionary."""
        results = EvaluationResults(
            episode_returns=[0.1],
            episode_lengths=[100],
            episode_sharpe_ratios=[0.5],
            total_return=0.1,
            annualized_return=0.2,
            sharpe_ratio=0.8,
            sortino_ratio=0.9,
            calmar_ratio=1.5,
            max_drawdown=0.01,
            win_rate=1.0,
            profit_factor=3.0,
            average_episode_length=100.0,
            volatility=0.1,
            downside_deviation=0.05,
            var_95=-0.005,
            var_99=-0.01,
            expected_shortfall_95=-0.007
        )

        result_dict = results.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['total_return'], 0.1)


class TestCheckpointManagerStandalone(unittest.TestCase):
    """Test CheckpointManager without complex dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / 'checkpoints'
        self.manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=3
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_manager_initialization(self):
        """Test checkpoint manager initialization."""
        self.assertEqual(self.manager.max_checkpoints, 3)
        self.assertTrue(self.manager.checkpoint_dir.exists())
        self.assertTrue(self.manager.models_dir.exists())

    def test_save_simple_checkpoint(self):
        """Test saving a simple checkpoint."""
        trainer_state = {
            'step': 1000,
            'episode': 50,
            'model_state_dict': {'param': torch.randn(5, 5)},
            'optimizer_state_dict': {},
            'curriculum_state': {},
            'performance_history': [],
            'config': {'learning_rate': 3e-4}
        }

        filepath = self.manager.save_checkpoint(
            trainer_state=trainer_state,
            step=1000,
            performance_metric=0.5
        )

        self.assertTrue(Path(filepath).exists())
        self.assertEqual(len(self.manager.checkpoint_info), 1)


class TestPPOConfigStandalone(unittest.TestCase):
    """Test basic config handling."""

    def test_config_as_dict(self):
        """Test config as dictionary."""
        config = {'learning_rate': 3e-4, 'n_steps': 2048}
        self.assertEqual(config['learning_rate'], 3e-4)
        self.assertEqual(config['n_steps'], 2048)


class TestLoggingStandalone(unittest.TestCase):
    """Test logging components without TensorBoard."""

    def test_metrics_tracker(self):
        """Test metrics tracker functionality."""
        from src.training.ppo.logging import MetricsTracker

        # Test metrics tracker
        tracker = MetricsTracker()
        tracker.update(loss=0.5, accuracy=0.8)
        self.assertIn('loss', tracker.metrics)
        self.assertEqual(tracker.metrics['loss'][0], 0.5)


if __name__ == '__main__':
    unittest.main()