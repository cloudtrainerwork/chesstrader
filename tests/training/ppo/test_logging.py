"""
Test PPO Logging System.

Tests for comprehensive TensorBoard logging, metrics tracking,
and visualization capabilities.
"""

import unittest
import tempfile
import shutil
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.training.ppo.logging import (
    PPOLogger,
    MetricsTracker,
    LoggingConfig,
    TENSORBOARD_AVAILABLE
)
from src.training.ppo.trainer import PPOConfig


class TestLoggingConfig(unittest.TestCase):
    """Test logging configuration."""

    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()

        self.assertEqual(config.log_dir, './logs/ppo_training')
        self.assertEqual(config.experiment_name, 'options_trading_v1')
        self.assertEqual(config.flush_secs, 30)
        self.assertEqual(config.max_queue_size, 100)
        self.assertTrue(config.create_plots)
        self.assertEqual(config.plot_freq, 1000)
        self.assertTrue(config.save_plots)
        self.assertEqual(config.plot_format, 'png')

    def test_custom_config(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            log_dir='./custom_logs',
            experiment_name='test_experiment',
            flush_secs=10,
            create_plots=False
        )

        self.assertEqual(config.log_dir, './custom_logs')
        self.assertEqual(config.experiment_name, 'test_experiment')
        self.assertEqual(config.flush_secs, 10)
        self.assertFalse(config.create_plots)


class TestMetricsTracker(unittest.TestCase):
    """Test metrics tracking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = MetricsTracker(window_size=5)

    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.window_size, 5)
        self.assertEqual(len(self.tracker.metrics), 0)
        self.assertEqual(len(self.tracker.global_metrics), 0)

    def test_update_metrics(self):
        """Test updating metrics."""
        self.tracker.update(loss=0.5, accuracy=0.8)

        self.assertIn('loss', self.tracker.metrics)
        self.assertIn('accuracy', self.tracker.metrics)
        self.assertEqual(len(self.tracker.metrics['loss']), 1)
        self.assertEqual(self.tracker.metrics['loss'][0], 0.5)

    def test_recent_average(self):
        """Test recent average calculation."""
        # Add some values
        for i in range(10):
            self.tracker.update(value=i)

        # Should only consider last 5 values (5, 6, 7, 8, 9)
        recent_avg = self.tracker.get_recent_average('value')
        expected_avg = np.mean([5, 6, 7, 8, 9])
        self.assertAlmostEqual(recent_avg, expected_avg, places=5)

    def test_recent_average_empty_metric(self):
        """Test recent average with empty metric."""
        avg = self.tracker.get_recent_average('nonexistent')
        self.assertEqual(avg, 0.0)

    def test_statistics(self):
        """Test statistics calculation."""
        values = [1, 2, 3, 4, 5]
        for v in values:
            self.tracker.update(test_metric=v)

        stats = self.tracker.get_statistics()
        self.assertIn('test_metric', stats)

        metric_stats = stats['test_metric']
        self.assertAlmostEqual(metric_stats['mean'], 3.0)
        self.assertAlmostEqual(metric_stats['min'], 1.0)
        self.assertAlmostEqual(metric_stats['max'], 5.0)
        self.assertEqual(metric_stats['count'], 5)
        self.assertEqual(metric_stats['latest'], 5.0)

    def test_window_size_limit(self):
        """Test that metrics respect window size limit."""
        # Add more values than window size
        for i in range(10):
            self.tracker.update(metric=i)

        # Should only keep last 5 values
        self.assertEqual(len(self.tracker.metrics['metric']), 5)
        # Should be [5, 6, 7, 8, 9]
        self.assertEqual(list(self.tracker.metrics['metric']), [5, 6, 7, 8, 9])

    def test_trend_calculation(self):
        """Test trend calculation."""
        # Increasing trend
        for i in range(10):
            self.tracker.update(increasing=i * 0.1)

        trend = self.tracker.get_trend('increasing', lookback=5)
        self.assertEqual(trend, 'increasing')

        # Decreasing trend
        for i in range(10):
            self.tracker.update(decreasing=-i * 0.1)

        trend = self.tracker.get_trend('decreasing', lookback=5)
        self.assertEqual(trend, 'decreasing')

        # Stable trend
        for i in range(10):
            self.tracker.update(stable=0.5)

        trend = self.tracker.get_trend('stable', lookback=5)
        self.assertEqual(trend, 'stable')

    def test_invalid_values(self):
        """Test handling of invalid values."""
        # Test with NaN and infinity
        self.tracker.update(
            valid=1.0,
            nan_value=float('nan'),
            inf_value=float('inf'),
            neg_inf=-float('inf')
        )

        # Only valid value should be stored
        self.assertEqual(len(self.tracker.metrics['valid']), 1)
        self.assertEqual(len(self.tracker.metrics['nan_value']), 0)
        self.assertEqual(len(self.tracker.metrics['inf_value']), 0)
        self.assertEqual(len(self.tracker.metrics['neg_inf']), 0)


class TestPPOLogger(unittest.TestCase):
    """Test PPO logger functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = str(Path(self.temp_dir) / 'logs')

        # Mock TensorBoard writer to avoid dependency issues
        self.mock_writer = Mock()

        # Patch SummaryWriter
        patcher = patch('src.training.ppo.logging.SummaryWriter')
        self.mock_summary_writer = patcher.start()
        self.mock_summary_writer.return_value = self.mock_writer
        self.addCleanup(patcher.stop)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = PPOLogger(
            log_dir=self.log_dir,
            experiment_name='test_experiment'
        )

        self.assertEqual(logger.experiment_name, 'test_experiment')
        self.assertIsInstance(logger.metrics_tracker, MetricsTracker)
        self.assertTrue(logger.log_dir.exists())

        # Check if TensorBoard writer was created (if available)
        if TENSORBOARD_AVAILABLE:
            self.mock_summary_writer.assert_called_once()

    def test_log_training_metrics(self):
        """Test logging training metrics."""
        logger = PPOLogger(self.log_dir, 'test')

        metrics = {
            'policy_loss': 0.5,
            'value_loss': 0.3,
            'entropy_loss': 0.1,
            'kl_divergence': 0.01,
            'clip_fraction': 0.2
        }

        logger.log_training_metrics(step=100, metrics=metrics)

        # Verify metrics were added to tracker
        self.assertIn('policy_loss', logger.metrics_tracker.metrics)
        self.assertEqual(logger.metrics_tracker.metrics['policy_loss'][0], 0.5)

        # Verify TensorBoard logging (if available)
        if TENSORBOARD_AVAILABLE:
            self.assertGreater(self.mock_writer.add_scalar.call_count, 0)

    def test_log_performance_metrics(self):
        """Test logging performance metrics."""
        logger = PPOLogger(self.log_dir, 'test')

        performance = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05,
            'win_rate': 0.6,
            'episode_length': 100
        }

        logger.log_performance_metrics(step=100, performance=performance)

        # Verify metrics were tracked
        self.assertIn('total_return', logger.metrics_tracker.metrics)
        self.assertEqual(logger.metrics_tracker.metrics['sharpe_ratio'][0], 1.2)

    def test_log_curriculum_metrics(self):
        """Test logging curriculum metrics."""
        logger = PPOLogger(self.log_dir, 'test')

        # Mock curriculum level
        mock_level = Mock()
        mock_level.difficulty_level = Mock()
        mock_level.difficulty_level.value = 2

        curriculum_state = {
            'level': mock_level,
            'advancement_rate': 0.1,
            'time_at_level': 1000,
            'performance_threshold': 0.5
        }

        logger.log_curriculum_metrics(step=100, curriculum_state=curriculum_state)

        if TENSORBOARD_AVAILABLE:
            # Should have called add_scalar for curriculum metrics
            scalar_calls = [call[0][0] for call in self.mock_writer.add_scalar.call_args_list]
            self.assertIn('Curriculum/DifficultyLevel', scalar_calls)

    def test_log_environment_metrics(self):
        """Test logging environment metrics."""
        logger = PPOLogger(self.log_dir, 'test')

        env_metrics = {
            'action_distribution': [10, 15, 20, 5],
            'market_volatility': 0.2,
            'trend_strength': 0.8,
            'terminal_reasons': {
                'max_steps': 5,
                'profit_target': 3,
                'stop_loss': 2
            }
        }

        logger.log_environment_metrics(step=100, env_metrics=env_metrics)

        if TENSORBOARD_AVAILABLE:
            # Should have logged action distribution and other metrics
            self.assertGreater(self.mock_writer.add_scalar.call_count, 0)

    def test_log_system_metrics(self):
        """Test logging system metrics."""
        logger = PPOLogger(self.log_dir, 'test')

        logger.log_system_metrics(step=100)

        if TENSORBOARD_AVAILABLE:
            # Should have logged system metrics
            scalar_calls = [call[0][0] for call in self.mock_writer.add_scalar.call_args_list]
            self.assertIn('System/StepsPerSecond', scalar_calls)
            self.assertIn('System/ElapsedTimeMinutes', scalar_calls)

    def test_log_hyperparameters(self):
        """Test logging hyperparameters."""
        logger = PPOLogger(self.log_dir, 'test')

        config = PPOConfig(
            learning_rate=1e-4,
            clip_epsilon=0.1,
            entropy_coef=0.02
        )

        logger.log_hyperparameters(config)

        if TENSORBOARD_AVAILABLE:
            # Should have called add_hparams
            self.mock_writer.add_hparams.assert_called_once()

    @patch('src.training.ppo.logging.plt')
    def test_create_performance_plots(self, mock_plt):
        """Test performance plot creation."""
        config = LoggingConfig(create_plots=True, save_plots=True)
        logger = PPOLogger(self.log_dir, 'test', config=config)

        # Add some sample data
        for i in range(10):
            logger.metrics_tracker.update(
                policy_loss=0.5 - i * 0.01,
                total_return=i * 0.1,
                sharpe_ratio=i * 0.1,
                kl_divergence=0.01 + i * 0.001
            )

        logger.create_performance_plots(step=100)

        # Should have called matplotlib functions
        mock_plt.subplots.assert_called_once()

    def test_create_performance_dashboard(self):
        """Test performance dashboard URL creation."""
        logger = PPOLogger(self.log_dir, 'test')

        dashboard_url = logger.create_performance_dashboard()

        if TENSORBOARD_AVAILABLE:
            self.assertIn('tensorboard --logdir', dashboard_url)
        else:
            self.assertEqual(dashboard_url, "TensorBoard not available")

    def test_flush_and_close(self):
        """Test logger flush and close."""
        logger = PPOLogger(self.log_dir, 'test')

        logger.flush()
        logger.close()

        if TENSORBOARD_AVAILABLE:
            self.mock_writer.flush.assert_called_once()
            self.mock_writer.close.assert_called_once()

    def test_logging_without_tensorboard(self):
        """Test logging functionality without TensorBoard."""
        # Temporarily disable TensorBoard
        with patch('src.training.ppo.logging.TENSORBOARD_AVAILABLE', False):
            logger = PPOLogger(self.log_dir, 'test')

            self.assertIsNone(logger.writer)

            # Should still work without TensorBoard
            metrics = {'policy_loss': 0.5, 'value_loss': 0.3}
            logger.log_training_metrics(step=100, metrics=metrics)

            # Metrics should still be tracked
            self.assertIn('policy_loss', logger.metrics_tracker.metrics)

    def test_custom_logging_config(self):
        """Test logger with custom configuration."""
        config = LoggingConfig(
            flush_secs=10,
            max_queue_size=50,
            create_plots=False
        )

        logger = PPOLogger(self.log_dir, 'test', config=config)

        self.assertEqual(logger.config.flush_secs, 10)
        self.assertEqual(logger.config.max_queue_size, 50)
        self.assertFalse(logger.config.create_plots)

    def test_concurrent_logging(self):
        """Test concurrent logging operations."""
        logger = PPOLogger(self.log_dir, 'test')

        # Simulate concurrent logging from multiple threads
        import threading

        def log_metrics(thread_id):
            for i in range(10):
                metrics = {
                    f'metric_{thread_id}': i * 0.1,
                    'shared_metric': i
                }
                logger.log_training_metrics(step=i + thread_id * 10, metrics=metrics)

        threads = []
        for tid in range(3):
            thread = threading.Thread(target=log_metrics, args=(tid,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All metrics should be tracked
        for tid in range(3):
            self.assertIn(f'metric_{tid}', logger.metrics_tracker.metrics)

        # Shared metric should have multiple values
        self.assertGreater(len(logger.metrics_tracker.metrics['shared_metric']), 10)


class TestLoggingIntegration(unittest.TestCase):
    """Integration tests for logging system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_full_logging_pipeline(self):
        """Test complete logging pipeline."""
        log_dir = str(Path(self.temp_dir) / 'logs')

        # Mock TensorBoard to avoid dependency
        with patch('src.training.ppo.logging.SummaryWriter') as mock_writer:
            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance

            logger = PPOLogger(log_dir, 'integration_test')

            # Simulate training loop with logging
            for step in range(0, 100, 10):
                # Training metrics
                training_metrics = {
                    'policy_loss': 0.5 * np.exp(-step / 50),
                    'value_loss': 0.3 * np.exp(-step / 50),
                    'entropy_loss': 0.1,
                    'kl_divergence': 0.01 + np.random.normal(0, 0.001)
                }
                logger.log_training_metrics(step, training_metrics)

                # Performance metrics every 20 steps
                if step % 20 == 0:
                    performance_metrics = {
                        'total_return': step * 0.001,
                        'sharpe_ratio': min(step * 0.01, 2.0),
                        'max_drawdown': max(step * 0.0005, 0.1),
                        'win_rate': 0.5 + step * 0.001
                    }
                    logger.log_performance_metrics(step, performance_metrics)

                # System metrics
                logger.log_system_metrics(step)

            # Get final statistics
            stats = logger.metrics_tracker.get_statistics()

            self.assertIn('policy_loss', stats)
            self.assertIn('total_return', stats)
            self.assertEqual(stats['policy_loss']['count'], 10)

            # Verify trends
            policy_trend = logger.metrics_tracker.get_trend('policy_loss')
            self.assertEqual(policy_trend, 'decreasing')  # Loss should decrease

            return_trend = logger.metrics_tracker.get_trend('total_return')
            self.assertEqual(return_trend, 'increasing')  # Return should increase

            logger.close()

    def test_logging_with_missing_metrics(self):
        """Test logging behavior with missing or None metrics."""
        log_dir = str(Path(self.temp_dir) / 'logs')

        with patch('src.training.ppo.logging.SummaryWriter'):
            logger = PPOLogger(log_dir, 'missing_metrics_test')

            # Log with missing metrics
            incomplete_metrics = {
                'policy_loss': 0.5,
                'value_loss': None,  # None value
                'entropy_loss': 0.1
                # Missing other expected metrics
            }

            # Should not crash
            logger.log_training_metrics(10, incomplete_metrics)

            # Only valid metrics should be tracked
            self.assertIn('policy_loss', logger.metrics_tracker.metrics)
            self.assertNotIn('value_loss', logger.metrics_tracker.metrics)


if __name__ == '__main__':
    unittest.main()
