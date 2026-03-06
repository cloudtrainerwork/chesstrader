"""
Comprehensive TensorBoard Logging for PPO Training.

This module implements detailed logging system for PPO training monitoring,
including policy metrics, environment performance, curriculum progression,
and real-time visualization capabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# Avoid circular import - PPOConfig will be passed as needed

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Configuration for logging system."""

    log_dir: str = './logs/ppo_training'
    experiment_name: str = 'options_trading_v1'
    flush_secs: int = 30
    max_queue_size: int = 100
    create_plots: bool = True
    plot_freq: int = 1000  # steps between plot updates
    save_plots: bool = True
    plot_format: str = 'png'


class MetricsTracker:
    """
    Tracks and aggregates metrics over time with rolling statistics.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.global_metrics = defaultdict(list)
        self.last_update = defaultdict(float)

    def update(self, **metrics: Union[float, int]):
        """
        Update metrics with new values.

        Args:
            **metrics: Metric name-value pairs to update
        """
        current_time = time.time()

        for name, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = float(value)

            try:
                if np.isfinite(value):  # Only track finite values
                    self.metrics[name].append(value)
                    self.global_metrics[name].append(value)
                    self.last_update[name] = current_time
            except TypeError:
                continue

    def get_recent_average(self, metric_name: str, window: Optional[int] = None) -> float:
        """
        Get recent average of a metric.

        Args:
            metric_name: Name of metric
            window: Window size (uses default if None)

        Returns:
            Recent average value or 0.0 if no data
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0

        window = window or self.window_size
        recent_values = list(self.metrics[metric_name])[-window:]
        return float(np.mean(recent_values))

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive statistics for all metrics.

        Returns:
            Dict with metric statistics
        """
        stats = {}

        for metric_name, values in self.metrics.items():
            if len(values) == 0:
                continue

            values_array = np.array(list(values))
            stats[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'count': len(values_array),
                'latest': float(values_array[-1])
            }

        return stats

    def get_trend(self, metric_name: str, lookback: int = 20) -> str:
        """
        Get trend direction for a metric.

        Args:
            metric_name: Name of metric
            lookback: Number of recent values to analyze

        Returns:
            Trend direction: 'increasing', 'decreasing', or 'stable'
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 'stable'

        values = np.array(list(self.metrics[metric_name])[-lookback:])
        if len(values) < 2:
            return 'stable'

        # Calculate linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'


class PPOLogger:
    """
    Comprehensive logging system for PPO training.

    Handles TensorBoard logging, metric tracking, and visualization
    for all aspects of PPO training including policy metrics,
    environment performance, and curriculum progression.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[LoggingConfig] = None
    ):
        """
        Initialize PPO logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name of experiment
            config: Logging configuration
        """
        self.config = config or LoggingConfig()
        self.experiment_name = experiment_name

        # Create timestamped log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / f"{experiment_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                flush_secs=self.config.flush_secs,
                max_queue=self.config.max_queue_size
            )
            logger.info(f"TensorBoard logging enabled: {self.log_dir}")
        else:
            self.writer = None
            logger.warning("TensorBoard not available, logging will be limited")

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()

        # Performance tracking
        self.start_time = time.time()
        self.last_log_time = time.time()

        # Plot settings
        if self.config.create_plots and PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            self.plots_dir = self.log_dir / 'plots'
            self.plots_dir.mkdir(exist_ok=True)
        elif self.config.create_plots and not PLOTTING_AVAILABLE:
            logger.warning("Plotting requested but matplotlib not available")

        logger.info(f"PPO Logger initialized: {self.log_dir}")

    def log_training_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Log PPO training metrics.

        Args:
            step: Training step
            metrics: Training metrics dictionary
        """
        # Update metrics tracker
        self.metrics_tracker.update(**metrics)

        if self.writer:
            # Core PPO metrics
            training_metrics = {
                'policy_loss': 'Training/PolicyLoss',
                'value_loss': 'Training/ValueLoss',
                'entropy_loss': 'Training/EntropyLoss',
                'total_loss': 'Training/TotalLoss',
                'kl_divergence': 'Training/KLDivergence',
                'clip_fraction': 'Training/ClipFraction',
                'explained_variance': 'Training/ExplainedVariance',
                'grad_norm': 'Training/GradientNorm'
            }

            for metric_key, tensorboard_key in training_metrics.items():
                if metric_key in metrics:
                    self.writer.add_scalar(tensorboard_key, metrics[metric_key], step)

            # Advantage and value metrics
            advantage_metrics = {
                'advantage_mean': 'Training/AdvantageeMean',
                'advantage_std': 'Training/AdvantageStd',
                'value_mean': 'Training/ValueMean',
                'value_std': 'Training/ValueStd',
                'returns_mean': 'Training/ReturnsMean',
                'returns_std': 'Training/ReturnsStd'
            }

            for metric_key, tensorboard_key in advantage_metrics.items():
                if metric_key in metrics:
                    self.writer.add_scalar(tensorboard_key, metrics[metric_key], step)

            # Learning rate and optimization metrics
            if 'learning_rate' in metrics:
                self.writer.add_scalar('Training/LearningRate', metrics['learning_rate'], step)

            if 'epoch' in metrics:
                self.writer.add_scalar('Training/Epoch', metrics['epoch'], step)

        logger.debug(f"Training metrics logged at step {step}")

    def log_performance_metrics(self, step: int, performance: Dict[str, float]):
        """
        Log environment and trading performance metrics.

        Args:
            step: Training step
            performance: Performance metrics dictionary
        """
        # Update metrics tracker
        self.metrics_tracker.update(**performance)

        if self.writer:
            # Episode metrics
            episode_metrics = {
                'total_return': 'Performance/EpisodeReturn',
                'episode_length': 'Performance/EpisodeLength',
                'win_rate': 'Performance/WinRate',
                'average_episode_length': 'Performance/AverageEpisodeLength'
            }

            for metric_key, tensorboard_key in episode_metrics.items():
                if metric_key in performance:
                    self.writer.add_scalar(tensorboard_key, performance[metric_key], step)

            # Financial metrics
            financial_metrics = {
                'sharpe_ratio': 'Performance/SharpeRatio',
                'max_drawdown': 'Performance/MaxDrawdown',
                'sortino_ratio': 'Performance/SortinoRatio',
                'calmar_ratio': 'Performance/CalmarRatio',
                'total_pnl': 'Performance/TotalPnL',
                'cumulative_return': 'Performance/CumulativeReturn'
            }

            for metric_key, tensorboard_key in financial_metrics.items():
                if metric_key in performance:
                    self.writer.add_scalar(tensorboard_key, performance[metric_key], step)

            # Risk metrics
            risk_metrics = {
                'volatility': 'Risk/Volatility',
                'var_95': 'Risk/VaR95',
                'var_99': 'Risk/VaR99',
                'beta': 'Risk/Beta',
                'correlation': 'Risk/Correlation'
            }

            for metric_key, tensorboard_key in risk_metrics.items():
                if metric_key in performance:
                    self.writer.add_scalar(tensorboard_key, performance[metric_key], step)

        logger.debug(f"Performance metrics logged at step {step}")

    def log_curriculum_metrics(self, step: int, curriculum_state: Dict[str, Any]):
        """
        Log curriculum learning progression metrics.

        Args:
            step: Training step
            curriculum_state: Curriculum state dictionary
        """
        if self.writer:
            # Curriculum level and progression
            if 'level' in curriculum_state:
                level_info = curriculum_state['level']
                if hasattr(level_info, 'difficulty_level'):
                    self.writer.add_scalar(
                        'Curriculum/DifficultyLevel',
                        level_info.difficulty_level.value,
                        step
                    )

            if 'advancement_rate' in curriculum_state:
                self.writer.add_scalar(
                    'Curriculum/AdvancementRate',
                    curriculum_state['advancement_rate'],
                    step
                )

            if 'time_at_level' in curriculum_state:
                self.writer.add_scalar(
                    'Curriculum/TimeAtLevel',
                    curriculum_state['time_at_level'],
                    step
                )

            # Performance thresholds
            if 'performance_threshold' in curriculum_state:
                self.writer.add_scalar(
                    'Curriculum/PerformanceThreshold',
                    curriculum_state['performance_threshold'],
                    step
                )

            # Strategy complexity metrics
            strategy_metrics = {
                'max_positions': 'Curriculum/MaxPositions',
                'strategy_count': 'Curriculum/StrategyCount',
                'complexity_score': 'Curriculum/ComplexityScore'
            }

            for metric_key, tensorboard_key in strategy_metrics.items():
                if metric_key in curriculum_state:
                    self.writer.add_scalar(
                        tensorboard_key,
                        curriculum_state[metric_key],
                        step
                    )

        logger.debug(f"Curriculum metrics logged at step {step}")

    def log_environment_metrics(self, step: int, env_metrics: Dict[str, Any]):
        """
        Log environment-specific metrics.

        Args:
            step: Training step
            env_metrics: Environment metrics dictionary
        """
        if self.writer:
            # Action distribution
            if 'action_distribution' in env_metrics:
                action_dist = env_metrics['action_distribution']
                if isinstance(action_dist, (list, np.ndarray)):
                    for i, count in enumerate(action_dist):
                        self.writer.add_scalar(f'Environment/Action_{i}_Count', count, step)

            # Market condition metrics
            market_metrics = {
                'market_volatility': 'Environment/MarketVolatility',
                'trend_strength': 'Environment/TrendStrength',
                'price_range': 'Environment/PriceRange',
                'volume_profile': 'Environment/VolumeProfile'
            }

            for metric_key, tensorboard_key in market_metrics.items():
                if metric_key in env_metrics:
                    self.writer.add_scalar(tensorboard_key, env_metrics[metric_key], step)

            # Terminal condition analysis
            if 'terminal_reasons' in env_metrics:
                terminal_reasons = env_metrics['terminal_reasons']
                for reason, count in terminal_reasons.items():
                    self.writer.add_scalar(f'Environment/Terminal_{reason}', count, step)

    def log_system_metrics(self, step: int):
        """
        Log system performance metrics.

        Args:
            step: Training step
        """
        if self.writer:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            time_since_last = current_time - self.last_log_time

            # Training speed
            steps_per_second = step / elapsed_time if elapsed_time > 0 else 0
            self.writer.add_scalar('System/StepsPerSecond', steps_per_second, step)
            self.writer.add_scalar('System/ElapsedTimeMinutes', elapsed_time / 60, step)

            # Memory usage (if available)
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent()
                self.writer.add_scalar('System/MemoryUsagePercent', memory_percent, step)
                self.writer.add_scalar('System/CPUUsagePercent', cpu_percent, step)
            except ImportError:
                pass

            # GPU usage (if available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_cached = torch.cuda.memory_cached() / 1024**2  # MB
                self.writer.add_scalar('System/GPUMemoryMB', gpu_memory, step)
                self.writer.add_scalar('System/GPUCachedMB', gpu_cached, step)

            self.last_log_time = current_time

    def log_hyperparameters(self, config: Any):
        """
        Log hyperparameters to TensorBoard.

        Args:
            config: PPO configuration object
        """
        if self.writer:
            hparams = {
                'learning_rate': config.learning_rate,
                'clip_epsilon': config.clip_epsilon,
                'entropy_coef': config.entropy_coef,
                'value_loss_coef': config.value_loss_coef,
                'max_grad_norm': config.max_grad_norm,
                'n_steps': config.n_steps,
                'n_epochs': config.n_epochs,
                'batch_size': config.batch_size,
                'gamma': config.gamma,
                'gae_lambda': config.gae_lambda,
                'n_envs': config.n_envs
            }

            # Add hparams to TensorBoard
            self.writer.add_hparams(hparams, {})

            # Log as scalars at step 0 for easy reference
            for param_name, param_value in hparams.items():
                self.writer.add_scalar(f'Hyperparameters/{param_name}', param_value, 0)

        logger.info("Hyperparameters logged")

    def create_performance_plots(self, step: int):
        """
        Create and save performance visualization plots.

        Args:
            step: Training step
        """
        if not self.config.create_plots or not PLOTTING_AVAILABLE:
            return

        try:
            # Get recent statistics
            stats = self.metrics_tracker.get_statistics()

            if not stats:
                return

            # Create performance dashboard
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'PPO Training Performance Dashboard - Step {step}', fontsize=16)

            # Plot 1: Training losses
            ax = axes[0, 0]
            if 'policy_loss' in stats:
                policy_loss = list(self.metrics_tracker.metrics['policy_loss'])
                ax.plot(policy_loss, label='Policy Loss', alpha=0.7)

            if 'value_loss' in stats:
                value_loss = list(self.metrics_tracker.metrics['value_loss'])
                ax.plot(value_loss, label='Value Loss', alpha=0.7)

            ax.set_title('Training Losses')
            ax.set_xlabel('Updates')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Episode returns
            ax = axes[0, 1]
            if 'total_return' in stats:
                returns = list(self.metrics_tracker.metrics['total_return'])
                ax.plot(returns, alpha=0.7, color='green')
                ax.set_title('Episode Returns')
                ax.set_xlabel('Episodes')
                ax.set_ylabel('Return')
                ax.grid(True, alpha=0.3)

                # Add trend line
                if len(returns) > 10:
                    z = np.polyfit(range(len(returns)), returns, 1)
                    p = np.poly1d(z)
                    ax.plot(range(len(returns)), p(range(len(returns))), '--', alpha=0.5, color='red')

            # Plot 3: Performance metrics
            ax = axes[0, 2]
            performance_metrics = ['sharpe_ratio', 'win_rate', 'max_drawdown']
            for i, metric in enumerate(performance_metrics):
                if metric in stats:
                    values = list(self.metrics_tracker.metrics[metric])
                    ax.plot(values, label=metric.replace('_', ' ').title(), alpha=0.7)

            ax.set_title('Performance Metrics')
            ax.set_xlabel('Evaluations')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 4: KL divergence and entropy
            ax = axes[1, 0]
            if 'kl_divergence' in stats:
                kl_div = list(self.metrics_tracker.metrics['kl_divergence'])
                ax.plot(kl_div, label='KL Divergence', alpha=0.7)

            if 'entropy_loss' in stats:
                entropy = list(self.metrics_tracker.metrics['entropy_loss'])
                ax.plot(entropy, label='Entropy', alpha=0.7)

            ax.set_title('Policy Exploration')
            ax.set_xlabel('Updates')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 5: Curriculum progression
            ax = axes[1, 1]
            # This would need curriculum-specific data
            ax.set_title('Curriculum Progression')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Difficulty Level')
            ax.grid(True, alpha=0.3)

            # Plot 6: System metrics
            ax = axes[1, 2]
            if hasattr(self, '_system_metrics'):
                ax.plot(self._system_metrics.get('steps_per_second', []),
                       label='Steps/Sec', alpha=0.7)
                ax.set_title('Training Speed')
                ax.set_xlabel('Time')
                ax.set_ylabel('Steps/Second')
                ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            if self.config.save_plots:
                plot_path = self.plots_dir / f'performance_dashboard_step_{step}.{self.config.plot_format}'
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')

            plt.close()

            logger.debug(f"Performance plots created for step {step}")

        except Exception as e:
            logger.error(f"Failed to create performance plots: {e}")

    def create_performance_dashboard(self) -> str:
        """
        Create performance dashboard URL.

        Returns:
            URL to TensorBoard dashboard
        """
        if self.writer:
            return f"tensorboard --logdir {self.log_dir}"
        else:
            return "TensorBoard not available"

    def flush(self):
        """Flush all pending logs."""
        if self.writer:
            self.writer.flush()

    def close(self):
        """Close logger and cleanup resources."""
        if self.writer:
            self.writer.close()
        logger.info("PPO Logger closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
