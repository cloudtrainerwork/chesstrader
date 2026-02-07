"""
Evaluation Metrics and Validation for PPO-trained Options Trading Agents.

This module implements comprehensive evaluation system for assessing agent
performance, including out-of-sample validation, risk-adjusted metrics,
statistical significance testing, and benchmark comparisons.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy import stats
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """
    Comprehensive evaluation results for a trading agent.
    """
    # Episode-level metrics
    episode_returns: List[float]
    episode_lengths: List[float]
    episode_sharpe_ratios: List[float]

    # Aggregate performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    average_episode_length: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    beta: float = 0.0
    correlation: float = 0.0

    # Trading-specific metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Strategy-specific metrics
    strategy_specific_metrics: Dict[str, float] = field(default_factory=dict)

    # Statistical confidence
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Evaluation metadata
    evaluation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    n_episodes: int = 0
    total_evaluation_time: float = 0.0
    market_conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, np.ndarray):
                result[field_name] = field_value.tolist()
            elif isinstance(field_value, (np.integer, np.floating)):
                result[field_name] = float(field_value)
            else:
                result[field_name] = field_value
        return result

    def summary(self) -> str:
        """Generate a summary report."""
        return f"""
        Evaluation Summary:
        Total Return: {self.total_return:.2%}
        Sharpe Ratio: {self.sharpe_ratio:.3f}
        Max Drawdown: {self.max_drawdown:.2%}
        Win Rate: {self.win_rate:.2%}
        Total Episodes: {self.n_episodes}
        """


class PerformanceMetrics:
    """
    Static methods for calculating various performance and risk metrics.
    """

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Array of returns
            target_return: Target return threshold

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0

        return float(np.mean(excess_returns) / downside_deviation * np.sqrt(252))

    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Array of returns

        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0

        annualized_return = PerformanceMetrics.annualized_return(returns)
        max_dd = PerformanceMetrics.max_drawdown(returns)

        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0.0

        return float(annualized_return / abs(max_dd))

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown (positive value)
        """
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return float(abs(np.min(drawdown)))

    @staticmethod
    def annualized_return(returns: np.ndarray) -> float:
        """
        Calculate annualized return.

        Args:
            returns: Array of returns

        Returns:
            Annualized return
        """
        if len(returns) == 0:
            return 0.0

        total_return = np.prod(1 + returns) - 1
        n_periods = len(returns)
        return float((1 + total_return) ** (252 / n_periods) - 1)

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array of returns
            confidence_level: Confidence level

        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0

        return float(np.percentile(returns, (1 - confidence_level) * 100))

    @staticmethod
    def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns: Array of returns
            confidence_level: Confidence level

        Returns:
            Expected shortfall
        """
        if len(returns) == 0:
            return 0.0

        var = PerformanceMetrics.value_at_risk(returns, confidence_level)
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var

        return float(np.mean(tail_returns))

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            returns: Array of returns

        Returns:
            Profit factor
        """
        if len(returns) == 0:
            return 0.0

        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]

        gross_profit = np.sum(winning_returns) if len(winning_returns) > 0 else 0.0
        gross_loss = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 0.0

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Calculate win rate.

        Args:
            returns: Array of returns

        Returns:
            Win rate (proportion of positive returns)
        """
        if len(returns) == 0:
            return 0.0

        return float(np.mean(returns > 0))


class AgentEvaluator:
    """
    Comprehensive evaluation system for trading agents.

    Provides out-of-sample validation, statistical testing, and
    comprehensive performance analysis.
    """

    def __init__(
        self,
        env_factory: Callable,
        n_eval_episodes: int = 50,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 1000,
        parallel_evaluation: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize agent evaluator.

        Args:
            env_factory: Function that creates evaluation environments
            n_eval_episodes: Number of episodes for evaluation
            confidence_level: Confidence level for statistical tests
            bootstrap_samples: Number of bootstrap samples
            parallel_evaluation: Whether to use parallel evaluation
            max_workers: Maximum number of worker threads
        """
        self.env_factory = env_factory
        self.n_eval_episodes = n_eval_episodes
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.parallel_evaluation = parallel_evaluation
        self.max_workers = max_workers

        # Risk-free rate (can be configurable)
        self.risk_free_rate = 0.02

        logger.info(f"AgentEvaluator initialized with {n_eval_episodes} episodes")

    def evaluate_agent(
        self,
        policy_network: torch.nn.Module,
        deterministic: bool = True,
        render: bool = False,
        save_results: bool = True,
        results_dir: Optional[str] = None
    ) -> EvaluationResults:
        """
        Comprehensive agent evaluation.

        Args:
            policy_network: Policy network to evaluate
            deterministic: Whether to use deterministic actions
            render: Whether to render episodes
            save_results: Whether to save results to file
            results_dir: Directory to save results

        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting agent evaluation with {self.n_eval_episodes} episodes")
        start_time = time.time()

        # Collect episode data
        if self.parallel_evaluation:
            episode_data = self._evaluate_parallel(policy_network, deterministic, render)
        else:
            episode_data = self._evaluate_sequential(policy_network, deterministic, render)

        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(episode_data)
        results.total_evaluation_time = time.time() - start_time
        results.n_episodes = len(episode_data)

        # Add confidence intervals
        results.confidence_intervals = self._calculate_confidence_intervals(episode_data)

        # Save results if requested
        if save_results:
            self._save_evaluation_results(results, results_dir)

        logger.info(f"Evaluation completed in {results.total_evaluation_time:.2f} seconds")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.3f}, Max DD: {results.max_drawdown:.2%}")

        return results

    def _evaluate_sequential(
        self,
        policy_network: torch.nn.Module,
        deterministic: bool,
        render: bool
    ) -> List[Dict[str, Any]]:
        """Sequential episode evaluation."""
        episode_data = []

        for episode in range(self.n_eval_episodes):
            episode_result = self._run_single_episode(
                policy_network, deterministic, render, episode
            )
            episode_data.append(episode_result)

            if (episode + 1) % 10 == 0:
                logger.info(f"Completed {episode + 1}/{self.n_eval_episodes} episodes")

        return episode_data

    def _evaluate_parallel(
        self,
        policy_network: torch.nn.Module,
        deterministic: bool,
        render: bool
    ) -> List[Dict[str, Any]]:
        """Parallel episode evaluation."""
        episode_data = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all episodes
            futures = {
                executor.submit(
                    self._run_single_episode,
                    policy_network, deterministic, False, episode  # No rendering in parallel
                ): episode
                for episode in range(self.n_eval_episodes)
            }

            # Collect results
            for future in as_completed(futures):
                episode = futures[future]
                try:
                    episode_result = future.result()
                    episode_data.append(episode_result)

                    if len(episode_data) % 10 == 0:
                        logger.info(f"Completed {len(episode_data)}/{self.n_eval_episodes} episodes")

                except Exception as e:
                    logger.error(f"Episode {episode} failed: {e}")

        return episode_data

    def _run_single_episode(
        self,
        policy_network: torch.nn.Module,
        deterministic: bool,
        render: bool,
        episode_id: int
    ) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        env = self.env_factory()
        obs = env.reset()

        episode_rewards = []
        episode_actions = []
        episode_values = []
        step_count = 0
        done = False

        policy_network.eval()
        with torch.no_grad():
            while not done:
                # Get action from policy
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, log_prob, value, _ = policy_network(obs_tensor)

                if deterministic:
                    if hasattr(policy_network, 'get_deterministic_action'):
                        action = policy_network.get_deterministic_action(obs_tensor)
                    # Otherwise use the action as is (should be deterministic for eval)

                if isinstance(action, torch.Tensor):
                    action_np = action.cpu().numpy()[0]
                else:
                    action_np = action

                # Step environment
                next_obs, reward, done, info = env.step(action_np)

                episode_rewards.append(reward)
                episode_actions.append(action_np)
                if isinstance(value, torch.Tensor):
                    episode_values.append(value.cpu().numpy()[0])

                obs = next_obs
                step_count += 1

                if render:
                    env.render()

                # Safety check for very long episodes
                if step_count >= 10000:
                    logger.warning(f"Episode {episode_id} exceeded 10000 steps, terminating")
                    done = True

        env.close()

        # Calculate episode metrics
        episode_return = sum(episode_rewards)
        episode_rewards_array = np.array(episode_rewards)

        # Episode-specific risk metrics
        episode_sharpe = PerformanceMetrics.sharpe_ratio(episode_rewards_array)
        episode_max_dd = PerformanceMetrics.max_drawdown(episode_rewards_array)

        return {
            'episode_id': episode_id,
            'episode_return': episode_return,
            'episode_length': step_count,
            'episode_rewards': episode_rewards,
            'episode_actions': episode_actions,
            'episode_values': episode_values,
            'episode_sharpe': episode_sharpe,
            'episode_max_drawdown': episode_max_dd,
            'info': info if 'info' in locals() else {}
        }

    def _calculate_comprehensive_metrics(self, episode_data: List[Dict[str, Any]]) -> EvaluationResults:
        """Calculate comprehensive performance metrics from episode data."""

        # Extract basic metrics
        episode_returns = [ep['episode_return'] for ep in episode_data]
        episode_lengths = [ep['episode_length'] for ep in episode_data]
        episode_sharpe_ratios = [ep['episode_sharpe'] for ep in episode_data]

        # Convert to numpy arrays
        returns_array = np.array(episode_returns)
        lengths_array = np.array(episode_lengths)

        # Basic performance metrics
        total_return = float(np.mean(returns_array))
        annualized_return = PerformanceMetrics.annualized_return(returns_array)

        # Risk-adjusted metrics
        sharpe_ratio = PerformanceMetrics.sharpe_ratio(returns_array, self.risk_free_rate)
        sortino_ratio = PerformanceMetrics.sortino_ratio(returns_array)
        calmar_ratio = PerformanceMetrics.calmar_ratio(returns_array)
        max_drawdown = PerformanceMetrics.max_drawdown(returns_array)

        # Risk metrics
        volatility = float(np.std(returns_array) * np.sqrt(252))
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = float(np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 0 else 0.0

        var_95 = PerformanceMetrics.value_at_risk(returns_array, 0.95)
        var_99 = PerformanceMetrics.value_at_risk(returns_array, 0.99)
        expected_shortfall_95 = PerformanceMetrics.expected_shortfall(returns_array, 0.95)

        # Trading metrics
        win_rate = PerformanceMetrics.win_rate(returns_array)
        profit_factor = PerformanceMetrics.profit_factor(returns_array)

        # Trade analysis
        winning_trades = np.sum(returns_array > 0)
        losing_trades = np.sum(returns_array < 0)
        total_trades = len(returns_array)

        winning_returns = returns_array[returns_array > 0]
        losing_returns = returns_array[returns_array < 0]

        average_win = float(np.mean(winning_returns)) if len(winning_returns) > 0 else 0.0
        average_loss = float(np.mean(losing_returns)) if len(losing_returns) > 0 else 0.0
        largest_win = float(np.max(returns_array)) if len(returns_array) > 0 else 0.0
        largest_loss = float(np.min(returns_array)) if len(returns_array) > 0 else 0.0

        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_runs(returns_array > 0, True)
        consecutive_losses = self._calculate_consecutive_runs(returns_array <= 0, True)

        # Create results object
        results = EvaluationResults(
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            episode_sharpe_ratios=episode_sharpe_ratios,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_episode_length=float(np.mean(lengths_array)),
            volatility=volatility,
            downside_deviation=downside_deviation,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=expected_shortfall_95,
            total_trades=int(total_trades),
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=int(consecutive_wins),
            consecutive_losses=int(consecutive_losses)
        )

        return results

    def _calculate_consecutive_runs(self, boolean_array: np.ndarray, target_value: bool) -> int:
        """Calculate maximum consecutive runs of target value."""
        if len(boolean_array) == 0:
            return 0

        runs = []
        current_run = 0

        for value in boolean_array:
            if value == target_value:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0

        if current_run > 0:
            runs.append(current_run)

        return max(runs) if runs else 0

    def _calculate_confidence_intervals(self, episode_data: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals using bootstrap resampling."""
        episode_returns = np.array([ep['episode_return'] for ep in episode_data])

        confidence_intervals = {}

        # Bootstrap metrics
        metrics_to_bootstrap = {
            'total_return': lambda x: np.mean(x),
            'sharpe_ratio': lambda x: PerformanceMetrics.sharpe_ratio(x, self.risk_free_rate),
            'max_drawdown': PerformanceMetrics.max_drawdown,
            'win_rate': PerformanceMetrics.win_rate,
            'profit_factor': PerformanceMetrics.profit_factor
        }

        for metric_name, metric_func in metrics_to_bootstrap.items():
            bootstrap_results = []

            for _ in range(self.bootstrap_samples):
                # Resample with replacement
                resampled_returns = np.random.choice(
                    episode_returns, size=len(episode_returns), replace=True
                )

                try:
                    metric_value = metric_func(resampled_returns)
                    if np.isfinite(metric_value):
                        bootstrap_results.append(metric_value)
                except:
                    continue

            if bootstrap_results:
                alpha = 1 - self.confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100

                lower_bound = np.percentile(bootstrap_results, lower_percentile)
                upper_bound = np.percentile(bootstrap_results, upper_percentile)

                confidence_intervals[metric_name] = (float(lower_bound), float(upper_bound))

        return confidence_intervals

    def compare_to_benchmark(
        self,
        agent_results: EvaluationResults,
        benchmark_results: EvaluationResults,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Statistical comparison of agent performance to benchmark.

        Args:
            agent_results: Agent evaluation results
            benchmark_results: Benchmark evaluation results
            significance_level: Statistical significance level

        Returns:
            Comparison results with statistical tests
        """
        comparison = {
            'performance_comparison': {},
            'statistical_tests': {},
            'summary': {}
        }

        # Compare key metrics
        metrics_to_compare = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
            'volatility', 'sortino_ratio', 'calmar_ratio'
        ]

        for metric in metrics_to_compare:
            agent_value = getattr(agent_results, metric)
            benchmark_value = getattr(benchmark_results, metric)

            comparison['performance_comparison'][metric] = {
                'agent': agent_value,
                'benchmark': benchmark_value,
                'difference': agent_value - benchmark_value,
                'relative_difference': (agent_value - benchmark_value) / abs(benchmark_value)
                                     if benchmark_value != 0 else float('inf')
            }

        # Statistical tests for returns
        agent_returns = np.array(agent_results.episode_returns)
        benchmark_returns = np.array(benchmark_results.episode_returns)

        # T-test for mean difference
        try:
            t_stat, p_value_ttest = stats.ttest_ind(agent_returns, benchmark_returns)
            comparison['statistical_tests']['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(p_value_ttest),
                'significant': p_value_ttest < significance_level
            }
        except:
            comparison['statistical_tests']['t_test'] = None

        # Wilcoxon rank-sum test (non-parametric)
        try:
            stat, p_value_wilcoxon = stats.ranksums(agent_returns, benchmark_returns)
            comparison['statistical_tests']['wilcoxon'] = {
                'statistic': float(stat),
                'p_value': float(p_value_wilcoxon),
                'significant': p_value_wilcoxon < significance_level
            }
        except:
            comparison['statistical_tests']['wilcoxon'] = None

        # Summary
        agent_better_metrics = sum(
            1 for metric in metrics_to_compare
            if getattr(agent_results, metric) > getattr(benchmark_results, metric)
        )

        comparison['summary'] = {
            'agent_outperforms_count': agent_better_metrics,
            'total_metrics_compared': len(metrics_to_compare),
            'statistical_significance': any(
                test['significant'] for test in comparison['statistical_tests'].values()
                if test is not None
            )
        }

        return comparison

    def generate_evaluation_report(self, results: EvaluationResults) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Evaluation results

        Returns:
            Formatted evaluation report
        """
        report = f"""
# Trading Agent Evaluation Report

**Evaluation Date:** {results.evaluation_date}
**Episodes:** {results.n_episodes}
**Evaluation Time:** {results.total_evaluation_time:.2f} seconds

## Performance Summary

### Returns
- **Total Return:** {results.total_return:.2%}
- **Annualized Return:** {results.annualized_return:.2%}
- **Average Episode Return:** {np.mean(results.episode_returns):.4f}

### Risk-Adjusted Performance
- **Sharpe Ratio:** {results.sharpe_ratio:.3f}
- **Sortino Ratio:** {results.sortino_ratio:.3f}
- **Calmar Ratio:** {results.calmar_ratio:.3f}

### Risk Metrics
- **Maximum Drawdown:** {results.max_drawdown:.2%}
- **Volatility (Annualized):** {results.volatility:.2%}
- **VaR (95%):** {results.var_95:.4f}
- **VaR (99%):** {results.var_99:.4f}
- **Expected Shortfall (95%):** {results.expected_shortfall_95:.4f}

### Trading Performance
- **Win Rate:** {results.win_rate:.2%}
- **Profit Factor:** {results.profit_factor:.2f}
- **Total Trades:** {results.total_trades}
- **Winning Trades:** {results.winning_trades}
- **Losing Trades:** {results.losing_trades}
- **Average Win:** {results.average_win:.4f}
- **Average Loss:** {results.average_loss:.4f}
- **Largest Win:** {results.largest_win:.4f}
- **Largest Loss:** {results.largest_loss:.4f}
- **Max Consecutive Wins:** {results.consecutive_wins}
- **Max Consecutive Losses:** {results.consecutive_losses}

### Episode Statistics
- **Average Episode Length:** {results.average_episode_length:.1f}
- **Episode Length Std:** {np.std(results.episode_lengths):.1f}

## Confidence Intervals

"""

        # Add confidence intervals
        for metric, (lower, upper) in results.confidence_intervals.items():
            report += f"- **{metric.replace('_', ' ').title()}:** [{lower:.4f}, {upper:.4f}]\n"

        # Add strategy-specific metrics if available
        if results.strategy_specific_metrics:
            report += "\n## Strategy-Specific Metrics\n\n"
            for metric, value in results.strategy_specific_metrics.items():
                report += f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n"

        report += "\n---\n*Report generated automatically by PPO Agent Evaluator*\n"

        return report

    def validate_statistical_significance(self, results: EvaluationResults) -> Dict[str, Any]:
        """
        Validate statistical significance of performance claims.

        Args:
            results: Evaluation results

        Returns:
            Statistical validation results
        """
        validation = {
            'sample_size_adequacy': {},
            'distribution_tests': {},
            'performance_significance': {},
            'recommendations': []
        }

        episode_returns = np.array(results.episode_returns)

        # Sample size adequacy
        n_episodes = len(episode_returns)
        validation['sample_size_adequacy'] = {
            'sample_size': n_episodes,
            'adequate_for_basic_stats': n_episodes >= 30,
            'adequate_for_robust_inference': n_episodes >= 50,
            'recommended_minimum': 100
        }

        # Normality tests
        try:
            shapiro_stat, shapiro_p = stats.shapiro(episode_returns)
            validation['distribution_tests']['normality'] = {
                'test': 'Shapiro-Wilk',
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
        except:
            validation['distribution_tests']['normality'] = None

        # Test if mean return is significantly different from zero
        try:
            t_stat, p_value = stats.ttest_1samp(episode_returns, 0)
            validation['performance_significance']['return_significance'] = {
                'test': 'One-sample t-test',
                'null_hypothesis': 'Mean return equals zero',
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_positive_return': (p_value < 0.05) and (t_stat > 0),
                'significant_negative_return': (p_value < 0.05) and (t_stat < 0)
            }
        except:
            validation['performance_significance']['return_significance'] = None

        # Recommendations based on results
        if n_episodes < 50:
            validation['recommendations'].append(
                "Consider increasing sample size for more robust statistical inference"
            )

        if results.sharpe_ratio > 1.0:
            validation['recommendations'].append(
                "High Sharpe ratio detected - validate with out-of-sample testing"
            )

        if results.max_drawdown > 0.2:
            validation['recommendations'].append(
                "High maximum drawdown - consider risk management improvements"
            )

        return validation

    def _save_evaluation_results(self, results: EvaluationResults, results_dir: Optional[str]):
        """Save evaluation results to file."""
        if results_dir:
            results_path = Path(results_dir)
        else:
            results_path = Path('./evaluation_results')

        results_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = results_path / f"evaluation_results_{timestamp}.json"

        with open(json_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

        # Save text report
        report_file = results_path / f"evaluation_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_evaluation_report(results))

        logger.info(f"Evaluation results saved to {results_path}")