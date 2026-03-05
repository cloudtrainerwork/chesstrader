"""
Position Management Evaluation System.

Provides comprehensive evaluation of trained position management policies
with trading-specific metrics, statistical validation, and benchmark comparisons.
Extends the base evaluation infrastructure with options trading analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import logging
from scipy import stats
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PositionEvaluationConfig:
    """Configuration for position evaluation system."""

    # Evaluation parameters
    n_episodes: int = 100  # Episodes per evaluation
    min_episodes_significance: int = 30  # Minimum for statistical significance
    confidence_level: float = 0.95  # Confidence level for intervals

    # Benchmark strategies
    benchmark_hold: bool = True  # Compare against buy-and-hold
    benchmark_simple_rules: bool = True  # Compare against rule-based system

    # Strategy-specific evaluation
    strategy_breakdown: bool = True  # Evaluate by strategy type
    time_decay_analysis: bool = True  # Analyze theta decay impact
    volatility_regime_analysis: bool = True  # Performance by vol regime

    # Risk metrics
    max_drawdown_threshold: float = 0.15  # Alert if drawdown > 15%
    sharpe_min_threshold: float = 1.0  # Minimum acceptable Sharpe ratio
    win_rate_min_threshold: float = 0.55  # Minimum win rate threshold

    # Output options
    save_detailed_results: bool = True
    create_performance_plots: bool = True
    export_trade_log: bool = True


@dataclass
class TradeRecord:
    """Individual trade record for detailed analysis."""

    strategy_type: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    pnl: float
    max_profit: float
    max_loss: float
    hold_duration: int
    exit_reason: str  # 'profit_target', 'stop_loss', 'expiration', 'manual'
    adjustments: int
    max_risk: float
    theta_decay: float
    vega_exposure: float
    entry_iv: float
    exit_iv: float


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""

    # Overall performance
    total_episodes: int
    total_trades: int
    win_rate: float
    avg_return: float
    std_return: float
    sharpe_ratio: float
    max_drawdown: float

    # Risk metrics
    var_95: float  # 95% Value at Risk
    expected_shortfall: float  # Expected loss beyond VaR
    calmar_ratio: float  # Return / Max Drawdown

    # Position management metrics
    avg_hold_time: float
    adjustment_frequency: float
    early_exit_rate: float  # Exits before expiration
    profit_target_hit_rate: float

    # Strategy breakdown
    strategy_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Statistical significance
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)

    # Benchmark comparison
    benchmark_results: Dict[str, float] = field(default_factory=dict)
    outperformance_significance: Dict[str, float] = field(default_factory=dict)

    # Detailed records
    trade_records: List[TradeRecord] = field(default_factory=list)


class PositionEvaluator:
    """
    Comprehensive position management policy evaluator.

    Evaluates trained position management policies with trading-specific metrics,
    statistical significance testing, and benchmark comparisons.
    """

    def __init__(
        self,
        config: Optional[PositionEvaluationConfig] = None,
        benchmark_strategies: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize position evaluator.

        Args:
            config: Evaluation configuration
            benchmark_strategies: Dictionary of benchmark strategy functions
        """
        self.config = config or PositionEvaluationConfig()
        self.benchmark_strategies = benchmark_strategies or {}
        self.results_cache = {}

        # Initialize benchmark strategies if enabled
        if self.config.benchmark_hold:
            self.benchmark_strategies['hold'] = self._hold_strategy

        if self.config.benchmark_simple_rules:
            self.benchmark_strategies['simple_rules'] = self._simple_rules_strategy

        logger.info(f"Position evaluator initialized with {len(self.benchmark_strategies)} benchmarks")

    def evaluate_policy(
        self,
        policy_network: Any,
        env_factory: Callable[[], Any],
        strategy_types: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> EvaluationResults:
        """
        Evaluate position management policy across multiple episodes.

        Args:
            policy_network: Trained position manager network
            env_factory: Factory function for creating environments
            strategy_types: List of strategy types to evaluate on
            save_path: Path to save detailed results

        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting policy evaluation for {self.config.n_episodes} episodes")

        # Run policy evaluation
        policy_results = self._run_policy_evaluation(
            policy_network, env_factory, strategy_types
        )

        # Run benchmark evaluations
        benchmark_results = {}
        for name, strategy in self.benchmark_strategies.items():
            logger.info(f"Evaluating benchmark: {name}")
            benchmark_results[name] = self._run_benchmark_evaluation(
                strategy, env_factory, strategy_types
            )

        # Calculate comprehensive metrics
        results = self._calculate_evaluation_results(
            policy_results, benchmark_results
        )

        # Statistical significance testing
        self._add_statistical_significance(results, policy_results, benchmark_results)

        # Strategy breakdown analysis
        if self.config.strategy_breakdown and strategy_types:
            self._add_strategy_breakdown(results, policy_results, strategy_types)

        # Save detailed results if requested
        if save_path and self.config.save_detailed_results:
            self._save_detailed_results(results, save_path)

        logger.info("Policy evaluation completed")
        return results

    def _run_policy_evaluation(
        self,
        policy_network: Any,
        env_factory: Callable[[], Any],
        strategy_types: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Run policy evaluation episodes."""
        results = {
            'returns': [],
            'trade_records': [],
            'episode_metrics': []
        }

        for episode in range(self.config.n_episodes):
            env = env_factory()

            # Set strategy type if specified
            if strategy_types and hasattr(env, 'set_strategy_type'):
                strategy = np.random.choice(strategy_types)
                env.set_strategy_type(strategy)

            # Run episode with policy
            episode_return, trade_record, metrics = self._run_episode(
                policy_network, env
            )

            results['returns'].append(episode_return)
            if trade_record:
                results['trade_records'].append(trade_record)
            results['episode_metrics'].append(metrics)

        return results

    def _run_episode(
        self,
        policy: Any,
        env: Any
    ) -> Tuple[float, Optional[TradeRecord], Dict[str, float]]:
        """Run single episode with policy."""
        obs = env.reset()
        total_return = 0.0
        episode_length = 0
        adjustments = 0
        max_profit = 0.0
        max_loss = 0.0

        # Track episode metrics
        metrics = {
            'max_position_value': 0.0,
            'max_risk_exposure': 0.0,
            'theta_decay_total': 0.0,
            'vega_exposure_avg': 0.0
        }

        done = False
        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _ = policy.act(obs_tensor)
                action = action.item()

            # Execute action
            next_obs, reward, done, info = env.step(action)

            total_return += reward
            episode_length += 1

            # Track adjustments
            if action in [2, 3]:  # ADJUST or ROLL actions
                adjustments += 1

            # Track profit/loss peaks
            if 'unrealized_pnl' in info:
                pnl = info['unrealized_pnl']
                max_profit = max(max_profit, pnl)
                max_loss = min(max_loss, pnl)

            # Update episode metrics
            if 'position_value' in info:
                metrics['max_position_value'] = max(
                    metrics['max_position_value'],
                    abs(info['position_value'])
                )

            if 'risk_exposure' in info:
                metrics['max_risk_exposure'] = max(
                    metrics['max_risk_exposure'],
                    info['risk_exposure']
                )

            if 'theta_decay' in info:
                metrics['theta_decay_total'] += info['theta_decay']

            obs = next_obs

        # Create trade record if trade completed
        trade_record = None
        if 'episode' in info and info['episode']:
            episode_info = info['episode']
            trade_record = TradeRecord(
                strategy_type=getattr(env, 'strategy_type', 'unknown'),
                entry_time=0,
                exit_time=episode_length,
                entry_price=episode_info.get('entry_price', 0.0),
                exit_price=episode_info.get('exit_price', 0.0),
                pnl=total_return,
                max_profit=max_profit,
                max_loss=max_loss,
                hold_duration=episode_length,
                exit_reason=episode_info.get('exit_reason', 'expiration'),
                adjustments=adjustments,
                max_risk=metrics['max_risk_exposure'],
                theta_decay=metrics['theta_decay_total'],
                vega_exposure=metrics['vega_exposure_avg'],
                entry_iv=episode_info.get('entry_iv', 0.0),
                exit_iv=episode_info.get('exit_iv', 0.0)
            )

        return total_return, trade_record, metrics

    def _run_benchmark_evaluation(
        self,
        benchmark_strategy: Callable,
        env_factory: Callable[[], Any],
        strategy_types: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Run benchmark strategy evaluation."""
        results = {
            'returns': [],
            'trade_records': [],
            'episode_metrics': []
        }

        for episode in range(self.config.n_episodes):
            env = env_factory()

            if strategy_types and hasattr(env, 'set_strategy_type'):
                strategy = np.random.choice(strategy_types)
                env.set_strategy_type(strategy)

            # Run episode with benchmark
            episode_return = self._run_benchmark_episode(benchmark_strategy, env)
            results['returns'].append(episode_return)

        return results

    def _run_benchmark_episode(self, benchmark_strategy: Callable, env: Any) -> float:
        """Run single episode with benchmark strategy."""
        obs = env.reset()
        total_return = 0.0

        done = False
        while not done:
            action = benchmark_strategy(obs, env)
            obs, reward, done, info = env.step(action)
            total_return += reward

        return total_return

    def _hold_strategy(self, obs: np.ndarray, env: Any) -> int:
        """Simple hold strategy benchmark."""
        return 0  # HOLD action

    def _simple_rules_strategy(self, obs: np.ndarray, env: Any) -> int:
        """Simple rule-based strategy benchmark."""
        # Extract position P&L from observation (assume it's in the observation)
        if len(obs) > 20:  # Ensure observation has enough elements
            pnl = obs[20] if len(obs) > 20 else 0.0

            # Simple rules: close if profit > 50% or loss > 30%
            if pnl > 0.5:
                return 1  # CLOSE
            elif pnl < -0.3:
                return 1  # CLOSE
            else:
                return 0  # HOLD

        return 0  # Default to HOLD

    def _calculate_evaluation_results(
        self,
        policy_results: Dict[str, List],
        benchmark_results: Dict[str, Dict[str, List]]
    ) -> EvaluationResults:
        """Calculate comprehensive evaluation metrics."""
        policy_returns = np.array(policy_results['returns'])

        # Basic performance metrics
        results = EvaluationResults(
            total_episodes=len(policy_returns),
            total_trades=len(policy_results['trade_records']),
            win_rate=np.mean(policy_returns > 0),
            avg_return=np.mean(policy_returns),
            std_return=np.std(policy_returns),
            sharpe_ratio=self._calculate_sharpe_ratio(policy_returns),
            max_drawdown=self._calculate_max_drawdown(policy_returns)
        )

        # Risk metrics
        results.var_95 = np.percentile(policy_returns, 5)
        results.expected_shortfall = np.mean(policy_returns[policy_returns <= results.var_95])
        results.calmar_ratio = results.avg_return / results.max_drawdown if results.max_drawdown > 0 else 0

        # Position management metrics
        trade_records = policy_results['trade_records']
        if trade_records:
            results.avg_hold_time = np.mean([t.hold_duration for t in trade_records])
            results.adjustment_frequency = np.mean([t.adjustments for t in trade_records])
            results.early_exit_rate = np.mean([
                1 if t.exit_reason != 'expiration' else 0 for t in trade_records
            ])
            results.profit_target_hit_rate = np.mean([
                1 if t.exit_reason == 'profit_target' else 0 for t in trade_records
            ])

        # Store trade records
        results.trade_records = trade_records

        # Benchmark comparison
        for benchmark_name, benchmark_data in benchmark_results.items():
            benchmark_returns = np.array(benchmark_data['returns'])
            results.benchmark_results[benchmark_name] = np.mean(benchmark_returns)

        return results

    def _add_statistical_significance(
        self,
        results: EvaluationResults,
        policy_results: Dict[str, List],
        benchmark_results: Dict[str, Dict[str, List]]
    ):
        """Add statistical significance testing to results."""
        policy_returns = np.array(policy_results['returns'])

        # Confidence intervals for key metrics
        alpha = 1 - self.config.confidence_level

        # Return confidence interval
        mean_return = np.mean(policy_returns)
        sem_return = stats.sem(policy_returns)
        ci_return = stats.t.interval(
            self.config.confidence_level,
            len(policy_returns) - 1,
            loc=mean_return,
            scale=sem_return
        )
        results.confidence_intervals['avg_return'] = ci_return

        # Sharpe ratio confidence interval (approximate)
        sharpe_sem = np.sqrt((1 + results.sharpe_ratio**2 / 2) / len(policy_returns))
        ci_sharpe = (
            results.sharpe_ratio - stats.norm.ppf(1 - alpha/2) * sharpe_sem,
            results.sharpe_ratio + stats.norm.ppf(1 - alpha/2) * sharpe_sem
        )
        results.confidence_intervals['sharpe_ratio'] = ci_sharpe

        # Benchmark comparison significance
        for benchmark_name, benchmark_data in benchmark_results.items():
            benchmark_returns = np.array(benchmark_data['returns'])

            # T-test for mean difference
            t_stat, p_value = stats.ttest_ind(policy_returns, benchmark_returns)
            results.p_values[f'{benchmark_name}_comparison'] = p_value
            results.outperformance_significance[benchmark_name] = p_value < alpha

    def _add_strategy_breakdown(
        self,
        results: EvaluationResults,
        policy_results: Dict[str, List],
        strategy_types: List[str]
    ):
        """Add strategy-specific performance breakdown."""
        trade_records = policy_results['trade_records']

        for strategy in strategy_types:
            strategy_trades = [t for t in trade_records if t.strategy_type == strategy]

            if strategy_trades:
                strategy_returns = [t.pnl for t in strategy_trades]
                strategy_performance = {
                    'n_trades': len(strategy_trades),
                    'win_rate': np.mean([1 if r > 0 else 0 for r in strategy_returns]),
                    'avg_return': np.mean(strategy_returns),
                    'sharpe_ratio': self._calculate_sharpe_ratio(np.array(strategy_returns)),
                    'max_drawdown': self._calculate_max_drawdown(np.array(strategy_returns)),
                    'avg_hold_time': np.mean([t.hold_duration for t in strategy_trades])
                }

                results.strategy_performance[strategy] = strategy_performance

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _save_detailed_results(self, results: EvaluationResults, save_path: str):
        """Save detailed evaluation results."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save summary results
        summary_path = save_dir / 'evaluation_summary.json'
        import json

        summary_data = {
            'total_episodes': results.total_episodes,
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'avg_return': results.avg_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'confidence_intervals': {k: list(v) for k, v in results.confidence_intervals.items()},
            'benchmark_results': results.benchmark_results,
            'strategy_performance': results.strategy_performance
        }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        # Save trade records if requested
        if self.config.export_trade_log and results.trade_records:
            trades_df = pd.DataFrame([
                {
                    'strategy_type': t.strategy_type,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'pnl': t.pnl,
                    'hold_duration': t.hold_duration,
                    'exit_reason': t.exit_reason,
                    'adjustments': t.adjustments,
                    'max_risk': t.max_risk
                }
                for t in results.trade_records
            ])

            trades_path = save_dir / 'trade_log.csv'
            trades_df.to_csv(trades_path, index=False)

        logger.info(f"Detailed results saved to: {save_dir}")


def evaluate_position_manager(
    position_manager_network: Any,
    env_factory: Callable[[], Any],
    strategy_types: Optional[List[str]] = None,
    config: Optional[PositionEvaluationConfig] = None,
    save_path: Optional[str] = None
) -> EvaluationResults:
    """
    Convenience function to evaluate position manager.

    Args:
        position_manager_network: Trained position manager network
        env_factory: Factory function for creating environments
        strategy_types: List of strategy types to evaluate on
        config: Evaluation configuration
        save_path: Path to save detailed results

    Returns:
        Comprehensive evaluation results
    """
    evaluator = PositionEvaluator(config)
    return evaluator.evaluate_policy(
        position_manager_network,
        env_factory,
        strategy_types,
        save_path
    )