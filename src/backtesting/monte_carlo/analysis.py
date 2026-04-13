"""
Statistical analysis and confidence intervals for Monte Carlo results

Provides comprehensive statistical analysis including confidence intervals,
significance testing, and risk assessment for backtesting validation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Statistical analyzer for Monte Carlo simulation results

    Provides confidence intervals, significance testing, and comprehensive
    risk analysis to validate backtesting results and quantify uncertainty.
    """

    def __init__(self, confidence_levels: List[int] = [90, 95, 99]):
        """
        Initialize statistical analyzer

        Args:
            confidence_levels: List of confidence levels for interval calculation
        """
        self.confidence_levels = confidence_levels
        self.results = {}

        logger.info(f"StatisticalAnalyzer initialized with confidence levels: {confidence_levels}")

    def calculate_confidence_intervals(self, monte_carlo_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate confidence intervals for Monte Carlo performance metrics

        Args:
            monte_carlo_results: DataFrame with simulation results

        Returns:
            Dictionary with confidence intervals for each metric
        """
        if len(monte_carlo_results) == 0:
            return {}

        logger.info(f"Calculating confidence intervals for {len(monte_carlo_results)} simulations")

        confidence_intervals = {}

        # Define metrics to analyze
        numeric_columns = monte_carlo_results.select_dtypes(include=[np.number]).columns
        metrics_to_analyze = [col for col in numeric_columns if col not in ['simulation_id']]

        for metric in metrics_to_analyze:
            if metric in monte_carlo_results.columns:
                values = monte_carlo_results[metric].values
                values = values[~np.isnan(values)]  # Remove NaN values

                if len(values) == 0:
                    continue

                metric_intervals = {}
                mean_value = np.mean(values)

                for confidence_level in self.confidence_levels:
                    # Calculate percentile-based confidence intervals
                    alpha = (100 - confidence_level) / 2
                    lower_percentile = alpha
                    upper_percentile = 100 - alpha

                    lower_bound = np.percentile(values, lower_percentile)
                    upper_bound = np.percentile(values, upper_percentile)

                    metric_intervals[f'{confidence_level}%'] = {
                        'lower': lower_bound,
                        'upper': upper_bound,
                        'mean': mean_value,
                        'std': np.std(values),
                        'median': np.median(values)
                    }

                confidence_intervals[metric] = metric_intervals

        return confidence_intervals

    def statistical_significance_test(self,
                                    strategy_returns: np.ndarray,
                                    benchmark_returns: np.ndarray,
                                    test_type: str = 't_test',
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test statistical significance of strategy vs benchmark

        Args:
            strategy_returns: Strategy performance results
            benchmark_returns: Benchmark performance results
            test_type: Type of statistical test ('t_test' or 'wilcoxon')
            alpha: Significance level

        Returns:
            Dictionary with test results and significance determination
        """
        # Handle edge cases
        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return {
                'error': 'Insufficient data for significance testing',
                'test_type': test_type,
                'confidence_level': 1 - alpha
            }

        if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
            return {
                'error': 'Need at least 2 samples per group',
                'test_type': test_type,
                'confidence_level': 1 - alpha
            }

        logger.info(f"Performing {test_type} significance test")

        try:
            if test_type == 't_test':
                # Independent t-test
                test_stat, p_value = stats.ttest_ind(strategy_returns, benchmark_returns)
            elif test_type == 'wilcoxon':
                # Non-parametric test
                if len(strategy_returns) == len(benchmark_returns):
                    # Paired Wilcoxon test
                    test_stat, p_value = stats.wilcoxon(strategy_returns, benchmark_returns)
                else:
                    # Mann-Whitney U test
                    test_stat, p_value = stats.mannwhitneyu(strategy_returns, benchmark_returns)
            else:
                return {
                    'error': f'Unknown test type: {test_type}',
                    'test_type': test_type
                }

            # Determine significance
            significant = p_value < alpha

            # Effect size (Cohen's d for t-test)
            if test_type == 't_test':
                pooled_std = np.sqrt(
                    ((len(strategy_returns) - 1) * np.var(strategy_returns, ddof=1) +
                     (len(benchmark_returns) - 1) * np.var(benchmark_returns, ddof=1)) /
                    (len(strategy_returns) + len(benchmark_returns) - 2)
                )
                effect_size = (np.mean(strategy_returns) - np.mean(benchmark_returns)) / pooled_std
            else:
                effect_size = None

            return {
                'test_statistic': test_stat,
                'p_value': p_value,
                'significant': significant,
                'confidence_level': 1 - alpha,
                'test_type': test_type,
                'effect_size': effect_size,
                'strategy_mean': np.mean(strategy_returns),
                'benchmark_mean': np.mean(benchmark_returns)
            }

        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return {
                'error': str(e),
                'test_type': test_type,
                'confidence_level': 1 - alpha
            }

    def risk_analysis(self, returns_distribution: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive risk analysis with tail statistics

        Args:
            returns_distribution: Distribution of returns from simulations

        Returns:
            Dictionary with VaR, CVaR, and tail risk metrics
        """
        if len(returns_distribution) == 0:
            return {}

        logger.info(f"Performing risk analysis on {len(returns_distribution)} return observations")

        # Remove NaN values
        returns = returns_distribution[~np.isnan(returns_distribution)]

        if len(returns) == 0:
            return {}

        # Value at Risk (VaR) calculations
        var_95 = np.percentile(returns, 5)    # 5th percentile
        var_99 = np.percentile(returns, 1)    # 1st percentile
        var_99_9 = np.percentile(returns, 0.1) # 0.1th percentile

        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])

        # Tail statistics
        tail_ratio = (var_95 / var_99) if var_99 != 0 else np.nan
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Extreme values
        worst_case = np.min(returns)
        best_case = np.max(returns)

        # Probability metrics
        prob_loss = np.mean(returns < 0)
        prob_large_loss = np.mean(returns < -0.10)  # Probability of >10% loss

        return {
            'var_95': var_95,
            'var_99': var_99,
            'var_99.9': var_99_9,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'worst_case': worst_case,
            'best_case': best_case,
            'prob_loss': prob_loss,
            'prob_large_loss': prob_large_loss,
            'volatility': np.std(returns),
            'semi_deviation': np.std(returns[returns < np.mean(returns)])
        }

    def _format_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format statistical results for clean reporting

        Args:
            raw_results: Raw statistical analysis results

        Returns:
            Formatted results with summary and interpretation
        """
        formatted = {
            'summary': '',
            'detailed_metrics': raw_results,
            'interpretation': {}
        }

        # Create summary text
        summary_parts = []

        if 'confidence_intervals' in raw_results:
            ci = raw_results['confidence_intervals']
            if 'total_return' in ci and '95%' in ci['total_return']:
                mean_return = ci['total_return']['95%']['mean']
                lower_ci = ci['total_return']['95%']['lower']
                upper_ci = ci['total_return']['95%']['upper']
                summary_parts.append(
                    f"Expected return: {mean_return:.2%} "
                    f"(95% confidence interval: {lower_ci:.2%} to {upper_ci:.2%})"
                )

        if 'risk_metrics' in raw_results:
            risk = raw_results['risk_metrics']
            if 'var_95' in risk:
                summary_parts.append(f"95% VaR: {risk['var_95']:.2%}")

        formatted['summary'] = '; '.join(summary_parts)

        # Add interpretation notes
        if 'risk_metrics' in raw_results:
            risk = raw_results['risk_metrics']
            interpretation = {}

            if 'skewness' in risk:
                if risk['skewness'] < -0.5:
                    interpretation['skewness'] = "Negative skewness indicates higher risk of large losses"
                elif risk['skewness'] > 0.5:
                    interpretation['skewness'] = "Positive skewness indicates potential for large gains"

            if 'prob_loss' in risk:
                if risk['prob_loss'] > 0.4:
                    interpretation['loss_probability'] = "High probability of losses indicates risky strategy"

            formatted['interpretation'] = interpretation

        return formatted

    def analyze_monte_carlo_results(self,
                                  monte_carlo_results: pd.DataFrame,
                                  benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Complete analysis of Monte Carlo simulation results

        Args:
            monte_carlo_results: DataFrame with simulation results
            benchmark_returns: Optional benchmark for comparison

        Returns:
            Comprehensive analysis results
        """
        logger.info("Starting comprehensive Monte Carlo analysis")

        results = {}

        # Calculate confidence intervals
        results['confidence_intervals'] = self.calculate_confidence_intervals(monte_carlo_results)

        # Risk analysis on returns
        if 'total_return' in monte_carlo_results.columns:
            returns = monte_carlo_results['total_return'].values
            results['risk_analysis'] = self.risk_analysis(returns)

        # Performance summary
        numeric_cols = monte_carlo_results.select_dtypes(include=[np.number]).columns
        performance_summary = {}
        for col in numeric_cols:
            if col != 'simulation_id':
                values = monte_carlo_results[col].dropna()
                if len(values) > 0:
                    performance_summary[col] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        results['performance_summary'] = performance_summary

        # Significance testing if benchmark provided
        if benchmark_returns is not None and 'total_return' in monte_carlo_results.columns:
            strategy_returns = monte_carlo_results['total_return'].values
            significance_test = self.statistical_significance_test(strategy_returns, benchmark_returns)
            results['significance_testing'] = {'vs_benchmark': significance_test}

        # Store results
        self.results = results

        return results

    def analyze_tail_behavior(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tail behavior and extreme scenarios

        Args:
            returns: Distribution of returns

        Returns:
            Tail behavior analysis results
        """
        if len(returns) == 0:
            return {}

        # Define tail thresholds
        left_tail_threshold = np.percentile(returns, 10)   # Bottom 10%
        right_tail_threshold = np.percentile(returns, 90)  # Top 10%

        # Left tail analysis (losses)
        left_tail_values = returns[returns <= left_tail_threshold]
        left_tail = {
            'extreme_scenarios': len(left_tail_values),
            'frequency': len(left_tail_values) / len(returns),
            'severity': np.mean(left_tail_values),
            'worst_case': np.min(left_tail_values) if len(left_tail_values) > 0 else 0
        }

        # Right tail analysis (gains)
        right_tail_values = returns[returns >= right_tail_threshold]
        right_tail = {
            'extreme_scenarios': len(right_tail_values),
            'frequency': len(right_tail_values) / len(returns),
            'best_case': np.max(right_tail_values) if len(right_tail_values) > 0 else 0
        }

        # Tail dependency (correlation between extremes)
        # Simple measure: ratio of tail variances to total variance
        total_variance = np.var(returns)
        left_tail_var = np.var(left_tail_values) if len(left_tail_values) > 1 else 0
        right_tail_var = np.var(right_tail_values) if len(right_tail_values) > 1 else 0

        tail_dependency = {
            'left_tail_concentration': left_tail_var / total_variance if total_variance > 0 else 0,
            'right_tail_concentration': right_tail_var / total_variance if total_variance > 0 else 0,
            'asymmetry': abs(left_tail['severity']) / right_tail['best_case'] if right_tail['best_case'] > 0 else np.inf
        }

        return {
            'left_tail': left_tail,
            'right_tail': right_tail,
            'tail_dependency': tail_dependency
        }