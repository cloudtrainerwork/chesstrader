"""
Bootstrap resampling for trade sequence analysis

Tests strategy robustness against trade sequence dependency by resampling
trade returns with replacement and analyzing performance distributions.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats

logger = logging.getLogger(__name__)


class BootstrapResampler:
    """
    Bootstrap resampler for trade sequence analysis

    Implements bootstrap resampling to test strategy robustness against
    trade sequence dependency risk and quantify performance uncertainty.
    """

    def __init__(self,
                 n_bootstrap_samples: int = 1000,
                 random_seed: int = 42):
        """
        Initialize bootstrap resampler

        Args:
            n_bootstrap_samples: Number of bootstrap samples to generate
            random_seed: Random seed for reproducible results
        """
        self.n_bootstrap_samples = n_bootstrap_samples
        self.random_seed = random_seed
        self.bootstrap_results = None

        # Set random seed for reproducible results
        np.random.seed(random_seed)

        logger.info(f"BootstrapResampler initialized with {n_bootstrap_samples} samples")

    def resample_trade_sequence(self, trade_returns: List[float]) -> pd.DataFrame:
        """
        Bootstrap resample trade returns to test sequence dependency

        Args:
            trade_returns: List of trade return values

        Returns:
            DataFrame with bootstrap sample results
        """
        if len(trade_returns) == 0:
            return pd.DataFrame()

        logger.info(f"Resampling {len(trade_returns)} trades with {self.n_bootstrap_samples} bootstrap samples")

        results = []

        for sample_id in range(self.n_bootstrap_samples):
            # Set seed for this specific sample to ensure reproducibility
            np.random.seed(self.random_seed + sample_id)

            # Resample trade returns with replacement
            resampled_returns = self._resample_with_replacement(trade_returns)

            # Calculate performance metrics for resampled sequence
            cumulative_returns, max_drawdown, final_return = self._calculate_performance_metrics(resampled_returns)

            # Store results
            result = {
                'sample_id': sample_id,
                'final_return': final_return,
                'max_drawdown': max_drawdown,
                'cumulative_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0,
                'trade_count': len(resampled_returns),
                'volatility': np.std(resampled_returns) if len(resampled_returns) > 1 else 0.0
            }

            # Calculate final portfolio value assuming 100k start
            result['final_value'] = 100000 * (1 + final_return)

            results.append(result)

        # Convert to DataFrame and store
        self.bootstrap_results = pd.DataFrame(results)
        return self.bootstrap_results

    def _calculate_performance_metrics(self, trade_returns: List[float]) -> Tuple[np.ndarray, float, float]:
        """
        Calculate performance metrics from trade sequence

        Args:
            trade_returns: Trade return sequence

        Returns:
            Tuple of (cumulative_returns, max_drawdown, final_return)
        """
        if len(trade_returns) == 0:
            return np.array([]), 0.0, 0.0

        # Convert to numpy array for calculations
        returns_array = np.array(trade_returns)

        # Calculate cumulative returns (compound growth)
        cumulative_returns = np.cumprod(1 + returns_array) - 1

        # Final return is the last cumulative return
        final_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0

        # Calculate maximum drawdown
        # Drawdown is the decline from peak to trough
        equity_curve = 1 + cumulative_returns  # Convert to equity values
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

        return cumulative_returns, max_drawdown, final_return

    def simulate_trade_sequences(self,
                                trade_returns: List[float],
                                initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run bootstrap simulation of trade sequences

        Args:
            trade_returns: Original trade return sequence
            initial_capital: Starting portfolio value

        Returns:
            Dictionary with simulation results and statistics
        """
        logger.info(f"Running bootstrap simulation with {len(trade_returns)} trades")

        # Generate bootstrap samples
        bootstrap_df = self.resample_trade_sequence(trade_returns)

        if len(bootstrap_df) == 0:
            return {
                'bootstrap_stats': pd.DataFrame(),
                'performance_distribution': {},
                'risk_metrics': {}
            }

        # Calculate performance distribution statistics
        final_returns = bootstrap_df['final_return'].values
        max_drawdowns = bootstrap_df['max_drawdown'].values

        performance_distribution = {
            'mean_return': np.mean(final_returns),
            'std_return': np.std(final_returns),
            'mean_drawdown': np.mean(max_drawdowns),
            'std_drawdown': np.std(max_drawdowns),
            'confidence_intervals': {
                '95%': {
                    'lower': np.percentile(final_returns, 2.5),
                    'upper': np.percentile(final_returns, 97.5)
                },
                '90%': {
                    'lower': np.percentile(final_returns, 5),
                    'upper': np.percentile(final_returns, 95)
                }
            }
        }

        # Calculate risk metrics
        risk_metrics = {
            'var_95': np.percentile(final_returns, 5),  # Value at Risk (5th percentile)
            'cvar_95': np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),  # Conditional VaR
            'sequence_dependency_risk': self._calculate_sequence_dependency_risk(trade_returns, final_returns),
            'worst_case_drawdown': np.min(max_drawdowns),
            'probability_of_loss': np.mean(final_returns < 0)
        }

        # Add final values scaled by initial capital
        bootstrap_df['final_value'] = initial_capital * (1 + bootstrap_df['final_return'])

        return {
            'bootstrap_stats': bootstrap_df,
            'performance_distribution': performance_distribution,
            'risk_metrics': risk_metrics
        }

    def _resample_with_replacement(self, trade_returns: List[float]) -> np.ndarray:
        """
        Resample trade returns with replacement

        Args:
            trade_returns: Original trade sequence

        Returns:
            Resampled trade sequence array
        """
        if len(trade_returns) == 0:
            return np.array([])

        # Bootstrap sampling: sample with replacement
        resampled_indices = np.random.choice(
            len(trade_returns),
            size=len(trade_returns),
            replace=True
        )

        # Return resampled sequence
        return np.array(trade_returns)[resampled_indices]

    def _calculate_sequence_dependency_risk(self, original_returns: List[float], bootstrap_returns: np.ndarray) -> float:
        """
        Calculate sequence dependency risk by comparing original vs bootstrap performance

        Args:
            original_returns: Original trade sequence
            bootstrap_returns: Bootstrap final returns distribution

        Returns:
            Sequence dependency risk measure
        """
        if len(original_returns) == 0:
            return 0.0

        # Calculate original sequence final return
        _, _, original_final_return = self._calculate_performance_metrics(original_returns)

        # Calculate where original return falls in bootstrap distribution
        percentile = stats.percentileofscore(bootstrap_returns, original_final_return)

        # Sequence dependency risk: how much the original sequence differs from random
        # Risk is higher when original performance is very different from bootstrap mean
        bootstrap_mean = np.mean(bootstrap_returns)
        risk_score = abs(original_final_return - bootstrap_mean) / np.std(bootstrap_returns) if np.std(bootstrap_returns) > 0 else 0.0

        return min(risk_score, 5.0)  # Cap at 5 standard deviations