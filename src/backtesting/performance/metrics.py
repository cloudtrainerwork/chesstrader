"""
Comprehensive performance metrics calculator for backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """
    Trading-specific performance metrics calculator

    Computes comprehensive performance metrics from equity curve and trade history
    with proper annualization and risk-adjusted measures.
    """

    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 252):
        """
        Initialize performance calculator

        Args:
            risk_free_rate: Annualized risk-free rate for Sharpe ratio
            trading_days: Trading days per year for annualization
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_all_metrics(self,
                            equity_curve: pd.DataFrame,
                            trade_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics

        Args:
            equity_curve: DataFrame with 'datetime' and 'total' columns
            trade_history: DataFrame with 'pnl' column for trade analysis

        Returns:
            Dictionary with all performance metrics
        """
        metrics = {}

        # Basic return metrics
        if not equity_curve.empty and 'total' in equity_curve.columns:
            initial_value = equity_curve['total'].iloc[0]
            final_value = equity_curve['total'].iloc[-1]

            metrics['total_return'] = (final_value - initial_value) / initial_value

            # Calculate period length for annualization
            if len(equity_curve) > 1:
                if 'datetime' in equity_curve.columns:
                    # Real timestamps: elapsed calendar time over calendar days/yr.
                    days = (equity_curve['datetime'].iloc[-1] - equity_curve['datetime'].iloc[0]).days
                    years = max(days / 365.25, 1 / self.trading_days)
                else:
                    # No timestamps: each row is a trading bar, so annualize by
                    # trading days per year, not calendar days. Dividing a
                    # trading-bar count by 365.25 over-annualizes the return.
                    bars = len(equity_curve)
                    years = max(bars / self.trading_days, 1 / self.trading_days)

                metrics['annualized_return'] = (1 + metrics['total_return']) ** (1/years) - 1
            else:
                metrics['annualized_return'] = 0.0

            # Risk metrics
            metrics['sharpe_ratio'] = self._calculate_sharpe(equity_curve)
            metrics['sortino_ratio'] = self._calculate_sortino(equity_curve)
            metrics['max_drawdown'] = self._calculate_max_drawdown(equity_curve)
            metrics['volatility'] = self._calculate_volatility(equity_curve)
            metrics['calmar_ratio'] = self._calculate_calmar_ratio(
                metrics['annualized_return'], metrics['max_drawdown']
            )
        else:
            # No equity curve data
            metrics.update({
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0
            })

        # Trade-based metrics
        if not trade_history.empty and 'pnl' in trade_history.columns:
            metrics['win_rate'] = self._calculate_win_rate(trade_history)
            metrics['profit_factor'] = self._calculate_profit_factor(trade_history)
        else:
            metrics.update({
                'win_rate': 0.0,
                'profit_factor': 0.0
            })

        return metrics

    def _calculate_sharpe(self, equity_curve: pd.DataFrame) -> float:
        """Calculate annualized Sharpe ratio"""
        try:
            if len(equity_curve) < 2:
                return 0.0

            # Calculate daily returns
            daily_returns = equity_curve['total'].pct_change().dropna()

            if len(daily_returns) == 0 or daily_returns.std() == 0:
                return 0.0

            # Annualize metrics
            mean_return = daily_returns.mean() * self.trading_days
            volatility = daily_returns.std() * np.sqrt(self.trading_days)

            # Sharpe ratio with risk-free rate adjustment
            return (mean_return - self.risk_free_rate) / volatility

        except Exception as e:
            logger.warning(f"Sharpe ratio calculation failed: {e}")
            return 0.0

    def _calculate_sortino(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if len(equity_curve) < 2:
                return 0.0

            daily_returns = equity_curve['total'].pct_change().dropna()

            if len(daily_returns) == 0:
                return 0.0

            mean_return = daily_returns.mean() * self.trading_days

            # Downside deviation (only negative returns)
            downside_returns = daily_returns[daily_returns < 0]

            if len(downside_returns) == 0:
                # No downside, return high Sortino
                return mean_return * 10  # Arbitrary high value

            downside_deviation = downside_returns.std() * np.sqrt(self.trading_days)

            if downside_deviation == 0:
                return 0.0

            return (mean_return - self.risk_free_rate) / downside_deviation

        except Exception as e:
            logger.warning(f"Sortino ratio calculation failed: {e}")
            return 0.0

    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(equity_curve) < 2:
                return 0.0

            equity_values = equity_curve['total'].values

            # Calculate running maximum
            running_max = np.maximum.accumulate(equity_values)

            # Calculate drawdowns
            drawdowns = (equity_values - running_max) / running_max

            # Return maximum (most negative) drawdown
            return np.min(drawdowns)

        except Exception as e:
            logger.warning(f"Max drawdown calculation failed: {e}")
            return 0.0

    def _calculate_volatility(self, equity_curve: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        try:
            if len(equity_curve) < 2:
                return 0.0

            daily_returns = equity_curve['total'].pct_change().dropna()

            if len(daily_returns) == 0:
                return 0.0

            return daily_returns.std() * np.sqrt(self.trading_days)

        except Exception as e:
            logger.warning(f"Volatility calculation failed: {e}")
            return 0.0

    def _calculate_win_rate(self, trade_history: pd.DataFrame) -> float:
        """Calculate win rate from trade history"""
        try:
            if trade_history.empty or 'pnl' not in trade_history.columns:
                return 0.0

            pnl_values = trade_history['pnl'].dropna()

            if len(pnl_values) == 0:
                return 0.0

            # Count winning trades (positive PnL)
            winning_trades = len(pnl_values[pnl_values > 0])
            total_trades = len(pnl_values)

            return winning_trades / total_trades

        except Exception as e:
            logger.warning(f"Win rate calculation failed: {e}")
            return 0.0

    def _calculate_profit_factor(self, trade_history: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            if trade_history.empty or 'pnl' not in trade_history.columns:
                return 0.0

            pnl_values = trade_history['pnl'].dropna()

            if len(pnl_values) == 0:
                return 0.0

            # Calculate gross profit and loss
            gross_profit = pnl_values[pnl_values > 0].sum()
            gross_loss = abs(pnl_values[pnl_values < 0].sum())

            if gross_loss == 0:
                # No losses - infinite profit factor, return high value
                return gross_profit if gross_profit > 0 else 0.0

            return gross_profit / gross_loss

        except Exception as e:
            logger.warning(f"Profit factor calculation failed: {e}")
            return 0.0

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)"""
        try:
            if max_drawdown == 0:
                return annualized_return * 10  # Arbitrary high value when no drawdown

            # Max drawdown is negative, so we use absolute value
            return annualized_return / abs(max_drawdown)

        except Exception as e:
            logger.warning(f"Calmar ratio calculation failed: {e}")
            return 0.0