"""
Tests for performance metrics calculator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtesting.performance.metrics import PerformanceCalculator


class TestPerformanceCalculator:
    """Test comprehensive performance metrics calculation"""

    def setup_method(self):
        """Setup test data"""
        self.calculator = PerformanceCalculator()

        # Create sample equity curve
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        # Simple upward trending equity curve with some volatility
        returns = np.random.normal(0.0005, 0.015, len(dates))  # ~12% annual return, 15% vol
        prices = 100000 * np.cumprod(1 + returns)

        self.equity_curve = pd.DataFrame({
            'datetime': dates,
            'total': prices
        })

        # Create sample trade history
        self.trade_history = pd.DataFrame({
            'datetime': dates[::20],  # Trades every 20 days
            'symbol': ['AAPL'] * len(dates[::20]),
            'pnl': np.random.normal(1000, 2000, len(dates[::20])),
            'quantity': [100] * len(dates[::20])
        })

    def test_metrics_calculation(self):
        """Test comprehensive performance metrics calculation"""
        # This test should FAIL initially since PerformanceCalculator doesn't exist
        metrics = self.calculator.calculate_all_metrics(
            self.equity_curve,
            self.trade_history
        )

        # Expected metrics
        expected_keys = [
            'total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'volatility', 'win_rate', 'profit_factor', 'calmar_ratio'
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], (int, float)), f"Metric {key} should be numeric"

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation with risk-free rate"""
        # Create known equity curve for predictable Sharpe ratio
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        # Strong positive mean with small, non-zero volatility -> high Sharpe.
        # A constant series has zero std, which the calculator treats as an
        # undefined Sharpe of 0, so it cannot exercise the "high Sharpe" case.
        returns = np.full(252, 0.001)
        returns[1::2] = 0.0008
        equity_values = 100000 * np.cumprod(1 + returns)

        equity_curve = pd.DataFrame({
            'datetime': dates,
            'total': equity_values
        })

        metrics = self.calculator.calculate_all_metrics(equity_curve, pd.DataFrame())

        # Consistent low-volatility positive returns -> high Sharpe and a
        # solidly positive annualized return (the old `> 10` asserted >1000%).
        assert metrics['sharpe_ratio'] > 10, "Sharpe ratio should be high for consistent returns"
        assert metrics['annualized_return'] > 0.10, "Annualized return should be solidly positive"

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Create equity curve with known drawdown
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = [100000, 110000, 120000, 90000, 85000, 95000, 130000]  # 29% drawdown
        values += [130000] * (100 - len(values))  # Pad with final value

        equity_curve = pd.DataFrame({
            'datetime': dates,
            'total': values
        })

        metrics = self.calculator.calculate_all_metrics(equity_curve, pd.DataFrame())

        # Should detect the 29.17% drawdown (from 120k to 85k)
        expected_drawdown = -((120000 - 85000) / 120000)
        assert abs(metrics['max_drawdown'] - expected_drawdown) < 0.001

    def test_win_rate_calculation(self):
        """Test win rate calculation from trade history"""
        # Create trade history with known win rate
        trade_history = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='D'),
            'pnl': [100, -50, 200, 150, -75, 300, -25, 180, -100, 250]  # 6 wins, 4 losses
        })

        metrics = self.calculator.calculate_all_metrics(pd.DataFrame(), trade_history)

        # Win rate should be 60% (6 of the 10 trades have positive PnL)
        expected_win_rate = 0.6
        assert abs(metrics['win_rate'] - expected_win_rate) < 0.001

    def test_profit_factor_calculation(self):
        """Test profit factor calculation"""
        # Trade history with specific profit/loss pattern
        trade_history = pd.DataFrame({
            'pnl': [100, 200, 300, -150, -250]  # Total profit: 600, Total loss: 400
        })

        metrics = self.calculator.calculate_all_metrics(pd.DataFrame(), trade_history)

        # Profit factor should be 600/400 = 1.5
        expected_profit_factor = 1.5
        assert abs(metrics['profit_factor'] - expected_profit_factor) < 0.001

    def test_edge_cases(self):
        """Test edge cases: no trades, all losses, all gains"""
        # No trades
        empty_trades = pd.DataFrame(columns=['pnl'])
        metrics = self.calculator.calculate_all_metrics(self.equity_curve, empty_trades)
        assert metrics['win_rate'] == 0.0
        assert metrics['profit_factor'] == 0.0

        # All losses
        loss_trades = pd.DataFrame({'pnl': [-100, -50, -200]})
        metrics = self.calculator.calculate_all_metrics(self.equity_curve, loss_trades)
        assert metrics['win_rate'] == 0.0
        assert metrics['profit_factor'] == 0.0

        # All gains
        gain_trades = pd.DataFrame({'pnl': [100, 50, 200]})
        metrics = self.calculator.calculate_all_metrics(self.equity_curve, gain_trades)
        assert metrics['win_rate'] == 1.0
        assert metrics['profit_factor'] > 0

    def test_annualization(self):
        """Test proper annualization using 252 trading days"""
        # One year of daily data
        # A full calendar year of daily points: a 20% total return annualizes
        # to ~20%. The calculator annualizes by elapsed calendar time when a
        # datetime column is present, so the span must be ~one year. (The
        # previous 252 freq='D' rows spanned only 251 days, so 20% annualized
        # to ~30%.)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        final_value = 120000
        equity_curve = pd.DataFrame({
            'datetime': dates,
            'total': np.linspace(100000, final_value, len(dates))
        })

        metrics = self.calculator.calculate_all_metrics(equity_curve, pd.DataFrame())

        # Annualized return should be approximately 20%
        expected_annual = 0.20
        assert abs(metrics['annualized_return'] - expected_annual) < 0.01