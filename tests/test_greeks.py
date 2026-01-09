"""
Tests for Greeks calculation and P/L analysis.
"""

import unittest
import math
from datetime import datetime, timedelta
from src.features.greeks import GreeksCalculator, ImpliedVolatilityEstimator
from src.features.pnl import PnLCalculator
from src.features.position_models import Position, StrategyType, OptionType


class TestGreeksCalculator(unittest.TestCase):
    """Test Greeks calculations."""

    def setUp(self):
        """Set up test calculator."""
        self.greeks_calc = GreeksCalculator()

    def test_call_delta_calculation(self):
        """Test call delta calculation."""
        # ATM call should have delta around 0.5
        delta = self.greeks_calc.calculate_delta(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='CALL'
        )

        self.assertGreater(delta, 0.4)
        self.assertLess(delta, 0.6)
        self.assertGreaterEqual(delta, -1.0)
        self.assertLessEqual(delta, 1.0)

    def test_put_delta_calculation(self):
        """Test put delta calculation."""
        # ATM put should have delta around -0.5
        delta = self.greeks_calc.calculate_delta(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='PUT'
        )

        self.assertLess(delta, -0.4)
        self.assertGreater(delta, -0.6)
        self.assertGreaterEqual(delta, -1.0)
        self.assertLessEqual(delta, 1.0)

    def test_gamma_calculation(self):
        """Test gamma calculation."""
        gamma = self.greeks_calc.calculate_gamma(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='CALL'
        )

        # Gamma should be positive and reasonable
        self.assertGreater(gamma, 0)
        self.assertLess(gamma, 10)  # Normalized gamma should be reasonable

    def test_theta_calculation(self):
        """Test theta calculation."""
        theta = self.greeks_calc.calculate_theta(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='CALL'
        )

        # Theta should be negative (time decay)
        self.assertLess(theta, 0)

    def test_vega_calculation(self):
        """Test vega calculation."""
        vega = self.greeks_calc.calculate_vega(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='CALL'
        )

        # Vega should be positive
        self.assertGreater(vega, 0)

    def test_edge_cases(self):
        """Test edge cases for Greeks calculations."""
        # Zero time to expiration
        delta = self.greeks_calc.calculate_delta(
            S=100, K=100, T=0, r=0.05, sigma=0.20, option_type='CALL'
        )
        self.assertEqual(delta, 0.0)

        # Zero volatility
        gamma = self.greeks_calc.calculate_gamma(
            S=100, K=100, T=0.25, r=0.05, sigma=0, option_type='CALL'
        )
        self.assertEqual(gamma, 0.0)

    def test_position_greeks(self):
        """Test position-level Greeks aggregation."""
        # Create test position
        position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=datetime(2024, 1, 1),
            expiration_date=datetime(2024, 2, 16),
            strikes=[10000],  # $100 strike
            option_types=[OptionType.CALL],
            quantities=[2],  # 2 contracts
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9500,
            current_underlying_price=10000  # ATM
        )

        greeks = self.greeks_calc.position_greeks(position, iv_estimates=[0.20])

        # Check that all Greeks are returned
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertIn('theta', greeks)
        self.assertIn('vega', greeks)

        # Delta should be positive for long call
        self.assertGreater(greeks['delta'], 0)

        # Gamma should be positive
        self.assertGreater(greeks['gamma'], 0)

        # Theta should be negative
        self.assertLess(greeks['theta'], 0)

        # Vega should be positive
        self.assertGreater(greeks['vega'], 0)


class TestImpliedVolatilityEstimator(unittest.TestCase):
    """Test IV estimation."""

    def setUp(self):
        """Set up test estimator."""
        self.iv_estimator = ImpliedVolatilityEstimator()

    def test_iv_estimation(self):
        """Test basic IV estimation."""
        iv = self.iv_estimator.estimate_iv('SPY')

        # IV should be reasonable (5% to 100%)
        self.assertGreater(iv, 0.05)
        self.assertLess(iv, 1.0)

    def test_position_iv_estimation(self):
        """Test position-level IV estimation."""
        position = Position(
            strategy_type=StrategyType.IRON_CONDOR,
            entry_date=datetime(2024, 1, 1),
            expiration_date=datetime(2024, 2, 16),
            strikes=[9500, 10000, 11000, 11500],
            option_types=[OptionType.PUT, OptionType.PUT, OptionType.CALL, OptionType.CALL],
            quantities=[1, -1, -1, 1],
            entry_prices=[200, 500, 600, 250],
            current_prices=[100, 300, 400, 150],
            underlying_price_at_entry=10500,
            current_underlying_price=10500
        )

        iv_estimates = self.iv_estimator.get_iv_for_position(position)

        # Should get IV for each leg
        self.assertEqual(len(iv_estimates), 4)

        # All IV estimates should be reasonable
        for iv in iv_estimates:
            self.assertGreater(iv, 0.05)
            self.assertLess(iv, 2.0)


class TestPnLCalculator(unittest.TestCase):
    """Test P/L calculations."""

    def setUp(self):
        """Set up test calculator and position."""
        self.pnl_calc = PnLCalculator()

        # Long call position for testing
        self.long_call = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=datetime(2024, 1, 1),
            expiration_date=datetime(2024, 2, 16),
            strikes=[10000],  # $100 strike
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],  # Paid $5.00
            current_prices=[300],  # Now worth $3.00
            underlying_price_at_entry=9500,
            current_underlying_price=10200
        )

    def test_unrealized_pnl(self):
        """Test unrealized P/L calculation."""
        pnl = self.pnl_calc.unrealized_pnl(self.long_call, [300])

        # Should be -200 cents (-$2.00)
        self.assertEqual(pnl, -200)

    def test_pnl_percentage(self):
        """Test P/L percentage calculation."""
        pnl_pct = self.pnl_calc.pnl_percentage(self.long_call, [300])

        # Should be -40% (-200/500)
        self.assertAlmostEqual(pnl_pct, -0.4, places=2)

    def test_percent_of_max_profit(self):
        """Test percent of max profit calculation."""
        # Long call has unlimited profit, so should handle gracefully
        pct_max_profit = self.pnl_calc.percent_of_max_profit(self.long_call, [300])

        # Should be 0 or small negative since position is losing
        self.assertLessEqual(pct_max_profit, 0.1)

    def test_percent_of_max_loss(self):
        """Test percent of max loss calculation."""
        pct_max_loss = self.pnl_calc.percent_of_max_loss(self.long_call, [300])

        # Should be 40% of max loss (200/500)
        self.assertAlmostEqual(pct_max_loss, 0.4, places=2)

    def test_days_held(self):
        """Test days held calculation."""
        days = self.pnl_calc.days_held(self.long_call)

        # Should be reasonable number of days
        expected_days = (datetime.now() - datetime(2024, 1, 1)).days
        self.assertEqual(days, expected_days)

    def test_comprehensive_metrics(self):
        """Test comprehensive P/L metrics calculation."""
        metrics = self.pnl_calc.calculate_pnl_metrics(self.long_call, [300])

        # Check all expected metrics are present
        expected_keys = [
            'entry_credit', 'current_value', 'unrealized_pnl',
            'pnl_percentage', 'percent_of_max_profit', 'percent_of_max_loss',
            'days_held'
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

        # Validate some key values
        self.assertEqual(metrics['unrealized_pnl'], -200.0)
        self.assertEqual(metrics['current_value'], 300.0)

    def test_position_performance_summary(self):
        """Test position performance summary."""
        summary = self.pnl_calc.position_performance_summary(self.long_call, [300])

        # Check main sections are present
        self.assertIn('pnl_metrics', summary)
        self.assertIn('position_structure', summary)
        self.assertIn('timing', summary)
        self.assertIn('risk_assessment', summary)

        # Check position structure
        pos_struct = summary['position_structure']
        self.assertEqual(pos_struct['strategy_type'], 'long_call')
        self.assertEqual(pos_struct['max_loss'], 500)  # Premium paid


if __name__ == '__main__':
    unittest.main()