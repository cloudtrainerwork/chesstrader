"""
Tests for position state feature engineering.
"""

import unittest
from datetime import datetime, timedelta
from src.features.position_models import Position, PositionZones, StrategyType, OptionType
from src.features.position_state import PositionState


class TestPositionModels(unittest.TestCase):
    """Test position data models and calculations."""

    def setUp(self):
        """Set up test position."""
        self.entry_date = datetime(2024, 1, 1)
        self.expiration_date = datetime(2024, 2, 16)

        # Iron Condor position for testing
        self.iron_condor = Position(
            strategy_type=StrategyType.IRON_CONDOR,
            entry_date=self.entry_date,
            expiration_date=self.expiration_date,
            strikes=[9500, 10000, 11000, 11500],  # Prices in cents
            option_types=[OptionType.PUT, OptionType.PUT, OptionType.CALL, OptionType.CALL],
            quantities=[1, -1, -1, 1],  # Short iron condor
            entry_prices=[200, 500, 600, 250],  # Prices in cents
            current_prices=[100, 300, 400, 150],
            underlying_price_at_entry=10500,  # $105.00 in cents
            current_underlying_price=10500
        )

        # Simple long call for testing
        self.long_call = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=self.entry_date,
            expiration_date=self.expiration_date,
            strikes=[10000],  # $100.00 strike
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],  # $5.00 premium
            current_prices=[300],  # $3.00 current
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

    def test_position_creation(self):
        """Test position object creation."""
        self.assertEqual(self.iron_condor.strategy_type, StrategyType.IRON_CONDOR)
        self.assertEqual(len(self.iron_condor.strikes), 4)
        self.assertEqual(self.iron_condor.days_held, (datetime.now() - self.entry_date).days)

    def test_position_validation(self):
        """Test position data validation."""
        with self.assertRaises(ValueError):
            Position(
                strategy_type=StrategyType.LONG_CALL,
                entry_date=self.entry_date,
                expiration_date=self.expiration_date,
                strikes=[10000],
                option_types=[OptionType.CALL, OptionType.PUT],  # Mismatched length
                quantities=[1],
                entry_prices=[500],
                current_prices=[300],
                underlying_price_at_entry=9800,
                current_underlying_price=9900
            )

    def test_unrealized_pnl_calculation(self):
        """Test P/L calculation."""
        # Iron condor P/L
        ic_pnl = self.iron_condor.calculate_unrealized_pnl()
        expected_current = 1 * 100 + (-1) * 300 + (-1) * 400 + 1 * 150  # = -450
        expected_entry = 1 * 200 + (-1) * 500 + (-1) * 600 + 1 * 250    # = -650
        expected_pnl = expected_current - expected_entry  # = 200
        self.assertEqual(ic_pnl, expected_pnl)

        # Long call P/L
        lc_pnl = self.long_call.calculate_unrealized_pnl()
        expected_pnl = 300 - 500  # = -200
        self.assertEqual(lc_pnl, expected_pnl)

    def test_max_profit_calculation(self):
        """Test maximum profit calculation."""
        # Iron condor max profit should be net credit
        ic_max = self.iron_condor.calculate_max_profit()
        net_credit = abs(1 * 200 + (-1) * 500 + (-1) * 600 + 1 * 250)  # 650
        self.assertEqual(ic_max, net_credit)

        # Long call has unlimited profit
        lc_max = self.long_call.calculate_max_profit()
        self.assertEqual(lc_max, 0)

    def test_debit_spread_max_profit_when_debit_exceeds_half_width(self):
        """Bull call spread max profit must be width - debit, even when the
        net debit exceeds half the spread width (regression for issue #16)."""
        # Width 1000c, net debit 800c (long 900c - short 100c). Max profit is
        # 1000 - 800 = 200c. The old max(width-debit, debit) returned 800c.
        bull_call_spread = Position(
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            entry_date=self.entry_date,
            expiration_date=self.expiration_date,
            strikes=[10000, 11000],
            option_types=[OptionType.CALL, OptionType.CALL],
            quantities=[1, -1],
            entry_prices=[900, 100],
            current_prices=[950, 120],
            underlying_price_at_entry=10000,
            current_underlying_price=10000
        )
        self.assertEqual(bull_call_spread.calculate_max_profit(), 200)

    def test_max_loss_calculation(self):
        """Test maximum loss calculation."""
        # Long call max loss is premium paid
        lc_loss = self.long_call.calculate_max_loss()
        self.assertEqual(lc_loss, 500)

    def test_breakeven_calculation(self):
        """Test breakeven calculation."""
        # Long call breakeven
        lc_breakevens = self.long_call.calculate_breakevens()
        expected_be = 10000 + 500  # Strike + premium
        self.assertEqual(lc_breakevens, [expected_be])

        # Iron condor should have two breakevens
        ic_breakevens = self.iron_condor.calculate_breakevens()
        self.assertEqual(len(ic_breakevens), 2)

    def test_price_zone_calculation(self):
        """Test price zone calculation."""
        zone = self.iron_condor.calculate_price_zone()
        self.assertIsInstance(zone, PositionZones)

        # Test long call in loss
        lc_zone = self.long_call.calculate_price_zone()
        self.assertIsInstance(lc_zone, PositionZones)


class TestPositionState(unittest.TestCase):
    """Test position state feature engineering."""

    def setUp(self):
        """Set up test position and feature calculator."""
        self.position_state = PositionState()

        self.test_position = Position(
            strategy_type=StrategyType.LONG_CALL,
            entry_date=datetime(2024, 1, 1),
            expiration_date=datetime(2024, 2, 16),
            strikes=[10000],
            option_types=[OptionType.CALL],
            quantities=[1],
            entry_prices=[500],
            current_prices=[300],
            underlying_price_at_entry=9800,
            current_underlying_price=9900
        )

    def test_position_state_calculation(self):
        """Test position state feature calculation."""
        features = self.position_state.calculate(self.test_position)

        # Check that we get expected features
        expected_features = [
            'price_zone', 'zone_velocity', 'distance_to_lower_breakeven',
            'distance_to_upper_breakeven', 'distance_to_lower_strike',
            'distance_to_upper_strike'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.index)

        # Validate no NaN values
        self.assertFalse(features.isnull().any())

        # Check feature ranges are reasonable
        for value in features:
            self.assertFalse(np.isnan(value))
            self.assertFalse(np.isinf(value))


if __name__ == '__main__':
    unittest.main()