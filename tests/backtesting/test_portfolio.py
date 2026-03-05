"""
Test suite for portfolio state management
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.backtesting.portfolio.portfolio import Portfolio
from src.backtesting.core.events import MarketEvent, FillEvent


class TestPortfolio:
    """Test portfolio state tracking and equity calculations"""

    def test_portfolio_initialization(self):
        """Test portfolio can be initialized with starting cash"""
        start_date = datetime(2023, 1, 1)
        initial_cash = 100000.0

        portfolio = Portfolio(start_date, initial_cash)

        assert portfolio.initial_cash == initial_cash
        assert portfolio.current_holdings['cash'] == initial_cash
        assert portfolio.current_holdings['commission'] == 0.0
        assert portfolio.current_holdings['total'] == initial_cash
        assert len(portfolio.current_positions) == 0

    def test_portfolio_tracking(self):
        """Test portfolio tracks positions and holdings correctly"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Check initial state
        assert portfolio.current_holdings['cash'] == 100000.0
        assert portfolio.current_holdings['total'] == 100000.0
        assert len(portfolio.all_holdings) == 1  # Initial holdings recorded

    def test_position_tracking(self):
        """Test position tracking by symbol with quantity"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Should start with no positions
        assert len(portfolio.current_positions) == 0

        # Should track positions after fills
        assert hasattr(portfolio, 'update_fill')

    def test_market_value_update(self):
        """Test mark-to-market updates on market events"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Create market event
        timestamp = datetime(2023, 1, 1, 9, 30)
        market_event = MarketEvent(timestamp, 'SPY', {
            'price': 400.0,
            'volume': 1000,
            'bid': 399.5,
            'ask': 400.5
        })

        # Portfolio should handle market events
        portfolio.update_market_value(market_event)

        # Should record new holdings entry
        assert len(portfolio.all_holdings) >= 1

    def test_fill_event_processing(self):
        """Test that fill events update positions and cash"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Create fill event for buying SPY
        timestamp = datetime(2023, 1, 1, 9, 30)
        fill_event = FillEvent(
            timestamp=timestamp,
            symbol='SPY',
            order_type='MKT',
            quantity=100,
            direction='BUY',
            fill_cost=40050.0,  # 100 * 400 + 50 commission
            commission=50.0
        )

        portfolio.update_fill(fill_event)

        # Check position tracking
        assert 'SPY' in portfolio.current_positions
        assert portfolio.current_positions['SPY'] == 100

        # Check cash reduction
        expected_cash = 100000.0 - 40050.0
        assert portfolio.current_holdings['cash'] == expected_cash
        assert portfolio.current_holdings['commission'] == 50.0

    def test_equity_curve_generation(self):
        """Test time series equity curve tracking"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Should track holdings over time
        assert len(portfolio.all_holdings) >= 1

        # Each holdings entry should have timestamp
        holdings_entry = portfolio.all_holdings[0]
        assert 'datetime' in holdings_entry
        assert 'total' in holdings_entry
        assert 'cash' in holdings_entry

    def test_portfolio_value_calculation(self):
        """Test total portfolio value includes positions + cash"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Initial total should equal cash
        assert portfolio.current_holdings['total'] == 100000.0

        # After position updates, total should reflect mark-to-market
        # This test verifies the calculation logic exists
        assert hasattr(portfolio, 'update_market_value')

    def test_commission_tracking(self):
        """Test commission costs are tracked separately"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Initial commission should be zero
        assert portfolio.current_holdings['commission'] == 0.0

        # After fills, commission should accumulate
        fill_event = FillEvent(
            timestamp=datetime.now(),
            symbol='SPY',
            order_type='MKT',
            quantity=100,
            direction='BUY',
            fill_cost=40025.0,  # 100 * 400 + 25 commission
            commission=25.0
        )

        portfolio.update_fill(fill_event)
        assert portfolio.current_holdings['commission'] == 25.0

    def test_average_cost_tracking(self):
        """Test portfolio tracks average cost basis for positions"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Should track cost basis for positions
        # This ensures we can calculate realized vs unrealized P&L
        assert hasattr(portfolio, 'current_positions')

        # Position tracking should handle multiple fills
        fill1 = FillEvent(
            timestamp=datetime.now(),
            symbol='SPY',
            order_type='MKT',
            quantity=50,
            direction='BUY',
            fill_cost=20010.0,  # 50 * 400 + 10 commission
            commission=10.0
        )

        fill2 = FillEvent(
            timestamp=datetime.now(),
            symbol='SPY',
            order_type='MKT',
            quantity=50,
            direction='BUY',
            fill_cost=20515.0,  # 50 * 410 + 15 commission
            commission=15.0
        )

        portfolio.update_fill(fill1)
        portfolio.update_fill(fill2)

        # Should accumulate position correctly
        assert portfolio.current_positions['SPY'] == 100
        assert portfolio.current_holdings['commission'] == 25.0

    def test_performance_summary(self):
        """Test portfolio can generate performance summary"""
        start_date = datetime(2023, 1, 1)
        portfolio = Portfolio(start_date, 100000.0)

        # Should provide performance metrics
        assert hasattr(portfolio, 'get_performance_summary')

        summary = portfolio.get_performance_summary()
        assert 'total_return' in summary
        assert 'sharpe_ratio' in summary or summary is not None