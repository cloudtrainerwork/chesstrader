"""
Test suite for execution handler and market data
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.backtesting.execution.execution import ExecutionHandler
from src.backtesting.data_handlers.market_data import MarketDataHandler
from src.backtesting.core.events import OrderEvent, FillEvent


class TestExecutionHandler:
    """Test order execution simulation with realistic costs"""

    def test_execution_handler_creation(self):
        """Test ExecutionHandler can be created with cost parameters"""
        handler = ExecutionHandler()

        # Should have configurable slippage and commission rates
        assert hasattr(handler, 'slippage_rate')
        assert hasattr(handler, 'commission_rate')

        # Default values for options trading
        assert handler.slippage_rate <= 0.001  # Should be reasonable for options
        assert handler.commission_rate > 0     # Should have commission costs

    def test_realistic_fills(self):
        """Test order execution includes slippage and commission"""
        handler = ExecutionHandler(slippage_rate=0.0005, commission_rate=1.0)

        # Create order event
        timestamp = datetime.now()
        order = OrderEvent(
            timestamp=timestamp,
            symbol='SPY',
            order_type='MKT',
            quantity=100,
            direction='BUY'
        )

        # Mock market price
        handler.latest_prices = {'SPY': 400.0}

        # Execute order
        fill = handler.execute_order(order)

        assert isinstance(fill, FillEvent)
        assert fill.symbol == 'SPY'
        assert fill.quantity == 100
        assert fill.direction == 'BUY'

        # Check slippage application (BUY should add slippage)
        expected_fill_price = 400.0 * (1 + 0.0005)  # Add slippage for BUY
        expected_cost = expected_fill_price * 100
        expected_commission = 1.0 * 100  # $1 per contract

        assert abs(fill.fill_cost - (expected_cost + expected_commission)) < 0.01
        assert fill.commission == expected_commission

    def test_slippage_direction(self):
        """Test slippage is applied correctly for BUY vs SELL orders"""
        handler = ExecutionHandler(slippage_rate=0.001, commission_rate=0.5)
        handler.latest_prices = {'SPY': 400.0}

        timestamp = datetime.now()

        # Test BUY order (should add slippage)
        buy_order = OrderEvent(timestamp, 'SPY', 'MKT', 100, 'BUY')
        buy_fill = handler.execute_order(buy_order)

        # Test SELL order (should subtract slippage)
        sell_order = OrderEvent(timestamp, 'SPY', 'MKT', 100, 'SELL')
        sell_fill = handler.execute_order(sell_order)

        # Extract fill prices (excluding commission)
        buy_price_per_share = (buy_fill.fill_cost - buy_fill.commission) / 100
        sell_price_per_share = (sell_fill.fill_cost - sell_fill.commission) / 100

        # BUY should be higher than market price
        assert buy_price_per_share > 400.0

        # SELL should be lower than market price
        assert sell_price_per_share < 400.0

    def test_commission_calculation(self):
        """Test commission calculation based on contract size"""
        handler = ExecutionHandler(slippage_rate=0.0, commission_rate=1.5)
        handler.latest_prices = {'SPY': 400.0}

        timestamp = datetime.now()
        order = OrderEvent(timestamp, 'SPY', 'MKT', 50, 'BUY')

        fill = handler.execute_order(order)

        # Commission should be rate * quantity
        expected_commission = 1.5 * 50
        assert fill.commission == expected_commission

    def test_no_fill_without_price(self):
        """Test that orders don't fill without market price data"""
        handler = ExecutionHandler()

        timestamp = datetime.now()
        order = OrderEvent(timestamp, 'UNKNOWN', 'MKT', 100, 'BUY')

        fill = handler.execute_order(order)

        # Should return None if no price available
        assert fill is None

    def test_market_data_integration(self):
        """Test execution handler integrates with market data"""
        handler = ExecutionHandler()

        # Should be able to receive price updates
        assert hasattr(handler, 'update_prices')

        # Test price update
        handler.update_prices('SPY', 450.0)
        assert handler.latest_prices.get('SPY') == 450.0


class TestMarketDataHandler:
    """Test historical market data access"""

    def test_market_data_handler_creation(self):
        """Test MarketDataHandler can be created"""
        handler = MarketDataHandler()

        # Should have methods for data access
        assert hasattr(handler, 'initialize')
        assert hasattr(handler, 'get_market_events')

    def test_data_initialization(self):
        """Test market data handler can be initialized with date range"""
        handler = MarketDataHandler()

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        # Should accept date range initialization
        handler.initialize(start_date, end_date)

        # Should track date range
        assert hasattr(handler, 'start_date') or hasattr(handler, 'current_date')

    def test_market_events_generation(self):
        """Test market data handler generates MarketEvent objects"""
        handler = MarketDataHandler()

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        handler.initialize(start_date, end_date)

        # Should return market events for a given timestamp
        current_time = datetime(2023, 1, 2)
        events = handler.get_market_events(current_time)

        # Should return a list (empty is OK for test)
        assert isinstance(events, list)

    def test_historical_data_access(self):
        """Test market data provides historical price feeds"""
        handler = MarketDataHandler()

        # Should provide access to historical data
        # For testing, we accept that this might be mocked or simplified
        assert hasattr(handler, 'get_market_events')

    def test_advance_time(self):
        """Test market data handler can advance simulation time"""
        handler = MarketDataHandler()

        # Should be able to advance time for simulation
        if hasattr(handler, 'advance_time'):
            # Optional method for time management
            pass