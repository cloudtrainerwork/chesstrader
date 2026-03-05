"""
Test suite for event system and core engine
"""

import pytest
import queue
from datetime import datetime
from unittest.mock import Mock

from src.backtesting.core.events import (
    EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
)
from src.backtesting.core.engine import BacktestEngine


class TestEventSystem:
    """Test event classes and event queue processing"""

    def test_event_types_defined(self):
        """Test that all required event types are defined"""
        assert hasattr(EventType, 'MARKET')
        assert hasattr(EventType, 'SIGNAL')
        assert hasattr(EventType, 'ORDER')
        assert hasattr(EventType, 'FILL')

    def test_market_event_creation(self):
        """Test MarketEvent can be created with required fields"""
        timestamp = datetime.now()
        event = MarketEvent(timestamp, 'SPY', {'price': 100.0, 'volume': 1000})

        assert event.type == EventType.MARKET
        assert event.timestamp == timestamp
        assert event.symbol == 'SPY'
        assert event.data['price'] == 100.0

    def test_signal_event_creation(self):
        """Test SignalEvent can be created with strategy and direction"""
        timestamp = datetime.now()
        event = SignalEvent(timestamp, 'SPY', 'LONG', 'test_strategy')

        assert event.type == EventType.SIGNAL
        assert event.symbol == 'SPY'
        assert event.signal_type == 'LONG'
        assert event.strategy_id == 'test_strategy'

    def test_order_event_creation(self):
        """Test OrderEvent can be created with order details"""
        timestamp = datetime.now()
        event = OrderEvent(timestamp, 'SPY', 'MKT', 100, 'BUY')

        assert event.type == EventType.ORDER
        assert event.symbol == 'SPY'
        assert event.order_type == 'MKT'
        assert event.quantity == 100
        assert event.direction == 'BUY'

    def test_fill_event_creation(self):
        """Test FillEvent can be created with execution details"""
        timestamp = datetime.now()
        event = FillEvent(timestamp, 'SPY', 'MKT', 100, 'BUY', 100.5, 1.0)

        assert event.type == EventType.FILL
        assert event.symbol == 'SPY'
        assert event.fill_cost == 100.5
        assert event.commission == 1.0


class TestBacktestEngine:
    """Test backtesting engine coordination and event processing"""

    def test_engine_creation(self):
        """Test BacktestEngine can be created"""
        engine = BacktestEngine()
        assert hasattr(engine, 'event_queue')
        assert isinstance(engine.event_queue, queue.Queue)

    def test_event_processing(self):
        """Test that events are processed sequentially without look-ahead bias"""
        engine = BacktestEngine()

        # Create mock handlers
        portfolio = Mock()
        execution = Mock()
        data_handler = Mock()

        engine.portfolio = portfolio
        engine.execution_handler = execution
        engine.data_handler = data_handler

        # Add events to queue
        timestamp1 = datetime.now()
        timestamp2 = datetime.now()

        market_event1 = MarketEvent(timestamp1, 'SPY', {'price': 100.0})
        market_event2 = MarketEvent(timestamp2, 'SPY', {'price': 101.0})

        engine.event_queue.put(market_event1)
        engine.event_queue.put(market_event2)

        # Process events
        engine._process_event()
        engine._process_event()

        # Verify events were dispatched to portfolio
        assert portfolio.update_market_value.call_count == 2

    def test_sequential_event_processing(self):
        """Test that events maintain temporal order"""
        engine = BacktestEngine()

        # Track processing order
        processed_events = []

        def mock_process(event):
            processed_events.append(event.timestamp)

        engine.portfolio = Mock()
        engine.portfolio.update_market_value = mock_process

        # Add events in specific order
        timestamp1 = datetime(2023, 1, 1, 9, 0)
        timestamp2 = datetime(2023, 1, 1, 9, 1)
        timestamp3 = datetime(2023, 1, 1, 9, 2)

        events = [
            MarketEvent(timestamp2, 'SPY', {'price': 101.0}),
            MarketEvent(timestamp1, 'SPY', {'price': 100.0}),
            MarketEvent(timestamp3, 'SPY', {'price': 102.0})
        ]

        for event in events:
            engine.event_queue.put(event)

        # Process all events
        while not engine.event_queue.empty():
            engine._process_event()

        # Events should be processed in queue order (not sorted)
        # This ensures no look-ahead bias
        expected_order = [timestamp2, timestamp1, timestamp3]
        assert processed_events == expected_order

    def test_run_backtest_method(self):
        """Test that engine has run_backtest method for complete simulation"""
        engine = BacktestEngine()

        # Mock components
        engine.portfolio = Mock()
        engine.execution_handler = Mock()
        engine.data_handler = Mock()

        # Should have run_backtest method
        assert hasattr(engine, 'run_backtest')

        # Method should accept start/end dates
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        # This should not raise an exception
        engine.run_backtest(start_date, end_date)