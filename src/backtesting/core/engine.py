"""
Event-driven backtesting engine

Coordinates market data, portfolio management, strategy signals,
and order execution in a sequential, bias-free manner.
"""

import queue
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from .events import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent


class BacktestEngine:
    """
    Event-driven backtesting engine

    Processes events sequentially to prevent look-ahead bias.
    Coordinates between market data, portfolio, strategies, and execution.
    """

    def __init__(self):
        """Initialize backtesting engine with empty event queue"""
        self.event_queue = queue.Queue()
        self.continue_backtest = True

        # Component handlers (set externally)
        self.data_handler = None
        self.portfolio = None
        self.execution_handler = None
        self.strategies = []

        # State tracking
        self.current_time = None
        self.events_processed = 0

        # Configure logging
        self.logger = logging.getLogger(__name__)

    def add_strategy(self, strategy):
        """Add a strategy to the backtesting engine"""
        self.strategies.append(strategy)

    def _process_event(self):
        """
        Process a single event from the queue

        Dispatches events to appropriate handlers based on event type.
        Maintains strict sequential processing to prevent look-ahead bias.
        """
        if self.event_queue.empty():
            return

        try:
            event = self.event_queue.get(block=False)
            self.events_processed += 1

            if event.type == EventType.MARKET:
                self._process_market_event(event)
            elif event.type == EventType.SIGNAL:
                self._process_signal_event(event)
            elif event.type == EventType.ORDER:
                self._process_order_event(event)
            elif event.type == EventType.FILL:
                self._process_fill_event(event)

            self.current_time = event.timestamp

        except queue.Empty:
            pass

    def _process_market_event(self, event: MarketEvent):
        """Process market data event"""
        # Update portfolio mark-to-market values
        if self.portfolio:
            self.portfolio.update_market_value(event)

        # Generate strategy signals
        for strategy in self.strategies:
            signals = strategy.generate_signals(event)
            for signal in signals:
                self.event_queue.put(signal)

    def _process_signal_event(self, event: SignalEvent):
        """Process strategy signal event"""
        if self.portfolio:
            orders = self.portfolio.generate_orders(event)
            for order in orders:
                self.event_queue.put(order)

    def _process_order_event(self, event: OrderEvent):
        """Process order execution event"""
        if self.execution_handler:
            fill_event = self.execution_handler.execute_order(event)
            if fill_event:
                self.event_queue.put(fill_event)

    def _process_fill_event(self, event: FillEvent):
        """Process order fill event"""
        if self.portfolio:
            self.portfolio.update_fill(event)

    def run_backtest(self, start_date: datetime, end_date: datetime):
        """
        Run complete backtest from start to end date

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Initialize components
        if self.data_handler:
            self.data_handler.initialize(start_date, end_date)

        if self.portfolio:
            self.portfolio.initialize(start_date)

        self.current_time = start_date
        self.continue_backtest = True

        # Main event loop
        while self.continue_backtest and self.current_time <= end_date:
            # Get new market data (if available)
            if self.data_handler:
                market_events = self.data_handler.get_market_events(self.current_time)
                for event in market_events:
                    self.event_queue.put(event)

            # Process all events in queue
            while not self.event_queue.empty() and self.continue_backtest:
                self._process_event()

            # Advance time (this would typically be driven by data handler)
            if self.data_handler and hasattr(self.data_handler, 'advance_time'):
                self.current_time = self.data_handler.advance_time()
            else:
                # Fallback: advance by one day
                self.current_time += timedelta(days=1)

        self.logger.info(f"Backtest completed. Processed {self.events_processed} events")

    def stop_backtest(self):
        """Stop the backtest gracefully"""
        self.continue_backtest = False

    def get_portfolio_summary(self):
        """Get final portfolio performance summary"""
        if self.portfolio:
            return self.portfolio.get_performance_summary()
        return None