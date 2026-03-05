"""
Market data handler for historical backtesting

Provides historical price data feeds using yfinance integration.
Generates MarketEvent objects for event-driven backtesting.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from ..core.events import MarketEvent


class MarketDataHandler:
    """
    Historical market data provider for backtesting

    Manages historical price data and generates MarketEvent objects
    for sequential processing in the backtesting engine.
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize market data handler

        Args:
            symbols: List of symbols to track (default: ['SPY'])
        """
        self.symbols = symbols or ['SPY']
        self.start_date = None
        self.end_date = None
        self.current_date = None

        # Historical data storage (symbol -> date -> data)
        self.historical_data: Dict[str, Dict[datetime, Dict]] = {}

        # Current simulation state
        self.is_initialized = False

        self.logger = logging.getLogger(__name__)

    def initialize(self, start_date: datetime, end_date: datetime):
        """
        Initialize market data for backtest period

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date
        self.is_initialized = True

        self.logger.info(f"Initialized market data for {start_date} to {end_date}")

        # In a real implementation, this would load data from yfinance
        # For now, we'll simulate some basic price data
        self._load_sample_data()

    def _load_sample_data(self):
        """Load sample historical data for testing"""
        # Generate sample price data for testing
        # In production, this would use yfinance to load real data

        base_prices = {'SPY': 400.0, 'QQQ': 300.0, 'IWM': 200.0}

        for symbol in self.symbols:
            if symbol not in base_prices:
                base_prices[symbol] = 100.0

            self.historical_data[symbol] = {}

            # Generate daily price data
            current_date = self.start_date
            price = base_prices[symbol]

            while current_date <= self.end_date:
                # Simple random walk for sample data
                price *= (1 + (hash(str(current_date) + symbol) % 41 - 20) / 10000)

                self.historical_data[symbol][current_date] = {
                    'open': price * 0.999,
                    'high': price * 1.002,
                    'low': price * 0.998,
                    'close': price,
                    'price': price,  # For compatibility
                    'volume': 1000000,
                    'bid': price * 0.9995,
                    'ask': price * 1.0005
                }

                current_date += timedelta(days=1)

        self.logger.info(f"Loaded sample data for {len(self.symbols)} symbols")

    def get_market_events(self, current_time: datetime) -> List[MarketEvent]:
        """
        Get market events for current simulation time

        Args:
            current_time: Current simulation timestamp

        Returns:
            List of MarketEvent objects for all symbols
        """
        if not self.is_initialized:
            return []

        events = []
        simulation_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

        for symbol in self.symbols:
            if (symbol in self.historical_data and
                simulation_date in self.historical_data[symbol]):

                market_data = self.historical_data[symbol][simulation_date]

                event = MarketEvent(
                    timestamp=current_time,
                    symbol=symbol,
                    data=market_data
                )

                events.append(event)

        return events

    def advance_time(self) -> Optional[datetime]:
        """
        Advance simulation time by one trading day

        Returns:
            Next simulation date or None if simulation complete
        """
        if not self.is_initialized or self.current_date is None:
            return None

        self.current_date += timedelta(days=1)

        if self.current_date > self.end_date:
            return None

        return self.current_date

    def get_latest_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """
        Get latest available price for a symbol

        Args:
            symbol: Trading symbol
            timestamp: Current timestamp

        Returns:
            Latest price or None if not available
        """
        simulation_date = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

        if (symbol in self.historical_data and
            simulation_date in self.historical_data[symbol]):

            return self.historical_data[symbol][simulation_date]['price']

        return None

    def add_symbol(self, symbol: str):
        """Add a new symbol to track"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)

            # If already initialized, load data for this symbol
            if self.is_initialized:
                self._load_sample_data_for_symbol(symbol)

    def _load_sample_data_for_symbol(self, symbol: str):
        """Load sample data for a single symbol"""
        # Simplified version of _load_sample_data for single symbol
        base_price = 100.0
        self.historical_data[symbol] = {}

        current_date = self.start_date
        price = base_price

        while current_date <= self.end_date:
            price *= (1 + (hash(str(current_date) + symbol) % 41 - 20) / 10000)

            self.historical_data[symbol][current_date] = {
                'open': price * 0.999,
                'high': price * 1.002,
                'low': price * 0.998,
                'close': price,
                'price': price,
                'volume': 1000000,
                'bid': price * 0.9995,
                'ask': price * 1.0005
            }

            current_date += timedelta(days=1)