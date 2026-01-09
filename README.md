# ChessTrader - Options AI Trading System

An AI-powered options trading system that applies game-theoretic frameworks and reinforcement learning to strategy selection and position management.

## Overview

ChessTrader models options strategies as bounded games (similar to chess) to leverage proven AI training methodologies, targeting 60%+ win rate with <15% drawdown through systematic regime detection and intelligent position management.

## Features

- Market regime detection and classification
- AI-powered options strategy selection
- Reinforcement learning position management
- Real-time data pipeline with caching
- Backtesting engine with walk-forward optimization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.data.providers import YFinanceProvider
from src.data.cache import CacheManager

# Initialize data provider and cache
provider = YFinanceProvider()
cache = CacheManager()

# Get price data
data = cache.get_price_history('SPY')
```

## Project Structure

```
src/
├── data/           # Data providers and caching
├── models/         # AI models and algorithms
├── strategies/     # Options trading strategies
└── analysis/       # Market analysis and backtesting
```

## License

This project is for educational and research purposes only. Not for actual trading.