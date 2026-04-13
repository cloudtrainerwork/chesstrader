# ChessTrader - AI-Powered Options Trading System

<div align="center">

![ChessTrader Logo](https://img.shields.io/badge/ChessTrader-AI%20Options%20Trading-blue?style=for-the-badge)

**Revolutionary options trading system combining chess AI with financial markets**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples)

</div>

## Overview

ChessTrader is a sophisticated AI-powered options trading system that applies chess engine principles to financial markets. By combining spatial neural networks, regime detection, and reinforcement learning, it provides intelligent strategy recommendations and comprehensive backtesting capabilities.

### 🎯 Core Value Proposition

- **Chess-Inspired AI**: Leverages spatial pattern recognition from chess engines
- **16 Options Strategies**: Complete coverage from conservative to aggressive approaches
- **Regime Detection**: AI identifies market conditions for optimal strategy selection
- **ML-Driven Recommendations**: Confidence-scored suggestions with risk assessment
- **Professional Backtesting**: Monte Carlo simulation with walk-forward optimization
- **Production Ready**: Both CLI and programmatic API interfaces

## Features

### 🧠 **Artificial Intelligence Engine**
- **Spatial Neural Networks**: Chess-inspired 7x6 board representation for options positions
- **Regime Detection**: 8-regime market classification with confidence scoring
- **Reinforcement Learning**: PPO-trained position management with game-theoretic modeling
- **Strategy Selection**: ML-powered ranking of 16 options strategies

### 📊 **Trading Strategies**
- **Neutral**: Iron Condor, Iron Butterfly, Short Straddle, Short Strangle
- **Directional**: Bull/Bear Call/Put Spreads, Covered Call, Protective Put
- **Volatility**: Long Straddle, Long Strangle, Calendar Spreads, Diagonal Spreads
- **Advanced**: Collar, Synthetic positions with dynamic risk management

### 🔬 **Backtesting & Analysis**
- **Event-Driven Engine**: Realistic order execution with slippage and commissions
- **Walk-Forward Optimization**: Out-of-sample validation with ML pipeline integration
- **Monte Carlo Simulation**: Statistical confidence intervals and risk metrics
- **Performance Reporting**: Professional tearsheets with HTML/PDF export

### 💻 **Interfaces**
- **CLI Application**: Rich-formatted command-line interface with progress indicators
- **Python API**: Async-first programmatic access for custom applications
- **Configuration**: Flexible JSON/environment variable configuration system

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chesstrader/chesstrader.git
cd chesstrader

# Install dependencies
pip install -r requirements.txt

# Install as editable package (enables 'chesstrader' command)
pip install -e .

# Verify installation
chesstrader --version
```

### Basic Usage

#### CLI Commands

```bash
# Get AI-powered strategy recommendations
chesstrader recommend AAPL
chesstrader recommend SPY --confidence 0.6 --max-results 5 --details

# Run comprehensive backtesting
chesstrader backtest --symbol AAPL
chesstrader backtest --symbol SPY --strategy iron_condor --start-date 2023-01-01 --save-report

# View help for any command
chesstrader --help
chesstrader recommend --help
chesstrader backtest --help
```

#### Programmatic API

```python
import asyncio
from src.main import OptionsAI

# Initialize the AI system
ai = OptionsAI()

# Get strategy recommendations
async def get_recommendations():
    recommendations = await ai.get_recommendations('AAPL')
    for rec in recommendations:
        print(f"{rec['strategy']}: {rec['confidence']:.2%} confidence")

# Run backtest
async def run_backtest():
    results = await ai.run_backtest({
        'symbol': 'SPY',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'strategy': 'iron_condor'
    })
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

# Execute async functions
asyncio.run(get_recommendations())
asyncio.run(run_backtest())
```

### Example Output

**Strategy Recommendations:**
```
🎯 Strategy Recommendations for AAPL
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Strategy      ┃ Confidence  ┃ Score ┃ Market Outlook ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Iron Condor   │    85.0%    │  92   │ neutral        │
│ Bull Put      │    72.0%    │  78   │ bullish        │
│ Calendar      │    68.0%    │  75   │ low_volatility │
└───────────────┴─────────────┴───────┴────────────────┘
```

**Backtesting Results:**
```
📊 Key Performance Metrics
┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Total Return  ┃  15.2%  ┃
┃ Sharpe Ratio  ┃   1.23  ┃
┃ Max Drawdown  ┃  -8.1%  ┃
┃ Win Rate      ┃  65.0%  ┃
┗━━━━━━━━━━━━━━━┻━━━━━━━━━┛

Overall Performance: ✅ Good
```

## System Architecture

ChessTrader implements a sophisticated multi-layer architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                     │
│  ┌─────────────────┐    ┌─────────────────────────────┐│
│  │   CLI (Typer)   │    │    Python API (Async)      ││
│  └─────────────────┘    └─────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                   Core AI Engine                       │
│  ┌─────────────────┐    ┌─────────────────────────────┐│
│  │ Strategy        │    │  Position Manager          ││
│  │ Recommender     │    │  (PPO + RL)                ││
│  └─────────────────┘    └─────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                 Neural Architecture                    │
│  ┌─────────────────┐    ┌─────────────────────────────┐│
│  │ Spatial Encoder │    │   Regime Detector          ││
│  │ (Chess-inspired)│    │   (8-regime classification)││
│  └─────────────────┘    └─────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                 Backtesting Engine                     │
│  ┌─────────────────┐    ┌─────────────────────────────┐│
│  │ Event-Driven    │    │  Monte Carlo Simulator     ││
│  │ Simulation      │    │  + Statistical Analysis    ││
│  └─────────────────┘    └─────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│               Strategy Framework                        │
│              16 Options Strategies                     │
│     Iron Condor • Straddles • Spreads • Collars       │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**: See `requirements.txt` for complete list
- **Memory**: 4GB+ RAM recommended for neural network operations
- **Storage**: 1GB+ for caching market data and model weights

### Key Dependencies

```
torch>=2.0.0              # Neural network framework
scikit-learn>=1.3.0        # Machine learning utilities
pandas>=2.0.0              # Data manipulation
yfinance>=0.2.28           # Market data provider
typer>=0.9.0               # CLI framework
rich>=13.7.0               # Terminal formatting
fastapi>=0.100.0           # Web API framework
pydantic>=2.0.0            # Configuration management
```

## Documentation

- **[API Reference](docs/api_reference.md)** - Complete programmatic API documentation
- **[CLI Guide](docs/cli_guide.md)** - Comprehensive command-line interface guide
- **[Examples](examples/)** - Working code examples and tutorials

## Examples

### Strategy Recommendation
```python
# examples/basic_usage.py - Get AI recommendations
from src.main import OptionsAI
import asyncio

async def main():
    ai = OptionsAI()

    # Get recommendations with custom confidence threshold
    ai.update_config(recommendation__confidence_threshold=0.7)

    recommendations = await ai.get_recommendations('SPY')
    for rec in recommendations:
        details = ai.get_strategy_details(rec['strategy'])
        print(f"Strategy: {rec['strategy']}")
        print(f"Confidence: {rec['confidence']:.2%}")
        print(f"Description: {details['description']}")
        print("---")

asyncio.run(main())
```

### Backtesting Workflow
```python
# examples/backtest_example.py - Comprehensive backtesting
from src.main import OptionsAI
import asyncio

async def main():
    ai = OptionsAI()

    # Configure backtest parameters
    config = {
        'symbol': 'QQQ',
        'start_date': '2022-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'commission': 0.65,
        'strategy': 'iron_condor'
    }

    # Run backtest with progress tracking
    results = await ai.run_backtest(config)

    # Analyze results
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")

    # Save detailed report
    ai.save_config('backtest_config.json')

asyncio.run(main())
```

## Development

### Project Structure
```
chesstrader/
├── src/                    # Source code
│   ├── main.py            # OptionsAI main class
│   ├── config.py          # Configuration management
│   ├── cli/               # Command-line interface
│   ├── api/               # Strategy recommendation API
│   ├── models/            # Neural network models
│   ├── strategies/        # Options strategy implementations
│   ├── backtesting/       # Backtesting engine
│   ├── features/          # Feature engineering
│   └── data/              # Data providers
├── tests/                 # Test suite
├── examples/              # Usage examples
├── docs/                  # Documentation
├── .planning/             # Development planning (GSD workflow)
└── requirements.txt       # Dependencies
```

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test files
pytest tests/test_main.py -v
pytest tests/cli/ -v
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Submit a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/chesstrader/chesstrader/issues)
- **Documentation**: [API Reference](docs/api_reference.md)
- **Examples**: [examples/](examples/)

## Acknowledgments

- Chess AI research community for spatial pattern recognition techniques
- Options trading community for strategy frameworks and risk management principles
- Open source machine learning ecosystem (PyTorch, scikit-learn, pandas)

---

<div align="center">

**Built with ❤️ for the intersection of artificial intelligence and quantitative finance**

[⭐ Star on GitHub](https://github.com/chesstrader/chesstrader) • [📖 Documentation](docs/) • [🚀 Examples](examples/)

</div>