# ChessTrader API Reference

Complete programmatic API documentation for ChessTrader Options AI system.

## Table of Contents

- [OptionsAI Class](#optionsai-class)
- [Configuration Management](#configuration-management)
- [Strategy Recommendations](#strategy-recommendations)
- [Backtesting](#backtesting)
- [Error Handling](#error-handling)
- [Examples](#examples)

## OptionsAI Class

The main entry point for all ChessTrader functionality.

### Constructor

```python
class OptionsAI:
    def __init__(self, config_path: Optional[str] = None, config: Optional[Config] = None)
```

**Parameters:**
- `config_path` (str, optional): Path to JSON configuration file
- `config` (Config, optional): Pre-configured Config instance (takes precedence over config_path)

**Example:**
```python
from src.main import OptionsAI

# Default configuration
ai = OptionsAI()

# With configuration file
ai = OptionsAI(config_path="config/production.json")

# With programmatic configuration
from src.config import Config
config = Config()
config.recommendation.confidence_threshold = 0.7
ai = OptionsAI(config=config)
```

### Class Methods

#### version()

Get ChessTrader version information.

```python
@classmethod
def version(cls) -> str
```

**Returns:** Version string

**Example:**
```python
version = OptionsAI.version()
print(f"ChessTrader v{version}")
```

### Instance Methods

#### get_recommendations()

Get AI-powered strategy recommendations for a symbol.

```python
async def get_recommendations(
    self,
    symbol: str,
    **kwargs
) -> List[Dict[str, Any]]
```

**Parameters:**
- `symbol` (str): Stock/ETF symbol to analyze (e.g., 'AAPL', 'SPY')
- `**kwargs`: Additional parameters for recommendation engine

**Returns:** List of recommendation dictionaries with the following keys:
- `strategy` (str): Strategy name (e.g., 'iron_condor', 'bull_put_spread')
- `confidence` (float): Confidence score between 0.0 and 1.0
- `score` (int): Numeric strategy score
- `market_outlook` (str): Expected market direction ('bullish', 'bearish', 'neutral')

**Example:**
```python
import asyncio
from src.main import OptionsAI

async def main():
    ai = OptionsAI()
    recommendations = await ai.get_recommendations('AAPL')

    for rec in recommendations:
        print(f"Strategy: {rec['strategy']}")
        print(f"Confidence: {rec['confidence']:.2%}")
        print(f"Score: {rec['score']}")
        print(f"Outlook: {rec['market_outlook']}")
        print("---")

asyncio.run(main())
```

#### run_backtest()

Execute comprehensive backtesting analysis.

```python
async def run_backtest(
    self,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `config` (dict, optional): Backtest configuration dictionary
- `**kwargs`: Additional backtest parameters

**Configuration Options:**
- `symbol` (str): Symbol to backtest
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date in 'YYYY-MM-DD' format
- `initial_capital` (float): Starting capital amount
- `commission` (float): Commission per contract
- `slippage` (float): Slippage percentage
- `max_position_size` (int): Maximum contracts per position
- `strategy` (str, optional): Specific strategy to test

**Returns:** Dictionary with performance metrics:
- `total_return` (float): Total return percentage
- `sharpe_ratio` (float): Risk-adjusted return metric
- `max_drawdown` (float): Maximum drawdown percentage
- `win_rate` (float): Percentage of profitable trades
- `total_trades` (int): Number of trades executed
- `avg_trade_return` (float): Average return per trade
- `volatility` (float): Return volatility
- `sortino_ratio` (float): Downside risk-adjusted return
- `calmar_ratio` (float): Return to maximum drawdown ratio
- `var_95` (float): Value at Risk (95% confidence)

**Example:**
```python
import asyncio
from src.main import OptionsAI

async def main():
    ai = OptionsAI()

    config = {
        'symbol': 'SPY',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'strategy': 'iron_condor'
    }

    results = await ai.run_backtest(config)

    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")

asyncio.run(main())
```

#### get_strategy_details()

Get detailed information about a specific strategy.

```python
def get_strategy_details(self, strategy_name: str) -> Dict[str, Any]
```

**Parameters:**
- `strategy_name` (str): Name of the strategy

**Returns:** Dictionary with strategy details:
- `name` (str): Strategy name
- `description` (str): Strategy description
- `risk_profile` (str): Risk level ('low', 'medium', 'high')
- `market_outlook` (str): Ideal market conditions
- `typical_duration` (str): Expected holding period
- `max_profit` (str): Maximum profit potential
- `max_loss` (str): Maximum loss potential

**Example:**
```python
from src.main import OptionsAI

ai = OptionsAI()
details = ai.get_strategy_details('iron_condor')

print(f"Strategy: {details['name']}")
print(f"Description: {details['description']}")
print(f"Risk Profile: {details['risk_profile']}")
print(f"Ideal Market: {details['market_outlook']}")
```

#### update_config()

Update configuration dynamically.

```python
def update_config(self, **kwargs) -> None
```

**Parameters:**
- `**kwargs`: Configuration values to update (supports dot notation with double underscores)

**Configuration Sections:**
- `recommendation__*`: Recommendation engine settings
- `backtesting__*`: Backtesting parameters
- `api__*`: API configuration
- `models__*`: Model paths and settings

**Example:**
```python
from src.main import OptionsAI

ai = OptionsAI()

# Update recommendation settings
ai.update_config(
    recommendation__confidence_threshold=0.7,
    recommendation__max_recommendations=5
)

# Update backtesting defaults
ai.update_config(
    backtesting__initial_capital=50000,
    backtesting__commission=1.0
)
```

#### save_config()

Save current configuration to file.

```python
def save_config(self, path: str) -> None
```

**Parameters:**
- `path` (str): Path to save configuration (JSON format)

**Example:**
```python
from src.main import OptionsAI

ai = OptionsAI()
ai.update_config(recommendation__confidence_threshold=0.8)
ai.save_config('config/my_settings.json')
```

## Configuration Management

ChessTrader uses Pydantic for type-safe configuration management.

### Config Class

```python
from src.config import Config

# Create default configuration
config = Config()

# Load from file
config = Config.from_file('config.json')

# Access nested settings
print(config.recommendation.confidence_threshold)
print(config.backtesting.initial_capital)
print(config.api.port)
```

### Configuration Sections

#### Recommendation Settings
```python
config.recommendation.confidence_threshold = 0.4  # Min confidence (0.0-1.0)
config.recommendation.max_recommendations = 3     # Max results (1-10)
config.recommendation.use_historical_data = True  # Use historical performance
```

#### Backtesting Settings
```python
config.backtesting.initial_capital = 100000.0    # Starting capital
config.backtesting.commission = 0.65             # Commission per contract
config.backtesting.slippage = 0.05               # Slippage percentage
config.backtesting.max_position_size = 10        # Max contracts per position
config.backtesting.risk_free_rate = 0.05         # Risk-free rate for Sharpe
```

#### API Settings
```python
config.api.host = "0.0.0.0"        # Server host
config.api.port = 8000             # Server port
config.api.rate_limit = 100        # Requests per minute
config.api.timeout = 30            # Request timeout seconds
```

#### Model Settings
```python
config.models.regime_model_path = "models/regime_detector.pth"
config.models.strategy_selector_path = "models/strategy_selector.pth"
config.models.position_manager_path = "models/position_manager.pth"
config.models.device = "cpu"       # or "cuda"
```

## Strategy Recommendations

### Available Strategies

ChessTrader supports 16 options strategies across different categories:

#### Neutral Strategies
- `iron_condor`: Limited risk/reward neutral strategy
- `iron_butterfly`: High probability neutral strategy
- `short_straddle`: Unlimited risk neutral strategy
- `short_strangle`: Wider neutral profit zone

#### Directional Strategies
- `bull_call_spread`: Limited risk bullish strategy
- `bull_put_spread`: Credit bullish strategy
- `bear_call_spread`: Credit bearish strategy
- `bear_put_spread`: Limited risk bearish strategy
- `covered_call`: Conservative income strategy
- `protective_put`: Downside protection strategy

#### Volatility Strategies
- `long_straddle`: Long volatility, unlimited profit
- `long_strangle`: Lower cost long volatility
- `calendar_spread`: Time decay strategy
- `diagonal_spread`: Advanced time/volatility play

#### Advanced Strategies
- `collar`: Protective strategy with income
- `synthetic_long`: Stock substitute with options

### Recommendation Process

1. **Market Analysis**: Current price, volatility, and trend analysis
2. **Regime Detection**: AI classification of market conditions
3. **Strategy Scoring**: ML-based ranking of suitable strategies
4. **Risk Assessment**: Confidence scoring and risk profiling
5. **Filtering**: Apply confidence thresholds and result limits

## Backtesting

### Backtest Engine Features

- **Event-Driven Simulation**: Realistic order execution
- **Transaction Costs**: Configurable commissions and slippage
- **Risk Management**: Position sizing and stop-loss integration
- **Performance Metrics**: Comprehensive statistical analysis
- **Monte Carlo**: Statistical validation and confidence intervals

### Performance Metrics

#### Return Metrics
- **Total Return**: Overall percentage gain/loss
- **Annualized Return**: Compound annual growth rate
- **Average Trade Return**: Mean return per trade

#### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (return/volatility)
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at confidence level

#### Trade Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss ratio
- **Average Win/Loss**: Mean profitable vs losing trade size
- **Trade Frequency**: Number of trades per period

## Error Handling

ChessTrader provides comprehensive error handling and user-friendly messages.

### Common Exceptions

```python
import asyncio
from src.main import OptionsAI

async def main():
    ai = OptionsAI()

    try:
        # Invalid symbol
        recommendations = await ai.get_recommendations('INVALID')
    except ValueError as e:
        print(f"Invalid symbol: {e}")

    try:
        # Invalid date range
        results = await ai.run_backtest({
            'symbol': 'AAPL',
            'start_date': '2025-01-01',  # Future date
            'end_date': '2024-01-01'     # Before start date
        })
    except ValueError as e:
        print(f"Invalid backtest configuration: {e}")

    try:
        # Unknown strategy
        details = ai.get_strategy_details('unknown_strategy')
    except KeyError as e:
        print(f"Strategy not found: {e}")

asyncio.run(main())
```

### Error Types

- **ValueError**: Invalid parameters or configuration
- **KeyError**: Unknown strategy or missing data
- **FileNotFoundError**: Configuration file not found
- **ConnectionError**: Market data provider unavailable
- **TimeoutError**: Operation exceeded timeout limit

## Examples

### Advanced Configuration

```python
from src.main import OptionsAI
from src.config import Config
import asyncio

# Create custom configuration
config = Config()
config.recommendation.confidence_threshold = 0.8
config.recommendation.max_recommendations = 5
config.backtesting.initial_capital = 250000
config.backtesting.commission = 0.50
config.api.timeout = 60

# Initialize with custom config
ai = OptionsAI(config=config)

async def main():
    # Get high-confidence recommendations
    recommendations = await ai.get_recommendations('SPY')

    if recommendations:
        # Get details for top recommendation
        top_strategy = recommendations[0]['strategy']
        details = ai.get_strategy_details(top_strategy)

        print(f"Top Recommendation: {top_strategy}")
        print(f"Confidence: {recommendations[0]['confidence']:.2%}")
        print(f"Description: {details['description']}")

        # Backtest the recommended strategy
        backtest_config = {
            'symbol': 'SPY',
            'strategy': top_strategy,
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }

        results = await ai.run_backtest(backtest_config)
        print(f"\nBacktest Results:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Win Rate: {results['win_rate']:.2%}")

asyncio.run(main())
```

### Batch Analysis

```python
from src.main import OptionsAI
import asyncio
import pandas as pd

async def analyze_symbols(symbols):
    """Analyze multiple symbols and compare recommendations."""
    ai = OptionsAI()
    results = []

    for symbol in symbols:
        try:
            recommendations = await ai.get_recommendations(symbol)
            if recommendations:
                top_rec = recommendations[0]
                results.append({
                    'symbol': symbol,
                    'strategy': top_rec['strategy'],
                    'confidence': top_rec['confidence'],
                    'score': top_rec['score']
                })
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    return pd.DataFrame(results)

async def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']

    print("Analyzing symbols...")
    df = await analyze_symbols(symbols)

    print(df.to_string(index=False))

    # Find most confident recommendations
    high_confidence = df[df['confidence'] > 0.7]
    print(f"\nHigh confidence recommendations:")
    print(high_confidence.to_string(index=False))

asyncio.run(main())
```

### Configuration from Environment

```python
import os
from src.main import OptionsAI

# Set environment variables
os.environ['CHESSTRADER_RECOMMENDATION__CONFIDENCE_THRESHOLD'] = '0.6'
os.environ['CHESSTRADER_BACKTESTING__INITIAL_CAPITAL'] = '150000'
os.environ['CHESSTRADER_API__PORT'] = '8080'

# Configuration automatically loads from environment
ai = OptionsAI()

print(f"Confidence threshold: {ai.config.recommendation.confidence_threshold}")
print(f"Initial capital: ${ai.config.backtesting.initial_capital:,.0f}")
print(f"API port: {ai.config.api.port}")
```

## Integration Patterns

### Web Application Integration

```python
from fastapi import FastAPI
from src.main import OptionsAI
import asyncio

app = FastAPI()
ai = OptionsAI()

@app.get("/recommendations/{symbol}")
async def get_recommendations(symbol: str, confidence: float = 0.4):
    """Get strategy recommendations via REST API."""
    ai.update_config(recommendation__confidence_threshold=confidence)
    recommendations = await ai.get_recommendations(symbol)
    return {"symbol": symbol, "recommendations": recommendations}

@app.post("/backtest")
async def run_backtest(config: dict):
    """Run backtest via REST API."""
    results = await ai.run_backtest(config)
    return {"results": results}
```

### Jupyter Notebook Usage

```python
# Cell 1: Setup
%load_ext autoreload
%autoreload 2

from src.main import OptionsAI
import asyncio
import pandas as pd
import matplotlib.pyplot as plt

ai = OptionsAI()

# Cell 2: Get recommendations
recommendations = await ai.get_recommendations('AAPL')
df = pd.DataFrame(recommendations)
df.head()

# Cell 3: Visualize confidence scores
plt.figure(figsize=(10, 6))
plt.bar(df['strategy'], df['confidence'])
plt.title('Strategy Confidence Scores for AAPL')
plt.xticks(rotation=45)
plt.ylabel('Confidence')
plt.show()

# Cell 4: Run backtest
config = {'symbol': 'AAPL', 'start_date': '2023-01-01', 'end_date': '2023-12-31'}
results = await ai.run_backtest(config)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```