# ChessTrader CLI Guide

Complete command-line interface guide for ChessTrader Options AI system.

## Table of Contents

- [Installation](#installation)
- [Global Options](#global-options)
- [Commands Overview](#commands-overview)
- [Recommend Command](#recommend-command)
- [Backtest Command](#backtest-command)
- [Configuration](#configuration)
- [Workflows](#workflows)
- [Troubleshooting](#troubleshooting)

## Installation

### Package Installation

```bash
# Install as package (recommended)
pip install -e .

# Verify installation
chesstrader --version
```

### Development Usage

```bash
# Run directly from source
python chesstrader.py --help

# Or with python -m
python -m src.cli.main --help
```

## Global Options

Available for all commands:

```bash
chesstrader [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to configuration file (JSON) | None |
| `--verbose` | `-v` | Enable verbose output | False |
| `--version` | | Show version and exit | |
| `--help` | | Show help message | |

### Examples

```bash
# Show version information
chesstrader --version

# Use custom configuration
chesstrader --config config/production.json recommend AAPL

# Enable verbose logging
chesstrader --verbose backtest --symbol SPY
```

## Commands Overview

ChessTrader CLI provides three main commands:

1. **`recommend`** - Get AI-powered strategy recommendations
2. **`backtest`** - Run comprehensive backtesting analysis
3. **`version`** - Display detailed version information

## Recommend Command

Get intelligent options strategy recommendations based on AI analysis.

### Basic Syntax

```bash
chesstrader recommend SYMBOL [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `SYMBOL` | string | Stock/ETF symbol to analyze | Yes |

### Options

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--confidence` | `-c` | float | Minimum confidence threshold (0.0-1.0) | 0.4 |
| `--max-results` | `-n` | int | Maximum number of recommendations (1-10) | 3 |
| `--details` | `-d` | flag | Show detailed strategy information | False |
| `--config` | | string | Path to configuration file | None |

### Examples

#### Basic Usage
```bash
# Get recommendations for Apple
chesstrader recommend AAPL

# Get recommendations for S&P 500 ETF
chesstrader recommend SPY
```

#### Advanced Usage
```bash
# Higher confidence threshold with more results
chesstrader recommend AAPL --confidence 0.7 --max-results 5

# Show detailed strategy information
chesstrader recommend MSFT --details

# Combine options
chesstrader recommend GOOGL -c 0.6 -n 4 -d
```

### Output Format

The recommend command displays results in a formatted table:

```
🎯 Strategy Recommendations for AAPL
Confidence threshold: 60.0% | Max results: 3

┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Strategy      ┃ Confidence  ┃ Score ┃ Market Outlook ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Iron Condor   │    85.0%    │  92   │ neutral        │
│ Bull Put      │    72.0%    │  78   │ bullish        │
│ Calendar      │    68.0%    │  75   │ low_volatility │
└───────────────┴─────────────┴───────┴────────────────┘

💡 Strategy Details:
1. Iron Condor: Low-risk neutral strategy with limited profit potential
2. Bull Put: Credit spread betting on bullish price movement
3. Calendar: Time decay strategy benefiting from low volatility

📊 Summary: 3 recommendations found | Average confidence: 75.0% | High confidence (≥70%): 3
```

#### Color Coding

- 🟢 **Green** (≥80%): High confidence recommendations
- 🟡 **Yellow** (≥60%): Medium confidence recommendations
- 🔴 **Red** (<60%): Lower confidence recommendations

## Backtest Command

Execute comprehensive backtesting analysis with professional reporting.

### Basic Syntax

```bash
chesstrader backtest [OPTIONS]
```

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--symbol` | string | Stock/ETF symbol to backtest | Required |
| `--strategy` | string | Specific strategy to test | AI recommendation |
| `--start-date` | string | Start date (YYYY-MM-DD) | 1 year ago |
| `--end-date` | string | End date (YYYY-MM-DD) | Today |
| `--capital` | float | Initial capital amount | 100000.0 |
| `--commission` | float | Commission per contract | 0.65 |
| `--max-positions` | int | Maximum contracts per position | 10 |
| `--output` | string | Output format (console/csv/html/pdf) | console |
| `--save-report` | flag | Save detailed report to file | False |
| `--config` | string | Path to configuration file | None |

### Examples

#### Basic Usage
```bash
# Backtest with default parameters
chesstrader backtest --symbol AAPL

# Backtest specific strategy
chesstrader backtest --symbol SPY --strategy iron_condor
```

#### Advanced Usage
```bash
# Custom date range and capital
chesstrader backtest --symbol MSFT \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --capital 250000

# Full configuration with report saving
chesstrader backtest --symbol QQQ \
  --strategy bull_put_spread \
  --start-date 2022-01-01 \
  --capital 150000 \
  --commission 1.0 \
  --max-positions 5 \
  --save-report \
  --output html
```

### Output Format

The backtest command displays comprehensive results in organized panels:

```
🔬 Backtesting Analysis for AAPL
Period: 2023-01-01 to 2023-12-31 | Capital: $100,000

📊 Key Performance Metrics
┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Total Return  ┃  15.2%  ┃
┃ Sharpe Ratio  ┃   1.23  ┃
┃ Max Drawdown  ┃  -8.1%  ┃
┃ Win Rate      ┃  65.0%  ┃
┃ Total Trades  ┃    24   ┃
┃ Avg Trade     ┃   1.8%  ┃
┗━━━━━━━━━━━━━━━┻━━━━━━━━━┛

⚠️ Risk Analysis
┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Volatility    ┃  12.4%  ┃
┃ Sortino Ratio ┃   1.45  ┃
┃ Calmar Ratio  ┃   1.88  ┃
┃ VaR (95%)     ┃  -3.2%  ┃
┗━━━━━━━━━━━━━━━┻━━━━━━━━━┛

📈 Trade Statistics
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Profitable Trades┃   16    ┃
┃ Loss Trades      ┃    8    ┃
┃ Largest Win      ┃   8.2%  ┃
┃ Largest Loss     ┃  -4.1%  ┃
┃ Avg Win          ┃   3.1%  ┃
┃ Avg Loss         ┃  -1.8%  ┃
┗━━━━━━━━━━━━━━━━━━┻━━━━━━━━━┛

Overall Performance: ✅ Good

Strategy: Iron Condor on AAPL
📄 Report saved: backtest_AAPL_iron_condor_20240412_143022.html
Format: HTML | Generated: 2024-04-12 14:30:22
```

### Available Strategies

When using `--strategy`, choose from:

#### Neutral Strategies
- `iron_condor`
- `iron_butterfly`
- `short_straddle`
- `short_strangle`

#### Directional Strategies
- `bull_call_spread`
- `bull_put_spread`
- `bear_call_spread`
- `bear_put_spread`
- `covered_call`
- `protective_put`

#### Volatility Strategies
- `long_straddle`
- `long_strangle`
- `calendar_spread`
- `diagonal_spread`

#### Advanced Strategies
- `collar`
- `synthetic_long`

## Configuration

ChessTrader supports flexible configuration through JSON files and environment variables.

### Configuration File Format

```json
{
  "recommendation": {
    "confidence_threshold": 0.6,
    "max_recommendations": 5,
    "use_historical_data": true
  },
  "backtesting": {
    "initial_capital": 150000,
    "commission": 0.50,
    "slippage": 0.03,
    "max_position_size": 15,
    "risk_free_rate": 0.05
  },
  "api": {
    "host": "localhost",
    "port": 8000,
    "timeout": 45
  },
  "models": {
    "device": "cpu"
  }
}
```

### Using Configuration Files

```bash
# Create configuration file
cat > config/trading.json << EOF
{
  "recommendation": {"confidence_threshold": 0.7},
  "backtesting": {"initial_capital": 200000}
}
EOF

# Use configuration file
chesstrader --config config/trading.json recommend AAPL
chesstrader --config config/trading.json backtest --symbol SPY
```

### Environment Variables

Set configuration through environment variables with `CHESSTRADER_` prefix:

```bash
# Set recommendation confidence
export CHESSTRADER_RECOMMENDATION__CONFIDENCE_THRESHOLD=0.7

# Set backtesting capital
export CHESSTRADER_BACKTESTING__INITIAL_CAPITAL=150000

# Set API port
export CHESSTRADER_API__PORT=8080

# Use in commands
chesstrader recommend AAPL
chesstrader backtest --symbol SPY
```

## Workflows

### Complete Analysis Workflow

1. **Get Recommendations**
```bash
# Step 1: Get AI recommendations
chesstrader recommend AAPL --confidence 0.6 --details
```

2. **Analyze Top Strategy**
```bash
# Step 2: Backtest recommended strategy
chesstrader backtest --symbol AAPL --strategy iron_condor --save-report
```

3. **Compare Alternatives**
```bash
# Step 3: Test alternative strategies
chesstrader backtest --symbol AAPL --strategy bull_put_spread
chesstrader backtest --symbol AAPL --strategy calendar_spread
```

### Multi-Symbol Analysis

```bash
# Analyze multiple symbols
for symbol in AAPL MSFT GOOGL TSLA SPY; do
  echo "=== Analyzing $symbol ==="
  chesstrader recommend $symbol --confidence 0.7
  echo ""
done
```

### Historical Performance Study

```bash
# Compare performance across different periods
chesstrader backtest --symbol SPY --start-date 2020-01-01 --end-date 2020-12-31
chesstrader backtest --symbol SPY --start-date 2021-01-01 --end-date 2021-12-31
chesstrader backtest --symbol SPY --start-date 2022-01-01 --end-date 2022-12-31
chesstrader backtest --symbol SPY --start-date 2023-01-01 --end-date 2023-12-31
```

### Strategy Comparison

```bash
# Compare all neutral strategies on same symbol
strategies=("iron_condor" "iron_butterfly" "short_straddle" "short_strangle")

for strategy in "${strategies[@]}"; do
  echo "=== Testing $strategy ==="
  chesstrader backtest --symbol SPY --strategy $strategy --start-date 2023-01-01
  echo ""
done
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Command Not Found

**Error:** `chesstrader: command not found`

**Solutions:**
```bash
# Install package
pip install -e .

# Or run directly
python chesstrader.py --help

# Check Python path
which python
pip list | grep chesstrader
```

#### 2. Import Errors

**Error:** `ModuleNotFoundError: No module named 'yfinance'`

**Solutions:**
```bash
# Install dependencies
pip install -r requirements.txt

# Check requirements
pip check

# Reinstall package
pip uninstall chesstrader
pip install -e .
```

#### 3. No Recommendations Found

**Output:** `No recommendations found for SYMBOL`

**Possible Causes:**
- Confidence threshold too high
- Invalid or unsupported symbol
- Insufficient market data
- Network connectivity issues

**Solutions:**
```bash
# Lower confidence threshold
chesstrader recommend AAPL --confidence 0.3

# Try different symbol
chesstrader recommend SPY

# Check network connectivity
ping finance.yahoo.com
```

#### 4. Backtest Errors

**Error:** `Invalid date range` or `Insufficient historical data`

**Solutions:**
```bash
# Use valid date range
chesstrader backtest --symbol AAPL --start-date 2023-01-01 --end-date 2023-12-31

# Check symbol validity
chesstrader recommend AAPL  # Should work if symbol is valid

# Use shorter date range
chesstrader backtest --symbol AAPL --start-date 2023-06-01
```

#### 5. Configuration Issues

**Error:** `Config file not found` or `Invalid JSON`

**Solutions:**
```bash
# Check file exists
ls -la config/trading.json

# Validate JSON
python -m json.tool config/trading.json

# Use absolute path
chesstrader --config /full/path/to/config.json recommend AAPL
```

### Getting Help

#### Command Help
```bash
# Global help
chesstrader --help

# Command-specific help
chesstrader recommend --help
chesstrader backtest --help
```

#### Debug Mode
```bash
# Enable verbose output
chesstrader --verbose recommend AAPL

# Check configuration
python -c "from src.config import Config; print(Config().dict())"
```

#### Log Analysis
```bash
# Check for error logs
tail -f chesstrader.log

# Enable debug logging
export CHESSTRADER_SYSTEM__LOG_LEVEL=DEBUG
chesstrader recommend AAPL
```

### Performance Tips

#### 1. Improve Recommendation Speed
```bash
# Use specific confidence threshold
chesstrader recommend AAPL --confidence 0.5

# Limit results
chesstrader recommend AAPL --max-results 3
```

#### 2. Optimize Backtesting
```bash
# Use shorter date ranges for faster results
chesstrader backtest --symbol AAPL --start-date 2023-09-01

# Reduce position size for faster simulation
chesstrader backtest --symbol AAPL --max-positions 5
```

#### 3. Configuration Caching
```bash
# Save frequently used configuration
chesstrader --config config/fast.json recommend AAPL

# Use environment variables for persistent settings
export CHESSTRADER_RECOMMENDATION__CONFIDENCE_THRESHOLD=0.6
```

## Advanced Usage

### Scripting with CLI

#### Bash Script Example
```bash
#!/bin/bash
# analyze_portfolio.sh

SYMBOLS=("AAPL" "MSFT" "GOOGL" "AMZN" "SPY")
CONFIDENCE=0.7

echo "Portfolio Analysis Report"
echo "========================"
date
echo ""

for symbol in "${SYMBOLS[@]}"; do
    echo "--- $symbol ---"

    # Get recommendations
    chesstrader recommend $symbol --confidence $CONFIDENCE --max-results 1

    # Get top recommendation and backtest it
    # (In practice, you'd parse the JSON output and extract strategy name)

    echo ""
done
```

#### Python Integration
```python
import subprocess
import json

def get_recommendations(symbol, confidence=0.6):
    """Get recommendations via CLI."""
    cmd = [
        'chesstrader', 'recommend', symbol,
        '--confidence', str(confidence),
        '--output', 'json'  # If JSON output is implemented
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        raise Exception(f"CLI error: {result.stderr}")

# Use in Python script
symbols = ['AAPL', 'MSFT', 'GOOGL']
for symbol in symbols:
    recs = get_recommendations(symbol, confidence=0.7)
    print(f"{symbol}: {recs[0]['strategy']} ({recs[0]['confidence']:.1%})")
```

### Integration with Other Tools

#### Pipeline with jq
```bash
# Extract specific fields (if JSON output available)
chesstrader recommend AAPL --output json | jq '.recommendations[0].strategy'

# Filter high-confidence recommendations
chesstrader recommend AAPL --output json | jq '.recommendations[] | select(.confidence > 0.7)'
```

#### CSV Export for Analysis
```bash
# Generate CSV reports
chesstrader backtest --symbol AAPL --output csv --save-report

# Process with other tools
cat backtest_AAPL_*.csv | head -5
```

This completes the comprehensive CLI guide for ChessTrader. The interface provides professional-grade options trading analysis accessible through simple, intuitive commands.