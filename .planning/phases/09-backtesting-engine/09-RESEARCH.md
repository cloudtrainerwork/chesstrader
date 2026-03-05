# Phase 9: Backtesting Engine - Research

**Researched:** 2026-03-04
**Domain:** Event-driven backtesting and quantitative finance simulation
**Confidence:** MEDIUM-HIGH

## Summary

Comprehensive backtesting for quantitative options trading requires an event-driven architecture that accurately simulates real-world trading conditions. The research identifies event-driven frameworks as the gold standard for realistic backtesting, with key components including data handlers, portfolio management, order execution simulation, and comprehensive performance measurement.

Event-driven backtesting prevents look-ahead bias and enables realistic modeling of order timing, slippage, and commissions. Walk-forward optimization provides robust strategy validation by continuously testing on new data, while Monte Carlo simulation adds statistical robustness by randomizing trade sequences to identify sequence risk.

**Primary recommendation:** Build a modular event-driven backtesting engine with integrated walk-forward optimization and Monte Carlo simulation capabilities using Python's quantitative finance ecosystem.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | >=2.0.0 | Time series and market data management | Industry standard for financial data manipulation |
| numpy | >=1.24.0 | Numerical computations and arrays | Foundation for all mathematical operations |
| yfinance | >=0.2.28 | Historical market data retrieval | Reliable, free market data source |
| matplotlib | >=3.7.0 | Performance visualization and charts | Standard plotting library for Python |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | >=1.10.0 | Statistical functions and optimization | For advanced statistical analysis and Monte Carlo |
| scikit-learn | >=1.3.0 | Performance metrics and ML utilities | For advanced performance analytics |
| pyfolio | Latest | Portfolio performance analysis | Professional-grade tearsheet generation |
| backtrader | >=1.9.76 | Alternative framework reference | If considering pre-built solutions |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom event loop | Backtrader | Custom gives full control but more development time |
| Basic metrics | PyFolio | Custom metrics are simpler but less comprehensive |
| yfinance | Bloomberg API | yfinance is free but has data limitations |

**Installation:**
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 yfinance>=0.2.28 matplotlib>=3.7.0 scipy>=1.10.0 scikit-learn>=1.3.0
pip install pyfolio  # For advanced performance analytics
```

## Architecture Patterns

### Recommended Project Structure
```
src/backtesting/
├── core/              # Core event-driven framework
├── data_handlers/     # Market data management
├── portfolio/         # Position and risk management
├── execution/         # Order execution and fills
├── performance/       # Metrics and reporting
├── monte_carlo/       # Monte Carlo simulation
└── walk_forward/      # Walk-forward optimization
```

### Pattern 1: Event-Driven Architecture
**What:** Central event queue that processes MarketEvent, SignalEvent, OrderEvent, and FillEvent objects in sequence
**When to use:** All backtesting scenarios - prevents look-ahead bias and enables realistic simulation
**Example:**
```python
# Source: QuantStart event-driven pattern
import queue
from abc import ABC, abstractmethod
from enum import Enum

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"

class Event(ABC):
    @property
    @abstractmethod
    def type(self):
        pass

class MarketEvent(Event):
    def __init__(self, timestamp, symbol, data):
        self.type = EventType.MARKET
        self.timestamp = timestamp
        self.symbol = symbol
        self.data = data

class BacktestEngine:
    def __init__(self):
        self.events = queue.Queue()

    def run_backtest(self):
        while True:
            try:
                event = self.events.get(False)
                if event.type == EventType.MARKET:
                    self.handle_market_event(event)
                elif event.type == EventType.SIGNAL:
                    self.handle_signal_event(event)
                # ... handle other events
            except queue.Empty:
                break
```

### Pattern 2: Portfolio State Management
**What:** Continuous marking-to-market and position tracking with time series equity curve generation
**When to use:** Essential for all backtesting - tracks positions, cash, and total equity over time
**Example:**
```python
# Source: Event-driven portfolio management pattern
class Portfolio:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_positions = {}  # {symbol: quantity}
        self.current_holdings = {
            'cash': initial_capital,
            'commission': 0.0,
            'total': initial_capital
        }
        self.all_holdings = []  # Time series

    def update_market_value(self, market_event):
        """Mark positions to market on every tick"""
        market_value = 0.0
        for symbol, quantity in self.current_positions.items():
            if quantity > 0:
                price = self.get_latest_price(symbol)
                market_value += quantity * price
                self.current_holdings[symbol] = quantity * price

        self.current_holdings['total'] = (
            self.current_holdings['cash'] + market_value
        )
        self.all_holdings.append(dict(self.current_holdings))
```

### Pattern 3: Order Execution Simulation
**What:** Realistic simulation of order fills with slippage, commissions, and market impact
**When to use:** Critical for accurate backtesting - models real trading costs and delays
**Example:**
```python
# Source: Execution handler with slippage modeling
class ExecutionHandler:
    def __init__(self, commission_rate=0.001, slippage_rate=0.0005):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def execute_order(self, order_event):
        # Simulate slippage
        if order_event.direction == 'BUY':
            fill_price = order_event.price * (1 + self.slippage_rate)
        else:
            fill_price = order_event.price * (1 - self.slippage_rate)

        # Calculate commission
        commission = abs(order_event.quantity) * fill_price * self.commission_rate

        return FillEvent(
            timestamp=order_event.timestamp,
            symbol=order_event.symbol,
            exchange='SIMULATED',
            quantity=order_event.quantity,
            direction=order_event.direction,
            fill_cost=fill_price,
            commission=commission
        )
```

### Anti-Patterns to Avoid
- **Vectorized backtesting without event queue:** Creates look-ahead bias and unrealistic order timing
- **Perfect order fills:** Always using exact bid/ask prices without slippage leads to overly optimistic results
- **Static commission models:** Using fixed dollar commissions instead of percentage-based for options

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Performance metrics calculation | Custom Sharpe/Sortino calculators | PyFolio or proven formulas | Edge cases in annualization, risk-free rates, and statistical robustness |
| Market data normalization | Custom data cleaning | yfinance + pandas | Handles splits, dividends, and data quality issues automatically |
| Monte Carlo simulation engine | Basic random sampling | SciPy + established statistical methods | Proper random number generation, seeding, and statistical validity |
| Walk-forward optimization logic | Custom rolling window validation | Established time series cross-validation patterns | Complex edge cases around data leakage and validation windows |

**Key insight:** Financial data has numerous edge cases (splits, dividends, holidays, data gaps) that mature libraries handle correctly, while custom solutions often miss critical details that affect backtest accuracy.

## Common Pitfalls

### Pitfall 1: Look-Ahead Bias
**What goes wrong:** Using future information in trading decisions, leading to unrealistically optimistic results
**Why it happens:** Vectorized operations and improper data handling can access future data points
**How to avoid:** Use strict event-driven architecture where each decision only has access to historical data up to that point
**Warning signs:** Backtest results that seem too good to be true, perfect market timing in historical data

### Pitfall 2: Survivorship Bias
**What goes wrong:** Only testing on stocks/options that remained liquid throughout the entire period
**Why it happens:** Using current universe of tradeable symbols to backtest historical periods
**How to avoid:** Use point-in-time universes and account for delisted/illiquid securities
**Warning signs:** All tested symbols show continuous trading history with no gaps

### Pitfall 3: Unrealistic Execution Assumptions
**What goes wrong:** Perfect fills at bid/ask prices with no slippage or market impact
**Why it happens:** Simplified execution models that don't account for real trading costs
**How to avoid:** Model realistic slippage (0.05-0.1% for liquid options), proper commission structures, and partial fills
**Warning signs:** Backtest performance deteriorates significantly in live trading

### Pitfall 4: Over-Optimization (Curve Fitting)
**What goes wrong:** Strategy parameters optimized to historical data don't perform on new data
**Why it happens:** Excessive parameter tuning without proper out-of-sample validation
**How to avoid:** Use walk-forward optimization with strict train/validation/test splits
**Warning signs:** Many parameters, complex rules, performance drops sharply on recent data

### Pitfall 5: Insufficient Position Sizing Risk Management
**What goes wrong:** Portfolio blowup from position sizes that seemed reasonable in backtest
**Why it happens:** Not modeling extreme scenarios or portfolio heat
**How to avoid:** Implement Kelly criterion sizing with maximum position limits and Monte Carlo stress testing
**Warning signs:** Large position sizes relative to portfolio, lack of maximum drawdown controls

## Code Examples

Verified patterns from research:

### Walk-Forward Optimization Implementation
```python
# Source: Walk-forward optimization pattern
import pandas as pd
from datetime import datetime, timedelta

class WalkForwardOptimizer:
    def __init__(self, strategy_class, train_window=252, test_window=63):
        self.strategy_class = strategy_class
        self.train_window = train_window
        self.test_window = test_window

    def optimize(self, data, param_grid):
        """Run walk-forward optimization"""
        results = []

        for start_idx in range(self.train_window, len(data), self.test_window):
            # Define training and testing periods
            train_end = start_idx
            train_start = start_idx - self.train_window
            test_start = start_idx
            test_end = min(start_idx + self.test_window, len(data))

            # Get training data
            train_data = data.iloc[train_start:train_end]

            # Optimize parameters on training data
            best_params = self._optimize_params(train_data, param_grid)

            # Test on out-of-sample data
            test_data = data.iloc[test_start:test_end]
            test_result = self._backtest_period(test_data, best_params)

            results.append({
                'period_start': data.index[test_start],
                'period_end': data.index[test_end-1],
                'params': best_params,
                'return': test_result['total_return'],
                'sharpe': test_result['sharpe_ratio'],
                'max_dd': test_result['max_drawdown']
            })

        return pd.DataFrame(results)
```

### Monte Carlo Simulation for Trade Sequence Risk
```python
# Source: Monte Carlo simulation for backtesting robustness
import numpy as np
from scipy import stats

class MonteCarloSimulator:
    def __init__(self, n_simulations=1000, random_seed=42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def simulate_trade_sequences(self, trade_returns):
        """Bootstrap resample trade returns to test sequence risk"""
        np.random.seed(self.random_seed)

        results = []
        n_trades = len(trade_returns)

        for _ in range(self.n_simulations):
            # Bootstrap resample trades
            resampled_returns = np.random.choice(
                trade_returns,
                size=n_trades,
                replace=True
            )

            # Calculate cumulative performance
            cumulative = np.cumprod(1 + resampled_returns)
            final_return = cumulative[-1] - 1
            max_drawdown = self._calculate_max_drawdown(cumulative)

            results.append({
                'final_return': final_return,
                'max_drawdown': max_drawdown
            })

        return pd.DataFrame(results)

    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown from cumulative returns"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Vectorized backtesting | Event-driven simulation | 2015-2020 | Eliminates look-ahead bias, enables realistic order modeling |
| Single backtest run | Monte Carlo + Walk-forward | 2020-2025 | Better risk assessment and out-of-sample validation |
| Simple commission models | Realistic execution costs | 2018-2025 | More accurate P&L expectations |
| Basic Sharpe ratio reporting | Comprehensive tearsheets | 2019-2025 | Better understanding of strategy characteristics |

**Deprecated/outdated:**
- Simple vectorized backtesting without event simulation
- Fixed dollar commission models for options
- Single-period optimization without walk-forward validation

## Open Questions

1. **Options-specific execution modeling**
   - What we know: General slippage models exist for equities
   - What's unclear: Best practices for options bid-ask spread simulation and early exercise modeling
   - Recommendation: Research options market microstructure and implement conservative spread assumptions

2. **Integration with existing RL training system**
   - What we know: Project has PPO training infrastructure in place
   - What's unclear: How to efficiently backtest RL-generated signals vs traditional rule-based strategies
   - Recommendation: Design plugin architecture for both signal types

3. **Real-time vs historical data reconciliation**
   - What we know: yfinance provides historical data
   - What's unclear: How to ensure backtesting data quality matches live trading data
   - Recommendation: Implement data quality checks and validation against known market events

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=7.4.0 |
| Config file | none — see Wave 0 |
| Quick run command | `python -m pytest tests/backtesting/ -x --tb=short` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BACK-01 | Event-driven backtest execution | integration | `pytest tests/test_backtest_engine.py::test_event_processing -x` | ❌ Wave 0 |
| BACK-02 | Walk-forward optimization | unit | `pytest tests/test_walk_forward.py::test_optimization_windows -x` | ❌ Wave 0 |
| BACK-03 | Monte Carlo simulation | unit | `pytest tests/test_monte_carlo.py::test_trade_resampling -x` | ❌ Wave 0 |
| BACK-04 | Performance metrics calculation | unit | `pytest tests/test_performance.py::test_sharpe_calculation -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/backtesting/ -x --tb=short`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backtesting/` — test directory structure
- [ ] `tests/backtesting/test_backtest_engine.py` — covers event-driven execution
- [ ] `tests/backtesting/test_walk_forward.py` — covers optimization framework
- [ ] `tests/backtesting/test_monte_carlo.py` — covers simulation robustness
- [ ] `tests/backtesting/test_performance.py` — covers metrics calculation
- [ ] `tests/conftest.py` — shared fixtures for market data and test portfolios

## Sources

### Primary (HIGH confidence)
- Event-driven backtesting architecture patterns from QuantStart methodology
- Python backtesting ecosystem survey from multiple 2024-2025 sources

### Secondary (MEDIUM confidence)
- Walk-forward optimization best practices from academic research (2024)
- Monte Carlo simulation techniques from quantitative trading literature
- Performance metrics standards from PyFolio documentation

### Tertiary (LOW confidence)
- Options-specific execution modeling (needs validation with options market makers)
- Integration patterns with RL systems (emerging area, limited established practices)

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM-HIGH - Well-established Python ecosystem for quantitative finance
- Architecture: HIGH - Event-driven pattern is industry standard with proven track record
- Pitfalls: HIGH - Well-documented common mistakes in backtesting literature

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (30 days - quantitative finance best practices are relatively stable)