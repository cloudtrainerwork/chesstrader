# Options AI Trading System
## Technical Design Document

**Version 1.0 | January 2026**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Data Architecture](#3-data-architecture)
4. [Neural Network Architecture](#4-neural-network-architecture)
5. [Reinforcement Learning Environment](#5-reinforcement-learning-environment)
6. [Training Pipeline](#6-training-pipeline)
7. [Backtesting Framework](#7-backtesting-framework)
8. [API Reference](#8-api-reference)
9. [Deployment](#9-deployment)
10. [Appendix](#appendix-glossary)

---

## 1. Introduction

### 1.1 Purpose

This Technical Design Document describes the architecture, components, algorithms, and implementation details of the Options AI Trading System. The system applies game-theoretic frameworks and reinforcement learning to options trading.

### 1.2 Core Innovation: The Game-Theoretic Framework

The fundamental innovation is modeling options strategies as bounded games, enabling transfer learning from chess-trained architectures:

| Chess Concept | Options Equivalent |
|---------------|-------------------|
| Board boundaries | Strike prices (Iron Condor wings) |
| Piece positions | Price relative to strikes |
| Legal moves | Valid adjustments (roll, close, widen) |
| Checkmate (lose) | Price breach / max loss |
| Stalemate (draw) | Scratch trade for small P/L |
| Time control | Theta decay / DTE |
| Position evaluation | Expected P/L, probability of profit |

### 1.3 Design Principles

1. **Modularity:** Each component can be developed, tested, and replaced independently
2. **Extensibility:** New strategies and regimes can be added without architectural changes
3. **Reproducibility:** All experiments are logged and reproducible
4. **Efficiency:** Optimized for both training throughput and inference latency

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERFACE LAYER                          │
│                  OptionsAI API │ CLI │ Visualization            │
├─────────────────────────────────────────────────────────────────┤
│                       EVALUATION LAYER                          │
│        Backtest Engine │ Walk-Forward │ Monte Carlo             │
├─────────────────────────────────────────────────────────────────┤
│                        TRAINING LAYER                           │
│         RL Environment │ PPO Trainer │ Curriculum               │
├─────────────────────────────────────────────────────────────────┤
│                         MODEL LAYER                             │
│      Regime Detector │ Strategy Selector │ Actor-Critic         │
├─────────────────────────────────────────────────────────────────┤
│                          DATA LAYER                             │
│           Data Pipeline │ Feature Engineer │ Storage            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| Data Pipeline | Fetch and process market data | `DataPipeline`, `YFinanceProvider` |
| Feature Engineer | Transform raw data to state vectors | `FeatureEngineer`, `RegimeState`, `PositionState` |
| Regime Detector | Classify market conditions | `RegimeDetector` (8 regimes) |
| Strategy Selector | Rank and recommend strategies | `StrategySelector` (16 strategies) |
| Position Manager | RL agent for position decisions | `ActorCritic`, `PolicyNetwork` |
| Backtest Engine | Historical simulation | `BacktestEngine`, `WalkForwardOptimizer` |

### 2.3 Data Flow

```
Market Data → Data Pipeline → Feature Engineer → State Vectors
                                                      ↓
                                              Regime Detector
                                                      ↓
                                              Strategy Selector
                                                      ↓
                                              Position Manager
                                                      ↓
                                              Action Recommendation
```

---

## 3. Data Architecture

### 3.1 Data Sources

| Data Type | Source | Update Frequency | Usage |
|-----------|--------|------------------|-------|
| Price Data | Yahoo Finance, Polygon | Real-time / EOD | OHLCV, technical indicators |
| Options Data | Tradier, Polygon | Real-time | Chains, Greeks, IV |
| VIX Data | Yahoo Finance | Real-time | Market volatility gauge |
| Events | Earnings Whispers | Daily | Earnings dates |

### 3.2 Regime State Vector (48 dimensions)

The regime state captures overall market conditions for classification:

| Category | Dimensions | Features |
|----------|------------|----------|
| Price Structure | 6 | Price vs SMA20/50/200, 52-week high/low distance, gap |
| Trend Indicators | 9 | ADX, +DI/-DI, MACD line/signal/histogram, EMA alignment, HH/LL |
| Volatility | 11 | IV, IV rank, IV percentile, HV20, VIX, VIX percentile, term structure, skew, BB width/position, ATR |
| Momentum | 6 | RSI, Stochastic K/D, ROC 5/10/20 |
| Volume | 3 | Volume ratio, OBV slope, volume trend |
| Support/Resistance | 6 | Nearest support/resistance, strength, distance, consolidation, range width |
| Events | 3 | Days to earnings, days to FOMC, days to OpEx |
| Market Context | 4 | SPY correlation, sector relative strength, market breadth, put/call ratio |

### 3.3 Position State Vector (24 dimensions)

The position state captures current position status for management decisions:

| Category | Dimensions | Features |
|----------|------------|----------|
| Strategy Identity | 1 | Strategy type (one-hot encoded separately) |
| Board Position | 6 | Price zone (-3 to +3), zone velocity, distance to upper/lower strikes, distance to breakevens |
| Time | 3 | DTE, percent time remaining, theta per day |
| Volatility | 3 | IV at entry, current IV, IV change |
| Greeks | 4 | Position delta, gamma, theta, vega |
| P/L Status | 5 | Entry credit, current value, unrealized P/L, % of max profit, % of max loss |
| Meta | 2 | Days held, adjustments made |

### 3.4 Database Schema

```sql
-- Price data cache
CREATE TABLE price_history (
    symbol VARCHAR(10),
    date DATE,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);

-- Regime classifications
CREATE TABLE regime_history (
    symbol VARCHAR(10),
    date DATE,
    regime VARCHAR(20),
    confidence DECIMAL(5,4),
    state_vector BLOB,
    PRIMARY KEY (symbol, date)
);

-- Trade history
CREATE TABLE trades (
    trade_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    strategy VARCHAR(30),
    entry_date DATE,
    exit_date DATE,
    entry_price DECIMAL(10,2),
    exit_price DECIMAL(10,2),
    pnl DECIMAL(10,2),
    adjustments INT
);
```

---

## 4. Neural Network Architecture

### 4.1 Spatial Encoder (Chess-Inspired)

The core architectural innovation treats market state as a spatial "board":

```python
class SpatialEncoder(nn.Module):
    """
    Treats market state as a 7x6x8 spatial grid:
    - 7 rows: Price zones from -3 to +3 (relative to strikes)
    - 6 columns: Time buckets (DTE: 45, 30, 21, 14, 7, 0)
    - 8 channels: Indicator categories ("piece types")
    """
    def __init__(self, input_size=48, hidden_size=256):
        self.board_h, self.board_w, self.channels = 7, 6, 8
        
        # Project input to board representation
        self.input_proj = nn.Linear(input_size, self.board_h * self.board_w * self.channels)
        
        # Convolutional layers (pattern extraction)
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Residual blocks (AlphaZero-inspired)
        self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(4)])
        
        # Output projection
        self.output_proj = nn.Linear(128 * self.board_h * self.board_w, hidden_size)
```

### 4.2 Regime Detector

Classifies market into 8 regimes using supervised learning:

| Regime | Characteristics | Best Strategies |
|--------|-----------------|-----------------|
| LOW_VOL_RANGE | IV rank < 30, weak trend | Iron Condor, Iron Butterfly |
| HIGH_VOL_RANGE | IV rank > 50, weak trend | Wide Iron Condor, Calendar |
| TRENDING_UP | ADX > 25, bullish alignment | Bull Call/Put Spread |
| TRENDING_DOWN | ADX > 25, bearish alignment | Bear Put/Call Spread |
| BREAKOUT_IMMINENT | BB compression, low IV | Long Straddle/Strangle |
| CRASH_SPIKE | VIX spike, high fear | Protective Put, Collar |
| IV_CRUSH_SETUP | Pre-earnings elevated IV | Short premium strategies |
| MEAN_REVERSION | Extreme RSI, stretched | Contrarian plays |

```python
class RegimeDetector(nn.Module):
    def __init__(self, input_size=48, hidden_size=256, num_regimes=8):
        self.encoder = SpatialEncoder(input_size, hidden_size)
        self.attention = TemporalAttention(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_regimes)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        attended = self.attention(encoded)
        logits = self.classifier(attended)
        return logits, F.softmax(logits, dim=-1)
```

### 4.3 Strategy Selector

Ranks 16 strategies given regime and market state:

| Strategy Type | Risk Profile | Ideal Conditions |
|---------------|--------------|------------------|
| Iron Condor | Defined risk, neutral | Low vol, range-bound |
| Iron Butterfly | Defined risk, neutral | Very low vol, tight range |
| Bull Call Spread | Defined risk, bullish | Uptrend, moderate vol |
| Bull Put Spread | Defined risk, bullish | Uptrend, elevated IV |
| Bear Put Spread | Defined risk, bearish | Downtrend, moderate vol |
| Bear Call Spread | Defined risk, bearish | Downtrend, elevated IV |
| Long Straddle | Unlimited profit, neutral | Pre-breakout, low IV |
| Long Strangle | Unlimited profit, neutral | Pre-breakout, very low IV |
| Short Straddle | Undefined risk, neutral | High IV, range-bound |
| Short Strangle | Undefined risk, neutral | High IV, wide range |
| Calendar Spread | Defined risk, neutral | Term structure play |
| Diagonal Spread | Defined risk, directional | Slow trend expected |
| Covered Call | Stock + short call | Mild bullish, income |
| Cash-Secured Put | Cash + short put | Bullish, want to own |
| Protective Put | Stock + long put | Hedging existing position |
| Collar | Stock + put + short call | Hedging, cap upside |

### 4.4 Actor-Critic Network (PPO)

Position management uses Proximal Policy Optimization:

```python
class ActorCritic(nn.Module):
    def __init__(self, position_state_size=24, market_state_size=48, 
                 hidden_size=256, num_actions=15):
        # Shared encoder
        self.position_encoder = nn.Linear(position_state_size, hidden_size // 2)
        self.market_encoder = SpatialEncoder(market_state_size, hidden_size // 2)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, position_state, market_state):
        pos_enc = F.relu(self.position_encoder(position_state))
        market_enc = self.market_encoder(market_state)
        combined = torch.cat([pos_enc, market_enc], dim=-1)
        
        action_logits = self.actor(combined)
        value = self.critic(combined)
        
        return action_logits, value
```

---

## 5. Reinforcement Learning Environment

### 5.1 Environment Design

The RL environment models options trading as a Gym-compatible bounded game:

```python
class OptionsEnvironment:
    """
    Gym-compatible environment where:
    - The "board" is price relative to strike boundaries
    - The "clock" is days to expiration
    - "Winning" is capturing theta without breach
    - "Losing" is price breaching boundaries
    """
    
    observation_space = spaces.Dict({
        'position_state': spaces.Box(low=-inf, high=inf, shape=(24,)),
        'market_state': spaces.Box(low=-inf, high=inf, shape=(48,))
    })
    
    action_space = spaces.Discrete(15)
```

### 5.2 Action Space

| ID | Action | Description |
|----|--------|-------------|
| 0 | HOLD | Do nothing, let position run |
| 1 | CLOSE_ALL | Exit entire position |
| 2 | CLOSE_PARTIAL_25 | Close 25% of position |
| 3 | CLOSE_PARTIAL_50 | Close 50% of position |
| 4 | CLOSE_CALL_SIDE | Close call side only (Iron Condor) |
| 5 | CLOSE_PUT_SIDE | Close put side only (Iron Condor) |
| 6 | ROLL_UP | Move strikes higher |
| 7 | ROLL_DOWN | Move strikes lower |
| 8 | ROLL_OUT | Extend expiration |
| 9 | WIDEN_SPREAD | Increase spread width |
| 10 | NARROW_SPREAD | Decrease spread width |
| 11 | ADD_PROTECTION | Buy protective option |
| 12 | TAKE_PROFIT_50 | Close at 50% of max profit |
| 13 | TAKE_PROFIT_75 | Close at 75% of max profit |
| 14 | INVERT_POSITION | Flip from bullish to bearish or vice versa |

### 5.3 Reward Function

Strategy-specific reward shaping:

**Iron Condor (Range-Bound):**
```python
def iron_condor_reward(state):
    zone = state.price_zone  # -3 to +3
    
    if abs(zone) < 0.5:
        return +2.0  # Safe zone bonus
    elif abs(zone) < 0.8:
        return +0.5  # Warning zone
    elif abs(zone) < 1.0:
        return -2.0  # Danger zone
    else:
        return -10.0  # Breach penalty
```

**Directional (Bull/Bear Spread):**
```python
def directional_reward(state, direction='bull'):
    price_move = state.price_change_pct
    
    if direction == 'bull':
        return price_move * 50  # Reward upward movement
    else:
        return -price_move * 50  # Reward downward movement
```

**Breakout (Long Straddle):**
```python
def breakout_reward(state):
    move = abs(state.price_change_pct)
    theta = state.position_theta
    
    return move * 100 + theta * 0.5  # Reward moves, penalize theta
```

**Universal Components:**
```python
def universal_reward(state, action_cost):
    pnl_change = state.current_pnl - state.previous_pnl
    
    return (
        pnl_change / 100  # P/L change (scaled)
        - action_cost / 50  # Transaction cost penalty
    )
```

### 5.4 Terminal Conditions

An episode ends when:

1. **Position Closed:** Agent executes CLOSE_ALL action
2. **Expiration:** DTE reaches 0
3. **Max Loss:** Unrealized loss exceeds max loss threshold
4. **End of Data:** Historical data exhausted

---

## 6. Training Pipeline

### 6.1 Curriculum Learning

Training follows a curriculum that progressively increases complexity:

| Phase | Strategies | Timesteps | Learning Goal |
|-------|------------|-----------|---------------|
| 1. Spreads | Bull/Bear Verticals | 50,000 | Learn spread dynamics |
| 2. Neutral | Iron Condor | 100,000 | Learn range management |
| 3. Volatility | Long Straddle | 50,000 | Learn vol plays |
| 4. Full | All 16 strategies | 100,000 | Strategy rotation |

```python
class CurriculumTrainer:
    phases = [
        {'strategies': ['bull_call_spread'], 'timesteps': 50000},
        {'strategies': ['iron_condor'], 'timesteps': 100000},
        {'strategies': ['long_straddle'], 'timesteps': 50000},
        {'strategies': ALL_STRATEGIES, 'timesteps': 100000}
    ]
    
    def train_curriculum(self):
        for phase in self.phases:
            env = MultiStrategyEnvironment(strategies=phase['strategies'])
            trainer = PPOTrainer(self.model, env)
            trainer.train(total_timesteps=phase['timesteps'])
```

### 6.2 PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 3e-4 | Adam optimizer LR |
| Discount (γ) | 0.99 | Future reward discount |
| GAE Lambda (λ) | 0.95 | Advantage estimation |
| Clip Epsilon | 0.2 | PPO clipping range |
| Value Loss Coef | 0.5 | Value function weight |
| Entropy Coef | 0.01 | Exploration bonus |
| Batch Size | 64 | Mini-batch size |
| Epochs per Update | 10 | Passes over buffer |
| Buffer Size | 2048 | Rollout buffer size |
| Max Grad Norm | 0.5 | Gradient clipping |

### 6.3 Transfer Learning from Chess

The SpatialEncoder can accept pre-trained weights from chess models:

```python
def load_pretrained_chess_weights(model, chess_checkpoint):
    """
    Transfer convolutional weights from chess model.
    Chess: 8x8 board, 12 piece types
    Options: 7x6 board, 8 indicator channels
    """
    chess_state = torch.load(chess_checkpoint)
    
    # Average across input channels to handle dimension mismatch
    conv1_weight = chess_state['conv1.weight']
    conv1_weight = conv1_weight[:, :8, :, :]  # Take first 8 channels
    
    model.encoder.conv1.weight.data = conv1_weight
    # ... similar for other conv layers
```

---

## 7. Backtesting Framework

### 7.1 Engine Design

Event-driven backtesting with realistic simulation assumptions:

```python
class BacktestEngine:
    def __init__(self,
                 initial_capital=100000,
                 commission_per_contract=0.65,
                 slippage_pct=0.001,
                 margin_requirement=0.20):
        self.capital = initial_capital
        self.commission = commission_per_contract
        self.slippage = slippage_pct
        self.margin_req = margin_requirement
```

### 7.2 Walk-Forward Optimization

Prevents overfitting by training on rolling windows:

```
|-------- Training (2 years) --------|----- Test (3 months) -----|
                                     |-------- Training ---------|-- Test --|
                                                                 |-- Train --|...
```

| Parameter | Value |
|-----------|-------|
| Training Window | 504 days (2 years) |
| Testing Window | 63 days (3 months) |
| Step Size | 21 days (monthly) |

### 7.3 Monte Carlo Simulation

Robustness testing through bootstrap sampling:

```python
class MonteCarloSimulator:
    def simulate_returns(self, daily_returns, num_days=252, num_sims=1000):
        simulations = np.zeros((num_sims, num_days))
        
        for i in range(num_sims):
            sampled = np.random.choice(daily_returns, size=num_days, replace=True)
            simulations[i] = np.cumprod(1 + sampled)
        
        return simulations
    
    def calculate_var(self, daily_returns, confidence=0.95, horizon=21):
        sims = self.simulate_returns(daily_returns, horizon)
        final_returns = sims[:, -1] - 1
        var = np.percentile(final_returns, (1 - confidence) * 100)
        return var
```

### 7.4 Performance Metrics

| Metric | Formula |
|--------|---------|
| Total Return | (Final Equity - Initial) / Initial |
| Annual Return | (1 + Total Return)^(1/years) - 1 |
| Sharpe Ratio | √252 × mean(daily returns) / std(daily returns) |
| Sortino Ratio | √252 × mean(daily returns) / std(negative returns) |
| Max Drawdown | min((equity - running max) / running max) |
| Win Rate | Winning trades / Total trades |
| Profit Factor | Gross profits / Gross losses |
| Average Trade | Total P/L / Number of trades |
| Average Winner | Mean P/L of winning trades |
| Average Loser | Mean P/L of losing trades |

---

## 8. API Reference

### 8.1 Main Interface

```python
from options_ai_system import OptionsAI

# Initialize
ai = OptionsAI(underlying='QQQ', device='cpu')

# Analyze market and get recommendation
analysis = ai.analyze()
print(f"Regime: {analysis['regime']}")
print(f"Strategy: {analysis['recommended_strategy']}")
ai.print_recommendation(analysis)

# Get position management action
action = ai.get_position_action(position_state, market_state)
print(f"Action: {action['action']}")

# Backtest a strategy
results = ai.backtest(strategy='iron_condor', days=252)
print(f"Sharpe: {results.sharpe_ratio}")
print(f"Win Rate: {results.win_rate}")

# Train the model
metrics = ai.train(total_timesteps=100000, use_curriculum=True)

# Save/Load model
ai.save_model('./checkpoints/model.pt')
ai.load_model('./checkpoints/model.pt')
```

### 8.2 Analysis Response Format

```python
{
    'symbol': 'QQQ',
    'timestamp': '2026-01-05T12:00:00',
    'price': 450.25,
    'regime': 'LOW_VOL_RANGE',
    'regime_confidence': 0.82,
    'regime_probabilities': {
        'LOW_VOL_RANGE': 0.82,
        'HIGH_VOL_RANGE': 0.08,
        'TRENDING_UP': 0.05,
        ...
    },
    'recommended_strategy': 'iron_condor',
    'top_strategies': [
        {'strategy': 'iron_condor', 'score': 0.85},
        {'strategy': 'iron_butterfly', 'score': 0.72},
        {'strategy': 'short_strangle', 'score': 0.68}
    ],
    'market_state': {
        'iv_rank': 25.5,
        'rsi': 52.3,
        'adx': 18.2,
        'bb_position': 0.45,
        'price_vs_sma_50': 1.2
    }
}
```

### 8.3 Command Line Interface

```bash
# Analyze a symbol
python -m options_ai_system analyze --symbol QQQ

# Backtest a strategy
python -m options_ai_system backtest --symbol SPY --strategy iron_condor --days 252

# Train the model
python -m options_ai_system train --timesteps 100000 --strategy iron_condor
```

---

## 9. Deployment

### 9.1 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0 | Neural networks, training |
| numpy | >=1.24 | Numerical operations |
| pandas | >=2.0 | Data manipulation |
| yfinance | >=0.2 | Market data fetching |
| scipy | >=1.10 | Statistical functions |
| matplotlib | >=3.7 | Visualization |

### 9.2 Installation

```bash
# Clone repository
git clone https://github.com/your-org/options-ai-system.git
cd options-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 9.3 Hardware Requirements

| Use Case | CPU | RAM | GPU | Storage |
|----------|-----|-----|-----|---------|
| Training | 8+ cores | 16GB+ | CUDA-compatible recommended | 50GB |
| Inference | 4+ cores | 8GB | Not required | 10GB |

### 9.4 Directory Structure

```
options_ai_system/
├── config/
│   └── settings.py          # Configuration
├── data/
│   ├── pipeline.py          # Data fetching
│   └── features.py          # Feature engineering
├── models/
│   └── networks.py          # Neural networks
├── training/
│   ├── environment.py       # RL environment
│   └── trainer.py           # Training loops
├── backtesting/
│   └── engine.py            # Backtest engine
├── docs/
│   ├── BRD.md               # Business requirements
│   └── Technical.md         # This document
├── main.py                  # Main interface
├── requirements.txt         # Dependencies
└── README.md                # Quick start guide
```

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| DTE | Days to Expiration - time remaining until option expires |
| IV | Implied Volatility - market's expectation of future volatility |
| HV | Historical Volatility - realized volatility from past prices |
| IV Rank | Current IV relative to past year's range (0-100) |
| Greeks | Option sensitivities: Delta (Δ), Gamma (Γ), Theta (Θ), Vega (ν) |
| PPO | Proximal Policy Optimization - RL algorithm |
| GAE | Generalized Advantage Estimation - variance reduction technique |
| ATM | At-the-Money - strike price equals underlying price |
| OTM | Out-of-the-Money - option has no intrinsic value |
| ITM | In-the-Money - option has intrinsic value |
| Spread | Options position with multiple legs |
| Iron Condor | Sell OTM call spread + OTM put spread (range-bound) |
| Straddle | Buy ATM call + ATM put (volatility play) |
| Vertical | Buy and sell same type at different strikes |
| Calendar | Same strike, different expirations |

---

*Document Version: 1.0*
*Last Updated: January 2026*