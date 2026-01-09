"""
Regime state feature engineering for market analysis.

Implements comprehensive feature extraction for 48-dimensional market state vectors
used in regime detection and position sizing algorithms.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from .base import FeatureEngineering, calculate_sma, calculate_ema, calculate_rolling_high, calculate_rolling_low


class PriceStructureFeatures(FeatureEngineering):
    """
    Price structure features for regime detection (6 dimensions).

    Features:
    1. price_vs_sma20: (price / SMA20) - 1
    2. price_vs_sma50: (price / SMA50) - 1
    3. price_vs_sma200: (price / SMA200) - 1
    4. distance_from_52w_high: (price - 52w_high) / 52w_high
    5. distance_from_52w_low: (price - 52w_low) / 52w_low
    6. gap_percentage: (open - prev_close) / prev_close
    """

    def calculate(self, symbol: str) -> pd.DataFrame:
        """
        Calculate price structure features for a symbol.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            DataFrame with 6 price structure features
        """
        # Get price data
        df = self.get_data(symbol, days=300)  # Extra data for 52-week calculations

        # Ensure we have required columns
        df = self.handle_missing_data(df)

        features = pd.DataFrame(index=df.index)

        # 1. Price vs SMA ratios
        sma20 = calculate_sma(df['Close'], 20)
        sma50 = calculate_sma(df['Close'], 50)
        sma200 = calculate_sma(df['Close'], 200)

        features['price_vs_sma20'] = (df['Close'] / sma20) - 1
        features['price_vs_sma50'] = (df['Close'] / sma50) - 1
        features['price_vs_sma200'] = (df['Close'] / sma200) - 1

        # 2. 52-week high/low distances
        high_52w = calculate_rolling_high(df['High'], 252)
        low_52w = calculate_rolling_low(df['Low'], 252)

        features['distance_from_52w_high'] = (df['Close'] - high_52w) / high_52w
        features['distance_from_52w_low'] = (df['Close'] - low_52w) / low_52w

        # 3. Gap percentage (today's open vs yesterday's close)
        prev_close = df['Close'].shift(1)
        features['gap_percentage'] = (df['Open'] - prev_close) / prev_close

        # Handle missing data and standardize
        features = self.handle_missing_data(features)

        # Standardize features to [-1, 1] range
        for col in features.columns:
            features[col] = self.standardize(features[col], method='robust')

        # Validate
        self.validate(features)

        return features


class TrendIndicators(FeatureEngineering):
    """
    Trend indicators using manual implementations (9 dimensions).

    Since pandas-ta is not available, we implement key indicators manually:
    1. ADX (14-period) - Average Directional Index
    2. +DI (14-period) - Positive Directional Indicator
    3. -DI (14-period) - Negative Directional Indicator
    4. MACD line - 12/26 EMA difference
    5. MACD signal - 9-period EMA of MACD
    6. MACD histogram - MACD - signal
    7. EMA alignment score - EMA9 vs EMA21 vs EMA50 alignment
    8. higher_highs - Binary flag for higher highs pattern
    9. lower_lows - Binary flag for lower lows pattern
    """

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX, +DI, and -DI indicators.

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # Directional Movement calculation
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low

        # Only count positive movements
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

        # Smoothed averages using Wilder's method (alpha = 1/period)
        alpha = 1.0 / period

        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        dm_plus_smooth = dm_plus.ewm(alpha=alpha, adjust=False).mean()
        dm_minus_smooth = dm_minus.ewm(alpha=alpha, adjust=False).mean()

        # Directional Indicators
        di_plus = 100 * (dm_plus_smooth / atr)
        di_minus = 100 * (dm_minus_smooth / atr)

        # ADX calculation
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx, di_plus, di_minus

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26,
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD line, signal line, and histogram.

        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate(self, symbol: str) -> pd.DataFrame:
        """
        Calculate trend indicators for a symbol.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            DataFrame with 9 trend indicator features
        """
        df = self.get_data(symbol, days=300)
        df = self.handle_missing_data(df)

        features = pd.DataFrame(index=df.index)

        # ADX and directional indicators
        adx, di_plus, di_minus = self.calculate_adx(df['High'], df['Low'], df['Close'])
        features['adx'] = adx / 100  # Normalize to [0, 1]
        features['di_plus'] = di_plus / 100
        features['di_minus'] = di_minus / 100

        # MACD indicators (normalized by price for relative comparison)
        macd_line, macd_signal, macd_hist = self.calculate_macd(df['Close'])
        features['macd_line'] = macd_line / df['Close']
        features['macd_signal'] = macd_signal / df['Close']
        features['macd_histogram'] = macd_hist / df['Close']

        # EMA alignment score
        ema9 = calculate_ema(df['Close'], 9)
        ema21 = calculate_ema(df['Close'], 21)
        ema50 = calculate_ema(df['Close'], 50)

        # Score based on EMA ordering: +1 if bullish alignment, -1 if bearish
        bullish_alignment = (ema9 > ema21) & (ema21 > ema50)
        bearish_alignment = (ema9 < ema21) & (ema21 < ema50)
        features['ema_alignment'] = bullish_alignment.astype(int) - bearish_alignment.astype(int)

        # Higher highs and lower lows patterns (20-day lookback)
        rolling_high = df['High'].rolling(20).max()
        rolling_low = df['Low'].rolling(20).min()

        features['higher_highs'] = (df['High'] == rolling_high).astype(int)
        features['lower_lows'] = (df['Low'] == rolling_low).astype(int)

        # Handle missing data and standardize
        features = self.handle_missing_data(features)

        # Standardize most features except binary flags
        for col in ['adx', 'di_plus', 'di_minus', 'macd_line', 'macd_signal', 'macd_histogram']:
            features[col] = self.standardize(features[col], method='robust')

        self.validate(features)
        return features


class MomentumIndicators(FeatureEngineering):
    """
    Momentum indicators using manual implementations (6 dimensions).

    Features:
    1. RSI (14-period) - Relative Strength Index
    2. Stochastic %K (14-period)
    3. Stochastic %D (3-period SMA of %K)
    4. ROC_5 (5-day rate of change)
    5. ROC_10 (10-day rate of change)
    6. ROC_20 (20-day rate of change)
    """

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic %K and %D."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    def calculate(self, symbol: str) -> pd.DataFrame:
        """
        Calculate momentum indicators for a symbol.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            DataFrame with 6 momentum indicator features
        """
        df = self.get_data(symbol, days=300)
        df = self.handle_missing_data(df)

        features = pd.DataFrame(index=df.index)

        # RSI
        rsi = self.calculate_rsi(df['Close'])
        features['rsi'] = (rsi / 50) - 1  # Normalize to [-1, 1] with 50 as midpoint

        # Stochastic oscillators
        stoch_k, stoch_d = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
        features['stoch_k'] = (stoch_k / 50) - 1  # Normalize to [-1, 1]
        features['stoch_d'] = (stoch_d / 50) - 1

        # Rate of Change indicators
        features['roc_5'] = df['Close'].pct_change(periods=5)
        features['roc_10'] = df['Close'].pct_change(periods=10)
        features['roc_20'] = df['Close'].pct_change(periods=20)

        # Handle missing data
        features = self.handle_missing_data(features)

        # Standardize ROC features (RSI and Stochastic already normalized)
        for col in ['roc_5', 'roc_10', 'roc_20']:
            features[col] = self.standardize(features[col], method='robust')

        self.validate(features)
        return features


class RegimeStateVector(FeatureEngineering):
    """
    Complete 48-dimensional regime state vector assembler.

    Combines all feature categories:
    - Price structure (6 dimensions)
    - Trend indicators (9 dimensions)
    - Momentum indicators (6 dimensions)
    - Volatility and market context (27 dimensions) - placeholder for Task 3
    """

    def __init__(self):
        super().__init__()
        self.price_features = PriceStructureFeatures()
        self.trend_features = TrendIndicators()
        self.momentum_features = MomentumIndicators()

    def calculate(self, symbol: str) -> pd.Series:
        """
        Calculate complete 48-dimensional state vector for a symbol.

        Note: Currently returns 21 dimensions (Tasks 1 & 2 complete).
        Task 3 will add remaining 27 dimensions.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Series with latest state vector (currently 21 dimensions)
        """
        # Calculate all feature categories
        price_feat = self.price_features.calculate(symbol)
        trend_feat = self.trend_features.calculate(symbol)
        momentum_feat = self.momentum_features.calculate(symbol)

        # Combine features (take most recent values)
        latest_price = price_feat.iloc[-1]
        latest_trend = trend_feat.iloc[-1]
        latest_momentum = momentum_feat.iloc[-1]

        # Concatenate all features
        state_vector = pd.concat([latest_price, latest_trend, latest_momentum])

        # Validate final output
        self.validate(state_vector)

        return state_vector