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

        # Rate of Change indicators - handle insufficient data
        min_periods_roc = min(len(df) - 1, 5)  # Ensure we have at least some data

        # Only calculate ROC if we have enough data
        if len(df) >= 6:
            features['roc_5'] = df['Close'].pct_change(periods=5)
        else:
            features['roc_5'] = df['Close'].pct_change(periods=min_periods_roc)

        if len(df) >= 11:
            features['roc_10'] = df['Close'].pct_change(periods=10)
        else:
            features['roc_10'] = df['Close'].pct_change(periods=min_periods_roc)

        if len(df) >= 21:
            features['roc_20'] = df['Close'].pct_change(periods=20)
        else:
            features['roc_20'] = df['Close'].pct_change(periods=min_periods_roc)

        # Handle missing data
        features = self.handle_missing_data(features)

        # For remaining NaN values (like the first values of ROC), fill with zeros
        features = features.fillna(0)

        # Standardize ROC features (RSI and Stochastic already normalized)
        for col in ['roc_5', 'roc_10', 'roc_20']:
            features[col] = self.standardize(features[col], method='robust')

        self.validate(features)
        return features


class VolatilityFeatures(FeatureEngineering):
    """
    Volatility features for regime detection (11 dimensions).

    Note: Some features require external data sources and are stubbed for MVP.
    """

    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20,
                                 std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper = sma + (std * std_mult)
        lower = sma - (std * std_mult)
        return upper, lower, sma

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate(self, symbol: str) -> pd.DataFrame:
        """Calculate volatility features for a symbol."""
        df = self.get_data(symbol, days=300)
        df = self.handle_missing_data(df)

        features = pd.DataFrame(index=df.index)

        # 1. Historical volatility (20-day rolling)
        returns = df['Close'].pct_change()
        features['historical_volatility_20'] = returns.rolling(20).std() * np.sqrt(252)  # Annualized

        # 2-4. Implied volatility features (stubbed - would need options data)
        features['implied_volatility'] = 0.2  # Stub with average IV
        features['iv_rank'] = 0.5  # Stub with neutral rank
        features['iv_percentile'] = 0.5  # Stub with neutral percentile

        # 5-6. VIX features (stubbed - would need VIX data)
        features['vix_level'] = 20.0  # Stub with average VIX
        features['vix_percentile'] = 0.5  # Stub with neutral percentile

        # 7. Term structure slope (stubbed - would need options data)
        features['term_structure_slope'] = 0.0  # Stub with flat term structure

        # 8. Put/call skew (stubbed - would need options data)
        features['put_call_skew'] = 0.0  # Stub with neutral skew

        # 9-10. Bollinger Bands features
        bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(df['Close'])
        features['bollinger_band_width'] = (bb_upper - bb_lower) / bb_middle
        features['bollinger_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

        # 11. ATR normalized by price
        atr = self.calculate_atr(df['High'], df['Low'], df['Close'])
        features['atr_normalized'] = atr / df['Close']

        # Handle missing data and normalize
        features = self.handle_missing_data(features)
        features = features.fillna(0)

        for col in features.columns:
            features[col] = self.standardize(features[col], method='robust')

        self.validate(features)
        return features


class VolumeFeatures(FeatureEngineering):
    """Volume-based features for regime detection (3 dimensions)."""

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def calculate(self, symbol: str) -> pd.DataFrame:
        """Calculate volume features for a symbol."""
        df = self.get_data(symbol, days=300)
        df = self.handle_missing_data(df)

        features = pd.DataFrame(index=df.index)

        # 1. Volume ratio (current volume vs 20-day average)
        avg_volume = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / avg_volume

        # 2. OBV slope (10-day linear regression slope)
        obv = self.calculate_obv(df['Close'], df['Volume'])
        obv_slope = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
        features['obv_slope'] = obv_slope

        # 3. Volume trend (5-day EMA slope)
        volume_ema = df['Volume'].ewm(span=5).mean()
        volume_trend = volume_ema.rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
        )
        features['volume_trend'] = volume_trend

        # Handle missing data and normalize
        features = self.handle_missing_data(features)
        features = features.fillna(0)

        for col in features.columns:
            features[col] = self.standardize(features[col], method='robust')

        self.validate(features)
        return features


class SupportResistanceFeatures(FeatureEngineering):
    """Support and resistance features for regime detection (6 dimensions)."""

    def find_support_resistance_levels(self, high: pd.Series, low: pd.Series,
                                     close: pd.Series, window: int = 20) -> Tuple[float, float]:
        """Simple support/resistance identification using rolling min/max."""
        recent_high = high.rolling(window).max().iloc[-1]
        recent_low = low.rolling(window).min().iloc[-1]
        return recent_low, recent_high

    def calculate(self, symbol: str) -> pd.DataFrame:
        """Calculate support/resistance features for a symbol."""
        df = self.get_data(symbol, days=300)
        df = self.handle_missing_data(df)

        features = pd.DataFrame(index=df.index)

        # Calculate rolling support/resistance levels
        support_levels = []
        resistance_levels = []

        for i in range(len(df)):
            if i < 20:
                support_levels.append(df['Low'].iloc[:i+1].min() if i > 0 else df['Low'].iloc[0])
                resistance_levels.append(df['High'].iloc[:i+1].max() if i > 0 else df['High'].iloc[0])
            else:
                support_levels.append(df['Low'].iloc[i-20:i+1].min())
                resistance_levels.append(df['High'].iloc[i-20:i+1].max())

        support_series = pd.Series(support_levels, index=df.index)
        resistance_series = pd.Series(resistance_levels, index=df.index)

        # 1. Distance to support (negative if below support)
        features['distance_to_support'] = (df['Close'] - support_series) / support_series

        # 2. Support strength (how often price bounced off this level)
        features['support_strength'] = 0.5  # Stub - would need bounce detection

        # 3. Distance to resistance (positive if below resistance)
        features['distance_to_resistance'] = (resistance_series - df['Close']) / resistance_series

        # 4. Resistance strength (how often price was rejected at this level)
        features['resistance_strength'] = 0.5  # Stub - would need rejection detection

        # 5. Consolidation score (range tightness)
        range_20d = (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        features['consolidation_score'] = range_20d / df['Close']

        # 6. Range width (current range relative to average)
        current_range = (df['High'] - df['Low']) / df['Close']
        avg_range = current_range.rolling(20).mean()
        features['range_width'] = current_range / avg_range

        # Handle missing data and normalize
        features = self.handle_missing_data(features)
        features = features.fillna(0)

        for col in features.columns:
            features[col] = self.standardize(features[col], method='robust')

        self.validate(features)
        return features


class MarketContextFeatures(FeatureEngineering):
    """Market context features for regime detection (4 dimensions)."""

    def calculate(self, symbol: str) -> pd.DataFrame:
        """Calculate market context features for a symbol."""
        df = self.get_data(symbol, days=300)
        df = self.handle_missing_data(df)

        features = pd.DataFrame(index=df.index)

        # 1. SPY correlation (20-day rolling correlation with SPY)
        if symbol != 'SPY':
            try:
                spy_data = self.get_data('SPY', days=300)
                # Align dates and calculate correlation
                aligned_data = pd.concat([df['Close'], spy_data['Close']], axis=1, keys=[symbol, 'SPY']).dropna()
                if len(aligned_data) > 20:
                    rolling_corr = aligned_data[symbol].rolling(20).corr(aligned_data['SPY'])
                    # Reindex to match original data
                    features['spy_correlation'] = rolling_corr.reindex(df.index).fillna(0.7)
                else:
                    features['spy_correlation'] = 0.7  # Default moderate correlation
            except:
                features['spy_correlation'] = 0.7  # Default if SPY data unavailable
        else:
            features['spy_correlation'] = 1.0  # Perfect correlation for SPY itself

        # 2. Sector relative strength (stubbed - would need sector ETF data)
        features['sector_relative_strength'] = 0.0  # Neutral relative strength

        # 3. Market breadth (stubbed - would need broad market data)
        features['market_breadth'] = 0.5  # Neutral breadth

        # 4. Put/call ratio (stubbed - would need options market data)
        features['put_call_ratio'] = 1.0  # Neutral put/call ratio

        # Handle missing data and normalize
        features = self.handle_missing_data(features)
        features = features.fillna(0)

        for col in features.columns:
            features[col] = self.standardize(features[col], method='robust')

        self.validate(features)
        return features


class EventFeatures(FeatureEngineering):
    """Event-based features for regime detection (3 dimensions)."""

    def calculate(self, symbol: str) -> pd.DataFrame:
        """Calculate event features for a symbol."""
        df = self.get_data(symbol, days=300)

        features = pd.DataFrame(index=df.index)

        # All event features are stubbed as they require external calendars
        # 1. Days to earnings (stubbed - would need earnings calendar)
        features['days_to_earnings'] = 30.0  # Assume average quarterly cycle

        # 2. Days to FOMC meeting (stubbed - would need FOMC calendar)
        features['days_to_fomc'] = 21.0  # Assume average meeting cycle

        # 3. Days to options expiration (stubbed - would need options calendar)
        features['days_to_opex'] = 10.0  # Assume average to monthly expiration

        # Normalize to [-1, 1] range
        for col in features.columns:
            features[col] = self.standardize(features[col], method='minmax')

        self.validate(features)
        return features


class RegimeStateVector(FeatureEngineering):
    """
    Complete 48-dimensional regime state vector assembler.

    Combines all feature categories:
    - Price structure (6 dimensions)
    - Trend indicators (9 dimensions)
    - Momentum indicators (6 dimensions)
    - Volatility features (11 dimensions)
    - Volume features (3 dimensions)
    - Support/resistance (6 dimensions)
    - Market context (4 dimensions)
    - Event features (3 dimensions)
    Total: 48 dimensions
    """

    def __init__(self):
        super().__init__()
        self.price_features = PriceStructureFeatures()
        self.trend_features = TrendIndicators()
        self.momentum_features = MomentumIndicators()
        self.volatility_features = VolatilityFeatures()
        self.volume_features = VolumeFeatures()
        self.support_resistance_features = SupportResistanceFeatures()
        self.market_context_features = MarketContextFeatures()
        self.event_features = EventFeatures()

    def calculate(self, symbol: str) -> pd.Series:
        """
        Calculate complete 48-dimensional state vector for a symbol.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Series with latest state vector (48 dimensions)
        """
        # Calculate all feature categories
        price_feat = self.price_features.calculate(symbol)
        trend_feat = self.trend_features.calculate(symbol)
        momentum_feat = self.momentum_features.calculate(symbol)
        volatility_feat = self.volatility_features.calculate(symbol)
        volume_feat = self.volume_features.calculate(symbol)
        support_resistance_feat = self.support_resistance_features.calculate(symbol)
        market_context_feat = self.market_context_features.calculate(symbol)
        event_feat = self.event_features.calculate(symbol)

        # Combine features (take most recent values)
        latest_price = price_feat.iloc[-1]
        latest_trend = trend_feat.iloc[-1]
        latest_momentum = momentum_feat.iloc[-1]
        latest_volatility = volatility_feat.iloc[-1]
        latest_volume = volume_feat.iloc[-1]
        latest_support_resistance = support_resistance_feat.iloc[-1]
        latest_market_context = market_context_feat.iloc[-1]
        latest_event = event_feat.iloc[-1]

        # Concatenate all features
        state_vector = pd.concat([
            latest_price, latest_trend, latest_momentum,
            latest_volatility, latest_volume, latest_support_resistance,
            latest_market_context, latest_event
        ])

        # Validate final output
        self.validate(state_vector)

        return state_vector