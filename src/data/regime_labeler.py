"""
Regime labeling system for historical market data.

Implements quantitative rules to classify market periods into 8 regime types
based on price action, momentum, volatility, and trend characteristics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from enum import IntEnum
import logging

from .providers import get_default_provider, DataProviderError
from ..features.base import calculate_sma, calculate_ema

logger = logging.getLogger(__name__)


class RegimeType(IntEnum):
    """Market regime classifications."""
    BULL_TRENDING = 0      # Strong upward momentum
    BEAR_TRENDING = 1      # Strong downward momentum
    HIGH_VOLATILITY = 2    # Elevated volatility environment
    LOW_VOLATILITY = 3     # Subdued volatility environment
    SIDEWAYS_RANGING = 4   # Consolidation patterns
    RECOVERY = 5           # Post-decline bounce patterns
    DISTRIBUTION = 6       # Pre-decline weakening
    CRISIS = 7             # Extreme stress conditions


class RegimeMetrics:
    """Container for regime classification metrics."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.close = data['Close']
        self.high = data['High']
        self.low = data['Low']
        self.volume = data['Volume']

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = self.close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_volatility(self, window: int = 20) -> pd.Series:
        """Calculate rolling volatility (annualized)."""
        returns = self.close.pct_change()
        vol = returns.rolling(window).std() * np.sqrt(252)
        return vol.fillna(vol.mean())

    def calculate_vix_proxy(self) -> pd.Series:
        """Calculate VIX proxy using rolling volatility."""
        vol = self.calculate_volatility(window=30)
        # Scale to approximate VIX range (10-80)
        vix_proxy = vol * 100
        return vix_proxy.clip(5, 100)

    def calculate_drawdown(self, window: int = 252) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        rolling_max = self.close.rolling(window).max()
        drawdown = (self.close - rolling_max) / rolling_max
        return drawdown.fillna(0)


class RegimeLabeler:
    """
    Historical market regime labeling system.

    Uses quantitative rules to classify market periods into 8 distinct regimes
    based on price structure, momentum, volatility, and trend characteristics.
    """

    def __init__(self, stability_window: int = 252):
        """
        Initialize regime labeler.

        Args:
            stability_window: Window for regime stability (default: 252 trading days)
        """
        self.stability_window = stability_window
        self.provider = get_default_provider()

        # Regime thresholds
        self.thresholds = {
            'rsi_high': 55,
            'rsi_low': 45,
            'vix_high': 25,
            'vix_low': 15,
            'vol_high_pct': 30,  # 30% annualized volatility
            'drawdown_crisis': -0.15,  # 15% drawdown for crisis
            'recovery_bounce_min': 0.05,  # 5% bounce minimum for recovery
            'distribution_weakness': -0.02,  # 2% weakness after strength
        }

    def _get_historical_data(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """Fetch historical data for regime analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        try:
            price_data = self.provider.get_price_history(
                symbol=symbol,
                start=start_date,
                end=end_date,
                interval="1d"
            )
            return price_data.data
        except DataProviderError as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise

    def _classify_regime(self, data: pd.DataFrame, idx: int) -> RegimeType:
        """
        Classify regime for a specific time point.

        Args:
            data: DataFrame with calculated indicators
            idx: Index position to classify

        Returns:
            RegimeType enum value
        """
        row = data.iloc[idx]

        # Get key indicators
        price = row['Close']
        sma20 = row['SMA20']
        sma50 = row['SMA50']
        rsi = row['RSI']
        volatility = row['Volatility']
        vix_proxy = row['VIX_Proxy']
        drawdown = row['Drawdown']

        # Crisis conditions (highest priority)
        if drawdown <= self.thresholds['drawdown_crisis']:
            return RegimeType.CRISIS

        # High volatility conditions
        if (vix_proxy > self.thresholds['vix_high'] or
            volatility > self.thresholds['vol_high_pct']):
            return RegimeType.HIGH_VOLATILITY

        # Low volatility conditions
        if (vix_proxy < self.thresholds['vix_low'] and
            volatility < 0.15):  # Less than 15% volatility
            return RegimeType.LOW_VOLATILITY

        # Recovery conditions (strong bounce from oversold)
        if idx >= 20:  # Need history for recovery detection
            recent_low = data['Close'].iloc[idx-20:idx].min()
            current_bounce = (price - recent_low) / recent_low
            oversold = data['RSI'].iloc[idx-10:idx].min() < 30

            if (current_bounce > self.thresholds['recovery_bounce_min'] and
                oversold and rsi > 40):
                return RegimeType.RECOVERY

        # Distribution conditions (weakness after strength)
        if idx >= 20:
            recent_high = data['Close'].iloc[idx-20:idx].max()
            recent_strength = any(data['RSI'].iloc[idx-10:idx] > 70)
            current_weakness = (price - recent_high) / recent_high

            if (recent_strength and
                current_weakness < self.thresholds['distribution_weakness'] and
                rsi < 60):
                return RegimeType.DISTRIBUTION

        # Bull trending conditions
        if (price > sma20 and price > sma50 and
            rsi > self.thresholds['rsi_high'] and
            volatility < 0.25):
            return RegimeType.BULL_TRENDING

        # Bear trending conditions
        if (price < sma20 and price < sma50 and
            rsi < self.thresholds['rsi_low'] and
            volatility < 0.25):
            return RegimeType.BEAR_TRENDING

        # Default to sideways/ranging
        return RegimeType.SIDEWAYS_RANGING

    def _apply_stability_filter(self, regimes: pd.Series) -> pd.Series:
        """Apply stability filter to prevent excessive regime switching."""
        filtered_regimes = regimes.copy()
        window = min(self.stability_window // 10, 25)  # 25-day stability window

        for i in range(window, len(regimes)):
            # Look at regime distribution in recent window
            recent_regimes = regimes.iloc[i-window:i]
            regime_counts = recent_regimes.value_counts()

            # If current regime is very different from recent consensus, smooth it
            if len(regime_counts) > 0:
                dominant_regime = regime_counts.index[0]
                dominant_pct = regime_counts.iloc[0] / len(recent_regimes)

                # If current regime differs and doesn't have strong recent support
                current_regime = regimes.iloc[i]
                current_support = regime_counts.get(current_regime, 0) / len(recent_regimes)

                if dominant_pct > 0.4 and current_support < 0.2:
                    # Smooth towards dominant regime
                    filtered_regimes.iloc[i] = dominant_regime

        return filtered_regimes

    def label_historical_data(self, symbol: str = 'SPY', years: int = 5) -> pd.DataFrame:
        """
        Label historical data with regime classifications.

        Args:
            symbol: Stock symbol to analyze (default: SPY)
            years: Years of historical data to analyze

        Returns:
            DataFrame with date index and regime labels
        """
        logger.info(f"Starting regime labeling for {symbol} over {years} years")

        # Get historical data
        data = self._get_historical_data(symbol, years)

        # Calculate indicators
        metrics = RegimeMetrics(data)

        data['SMA20'] = calculate_sma(data['Close'], 20)
        data['SMA50'] = calculate_sma(data['Close'], 50)
        data['SMA200'] = calculate_sma(data['Close'], 200)
        data['RSI'] = metrics.calculate_rsi()
        data['Volatility'] = metrics.calculate_volatility()
        data['VIX_Proxy'] = metrics.calculate_vix_proxy()
        data['Drawdown'] = metrics.calculate_drawdown()

        # Fill NaN values
        data = data.ffill().bfill()

        # Apply regime classification
        regimes = []
        for i in range(len(data)):
            if i < 50:  # Need sufficient history for stable classification
                regimes.append(RegimeType.SIDEWAYS_RANGING)  # Default for early periods
            else:
                regime = self._classify_regime(data, i)
                regimes.append(regime)

        data['Regime'] = regimes

        # Apply stability filtering
        data['Regime'] = self._apply_stability_filter(data['Regime'])

        # Create result DataFrame
        result = pd.DataFrame({
            'Date': data.index,
            'Regime': data['Regime'],
            'RegimeName': [RegimeType(r).name for r in data['Regime']],
            'Close': data['Close'],
            'RSI': data['RSI'],
            'Volatility': data['Volatility'],
            'VIX_Proxy': data['VIX_Proxy'],
            'Drawdown': data['Drawdown']
        })

        result.set_index('Date', inplace=True)

        logger.info(f"Regime labeling complete. Labeled {len(result)} periods")
        self._log_regime_distribution(result)

        return result

    def _log_regime_distribution(self, labeled_data: pd.DataFrame) -> None:
        """Log regime distribution statistics."""
        regime_counts = labeled_data['Regime'].value_counts()
        total_periods = len(labeled_data)

        logger.info("Regime Distribution:")
        for regime_num, count in regime_counts.items():
            regime_name = RegimeType(regime_num).name
            percentage = (count / total_periods) * 100
            logger.info(f"  {regime_name}: {count} periods ({percentage:.1f}%)")

        # Validate no regime dominates >40%
        max_percentage = regime_counts.iloc[0] / total_periods
        if max_percentage > 0.4:
            logger.warning(f"Regime {RegimeType(regime_counts.index[0]).name} dominates "
                          f"{max_percentage:.1%} of periods (>40% threshold)")

    def validate_regime_labels(self, labeled_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate regime labeling quality.

        Returns:
            Dictionary with validation results
        """
        results = {}

        # Check temporal coverage
        results['sufficient_data'] = len(labeled_data) >= (5 * 252)  # 5 years of trading days

        # Check all regimes represented
        unique_regimes = set(labeled_data['Regime'].unique())
        all_regimes = set(range(8))
        results['all_regimes_present'] = unique_regimes == all_regimes

        # Check no regime dominates >40%
        regime_counts = labeled_data['Regime'].value_counts(normalize=True)
        results['no_regime_dominance'] = regime_counts.iloc[0] <= 0.4

        # Check reasonable regime transitions (not too frequent)
        regime_changes = (labeled_data['Regime'] != labeled_data['Regime'].shift(1)).sum()
        change_rate = regime_changes / len(labeled_data)
        results['stable_regimes'] = change_rate < 0.1  # Less than 10% daily changes

        return results


def create_regime_labels(symbol: str = 'SPY', years: int = 5) -> pd.DataFrame:
    """
    Convenience function to create regime labels for a symbol.

    Args:
        symbol: Stock symbol to analyze
        years: Years of historical data

    Returns:
        DataFrame with regime labels
    """
    labeler = RegimeLabeler()
    return labeler.label_historical_data(symbol, years)