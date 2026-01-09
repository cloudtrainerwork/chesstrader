"""
Base classes for feature engineering.

Provides abstract base class and common utilities for feature extraction.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from ..data import get_default_provider, CacheManager, MarketDataRequest


class FeatureEngineering(ABC):
    """
    Abstract base class for feature engineering components.

    Provides standardized interface and common utilities for feature extraction,
    normalization, and validation.
    """

    def __init__(self):
        self.data_provider = get_default_provider()
        self.cache_manager = CacheManager()

    def get_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """
        Get price data for a symbol.

        Args:
            symbol: Stock symbol to fetch
            days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        try:
            from datetime import datetime, timedelta

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            price_data = self.cache_manager.get_price_history(
                symbol=symbol,
                start=start_date,
                end=end_date,
                interval="1d"
            )

            if not price_data.is_valid:
                raise ValueError(f"No valid data available for {symbol}")

            return price_data.data
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {symbol}: {e}")

    def standardize(self, values: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        Standardize feature values to approximately [-1, 1] range.

        Args:
            values: Series of feature values to standardize
            method: Standardization method ('zscore', 'minmax', 'robust')

        Returns:
            Standardized series
        """
        if method == 'zscore':
            # Z-score normalization, then tanh to bound to [-1, 1]
            mean = values.mean()
            std = values.std()
            if std == 0:
                return pd.Series(np.zeros(len(values)), index=values.index)
            z_scores = (values - mean) / std
            return np.tanh(z_scores)

        elif method == 'minmax':
            # Min-max normalization to [-1, 1]
            min_val = values.min()
            max_val = values.max()
            if max_val == min_val:
                return pd.Series(np.zeros(len(values)), index=values.index)
            return 2 * (values - min_val) / (max_val - min_val) - 1

        elif method == 'robust':
            # Robust normalization using percentiles
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            iqr = q75 - q25
            if iqr == 0:
                return pd.Series(np.zeros(len(values)), index=values.index)
            median = values.median()
            return np.tanh((values - median) / iqr)

        else:
            raise ValueError(f"Unknown standardization method: {method}")

    def validate(self, features: Union[pd.DataFrame, pd.Series]) -> bool:
        """
        Validate feature output for quality and completeness.

        Args:
            features: Feature values to validate

        Returns:
            True if features pass validation

        Raises:
            ValueError: If features fail validation
        """
        if features is None or features.empty:
            raise ValueError("Features cannot be None or empty")

        # Check for NaN values
        if features.isnull().any().any() if isinstance(features, pd.DataFrame) else features.isnull().any():
            raise ValueError("Features contain NaN values")

        # Check for infinite values
        if np.isinf(features).any().any() if isinstance(features, pd.DataFrame) else np.isinf(features).any():
            raise ValueError("Features contain infinite values")

        # Check feature range (should be roughly [-1, 1] for neural networks)
        if isinstance(features, pd.DataFrame):
            for col in features.columns:
                if features[col].max() > 5 or features[col].min() < -5:
                    print(f"Warning: Feature {col} has extreme values (range: {features[col].min():.3f} to {features[col].max():.3f})")
        else:
            if features.max() > 5 or features.min() < -5:
                print(f"Warning: Features have extreme values (range: {features.min():.3f} to {features.max():.3f})")

        return True

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data with forward fill then backward fill.

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled
        """
        return df.ffill().bfill()

    @abstractmethod
    def calculate(self, symbol: str) -> Union[pd.DataFrame, pd.Series]:
        """
        Calculate features for a given symbol.

        Args:
            symbol: Stock symbol to calculate features for

        Returns:
            Calculated features
        """
        pass


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=window, min_periods=1).mean()


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=span, adjust=False).mean()


def calculate_rolling_high(prices: pd.Series, window: int) -> pd.Series:
    """Calculate rolling maximum over window."""
    return prices.rolling(window=window, min_periods=1).max()


def calculate_rolling_low(prices: pd.Series, window: int) -> pd.Series:
    """Calculate rolling minimum over window."""
    return prices.rolling(window=window, min_periods=1).min()