"""
Training data assembly and preprocessing for regime detection.

Combines regime features with regime labels into training-ready datasets
with proper splits and validation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import logging

from .regime_labeler import RegimeLabeler, RegimeType
from ..features.regime_features import RegimeStateVector
from ..features.base import FeatureEngineering

logger = logging.getLogger(__name__)


class RegimeDataset(Dataset):
    """
    PyTorch Dataset for regime detection training.

    Pairs 48-dimensional feature vectors with regime labels for neural network training.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, dates: pd.DatetimeIndex):
        """
        Initialize dataset.

        Args:
            features: Feature array of shape (N, 48)
            labels: Label array of shape (N,) with regime indices
            dates: DatetimeIndex for temporal reference
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.dates = dates

        # Validate inputs
        assert len(features) == len(labels) == len(dates), "Length mismatch in inputs"
        assert features.shape[1] == 48, f"Expected 48 features, got {features.shape[1]}"
        assert np.all((labels >= 0) & (labels <= 7)), "Labels must be in range [0, 7]"

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def get_regime_distribution(self) -> Dict[str, int]:
        """Get distribution of regimes in dataset."""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {RegimeType(int(u)).name: int(c) for u, c in zip(unique, counts)}


class TrainingDataCollector:
    """
    Collects and prepares training data for regime detection.

    Combines historical market data, regime labeling, and feature engineering
    into a complete training dataset with proper splits.
    """

    def __init__(self, symbol: str = 'SPY', years: int = 5, feature_window: int = 48):
        """
        Initialize data collector.

        Args:
            symbol: Stock symbol to analyze
            years: Years of historical data
            feature_window: Number of feature dimensions (should be 48)
        """
        self.symbol = symbol
        self.years = years
        self.feature_window = feature_window

        self.regime_labeler = RegimeLabeler()
        self.feature_extractor = RegimeStateVector()

        self._regime_data = None
        self._feature_data = None
        self._aligned_data = None

    def collect_regime_labels(self) -> pd.DataFrame:
        """Collect regime labels for historical data."""
        logger.info(f"Collecting regime labels for {self.symbol}")
        self._regime_data = self.regime_labeler.label_historical_data(
            symbol=self.symbol, years=self.years
        )
        return self._regime_data

    def collect_feature_data(self) -> pd.DataFrame:
        """Collect feature data using RegimeStateVector."""
        logger.info(f"Collecting feature data for {self.symbol}")

        # Calculate features for multiple time periods
        feature_calculator = self.feature_extractor.price_features

        # Get sufficient data for feature calculation
        data = feature_calculator.get_data(self.symbol, days=self.years * 365)

        # Calculate all feature categories
        price_feat = self.feature_extractor.price_features.calculate(self.symbol)
        trend_feat = self.feature_extractor.trend_features.calculate(self.symbol)
        momentum_feat = self.feature_extractor.momentum_features.calculate(self.symbol)
        volatility_feat = self.feature_extractor.volatility_features.calculate(self.symbol)
        volume_feat = self.feature_extractor.volume_features.calculate(self.symbol)
        support_resistance_feat = self.feature_extractor.support_resistance_features.calculate(self.symbol)
        market_context_feat = self.feature_extractor.market_context_features.calculate(self.symbol)
        event_feat = self.feature_extractor.event_features.calculate(self.symbol)

        # Combine all features
        feature_dfs = [
            price_feat, trend_feat, momentum_feat, volatility_feat,
            volume_feat, support_resistance_feat, market_context_feat, event_feat
        ]

        self._feature_data = pd.concat(feature_dfs, axis=1)
        logger.info(f"Collected {self._feature_data.shape[1]} features for {len(self._feature_data)} periods")

        return self._feature_data

    def align_features_and_labels(self) -> pd.DataFrame:
        """Align feature vectors with regime labels temporally."""
        if self._regime_data is None:
            self.collect_regime_labels()
        if self._feature_data is None:
            self.collect_feature_data()

        logger.info("Aligning features with regime labels")

        # Find common date range
        common_dates = self._regime_data.index.intersection(self._feature_data.index)
        logger.info(f"Found {len(common_dates)} common dates")

        # Align data
        aligned_regimes = self._regime_data.loc[common_dates]
        aligned_features = self._feature_data.loc[common_dates]

        # Combine into single DataFrame
        self._aligned_data = pd.concat([aligned_features, aligned_regimes[['Regime']]], axis=1)

        # Remove any rows with NaN values
        initial_length = len(self._aligned_data)
        self._aligned_data = self._aligned_data.dropna()
        final_length = len(self._aligned_data)

        if initial_length != final_length:
            logger.warning(f"Removed {initial_length - final_length} rows with NaN values")

        logger.info(f"Aligned dataset: {len(self._aligned_data)} samples with {self._aligned_data.shape[1]-1} features")

        return self._aligned_data

    def create_training_arrays(self) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Create training arrays from aligned data.

        Returns:
            Tuple of (features, labels, dates)
        """
        if self._aligned_data is None:
            self.align_features_and_labels()

        # Separate features and labels
        feature_columns = [col for col in self._aligned_data.columns if col != 'Regime']
        features = self._aligned_data[feature_columns].values
        labels = self._aligned_data['Regime'].values
        dates = self._aligned_data.index

        # Validate feature dimensions
        if features.shape[1] != 48:
            logger.error(f"Expected 48 features, got {features.shape[1]}")
            raise ValueError(f"Feature dimension mismatch: expected 48, got {features.shape[1]}")

        # Validate feature ranges (should be in [-1, 1])
        feature_min = features.min()
        feature_max = features.max()
        if feature_min < -2 or feature_max > 2:
            logger.warning(f"Features outside expected range [-1, 1]: min={feature_min:.3f}, max={feature_max:.3f}")

        # Validate labels
        unique_labels = np.unique(labels)
        logger.info(f"Unique regime labels: {sorted(unique_labels)}")

        return features.astype(np.float32), labels.astype(np.int64), dates

    def validate_data_quality(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, bool]:
        """
        Validate training data quality.

        Returns:
            Dictionary with validation results
        """
        results = {}

        # Check for NaN/inf values
        results['no_nan_features'] = not np.any(np.isnan(features))
        results['no_inf_features'] = not np.any(np.isinf(features))
        results['valid_labels'] = np.all((labels >= 0) & (labels <= 7))

        # Check feature normalization
        results['features_normalized'] = np.all((features >= -3) & (features <= 3))  # Lenient check

        # Check regime distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        results['all_regimes_present'] = len(unique_labels) == 8

        if len(counts) > 0:
            max_regime_pct = counts.max() / len(labels)
            results['no_regime_dominance'] = max_regime_pct <= 0.4
        else:
            results['no_regime_dominance'] = False

        # Log validation results
        logger.info("Data quality validation:")
        for check, passed in results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {check}: {status}")

        return results


def create_temporal_splits(features: np.ndarray, labels: np.ndarray, dates: pd.DatetimeIndex,
                          train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
                          ) -> Tuple[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex], ...]:
    """
    Create temporal train/val/test splits preserving chronological order.

    Args:
        features: Feature array
        labels: Label array
        dates: DatetimeIndex
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Tuple of (train_data, val_data, test_data) where each is (features, labels, dates)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n_samples = len(features)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    # Split data temporally
    train_features = features[:train_end]
    train_labels = labels[:train_end]
    train_dates = dates[:train_end]

    val_features = features[train_end:val_end]
    val_labels = labels[train_end:val_end]
    val_dates = dates[train_end:val_end]

    test_features = features[val_end:]
    test_labels = labels[val_end:]
    test_dates = dates[val_end:]

    logger.info(f"Temporal splits: train={len(train_features)}, val={len(val_features)}, test={len(test_features)}")

    # Verify all regimes present in training set
    train_regimes = set(train_labels)
    if len(train_regimes) < 8:
        missing_regimes = set(range(8)) - train_regimes
        logger.warning(f"Missing regimes in training set: {missing_regimes}")

    return ((train_features, train_labels, train_dates),
            (val_features, val_labels, val_dates),
            (test_features, test_labels, test_dates))


def create_balanced_sampler(labels: np.ndarray) -> torch.utils.data.WeightedRandomSampler:
    """
    Create balanced sampler for handling class imbalance.

    Args:
        labels: Label array

    Returns:
        WeightedRandomSampler for balanced sampling
    """
    # Calculate class weights
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )

    # Create sample weights
    sample_weights = np.array([class_weights[label] for label in labels])
    sample_weights = torch.DoubleTensor(sample_weights)

    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def create_data_loaders(symbol: str = 'SPY', years: int = 5, batch_size: int = 32,
                       num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        symbol: Stock symbol
        years: Years of historical data
        batch_size: Batch size for training
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating data loaders for {symbol} with {years} years of data")

    # Collect and prepare data
    collector = TrainingDataCollector(symbol=symbol, years=years)
    features, labels, dates = collector.create_training_arrays()

    # Validate data quality
    validation_results = collector.validate_data_quality(features, labels)
    if not all(validation_results.values()):
        failed_checks = [k for k, v in validation_results.items() if not v]
        logger.error(f"Data quality validation failed: {failed_checks}")
        raise ValueError(f"Data quality issues detected: {failed_checks}")

    # Create temporal splits
    (train_features, train_labels, train_dates), \
    (val_features, val_labels, val_dates), \
    (test_features, test_labels, test_dates) = create_temporal_splits(features, labels, dates)

    # Create datasets
    train_dataset = RegimeDataset(train_features, train_labels, train_dates)
    val_dataset = RegimeDataset(val_features, val_labels, val_dates)
    test_dataset = RegimeDataset(test_features, test_labels, test_dates)

    # Create balanced sampler for training
    train_sampler = create_balanced_sampler(train_labels)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    logger.info("Data loaders created successfully")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Log regime distributions
    logger.info("Training set regime distribution:")
    train_dist = train_dataset.get_regime_distribution()
    for regime, count in train_dist.items():
        percentage = (count / len(train_dataset)) * 100
        logger.info(f"  {regime}: {count} ({percentage:.1f}%)")

    return train_loader, val_loader, test_loader


class TrainingDataset:
    """
    High-level interface for creating training datasets.

    Provides a simple interface for getting prepared training data
    with all necessary preprocessing and validation.
    """

    def __init__(self, symbol: str = 'SPY', years: int = 5):
        """Initialize dataset builder."""
        self.symbol = symbol
        self.years = years
        self.collector = TrainingDataCollector(symbol, years)

        self._features = None
        self._labels = None
        self._dates = None

    def prepare_data(self) -> None:
        """Prepare the complete dataset."""
        self._features, self._labels, self._dates = self.collector.create_training_arrays()

    @property
    def features(self) -> np.ndarray:
        """Get feature array."""
        if self._features is None:
            self.prepare_data()
        return self._features

    @property
    def labels(self) -> np.ndarray:
        """Get label array."""
        if self._labels is None:
            self.prepare_data()
        return self._labels

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Get date index."""
        if self._dates is None:
            self.prepare_data()
        return self._dates

    def __len__(self) -> int:
        return len(self.features)

    def get_splits(self) -> Tuple[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex], ...]:
        """Get temporal train/val/test splits."""
        return create_temporal_splits(self.features, self.labels, self.dates)