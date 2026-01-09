"""
Data utilities and preprocessing pipeline for regime detection.

Provides data augmentation, preprocessing, and analysis utilities
for enhanced training data quality.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyzes feature importance and distributions in training data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize analyzer.

        Args:
            features: Feature array of shape (N, 48)
            labels: Label array of shape (N,)
            feature_names: Optional list of feature names
        """
        self.features = features
        self.labels = labels
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]

    def calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance using correlation with regime changes.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance_scores = {}

        for i, feature_name in enumerate(self.feature_names):
            feature_values = self.features[:, i]

            # Calculate correlation with regime transitions
            regime_changes = np.diff(self.labels, prepend=self.labels[0])
            change_magnitude = np.abs(regime_changes)

            # Feature importance based on correlation with regime changes
            if len(feature_values) > 1 and np.std(feature_values) > 0:
                correlation = np.corrcoef(feature_values[1:], change_magnitude)[0, 1]
                importance_scores[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance_scores[feature_name] = 0.0

        return importance_scores

    def analyze_feature_distributions(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature distributions by regime.

        Returns:
            Dictionary with statistics by regime
        """
        analysis = {}

        for regime in np.unique(self.labels):
            regime_mask = self.labels == regime
            regime_features = self.features[regime_mask]

            regime_stats = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_values = regime_features[:, i]
                if len(feature_values) > 0:
                    regime_stats[feature_name] = {
                        'mean': np.mean(feature_values),
                        'std': np.std(feature_values),
                        'min': np.min(feature_values),
                        'max': np.max(feature_values)
                    }

            analysis[f'regime_{regime}'] = regime_stats

        return analysis

    def get_correlation_matrix(self) -> np.ndarray:
        """Calculate feature correlation matrix."""
        return np.corrcoef(self.features.T)

    def detect_outliers(self, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outlier samples using z-score method.

        Args:
            threshold: Z-score threshold for outlier detection

        Returns:
            Boolean array indicating outliers
        """
        z_scores = np.abs((self.features - np.mean(self.features, axis=0)) /
                         (np.std(self.features, axis=0) + 1e-8))
        outliers = np.any(z_scores > threshold, axis=1)
        return outliers


class DataAugmenter:
    """Provides data augmentation techniques for regime detection training."""

    def __init__(self, noise_std: float = 0.05, augmentation_factor: int = 2):
        """
        Initialize augmenter.

        Args:
            noise_std: Standard deviation for noise injection
            augmentation_factor: Factor by which to increase dataset size
        """
        self.noise_std = noise_std
        self.augmentation_factor = augmentation_factor

    def add_gaussian_noise(self, features: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to features.

        Args:
            features: Input feature array

        Returns:
            Features with added noise
        """
        noise = np.random.normal(0, self.noise_std, features.shape)
        return features + noise

    def temporal_jitter(self, features: np.ndarray, labels: np.ndarray,
                       jitter_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal jittering by slightly shifting time series.

        Args:
            features: Feature array
            labels: Label array
            jitter_window: Maximum shift in samples

        Returns:
            Jittered features and corresponding labels
        """
        jittered_features = []
        jittered_labels = []

        for i in range(len(features)):
            # Random jitter within window
            shift = np.random.randint(-jitter_window, jitter_window + 1)
            jittered_idx = max(0, min(len(features) - 1, i + shift))

            jittered_features.append(features[jittered_idx])
            jittered_labels.append(labels[jittered_idx])

        return np.array(jittered_features), np.array(jittered_labels)

    def augment_dataset(self, features: np.ndarray, labels: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full augmentation pipeline.

        Args:
            features: Original features
            labels: Original labels

        Returns:
            Augmented features and labels
        """
        augmented_features = [features]
        augmented_labels = [labels]

        for _ in range(self.augmentation_factor - 1):
            # Add noise augmentation
            noisy_features = self.add_gaussian_noise(features)

            # Apply temporal jitter
            jittered_features, jittered_labels = self.temporal_jitter(noisy_features, labels)

            augmented_features.append(jittered_features)
            augmented_labels.append(jittered_labels)

        # Combine all augmentations
        final_features = np.vstack(augmented_features)
        final_labels = np.concatenate(augmented_labels)

        logger.info(f"Dataset augmented from {len(features)} to {len(final_features)} samples")

        return final_features, final_labels


class FeaturePreprocessor:
    """Handles feature preprocessing and normalization."""

    def __init__(self, method: str = 'robust', feature_range: Tuple[float, float] = (-1, 1)):
        """
        Initialize preprocessor.

        Args:
            method: Scaling method ('robust', 'standard', 'minmax')
            feature_range: Target range for features
        """
        self.method = method
        self.feature_range = feature_range
        self.scaler = None
        self._is_fitted = False

    def fit(self, features: np.ndarray) -> 'FeaturePreprocessor':
        """
        Fit preprocessor to training data.

        Args:
            features: Training features

        Returns:
            Self for method chaining
        """
        if self.method == 'robust':
            self.scaler = RobustScaler()
        elif self.method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.method}")

        self.scaler.fit(features)
        self._is_fitted = True

        logger.info(f"Fitted {self.method} scaler to {features.shape[0]} samples")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted preprocessor.

        Args:
            features: Features to transform

        Returns:
            Transformed features
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        scaled_features = self.scaler.transform(features)

        # Clip to target range if specified
        if self.feature_range is not None:
            min_val, max_val = self.feature_range
            scaled_features = np.clip(scaled_features, min_val, max_val)

        return scaled_features.astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(features).transform(features)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform features back to original scale."""
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted.")

        return self.scaler.inverse_transform(features)


class DataStatistics:
    """Provides comprehensive data statistics and reporting."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, dates: pd.DatetimeIndex):
        """Initialize statistics calculator."""
        self.features = features
        self.labels = labels
        self.dates = dates

    def generate_report(self) -> Dict[str, any]:
        """
        Generate comprehensive data report.

        Returns:
            Dictionary with detailed statistics
        """
        report = {}

        # Basic statistics
        report['dataset_info'] = {
            'num_samples': len(self.features),
            'num_features': self.features.shape[1],
            'date_range': {
                'start': str(self.dates.min()),
                'end': str(self.dates.max()),
                'duration_days': (self.dates.max() - self.dates.min()).days
            }
        }

        # Feature statistics
        report['feature_stats'] = {
            'mean': np.mean(self.features, axis=0).tolist(),
            'std': np.std(self.features, axis=0).tolist(),
            'min': np.min(self.features, axis=0).tolist(),
            'max': np.max(self.features, axis=0).tolist(),
            'nan_count': np.sum(np.isnan(self.features), axis=0).tolist(),
            'inf_count': np.sum(np.isinf(self.features), axis=0).tolist()
        }

        # Label distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)

        report['label_distribution'] = {
            'regime_counts': {int(label): int(count) for label, count in zip(unique_labels, counts)},
            'regime_percentages': {int(label): float(count / total_samples)
                                 for label, count in zip(unique_labels, counts)},
            'most_common_regime': int(unique_labels[np.argmax(counts)]),
            'least_common_regime': int(unique_labels[np.argmin(counts)]),
            'balance_ratio': float(np.min(counts) / np.max(counts))
        }

        # Temporal statistics
        regime_changes = np.sum(self.labels[1:] != self.labels[:-1])
        report['temporal_stats'] = {
            'regime_changes': int(regime_changes),
            'change_frequency': float(regime_changes / len(self.labels)),
            'average_regime_duration': float(len(self.labels) / (regime_changes + 1))
        }

        return report

    def print_summary(self) -> None:
        """Print a formatted summary of dataset statistics."""
        report = self.generate_report()

        print("\n" + "="*60)
        print("REGIME DETECTION DATASET SUMMARY")
        print("="*60)

        # Dataset info
        info = report['dataset_info']
        print(f"\nDataset Information:")
        print(f"  Samples: {info['num_samples']:,}")
        print(f"  Features: {info['num_features']}")
        print(f"  Date Range: {info['date_range']['start']} to {info['date_range']['end']}")
        print(f"  Duration: {info['date_range']['duration_days']} days")

        # Feature summary
        feat_stats = report['feature_stats']
        print(f"\nFeature Summary:")
        print(f"  Mean range: [{np.min(feat_stats['mean']):.3f}, {np.max(feat_stats['mean']):.3f}]")
        print(f"  Std range: [{np.min(feat_stats['std']):.3f}, {np.max(feat_stats['std']):.3f}]")
        print(f"  Value range: [{np.min(feat_stats['min']):.3f}, {np.max(feat_stats['max']):.3f}]")
        print(f"  NaN values: {np.sum(feat_stats['nan_count'])}")
        print(f"  Inf values: {np.sum(feat_stats['inf_count'])}")

        # Label distribution
        label_dist = report['label_distribution']
        print(f"\nRegime Distribution:")
        for regime_id, count in label_dist['regime_counts'].items():
            percentage = label_dist['regime_percentages'][regime_id] * 100
            from ..data.regime_labeler import RegimeType
            regime_name = RegimeType(regime_id).name
            print(f"  {regime_name}: {count:,} ({percentage:.1f}%)")

        print(f"\nBalance Statistics:")
        print(f"  Balance ratio: {label_dist['balance_ratio']:.3f}")
        print(f"  Most common: {RegimeType(label_dist['most_common_regime']).name}")
        print(f"  Least common: {RegimeType(label_dist['least_common_regime']).name}")

        # Temporal statistics
        temporal = report['temporal_stats']
        print(f"\nTemporal Characteristics:")
        print(f"  Regime changes: {temporal['regime_changes']:,}")
        print(f"  Change frequency: {temporal['change_frequency']:.1%}")
        print(f"  Avg regime duration: {temporal['average_regime_duration']:.1f} days")

        print("="*60)


def create_feature_names() -> List[str]:
    """
    Create standardized feature names for the 48-dimensional feature vector.

    Returns:
        List of 48 feature names
    """
    names = []

    # Price structure features (6)
    price_features = [
        'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
        'distance_from_52w_high', 'distance_from_52w_low', 'gap_percentage'
    ]
    names.extend(price_features)

    # Trend indicators (9)
    trend_features = [
        'adx', 'di_plus', 'di_minus', 'macd_line', 'macd_signal',
        'macd_histogram', 'ema_alignment', 'higher_highs', 'lower_lows'
    ]
    names.extend(trend_features)

    # Momentum indicators (6)
    momentum_features = [
        'rsi', 'stoch_k', 'stoch_d', 'roc_5', 'roc_10', 'roc_20'
    ]
    names.extend(momentum_features)

    # Volatility features (11)
    volatility_features = [
        'historical_volatility_20', 'implied_volatility', 'iv_rank', 'iv_percentile',
        'vix_level', 'vix_percentile', 'term_structure_slope', 'put_call_skew',
        'bollinger_band_width', 'bollinger_position', 'atr_normalized'
    ]
    names.extend(volatility_features)

    # Volume features (3)
    volume_features = ['volume_ratio', 'obv_slope', 'volume_trend']
    names.extend(volume_features)

    # Support/resistance features (6)
    support_features = [
        'distance_to_support', 'support_strength', 'distance_to_resistance',
        'resistance_strength', 'consolidation_score', 'range_width'
    ]
    names.extend(support_features)

    # Market context features (4)
    context_features = [
        'spy_correlation', 'sector_relative_strength', 'market_breadth', 'put_call_ratio'
    ]
    names.extend(context_features)

    # Event features (3)
    event_features = ['days_to_earnings', 'days_to_fomc', 'days_to_opex']
    names.extend(event_features)

    assert len(names) == 48, f"Expected 48 features, got {len(names)}"
    return names


def validate_data_pipeline(features: np.ndarray, labels: np.ndarray, dates: pd.DatetimeIndex
                          ) -> Dict[str, bool]:
    """
    Comprehensive validation of the complete data pipeline.

    Args:
        features: Feature array
        labels: Label array
        dates: Date index

    Returns:
        Dictionary with validation results
    """
    validation = {}

    # Shape validation
    validation['correct_feature_dims'] = features.shape[1] == 48
    validation['matching_lengths'] = len(features) == len(labels) == len(dates)

    # Data type validation
    validation['features_float'] = features.dtype in [np.float32, np.float64]
    validation['labels_int'] = labels.dtype in [np.int32, np.int64]

    # Value validation
    validation['no_nan_features'] = not np.any(np.isnan(features))
    validation['no_inf_features'] = not np.any(np.isinf(features))
    validation['valid_label_range'] = np.all((labels >= 0) & (labels <= 7))

    # Distribution validation
    unique_labels = np.unique(labels)
    validation['all_regimes_present'] = len(unique_labels) == 8

    if len(unique_labels) > 0:
        _, counts = np.unique(labels, return_counts=True)
        max_regime_pct = counts.max() / len(labels)
        validation['balanced_regimes'] = max_regime_pct <= 0.4
    else:
        validation['balanced_regimes'] = False

    # Feature range validation (should be roughly normalized)
    feature_std = np.std(features)
    validation['reasonable_feature_scale'] = 0.1 <= feature_std <= 10

    # Temporal validation
    validation['chronological_order'] = dates.is_monotonic_increasing
    validation['sufficient_history'] = len(dates) >= (3 * 252)  # At least 3 years

    # Log validation summary
    passed = sum(validation.values())
    total = len(validation)
    logger.info(f"Data pipeline validation: {passed}/{total} checks passed")

    for check, result in validation.items():
        if not result:
            logger.warning(f"Validation failed: {check}")

    return validation