"""
Comprehensive tests for training metrics and monitoring system.

Tests regime classification metrics, confidence calibration, temporal stability,
training progress analysis, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.training.metrics import (
    calculate_regime_metrics, calculate_calibration_metrics,
    calculate_temporal_stability_metrics, track_training_progress,
    calculate_top_k_accuracy, calculate_expected_calibration_error,
    calculate_maximum_calibration_error, calculate_confidence_accuracy_correlation,
    calculate_regime_transition_matrix, calculate_regime_persistence,
    calculate_convergence_rate, calculate_loss_smoothness,
    calculate_overfitting_score, assess_training_health
)


@pytest.fixture
def sample_predictions():
    """Create sample prediction data."""
    np.random.seed(42)
    n_samples = 100

    # Regime predictions (0-7)
    predictions = np.random.randint(0, 8, n_samples)

    # Regime probabilities (should sum to 1)
    probabilities = np.random.dirichlet(np.ones(8), n_samples)

    # True labels
    labels = np.random.randint(0, 8, n_samples)

    # Confidence scores
    confidence_scores = np.random.beta(2, 2, n_samples)  # Beta distribution for [0,1] values

    return predictions, probabilities, labels, confidence_scores


@pytest.fixture
def sample_temporal_data():
    """Create sample temporal prediction data."""
    np.random.seed(42)
    n_samples = 200

    # Create temporal predictions with some persistence
    predictions = []
    current_regime = np.random.randint(0, 8)
    for i in range(n_samples):
        if np.random.random() < 0.1:  # 10% chance of regime change
            current_regime = np.random.randint(0, 8)
        predictions.append(current_regime)

    predictions = np.array(predictions)
    probabilities = np.random.dirichlet(np.ones(8), n_samples)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    return predictions, probabilities, dates


@pytest.fixture
def sample_training_history():
    """Create sample training history data."""
    epochs = 20

    # Simulate decreasing loss with some noise
    base_loss = np.exp(-np.linspace(0, 2, epochs))
    train_loss = base_loss + np.random.normal(0, 0.01, epochs)
    val_loss = base_loss * 1.1 + np.random.normal(0, 0.02, epochs)

    # Simulate increasing accuracy
    val_accuracy = 1 - base_loss * 0.8 + np.random.normal(0, 0.01, epochs)
    val_accuracy = np.clip(val_accuracy, 0, 1)

    # Learning rate schedule
    learning_rate = [1e-3] * 10 + [5e-4] * 6 + [2.5e-4] * 4

    return {
        'train_loss': train_loss.tolist(),
        'val_loss': val_loss.tolist(),
        'val_accuracy': val_accuracy.tolist(),
        'val_classification_acc': (val_accuracy * 0.95).tolist(),
        'val_confidence_mae': (base_loss * 0.1).tolist(),
        'learning_rate': learning_rate
    }


class TestRegimeMetrics:
    """Test regime classification metrics calculation."""

    def test_calculate_regime_metrics_basic(self, sample_predictions):
        """Test basic regime metrics calculation."""
        predictions, probabilities, labels, _ = sample_predictions

        metrics = calculate_regime_metrics(predictions, probabilities, labels)

        # Check required metrics exist
        assert 'overall_accuracy' in metrics
        assert 'balanced_accuracy' in metrics
        assert 'macro_precision' in metrics
        assert 'macro_recall' in metrics
        assert 'macro_f1' in metrics
        assert 'top_2_accuracy' in metrics
        assert 'top_3_accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert 'per_regime_metrics' in metrics

        # Check value ranges
        assert 0 <= metrics['overall_accuracy'] <= 1
        assert 0 <= metrics['balanced_accuracy'] <= 1
        assert 0 <= metrics['macro_precision'] <= 1
        assert 0 <= metrics['macro_recall'] <= 1
        assert 0 <= metrics['macro_f1'] <= 1
        assert 0 <= metrics['top_2_accuracy'] <= 1
        assert 0 <= metrics['top_3_accuracy'] <= 1

        # Check confusion matrix shape
        assert metrics['confusion_matrix'].shape == (8, 8)

        # Check per-regime metrics structure
        assert len(metrics['per_regime_metrics']) == 8

    def test_top_k_accuracy(self, sample_predictions):
        """Test top-k accuracy calculation."""
        _, probabilities, labels, _ = sample_predictions

        top_1_acc = calculate_top_k_accuracy(probabilities, labels, k=1)
        top_2_acc = calculate_top_k_accuracy(probabilities, labels, k=2)
        top_8_acc = calculate_top_k_accuracy(probabilities, labels, k=8)

        # Top-k accuracy should increase with k
        assert top_1_acc <= top_2_acc
        assert top_2_acc <= top_8_acc
        assert top_8_acc == 1.0  # All regimes covered

        # Values should be in valid range
        assert 0 <= top_1_acc <= 1
        assert 0 <= top_2_acc <= 1

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        n_samples = 50
        labels = np.random.randint(0, 8, n_samples)
        predictions = labels.copy()  # Perfect predictions

        # Create perfect probabilities
        probabilities = np.zeros((n_samples, 8))
        probabilities[np.arange(n_samples), labels] = 1.0

        metrics = calculate_regime_metrics(predictions, probabilities, labels)

        assert metrics['overall_accuracy'] == 1.0
        assert metrics['balanced_accuracy'] == 1.0
        assert metrics['top_2_accuracy'] == 1.0

    def test_random_predictions(self):
        """Test metrics with random predictions."""
        n_samples = 1000  # Large sample for stable statistics
        np.random.seed(42)

        labels = np.random.randint(0, 8, n_samples)
        predictions = np.random.randint(0, 8, n_samples)
        probabilities = np.random.dirichlet(np.ones(8), n_samples)

        metrics = calculate_regime_metrics(predictions, probabilities, labels)

        # Random predictions should have ~12.5% accuracy
        assert 0.05 <= metrics['overall_accuracy'] <= 0.25


class TestCalibrationMetrics:
    """Test confidence calibration metrics."""

    def test_calculate_calibration_metrics_basic(self, sample_predictions):
        """Test basic calibration metrics calculation."""
        predictions, probabilities, labels, confidence_scores = sample_predictions

        calibration = calculate_calibration_metrics(
            probabilities, predictions, labels, confidence_scores
        )

        # Check required metrics exist
        assert 'expected_calibration_error' in calibration
        assert 'maximum_calibration_error' in calibration
        assert 'reliability_diagram' in calibration
        assert 'confidence_statistics' in calibration
        assert 'confidence_accuracy_correlation' in calibration
        assert 'confidence_by_correctness' in calibration

        # Check ECE and MCE ranges
        assert 0 <= calibration['expected_calibration_error'] <= 1
        assert 0 <= calibration['maximum_calibration_error'] <= 1

        # Check confidence statistics
        conf_stats = calibration['confidence_statistics']
        assert 0 <= conf_stats['mean_confidence'] <= 1
        assert conf_stats['std_confidence'] >= 0
        assert 0 <= conf_stats['min_confidence'] <= 1
        assert 0 <= conf_stats['max_confidence'] <= 1

    def test_expected_calibration_error(self, sample_predictions):
        """Test ECE calculation."""
        predictions, probabilities, labels, _ = sample_predictions

        # Use max probability as confidence
        max_probs = np.max(probabilities, axis=1)
        correct_predictions = (predictions == labels).astype(float)

        ece = calculate_expected_calibration_error(max_probs, correct_predictions)

        assert 0 <= ece <= 1
        assert isinstance(ece, float)

    def test_maximum_calibration_error(self, sample_predictions):
        """Test MCE calculation."""
        predictions, probabilities, labels, _ = sample_predictions

        max_probs = np.max(probabilities, axis=1)
        correct_predictions = (predictions == labels).astype(float)

        mce = calculate_maximum_calibration_error(max_probs, correct_predictions)

        assert 0 <= mce <= 1
        assert isinstance(mce, float)

    def test_confidence_accuracy_correlation(self, sample_predictions):
        """Test confidence-accuracy correlation."""
        _, _, _, confidence_scores = sample_predictions

        # Create artificial correct/incorrect predictions
        correct_predictions = np.random.binomial(1, confidence_scores)

        corr = calculate_confidence_accuracy_correlation(confidence_scores, correct_predictions)

        assert -1 <= corr <= 1
        assert isinstance(corr, float)

    def test_perfect_calibration(self):
        """Test calibration with perfectly calibrated model."""
        n_samples = 1000
        np.random.seed(42)

        # Create perfectly calibrated predictions
        confidence_scores = np.random.uniform(0, 1, n_samples)
        correct_predictions = np.random.binomial(1, confidence_scores)

        ece = calculate_expected_calibration_error(confidence_scores, correct_predictions.astype(float))

        # Should be close to 0 for perfectly calibrated model
        assert ece < 0.1  # Allow some random variation


class TestTemporalStabilityMetrics:
    """Test temporal stability metrics."""

    def test_calculate_temporal_stability_basic(self, sample_temporal_data):
        """Test basic temporal stability calculation."""
        predictions, probabilities, dates = sample_temporal_data

        temporal = calculate_temporal_stability_metrics(predictions, probabilities, dates)

        # Check required metrics exist
        assert 'prediction_stability' in temporal
        assert 'prediction_changes' in temporal
        assert 'window_analysis' in temporal
        assert 'transition_matrix' in temporal
        assert 'regime_persistence' in temporal

        # Check stability ranges
        assert 0 <= temporal['prediction_stability'] <= 1
        assert temporal['prediction_changes'] >= 0

        # Check transition matrix shape
        assert temporal['transition_matrix'].shape == (8, 8)

    def test_regime_transition_matrix(self, sample_temporal_data):
        """Test regime transition matrix calculation."""
        predictions, _, _ = sample_temporal_data

        transition_matrix = calculate_regime_transition_matrix(predictions)

        assert transition_matrix.shape == (8, 8)

        # Each row should sum to approximately 1 (or 0 if regime never occurs)
        row_sums = np.sum(transition_matrix, axis=1)
        non_zero_rows = row_sums > 0
        assert np.allclose(row_sums[non_zero_rows], 1.0, atol=1e-10)

    def test_regime_persistence(self, sample_temporal_data):
        """Test regime persistence calculation."""
        predictions, _, _ = sample_temporal_data

        persistence = calculate_regime_persistence(predictions)

        # Should have statistics for all regimes
        assert len(persistence) == 8

        # Check structure of each regime's statistics
        for regime_stats in persistence.values():
            assert 'mean_length' in regime_stats
            assert 'median_length' in regime_stats
            assert 'max_length' in regime_stats
            assert 'std_length' in regime_stats
            assert 'occurrences' in regime_stats

            # All values should be non-negative
            assert regime_stats['mean_length'] >= 0
            assert regime_stats['median_length'] >= 0
            assert regime_stats['max_length'] >= 0
            assert regime_stats['std_length'] >= 0
            assert regime_stats['occurrences'] >= 0

    def test_insufficient_temporal_data(self):
        """Test handling of insufficient temporal data."""
        predictions = np.array([0, 1, 2])  # Too few samples
        probabilities = np.random.dirichlet(np.ones(8), 3)
        dates = pd.date_range('2023-01-01', periods=3)

        temporal = calculate_temporal_stability_metrics(predictions, probabilities, dates)

        assert 'insufficient_data' in temporal
        assert temporal['insufficient_data'] is True


class TestTrainingProgressAnalysis:
    """Test training progress analysis."""

    def test_track_training_progress_basic(self, sample_training_history):
        """Test basic training progress analysis."""
        progress = track_training_progress(sample_training_history)

        # Check required sections exist
        assert 'training_loss' in progress
        assert 'validation_loss' in progress
        assert 'validation_accuracy' in progress
        assert 'learning_rate' in progress
        assert 'training_health' in progress

        # Check training loss analysis
        train_analysis = progress['training_loss']
        assert 'initial_loss' in train_analysis
        assert 'final_loss' in train_analysis
        assert 'improvement' in train_analysis
        assert 'improvement_pct' in train_analysis
        assert 'convergence_rate' in train_analysis

    def test_convergence_rate_calculation(self, sample_training_history):
        """Test convergence rate calculation."""
        loss_values = np.array(sample_training_history['train_loss'])

        conv_rate = calculate_convergence_rate(loss_values)

        assert conv_rate >= 0  # Should be non-negative
        assert isinstance(conv_rate, float)

    def test_loss_smoothness_calculation(self, sample_training_history):
        """Test loss smoothness calculation."""
        loss_values = np.array(sample_training_history['train_loss'])

        smoothness = calculate_loss_smoothness(loss_values)

        assert smoothness >= 0
        assert isinstance(smoothness, float)

    def test_overfitting_score_calculation(self, sample_training_history):
        """Test overfitting score calculation."""
        train_loss = np.array(sample_training_history['train_loss'])
        val_loss = np.array(sample_training_history['val_loss'])

        overfitting = calculate_overfitting_score(train_loss, val_loss)

        assert isinstance(overfitting, float)

    def test_training_health_assessment(self, sample_training_history):
        """Test training health assessment."""
        health = assess_training_health(sample_training_history)

        assert 'health_score' in health
        assert 'status' in health
        assert 'issues' in health
        assert 'recommendations' in health

        assert 0 <= health['health_score'] <= 1
        assert health['status'] in ['healthy', 'concerning', 'problematic']
        assert isinstance(health['issues'], list)
        assert isinstance(health['recommendations'], list)

    def test_empty_training_history(self):
        """Test handling of empty training history."""
        empty_history = {}

        progress = track_training_progress(empty_history)

        assert 'insufficient_data' in progress
        assert progress['insufficient_data'] is True

    def test_minimal_training_history(self):
        """Test handling of minimal training history."""
        minimal_history = {
            'train_loss': [1.0],
            'val_loss': [1.1],
            'val_accuracy': [0.5]
        }

        progress = track_training_progress(minimal_history)

        # Should handle gracefully without crashing
        assert isinstance(progress, dict)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_predictions(self):
        """Test handling of empty prediction arrays."""
        empty_predictions = np.array([])
        empty_probabilities = np.empty((0, 8))
        empty_labels = np.array([])
        empty_confidence = np.array([])

        # Should handle gracefully or raise appropriate errors
        try:
            metrics = calculate_regime_metrics(empty_predictions, empty_probabilities, empty_labels)
        except (ValueError, ZeroDivisionError):
            pass  # Expected for empty data

    def test_single_class_predictions(self):
        """Test handling when all predictions are the same class."""
        n_samples = 50
        predictions = np.zeros(n_samples)  # All class 0
        probabilities = np.zeros((n_samples, 8))
        probabilities[:, 0] = 1.0  # All probability on class 0
        labels = np.random.randint(0, 8, n_samples)  # Mixed labels

        metrics = calculate_regime_metrics(predictions, probabilities, labels)

        # Should handle gracefully
        assert isinstance(metrics, dict)
        assert 'overall_accuracy' in metrics

    def test_extreme_confidence_values(self):
        """Test handling of extreme confidence values."""
        n_samples = 50
        predictions = np.random.randint(0, 8, n_samples)
        probabilities = np.random.dirichlet(np.ones(8), n_samples)
        labels = np.random.randint(0, 8, n_samples)

        # Test with all zeros and all ones
        zero_confidence = np.zeros(n_samples)
        ones_confidence = np.ones(n_samples)

        for conf in [zero_confidence, ones_confidence]:
            calibration = calculate_calibration_metrics(
                probabilities, predictions, labels, conf
            )
            assert isinstance(calibration, dict)

    def test_nan_handling(self):
        """Test handling of NaN values in inputs."""
        n_samples = 50
        predictions = np.random.randint(0, 8, n_samples)
        probabilities = np.random.dirichlet(np.ones(8), n_samples)
        labels = np.random.randint(0, 8, n_samples)
        confidence = np.random.rand(n_samples)

        # Introduce some NaN values
        confidence[::10] = np.nan

        # Should handle or raise appropriate errors
        try:
            calibration = calculate_calibration_metrics(
                probabilities, predictions, labels, confidence
            )
        except ValueError:
            pass  # Expected for NaN values

    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths."""
        predictions = np.array([0, 1, 2])
        probabilities = np.random.dirichlet(np.ones(8), 5)  # Different length
        labels = np.array([1, 2])  # Different length
        confidence = np.array([0.5, 0.6, 0.7, 0.8])  # Different length

        # Should raise appropriate errors
        with pytest.raises((ValueError, IndexError)):
            calculate_regime_metrics(predictions, probabilities, labels)