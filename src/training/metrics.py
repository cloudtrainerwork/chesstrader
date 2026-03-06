"""
Comprehensive validation metrics and monitoring system for regime detection.

Provides detailed analysis of model performance including regime classification
accuracy, confidence calibration, temporal stability, and training progress tracking.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    balanced_accuracy_score, classification_report
)
from sklearn.calibration import calibration_curve
import logging

# Import RegimeType enum for regime name mapping
try:
    from ..data.regime_labeler import RegimeType
except ImportError:
    # Fallback enum if import fails
    from enum import IntEnum
    class RegimeType(IntEnum):
        BULL_TRENDING = 0
        BEAR_TRENDING = 1
        HIGH_VOLATILITY = 2
        LOW_VOLATILITY = 3
        SIDEWAYS_RANGING = 4
        RECOVERY = 5
        DISTRIBUTION = 6
        CRISIS = 7

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Lightweight metrics calculator for training loops.

    Provides simple performance metrics for PPO trainers and position manager.
    """

    def __init__(self,
                 track_sharpe: bool = True,
                 track_drawdown: bool = True,
                 track_win_rate: bool = True):
        self.track_sharpe = track_sharpe
        self.track_drawdown = track_drawdown
        self.track_win_rate = track_win_rate

    def calculate(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic performance metrics from episode returns."""
        if returns is None or len(returns) == 0:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}

        returns = np.asarray(returns, dtype=float)
        metrics = {'total_return': float(np.mean(returns))}

        if self.track_sharpe:
            std = float(np.std(returns))
            metrics['sharpe_ratio'] = float(np.mean(returns) / std) if std > 0 else 0.0

        if self.track_drawdown:
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
            metrics['max_drawdown'] = max_drawdown

        if self.track_win_rate:
            metrics['win_rate'] = float(np.mean(returns > 0))

        return metrics


def calculate_regime_metrics(predictions: np.ndarray, probabilities: np.ndarray,
                           labels: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive regime classification metrics.

    Args:
        predictions: Predicted regime indices (N,)
        probabilities: Regime probabilities (N, 8)
        labels: True regime labels (N,)

    Returns:
        Dictionary with classification metrics
    """
    # Basic accuracy metrics
    overall_accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)

    # Per-regime precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Weighted averages
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    # Confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)

    # Per-regime metrics dictionary
    regime_metrics = {}
    for i in range(8):
        regime_name = RegimeType(i).name
        regime_metrics[regime_name] = {
            'precision': precision[i] if i < len(precision) else 0.0,
            'recall': recall[i] if i < len(recall) else 0.0,
            'f1_score': f1[i] if i < len(f1) else 0.0,
            'support': support[i] if i < len(support) else 0,
            'accuracy': conf_matrix[i, i] / support[i] if i < len(support) and support[i] > 0 else 0.0
        }

    # Top-k accuracy (regime is in top k predictions)
    top_2_accuracy = calculate_top_k_accuracy(probabilities, labels, k=2)
    top_3_accuracy = calculate_top_k_accuracy(probabilities, labels, k=3)

    return {
        'overall_accuracy': overall_accuracy,
        'balanced_accuracy': balanced_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'top_2_accuracy': top_2_accuracy,
        'top_3_accuracy': top_3_accuracy,
        'confusion_matrix': conf_matrix,
        'per_regime_metrics': regime_metrics,
        'class_distribution': dict(zip(*np.unique(labels, return_counts=True)))
    }


def calculate_top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Calculate top-k accuracy for regime predictions."""
    if k >= probabilities.shape[1]:
        return 1.0

    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels[:, np.newaxis], axis=1)
    return np.mean(correct)


def calculate_calibration_metrics(probabilities: np.ndarray, predictions: np.ndarray,
                                labels: np.ndarray, confidence_scores: np.ndarray,
                                n_bins: int = 10) -> Dict[str, Any]:
    """
    Calculate confidence calibration metrics.

    Args:
        probabilities: Regime probabilities (N, 8)
        predictions: Predicted regime indices (N,)
        labels: True regime labels (N,)
        confidence_scores: Model confidence scores (N,)
        n_bins: Number of bins for calibration analysis

    Returns:
        Dictionary with calibration metrics
    """
    # Extract confidence values (max probability for each prediction)
    max_probs = np.max(probabilities, axis=1)

    # Binary correctness
    correct_predictions = (predictions == labels).astype(float)

    # Reliability diagram data (calibration curve)
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            correct_predictions, max_probs, n_bins=n_bins, strategy='uniform'
        )
    except ValueError:
        # Handle edge cases
        fraction_of_positives = np.array([])
        mean_predicted_value = np.array([])

    # Expected Calibration Error (ECE)
    ece = calculate_expected_calibration_error(max_probs, correct_predictions, n_bins)

    # Maximum Calibration Error (MCE)
    mce = calculate_maximum_calibration_error(max_probs, correct_predictions, n_bins)

    # Confidence statistics
    confidence_stats = {
        'mean_confidence': np.mean(confidence_scores),
        'std_confidence': np.std(confidence_scores),
        'min_confidence': np.min(confidence_scores),
        'max_confidence': np.max(confidence_scores),
        'median_confidence': np.median(confidence_scores)
    }

    # Confidence vs accuracy correlation
    confidence_accuracy_corr = calculate_confidence_accuracy_correlation(
        confidence_scores, correct_predictions
    )

    # Confidence distribution by correctness
    correct_mask = correct_predictions == 1
    incorrect_mask = correct_predictions == 0

    confidence_correct = confidence_scores[correct_mask] if np.any(correct_mask) else np.array([])
    confidence_incorrect = confidence_scores[incorrect_mask] if np.any(incorrect_mask) else np.array([])

    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce,
        'reliability_diagram': {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        },
        'confidence_statistics': confidence_stats,
        'confidence_accuracy_correlation': confidence_accuracy_corr,
        'confidence_by_correctness': {
            'correct_mean': np.mean(confidence_correct) if len(confidence_correct) > 0 else 0.0,
            'correct_std': np.std(confidence_correct) if len(confidence_correct) > 0 else 0.0,
            'incorrect_mean': np.mean(confidence_incorrect) if len(confidence_incorrect) > 0 else 0.0,
            'incorrect_std': np.std(confidence_incorrect) if len(confidence_incorrect) > 0 else 0.0
        }
    }


def calculate_expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                                       n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calculate_maximum_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                                      n_bins: int = 10) -> float:
    """Calculate Maximum Calibration Error (MCE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

    return mce


def calculate_confidence_accuracy_correlation(confidence_scores: np.ndarray,
                                            correct_predictions: np.ndarray) -> float:
    """Calculate correlation between confidence scores and prediction correctness."""
    try:
        correlation = np.corrcoef(confidence_scores.flatten(), correct_predictions)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except:
        return 0.0


def calculate_temporal_stability_metrics(predictions: np.ndarray, probabilities: np.ndarray,
                                       dates: pd.DatetimeIndex,
                                       window_size: int = 30) -> Dict[str, Any]:
    """
    Calculate temporal stability metrics for regime predictions.

    Args:
        predictions: Predicted regime indices over time (N,)
        probabilities: Regime probabilities over time (N, 8)
        dates: Corresponding dates
        window_size: Window size for stability analysis

    Returns:
        Dictionary with temporal stability metrics
    """
    if len(predictions) < window_size:
        return {'insufficient_data': True, 'message': f'Need at least {window_size} samples'}

    # Prediction smoothness (how often predictions change)
    prediction_changes = np.sum(np.diff(predictions) != 0)
    prediction_stability = 1 - (prediction_changes / (len(predictions) - 1))

    # Rolling window analysis
    window_metrics = []
    for i in range(len(predictions) - window_size + 1):
        window_preds = predictions[i:i + window_size]
        window_probs = probabilities[i:i + window_size]

        # Dominant regime in window
        dominant_regime = np.bincount(window_preds).argmax()
        dominant_regime_pct = np.mean(window_preds == dominant_regime)

        # Average confidence in window
        max_probs = np.max(window_probs, axis=1)
        avg_confidence = np.mean(max_probs)

        # Regime diversity (number of unique regimes)
        unique_regimes = len(np.unique(window_preds))

        window_metrics.append({
            'start_date': dates[i],
            'end_date': dates[i + window_size - 1],
            'dominant_regime': dominant_regime,
            'dominant_regime_percentage': dominant_regime_pct,
            'average_confidence': avg_confidence,
            'unique_regimes': unique_regimes
        })

    # Aggregate temporal metrics
    window_df = pd.DataFrame(window_metrics)

    # Regime transition analysis
    transition_matrix = calculate_regime_transition_matrix(predictions)

    # Persistence analysis (how long each regime lasts)
    regime_persistence = calculate_regime_persistence(predictions)

    return {
        'prediction_stability': prediction_stability,
        'prediction_changes': int(prediction_changes),
        'window_analysis': {
            'avg_dominant_regime_pct': window_df['dominant_regime_percentage'].mean(),
            'avg_confidence': window_df['average_confidence'].mean(),
            'avg_unique_regimes': window_df['unique_regimes'].mean(),
            'regime_diversity_std': window_df['unique_regimes'].std()
        },
        'transition_matrix': transition_matrix,
        'regime_persistence': regime_persistence,
        'temporal_windows': window_metrics
    }


def calculate_regime_transition_matrix(predictions: np.ndarray) -> np.ndarray:
    """Calculate regime transition probability matrix."""
    n_regimes = 8
    transition_matrix = np.zeros((n_regimes, n_regimes))

    for i in range(len(predictions) - 1):
        current_regime = predictions[i]
        next_regime = predictions[i + 1]
        if 0 <= current_regime < n_regimes and 0 <= next_regime < n_regimes:
            transition_matrix[current_regime, next_regime] += 1

    # Normalize to probabilities
    row_sums = transition_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]

    return transition_matrix


def calculate_regime_persistence(predictions: np.ndarray) -> Dict[str, Any]:
    """Calculate how long each regime persists on average."""
    persistence_data = {i: [] for i in range(8)}

    current_regime = predictions[0] if len(predictions) > 0 else None
    current_length = 1

    for i in range(1, len(predictions)):
        if predictions[i] == current_regime:
            current_length += 1
        else:
            if 0 <= current_regime < 8:
                persistence_data[current_regime].append(current_length)
            current_regime = predictions[i]
            current_length = 1

    # Add final sequence
    if 0 <= current_regime < 8:
        persistence_data[current_regime].append(current_length)

    # Calculate statistics
    persistence_stats = {}
    for regime in range(8):
        regime_name = RegimeType(regime).name
        lengths = persistence_data[regime]
        if lengths:
            persistence_stats[regime_name] = {
                'mean_length': np.mean(lengths),
                'median_length': np.median(lengths),
                'max_length': np.max(lengths),
                'std_length': np.std(lengths),
                'occurrences': len(lengths)
            }
        else:
            persistence_stats[regime_name] = {
                'mean_length': 0,
                'median_length': 0,
                'max_length': 0,
                'std_length': 0,
                'occurrences': 0
            }

    return persistence_stats


def track_training_progress(training_history: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Analyze training progress and convergence patterns.

    Args:
        training_history: Dictionary with training metrics over epochs

    Returns:
        Dictionary with training progress analysis
    """
    if not training_history or not training_history.get('train_loss'):
        return {'insufficient_data': True}

    # Extract metrics
    train_loss = np.array(training_history['train_loss'])
    val_loss = np.array(training_history.get('val_loss', []))
    val_accuracy = np.array(training_history.get('val_accuracy', []))
    learning_rates = np.array(training_history.get('learning_rate', []))

    progress_metrics = {}

    # Training loss analysis
    if len(train_loss) > 1:
        progress_metrics['training_loss'] = {
            'initial_loss': float(train_loss[0]),
            'final_loss': float(train_loss[-1]),
            'improvement': float(train_loss[0] - train_loss[-1]),
            'improvement_pct': float((train_loss[0] - train_loss[-1]) / train_loss[0] * 100),
            'convergence_rate': calculate_convergence_rate(train_loss),
            'smoothness': calculate_loss_smoothness(train_loss)
        }

    # Validation analysis
    if len(val_loss) > 1:
        progress_metrics['validation_loss'] = {
            'initial_loss': float(val_loss[0]),
            'final_loss': float(val_loss[-1]),
            'best_loss': float(np.min(val_loss)),
            'best_epoch': int(np.argmin(val_loss)),
            'overfitting_score': calculate_overfitting_score(train_loss, val_loss)
        }

    if len(val_accuracy) > 1:
        progress_metrics['validation_accuracy'] = {
            'initial_accuracy': float(val_accuracy[0]),
            'final_accuracy': float(val_accuracy[-1]),
            'best_accuracy': float(np.max(val_accuracy)),
            'best_epoch': int(np.argmax(val_accuracy)),
            'stability': calculate_metric_stability(val_accuracy)
        }

    # Learning rate analysis
    if len(learning_rates) > 1:
        lr_reductions = np.sum(np.diff(learning_rates) < 0)
        progress_metrics['learning_rate'] = {
            'initial_lr': float(learning_rates[0]),
            'final_lr': float(learning_rates[-1]),
            'reductions': int(lr_reductions),
            'reduction_factor': float(learning_rates[-1] / learning_rates[0])
        }

    # Overall training health
    progress_metrics['training_health'] = assess_training_health(training_history)

    return progress_metrics


def calculate_convergence_rate(loss_values: np.ndarray, window_size: int = 10) -> float:
    """Calculate the rate of loss convergence."""
    if len(loss_values) < window_size * 2:
        return 0.0

    # Compare early and late windows
    early_window = np.mean(loss_values[:window_size])
    late_window = np.mean(loss_values[-window_size:])

    # Rate of improvement per epoch
    epochs_elapsed = len(loss_values) - window_size
    convergence_rate = (early_window - late_window) / epochs_elapsed

    return max(0.0, convergence_rate)  # Non-negative rate


def calculate_loss_smoothness(loss_values: np.ndarray) -> float:
    """Calculate how smooth the loss curve is (lower is smoother)."""
    if len(loss_values) < 3:
        return 0.0

    # Calculate second derivative (curvature)
    second_derivative = np.diff(loss_values, n=2)
    smoothness = np.std(second_derivative)

    return smoothness


def calculate_overfitting_score(train_loss: np.ndarray, val_loss: np.ndarray) -> float:
    """Calculate overfitting score (positive means overfitting)."""
    min_len = min(len(train_loss), len(val_loss))
    if min_len < 2:
        return 0.0

    # Use last portion of training
    window_size = max(1, min_len // 4)
    recent_train = np.mean(train_loss[-window_size:])
    recent_val = np.mean(val_loss[-window_size:])

    # Overfitting score: positive means val loss > train loss
    overfitting_score = (recent_val - recent_train) / recent_train

    return overfitting_score


def calculate_metric_stability(metric_values: np.ndarray, window_size: int = 10) -> float:
    """Calculate stability of a metric (lower standard deviation is more stable)."""
    if len(metric_values) < window_size:
        return np.std(metric_values) if len(metric_values) > 1 else 0.0

    # Use recent window for stability assessment
    recent_values = metric_values[-window_size:]
    stability = 1.0 / (1.0 + np.std(recent_values))  # Higher is more stable

    return stability


def assess_training_health(training_history: Dict[str, List[float]]) -> Dict[str, Any]:
    """Assess overall health of training process."""
    health_score = 1.0
    issues = []
    recommendations = []

    train_loss = training_history.get('train_loss', [])
    val_loss = training_history.get('val_loss', [])
    val_accuracy = training_history.get('val_accuracy', [])

    # Check for decreasing training loss
    if len(train_loss) > 5:
        recent_trend = np.polyfit(range(len(train_loss[-10:])), train_loss[-10:], 1)[0]
        if recent_trend > 0:  # Loss increasing
            health_score *= 0.8
            issues.append("Training loss not decreasing")
            recommendations.append("Consider reducing learning rate or checking data quality")

    # Check for overfitting
    if len(train_loss) >= len(val_loss) >= 5:
        overfitting_score = calculate_overfitting_score(np.array(train_loss), np.array(val_loss))
        if overfitting_score > 0.2:  # 20% gap
            health_score *= 0.7
            issues.append("Potential overfitting detected")
            recommendations.append("Consider regularization, dropout, or early stopping")

    # Check for validation accuracy improvement
    if len(val_accuracy) > 5:
        accuracy_trend = np.polyfit(range(len(val_accuracy[-10:])), val_accuracy[-10:], 1)[0]
        if accuracy_trend < 0:  # Accuracy decreasing
            health_score *= 0.9
            issues.append("Validation accuracy declining")
            recommendations.append("Consider early stopping or model architecture changes")

    # Check for training stagnation
    if len(val_accuracy) > 10:
        recent_variance = np.var(val_accuracy[-5:])
        if recent_variance < 1e-6:  # Very low variance
            health_score *= 0.9
            issues.append("Training appears to have stagnated")
            recommendations.append("Consider learning rate adjustment or model complexity")

    return {
        'health_score': health_score,
        'status': 'healthy' if health_score > 0.8 else 'concerning' if health_score > 0.6 else 'problematic',
        'issues': issues,
        'recommendations': recommendations
    }
