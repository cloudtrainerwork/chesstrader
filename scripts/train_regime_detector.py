#!/usr/bin/env python3
"""
Training script for regime detection neural network.

Provides end-to-end training pipeline with data loading, model training,
validation, checkpointing, and comprehensive evaluation reporting.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import numpy as np
import pandas as pd

# Optional plotting dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config, TrainingConfig
from src.models.regime_detector import RegimeDetector
from src.training.trainer import RegimeTrainer
from src.training.metrics import (
    calculate_regime_metrics, calculate_calibration_metrics,
    calculate_temporal_stability_metrics, track_training_progress
)
from src.data.training_data import create_data_loaders, TrainingDataset
from src.models.model_utils import export_model_for_inference, get_model_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.training.logs_dir / 'training.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_arguments() -> argparse.ArgumentParser:
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train regime detection neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    parser.add_argument('--symbol', type=str, default='SPY',
                      help='Stock symbol to train on')
    parser.add_argument('--years', type=int, default=5,
                      help='Years of historical data to use')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Training batch size (uses config default if not specified)')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=None,
                      help='Maximum number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=None,
                      help='Learning rate')
    parser.add_argument('--patience', type=int, default=None,
                      help='Early stopping patience')

    # Model configuration
    parser.add_argument('--hidden-dims', type=str, default=None,
                      help='Hidden layer dimensions as comma-separated values (e.g., "128,64,32")')
    parser.add_argument('--dropout-rate', type=float, default=None,
                      help='Dropout rate for regularization')

    # Device and optimization
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cuda', 'cpu'],
                      help='Device to use for training')
    parser.add_argument('--no-mixed-precision', action='store_true',
                      help='Disable mixed precision training')

    # Checkpointing and output
    parser.add_argument('--resume-from', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for models and results')
    parser.add_argument('--save-best-only', action='store_true',
                      help='Only save the best model (not periodic checkpoints)')

    # Evaluation and visualization
    parser.add_argument('--skip-final-eval', action='store_true',
                      help='Skip comprehensive final evaluation')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save training progress and evaluation plots')
    parser.add_argument('--export-inference', action='store_true',
                      help='Export best model for inference deployment')

    # Validation and testing
    parser.add_argument('--validation-split', type=float, default=0.15,
                      help='Validation set ratio')
    parser.add_argument('--test-split', type=float, default=0.15,
                      help='Test set ratio')

    # Logging and debugging
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')

    return parser


def setup_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from arguments and defaults."""
    training_config = TrainingConfig()

    # Override with command line arguments if provided
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.epochs is not None:
        training_config.max_epochs = args.epochs
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.patience is not None:
        training_config.patience = args.patience

    # Device configuration
    if args.device != 'auto':
        training_config.device = args.device

    # Mixed precision
    if args.no_mixed_precision:
        training_config.mixed_precision = False

    # Output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
        training_config.checkpoint_dir = output_path / 'checkpoints'
        training_config.logs_dir = output_path / 'logs'

    return training_config


def setup_model(args: argparse.Namespace) -> RegimeDetector:
    """Create and configure the model."""
    # Parse hidden dimensions if provided
    hidden_dims = (128, 64, 32)  # Default
    if args.hidden_dims:
        try:
            hidden_dims = tuple(map(int, args.hidden_dims.split(',')))
        except ValueError:
            logger.warning(f"Invalid hidden dims format: {args.hidden_dims}, using default")

    # Create model
    model = RegimeDetector(
        input_dim=48,
        hidden_dims=hidden_dims,
        num_regimes=8,
        dropout_rate=args.dropout_rate if args.dropout_rate is not None else 0.2
    )

    logger.info(f"Created model: {model}")
    return model


def setup_reproducibility(seed: int) -> None:
    """Setup random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def create_training_summary(trainer: RegimeTrainer, final_metrics: Dict[str, Any],
                          training_time: float, args: argparse.Namespace) -> Dict[str, Any]:
    """Create comprehensive training summary."""
    model_summary = trainer.get_model_summary()

    summary = {
        'training_info': {
            'symbol': args.symbol,
            'years': args.years,
            'epochs_completed': trainer.current_epoch + 1,
            'best_val_accuracy': trainer.best_val_accuracy,
            'best_val_loss': trainer.best_val_loss,
            'training_time_minutes': training_time / 60,
            'final_learning_rate': trainer.optimizer.param_groups[0]['lr']
        },
        'model_info': model_summary,
        'final_metrics': final_metrics,
        'training_history': trainer.training_history,
        'config': {
            'batch_size': trainer.config.batch_size,
            'learning_rate': trainer.config.learning_rate,
            'patience': trainer.config.patience,
            'mixed_precision': trainer.config.mixed_precision,
            'device': str(trainer.device)
        },
        'timestamp': datetime.now().isoformat()
    }

    return summary


def save_training_plots(trainer: RegimeTrainer, output_dir: Path) -> None:
    """Save training progress plots."""
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting libraries not available, skipping plot generation")
        return

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    history = trainer.training_history

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy curves
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].plot(history['val_classification_acc'], label='Classification Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning rate
    axes[1, 0].plot(history['learning_rate'])
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)

    # Confidence MAE
    axes[1, 1].plot(history['val_confidence_mae'], color='red')
    axes[1, 1].set_title('Confidence Mean Absolute Error')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Training plots saved to {plots_dir}")


def evaluate_model_comprehensive(trainer: RegimeTrainer, test_loader, output_dir: Path) -> Dict[str, Any]:
    """Perform comprehensive model evaluation."""
    logger.info("Starting comprehensive model evaluation...")

    # Run validation on test set
    test_metrics = trainer.validate_epoch(test_loader)

    # Calculate detailed metrics
    regime_metrics = calculate_regime_metrics(
        test_metrics['predictions'],
        test_metrics['regime_probs'],
        test_metrics['labels']
    )

    calibration_metrics = calculate_calibration_metrics(
        test_metrics['regime_probs'],
        test_metrics['predictions'],
        test_metrics['labels'],
        test_metrics['confidence_scores'].flatten()
    )

    # Create temporal analysis if test data has sufficient samples
    if len(test_metrics['predictions']) > 50:
        # Create mock dates for temporal analysis
        test_dates = pd.date_range('2023-01-01', periods=len(test_metrics['predictions']))
        temporal_metrics = calculate_temporal_stability_metrics(
            test_metrics['predictions'],
            test_metrics['regime_probs'],
            test_dates
        )
    else:
        temporal_metrics = {'insufficient_data': True}

    # Training progress analysis
    progress_metrics = track_training_progress(trainer.training_history)

    comprehensive_metrics = {
        'test_performance': {
            'accuracy': test_metrics['accuracy'],
            'loss': test_metrics['losses']['total'],
            'classification_accuracy': test_metrics['classification_accuracy'],
            'confidence_mae': test_metrics['confidence_mae']
        },
        'regime_analysis': regime_metrics,
        'calibration_analysis': calibration_metrics,
        'temporal_stability': temporal_metrics,
        'training_analysis': progress_metrics
    }

    # Save detailed results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_compatible = convert_numpy_to_json(comprehensive_metrics)
        json.dump(json_compatible, f, indent=2, default=str)

    logger.info(f"Comprehensive evaluation completed. Results saved to {results_file}")
    return comprehensive_metrics


def convert_numpy_to_json(obj):
    """Convert numpy arrays and types to JSON-compatible formats."""
    if isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def main():
    """Main training function."""
    # Parse arguments
    parser = setup_arguments()
    args = parser.parse_args()

    # Setup logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Setup reproducibility
    setup_reproducibility(args.seed)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"training_runs/regime_detector_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Training output directory: {output_dir}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Setup configuration
        training_config = setup_training_config(args)
        training_config.checkpoint_dir = output_dir / 'checkpoints'
        training_config.logs_dir = output_dir / 'logs'
        training_config.checkpoint_dir.mkdir(exist_ok=True)
        training_config.logs_dir.mkdir(exist_ok=True)

        # Create model
        model = setup_model(args)

        # Create trainer
        trainer = RegimeTrainer(model, training_config)

        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming training from {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)

        # Load data
        logger.info(f"Loading training data for {args.symbol} ({args.years} years)...")
        train_loader, val_loader, test_loader = create_data_loaders(
            symbol=args.symbol,
            years=args.years,
            batch_size=training_config.batch_size
        )

        logger.info(f"Data loaded successfully:")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")

        # Start training
        logger.info("Starting training...")
        start_time = time.time()

        training_history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config.max_epochs
        )

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f} seconds")

        # Final evaluation
        if not args.skip_final_eval:
            final_metrics = evaluate_model_comprehensive(trainer, test_loader, output_dir)
        else:
            final_metrics = {'skipped': True}

        # Create training summary
        training_summary = create_training_summary(trainer, final_metrics, training_time, args)

        # Save training summary
        summary_file = output_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(convert_numpy_to_json(training_summary), f, indent=2, default=str)

        # Save training plots
        if args.save_plots:
            save_training_plots(trainer, output_dir)

        # Export inference model
        if args.export_inference:
            inference_path = output_dir / 'inference_model.pth'
            export_model_for_inference(trainer.model, inference_path)
            logger.info(f"Inference model exported to {inference_path}")

        # Final results
        logger.info("=== TRAINING COMPLETED ===")
        logger.info(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Total epochs: {trainer.current_epoch + 1}")
        logger.info(f"Training time: {training_time:.1f}s")
        logger.info(f"Results saved to: {output_dir}")

        # Print final test metrics if available
        if not args.skip_final_eval and 'test_performance' in final_metrics:
            test_perf = final_metrics['test_performance']
            logger.info("=== FINAL TEST RESULTS ===")
            logger.info(f"Test accuracy: {test_perf['accuracy']:.4f}")
            logger.info(f"Test loss: {test_perf['loss']:.4f}")
            logger.info(f"Classification accuracy: {test_perf['classification_accuracy']:.4f}")
            logger.info(f"Confidence MAE: {test_perf['confidence_mae']:.4f}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception("Full traceback:")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())