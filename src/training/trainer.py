"""
Comprehensive training infrastructure for regime detection neural network.

Implements RegimeTrainer with combined loss functions, validation metrics,
early stopping, learning rate scheduling, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime
import json
import time

from ..config import TrainingConfig, config
from ..models.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class RegimeTrainer:
    """
    Comprehensive trainer for regime detection neural networks.

    Features:
    - Combined classification + confidence regression loss
    - Mixed precision training for efficiency
    - Early stopping with configurable patience
    - Learning rate scheduling with ReduceLROnPlateau
    - Model checkpointing with state management
    - Comprehensive validation metrics tracking
    - Device management with automatic CUDA detection
    """

    def __init__(self, model: RegimeDetector, training_config: Optional[TrainingConfig] = None):
        """
        Initialize the regime trainer.

        Args:
            model: RegimeDetector neural network
            training_config: Training configuration (uses global config if None)
        """
        self.model = model
        self.config = training_config or config.training

        # Device setup
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.MSELoss()

        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            min_lr=self.config.lr_scheduler_min_lr,
            verbose=True
        )

        # Mixed precision training
        self.scaler = GradScaler() if self.config.mixed_precision and self.device.type == 'cuda' else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0

        # Metrics tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_classification_acc': [],
            'val_confidence_mae': [],
            'learning_rate': []
        }

        logger.info(f"RegimeTrainer initialized on {self.device}")
        logger.info(f"Mixed precision training: {'enabled' if self.scaler is not None else 'disabled'}")

    def _setup_device(self) -> torch.device:
        """Setup computing device based on configuration."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        if device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")
        else:
            logger.info("Using CPU")

        return device

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor,
                    confidence_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined classification and confidence loss.

        Args:
            outputs: Model outputs (batch_size, 9) - 8 regime probs + 1 confidence
            targets: Regime labels (batch_size,)
            confidence_targets: Target confidence scores (batch_size, 1) - if None, uses max prob as target

        Returns:
            Dictionary with loss components and total loss
        """
        # Split outputs
        regime_probs = outputs[:, :8]  # First 8 are regime probabilities
        predicted_confidence = outputs[:, 8:9]  # Last 1 is confidence score

        # Classification loss
        classification_loss = self.classification_loss(regime_probs, targets)

        # Confidence loss - if no targets provided, use max probability as pseudo-target
        if confidence_targets is None:
            # Use max probability as confidence target (unsupervised confidence learning)
            confidence_targets = torch.max(regime_probs, dim=1)[0].unsqueeze(1).detach()

        confidence_loss = self.confidence_loss(predicted_confidence, confidence_targets)

        # Combined loss
        total_loss = (
            self.config.loss_weights['classification'] * classification_loss +
            self.config.loss_weights['confidence'] * confidence_loss
        )

        return {
            'total': total_loss,
            'classification': classification_loss,
            'confidence': confidence_loss
        }

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Tuple of (features, labels)

        Returns:
            Dictionary with loss components
        """
        self.model.train()
        features, labels = batch
        features, labels = features.to(self.device), labels.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.model(features)
                losses = self.compute_loss(outputs, labels)
                loss = losses['total']

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.gradient_clip_threshold > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_threshold
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            outputs = self.model(features)
            losses = self.compute_loss(outputs, labels)
            loss = losses['total']

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_threshold > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_threshold
                )

            # Optimizer step
            self.optimizer.step()

        # Return loss components as floats
        return {k: v.item() for k, v in losses.items()}

    def validate_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute a single validation step.

        Args:
            batch: Tuple of (features, labels)

        Returns:
            Dictionary with loss components and predictions
        """
        self.model.eval()
        features, labels = batch
        features, labels = features.to(self.device), labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(features)
            losses = self.compute_loss(outputs, labels)

            # Get predictions
            regime_probs = outputs[:, :8]
            confidence_scores = outputs[:, 8:9]
            predicted_regimes = torch.argmax(regime_probs, dim=1)

            # Calculate accuracy
            correct = (predicted_regimes == labels).float()
            accuracy = correct.mean()

            return {
                'losses': {k: v.item() for k, v in losses.items()},
                'accuracy': accuracy.item(),
                'predictions': predicted_regimes.cpu().numpy(),
                'regime_probs': regime_probs.cpu().numpy(),
                'confidence_scores': confidence_scores.cpu().numpy(),
                'labels': labels.cpu().numpy()
            }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with epoch training metrics
        """
        epoch_losses = {'total': [], 'classification': [], 'confidence': []}

        for batch_idx, batch in enumerate(train_loader):
            losses = self.train_step(batch)

            for key in epoch_losses:
                epoch_losses[key].append(losses[key])

            # Log progress periodically
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}: Loss = {losses['total']:.4f}")

        # Calculate epoch averages
        epoch_metrics = {key: np.mean(values) for key, values in epoch_losses.items()}

        return epoch_metrics

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with epoch validation metrics
        """
        epoch_losses = {'total': [], 'classification': [], 'confidence': []}
        epoch_accuracies = []
        all_predictions = []
        all_regime_probs = []
        all_confidence_scores = []
        all_labels = []

        for batch in val_loader:
            val_results = self.validate_step(batch)

            # Collect losses
            for key in epoch_losses:
                epoch_losses[key].append(val_results['losses'][key])

            # Collect metrics
            epoch_accuracies.append(val_results['accuracy'])
            all_predictions.extend(val_results['predictions'])
            all_regime_probs.append(val_results['regime_probs'])
            all_confidence_scores.append(val_results['confidence_scores'])
            all_labels.extend(val_results['labels'])

        # Calculate epoch metrics
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        avg_accuracy = np.mean(epoch_accuracies)

        # Combine predictions for detailed analysis
        all_regime_probs = np.vstack(all_regime_probs)
        all_confidence_scores = np.vstack(all_confidence_scores)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate classification accuracy and confidence MAE
        classification_acc = np.mean(all_predictions == all_labels)
        confidence_mae = np.mean(np.abs(all_confidence_scores - np.max(all_regime_probs, axis=1, keepdims=True)))

        return {
            'losses': avg_losses,
            'accuracy': avg_accuracy,
            'classification_accuracy': classification_acc,
            'confidence_mae': confidence_mae,
            'predictions': all_predictions,
            'regime_probs': all_regime_probs,
            'confidence_scores': all_confidence_scores,
            'labels': all_labels
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model for specified epochs with validation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses config if None)

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.max_epochs
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase
            if epoch % self.config.validation_frequency == 0:
                val_metrics = self.validate_epoch(val_loader)

                # Update learning rate scheduler
                self.scheduler.step(val_metrics['losses']['total'])

                # Track metrics
                self.training_history['train_loss'].append(train_metrics['total'])
                self.training_history['val_loss'].append(val_metrics['losses']['total'])
                self.training_history['val_accuracy'].append(val_metrics['accuracy'])
                self.training_history['val_classification_acc'].append(val_metrics['classification_accuracy'])
                self.training_history['val_confidence_mae'].append(val_metrics['confidence_mae'])
                self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

                # Check for improvement
                improved = val_metrics['accuracy'] > self.best_val_accuracy
                if improved:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.best_val_loss = val_metrics['losses']['total']
                    self.epochs_without_improvement = 0

                    # Save best model
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)
                else:
                    self.epochs_without_improvement += 1

                # Logging
                epoch_time = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): "
                    f"Train Loss: {train_metrics['total']:.4f}, "
                    f"Val Loss: {val_metrics['losses']['total']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

                # Early stopping check
                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Periodic checkpointing
            if epoch % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_metrics if epoch % self.config.validation_frequency == 0 else None)

        logger.info("Training completed")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")

        return self.training_history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model checkpoint with full training state.

        Args:
            filename: Checkpoint filename
            epoch: Current epoch number
            metrics: Optional validation metrics
        """
        checkpoint_path = self.config.checkpoint_dir / filename

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'num_regimes': self.model.num_regimes,
                'dropout_rate': self.model.dropout_rate
            },
            'timestamp': datetime.now().isoformat(),
        }

        if metrics:
            checkpoint_data['validation_metrics'] = metrics

        if self.scaler:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint_data, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, filename: str, load_optimizer: bool = True) -> None:
        """
        Load model checkpoint and restore training state.

        Args:
            filename: Checkpoint filename
            load_optimizer: Whether to restore optimizer state
        """
        checkpoint_path = self.config.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Validate model compatibility
        model_config = checkpoint_data.get('config', {})
        if model_config.get('input_dim') != self.model.input_dim:
            raise ValueError(f"Model input dimension mismatch: expected {self.model.input_dim}, got {model_config.get('input_dim')}")

        # Load model state
        self.model.load_state_dict(checkpoint_data['model_state_dict'])

        # Load training state
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            if self.scaler and 'scaler_state_dict' in checkpoint_data:
                self.scaler.load_state_dict(checkpoint_data['scaler_state_dict'])

        # Restore trainer state
        self.current_epoch = checkpoint_data['epoch']
        self.training_history = checkpoint_data.get('training_history', self.training_history)
        self.best_val_accuracy = checkpoint_data.get('best_val_accuracy', 0.0)
        self.best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
        self.epochs_without_improvement = checkpoint_data.get('epochs_without_improvement', 0)

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Restored to epoch {self.current_epoch}, best val accuracy: {self.best_val_accuracy:.4f}")

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the most recent ones."""
        if not self.config.checkpoint_dir.exists():
            return

        # Get all checkpoint files
        checkpoint_files = list(self.config.checkpoint_dir.glob('checkpoint_epoch_*.pth'))

        # Keep only the most recent checkpoints
        if len(checkpoint_files) > self.config.max_checkpoints:
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoint_files[:-self.config.max_checkpoints]:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary with parameter counts and configuration."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'architecture': f"{self.model.input_dim} → {' → '.join(map(str, self.model.hidden_dims))} → {self.model.num_regimes}+1",
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': str(self.device),
            'mixed_precision': self.scaler is not None,
            'current_epoch': self.current_epoch,
            'best_val_accuracy': self.best_val_accuracy
        }