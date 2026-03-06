"""
Transfer learning trainer for options domain fine-tuning.

Implements TransferTrainer class for fine-tuning chess-inspired neural networks
on options trading tasks with progressive unfreezing and domain-specific loss functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass

from .chess_adapter import ChessWeightAdapter, load_and_adapt_chess_weights


@dataclass
class TransferTrainingConfig:
    """Configuration for transfer learning training."""

    # Learning rates
    base_learning_rate: float = 1e-4
    transferred_lr_factor: float = 0.1  # Reduce LR for transferred weights
    new_layer_lr_factor: float = 1.0  # Full LR for new layers

    # Progressive unfreezing
    freeze_stages: List[str] = None  # Stages to initially freeze
    unfreeze_schedule: Dict[int, List[str]] = None  # epoch -> layers to unfreeze
    unfreeze_after_epochs: int = 5  # Start unfreezing after N epochs

    # Loss function weights
    strategy_accuracy_weight: float = 0.7
    regime_classification_weight: float = 0.3
    regularization_weight: float = 0.01

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0

    # Transfer learning mode
    transfer_mode: str = 'fine_tune'  # 'fine_tune' or 'feature_extraction'

    def __post_init__(self):
        """Set default values after initialization."""
        if self.freeze_stages is None:
            self.freeze_stages = ['spatial_encoder', 'feature_extractor.stages.0']

        if self.unfreeze_schedule is None:
            self.unfreeze_schedule = {
                5: ['feature_extractor.stages.1'],
                10: ['feature_extractor.stages.0'],
                15: ['spatial_encoder']
            }


class TransferTrainer:
    """
    Trainer for chess-to-options transfer learning.

    Handles loading chess weights, progressive unfreezing, and domain-specific
    fine-tuning with combined loss functions for strategy evaluation and regime classification.
    """

    def __init__(self,
                 model: nn.Module,
                 config: TransferTrainingConfig,
                 device: Optional[torch.device] = None):
        """
        Initialize transfer trainer.

        Args:
            model: Target model for fine-tuning
            config: Training configuration
            device: Training device (CPU/GPU)
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []

        # Layer tracking for progressive unfreezing
        self.frozen_layers = set()
        self.transferred_layers = set()
        self.new_layers = set()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def load_chess_weights(self,
                          chess_model_path: Union[str, Path],
                          weight_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Load and adapt chess model weights.

        Args:
            chess_model_path: Path to chess model file
            weight_mapping: Custom mapping between chess and target weights

        Returns:
            Dictionary with loading results and statistics
        """
        self.logger.info(f"Loading chess weights from {chess_model_path}")

        try:
            # Load and adapt weights
            adapted_weights = load_and_adapt_chess_weights(
                chess_model_path,
                self.model,
                target_spatial_dims=(7, 6)
            )

            model_state = self.model.state_dict()
            compatible_weights = {}
            for key, value in adapted_weights.items():
                if key in model_state and model_state[key].shape == value.shape:
                    compatible_weights[key] = value

            # Load weights into model
            load_result = self.model.load_state_dict(compatible_weights, strict=False)

            # Track which layers were transferred
            self.transferred_layers.update(adapted_weights.keys())
            model_keys = set(model_state.keys())
            self.new_layers = model_keys - self.transferred_layers
            if not self.new_layers:
                self.new_layers = {key for key in model_keys if key.startswith("output_head.")}

            self.logger.info(f"Loaded {len(compatible_weights)} chess weight layers")
            self.logger.info(f"Missing keys: {len(load_result.missing_keys)}")
            self.logger.info(f"Unexpected keys: {len(load_result.unexpected_keys)}")

            return {
                'loaded_layers': list(adapted_weights.keys()),
                'missing_keys': load_result.missing_keys,
                'unexpected_keys': load_result.unexpected_keys,
                'transferred_layers': list(self.transferred_layers),
                'new_layers': list(self.new_layers)
            }

        except Exception as e:
            self.logger.error(f"Failed to load chess weights: {e}")
            raise

    def freeze_chess_layers(self, layer_patterns: Optional[List[str]] = None):
        """
        Freeze specified layers for transfer learning.

        Args:
            layer_patterns: List of layer name patterns to freeze
        """
        if layer_patterns is None:
            layer_patterns = self.config.freeze_stages

        frozen_count = 0
        for name, param in self.model.named_parameters():
            should_freeze = any(pattern in name for pattern in layer_patterns)
            if should_freeze:
                param.requires_grad = False
                self.frozen_layers.add(name)
                frozen_count += 1

        self.logger.info(f"Frozen {frozen_count} parameters in {len(layer_patterns)} layer groups")

    def unfreeze_layers(self, layer_patterns: List[str]):
        """
        Unfreeze specified layers.

        Args:
            layer_patterns: List of layer name patterns to unfreeze
        """
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            should_unfreeze = any(pattern in name for pattern in layer_patterns)
            if should_unfreeze and name in self.frozen_layers:
                param.requires_grad = True
                self.frozen_layers.remove(name)
                unfrozen_count += 1

        self.logger.info(f"Unfroze {unfrozen_count} parameters in {len(layer_patterns)} layer groups")

    def fine_tune_schedule(self) -> bool:
        """
        Apply progressive unfreezing schedule.

        Returns:
            True if any layers were unfrozen, False otherwise
        """
        if self.current_epoch in self.config.unfreeze_schedule:
            layers_to_unfreeze = self.config.unfreeze_schedule[self.current_epoch]
            self.unfreeze_layers(layers_to_unfreeze)
            return True
        return False

    def setup_optimizers(self) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """
        Setup optimizers with different learning rates for transferred vs new layers.

        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Group parameters by transfer status
        transferred_params = []
        new_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.transferred_layers:
                    transferred_params.append(param)
                else:
                    new_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = []

        if transferred_params:
            param_groups.append({
                'params': transferred_params,
                'lr': self.config.base_learning_rate * self.config.transferred_lr_factor,
                'weight_decay': 1e-5
            })

        if new_params:
            param_groups.append({
                'params': new_params,
                'lr': self.config.base_learning_rate * self.config.new_layer_lr_factor,
                'weight_decay': 1e-4
            })

        optimizer = optim.AdamW(param_groups)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return optimizer, scheduler

    def options_domain_loss(self,
                           outputs: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for options domain.

        Args:
            outputs: Model outputs dict with 'strategy_scores' and 'regime_logits'
            targets: Target dict with 'strategy_labels' and 'regime_labels'

        Returns:
            Tuple of (total_loss, loss_components)
        """
        losses = {}

        # Strategy evaluation accuracy loss
        if 'strategy_scores' in outputs and 'strategy_labels' in targets:
            strategy_loss = F.mse_loss(outputs['strategy_scores'], targets['strategy_labels'])
            losses['strategy_loss'] = strategy_loss * self.config.strategy_accuracy_weight

        # Regime classification loss
        if 'regime_logits' in outputs and 'regime_labels' in targets:
            regime_loss = F.cross_entropy(outputs['regime_logits'], targets['regime_labels'])
            losses['regime_loss'] = regime_loss * self.config.regime_classification_weight

        # L2 regularization on new layers
        if self.config.regularization_weight > 0:
            l2_reg = 0.0
            for name, param in self.model.named_parameters():
                if name in self.new_layers and param.requires_grad:
                    l2_reg += torch.norm(param, p=2)
            losses['l2_regularization'] = l2_reg * self.config.regularization_weight

        # Combine losses
        total_loss = sum(losses.values())

        # Convert to float for logging
        loss_components = {k: v.item() if isinstance(v, torch.Tensor) else v
                          for k, v in losses.items()}
        loss_components['total_loss'] = total_loss.item()

        return total_loss, loss_components

    def transfer_learning_train(self,
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Execute transfer learning training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save best model

        Returns:
            Dictionary with training results and metrics
        """
        self.logger.info("Starting transfer learning training")

        # Setup training components
        optimizer, scheduler = self.setup_optimizers()
        best_model_state = None

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Apply progressive unfreezing schedule
            unfroze_layers = self.fine_tune_schedule()
            if unfroze_layers:
                # Update optimizer with newly unfrozen parameters
                optimizer, scheduler = self.setup_optimizers()

            # Training phase
            train_metrics = self._train_epoch(train_loader, optimizer)

            # Validation phase
            val_metrics = self._validate_epoch(val_loader)

            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])

            # Early stopping and model saving
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                if save_path:
                    torch.save({
                        'model_state_dict': best_model_state,
                        'epoch': epoch,
                        'val_loss': self.best_val_loss,
                        'config': self.config
                    }, save_path)
            else:
                self.patience_counter += 1

            # Log epoch results
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
                'lr': optimizer.param_groups[0]['lr'],
                'frozen_params': len(self.frozen_layers)
            }
            epoch_info.update({f'train_{k}': v for k, v in train_metrics.items()})
            epoch_info.update({f'val_{k}': v for k, v in val_metrics.items()})

            self.training_history.append(epoch_info)

            self.logger.info(
                f"Epoch {epoch:3d}: train_loss={train_metrics['total_loss']:.4f}, "
                f"val_loss={val_metrics['total_loss']:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        training_results = {
            'final_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'training_history': self.training_history,
            'transferred_layers': list(self.transferred_layers),
            'new_layers': list(self.new_layers),
            'final_frozen_layers': list(self.frozen_layers)
        }

        self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        return training_results

    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        epoch_losses = {}
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items() if k != 'targets'}
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch['targets'].items()}

            # Forward pass
            optimizer.zero_grad()
            outputs = self._forward_model(inputs)

            # Compute loss
            loss, loss_components = self.options_domain_loss(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            optimizer.step()

            # Accumulate losses
            batch_size = len(next(iter(inputs.values())))
            total_samples += batch_size

            for key, value in loss_components.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value * batch_size

        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= total_samples

        return epoch_losses

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Execute one validation epoch."""
        self.model.eval()
        epoch_losses = {}
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k != 'targets'}
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch['targets'].items()}

                # Forward pass
                outputs = self._forward_model(inputs)

                # Compute loss
                _, loss_components = self.options_domain_loss(outputs, targets)

                # Accumulate losses
                batch_size = len(next(iter(inputs.values())))
                total_samples += batch_size

                for key, value in loss_components.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value * batch_size

        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= total_samples

        return epoch_losses

    def _forward_model(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, dict) and "spatial_tensor" in inputs and hasattr(self.model, "market_encoder"):
            spatial_tensor = inputs["spatial_tensor"]
            market_features = inputs.get("market_features")
            if market_features is None:
                batch_size = spatial_tensor.shape[0]
                feature_dim = getattr(getattr(self.model, "regime_detector", None), "input_dim", 48)
                market_features = torch.zeros(
                    batch_size,
                    feature_dim,
                    device=spatial_tensor.device,
                    dtype=spatial_tensor.dtype,
                )
            return self.model(spatial_tensor, market_features)
        return self.model(inputs)

    def get_transfer_effectiveness_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics to evaluate transfer learning effectiveness.

        Returns:
            Dictionary with transfer effectiveness metrics
        """
        if not self.training_history:
            return {}

        final_metrics = self.training_history[-1]
        initial_metrics = self.training_history[0]

        effectiveness = {
            'loss_improvement': initial_metrics['val_loss'] - final_metrics['val_loss'],
            'loss_improvement_pct': (initial_metrics['val_loss'] - final_metrics['val_loss']) /
                                   initial_metrics['val_loss'] * 100,
            'epochs_to_best': next(
                (i for i, metrics in enumerate(self.training_history)
                 if metrics['val_loss'] == self.best_val_loss), -1
            ),
            'transferred_layer_count': len(self.transferred_layers),
            'new_layer_count': len(self.new_layers),
            'final_frozen_layer_count': len(self.frozen_layers),
            'convergence_speed': len(self.training_history)
        }

        return effectiveness


def create_transfer_trainer_from_chess_model(chess_model_path: Union[str, Path],
                                           target_model: nn.Module,
                                           config: Optional[TransferTrainingConfig] = None,
                                           device: Optional[torch.device] = None) -> TransferTrainer:
    """
    Convenience function to create a transfer trainer with chess weights pre-loaded.

    Args:
        chess_model_path: Path to chess model file
        target_model: Target model for fine-tuning
        config: Training configuration (optional)
        device: Training device (optional)

    Returns:
        Initialized TransferTrainer with chess weights loaded
    """
    config = config or TransferTrainingConfig()
    trainer = TransferTrainer(target_model, config, device)

    # Load chess weights
    load_result = trainer.load_chess_weights(chess_model_path)

    # Apply initial freezing
    trainer.freeze_chess_layers()

    return trainer
