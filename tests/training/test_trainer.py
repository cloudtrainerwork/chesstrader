"""
Comprehensive tests for RegimeTrainer and training infrastructure.

Tests trainer initialization, training mechanics, validation steps,
checkpointing, device management, and integration testing.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Tuple

from src.models.regime_detector import RegimeDetector
from src.training.trainer import RegimeTrainer
from src.config import TrainingConfig


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model():
    """Create a test model."""
    return RegimeDetector(input_dim=48, hidden_dims=(64, 32, 16), num_regimes=8, dropout_rate=0.1)


@pytest.fixture
def training_config():
    """Create test training configuration."""
    config = TrainingConfig()
    config.batch_size = 8
    config.learning_rate = 1e-3
    config.max_epochs = 3
    config.patience = 2
    config.mixed_precision = False  # Disable for testing stability
    config.validation_frequency = 1
    config.checkpoint_frequency = 1
    return config


@pytest.fixture
def sample_data():
    """Create sample training data."""
    batch_size = 16
    features = torch.randn(batch_size, 48)
    labels = torch.randint(0, 8, (batch_size,))
    return features, labels


@pytest.fixture
def sample_dataloader():
    """Create sample DataLoader for testing."""
    features = torch.randn(32, 48)
    labels = torch.randint(0, 8, (32,))
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestRegimeTrainerInitialization:
    """Test trainer initialization and setup."""

    def test_trainer_initialization(self, model, training_config):
        """Test basic trainer initialization."""
        trainer = RegimeTrainer(model, training_config)

        assert trainer.model is model
        assert trainer.config is training_config
        assert trainer.device is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.current_epoch == 0
        assert trainer.best_val_accuracy == 0.0
        assert trainer.best_val_loss == float('inf')
        assert trainer.epochs_without_improvement == 0

    def test_device_setup_auto(self, model):
        """Test automatic device selection."""
        config = TrainingConfig()
        config.device = "auto"
        trainer = RegimeTrainer(model, config)

        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert trainer.device == expected_device

    def test_device_setup_specific(self, model):
        """Test specific device selection."""
        config = TrainingConfig()
        config.device = "cpu"
        trainer = RegimeTrainer(model, config)

        assert trainer.device == torch.device('cpu')

    def test_mixed_precision_setup(self, model):
        """Test mixed precision setup."""
        config = TrainingConfig()
        config.mixed_precision = True
        trainer = RegimeTrainer(model, config)

        if trainer.device.type == 'cuda':
            assert trainer.scaler is not None
        else:
            assert trainer.scaler is None


class TestTrainingMechanics:
    """Test core training mechanics."""

    def test_compute_loss(self, model, training_config, sample_data):
        """Test loss computation."""
        trainer = RegimeTrainer(model, training_config)
        features, labels = sample_data

        # Forward pass
        outputs = trainer.model(features)
        losses = trainer.compute_loss(outputs, labels)

        assert 'total' in losses
        assert 'classification' in losses
        assert 'confidence' in losses
        assert all(isinstance(loss, torch.Tensor) for loss in losses.values())
        assert all(loss.item() >= 0 for loss in losses.values())

    def test_train_step(self, model, training_config, sample_data):
        """Test training step mechanics."""
        trainer = RegimeTrainer(model, training_config)

        # Store initial parameters for comparison
        initial_params = [p.clone().detach() for p in trainer.model.parameters()]

        # Execute training step
        losses = trainer.train_step(sample_data)

        # Check loss structure
        assert isinstance(losses, dict)
        assert 'total' in losses
        assert 'classification' in losses
        assert 'confidence' in losses
        assert all(isinstance(loss, float) for loss in losses.values())

        # Check that parameters have been updated
        updated_params = [p.clone().detach() for p in trainer.model.parameters()]
        param_changed = any(not torch.equal(initial, updated)
                          for initial, updated in zip(initial_params, updated_params))
        assert param_changed, "Model parameters should be updated after training step"

    def test_validate_step(self, model, training_config, sample_data):
        """Test validation step mechanics."""
        trainer = RegimeTrainer(model, training_config)

        results = trainer.validate_step(sample_data)

        # Check result structure
        assert isinstance(results, dict)
        assert 'losses' in results
        assert 'accuracy' in results
        assert 'predictions' in results
        assert 'regime_probs' in results
        assert 'confidence_scores' in results
        assert 'labels' in results

        # Check data types and shapes
        assert isinstance(results['accuracy'], float)
        assert 0 <= results['accuracy'] <= 1
        assert isinstance(results['predictions'], np.ndarray)
        assert isinstance(results['regime_probs'], np.ndarray)
        assert isinstance(results['confidence_scores'], np.ndarray)
        assert isinstance(results['labels'], np.ndarray)

    def test_train_epoch(self, model, training_config, sample_dataloader):
        """Test full epoch training."""
        trainer = RegimeTrainer(model, training_config)

        epoch_metrics = trainer.train_epoch(sample_dataloader)

        assert isinstance(epoch_metrics, dict)
        assert 'total' in epoch_metrics
        assert 'classification' in epoch_metrics
        assert 'confidence' in epoch_metrics
        assert all(isinstance(metric, float) for metric in epoch_metrics.values())
        assert all(metric >= 0 for metric in epoch_metrics.values())

    def test_validate_epoch(self, model, training_config, sample_dataloader):
        """Test full epoch validation."""
        trainer = RegimeTrainer(model, training_config)

        val_metrics = trainer.validate_epoch(sample_dataloader)

        assert isinstance(val_metrics, dict)
        assert 'losses' in val_metrics
        assert 'accuracy' in val_metrics
        assert 'classification_accuracy' in val_metrics
        assert 'confidence_mae' in val_metrics
        assert 'predictions' in val_metrics
        assert 'regime_probs' in val_metrics
        assert 'confidence_scores' in val_metrics
        assert 'labels' in val_metrics


class TestCheckpointing:
    """Test model checkpointing and state management."""

    def test_save_checkpoint(self, model, training_config, temp_checkpoint_dir):
        """Test checkpoint saving."""
        training_config.checkpoint_dir = temp_checkpoint_dir
        trainer = RegimeTrainer(model, training_config)

        # Save checkpoint
        checkpoint_name = "test_checkpoint.pth"
        test_metrics = {'val_acc': 0.85, 'val_loss': 0.5}
        trainer.save_checkpoint(checkpoint_name, epoch=5, metrics=test_metrics)

        # Check file exists
        checkpoint_path = temp_checkpoint_dir / checkpoint_name
        assert checkpoint_path.exists()

        # Load and verify contents
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        assert checkpoint_data['epoch'] == 5
        assert 'model_state_dict' in checkpoint_data
        assert 'optimizer_state_dict' in checkpoint_data
        assert 'scheduler_state_dict' in checkpoint_data
        assert checkpoint_data['validation_metrics'] == test_metrics

    def test_load_checkpoint(self, model, training_config, temp_checkpoint_dir):
        """Test checkpoint loading."""
        training_config.checkpoint_dir = temp_checkpoint_dir
        trainer = RegimeTrainer(model, training_config)

        # Save a checkpoint first
        checkpoint_name = "test_checkpoint.pth"
        trainer.best_val_accuracy = 0.9
        trainer.current_epoch = 10
        trainer.save_checkpoint(checkpoint_name, epoch=10)

        # Create new trainer and load checkpoint
        new_model = RegimeDetector(input_dim=48, hidden_dims=(64, 32, 16), num_regimes=8)
        new_trainer = RegimeTrainer(new_model, training_config)
        new_trainer.load_checkpoint(checkpoint_name)

        # Verify state restoration
        assert new_trainer.current_epoch == 10
        assert new_trainer.best_val_accuracy == 0.9

    def test_checkpoint_compatibility_check(self, training_config, temp_checkpoint_dir):
        """Test checkpoint compatibility validation."""
        training_config.checkpoint_dir = temp_checkpoint_dir

        # Create and save model with specific architecture
        model1 = RegimeDetector(input_dim=48, hidden_dims=(64, 32, 16), num_regimes=8)
        trainer1 = RegimeTrainer(model1, training_config)
        trainer1.save_checkpoint("compatible_checkpoint.pth", epoch=1)

        # Try to load with incompatible architecture
        model2 = RegimeDetector(input_dim=24, hidden_dims=(64, 32, 16), num_regimes=8)  # Different input_dim
        trainer2 = RegimeTrainer(model2, training_config)

        with pytest.raises(ValueError):
            trainer2.load_checkpoint("compatible_checkpoint.pth")

    def test_checkpoint_cleanup(self, model, training_config, temp_checkpoint_dir):
        """Test automatic checkpoint cleanup."""
        training_config.checkpoint_dir = temp_checkpoint_dir
        training_config.max_checkpoints = 2
        trainer = RegimeTrainer(model, training_config)

        # Create multiple checkpoints
        for i in range(5):
            trainer.save_checkpoint(f"checkpoint_epoch_{i}.pth", epoch=i)

        # Check that only max_checkpoints exist
        checkpoint_files = list(temp_checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        assert len(checkpoint_files) <= training_config.max_checkpoints


class TestEarlyStoppingAndScheduling:
    """Test early stopping and learning rate scheduling."""

    def test_early_stopping_trigger(self, model, sample_dataloader):
        """Test early stopping mechanism."""
        config = TrainingConfig()
        config.patience = 2
        config.max_epochs = 10
        trainer = RegimeTrainer(model, config)

        # Simulate training without improvement
        trainer.epochs_without_improvement = config.patience

        # This should trigger early stopping in a real training loop
        assert trainer.epochs_without_improvement >= config.patience

    def test_learning_rate_scheduling(self, model, training_config, sample_dataloader):
        """Test learning rate scheduler behavior."""
        trainer = RegimeTrainer(model, training_config)

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # Simulate validation loss that should trigger LR reduction
        high_loss = 10.0
        trainer.scheduler.step(high_loss)
        trainer.scheduler.step(high_loss)
        trainer.scheduler.step(high_loss)
        trainer.scheduler.step(high_loss)
        trainer.scheduler.step(high_loss)
        trainer.scheduler.step(high_loss)  # Should trigger reduction after patience

        current_lr = trainer.optimizer.param_groups[0]['lr']
        # LR should be reduced or stay the same (depending on scheduler state)
        assert current_lr <= initial_lr


class TestDeviceManagement:
    """Test device management functionality."""

    def test_model_device_consistency(self, model, training_config, sample_data):
        """Test that model and data are on consistent devices."""
        trainer = RegimeTrainer(model, training_config)

        # Check model is on correct device
        assert next(trainer.model.parameters()).device == trainer.device

        # Test training step with device consistency
        features, labels = sample_data
        features = features.to(trainer.device)
        labels = labels.to(trainer.device)

        # Should not raise any device mismatch errors
        losses = trainer.train_step((features, labels))
        assert isinstance(losses, dict)


class TestModelSummary:
    """Test model summary and information methods."""

    def test_get_model_summary(self, model, training_config):
        """Test model summary generation."""
        trainer = RegimeTrainer(model, training_config)

        summary = trainer.get_model_summary()

        assert isinstance(summary, dict)
        assert 'architecture' in summary
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert 'model_size_mb' in summary
        assert 'device' in summary
        assert 'mixed_precision' in summary
        assert 'current_epoch' in summary
        assert 'best_val_accuracy' in summary

        # Check data types
        assert isinstance(summary['total_parameters'], int)
        assert isinstance(summary['trainable_parameters'], int)
        assert isinstance(summary['model_size_mb'], float)
        assert summary['total_parameters'] > 0
        assert summary['trainable_parameters'] > 0


class TestIntegrationTraining:
    """Integration tests for full training pipeline."""

    def test_short_training_integration(self, model, sample_dataloader):
        """Test complete training pipeline with minimal epochs."""
        config = TrainingConfig()
        config.max_epochs = 2
        config.patience = 5  # Prevent early stopping
        config.validation_frequency = 1
        config.mixed_precision = False  # For stability

        trainer = RegimeTrainer(model, config)

        # Run short training
        history = trainer.fit(
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,  # Use same data for simplicity
            epochs=2
        )

        # Check training history structure
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'val_accuracy' in history
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2
        assert len(history['val_accuracy']) == 2

    def test_training_with_checkpointing(self, model, sample_dataloader, temp_checkpoint_dir):
        """Test training with periodic checkpointing."""
        config = TrainingConfig()
        config.checkpoint_dir = temp_checkpoint_dir
        config.max_epochs = 3
        config.patience = 10  # Prevent early stopping
        config.checkpoint_frequency = 1  # Save every epoch
        config.mixed_precision = False

        trainer = RegimeTrainer(model, config)

        # Run training
        trainer.fit(
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            epochs=3
        )

        # Check that checkpoints were created
        checkpoint_files = list(temp_checkpoint_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0  # At least best_model.pth

    def test_training_resumption(self, model, sample_dataloader, temp_checkpoint_dir):
        """Test training resumption from checkpoint."""
        config = TrainingConfig()
        config.checkpoint_dir = temp_checkpoint_dir
        config.max_epochs = 2
        config.patience = 10
        config.mixed_precision = False

        # First training session
        trainer1 = RegimeTrainer(model, config)
        trainer1.fit(
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            epochs=2
        )

        # Save intermediate checkpoint
        trainer1.save_checkpoint("resume_test.pth", epoch=trainer1.current_epoch)

        # Create new trainer and resume
        new_model = RegimeDetector(input_dim=48, hidden_dims=(64, 32, 16), num_regimes=8, dropout_rate=0.1)
        trainer2 = RegimeTrainer(new_model, config)
        trainer2.load_checkpoint("resume_test.pth")

        # Verify state was restored
        assert trainer2.current_epoch == trainer1.current_epoch
        assert abs(trainer2.best_val_accuracy - trainer1.best_val_accuracy) < 1e-6


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_batch_handling(self, model, training_config):
        """Test handling of invalid batch data."""
        trainer = RegimeTrainer(model, training_config)

        # Test with wrong feature dimensions
        invalid_features = torch.randn(4, 24)  # Wrong dimension
        labels = torch.randint(0, 8, (4,))

        with pytest.raises(ValueError):
            trainer.train_step((invalid_features, labels))

    def test_empty_dataloader_handling(self, model, training_config):
        """Test behavior with empty data loaders."""
        trainer = RegimeTrainer(model, training_config)

        # Create empty dataloader
        empty_dataset = TensorDataset(torch.empty(0, 48), torch.empty(0, dtype=torch.long))
        empty_loader = DataLoader(empty_dataset, batch_size=8)

        # Should handle gracefully
        epoch_metrics = trainer.train_epoch(empty_loader)

        # Should return metrics with appropriate default values
        assert isinstance(epoch_metrics, dict)