"""
Tests for transfer learning trainer module.

Tests TransferTrainer class functionality including chess weight integration,
progressive unfreezing, and options domain fine-tuning.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.models.transfer_trainer import (
    TransferTrainer, TransferTrainingConfig,
    create_transfer_trainer_from_chess_model
)
from src.models.spatial_net import SpatialNet, SpatialNetConfig


class SimpleCNNModel(nn.Module):
    """Simple CNN model for testing."""

    def __init__(self, input_channels=20, num_classes=16):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.output_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if isinstance(x, dict) and 'spatial_tensor' in x:
            x = x['spatial_tensor']

        x = self.spatial_encoder(x)
        x = self.feature_extractor(x)
        strategy_scores = self.output_head(x)

        return {
            'strategy_scores': strategy_scores,
            'regime_logits': strategy_scores[:, :4]  # Use first 4 as regime logits
        }


class TestTransferTrainingConfig:
    """Test transfer training configuration."""

    def test_default_config_initialization(self):
        """Test default configuration values."""
        config = TransferTrainingConfig()

        assert config.base_learning_rate == 1e-4
        assert config.transferred_lr_factor == 0.1
        assert config.new_layer_lr_factor == 1.0
        assert config.strategy_accuracy_weight == 0.7
        assert config.regime_classification_weight == 0.3
        assert config.transfer_mode == 'fine_tune'

        # Check default freeze stages and unfreeze schedule
        assert 'spatial_encoder' in config.freeze_stages
        assert 5 in config.unfreeze_schedule

    def test_custom_config_initialization(self):
        """Test custom configuration initialization."""
        custom_freeze_stages = ['encoder', 'decoder']
        custom_unfreeze_schedule = {3: ['decoder'], 6: ['encoder']}

        config = TransferTrainingConfig(
            base_learning_rate=2e-4,
            freeze_stages=custom_freeze_stages,
            unfreeze_schedule=custom_unfreeze_schedule
        )

        assert config.base_learning_rate == 2e-4
        assert config.freeze_stages == custom_freeze_stages
        assert config.unfreeze_schedule == custom_unfreeze_schedule


class TestTransferTrainer:
    """Test transfer trainer functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return SimpleCNNModel(input_channels=20, num_classes=16)

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TransferTrainingConfig(
            base_learning_rate=1e-3,
            num_epochs=5,
            batch_size=4,
            early_stopping_patience=3,
            freeze_stages=['spatial_encoder'],
            unfreeze_schedule={2: ['spatial_encoder']}
        )

    @pytest.fixture
    def trainer(self, simple_model, config):
        """Create transfer trainer instance."""
        return TransferTrainer(simple_model, config, device=torch.device('cpu'))

    @pytest.fixture
    def sample_chess_weights(self):
        """Create sample chess weights."""
        return {
            'spatial_encoder.0.weight': torch.randn(32, 20, 3, 3),
            'spatial_encoder.0.bias': torch.randn(32),
            'spatial_encoder.2.weight': torch.randn(64, 32, 3, 3),
            'spatial_encoder.2.bias': torch.randn(64),
            'feature_extractor.0.weight': torch.randn(128, 64, 3, 3),
            'feature_extractor.0.bias': torch.randn(128),
            'output_head.0.weight': torch.randn(64, 128),
            'output_head.0.bias': torch.randn(64),
            'output_head.3.weight': torch.randn(16, 64),
            'output_head.3.bias': torch.randn(16)
        }

    @pytest.fixture
    def temp_chess_model(self, sample_chess_weights):
        """Create temporary chess model file."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(sample_chess_weights, f.name)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def sample_dataloader(self):
        """Create sample data loader for testing."""
        batch_size = 4
        spatial_data = torch.randn(20, 20, 7, 6)  # 20 samples
        strategy_labels = torch.randn(20, 16)
        regime_labels = torch.randint(0, 4, (20,))

        # Create dataset with proper structure
        dataset_inputs = []
        dataset_targets = []

        for i in range(20):
            inputs = {'spatial_tensor': spatial_data[i]}
            targets = {
                'strategy_labels': strategy_labels[i],
                'regime_labels': regime_labels[i]
            }
            dataset_inputs.append(inputs)
            dataset_targets.append(targets)

        # Custom collate function
        def collate_fn(batch):
            batch_inputs = {}
            batch_targets = {}

            # Collect spatial tensors
            spatial_tensors = [item[0]['spatial_tensor'] for item in batch]
            batch_inputs['spatial_tensor'] = torch.stack(spatial_tensors)

            # Collect targets
            strategy_labels = [item[1]['strategy_labels'] for item in batch]
            regime_labels = [item[1]['regime_labels'] for item in batch]

            batch_targets['strategy_labels'] = torch.stack(strategy_labels)
            batch_targets['regime_labels'] = torch.stack(regime_labels)

            return {'targets': batch_targets, **batch_inputs}

        # Create dataset and dataloader
        combined_dataset = list(zip(dataset_inputs, dataset_targets))
        return DataLoader(combined_dataset, batch_size=batch_size, collate_fn=collate_fn)

    def test_trainer_initialization(self, simple_model, config):
        """Test trainer initialization."""
        trainer = TransferTrainer(simple_model, config, device=torch.device('cpu'))

        assert trainer.model == simple_model
        assert trainer.config == config
        assert trainer.device == torch.device('cpu')
        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float('inf')
        assert len(trainer.frozen_layers) == 0
        assert len(trainer.transferred_layers) == 0
        assert len(trainer.new_layers) == 0

    @patch('src.models.transfer_trainer.load_and_adapt_chess_weights')
    def test_load_chess_weights_success(self, mock_load, trainer, temp_chess_model, sample_chess_weights):
        """Test successful chess weight loading."""
        mock_load.return_value = sample_chess_weights

        result = trainer.load_chess_weights(temp_chess_model)

        assert 'loaded_layers' in result
        assert 'transferred_layers' in result
        assert 'new_layers' in result
        assert len(trainer.transferred_layers) > 0
        assert len(trainer.new_layers) > 0

    @patch('src.models.transfer_trainer.load_and_adapt_chess_weights')
    def test_load_chess_weights_failure(self, mock_load, trainer, temp_chess_model):
        """Test chess weight loading failure."""
        mock_load.side_effect = RuntimeError("Loading failed")

        with pytest.raises(RuntimeError, match="Loading failed"):
            trainer.load_chess_weights(temp_chess_model)

    def test_freeze_chess_layers(self, trainer):
        """Test freezing chess layers."""
        # Initially all parameters should require grad
        total_params = sum(1 for p in trainer.model.parameters())
        grad_params = sum(1 for p in trainer.model.parameters() if p.requires_grad)
        assert grad_params == total_params

        # Freeze spatial encoder layers
        trainer.freeze_chess_layers(['spatial_encoder'])

        # Check that some parameters are frozen
        frozen_params = sum(1 for p in trainer.model.parameters() if not p.requires_grad)
        assert frozen_params > 0
        assert len(trainer.frozen_layers) > 0

    def test_unfreeze_layers(self, trainer):
        """Test unfreezing layers."""
        # First freeze some layers
        trainer.freeze_chess_layers(['spatial_encoder'])
        initial_frozen_count = len(trainer.frozen_layers)
        assert initial_frozen_count > 0

        # Then unfreeze them
        trainer.unfreeze_layers(['spatial_encoder'])

        # Check that layers are unfrozen
        final_frozen_count = len(trainer.frozen_layers)
        assert final_frozen_count < initial_frozen_count

    def test_fine_tune_schedule(self, trainer):
        """Test progressive unfreezing schedule."""
        # Set up initial frozen state
        trainer.freeze_chess_layers(['spatial_encoder'])
        initial_frozen_count = len(trainer.frozen_layers)

        # Test schedule at wrong epoch (should not unfreeze)
        trainer.current_epoch = 1
        unfroze = trainer.fine_tune_schedule()
        assert not unfroze
        assert len(trainer.frozen_layers) == initial_frozen_count

        # Test schedule at correct epoch (should unfreeze)
        trainer.current_epoch = 2  # Based on config fixture
        unfroze = trainer.fine_tune_schedule()
        assert unfroze
        assert len(trainer.frozen_layers) < initial_frozen_count

    def test_setup_optimizers_with_transferred_layers(self, trainer, sample_chess_weights):
        """Test optimizer setup with transferred layers."""
        # Simulate loaded chess weights
        trainer.transferred_layers = set(sample_chess_weights.keys())
        model_keys = set(trainer.model.state_dict().keys())
        trainer.new_layers = model_keys - trainer.transferred_layers

        optimizer, scheduler = trainer.setup_optimizers()

        assert isinstance(optimizer, optim.AdamW)
        assert isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

        # Should have parameter groups for transferred and new layers
        assert len(optimizer.param_groups) >= 1

    def test_options_domain_loss_complete(self, trainer):
        """Test options domain loss computation with all components."""
        outputs = {
            'strategy_scores': torch.randn(4, 16),
            'regime_logits': torch.randn(4, 4)
        }
        targets = {
            'strategy_labels': torch.randn(4, 16),
            'regime_labels': torch.randint(0, 4, (4,))
        }

        # Add some new layers for regularization
        trainer.new_layers = {'output_head.0.weight', 'output_head.3.weight'}

        loss, components = trainer.options_domain_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert 'strategy_loss' in components
        assert 'regime_loss' in components
        assert 'l2_regularization' in components
        assert 'total_loss' in components

    def test_options_domain_loss_partial(self, trainer):
        """Test options domain loss with partial outputs."""
        # Test with only strategy outputs
        outputs = {'strategy_scores': torch.randn(4, 16)}
        targets = {'strategy_labels': torch.randn(4, 16)}

        loss, components = trainer.options_domain_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert 'strategy_loss' in components
        assert 'regime_loss' not in components

    @patch('src.models.transfer_trainer.load_and_adapt_chess_weights')
    def test_transfer_learning_train_integration(self, mock_load, trainer, sample_chess_weights,
                                                sample_dataloader, tmp_path):
        """Test complete transfer learning training integration."""
        # Setup mock chess weight loading
        mock_load.return_value = sample_chess_weights
        trainer.transferred_layers = set(sample_chess_weights.keys())
        model_keys = set(trainer.model.state_dict().keys())
        trainer.new_layers = model_keys - trainer.transferred_layers

        # Use small config for fast test
        trainer.config.num_epochs = 2
        trainer.config.early_stopping_patience = 1

        # Run training
        save_path = tmp_path / "test_model.pth"
        results = trainer.transfer_learning_train(sample_dataloader, sample_dataloader, save_path)

        assert 'final_val_loss' in results
        assert 'total_epochs' in results
        assert 'training_history' in results
        assert len(trainer.training_history) > 0

        # Check that model was saved
        assert save_path.exists()

    def test_train_epoch_functionality(self, trainer, sample_dataloader):
        """Test training epoch execution."""
        optimizer = optim.Adam(trainer.model.parameters(), lr=1e-3)

        # Setup some transferred layers for realistic test
        trainer.transferred_layers = {'spatial_encoder.0.weight', 'spatial_encoder.2.weight'}
        trainer.new_layers = set(trainer.model.state_dict().keys()) - trainer.transferred_layers

        metrics = trainer._train_epoch(sample_dataloader, optimizer)

        assert isinstance(metrics, dict)
        assert 'total_loss' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_validate_epoch_functionality(self, trainer, sample_dataloader):
        """Test validation epoch execution."""
        # Setup some transferred layers for realistic test
        trainer.transferred_layers = {'spatial_encoder.0.weight', 'spatial_encoder.2.weight'}
        trainer.new_layers = set(trainer.model.state_dict().keys()) - trainer.transferred_layers

        metrics = trainer._validate_epoch(sample_dataloader)

        assert isinstance(metrics, dict)
        assert 'total_loss' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_get_transfer_effectiveness_metrics_empty_history(self, trainer):
        """Test effectiveness metrics with empty training history."""
        metrics = trainer.get_transfer_effectiveness_metrics()
        assert metrics == {}

    def test_get_transfer_effectiveness_metrics_with_history(self, trainer):
        """Test effectiveness metrics with training history."""
        # Create mock training history
        trainer.training_history = [
            {'val_loss': 2.0},
            {'val_loss': 1.5},
            {'val_loss': 1.0}
        ]
        trainer.best_val_loss = 1.0
        trainer.transferred_layers = {'layer1', 'layer2'}
        trainer.new_layers = {'layer3'}
        trainer.frozen_layers = {'layer1'}

        metrics = trainer.get_transfer_effectiveness_metrics()

        assert 'loss_improvement' in metrics
        assert 'loss_improvement_pct' in metrics
        assert 'epochs_to_best' in metrics
        assert 'transferred_layer_count' in metrics
        assert 'new_layer_count' in metrics
        assert metrics['loss_improvement'] == 1.0  # 2.0 - 1.0
        assert metrics['transferred_layer_count'] == 2
        assert metrics['new_layer_count'] == 1


class TestCreateTransferTrainerFromChessModel:
    """Test convenience function for creating transfer trainer."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return SimpleCNNModel()

    @pytest.fixture
    def temp_chess_model(self):
        """Create temporary chess model file."""
        weights = {
            'spatial_encoder.0.weight': torch.randn(32, 20, 3, 3),
            'spatial_encoder.0.bias': torch.randn(32)
        }
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(weights, f.name)
            yield f.name
        os.unlink(f.name)

    @patch('src.models.transfer_trainer.TransferTrainer.load_chess_weights')
    @patch('src.models.transfer_trainer.TransferTrainer.freeze_chess_layers')
    def test_create_trainer_from_chess_model(self, mock_freeze, mock_load,
                                           simple_model, temp_chess_model):
        """Test creating trainer from chess model."""
        mock_load.return_value = {'loaded_layers': ['layer1']}

        config = TransferTrainingConfig(base_learning_rate=2e-4)
        trainer = create_transfer_trainer_from_chess_model(
            temp_chess_model, simple_model, config
        )

        assert isinstance(trainer, TransferTrainer)
        assert trainer.config.base_learning_rate == 2e-4
        mock_load.assert_called_once_with(temp_chess_model)
        mock_freeze.assert_called_once()

    @patch('src.models.transfer_trainer.TransferTrainer.load_chess_weights')
    @patch('src.models.transfer_trainer.TransferTrainer.freeze_chess_layers')
    def test_create_trainer_default_config(self, mock_freeze, mock_load,
                                          simple_model, temp_chess_model):
        """Test creating trainer with default configuration."""
        mock_load.return_value = {'loaded_layers': ['layer1']}

        trainer = create_transfer_trainer_from_chess_model(temp_chess_model, simple_model)

        assert isinstance(trainer, TransferTrainer)
        assert isinstance(trainer.config, TransferTrainingConfig)
        assert trainer.config.base_learning_rate == 1e-4  # Default value


class TestIntegrationWithSpatialNet:
    """Integration tests with SpatialNet model."""

    @pytest.fixture
    def spatial_net_model(self):
        """Create SpatialNet model for integration testing."""
        config = SpatialNetConfig(
            base_channels=32,
            num_stages=2,
            blocks_per_stage=1,
            output_features=128,
            strategy_classes=16
        )
        return SpatialNet(config)

    @pytest.fixture
    def spatial_config(self):
        """Create config suitable for SpatialNet."""
        return TransferTrainingConfig(
            base_learning_rate=5e-4,
            num_epochs=3,
            batch_size=2,
            freeze_stages=['spatial_encoder.encoder', 'feature_extractor.stages.0'],
            unfreeze_schedule={1: ['feature_extractor.stages.0'], 2: ['spatial_encoder.encoder']}
        )

    @pytest.fixture
    def spatial_dataloader(self):
        """Create data loader compatible with SpatialNet."""
        batch_size = 2

        # Create sample data matching SpatialNet expected input
        spatial_data = torch.randn(10, 20, 7, 6)  # 10 samples, 20 channels, 7x6 spatial
        strategy_labels = torch.randn(10, 16)
        regime_labels = torch.randint(0, 4, (10,))

        dataset_inputs = []
        dataset_targets = []

        for i in range(10):
            inputs = {'spatial_tensor': spatial_data[i]}
            targets = {
                'strategy_labels': strategy_labels[i],
                'regime_labels': regime_labels[i]
            }
            dataset_inputs.append(inputs)
            dataset_targets.append(targets)

        def collate_fn(batch):
            batch_inputs = {}
            batch_targets = {}

            spatial_tensors = [item[0]['spatial_tensor'] for item in batch]
            batch_inputs['spatial_tensor'] = torch.stack(spatial_tensors)

            strategy_labels = [item[1]['strategy_labels'] for item in batch]
            regime_labels = [item[1]['regime_labels'] for item in batch]

            batch_targets['strategy_labels'] = torch.stack(strategy_labels)
            batch_targets['regime_labels'] = torch.stack(regime_labels)

            return {'targets': batch_targets, **batch_inputs}

        combined_dataset = list(zip(dataset_inputs, dataset_targets))
        return DataLoader(combined_dataset, batch_size=batch_size, collate_fn=collate_fn)

    def test_transfer_trainer_with_spatial_net(self, spatial_net_model, spatial_config, spatial_dataloader):
        """Test transfer trainer integration with SpatialNet."""
        trainer = TransferTrainer(spatial_net_model, spatial_config, device=torch.device('cpu'))

        # Test that trainer works with SpatialNet
        assert trainer.model == spatial_net_model

        # Test freezing layers
        trainer.freeze_chess_layers()
        frozen_count = len(trainer.frozen_layers)
        assert frozen_count > 0

        # Test optimizer setup
        trainer.new_layers = set(trainer.model.state_dict().keys())
        optimizer, scheduler = trainer.setup_optimizers()
        assert isinstance(optimizer, optim.AdamW)

        # Test single epoch (quick integration test)
        metrics = trainer._validate_epoch(spatial_dataloader)
        assert 'total_loss' in metrics

    @patch('src.models.transfer_trainer.load_and_adapt_chess_weights')
    def test_spatial_net_with_chess_weights(self, mock_load, spatial_net_model, spatial_config):
        """Test SpatialNet with mock chess weights."""
        # Create mock weights that match SpatialNet structure
        mock_weights = {
            'spatial_encoder.encoder.initial_conv.weight': torch.randn(32, 20, 7, 6),
            'spatial_encoder.encoder.initial_conv.bias': torch.randn(32),
            'feature_extractor.stages.0.blocks.0.conv1.weight': torch.randn(32, 32, 3, 3),
        }
        mock_load.return_value = mock_weights

        trainer = TransferTrainer(spatial_net_model, spatial_config, device=torch.device('cpu'))

        # Test chess weight loading
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            result = trainer.load_chess_weights(f.name)

        assert len(trainer.transferred_layers) > 0
        assert len(trainer.new_layers) > 0


if __name__ == '__main__':
    pytest.main([__file__])