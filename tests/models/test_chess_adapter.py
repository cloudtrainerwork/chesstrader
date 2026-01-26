"""
Tests for chess weight adapter module.

Tests ChessWeightAdapter class functionality including weight loading,
spatial dimension adaptation, and compatibility checking.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.models.chess_adapter import ChessWeightAdapter, load_and_adapt_chess_weights
from src.models.spatial_net import SpatialNet, SpatialNetConfig


class TestChessWeightAdapter:
    """Test chess weight adapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create a chess weight adapter instance."""
        return ChessWeightAdapter(target_spatial_dims=(7, 6))

    @pytest.fixture
    def sample_chess_weights(self):
        """Create sample chess weights dictionary."""
        return {
            'conv1.weight': torch.randn(64, 3, 8, 8),
            'conv1.bias': torch.randn(64),
            'conv2.weight': torch.randn(128, 64, 8, 8),
            'conv2.bias': torch.randn(128),
            'residual_blocks.0.conv1.weight': torch.randn(64, 64, 3, 3),
            'residual_blocks.0.bn1.weight': torch.randn(64),
            'residual_blocks.0.bn1.bias': torch.randn(64),
            'residual_blocks.1.conv1.weight': torch.randn(128, 64, 3, 3),
            'fc.weight': torch.randn(512, 64),  # 8x8 = 64 spatial positions
            'fc.bias': torch.randn(512)
        }

    @pytest.fixture
    def sample_target_model(self):
        """Create a sample target model."""
        config = SpatialNetConfig(
            base_channels=64,
            num_stages=2,
            blocks_per_stage=1,
            output_features=512,
            strategy_classes=16
        )
        return SpatialNet(config)

    @pytest.fixture
    def temp_model_file(self, sample_chess_weights):
        """Create temporary model file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(sample_chess_weights, f.name)
            yield f.name
        os.unlink(f.name)

    def test_adapter_initialization(self):
        """Test adapter initialization with different spatial dimensions."""
        adapter = ChessWeightAdapter()
        assert adapter.target_height == 7
        assert adapter.target_width == 6

        adapter_custom = ChessWeightAdapter(target_spatial_dims=(5, 8))
        assert adapter_custom.target_height == 5
        assert adapter_custom.target_width == 8

    def test_load_pytorch_weights_success(self, adapter, temp_model_file):
        """Test successful loading of PyTorch weights."""
        weights = adapter.load_chess_weights(temp_model_file, model_type='pytorch')

        assert isinstance(weights, dict)
        assert 'conv1.weight' in weights
        assert weights['conv1.weight'].shape == (64, 3, 8, 8)

    def test_load_chess_weights_file_not_found(self, adapter):
        """Test loading weights from non-existent file."""
        with pytest.raises(FileNotFoundError):
            adapter.load_chess_weights('/nonexistent/model.pth')

    def test_load_chess_weights_unsupported_format(self, adapter):
        """Test loading weights from unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            with pytest.raises(ValueError, match="Unsupported model format"):
                adapter.load_chess_weights(f.name)

    def test_load_chess_weights_different_checkpoint_formats(self, adapter):
        """Test loading weights from different checkpoint formats."""
        # Test state_dict format
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            state_dict = {'conv1.weight': torch.randn(64, 3, 8, 8)}
            torch.save({'state_dict': state_dict}, f.name)
            weights = adapter.load_chess_weights(f.name)
            assert 'conv1.weight' in weights
        os.unlink(f.name)

        # Test model_state_dict format
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            state_dict = {'conv1.weight': torch.randn(64, 3, 8, 8)}
            torch.save({'model_state_dict': state_dict}, f.name)
            weights = adapter.load_chess_weights(f.name)
            assert 'conv1.weight' in weights
        os.unlink(f.name)

    def test_is_conv_weight(self, adapter):
        """Test convolutional weight detection."""
        # Test convolutional weights
        assert adapter._is_conv_weight('conv1.weight', torch.randn(64, 3, 3, 3))
        assert adapter._is_conv_weight('Conv2.weight', torch.randn(128, 64, 5, 5))
        assert adapter._is_conv_weight('spatial_encoder.weight', torch.randn(32, 16, 1, 1))

        # Test non-convolutional weights
        assert not adapter._is_conv_weight('bn1.weight', torch.randn(64))
        assert not adapter._is_conv_weight('bias', torch.randn(128))

    def test_adapt_conv2d_weight_8x8_to_7x6(self, adapter):
        """Test adapting conv2d weights from 8x8 to 7x6."""
        # Test standard conv weight
        weight = torch.randn(64, 32, 8, 8)
        adapted = adapter._adapt_conv2d_weight(weight)

        assert adapted.shape == (64, 32, 7, 6)

        # Test weight already correct size
        correct_weight = torch.randn(64, 32, 7, 6)
        adapted_correct = adapter._adapt_conv2d_weight(correct_weight)
        assert torch.equal(adapted_correct, correct_weight)

    def test_adapt_linear_weight(self, adapter):
        """Test adapting linear weights with spatial encoding."""
        # Test linear weight with 64 inputs (8x8 spatial)
        weight = torch.randn(512, 64)
        adapted = adapter._adapt_linear_weight(weight)

        assert adapted.shape == (512, 42)  # 7x6 = 42

        # Test linear weight without spatial encoding
        regular_weight = torch.randn(256, 128)
        adapted_regular = adapter._adapt_linear_weight(regular_weight)
        assert torch.equal(adapted_regular, regular_weight)

    def test_adapt_conv_layers(self, adapter, sample_chess_weights):
        """Test adapting all convolutional layers."""
        adapted = adapter.adapt_conv_layers(sample_chess_weights)

        # Check conv layer adaptations
        assert adapted['conv1.weight'].shape == (64, 3, 7, 6)
        assert adapted['conv2.weight'].shape == (128, 64, 7, 6)

        # Check that non-conv weights are preserved
        assert torch.equal(adapted['conv1.bias'], sample_chess_weights['conv1.bias'])

        # Check residual block conv adaptation
        assert adapted['residual_blocks.0.conv1.weight'].shape[2:] == (3, 3)  # 3x3 unchanged

    def test_adapt_conv_layers_with_mapping(self, adapter, sample_chess_weights):
        """Test adapting conv layers with custom key mapping."""
        mapping = {
            'conv1.weight': 'spatial_encoder.conv1.weight',
            'conv2.weight': 'feature_extractor.conv1.weight'
        }

        adapted = adapter.adapt_conv_layers(sample_chess_weights, mapping)

        assert 'spatial_encoder.conv1.weight' in adapted
        assert 'feature_extractor.conv1.weight' in adapted
        assert adapted['spatial_encoder.conv1.weight'].shape == (64, 3, 7, 6)

    def test_group_residual_weights(self, adapter, sample_chess_weights):
        """Test grouping weights by residual blocks."""
        groups = adapter._group_residual_weights(sample_chess_weights)

        assert 'residual_blocks.0' in groups
        assert 'residual_blocks.1' in groups
        assert 'conv1.weight' in groups['residual_blocks.0']
        assert 'bn1.weight' in groups['residual_blocks.0']

    def test_adapt_residual_blocks(self, adapter, sample_chess_weights):
        """Test adapting residual block weights."""
        adapted = adapter.adapt_residual_blocks(sample_chess_weights)

        # Check that residual block weights are properly grouped and adapted
        assert 'residual_blocks.0.conv1.weight' in adapted
        assert 'residual_blocks.1.conv1.weight' in adapted

        # Check conv weights are adapted while preserving 3x3 kernel size
        conv_weight = adapted['residual_blocks.0.conv1.weight']
        assert conv_weight.shape[2:] == (3, 3)

    def test_adapt_residual_blocks_with_mapping(self, adapter, sample_chess_weights):
        """Test adapting residual blocks with custom block mapping."""
        block_mapping = {
            'residual_blocks.0': 'feature_extractor.blocks.0',
            'residual_blocks.1': 'feature_extractor.blocks.1'
        }

        adapted = adapter.adapt_residual_blocks(sample_chess_weights, block_mapping)

        assert 'feature_extractor.blocks.0.conv1.weight' in adapted
        assert 'feature_extractor.blocks.1.conv1.weight' in adapted

    def test_weight_compatibility_check(self, adapter, sample_chess_weights, sample_target_model):
        """Test compatibility checking between chess weights and target model."""
        report = adapter.weight_compatibility_check(sample_chess_weights, sample_target_model)

        assert 'compatible_layers' in report
        assert 'incompatible_layers' in report
        assert 'missing_in_chess' in report
        assert 'extra_in_chess' in report
        assert 'size_mismatches' in report
        assert 'total_chess_params' in report
        assert 'total_target_params' in report

        # Should find some incompatibilities due to different architectures
        assert len(report['missing_in_chess']) > 0 or len(report['size_mismatches']) > 0

    def test_create_weight_mapping_exact(self, adapter, sample_target_model):
        """Test creating weight mapping with exact strategy."""
        # Create chess weights with some matching keys
        target_keys = list(sample_target_model.state_dict().keys())
        chess_weights = {key: torch.randn(2, 2) for key in target_keys[:3]}
        chess_weights.update({'extra_key': torch.randn(2, 2)})

        mapping = adapter.create_weight_mapping(chess_weights, sample_target_model, 'exact')

        # Should only map exactly matching keys
        for key in target_keys[:3]:
            assert mapping[key] == key
        assert 'extra_key' not in mapping

    def test_create_weight_mapping_auto(self, adapter, sample_chess_weights, sample_target_model):
        """Test creating weight mapping with auto strategy."""
        mapping = adapter.create_weight_mapping(sample_chess_weights, sample_target_model, 'auto')

        assert isinstance(mapping, dict)
        # Should create some mappings even if not perfect matches

    def test_find_fuzzy_match(self, adapter):
        """Test fuzzy matching functionality."""
        target_keys = {
            'spatial_encoder.conv1.weight',
            'feature_extractor.conv2.weight',
            'attention.query.weight'
        }

        # Test good match
        match = adapter._find_fuzzy_match('conv1.weight', target_keys)
        assert match == 'spatial_encoder.conv1.weight'

        # Test no match
        no_match = adapter._find_fuzzy_match('completely_different.weight', target_keys)
        assert no_match is None

    @patch('warnings.warn')
    def test_load_onnx_weights_warning(self, mock_warn, adapter):
        """Test that ONNX loading shows appropriate warning."""
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            weights = adapter._load_onnx_weights(Path(f.name))
            assert weights == {}
            mock_warn.assert_called_once()


class TestLoadAndAdaptChessWeights:
    """Test convenience function for loading and adapting chess weights."""

    @pytest.fixture
    def temp_chess_model(self):
        """Create temporary chess model file."""
        weights = {
            'conv1.weight': torch.randn(32, 3, 8, 8),
            'conv1.bias': torch.randn(32),
            'fc.weight': torch.randn(16, 32),
            'fc.bias': torch.randn(16)
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(weights, f.name)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def simple_target_model(self):
        """Create simple target model."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 16)
        )

    @patch('warnings.warn')
    def test_load_and_adapt_chess_weights_success(self, mock_warn, temp_chess_model, simple_target_model):
        """Test successful loading and adaptation."""
        adapted_weights = load_and_adapt_chess_weights(
            temp_chess_model,
            simple_target_model,
            target_spatial_dims=(7, 6)
        )

        assert isinstance(adapted_weights, dict)
        # Should contain some adapted weights
        assert len(adapted_weights) > 0

        # Check that warning is issued for incompatible layers
        mock_warn.assert_called()

    def test_load_and_adapt_chess_weights_custom_dims(self, temp_chess_model, simple_target_model):
        """Test loading with custom spatial dimensions."""
        adapted_weights = load_and_adapt_chess_weights(
            temp_chess_model,
            simple_target_model,
            target_spatial_dims=(5, 4)
        )

        assert isinstance(adapted_weights, dict)


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
            strategy_classes=8
        )
        return SpatialNet(config)

    @pytest.fixture
    def compatible_chess_weights(self):
        """Create chess weights compatible with SpatialNet structure."""
        return {
            'spatial_encoder.initial_conv.weight': torch.randn(32, 20, 8, 8),  # 20 input channels
            'spatial_encoder.initial_conv.bias': torch.randn(32),
            'feature_extractor.stages.0.blocks.0.conv1.weight': torch.randn(32, 32, 3, 3),
            'feature_extractor.stages.0.blocks.0.conv1.bias': torch.randn(32),
            'output_head.classifier.weight': torch.randn(8, 128),
            'output_head.classifier.bias': torch.randn(8)
        }

    def test_spatial_net_weight_adaptation(self, spatial_net_model, compatible_chess_weights):
        """Test adapting weights specifically for SpatialNet."""
        adapter = ChessWeightAdapter()

        # Adapt weights
        adapted_weights = adapter.adapt_conv_layers(compatible_chess_weights)

        # Check that spatial encoder conv weights are adapted
        spatial_conv_weight = adapted_weights['spatial_encoder.initial_conv.weight']
        assert spatial_conv_weight.shape == (32, 20, 7, 6)

        # Check compatibility with model
        report = adapter.weight_compatibility_check(adapted_weights, spatial_net_model)

        # Should have at least some compatible layers
        assert len(report['compatible_layers']) > 0

    def test_loading_adapted_weights_into_spatial_net(self, spatial_net_model):
        """Test loading adapted weights into SpatialNet model."""
        # Create mock adapted weights
        state_dict = spatial_net_model.state_dict()
        adapted_weights = {}

        for key, tensor in list(state_dict.items())[:3]:  # Take first 3 layers
            adapted_weights[key] = torch.randn_like(tensor)

        # Load weights (should not raise error)
        spatial_net_model.load_state_dict(adapted_weights, strict=False)

        # Verify weights were loaded
        for key in adapted_weights:
            assert torch.allclose(spatial_net_model.state_dict()[key], adapted_weights[key])


if __name__ == '__main__':
    pytest.main([__file__])