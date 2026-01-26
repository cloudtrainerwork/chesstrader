"""
Tests for residual blocks implementation.

Comprehensive test suite covering ResidualBlock, ResidualStack, and
ChessInspiredFeatureExtractor classes.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.residual_blocks import (
    ResidualBlock,
    ResidualStack,
    ChessInspiredFeatureExtractor
)


class TestResidualBlock:
    """Test cases for ResidualBlock class."""

    def test_residual_block_initialization(self):
        """Test ResidualBlock initialization with different configurations."""
        # Basic initialization
        block = ResidualBlock(in_channels=32, out_channels=64)
        assert block.in_channels == 32
        assert block.out_channels == 64
        assert block.stride == 1

        # Custom stride
        block_stride = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        assert block_stride.stride == 2

        # Skip connection should be created when channels differ
        assert block.skip_connection is not None

        # Skip connection should be None when channels match and stride=1
        same_channels_block = ResidualBlock(in_channels=32, out_channels=32)
        assert same_channels_block.skip_connection is None

    def test_residual_block_forward_pass(self):
        """Test ResidualBlock forward pass with different input sizes."""
        block = ResidualBlock(in_channels=3, out_channels=32)

        # Test with chess board dimensions (7x6)
        input_tensor = torch.randn(4, 3, 7, 6)  # batch=4
        output = block(input_tensor)

        # Check output shape
        assert output.shape == (4, 32, 7, 6)

        # Test with single batch
        single_input = torch.randn(1, 3, 7, 6)
        single_output = block(single_input)
        assert single_output.shape == (1, 32, 7, 6)

    def test_residual_block_gradient_flow(self):
        """Test that gradients flow through residual connections properly."""
        block = ResidualBlock(in_channels=32, out_channels=32)

        # Create input with requires_grad
        input_tensor = torch.randn(2, 32, 7, 6, requires_grad=True)

        # Forward pass
        output = block(input_tensor)

        # Compute loss (sum of all outputs)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that input has gradients
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)

    def test_residual_block_with_stride(self):
        """Test ResidualBlock with stride > 1."""
        block = ResidualBlock(in_channels=32, out_channels=64, stride=2)

        input_tensor = torch.randn(2, 32, 8, 8)  # Larger input for stride test
        output = block(input_tensor)

        # Stride 2 should halve spatial dimensions
        assert output.shape == (2, 64, 4, 4)

    def test_residual_block_weight_initialization(self):
        """Test that weights are properly initialized."""
        block = ResidualBlock(in_channels=16, out_channels=32)

        # Check that weights are not zero
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                assert not torch.all(module.weight == 0)
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm weights should be initialized to 1, bias to 0
                assert torch.all(module.weight == 1.0)
                assert torch.all(module.bias == 0.0)


class TestResidualStack:
    """Test cases for ResidualStack class."""

    def test_residual_stack_initialization(self):
        """Test ResidualStack initialization."""
        stack = ResidualStack(
            in_channels=3,
            hidden_channels=32,
            out_channels=64,
            num_blocks=4
        )

        assert stack.in_channels == 3
        assert stack.hidden_channels == 32
        assert stack.out_channels == 64
        assert stack.num_blocks == 4
        assert len(stack.blocks) == 4

    def test_residual_stack_single_block(self):
        """Test ResidualStack with single block."""
        stack = ResidualStack(
            in_channels=3,
            hidden_channels=32,
            out_channels=64,
            num_blocks=1
        )

        assert len(stack.blocks) == 1

        # Test forward pass
        input_tensor = torch.randn(2, 3, 7, 6)
        output = stack(input_tensor)
        assert output.shape == (2, 64, 7, 6)

    def test_residual_stack_invalid_blocks(self):
        """Test ResidualStack with invalid number of blocks."""
        with pytest.raises(ValueError, match="Number of blocks must be at least 1"):
            ResidualStack(
                in_channels=3,
                hidden_channels=32,
                out_channels=64,
                num_blocks=0
            )

    def test_residual_stack_forward_pass(self):
        """Test ResidualStack forward pass preserves chess board dimensions."""
        stack = ResidualStack(
            in_channels=4,  # Multi-channel input from MarketEncoder
            hidden_channels=64,
            out_channels=128,
            num_blocks=6
        )

        # Chess board dimensions
        input_tensor = torch.randn(8, 4, 7, 6)
        output = stack(input_tensor)

        # Should preserve spatial dimensions
        assert output.shape == (8, 128, 7, 6)

    def test_residual_stack_feature_maps(self):
        """Test getting intermediate feature maps."""
        stack = ResidualStack(
            in_channels=3,
            hidden_channels=32,
            out_channels=64,
            num_blocks=3
        )

        input_tensor = torch.randn(2, 3, 7, 6)
        feature_maps = stack.get_feature_maps(input_tensor)

        # Should have 3 feature maps (one from each block)
        assert len(feature_maps) == 3

        # Check shapes
        assert feature_maps[0].shape == (2, 32, 7, 6)  # First block output
        assert feature_maps[1].shape == (2, 32, 7, 6)  # Second block output
        assert feature_maps[2].shape == (2, 64, 7, 6)  # Final block output

    def test_residual_stack_parameter_count(self):
        """Test parameter counting."""
        stack = ResidualStack(
            in_channels=3,
            hidden_channels=32,
            out_channels=64,
            num_blocks=2
        )

        param_count = stack.count_parameters()

        # Should be positive number
        assert param_count > 0

        # Compare with manual count
        manual_count = sum(p.numel() for p in stack.parameters())
        assert param_count == manual_count


class TestChessInspiredFeatureExtractor:
    """Test cases for ChessInspiredFeatureExtractor class."""

    def test_feature_extractor_initialization(self):
        """Test ChessInspiredFeatureExtractor initialization."""
        extractor = ChessInspiredFeatureExtractor(
            input_channels=4,
            base_channels=32,
            num_stages=3,
            blocks_per_stage=2
        )

        assert extractor.input_channels == 4
        assert extractor.base_channels == 32
        assert extractor.num_stages == 3
        assert extractor.blocks_per_stage == 2
        assert len(extractor.stages) == 3

    def test_feature_extractor_spatial_output(self):
        """Test feature extractor with spatial output (no global pooling)."""
        extractor = ChessInspiredFeatureExtractor(
            input_channels=4,
            base_channels=32,
            num_stages=2,
            blocks_per_stage=2
        )

        # Test with chess board input
        input_tensor = torch.randn(4, 4, 7, 6)
        output = extractor(input_tensor)

        # Should preserve spatial dimensions, increase channels
        # Stage 1: 4 -> 32, Stage 2: 32 -> 64
        assert output.shape == (4, 64, 7, 6)

    def test_feature_extractor_global_features(self):
        """Test feature extractor with global feature output."""
        extractor = ChessInspiredFeatureExtractor(
            input_channels=3,
            base_channels=32,
            num_stages=2,
            blocks_per_stage=2,
            output_features=256
        )

        input_tensor = torch.randn(4, 3, 7, 6)
        output = extractor(input_tensor)

        # Should output global features
        assert output.shape == (4, 256)

    def test_feature_extractor_invalid_input_dimensions(self):
        """Test feature extractor with invalid input dimensions."""
        extractor = ChessInspiredFeatureExtractor(input_channels=3)

        # Wrong spatial dimensions
        wrong_spatial = torch.randn(2, 3, 8, 8)  # Should be 7x6
        with pytest.raises(ValueError, match="Expected spatial dimensions \\(7, 6\\)"):
            extractor(wrong_spatial)

        # Wrong number of dimensions
        wrong_dims = torch.randn(3, 7, 6)  # Missing batch dimension
        with pytest.raises(ValueError, match="Expected 4D input"):
            extractor(wrong_dims)

    def test_feature_extractor_channel_progression(self):
        """Test that channels progress correctly through stages."""
        extractor = ChessInspiredFeatureExtractor(
            input_channels=3,
            base_channels=16,
            num_stages=3,
            blocks_per_stage=1
        )

        # Channels should be: 3 -> 16 -> 32 -> 64
        input_tensor = torch.randn(2, 3, 7, 6)

        # Get intermediate outputs by manually passing through stages
        x = input_tensor
        stage_outputs = []

        for stage in extractor.stages:
            x = stage(x)
            stage_outputs.append(x.shape[1])  # Channel dimension

        # Check channel progression
        assert stage_outputs == [16, 32, 64]

    def test_feature_extractor_integration_with_market_encoder(self):
        """Test integration with MarketEncoder output format."""
        # MarketEncoder outputs (batch, 4, 7, 6) tensors
        # with channels: [Position, Regime, Confidence, Uncertainty]

        extractor = ChessInspiredFeatureExtractor(
            input_channels=4,  # Match MarketEncoder output
            base_channels=64,
            num_stages=3,
            blocks_per_stage=3,
            output_features=512
        )

        # Simulate MarketEncoder output
        market_encoded = torch.randn(16, 4, 7, 6)

        # Extract features
        features = extractor(market_encoded)

        # Should produce global feature vector
        assert features.shape == (16, 512)

        # Features should be reasonable (not NaN or inf)
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))


class TestIntegration:
    """Integration tests for residual block components."""

    def test_end_to_end_chess_pipeline(self):
        """Test complete pipeline from spatial input to features."""
        # Simulate complete pipeline
        batch_size = 8

        # 1. Multi-channel spatial input (from MarketEncoder)
        spatial_input = torch.randn(batch_size, 4, 7, 6)

        # 2. Feature extraction
        feature_extractor = ChessInspiredFeatureExtractor(
            input_channels=4,
            base_channels=32,
            num_stages=4,  # Deep network
            blocks_per_stage=2,
            output_features=256
        )

        features = feature_extractor(spatial_input)

        # 3. Verify output format
        assert features.shape == (batch_size, 256)
        assert not torch.any(torch.isnan(features))
        assert not torch.any(torch.isinf(features))

    def test_gradient_flow_through_deep_stack(self):
        """Test gradient flow through deep residual stack."""
        # Create deep feature extractor
        extractor = ChessInspiredFeatureExtractor(
            input_channels=3,
            base_channels=32,
            num_stages=4,
            blocks_per_stage=3,
            output_features=128
        )

        # Input with gradient tracking
        input_tensor = torch.randn(4, 3, 7, 6, requires_grad=True)

        # Forward pass
        output = extractor(input_tensor)

        # Compute loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist and are reasonable
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)

        # Check that gradients are not too large (gradient explosion check)
        grad_norm = torch.norm(input_tensor.grad)
        assert grad_norm < 100.0  # Reasonable upper bound

    def test_device_compatibility(self):
        """Test that models work on different devices."""
        extractor = ChessInspiredFeatureExtractor(
            input_channels=3,
            base_channels=16,
            num_stages=2,
            blocks_per_stage=1
        )

        # Test on CPU
        cpu_input = torch.randn(2, 3, 7, 6)
        cpu_output = extractor(cpu_input)
        assert cpu_output.device.type == 'cpu'

        # Test on GPU if available
        if torch.cuda.is_available():
            extractor_gpu = extractor.cuda()
            gpu_input = cpu_input.cuda()
            gpu_output = extractor_gpu(gpu_input)
            assert gpu_output.device.type == 'cuda'

    def test_batch_size_handling(self):
        """Test handling of different batch sizes including single batch."""
        extractor = ChessInspiredFeatureExtractor(
            input_channels=3,
            base_channels=32,
            num_stages=2,
            blocks_per_stage=2
        )

        # Test various batch sizes
        for batch_size in [1, 4, 16, 32]:
            input_tensor = torch.randn(batch_size, 3, 7, 6)
            output = extractor(input_tensor)

            expected_shape = (batch_size, 64, 7, 6)  # Spatial output
            assert output.shape == expected_shape


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])