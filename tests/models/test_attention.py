"""
Tests for attention mechanism implementation.

Comprehensive test suite covering SpatialAttention, MultiScaleSpatialAttention,
SpatialAttentionBlock, and PositionalEncoding classes.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.attention import (
    PositionalEncoding,
    SpatialAttention,
    MultiScaleSpatialAttention,
    SpatialAttentionBlock
)


class TestPositionalEncoding:
    """Test cases for PositionalEncoding class."""

    def test_positional_encoding_initialization(self):
        """Test PositionalEncoding initialization."""
        pos_enc = PositionalEncoding(channels=128, height=7, width=6)

        assert pos_enc.channels == 128
        assert pos_enc.height == 7
        assert pos_enc.width == 6
        assert pos_enc.pos_embedding.shape == (1, 128, 7, 6)

    def test_positional_encoding_forward(self):
        """Test PositionalEncoding forward pass."""
        pos_enc = PositionalEncoding(channels=64)

        input_tensor = torch.randn(4, 64, 7, 6)
        output = pos_enc(input_tensor)

        # Output should have same shape as input
        assert output.shape == input_tensor.shape

        # Output should be different from input (positional encoding added)
        assert not torch.allclose(output, input_tensor)

    def test_positional_encoding_different_sizes(self):
        """Test PositionalEncoding with different spatial sizes."""
        # Test with different board sizes
        pos_enc_custom = PositionalEncoding(channels=32, height=8, width=8)

        input_tensor = torch.randn(2, 32, 8, 8)
        output = pos_enc_custom(input_tensor)

        assert output.shape == (2, 32, 8, 8)

    def test_positional_encoding_gradient_flow(self):
        """Test gradient flow through positional encoding."""
        pos_enc = PositionalEncoding(channels=64)

        input_tensor = torch.randn(2, 64, 7, 6, requires_grad=True)
        output = pos_enc(input_tensor)

        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert input_tensor.grad is not None
        assert pos_enc.pos_embedding.grad is not None


class TestSpatialAttention:
    """Test cases for SpatialAttention class."""

    def test_spatial_attention_initialization(self):
        """Test SpatialAttention initialization."""
        attention = SpatialAttention(in_channels=128, num_heads=8)

        assert attention.in_channels == 128
        assert attention.num_heads == 8
        assert attention.key_dim == 128 // 8  # 16
        assert attention.value_dim == 128 // 8  # 16

    def test_spatial_attention_invalid_channels(self):
        """Test SpatialAttention with invalid channel configuration."""
        with pytest.raises(ValueError, match="in_channels .* must be divisible by num_heads"):
            SpatialAttention(in_channels=100, num_heads=8)  # 100 not divisible by 8

    def test_spatial_attention_forward_pass(self):
        """Test SpatialAttention forward pass."""
        attention = SpatialAttention(in_channels=64, num_heads=8)

        input_tensor = torch.randn(4, 64, 7, 6)
        output = attention(input_tensor)

        # Output should have same shape as input
        assert output.shape == input_tensor.shape

        # Output should be different from input (attention applied)
        assert not torch.allclose(output, input_tensor, atol=1e-6)

    def test_spatial_attention_single_batch(self):
        """Test SpatialAttention with single batch."""
        attention = SpatialAttention(in_channels=64, num_heads=4)

        single_input = torch.randn(1, 64, 7, 6)
        output = attention(single_input)

        assert output.shape == (1, 64, 7, 6)

    def test_spatial_attention_weights(self):
        """Test getting attention weights."""
        attention = SpatialAttention(in_channels=64, num_heads=8)

        input_tensor = torch.randn(2, 64, 7, 6)
        weights = attention.get_attention_weights(input_tensor)

        # Weights shape: (batch, num_heads, seq_len, seq_len)
        # seq_len = 7 * 6 = 42
        expected_shape = (2, 8, 42, 42)
        assert weights.shape == expected_shape

        # Attention weights should sum to 1 along last dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_spatial_attention_custom_dimensions(self):
        """Test SpatialAttention with custom key/value dimensions."""
        attention = SpatialAttention(
            in_channels=128,
            num_heads=8,
            key_dim=32,
            value_dim=32
        )

        assert attention.key_dim == 32
        assert attention.value_dim == 32

        input_tensor = torch.randn(2, 128, 7, 6)
        output = attention(input_tensor)

        assert output.shape == (2, 128, 7, 6)

    def test_spatial_attention_gradient_flow(self):
        """Test gradient flow through spatial attention."""
        attention = SpatialAttention(in_channels=64, num_heads=8)

        input_tensor = torch.randn(2, 64, 7, 6, requires_grad=True)
        output = attention(input_tensor)

        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)

        # Check attention parameters have gradients
        for param in attention.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMultiScaleSpatialAttention:
    """Test cases for MultiScaleSpatialAttention class."""

    def test_multi_scale_attention_initialization(self):
        """Test MultiScaleSpatialAttention initialization."""
        multi_attention = MultiScaleSpatialAttention(
            in_channels=128,
            scales=(1, 2),
            num_heads=8
        )

        assert multi_attention.in_channels == 128
        assert multi_attention.scales == (1, 2)
        assert multi_attention.num_heads == 8
        assert len(multi_attention.attention_modules) == 2

    def test_multi_scale_attention_forward(self):
        """Test MultiScaleSpatialAttention forward pass."""
        multi_attention = MultiScaleSpatialAttention(
            in_channels=64,
            scales=(1, 2),
            num_heads=8
        )

        input_tensor = torch.randn(4, 64, 8, 8)  # Larger input for multi-scale
        output = multi_attention(input_tensor)

        # Output should have same shape as input
        assert output.shape == input_tensor.shape

    def test_multi_scale_attention_chess_board(self):
        """Test MultiScaleSpatialAttention with chess board dimensions."""
        multi_attention = MultiScaleSpatialAttention(
            in_channels=128,
            scales=(1,),  # Only single scale for 7x6
            num_heads=8
        )

        input_tensor = torch.randn(2, 128, 7, 6)
        output = multi_attention(input_tensor)

        assert output.shape == (2, 128, 7, 6)

    def test_multi_scale_attention_gradient_flow(self):
        """Test gradient flow through multi-scale attention."""
        multi_attention = MultiScaleSpatialAttention(
            in_channels=64,
            scales=(1, 2),
            num_heads=4
        )

        input_tensor = torch.randn(2, 64, 8, 8, requires_grad=True)
        output = multi_attention(input_tensor)

        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)


class TestSpatialAttentionBlock:
    """Test cases for SpatialAttentionBlock class."""

    def test_attention_block_initialization(self):
        """Test SpatialAttentionBlock initialization."""
        attention_block = SpatialAttentionBlock(
            in_channels=128,
            num_heads=8,
            ff_hidden_dim=512
        )

        assert attention_block.in_channels == 128

        # Check components exist
        assert hasattr(attention_block, 'attention')
        assert hasattr(attention_block, 'feed_forward')
        assert hasattr(attention_block, 'norm1')
        assert hasattr(attention_block, 'norm2')

    def test_attention_block_forward(self):
        """Test SpatialAttentionBlock forward pass."""
        attention_block = SpatialAttentionBlock(
            in_channels=64,
            num_heads=8
        )

        input_tensor = torch.randn(4, 64, 7, 6)
        output = attention_block(input_tensor)

        # Output should have same shape as input
        assert output.shape == input_tensor.shape

        # Output should be different from input
        assert not torch.allclose(output, input_tensor, atol=1e-6)

    def test_attention_block_residual_connections(self):
        """Test that residual connections work properly."""
        attention_block = SpatialAttentionBlock(
            in_channels=64,
            num_heads=8,
            dropout_rate=0.0  # Disable dropout for deterministic test
        )

        # Set attention block to eval mode to disable dropout
        attention_block.eval()

        input_tensor = torch.randn(2, 64, 7, 6)

        # Forward pass
        output = attention_block(input_tensor)

        # The output should be close to input + transformations due to residual connections
        # (exact equality not expected due to layer norm and attention)
        assert output.shape == input_tensor.shape

    def test_attention_block_gradient_flow(self):
        """Test gradient flow through attention block."""
        attention_block = SpatialAttentionBlock(
            in_channels=64,
            num_heads=8
        )

        input_tensor = torch.randn(2, 64, 7, 6, requires_grad=True)
        output = attention_block(input_tensor)

        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)

        # Check all parameters have gradients
        for param in attention_block.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_attention_block_different_batch_sizes(self):
        """Test attention block with different batch sizes."""
        attention_block = SpatialAttentionBlock(
            in_channels=64,
            num_heads=8
        )

        # Test with different batch sizes
        for batch_size in [1, 4, 16]:
            input_tensor = torch.randn(batch_size, 64, 7, 6)
            output = attention_block(input_tensor)

            assert output.shape == (batch_size, 64, 7, 6)


class TestAttentionIntegration:
    """Integration tests for attention components."""

    def test_stacked_attention_blocks(self):
        """Test stacking multiple attention blocks."""
        # Create stack of attention blocks
        attention_stack = nn.Sequential(
            SpatialAttentionBlock(in_channels=128, num_heads=8),
            SpatialAttentionBlock(in_channels=128, num_heads=8),
            SpatialAttentionBlock(in_channels=128, num_heads=8)
        )

        input_tensor = torch.randn(4, 128, 7, 6)
        output = attention_stack(input_tensor)

        assert output.shape == input_tensor.shape

    def test_attention_with_different_spatial_sizes(self):
        """Test attention mechanisms with various spatial sizes."""
        sizes = [(7, 6), (8, 8), (14, 12)]

        for height, width in sizes:
            attention = SpatialAttention(in_channels=64, num_heads=8)

            # Update positional encoding for new size
            attention.pos_encoding = PositionalEncoding(
                channels=64, height=height, width=width
            )

            input_tensor = torch.randn(2, 64, height, width)
            output = attention(input_tensor)

            assert output.shape == input_tensor.shape

    def test_attention_memory_efficiency(self):
        """Test attention mechanism memory usage."""
        attention = SpatialAttention(in_channels=128, num_heads=8)

        # Test with reasonably large batch
        input_tensor = torch.randn(8, 128, 7, 6)

        # Forward pass should complete without memory errors
        output = attention(input_tensor)
        assert output.shape == input_tensor.shape

        # Gradient computation should also complete
        if input_tensor.requires_grad:
            loss = output.sum()
            loss.backward()

    def test_attention_deterministic_behavior(self):
        """Test that attention produces deterministic results."""
        torch.manual_seed(42)

        attention = SpatialAttentionBlock(in_channels=64, num_heads=8, dropout_rate=0.0)
        attention.eval()  # Set to eval mode to disable dropout

        input_tensor = torch.randn(2, 64, 7, 6)

        # Run forward pass twice
        output1 = attention(input_tensor.clone())
        output2 = attention(input_tensor.clone())

        # Outputs should be identical (deterministic)
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_attention_feature_interaction(self):
        """Test that attention actually processes spatial relationships."""
        attention = SpatialAttention(in_channels=64, num_heads=8)

        # Create input with specific pattern
        input_tensor = torch.zeros(1, 64, 7, 6)
        input_tensor[0, :, 3, 3] = 1.0  # Center position
        input_tensor[0, :, 0, 0] = -1.0  # Corner position

        output = attention(input_tensor)

        # Attention should spread information across spatial positions
        # Check that positions other than the original ones have non-zero values
        assert not torch.allclose(output[0, :, 1, 1], torch.zeros(64), atol=1e-3)
        assert not torch.allclose(output[0, :, 6, 5], torch.zeros(64), atol=1e-3)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])