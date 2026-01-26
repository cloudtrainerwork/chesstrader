"""
Self-attention mechanism for spatial position relationship modeling.

Implements SpatialAttention class with multi-head attention for processing
spatial relationships in 7x6 chess-inspired options strategy representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for 7x6 spatial board positions.

    Creates learnable position embeddings for each spatial location
    to help attention mechanism understand spatial relationships.
    """

    def __init__(self, channels: int, height: int = 7, width: int = 6):
        """
        Initialize positional encoding.

        Args:
            channels: Number of feature channels
            height: Board height (default: 7 for price zones)
            width: Board width (default: 6 for option legs)
        """
        super(PositionalEncoding, self).__init__()

        self.channels = channels
        self.height = height
        self.width = width

        # Create learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, height, width) * 0.02)

        # Alternative: Sinusoidal position encoding (comment out learnable version to use)
        # self.register_buffer('pos_embedding', self._create_sinusoidal_encoding())

    def _create_sinusoidal_encoding(self) -> torch.Tensor:
        """
        Create sinusoidal positional encoding (alternative to learnable).

        Returns:
            Fixed sinusoidal position encoding tensor
        """
        pos_enc = torch.zeros(1, self.channels, self.height, self.width)

        # Create position indices
        pos_h = torch.arange(self.height).float().unsqueeze(1).repeat(1, self.width)
        pos_w = torch.arange(self.width).float().unsqueeze(0).repeat(self.height, 1)

        # Generate encoding for each channel
        for c in range(self.channels):
            # Alternate between height and width based encodings
            if c % 4 == 0:
                pos_enc[0, c] = torch.sin(pos_h / (10000 ** (c / self.channels)))
            elif c % 4 == 1:
                pos_enc[0, c] = torch.cos(pos_h / (10000 ** (c / self.channels)))
            elif c % 4 == 2:
                pos_enc[0, c] = torch.sin(pos_w / (10000 ** (c / self.channels)))
            else:
                pos_enc[0, c] = torch.cos(pos_w / (10000 ** (c / self.channels)))

        return pos_enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input features.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Input with positional encoding added
        """
        return x + self.pos_embedding


class SpatialAttention(nn.Module):
    """
    Multi-head self-attention for spatial position relationship modeling.

    Processes spatial relationships between different positions on the 7x6
    options strategy board using multi-head attention mechanism.
    """

    def __init__(self,
                 in_channels: int,
                 num_heads: int = 8,
                 key_dim: Optional[int] = None,
                 value_dim: Optional[int] = None,
                 dropout_rate: float = 0.1):
        """
        Initialize spatial attention module.

        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads (default: 8)
            key_dim: Dimension of keys/queries (default: in_channels // num_heads)
            value_dim: Dimension of values (default: in_channels // num_heads)
            dropout_rate: Dropout rate for attention weights
        """
        super(SpatialAttention, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads

        # Calculate dimensions
        if key_dim is None:
            key_dim = in_channels // num_heads
        if value_dim is None:
            value_dim = in_channels // num_heads

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = math.sqrt(key_dim)

        # Ensure dimensions are compatible
        if in_channels % num_heads != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})")

        # Linear projections for keys, queries, values
        self.query_proj = nn.Linear(in_channels, num_heads * key_dim)
        self.key_proj = nn.Linear(in_channels, num_heads * key_dim)
        self.value_proj = nn.Linear(in_channels, num_heads * value_dim)

        # Output projection
        self.output_proj = nn.Linear(num_heads * value_dim, in_channels)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(in_channels)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout_rate)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(in_channels)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize projection weights."""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Attention-processed tensor of same shape
        """
        batch_size, channels, height, width = x.shape

        # Add positional encoding
        x_pos = self.pos_encoding(x)

        # Store residual connection
        residual = x_pos

        # Reshape for attention: (batch, height*width, channels)
        x_flat = x_pos.view(batch_size, channels, height * width).transpose(1, 2)

        # Generate queries, keys, values
        queries = self.query_proj(x_flat)  # (batch, seq_len, num_heads * key_dim)
        keys = self.key_proj(x_flat)       # (batch, seq_len, num_heads * key_dim)
        values = self.value_proj(x_flat)   # (batch, seq_len, num_heads * value_dim)

        # Reshape for multi-head attention
        seq_len = height * width

        queries = queries.view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.value_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, values)

        # Reshape back to original format
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.num_heads * self.value_dim)

        # Output projection
        output = self.output_proj(attention_output)

        # Reshape back to spatial format
        output = output.transpose(1, 2).view(batch_size, channels, height, width)

        # Residual connection and layer norm
        output = self.layer_norm((output + residual).contiguous().view(batch_size, height * width, channels))
        output = output.view(batch_size, channels, height, width)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization and analysis.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size, channels, height, width = x.shape

        # Add positional encoding
        x_pos = self.pos_encoding(x)

        # Reshape for attention
        x_flat = x_pos.view(batch_size, channels, height * width).transpose(1, 2)

        # Generate queries and keys
        queries = self.query_proj(x_flat)
        keys = self.key_proj(x_flat)

        # Reshape for multi-head attention
        seq_len = height * width
        queries = queries.view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)

        # Compute attention scores and weights
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        return attention_weights

    def __repr__(self) -> str:
        """String representation of SpatialAttention."""
        return (f"SpatialAttention(\n"
                f"  in_channels={self.in_channels}\n"
                f"  num_heads={self.num_heads}\n"
                f"  key_dim={self.key_dim}\n"
                f"  value_dim={self.value_dim}\n"
                f"  scale={self.scale:.3f}\n"
                f")")


class MultiScaleSpatialAttention(nn.Module):
    """
    Multi-scale spatial attention for capturing relationships at different scales.

    Combines attention at different spatial resolutions to capture both
    local and global relationships in the options strategy representation.
    """

    def __init__(self,
                 in_channels: int,
                 scales: Tuple[int, ...] = (1, 2),
                 num_heads: int = 8,
                 dropout_rate: float = 0.1):
        """
        Initialize multi-scale spatial attention.

        Args:
            in_channels: Number of input channels
            scales: Downsampling scales for multi-scale processing
            num_heads: Number of attention heads per scale
            dropout_rate: Dropout rate
        """
        super(MultiScaleSpatialAttention, self).__init__()

        self.in_channels = in_channels
        self.scales = scales
        self.num_heads = num_heads

        # Create attention modules for each scale
        self.attention_modules = nn.ModuleDict()

        for scale in scales:
            self.attention_modules[f'scale_{scale}'] = SpatialAttention(
                in_channels=in_channels,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Conv2d(in_channels * len(scales), in_channels, kernel_size=1)

        # Final layer normalization
        self.final_norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Multi-scale attention output
        """
        batch_size, channels, height, width = x.shape
        scale_outputs = []

        for scale in self.scales:
            if scale == 1:
                # Original scale
                scale_input = x
                attention_name = f'scale_{scale}'
                scale_output = self.attention_modules[attention_name](scale_input)
            else:
                # Downsampled scale
                scale_input = F.avg_pool2d(x, kernel_size=scale, stride=scale)

                # Create temporary attention module with correct dimensions
                downscaled_height = scale_input.shape[2]
                downscaled_width = scale_input.shape[3]

                # Create a new SpatialAttention for this scale with correct dimensions
                temp_attention = SpatialAttention(
                    in_channels=self.in_channels,
                    num_heads=self.num_heads,
                    dropout_rate=0.1
                )
                temp_attention.pos_encoding = PositionalEncoding(
                    channels=self.in_channels,
                    height=downscaled_height,
                    width=downscaled_width
                )

                scale_output = temp_attention(scale_input)

            # Upsample back to original size if needed
            if scale != 1:
                scale_output = F.interpolate(
                    scale_output,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )

            scale_outputs.append(scale_output)

        # Concatenate multi-scale features
        multi_scale_features = torch.cat(scale_outputs, dim=1)

        # Fuse features
        fused_output = self.fusion(multi_scale_features)

        # Apply layer normalization
        fused_output = self.final_norm(
            fused_output.view(batch_size, height * width, channels)
        ).view(batch_size, channels, height, width)

        # Residual connection
        output = fused_output + x

        return output


class SpatialAttentionBlock(nn.Module):
    """
    Complete spatial attention block with residual connections and normalization.

    Combines spatial attention with feed-forward network and residual connections
    similar to transformer architecture but optimized for spatial data.
    """

    def __init__(self,
                 in_channels: int,
                 num_heads: int = 8,
                 ff_hidden_dim: Optional[int] = None,
                 dropout_rate: float = 0.1):
        """
        Initialize spatial attention block.

        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads
            ff_hidden_dim: Hidden dimension for feed-forward network
            dropout_rate: Dropout rate
        """
        super(SpatialAttentionBlock, self).__init__()

        self.in_channels = in_channels

        if ff_hidden_dim is None:
            ff_hidden_dim = in_channels * 4

        # Spatial attention
        self.attention = SpatialAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, ff_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(ff_hidden_dim, in_channels, kernel_size=1),
            nn.Dropout(dropout_rate)
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention block.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Processed tensor with same shape
        """
        batch_size, channels, height, width = x.shape

        # First sub-layer: spatial attention with residual connection
        attn_input = self.norm1(x.view(batch_size, height * width, channels))
        attn_input = attn_input.view(batch_size, channels, height, width)

        attn_output = self.attention(attn_input)
        x = x + attn_output

        # Second sub-layer: feed-forward with residual connection
        ff_input = self.norm2(x.view(batch_size, height * width, channels))
        ff_input = ff_input.view(batch_size, channels, height, width)

        ff_output = self.feed_forward(ff_input)
        x = x + ff_output

        return x

    def __repr__(self) -> str:
        """String representation of SpatialAttentionBlock."""
        return (f"SpatialAttentionBlock(\n"
                f"  in_channels={self.in_channels}\n"
                f"  attention={self.attention}\n"
                f")")


if __name__ == '__main__':
    # Test the attention modules
    batch_size = 4
    channels = 128
    height, width = 7, 6

    # Test SpatialAttention
    attention = SpatialAttention(in_channels=channels, num_heads=8)
    x = torch.randn(batch_size, channels, height, width)

    print(f"Input shape: {x.shape}")
    output = attention(x)
    print(f"Attention output shape: {output.shape}")

    # Test attention weights
    weights = attention.get_attention_weights(x)
    print(f"Attention weights shape: {weights.shape}")

    # Test MultiScaleSpatialAttention
    multi_scale_attention = MultiScaleSpatialAttention(in_channels=channels)
    multi_output = multi_scale_attention(x)
    print(f"Multi-scale attention output shape: {multi_output.shape}")

    # Test SpatialAttentionBlock
    attention_block = SpatialAttentionBlock(in_channels=channels)
    block_output = attention_block(x)
    print(f"Attention block output shape: {block_output.shape}")

    print("All attention modules working correctly!")