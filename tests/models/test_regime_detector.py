"""
Comprehensive test suite for RegimeDetector neural network.

Tests input/output shapes, forward pass functionality, device compatibility,
and numerical stability with various input scenarios.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from src.models.regime_detector import RegimeDetector


class TestRegimeDetector:
    """Test suite for RegimeDetector neural network."""

    @pytest.fixture
    def model(self):
        """Create a RegimeDetector instance for testing."""
        return RegimeDetector()

    @pytest.fixture
    def sample_input(self):
        """Create a sample 48-dimensional input tensor."""
        return torch.randn(1, 48)

    @pytest.fixture
    def batch_input(self):
        """Create a batch of 48-dimensional input tensors."""
        return torch.randn(4, 48)

    def test_model_initialization(self, model):
        """Test model initialization and architecture."""
        # Test default parameters
        assert model.input_dim == 48
        assert model.hidden_dims == (128, 64, 32)
        assert model.num_regimes == 8
        assert model.dropout_rate == 0.2

        # Test custom parameters
        custom_model = RegimeDetector(
            input_dim=64,
            hidden_dims=(256, 128, 64),
            num_regimes=10,
            dropout_rate=0.3
        )
        assert custom_model.input_dim == 64
        assert custom_model.hidden_dims == (256, 128, 64)
        assert custom_model.num_regimes == 10
        assert custom_model.dropout_rate == 0.3

    def test_model_architecture(self, model):
        """Test model architecture and parameter count."""
        # Check layers exist
        assert hasattr(model, 'input_layer')
        assert hasattr(model, 'hidden1')
        assert hasattr(model, 'hidden2')
        assert hasattr(model, 'regime_output')
        assert hasattr(model, 'confidence_output')

        # Check layer dimensions
        assert model.input_layer.in_features == 48
        assert model.input_layer.out_features == 128
        assert model.hidden1.in_features == 128
        assert model.hidden1.out_features == 64
        assert model.hidden2.in_features == 64
        assert model.hidden2.out_features == 32
        assert model.regime_output.in_features == 32
        assert model.regime_output.out_features == 8
        assert model.confidence_output.in_features == 32
        assert model.confidence_output.out_features == 1

        # Check batch norm layers
        assert hasattr(model, 'input_bn')
        assert hasattr(model, 'hidden1_bn')
        assert hasattr(model, 'hidden2_bn')

    def test_input_shape_validation(self, model):
        """Test input shape validation."""
        # Valid input
        valid_input = torch.randn(1, 48)
        output = model.eval()(valid_input)
        assert output.shape == (1, 9)

        # Invalid input dimensions
        with pytest.raises(ValueError, match="Expected input shape"):
            model(torch.randn(1, 47))  # Wrong feature dimension

        with pytest.raises(ValueError, match="Expected input shape"):
            model(torch.randn(48))  # Missing batch dimension

        with pytest.raises(ValueError, match="Expected input shape"):
            model(torch.randn(1, 48, 1))  # Too many dimensions

    def test_output_shape_validation(self, model, sample_input, batch_input):
        """Test output shape correctness."""
        model.eval()

        # Single sample
        output = model(sample_input)
        assert output.shape == (1, 9)
        assert output.dim() == 2

        # Batch
        batch_output = model(batch_input)
        assert batch_output.shape == (4, 9)
        assert batch_output.dim() == 2

        # Large batch
        large_batch = torch.randn(32, 48)
        large_output = model(large_batch)
        assert large_output.shape == (32, 9)

    def test_output_value_ranges(self, model, batch_input):
        """Test output value ranges and constraints."""
        model.eval()
        output = model(batch_input)

        # Split regime probabilities and confidence
        regime_probs = output[:, :8]
        confidence = output[:, 8:]

        # Test regime probabilities (should be softmax output)
        assert torch.all(regime_probs >= 0), "Regime probabilities should be non-negative"
        assert torch.all(regime_probs <= 1), "Regime probabilities should be <= 1"

        # Test that probabilities sum to 1 (within tolerance)
        prob_sums = torch.sum(regime_probs, dim=1)
        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), atol=1e-6, rtol=1e-6)

        # Test confidence scores (should be sigmoid output)
        assert torch.all(confidence >= 0), "Confidence scores should be non-negative"
        assert torch.all(confidence <= 1), "Confidence scores should be <= 1"

    def test_forward_pass_training_mode(self, model, batch_input):
        """Test forward pass in training mode."""
        model.train()
        output = model(batch_input)
        assert output.shape == (4, 9)
        assert output.requires_grad  # Should track gradients in training mode

    def test_forward_pass_eval_mode(self, model, sample_input):
        """Test forward pass in evaluation mode."""
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
            assert output.shape == (1, 9)
            assert not output.requires_grad  # Should not track gradients

    def test_predict_regime_method(self, model, batch_input):
        """Test the predict_regime method."""
        model.eval()
        predicted_regimes, regime_probs, confidence = model.predict_regime(batch_input)

        # Check output shapes
        assert predicted_regimes.shape == (4,)
        assert regime_probs.shape == (4, 8)
        assert confidence.shape == (4, 1)

        # Check predicted regime indices are valid
        assert torch.all(predicted_regimes >= 0)
        assert torch.all(predicted_regimes < 8)

        # Check that predicted regimes match argmax of probabilities
        expected_regimes = torch.argmax(regime_probs, dim=1)
        torch.testing.assert_close(predicted_regimes, expected_regimes)

    def test_calculate_uncertainty_method(self, model, batch_input):
        """Test the calculate_uncertainty method."""
        model.eval()
        uncertainty = model.calculate_uncertainty(batch_input)

        # Check output shape
        assert uncertainty.shape == (4, 1)

        # Check uncertainty values are in valid range [0, 1]
        assert torch.all(uncertainty >= 0)
        assert torch.all(uncertainty <= 1)

        # Test with uniform probabilities (high uncertainty)
        uniform_input = torch.zeros(1, 48)  # This should lead to more uniform probabilities
        uniform_uncertainty = model.calculate_uncertainty(uniform_input)
        assert uniform_uncertainty.shape == (1, 1)

    def test_device_handling(self, model):
        """Test device handling functionality."""
        # Test get_device
        device = model.get_device()
        assert isinstance(device, torch.device)

        # Test to_device (CPU to CPU)
        cpu_device = torch.device('cpu')
        model_cpu = model.to_device(cpu_device)
        assert model_cpu.get_device() == cpu_device

        # Test GPU if available
        if torch.cuda.is_available():
            gpu_device = torch.device('cuda:0')
            model_gpu = model.to_device(gpu_device)
            assert model_gpu.get_device() == gpu_device

            # Test forward pass on GPU
            gpu_input = torch.randn(2, 48).to(gpu_device)
            gpu_output = model_gpu(gpu_input)
            assert gpu_output.device == gpu_device
            assert gpu_output.shape == (2, 9)

    def test_numerical_stability(self, model):
        """Test numerical stability with edge cases."""
        model.eval()

        # Test with all zeros
        zero_input = torch.zeros(1, 48)
        zero_output = model(zero_input)
        assert not torch.isnan(zero_output).any(), "Output contains NaN for zero input"
        assert not torch.isinf(zero_output).any(), "Output contains Inf for zero input"

        # Test with large values
        large_input = torch.full((1, 48), 100.0)
        large_output = model(large_input)
        assert not torch.isnan(large_output).any(), "Output contains NaN for large input"
        assert not torch.isinf(large_output).any(), "Output contains Inf for large input"

        # Test with small values
        small_input = torch.full((1, 48), 1e-6)
        small_output = model(small_input)
        assert not torch.isnan(small_output).any(), "Output contains NaN for small input"
        assert not torch.isinf(small_output).any(), "Output contains Inf for small input"

        # Test with negative values
        negative_input = torch.full((1, 48), -10.0)
        negative_output = model(negative_input)
        assert not torch.isnan(negative_output).any(), "Output contains NaN for negative input"
        assert not torch.isinf(negative_output).any(), "Output contains Inf for negative input"

    def test_weight_initialization(self, model):
        """Test weight initialization."""
        # Check that weights are not all zeros
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert not torch.all(module.weight == 0), "Linear layer weights are all zero"
                if module.bias is not None:
                    assert torch.all(module.bias == 0), "Linear layer biases should be zero"

            elif isinstance(module, torch.nn.BatchNorm1d):
                assert torch.all(module.weight == 1), "BatchNorm weights should be initialized to 1"
                assert torch.all(module.bias == 0), "BatchNorm biases should be initialized to 0"

    def test_gradient_flow(self, model, batch_input):
        """Test gradient flow during backpropagation."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Forward pass
        output = model(batch_input)
        loss = torch.mean(output)  # Simple loss for testing

        # Backward pass
        loss.backward()

        # Check that gradients exist and are not all zero
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

        assert len(grad_norms) > 0, "No gradients found"
        assert any(norm > 0 for norm in grad_norms), "All gradients are zero"

    def test_reproducibility(self, model):
        """Test model reproducibility with same inputs."""
        model.eval()
        torch.manual_seed(42)
        input_tensor = torch.randn(2, 48)

        # First forward pass
        torch.manual_seed(0)
        output1 = model(input_tensor)

        # Second forward pass with same seed
        torch.manual_seed(0)
        output2 = model(input_tensor)

        # Outputs should be identical in eval mode
        torch.testing.assert_close(output1, output2)

    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes."""
        model.eval()
        batch_sizes = [1, 2, 8, 16, 32]

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 48)
            output = model(input_tensor)
            assert output.shape == (batch_size, 9), f"Failed for batch size {batch_size}"

    def test_model_string_representation(self, model):
        """Test model string representation."""
        model_str = str(model)
        assert "RegimeDetector" in model_str
        assert "48 → 128 → 64 → 32 → 8+1" in model_str
        assert "Total params:" in model_str
        assert "Trainable params:" in model_str
        assert "Dropout rate: 0.2" in model_str
        assert "Device: cpu" in model_str

    def test_no_dropout_during_inference(self, model):
        """Test that dropout is disabled during inference."""
        model.eval()
        input_tensor = torch.randn(5, 48)

        # Multiple forward passes should give identical results in eval mode
        output1 = model(input_tensor)
        output2 = model(input_tensor)
        output3 = model(input_tensor)

        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(output2, output3)

    @pytest.mark.parametrize("input_dim,hidden_dims,num_regimes", [
        (48, (128, 64, 32), 8),
        (32, (64, 32, 16), 4),
        (64, (256, 128, 64), 12),
    ])
    def test_custom_architectures(self, input_dim, hidden_dims, num_regimes):
        """Test custom model architectures."""
        model = RegimeDetector(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_regimes=num_regimes
        )

        # Test forward pass
        input_tensor = torch.randn(2, input_dim)
        model.eval()
        output = model(input_tensor)

        expected_output_dim = num_regimes + 1  # regimes + confidence
        assert output.shape == (2, expected_output_dim)

        # Test regime probabilities sum to 1
        regime_probs = output[:, :num_regimes]
        prob_sums = torch.sum(regime_probs, dim=1)
        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), atol=1e-6, rtol=1e-6)


class TestRegimeDetectorIntegration:
    """Integration tests for RegimeDetector with real-world scenarios."""

    def test_regime_feature_vector_compatibility(self):
        """Test compatibility with RegimeStateVector output."""
        # This test ensures the model works with actual feature vectors
        model = RegimeDetector()
        model.eval()

        # Simulate normalized feature vector from RegimeStateVector (48 dimensions, [-1, 1] range)
        feature_vector = torch.rand(1, 48) * 2 - 1  # Random values in [-1, 1]
        output = model(feature_vector)

        assert output.shape == (1, 9)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_batch_prediction_workflow(self):
        """Test typical batch prediction workflow."""
        model = RegimeDetector()
        model.eval()

        # Simulate batch of market states
        batch_features = torch.randn(10, 48)

        # Get predictions
        predicted_regimes, regime_probs, confidence = model.predict_regime(batch_features)
        uncertainty = model.calculate_uncertainty(batch_features)

        # Verify all outputs are coherent
        assert predicted_regimes.shape == (10,)
        assert regime_probs.shape == (10, 8)
        assert confidence.shape == (10, 1)
        assert uncertainty.shape == (10, 1)

        # Check that all predictions are valid
        assert torch.all(predicted_regimes >= 0) and torch.all(predicted_regimes < 8)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
        assert torch.all(uncertainty >= 0) and torch.all(uncertainty <= 1)

    def test_training_loop_simulation(self):
        """Test a simulated training loop."""
        model = RegimeDetector()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Simulate training data
        batch_size = 8
        features = torch.randn(batch_size, 48)
        labels = torch.randint(0, 8, (batch_size,))  # Random regime labels

        model.train()

        # Forward pass
        output = model(features)
        regime_logits = output[:, :8]  # Use logits for loss calculation

        # Loss calculation
        loss = criterion(regime_logits, labels)
        assert loss.item() > 0

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients were computed
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.all(param.grad == 0), "Some gradients should be non-zero"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])