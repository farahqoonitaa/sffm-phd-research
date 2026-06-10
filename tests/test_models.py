"""Tests for SFFM models."""

import torch
import pytest
from src.sffm.models import DeepKernelSFFM, RotaryPositionalEmbedding, RBFKernel


def test_rotary_positional_embedding():
    """Test RoPE implementation."""
    rope = RotaryPositionalEmbedding(dim=128)
    x = torch.randn(2, 10, 128)  # (batch, seq_len, dim)
    y = rope(x)
    assert y.shape == x.shape


def test_rbf_kernel():
    """Test RBF kernel."""
    kernel = RBFKernel()
    x1 = torch.randn(5, 10)
    x2 = torch.randn(3, 10)
    K = kernel(x1, x2)
    assert K.shape == (5, 3)
    assert torch.all(K >= 0) and torch.all(K <= 1)


def test_deep_kernel_sffm():
    """Test DeepKernelSFFM model."""
    model = DeepKernelSFFM(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        output_dim=1,
        num_inducing=50
    )
    
    event_ids = torch.randint(0, 100, (4, 20))  # (batch, seq_len)
    mean, std = model(event_ids)
    
    assert mean.shape == (4, 20, 1)
    assert std.shape == (4, 20, 1)
    assert torch.all(std > 0)  # Uncertainty must be positive


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
