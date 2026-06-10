"""Deep Kernel Learning: Gaussian Process with Learned Kernel via Transformer Embeddings.

This module implements the core uncertainty quantification component for Phase 2 of the SFFM project.
Follows the framework of Wilson et al. (2016) and Chen, Fan & Wang (2023).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for irregular event timing.
    
    Handles variable-length sequences with actual timestamps.
    Reference: Su et al. (2021), applied to financial event sequences.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply rotary embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            positions: Optional positional indices or timestamps (batch, seq_len)
        
        Returns:
            Rotated embeddings (batch, seq_len, dim)
        """
        seq_len = x.shape[1]
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        else:
            positions = positions.float()
        
        # Compute angles
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        # Apply rotation matrix
        x_rot = torch.cat([
            x[..., :self.dim//2] * cos_emb[..., :self.dim//2] - x[..., self.dim//2:] * sin_emb[..., self.dim//2:],
            x[..., :self.dim//2] * sin_emb[..., :self.dim//2] + x[..., self.dim//2:] * cos_emb[..., self.dim//2:]
        ], dim=-1)
        
        return x_rot


class RBFKernel(nn.Module):
    """Radial Basis Function (RBF) kernel for Gaussian processes.
    
    κ(x, x') = exp(-γ ||x - x'||²)
    """
    
    def __init__(self, lengthscale_init: float = 1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(lengthscale_init)))
    
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute RBF kernel matrix.
        
        Args:
            x1: First set of points (m, dim)
            x2: Second set of points (n, dim)
        
        Returns:
            Kernel matrix (m, n)
        """
        if x2 is None:
            x2 = x1
        
        lengthscale = self.log_lengthscale.exp()
        
        # Squared distances
        x1_norm = (x1 ** 2).sum(dim=-1, keepdim=True)
        x2_norm = (x2 ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)
        dist_sq = x1_norm + x2_norm - 2 * torch.matmul(x1, x2.transpose(-2, -1))
        dist_sq = torch.clamp(dist_sq, min=1e-8)
        
        K = torch.exp(-dist_sq / (2 * lengthscale ** 2))
        return K


class VariationalSparseGP(nn.Module):
    """Variational Sparse Gaussian Process for scalable uncertainty quantification.
    
    Approximates posterior using M inducing points (M << N) and stochastic ELBO optimization.
    Reference: Tran et al. (2013), applied to sequential financial data.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, num_inducing: int = 1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        
        # Inducing points
        self.register_parameter(
            'inducing_points',
            nn.Parameter(torch.randn(num_inducing, input_dim))
        )
        
        # Variational parameters
        self.register_parameter(
            'variational_mean',
            nn.Parameter(torch.randn(num_inducing, output_dim))
        )
        
        self.register_parameter(
            'log_noise_var',
            nn.Parameter(torch.tensor(math.log(0.01)))
        )
        
        self.kernel = RBFKernel()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance at new points.
        
        Args:
            x: Input points (batch*seq_len, input_dim)
        
        Returns:
            mean: (batch*seq_len, output_dim)
            variance: (batch*seq_len, output_dim)
        """
        K_zz = self.kernel(self.inducing_points)
        K_xz = self.kernel(x, self.inducing_points)
        K_xx = self.kernel(x, x).diagonal()
        
        # Add numerical stability
        noise_var = self.log_noise_var.exp()
        K_zz_reg = K_zz + 1e-6 * torch.eye(self.num_inducing, device=K_zz.device)
        
        # Predictive mean
        K_zz_inv = torch.linalg.inv(K_zz_reg)
        mean = torch.matmul(K_xz, torch.matmul(K_zz_inv, self.variational_mean))
        
        # Predictive variance (simplified)
        var = K_xx.unsqueeze(-1) + noise_var
        
        return mean, torch.sqrt(torch.clamp(var, min=1e-8))


class DeepKernelSFFM(nn.Module):
    """Sequential Financial Foundation Model with Deep Kernel Learning.
    
    Combines:
    - Transformer encoder with RoPE embeddings
    - Learned kernel (Deep Kernel Learning)
    - Variational Sparse GP for uncertainty quantification
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        output_dim: int = 1,
        num_inducing: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rope = RotaryPositionalEmbedding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gp = VariationalSparseGP(
            input_dim=hidden_dim,
            output_dim=output_dim,
            num_inducing=num_inducing
        )
    
    def forward(self, event_ids: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: sequence -> latent representation -> uncertainty-aware predictions.
        
        Args:
            event_ids: Event token IDs (batch, seq_len)
            positions: Optional timestamps (batch, seq_len)
        
        Returns:
            mean: Predicted risk/price (batch, seq_len, 1)
            std: Predicted uncertainty (batch, seq_len, 1)
        """
        batch_size, seq_len = event_ids.shape
        
        x = self.embedding(event_ids)
        x = self.rope(x, positions)
        z = self.transformer(x)
        
        z_flat = z.reshape(-1, z.shape[-1])
        mean, std = self.gp(z_flat)
        
        mean = mean.reshape(batch_size, seq_len, -1)
        std = std.reshape(batch_size, seq_len, -1)
        
        return mean, std
