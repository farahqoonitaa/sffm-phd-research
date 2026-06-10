"""Tests for fairness algorithms."""

import torch
import pytest
from src.sffm.fairness import FairnessLosses


def test_demographic_parity():
    """Test demographic parity fairness."""
    loss_fn = FairnessLosses(method='demographic_parity')
    pred = torch.rand(100)
    target = torch.rand(100)
    protected_attr = torch.randint(0, 2, (100,))
    features = torch.randn(100, 10)
    
    loss, metrics = loss_fn(pred, target, protected_attr, features)
    assert loss.item() >= 0
    assert 'dp_loss' in metrics


def test_equalized_odds():
    """Test equalized odds fairness."""
    loss_fn = FairnessLosses(method='equalized_odds')
    pred = torch.rand(100)
    target = torch.randint(0, 2, (100,)).float()
    protected_attr = torch.randint(0, 2, (100,))
    features = torch.randn(100, 10)
    
    loss, metrics = loss_fn(pred, target, protected_attr, features)
    assert loss.item() >= 0
    assert 'eo_loss' in metrics


def test_peer_induced_fairness():
    """Test peer-induced fairness (RECOMMENDED)."""
    loss_fn = FairnessLosses(method='peer_induced')
    pred = torch.rand(100)
    target = torch.rand(100)
    protected_attr = torch.randint(0, 2, (100,))
    features = torch.randn(100, 10)
    
    loss, metrics = loss_fn(pred, target, protected_attr, features)
    assert loss.item() >= 0
    assert 'pif_loss' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
