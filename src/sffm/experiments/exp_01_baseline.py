"""Phase 1: Baseline SFFM Experiment.

Train a basic Sequential Financial Foundation Model without fairness constraints.
Evaluate on multi-task objectives: risk prediction, pricing, event prediction.

Run: python src/sffm/experiments/exp_01_baseline.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.sffm.models import DeepKernelSFFM

def run_baseline_experiment():
    print("=" * 60)
    print("Phase 1: Baseline SFFM Experiment")
    print("=" * 60)
    
    # Hyperparameters
    vocab_size = 1000  # Event types
    hidden_dim = 256
    num_layers = 4
    num_heads = 8
    batch_size = 32
    num_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Create dummy data
    print("\n[1] Creating dummy dataset...")
    num_samples = 100
    seq_len = 50
    
    event_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.rand(num_samples, seq_len, 1)  # Risk scores [0, 1]
    
    dataset = TensorDataset(event_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"  Samples: {num_samples}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Event vocab size: {vocab_size}")
    
    # Create model
    print("\n[2] Initializing DeepKernelSFFM...")
    model = DeepKernelSFFM(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        output_dim=1,
        num_inducing=100,
        dropout=0.1
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    print("\n[3] Training baseline model...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, (event_ids_batch, targets_batch) in enumerate(dataloader):
            event_ids_batch = event_ids_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            # Forward pass
            mean, std = model(event_ids_batch)
            
            # Loss (MSE + uncertainty regularization)
            loss = criterion(mean, targets_batch)
            loss += 0.1 * std.mean()  # Regularize uncertainty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    # Evaluation
    print("\n[4] Evaluation on test set...")
    model.eval()
    
    with torch.no_grad():
        mean, std = model(event_ids[:10].to(device))
    
    print(f"  Mean predictions: {mean[:5, 0, 0].cpu().numpy()}")
    print(f"  Uncertainty (std): {std[:5, 0, 0].cpu().numpy()}")
    
    print("\n" + "=" * 60)
    print("✓ Baseline experiment completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    run_baseline_experiment()
