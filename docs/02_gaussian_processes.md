# Gaussian Processes & Deep Kernel Learning

## Why Gaussian Processes for Financial Uncertainty?

### The Problem with Point Estimates

Standard sequential models produce **point estimates** without confidence bounds:

```
Input: Customer transactions → Model → Risk: 0.72
                                        (no confidence bounds!)
```

**Regulatory gaps**:
- FCA Consumer Duty mandates transparency
- EU AI Act requires calibration demonstration
- No way to identify when model is uncertain

### The Gaussian Process Solution

GPs provide **principled, Bayesian uncertainty quantification**:

```
Input: Customer transactions 
  ↓
Transformer Encoder (learns representations)
  ↓
Gaussian Process Prior (over latent embeddings)
  ↓
Calibrated Predictive Distribution
  Mean: 0.72, Std: ±0.08 (95% confidence)
```

## Key Concepts

### 1. Gaussian Process Basics

A GP is a distribution over functions:
```
f(x) ~ GP(m(x), k(x, x'))

m(x) = prior mean function
k(x, x') = covariance (kernel) function
```

### 2. Deep Kernel Learning

Instead of fixed kernel, learn it from data:
```
                  Input Sequence
                       ↓
                 Transformer φ(·)
                       ↓
                   Latent z = φ(x)
                       ↓
            GP with learned kernel κ(φ(x), φ(x'))
                       ↓
                Predictive Distribution
              (mean + calibrated uncertainty)
```

### 3. Variational Sparse GP

Scales to large datasets using:
- **Inducing points**: M << N (e.g., 1,000 vs. 100M)
- **Variational inference**: O(M³) instead of O(N³)
- **Stochastic optimization**: mini-batch training

## Implementation in SFFM

See `src/sffm/models/deep_kernel_learning.py`:

1. **RotaryPositionalEmbedding** - handles irregular event timing
2. **RBFKernel** - learnable similarity measure
3. **VariationalSparseGP** - scalable inference
4. **DeepKernelSFFM** - complete model

## Calibration & Validation

**Expected Calibration Error (ECE)**:
```
ECE = |P(accurate | confidence) - confidence|

Goal: ECE < 0.05 (well-calibrated)
```

**Reliability Diagram**:
```
y-axis: Accuracy
x-axis: Predicted confidence

Perfect calibration: diagonal line y=x
```

## References

- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*
- Wilson et al. (2016). Deep Kernel Learning. *AISTATS 2016*
- Chen, Fan & Wang (2023). Multivariate Gaussian processes. *METRON*
