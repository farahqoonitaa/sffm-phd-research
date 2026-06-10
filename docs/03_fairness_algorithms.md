# Fairness Algorithms: Six Approaches Compared

## The Fairness-Uncertainty Trilemma

```
                    Accuracy
                      /\
                     /  \
                    /    \
              High/      \Low
                /          \
          Fairness -------- Uncertainty
         (Equal)          (Calibrated)
```

## Five Fairness Algorithms

### 1. Demographic Parity
**Goal**: Equal average predictions across groups
- **Pros**: Simple, interpretable
- **Cons**: Ignores merit; can hurt qualified minorities
- **Use case**: Simple compliance checks

### 2. Equalized Odds
**Goal**: Equal error rates across groups
- **Pros**: Balanced approach
- **Cons**: Requires labeled outcomes
- **Use case**: Credit/insurance decisions

### 3. Calibration Fairness
**Goal**: Equal uncertainty across groups
- **Pros**: Integrates with GP directly
- **Cons**: May conflict with group fairness
- **Use case**: Regulatory compliance

### 4. Counterfactual Fairness
**Goal**: Decisions invariant to protected attribute
- **Pros**: Philosophically sound (causal fairness)
- **Cons**: Requires causal graph
- **Use case**: Research/auditing

### 5. Peer-Induced Fairness ⭐ (RECOMMENDED)
**Goal**: Distinguish discrimination from genuine disadvantage
- **Pros**: Legally defensible, works with informal-economy data
- **Cons**: Requires peer matching
- **Use case**: FCA compliance, heterogeneous populations

## Comparison Matrix

| Algorithm | Complexity | Regulatory | UQ Integration | Informal Economy |
|-----------|-----------|-----------|----------------|-----------------|
| Demographic Parity | Low | Low | Good | Risky |
| Equalized Odds | Medium | Medium | Good | Good |
| Calibration | Medium | High | Excellent | Medium |
| Counterfactual | High | Medium | Partial | Good |
| **Peer-Induced** | **Medium** | **Very High** | **Excellent** | **Excellent** |

## Expected Results (Phase 3)

| Model | Accuracy | Fairness Disc. | Calib. UQ |
|-------|----------|----------------|----------|
| Baseline | 92% | 45% | Good |
| Model A (DP) | 78% | 5% | Fair |
| Model B (EO) | 85% | 18% | Good |
| Model C (Cal) | 88% | 25% | Excellent |
| **Model E (PIF)** | **90%** | **12%** | **Excellent** |

## Implementation

See `src/sffm/fairness/fairness_losses.py` for all 5 algorithms ready to use in training:

```python
from src.sffm.fairness import FairnessLosses

loss_fn = FairnessLosses(method='peer_induced', lambda_fair=0.3)
fair_loss, metrics = loss_fn(pred, target, protected_attr, features)

# In training loop:
total_loss = task_loss + fair_loss
```

## References

- Fang, Chen & Ansell (2024). Peer-induced Fairness. *arXiv:2408.02558*
- Kehrenberg, Chen & Quadrianto (2020). Tuning fairness by balancing labels. *Frontiers in AI*
- Hurlin et al. (2024). The Fairness of Credit Scoring Models. *arXiv:2205.10200*
