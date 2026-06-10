# Sequential Financial Foundation Model (SFFM)
## Fairness-Aware Sequential Foundation Models for Financial Customer Behaviour: Uncertainty Quantification and Equitable Pricing in Heterogeneous Populations

**PhD Studentship** | University of Edinburgh Business School | Lloyds Banking Group Collaboration  
**Applicant**: Farah Qoonita Syuhaila  
**Email**: farahqoonita2@gmail.com  
**Supervisor**: Dr. Zexun Chen, Lecturer in Predictive Analytics  
**Deadline**: 18 June 2026

---

## 🎯 Project Overview

This repository houses the research, code, and documentation for a four-year FinTech PhD exploring **uncertainty-aware sequential foundation models for financial customer behaviour**, with emphasis on:

1. **Gaussian Process-based Uncertainty Quantification** (Deep Kernel Learning)
2. **In-Training Fairness Constraints** (Peer-Induced Fairness Framework)
3. **Heterogeneous Population Generalisation** (Western vs. Informal-Economy Data)

---

## 🔍 Core Research Questions

| RQ | Focus | Phase |
|---|---|---|
| **RQ1** | Can a sequential foundation model produce well-calibrated predictions across diverse customer populations? | Year 1 |
| **RQ2** | How do Gaussian-process priors enable principled uncertainty quantification in pricing decisions? | Year 2 |
| **RQ3** | How does peer-induced fairness scale from post-hoc audit to in-training constraint? | Year 3 |
| **RQ4** | Do FFMs trained on Western data systematically misrepresent uncertainty/fairness for informal-economy customers? | Year 4 |

---

## 📚 Quick Start: Learn by Doing

### Installation

```bash
# Clone the repository
git clone https://github.com/farahqoonitaa/sffm-phd-research.git
cd sffm-phd-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Steps

```bash
# 1. Explore Gaussian Processes (interactive)
jupyter notebook notebooks/01_gp_intro.ipynb

# 2. Run baseline SFFM
python src/experiments/exp_01_baseline.py

# 3. Test fairness algorithms
python src/experiments/exp_02_fairness.py

# 4. Run unit tests
pytest tests/
```

---

## 🏗️ Repository Structure

```
sffm-phd-research/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package configuration
│
├── docs/                               # Documentation
│   ├── 01_research_proposal.md         # Full 18-page proposal
│   ├── 02_gaussian_processes.md        # GP theory & Deep Kernel Learning
│   └── 03_fairness_algorithms.md       # Fairness framework deep-dive
│
├── src/
│   └── sffm/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── deep_kernel_learning.py # Phase 2: GP uncertainty
│       │   └── transformer.py           # Transformer encoder
│       │
│       ├── fairness/
│       │   ├── __init__.py
│       │   └── fairness_losses.py      # Phase 3: 5 fairness algorithms
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   └── loaders.py              # Data loading utilities
│       │
│       └── experiments/
│           ├── __init__.py
│           ├── exp_01_baseline.py      # Phase 1: Baseline SFFM
│           └── exp_02_fairness.py      # Phase 3: Fairness constraints
│
├── notebooks/
│   ├── 01_gp_intro.ipynb               # Learn Gaussian Processes
│   ├── 02_fairness_experiment.ipynb    # Try fairness algorithms
│   └── 03_uncertainty_calibration.ipynb # Calibration walkthrough
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py                  # Model tests
│   ├── test_fairness.py                # Fairness tests
│   └── test_uncertainty.py             # Uncertainty tests
│
and results/                            # (Generated after experiments)
```

---

## 🧠 Gaussian Processes: Core Innovation

### Why Gaussian Processes?

**Problem**: Standard sequential models (BERT4Rec, Transformers) produce **point estimates** without uncertainty.

```
Input: Customer transactions → Model → Risk: 0.72
                                        (no confidence bounds!)
```

**Solution**: Gaussian processes + Deep Kernel Learning

```
Input: Customer transactions 
  ↓
Transformer Encoder (learns representations)
  ↓
Gaussian Process Prior (over latent embeddings)
  ↓
Calibrated Predictive Distribution
  Mean: 0.72, Std: ±0.08 (95% confidence interval)
```

**Regulatory Alignment**:
- ✅ FCA Consumer Duty: "lenders must understand model uncertainty"
- ✅ EU AI Act: "high-risk AI systems must demonstrate calibration"

### Implementation

See `src/sffm/models/deep_kernel_learning.py` for:
- **RotaryPositionalEmbedding** (RoPE) — handles irregular event timing
- **RBFKernel** — learnable kernel for customer similarity
- **VariationalSparseGP** — scalable GP inference (O(M³) instead of O(N³))
- **DeepKernelSFFM** — complete end-to-end model

---

## ⚖️ Fairness Framework: Six Algorithms

### The Fairness-Uncertainty Trilemma

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

We explore **5 different fairness approaches**:

1. **Demographic Parity** — Equal avg predictions across groups (simple but risky)
2. **Equalized Odds** — Equal error rates (balanced approach)
3. **Calibration Fairness** — Equal uncertainty across groups
4. **Counterfactual Fairness** — Causal fairness (philosophically sound)
5. **Peer-Induced Fairness** ⭐ — "Would a similar peer from another group get better treatment?" (RECOMMENDED)

### Implementation

See `src/sffm/fairness/fairness_losses.py` for:
- `FairnessLosses` class with all 5 algorithms
- Integration with GP uncertainty
- Training-loop ready (add λ * fairness_loss to your objective)

**Expected Results** (Phase 3):
| Model | Accuracy | Fairness Discrimination | Calibration |
|-------|----------|----------------------|-------------|
| Baseline | 92% | 45% | Good |
| Model A (DP) | 78% | 5% | Fair |
| Model B (EO) | 85% | 18% | Good |
| Model C (Cal) | 88% | 25% | Excellent |
| **Model E (PIF)** | **90%** | **12%** | **Excellent** |

---

## 📖 Learning Resources

### For Understanding Gaussian Processes
1. Start: `notebooks/01_gp_intro.ipynb` (interactive tutorial)
2. Theory: `docs/02_gaussian_processes.md` (mathematical foundation)
3. Code: `src/sffm/models/deep_kernel_learning.py` (implementation)

### For Experimenting with Fairness
1. Start: `notebooks/02_fairness_experiment.ipynb` (hands-on lab)
2. Theory: `docs/03_fairness_algorithms.md` (algorithm comparison)
3. Code: `src/sffm/fairness/fairness_losses.py` (all 5 algorithms)

### For Uncertainty Calibration
1. Start: `notebooks/03_uncertainty_calibration.ipynb`
2. Metrics: Expected Calibration Error (ECE), reliability diagrams
3. Validation: Compare GP vs. Monte Carlo Dropout vs. Deep Ensembles

---

## 🚀 Experimental Phases

### Phase 1: Baseline SFFM (Year 1)
```bash
python src/experiments/exp_01_baseline.py
```
- Transformer encoder with multi-task pre-training
- Baseline evaluation on risk/pricing prediction
- Generates: `results/phase_1_baseline/`

### Phase 2: Uncertainty Quantification (Year 2)
```bash
python src/experiments/exp_02_uncertainty.py
```
- Deep Kernel Learning integration
- Calibration validation (ECE, reliability diagrams)
- Comparison: GP vs. MC Dropout vs. Ensembles
- Generates: `results/phase_2_uncertainty/`

### Phase 3: Fairness Constraints (Year 3)
```bash
python src/experiments/exp_03_fairness.py
```
- 5 fairness algorithms (A-E models)
- Fairness-accuracy-uncertainty trade-off analysis
- Generates: `results/phase_3_fairness/`

### Phase 4: Heterogeneity Analysis (Year 4)
```bash
python src/experiments/exp_04_heterogeneity.py
```
- UK vs. Indonesia customer comparison
- Informal-economy generalisation gap
- Policy recommendations
- Generates: `results/phase_4_heterogeneity/`

---

## 📊 Expected Contributions

### Methodological
✅ First integration of Deep-Kernel GP uncertainty into sequential financial foundation models

### Theoretical
✅ Extension of peer-induced fairness from post-hoc audit to in-training constraint

### Empirical
✅ First comparative evaluation across Western (UK) and informal-economy (Indonesia) data

### Policy
✅ Framework for FCA, OJK, ASEAN regulators on AI fairness & uncertainty standards

---

## 🔗 Key References

**Gaussian Processes**:
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*
- Chen, Fan & Wang (2023). Multivariate Gaussian processes. *METRON*
- Wilson et al. (2016). Deep Kernel Learning. *AISTATS 2016*

**Financial Foundation Models**:
- Braithwaite et al. (2025). nuFormer: Transformer for Financial Data
- Dou et al. (2025). TransactionGPT. *Visa Research*

**Fairness**:
- Fang, Chen & Ansell (2024). Peer-induced Fairness. *arXiv:2408.02558*
- Kehrenberg, Chen & Quadrianto (2020). Tuning fairness by balancing labels. *Frontiers in AI*

**Regulatory**:
- FCA (2023). Consumer Duty Regulation
- EU (2024). Artificial Intelligence Act (Regulation 2024/1689)

---

## 👤 About the Researcher

**Farah Qoonita Syuhaila**  
MBA (Cum Laude, ITB), BEng (Engineering Physics, ITS)  
Lead Business Analyst & AI Product Manager at Blendmedia  
PhD Candidate, University of Edinburgh Business School

**Email**: farahqoonita2@gmail.com  
**GitHub**: [@farahqoonitaa](https://github.com/farahqoonitaa)

---

## 📝 License

MIT License — see LICENSE file

---

**Last Updated**: June 2026 | **Status**: Active Research Project
