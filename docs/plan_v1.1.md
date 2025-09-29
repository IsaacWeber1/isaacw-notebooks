# LBM Collision Operator Learning: First-Pass Plan v1.1

*Comprehensive plan for learning collision/transition operators in multicomponent/fluctuating Lattice Boltzmann systems*

## 1. Objectives & Scope

### Core Objective
- **Learn collision/transition operator** for LBM systems using simplest possible approach
- **Target**: Single-site mapping from 38-channel state → 38-channel updates (Δᵢ)
- **State representation**: Combine f and g populations into one vector: x⃗ → Nx × Ny × Nz × (19 + 19)
- **Direction**: Start with simplest baseline (ridge/2-layer MLP), add structure only if needed

### Research Scope
- **Unified state**: Treat each lattice site as single 38-dimensional vector (no population separation)
- **Conservation-aware**: Preserve total mass/momentum across all 38 channels
- **Proof-of-concept**: Validate that ML can learn collision dynamics at all
- **Efficiency goal**: Computational speedup vs traditional BGK/MRT collision

### Out of Scope (Phase 1)
- Full rollout stability (beyond basic validation)
- Thermal/electromagnetic extensions
- Production-scale deployment

## 2. Data Inventory & Specification

### Current Data Assets
- **Verified**: `/data_loading.md` documents loader with verified output
- **Sample data**: ~100 timesteps each of population_f/ and population_g/
- **Format**: HyperCLaw-V1.1 AMReX/BoxLib plotfiles
- **Combined representation**: 38 channels per lattice site (f0-f18, g0-g18 flattened)
- **Target shape**: (Nx, Ny, Nz, 38) per timestep

### Programmatic Discovery Required
- **Grid dimensions**: Exact (Nx, Ny, Nz) from headers
- **Channel validation**: Confirm 19+19=38 channels total
- **Data ranges**: Min/max/mean for each of 38 channels
- **Conservation baseline**: Mass/momentum on raw data
- **Simple statistics**: Per-channel variance, correlations

### Missing Information (Open Questions)
- **Full dataset specs**: Server data dimensions, total timesteps
- **Physics parameters**: Reynolds number, mixture ratios, forcing
- **Fluctuation presence**: Are thermal fluctuations included in current samples?
- **Conservation tolerances**: Expected precision for mass/momentum

## 3. Target Definition

### Learning Target (Simplified)
- **Single target**: 38-channel increment Δ from 38-channel input
- **Mapping**: state(t) + Δ(state(t)) = state(t+1)
- **No population separation**: Treat f and g as unified 38-dimensional vector

### Invariants to Enforce
- **Total mass conservation**: Σᵢ₌₀³⁷ Δᵢ = 0 (sum over all 38 channels)
- **Total momentum conservation**: Σᵢ₌₀³⁷ cᵢ Δᵢ = 0 (using appropriate velocity vectors)
- **Physical bounds**: All 38 channels ≥ 0 after update

### Locality Assumptions
- **Single-site**: 38 input channels → 38 output channels per lattice point
- **No spatial coupling**: Each site processed independently
- **No population distinction**: f and g treated identically in model

## 4. Evaluation & Metrics

*All metrics below to be reported in `docs/first_pass_report.md` during D2.5*

### Conservation Fidelity
- **Mass/momentum errors**: Mean absolute deviation, RMSE (report in first_pass_report.md)
- **Conservation drift**: Accumulation over multiple timesteps (report in first_pass_report.md)
- **Statistical tests**: Distribution of conservation violations (report in first_pass_report.md)

### Physical Validity
- **Non-negativity**: Fraction of negative distributions (report in first_pass_report.md)
- **Realizability**: H-theorem compliance, entropy production (report in first_pass_report.md)
- **Equilibrium approach**: Maxwell-Boltzmann convergence in quiescent regions (report in first_pass_report.md)

### Distributional Matching
- **Point-wise accuracy**: MSE on collision increments (report in first_pass_report.md)
- **Statistical moments**: Mean, variance, skewness of Δᵢ (report in first_pass_report.md)
- **Spatial correlations**: Two-point functions, structure factors (report in first_pass_report.md)
- **Temporal consistency**: Autocorrelation preservation (report in first_pass_report.md)

### Rollout Stability
- **Short-term**: 10-50 timestep prediction stability (report in first_pass_report.md)
- **Conservation tracking**: Cumulative mass/momentum drift (report in first_pass_report.md)
- **Blow-up detection**: NaN/Inf emergence, growth rates (report in first_pass_report.md)

### Computational Efficiency
- **Throughput**: Operations per lattice site vs BGK (report in first_pass_report.md)
- **Memory footprint**: Model size vs traditional collision (report in first_pass_report.md)
- **Scalability**: Performance on different grid sizes (report in first_pass_report.md)

## 5. Baselines & Candidate Approaches

### Baseline Methods (D2.5 will select ONE)
1. **Ridge regression**: Δ = A × state + b, where A is 38×38 matrix with L2 regularization
2. **2-layer MLP**: 38 → 128 → 38, ReLU activation, MSE loss

### Future Candidates (Post-D2.5)
1. **Constrained networks**: Hard-coded conservation via architecture
2. **Diffusion models**: Conditional generation P(Δ | f,g) following Kohl et al. 2024
3. **Physics-informed neural ODEs**: Collision as learned dynamical system

### Constraint Enforcement Strategies
- **Soft constraints**: Penalty terms in loss function
- **Hard constraints**: Lagrange multipliers, projection layers
- **Augmented Lagrangian**: Iterative constraint satisfaction

## 6. Milestones (D0-D3), Risks & Open Questions

### D0: Data Specification (Week 1-2)
**Deliverables:**
- Programmatic data loader outputting (num_sites, 38) arrays
- Grid dimensions (Nx, Ny, Nz) and confirm 38 total channels
- Per-channel statistics: min/max/mean/std for all 38 channels
- Conservation validation: total mass/momentum on raw data

**Risks:**
- Missing dependencies (numpy, yt) in environment
- Data format inconsistencies between f/g populations
- Insufficient sample size for meaningful statistics

### D1: Target Definition & Constraints (Week 3)
**Deliverables:**
- Simple target: 38→38 mapping for collision increments
- Conservation constraints: sum(Δ)=0, sum(cᵢ×Δ)=0
- Single-site processing confirmed (no neighborhood needed)
- Reference BGK collision operator (if parameters available)

**Risks:**
- Ambiguous physics constraints from dataset
- Locality assumption insufficient for accuracy
- BGK parameters unknown/not documented

### D2: Evaluation Protocol (Week 4)
**Deliverables:**
- Automated conservation checking suite
- Physical validity tests (non-negativity, realizability)
- Rollout stability assessment framework
- Performance benchmarking infrastructure

**Risks:**
- Computational bottlenecks in evaluation
- Unclear thresholds for "acceptable" conservation errors
- Limited rollout horizon due to accumulated errors

### D2.5: Baseline Train & Report (Week 4-5)
**Deliverables:**
- Train exactly one baseline on flattened 38-channel data from local samples
- Input: (num_samples, 38), Output: (num_samples, 38)
- Save trained model to `models/baseline.{pt|pkl}`
- Write metrics to `artifacts/metrics.json`
- Generate `docs/first_pass_report.md` with:
  - Conservation errors on 38-channel updates
  - MSE on collision predictions
  - Training convergence
  - Performance vs complexity analysis

**Execution:**
- Single script: `experiments/first_pass_train_and_eval.py`
- Flatten f,g → single 38D vector per site
- No population-specific logic or branching

### D3: Risk Assessment & Next Steps (Week 5)
**Deliverables:**
- Comprehensive risk register with mitigation strategies
- Gap analysis: data/method/evaluation limitations
- Recommendations for model architecture selection
- Server dataset integration plan

**Open Questions:**
- Fluctuation modeling: deterministic vs stochastic operators?
- Multi-scale effects: sub-grid vs resolved collision dynamics
- Generalization: single operating point vs parametric families

## 7. Repository Organization Plan

### Directory Structure
```
isaacw-notebooks/
├── data_loading/           # Data I/O utilities
│   ├── discover_specs.py   # Programmatic data discovery
│   ├── loader_validation.py # Conservation checks
│   └── sample_analysis.py  # Statistical summaries
├── targets/                # Learning target definitions
│   ├── collision_operators.py # Δᵢ mappings
│   ├── constraints.py      # Physics invariants
│   └── baselines.py        # BGK/linear implementations
├── evaluation/             # Metrics and validation
│   ├── conservation.py     # Mass/momentum checks
│   ├── validity.py         # Physical bounds
│   ├── rollouts.py         # Stability assessment
│   └── benchmarks.py       # Performance comparison
├── models/                 # Neural network implementations
│   ├── baseline.pt         # Saved baseline model (after D2.5)
│   ├── mlp.py             # Dense networks
│   └── ridge.py           # Linear baseline
├── artifacts/              # Training outputs
│   └── metrics.json        # All evaluation metrics from D2.5
├── experiments/           # Training and analysis scripts
│   ├── D0_data_spec.py    # Milestone deliverables
│   ├── D1_targets.py
│   ├── D2_evaluation.py
│   ├── first_pass_train_and_eval.py  # Single-pass baseline training
│   └── D3_assessment.py
└── docs/                  # Documentation
    ├── plan_v1.1.md        # This document
    ├── chat_notes.md       # ChatGPT feedback
    ├── first_pass_report.md # D2.5 results summary
    └── literature/         # References and summaries
```

### Code Standards
- **Type hints**: All functions with mypy compatibility
- **Docstrings**: NumPy style for physics equations
- **Testing**: pytest with conservation law validation
- **Notebooks**: Exploratory analysis, not production code

## 8. Reproducibility & Environment

### Environment Management
- **Python**: 3.9+ with conda/mamba environment
- **Core dependencies**: numpy, scipy, torch/sklearn, matplotlib
- **Data I/O**: yt (AMReX reader), h5py for processed data
- **ML frameworks**: PyTorch or scikit-learn for baseline

### Reproducibility Requirements
- **Random seeds**: Fixed for data loading, model initialization
- **Version pinning**: requirements.txt with exact versions
- **Data versioning**: Hash validation for sample populations
- **Experiment tracking**: JSON metrics for systematic comparison

### Computational Resources
- **Local development**: Sample data, small models
- **Server scaling**: Full dataset, production training
- **GPU requirements**: Optional for baseline (CPU sufficient)
- **Memory estimates**: 8GB for baseline on samples

## Next Steps

1. **Run D2.5 script locally end-to-end**: Execute `experiments/first_pass_train_and_eval.py`
2. **Analyze first_pass_report.md**: Identify critical limitations
3. **Decide model progression**: Ridge vs MLP performance determines next architecture
4. **Prepare server deployment**: Based on local baseline results

## References

- **LBM foundations**: Krüger et al. "The Lattice Boltzmann Method" (2017)
- **Fluctuating LBM**: Dünweg, Schiller & Ladd, Phys. Rev. E 76, 036704 (2007)
- **Diffusion for fluids**: Kohl et al. arXiv:2309.01745 (2024)
- **Flow matching**: Lienen et al. arXiv:2306.01776 (2024)

---

*Document version: v1.1*
*Last updated: 2025-09-29*
*Next review: After D2.5 completion*