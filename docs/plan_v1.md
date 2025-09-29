# LBM Collision Operator Learning: First-Pass Plan v1

*Comprehensive plan for learning collision/transition operators in multicomponent/fluctuating Lattice Boltzmann systems*

## 1. Objectives & Scope

### Core Objective
- **Learn collision/transition operator** for multicomponent LBM systems (f, g populations)
- **Target**: Local mapping from pre-collision distributions → post-collision increments (Δᵢ)
- **Context**: Binary mixture with BGK collision, potential extension to fluctuating LBM
- **Direction**: Start simple (perceptron baseline), evolve toward diffusion models as surrogates

### Research Scope
- **Multicomponent coupling**: Joint (f,g) → (Δf,Δg) operator learning
- **Conservation-aware**: Preserve mass/momentum invariants
- **Fluctuating extensions**: Preparation for stochastic collision operators
- **Efficiency goal**: Computational speedup vs traditional BGK/MRT collision

### Out of Scope (Phase 1)
- Full rollout stability (beyond basic validation)
- Thermal/electromagnetic extensions
- Production-scale deployment

## 2. Data Inventory & Specification

### Current Data Assets
- **Verified**: `/data_loading.md` documents loader with verified output
- **Sample populations**:
  - `population_f/`: ~100 plotfile directories (plt1001001-plt1001100)
  - `population_g/`: ~100 plotfile directories (parallel structure)
- **Format**: HyperCLaw-V1.1 AMReX/BoxLib plotfiles
- **Fields**: 19 channels per population (f0-f18, g0-g18)

### Programmatic Discovery Required
- **Grid dimensions**: Header suggests 3D, need exact (Nx, Ny, Nz)
- **Timestep spacing**: Δt and physical time units
- **Boundary conditions**: Periodic vs other
- **Data range/scales**: Min/max values, typical magnitudes
- **Correlation structure**: Spatial/temporal dependencies

### Missing Information (Open Questions)
- **Full dataset specs**: Server data dimensions, total timesteps
- **Physics parameters**: Reynolds number, mixture ratios, forcing
- **Fluctuation presence**: Are thermal fluctuations included in current samples?
- **Conservation tolerances**: Expected precision for mass/momentum

## 3. Target Definition

### Learning Target Options
1. **Δᵢ increments**: fᵢ(t+1) = fᵢ(t) + Δᵢ(fₜ, gₜ)
2. **Transition probabilities**: P(f_{t+1}, g_{t+1} | f_t, g_t)
3. **Post-collision states**: f*ᵢ, g*ᵢ directly

### Invariants to Enforce
- **Mass conservation**: Σᵢ Δfᵢ = 0, Σᵢ Δgᵢ = 0
- **Momentum conservation**: Σᵢ cᵢ Δfᵢ + Σᵢ cᵢ Δgᵢ = 0
- **Physical bounds**: fᵢ, gᵢ ≥ 0 (or small tolerance)
- **Component-wise constraints**: TBD from dataset analysis

### Locality Assumptions
- **Input window**: Single-site (38 channels) vs small neighborhood
- **D3Q19 structure**: 19 velocity directions, 3D lattice
- **Coupling terms**: Linear vs nonlinear (f,g) interactions

## 4. Evaluation & Metrics

### Conservation Fidelity
- **Mass/momentum errors**: Mean absolute deviation, RMSE
- **Conservation drift**: Accumulation over multiple timesteps
- **Statistical tests**: Distribution of conservation violations

### Physical Validity
- **Non-negativity**: Fraction of negative distributions
- **Realizability**: H-theorem compliance, entropy production
- **Equilibrium approach**: Maxwell-Boltzmann convergence in quiescent regions

### Distributional Matching
- **Point-wise accuracy**: MSE on collision increments
- **Statistical moments**: Mean, variance, skewness of Δᵢ
- **Spatial correlations**: Two-point functions, structure factors
- **Temporal consistency**: Autocorrelation preservation

### Rollout Stability
- **Short-term**: 10-50 timestep prediction stability
- **Conservation tracking**: Cumulative mass/momentum drift
- **Blow-up detection**: NaN/Inf emergence, growth rates

### Computational Efficiency
- **Throughput**: Operations per lattice site vs BGK
- **Memory footprint**: Model size vs traditional collision
- **Scalability**: Performance on different grid sizes

## 5. Baselines & Candidate Approaches

### Baseline Methods
1. **BGK collision**: Traditional single-relaxation-time operator
2. **Linear regression**: Δᵢ = Σⱼ Aᵢⱼ (fⱼ + gⱼ) + bᵢ
3. **Random forest**: Non-parametric regression with physical constraints

### Neural Network Candidates
1. **Dense MLP**: 38 → hidden layers → 38, physics-informed loss
2. **Constrained networks**: Hard-coded conservation via architecture
3. **Mixture density networks**: Stochastic collision for fluctuating LBM

### Advanced Approaches (Literature-motivated)
1. **Diffusion models**: Conditional generation P(Δ | f,g) following Kohl et al. 2024
2. **Flow matching**: Continuous normalizing flows for collision dynamics
3. **Physics-informed neural ODEs**: Collision as learned dynamical system

### Constraint Enforcement Strategies
- **Soft constraints**: Penalty terms in loss function
- **Hard constraints**: Lagrange multipliers, projection layers
- **Augmented Lagrangian**: Iterative constraint satisfaction

## 6. Milestones (D0-D3), Risks & Open Questions

### D0: Data Specification (Week 1-2)
**Deliverables:**
- Programmatic data loader for sample populations
- Grid dimensions, channel layout, temporal spacing
- Statistical summary: ranges, distributions, correlations
- Conservation validation on raw data

**Risks:**
- Missing dependencies (numpy, yt) in environment
- Data format inconsistencies between f/g populations
- Insufficient sample size for meaningful statistics

### D1: Target Definition & Constraints (Week 3)
**Deliverables:**
- Formal mathematical definition of learning target
- Conservation constraint specifications with tolerances
- Locality analysis: single-site vs neighborhood requirements
- Baseline BGK implementation for comparison

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
│   ├── mlp.py             # Dense networks
│   ├── constrained.py     # Physics-informed architectures
│   └── diffusion.py       # Generative models (future)
├── experiments/           # Training and analysis scripts
│   ├── D0_data_spec.py    # Milestone deliverables
│   ├── D1_targets.py
│   ├── D2_evaluation.py
│   └── D3_assessment.py
└── docs/                  # Documentation
    ├── plan_v1.md         # This document
    ├── chat_notes.md      # ChatGPT feedback
    └── literature/        # References and summaries
```

### Code Standards
- **Type hints**: All functions with mypy compatibility
- **Docstrings**: NumPy style for physics equations
- **Testing**: pytest with conservation law validation
- **Notebooks**: Exploratory analysis, not production code

## 8. Reproducibility & Environment

### Environment Management
- **Python**: 3.9+ with conda/mamba environment
- **Core dependencies**: numpy, scipy, torch/jax, matplotlib
- **Data I/O**: yt (AMReX reader), h5py for processed data
- **ML frameworks**: PyTorch primary, JAX for diffusion models

### Reproducibility Requirements
- **Random seeds**: Fixed for data loading, model initialization
- **Version pinning**: requirements.txt with exact versions
- **Data versioning**: Hash validation for sample populations
- **Experiment tracking**: wandb/mlflow for systematic comparison

### Computational Resources
- **Local development**: Sample data, small models
- **Server scaling**: Full dataset, production training
- **GPU requirements**: CUDA for neural network training
- **Memory estimates**: 32GB+ for full population loading

## Next Steps

1. **Execute D0**: Implement data discovery scripts, validate conservation
2. **Literature review**: Diffusion models for fluid simulation (Kohl, Lienen)
3. **Baseline implementation**: BGK collision for comparison
4. **Community engagement**: Connect with LBM/ML-for-physics groups

## References

- **LBM foundations**: Krüger et al. "The Lattice Boltzmann Method" (2017)
- **Fluctuating LBM**: Dünweg, Schiller & Ladd, Phys. Rev. E 76, 036704 (2007)
- **Diffusion for fluids**: Kohl et al. arXiv:2309.01745 (2024)
- **Flow matching**: Lienen et al. arXiv:2306.01776 (2024)

---

*Document version: v1.0*
*Last updated: 2025-09-29*
*Next review: After D0 completion*