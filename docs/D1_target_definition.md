# D1: Target Definition & Constraints

**Milestone**: D1 - Target Definition & Constraints
**Date**: 2025-09-29
**Status**: COMPLETED

## Executive Summary

Successfully defined the learning target and physics constraints for 38-channel LBM collision operator learning. The target is formulated as collision increments with realistic conservation tolerances based on D0 findings.

**Key Deliverables:**
- **Target**: Collision increments Δ = state[t+1] - state[t]
- **Constraints**: Mass (5% tolerance) and momentum (2% tolerance) conservation
- **Implementation**: Complete Python modules with validation on real data

## Target Definition

### Mathematical Formulation

For the simplified 38-channel approach:

**Input State**: `x ∈ ℝ³⁸` where `x = [f₀, f₁, ..., f₁₈, g₀, g₁, ..., g₁₈]`

**Learning Target**: `Δ ∈ ℝ³⁸` such that `x[t+1] = x[t] + Δ(x[t])`

**Objective**: Learn function `Δ(·)` that maps pre-collision states to collision increments

### Design Rationale

1. **Incremental Learning**: Δ represents the "change" caused by collision
2. **Unified Representation**: No f/g separation required
3. **Additive Update**: Simple and numerically stable
4. **Locality**: Single-site processing (38 → 38 mapping)

## Physics Constraints

### Conservation Laws

Based on D0 analysis showing 1-4% violations in real data:

#### Mass Conservation
- **Constraint**: `|Σᵢ₌₀³⁷ Δᵢ| < τₘ`
- **Tolerance**: `τₘ = 0.05` (5% relative tolerance)
- **Rationale**: Observed violations ~1-4% in discrete LBM

#### Momentum Conservation
- **Constraint**: `|Σᵢ₌₀³⁷ cᵢ Δᵢ| < τₚ` per component
- **Tolerance**: `τₚ = 0.02` (2% per component)
- **Velocity Set**: D3Q19 applied to both f and g populations

### Physical Bounds
- **Non-negativity**: `x[t] + Δ ≥ -10⁻¹⁰` (small numerical tolerance)
- **Bounded growth**: `|Δᵢ| < 0.5` (prevent explosive changes)

### Constraint Implementation

**Soft Constraints** (preferred for ML training):
```python
loss = mse_loss + λₘ × mass_penalty + λₚ × momentum_penalty

mass_penalty = |Σ Δᵢ|
momentum_penalty = |Σ cᵢ × Δᵢ|
```

**Hard Constraints** (alternative):
- Lagrange multipliers
- Projection layers
- Constrained optimization

## Implementation

### Module Structure
```
targets/
├── __init__.py                 # Module interface
├── collision_targets.py        # Target computation and analysis
├── constraints.py              # Physics constraint checking
└── validation.py               # Real data validation
```

### Key Functions

#### Target Computation
```python
from targets import compute_collision_deltas, create_training_dataset

# Compute targets from data
deltas = compute_collision_deltas(pre_states, post_states)

# Create ML dataset
X, y = create_training_dataset(pre_states, post_states)
```

#### Constraint Checking
```python
from targets import validate_collision_pair

# Comprehensive validation
results = validate_collision_pair(pre_states, post_states)
print(results['summary']['overall_status'])  # PASS/VIOLATIONS_DETECTED
```

#### Penalty Functions
```python
from targets import create_constraint_penalty_loss

# For training loss
penalty, components = create_constraint_penalty_loss(predicted_deltas, pre_states)
total_loss = mse_loss + penalty
```

## Validation on Real Data

### Test Configuration
- **Data**: 5 timesteps from local samples
- **Samples**: 10,000 collision pairs
- **Validation**: Target definition + constraint checking

### Key Findings

#### Target Statistics
- **Delta magnitude**: Mean ~10⁻³, Max ~10⁻²
- **Reconstruction**: Perfect (error < 10⁻¹⁰)
- **Distribution**: Physically reasonable increments

#### Constraint Analysis
- **Mass violations**: ~3% of samples exceed 5% tolerance
- **Momentum violations**: ~2% of samples exceed 2% tolerance
- **Physical bounds**: >99.9% satisfy non-negativity

#### Status: **READY FOR TRAINING** ✅

## Model-Ready Specification

### Training Setup
```python
# Input/Output specification
input_shape = (batch_size, 38)    # Pre-collision states
output_shape = (batch_size, 38)   # Predicted deltas
target_shape = (batch_size, 38)   # True deltas from data

# Loss function
def physics_informed_loss(y_pred, y_true, x_input):
    mse = mean_squared_error(y_pred, y_true)
    penalty, _ = create_constraint_penalty_loss(y_pred, x_input)
    return mse + 0.1 * penalty  # Adjust penalty weight as needed
```

### Baseline Models (D2.5 Candidates)
1. **Ridge Regression**: `Δ = A × x + b` with L2 regularization
2. **2-Layer MLP**: `38 → 128 → 38` with ReLU, MSE + penalty loss

## Comparison to Traditional Approaches

| Aspect | Traditional BGK | Learned Δ (This Work) |
|--------|-----------------|------------------------|
| **Input** | f, g separately | Combined 38D vector |
| **Output** | Post-collision f, g | Collision increments Δ |
| **Conservation** | Exact (continuous) | Soft constraints (~5%) |
| **Computation** | Matrix operations | Neural network forward pass |
| **Flexibility** | Fixed physics | Data-driven adaptation |

## Next Steps for D2.5

### Immediate Actions
1. **Select baseline model**: Ridge regression or MLP based on computational constraints
2. **Implement training loop**: Using physics-informed loss with validated tolerances
3. **Evaluation metrics**: Conservation errors, MSE, rollout stability
4. **Comparison baseline**: Use observed data violations as target performance

### Training Protocol
1. Load full collision dataset (98 × 32K = 3.2M samples)
2. Train with physics-informed loss (MSE + constraint penalties)
3. Validate on held-out timesteps
4. Generate comprehensive metrics report
5. Compare to observed data violation levels

## Risk Assessment

### Low Risk ✅
- Target definition mathematically sound
- Constraints based on real data analysis
- Implementation validated on sample data
- Clear path to D2.5 training

### Medium Risk ⚠️
- Constraint penalty weights need tuning
- Model capacity requirements unknown
- Full dataset memory/compute requirements

### Mitigation Strategies
- Start with ridge regression (minimal hyperparameters)
- Adjust penalty weights based on training convergence
- Use data subsampling if memory constrained

---

**D1 Status: COMPLETE** ✅
**Ready for D2.5: Baseline Train & Report** ✅

Target definition provides clear, testable objective for collision operator learning with realistic physics constraints derived from actual LBM data behavior.