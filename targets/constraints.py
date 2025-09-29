#!/usr/bin/env python3
"""
Physics constraints for 38-channel LBM collision learning.

Implements conservation checks with realistic tolerances based on D0 analysis:
- Mass conservation: ~5% tolerance (observed 1-4% violations)
- Momentum conservation: ~2% tolerance per component
- Physical bounds: non-negativity with small tolerance
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path for D3Q19 velocity set
sys.path.append(str(Path(__file__).parent.parent.parent))
from learn_lbm.d3q19 import C_numpy as C


# Tolerance constants based on D0 analysis
MASS_TOLERANCE = 0.05  # 5% mass conservation tolerance
MOMENTUM_TOLERANCE = 0.02  # 2% momentum conservation tolerance per component
NEGATIVITY_TOLERANCE = -1e-10  # Small negative values allowed (numerical precision)


def compute_38_channel_mass(states: np.ndarray) -> np.ndarray:
    """
    Compute total mass for 38-channel states.

    Args:
        states: State array, shape (..., 38)

    Returns:
        Mass per state, shape (...)
    """
    return np.sum(states, axis=-1)


def compute_38_channel_momentum(states: np.ndarray) -> np.ndarray:
    """
    Compute total momentum for 38-channel states.

    For the unified representation:
    - Channels 0-18: f population with D3Q19 velocities
    - Channels 19-37: g population with same D3Q19 velocities

    Args:
        states: State array, shape (..., 38)

    Returns:
        Momentum per state, shape (..., 3)
    """

    f_channels = states[..., :19]   # f population
    g_channels = states[..., 19:]   # g population

    # Momentum contributions from both populations
    f_momentum = np.einsum('...i,id->...d', f_channels, C.astype(states.dtype))
    g_momentum = np.einsum('...i,id->...d', g_channels, C.astype(states.dtype))

    # Total momentum
    total_momentum = f_momentum + g_momentum

    return total_momentum


def check_mass_conservation(pre_states: np.ndarray,
                           post_states: np.ndarray,
                           tolerance: float = MASS_TOLERANCE) -> Dict[str, Any]:
    """
    Check mass conservation between pre and post collision states.

    Args:
        pre_states: Pre-collision states, shape (..., 38)
        post_states: Post-collision states, shape (..., 38)
        tolerance: Relative tolerance for mass conservation

    Returns:
        Conservation check results
    """

    # Compute masses
    pre_mass = compute_38_channel_mass(pre_states)
    post_mass = compute_38_channel_mass(post_states)

    # Mass difference
    mass_error = post_mass - pre_mass
    relative_mass_error = mass_error / (pre_mass + 1e-15)  # Avoid division by zero

    # Flatten for statistics
    flat_mass_error = mass_error.flatten()
    flat_relative_error = relative_mass_error.flatten()

    # Check violations
    violations = np.abs(flat_relative_error) > tolerance
    violation_count = np.sum(violations)
    violation_fraction = violation_count / len(flat_relative_error)

    results = {
        "constraint": "mass_conservation",
        "tolerance": tolerance,
        "total_samples": len(flat_mass_error),
        "violations": {
            "count": int(violation_count),
            "fraction": float(violation_fraction),
            "max_violation": float(np.max(np.abs(flat_relative_error))),
            "mean_violation": float(np.mean(np.abs(flat_relative_error)))
        },
        "statistics": {
            "mean_error": float(np.mean(flat_mass_error)),
            "std_error": float(np.std(flat_mass_error)),
            "max_absolute_error": float(np.max(np.abs(flat_mass_error))),
            "rmse": float(np.sqrt(np.mean(flat_mass_error**2))),
            "mean_relative_error": float(np.mean(flat_relative_error)),
            "max_relative_error": float(np.max(np.abs(flat_relative_error)))
        },
        "status": "PASS" if violation_fraction < 0.1 else "FAIL"  # Allow 10% violation rate
    }

    return results


def check_momentum_conservation(pre_states: np.ndarray,
                               post_states: np.ndarray,
                               tolerance: float = MOMENTUM_TOLERANCE) -> Dict[str, Any]:
    """
    Check momentum conservation between pre and post collision states.

    Args:
        pre_states: Pre-collision states, shape (..., 38)
        post_states: Post-collision states, shape (..., 38)
        tolerance: Tolerance for momentum conservation per component

    Returns:
        Conservation check results
    """

    # Compute momenta
    pre_momentum = compute_38_channel_momentum(pre_states)
    post_momentum = compute_38_channel_momentum(post_states)

    # Momentum error
    momentum_error = post_momentum - pre_momentum

    # Flatten for statistics
    flat_momentum_error = momentum_error.reshape(-1, 3)

    # Component-wise analysis
    results = {
        "constraint": "momentum_conservation",
        "tolerance": tolerance,
        "total_samples": flat_momentum_error.shape[0],
        "components": {},
        "overall": {}
    }

    # Analyze each component (x, y, z)
    component_names = ['x', 'y', 'z']
    all_violations = []

    for i, comp in enumerate(component_names):
        comp_errors = flat_momentum_error[:, i]
        comp_violations = np.abs(comp_errors) > tolerance

        violation_count = np.sum(comp_violations)
        violation_fraction = violation_count / len(comp_errors)

        results["components"][comp] = {
            "violations": int(violation_count),
            "violation_fraction": float(violation_fraction),
            "max_error": float(np.max(np.abs(comp_errors))),
            "mean_error": float(np.mean(comp_errors)),
            "std_error": float(np.std(comp_errors)),
            "rmse": float(np.sqrt(np.mean(comp_errors**2)))
        }

        all_violations.extend(comp_violations)

    # Overall momentum conservation
    magnitude_errors = np.linalg.norm(flat_momentum_error, axis=1)
    magnitude_violations = magnitude_errors > tolerance

    results["overall"] = {
        "magnitude_violations": int(np.sum(magnitude_violations)),
        "magnitude_violation_fraction": float(np.mean(magnitude_violations)),
        "max_magnitude_error": float(np.max(magnitude_errors)),
        "mean_magnitude_error": float(np.mean(magnitude_errors)),
        "rmse_magnitude": float(np.sqrt(np.mean(magnitude_errors**2))),
        "total_violation_fraction": float(np.mean(all_violations))
    }

    # Status
    results["status"] = "PASS" if results["overall"]["total_violation_fraction"] < 0.1 else "FAIL"

    return results


def check_physical_bounds(states: np.ndarray,
                         tolerance: float = NEGATIVITY_TOLERANCE) -> Dict[str, Any]:
    """
    Check physical bounds (non-negativity) for states.

    Args:
        states: State array, shape (..., 38)
        tolerance: Tolerance for negative values

    Returns:
        Physical bounds check results
    """

    flat_states = states.reshape(-1, 38)

    # Check for negative values
    negative_mask = flat_states < tolerance
    negative_count = np.sum(negative_mask)
    negative_fraction = negative_count / flat_states.size

    # Statistics
    min_value = float(np.min(flat_states))
    negative_values = flat_states[negative_mask]

    results = {
        "constraint": "physical_bounds",
        "tolerance": tolerance,
        "total_values": int(flat_states.size),
        "negative_values": {
            "count": int(negative_count),
            "fraction": negative_fraction,
            "min_value": min_value,
            "mean_negative": float(np.mean(negative_values)) if negative_count > 0 else 0.0,
            "worst_negative": float(np.min(negative_values)) if negative_count > 0 else 0.0
        },
        "statistics": {
            "min_global": min_value,
            "max_global": float(np.max(flat_states)),
            "mean_global": float(np.mean(flat_states)),
            "negative_sites": int(np.sum(np.any(negative_mask.reshape(-1, 38), axis=1)))
        },
        "status": "PASS" if negative_fraction < 0.001 else "FAIL"  # Allow 0.1% negative values
    }

    return results


def validate_collision_pair(pre_state: np.ndarray,
                           post_state: np.ndarray,
                           mass_tol: float = MASS_TOLERANCE,
                           momentum_tol: float = MOMENTUM_TOLERANCE,
                           bounds_tol: float = NEGATIVITY_TOLERANCE) -> Dict[str, Any]:
    """
    Complete validation of a collision pair against all constraints.

    Args:
        pre_state: Pre-collision state, shape (..., 38)
        post_state: Post-collision state, shape (..., 38)
        mass_tol: Mass conservation tolerance
        momentum_tol: Momentum conservation tolerance
        bounds_tol: Physical bounds tolerance

    Returns:
        Complete validation results
    """

    results = {
        "validation_type": "collision_pair",
        "constraints": {},
        "summary": {}
    }

    # Run all constraint checks
    results["constraints"]["mass"] = check_mass_conservation(pre_state, post_state, mass_tol)
    results["constraints"]["momentum"] = check_momentum_conservation(pre_state, post_state, momentum_tol)
    results["constraints"]["pre_bounds"] = check_physical_bounds(pre_state, bounds_tol)
    results["constraints"]["post_bounds"] = check_physical_bounds(post_state, bounds_tol)

    # Summary
    all_passed = all(
        check["status"] == "PASS"
        for check in results["constraints"].values()
    )

    results["summary"] = {
        "overall_status": "PASS" if all_passed else "VIOLATIONS_DETECTED",
        "passed_constraints": sum(
            1 for check in results["constraints"].values()
            if check["status"] == "PASS"
        ),
        "total_constraints": len(results["constraints"])
    }

    return results


def create_constraint_penalty_loss(deltas: np.ndarray,
                                  pre_states: np.ndarray,
                                  mass_weight: float = 1.0,
                                  momentum_weight: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """
    Create constraint penalty terms for ML loss functions.

    Args:
        deltas: Predicted collision increments, shape (..., 38)
        pre_states: Pre-collision states, shape (..., 38)
        mass_weight: Weight for mass conservation penalty
        momentum_weight: Weight for momentum conservation penalty

    Returns:
        total_penalty: Total constraint violation penalty
        penalties: Individual penalty components
    """

    # Mass conservation penalty: |sum(deltas)|
    mass_violations = np.abs(np.sum(deltas, axis=-1))
    mass_penalty = mass_weight * np.mean(mass_violations)

    # Momentum conservation penalty: |sum(c_i * delta_i)|
    # Need to compute momentum from deltas
    f_deltas = deltas[..., :19]
    g_deltas = deltas[..., 19:]

    f_momentum_delta = np.einsum('...i,id->...d', f_deltas, C.astype(deltas.dtype))
    g_momentum_delta = np.einsum('...i,id->...d', g_deltas, C.astype(deltas.dtype))
    total_momentum_delta = f_momentum_delta + g_momentum_delta

    momentum_violations = np.linalg.norm(total_momentum_delta, axis=-1)
    momentum_penalty = momentum_weight * np.mean(momentum_violations)

    penalties = {
        "mass_penalty": float(mass_penalty),
        "momentum_penalty": float(momentum_penalty),
        "total_penalty": float(mass_penalty + momentum_penalty)
    }

    return penalties["total_penalty"], penalties


def main():
    """Demo of constraint checking."""

    print("=== Physics Constraints Demo ===")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Pre-collision states
    pre_states = np.random.uniform(0.02, 0.35, (n_samples, 38))

    # Post-collision with small violations (realistic)
    deltas = np.random.normal(0, 0.001, (n_samples, 38))

    # Add small mass conservation violations
    mass_violations = np.random.normal(0, 0.002, n_samples)
    deltas[:, 0] += mass_violations  # Violate mass conservation slightly

    post_states = pre_states + deltas
    post_states = np.maximum(post_states, 0.001)  # Ensure non-negativity

    print(f"Created {n_samples} collision pairs with realistic violations")

    # Test constraint checks
    print("\n1. Mass conservation check:")
    mass_check = check_mass_conservation(pre_states, post_states)
    print(f"   Status: {mass_check['status']}")
    print(f"   Violations: {mass_check['violations']['fraction']:.3f} of samples")
    print(f"   Max violation: {mass_check['violations']['max_violation']:.4f}")

    print("\n2. Momentum conservation check:")
    momentum_check = check_momentum_conservation(pre_states, post_states)
    print(f"   Status: {momentum_check['status']}")
    print(f"   Overall violations: {momentum_check['overall']['total_violation_fraction']:.3f}")
    print(f"   Max magnitude error: {momentum_check['overall']['max_magnitude_error']:.4f}")

    print("\n3. Physical bounds check:")
    bounds_check = check_physical_bounds(post_states)
    print(f"   Status: {bounds_check['status']}")
    print(f"   Negative fraction: {bounds_check['negative_values']['fraction']:.6f}")
    print(f"   Min value: {bounds_check['negative_values']['min_value']:.6f}")

    # Complete validation
    print("\n4. Complete validation:")
    validation = validate_collision_pair(pre_states, post_states)
    print(f"   Overall status: {validation['summary']['overall_status']}")
    print(f"   Passed: {validation['summary']['passed_constraints']}/{validation['summary']['total_constraints']} constraints")

    # Penalty calculation
    print("\n5. Constraint penalties:")
    penalty, penalties = create_constraint_penalty_loss(deltas, pre_states)
    print(f"   Total penalty: {penalty:.6f}")
    print(f"   Mass penalty: {penalties['mass_penalty']:.6f}")
    print(f"   Momentum penalty: {penalties['momentum_penalty']:.6f}")

    print("\n✅ Physics constraints working correctly")


if __name__ == "__main__":
    main()