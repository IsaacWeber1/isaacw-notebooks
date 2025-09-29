#!/usr/bin/env python3
"""
Collision target definitions for 38-channel LBM learning.

Defines the learning target as collision increments:
Δ = state[t+1] - state[t]

Where state is the 38-channel combined [f₀...f₁₈, g₀...g₁₈] vector.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CollisionTarget:
    """
    Defines the target for collision operator learning.

    For the simplified 38-channel approach:
    - Input: pre_collision state (38 channels)
    - Output: collision increment Δ (38 channels)
    - Update rule: post_collision = pre_collision + Δ
    """
    name: str = "collision_increments"
    input_channels: int = 38
    output_channels: int = 38
    description: str = "Learn collision increments Δ such that state[t+1] = state[t] + Δ"


def compute_collision_deltas(pre_collision: np.ndarray,
                            post_collision: np.ndarray) -> np.ndarray:
    """
    Compute collision increments from pre/post collision states.

    Args:
        pre_collision: Pre-collision state, shape (..., 38)
        post_collision: Post-collision state, shape (..., 38)

    Returns:
        deltas: Collision increments, shape (..., 38)

    The target learning problem is:
        Given: pre_collision state
        Predict: deltas
        Such that: post_collision ≈ pre_collision + deltas
    """

    # Validate inputs
    if pre_collision.shape != post_collision.shape:
        raise ValueError(f"Shape mismatch: pre={pre_collision.shape}, post={post_collision.shape}")

    if pre_collision.shape[-1] != 38:
        raise ValueError(f"Expected 38 channels, got {pre_collision.shape[-1]}")

    # Simple increment calculation
    deltas = post_collision - pre_collision

    # Validate no NaN/Inf
    if np.any(np.isnan(deltas)) or np.any(np.isinf(deltas)):
        raise ValueError("NaN or Inf values detected in collision deltas")

    return deltas


def analyze_delta_statistics(deltas: np.ndarray) -> dict:
    """
    Analyze statistical properties of collision increments.

    Args:
        deltas: Collision increments, shape (..., 38)

    Returns:
        Statistics dictionary with per-channel and global analysis
    """

    # Flatten spatial dimensions if present
    if deltas.ndim > 2:
        flat_deltas = deltas.reshape(-1, 38)
    else:
        flat_deltas = deltas

    stats = {
        "shape": deltas.shape,
        "total_samples": flat_deltas.shape[0],
        "channels": flat_deltas.shape[1],
        "per_channel": {},
        "global": {}
    }

    # Per-channel statistics
    for ch in range(38):
        ch_data = flat_deltas[:, ch]

        # Determine population (f: 0-18, g: 19-37)
        pop_type = "f" if ch < 19 else "g"
        ch_in_pop = ch if ch < 19 else ch - 19
        channel_name = f"{pop_type}{ch_in_pop}"

        stats["per_channel"][channel_name] = {
            "index": ch,
            "population": pop_type,
            "mean": float(np.mean(ch_data)),
            "std": float(np.std(ch_data)),
            "min": float(np.min(ch_data)),
            "max": float(np.max(ch_data)),
            "median": float(np.median(ch_data)),
            "q25": float(np.percentile(ch_data, 25)),
            "q75": float(np.percentile(ch_data, 75)),
            "near_zero": int(np.sum(np.abs(ch_data) < 1e-10)),
            "magnitude_mean": float(np.mean(np.abs(ch_data))),
            "magnitude_max": float(np.max(np.abs(ch_data)))
        }

    # Global statistics
    all_deltas = flat_deltas.flatten()
    stats["global"] = {
        "mean_increment": float(np.mean(all_deltas)),
        "std_increment": float(np.std(all_deltas)),
        "min_increment": float(np.min(all_deltas)),
        "max_increment": float(np.max(all_deltas)),
        "magnitude_mean": float(np.mean(np.abs(all_deltas))),
        "magnitude_max": float(np.max(np.abs(all_deltas))),
        "magnitude_std": float(np.std(np.abs(all_deltas))),
        "zero_fraction": float(np.mean(np.abs(all_deltas) < 1e-10)),
        "large_changes": int(np.sum(np.abs(all_deltas) > 0.01))  # Changes > 1%
    }

    return stats


def create_training_dataset(pre_states: np.ndarray,
                           post_states: np.ndarray,
                           max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset from collision state pairs.

    Args:
        pre_states: Pre-collision states, shape (N, ..., 38)
        post_states: Post-collision states, shape (N, ..., 38)
        max_samples: Maximum number of samples to use (for memory management)

    Returns:
        X: Input features (pre-collision states), shape (M, 38)
        y: Target outputs (collision deltas), shape (M, 38)

    Where M = min(max_samples, total_available_samples)
    """

    # Compute deltas
    deltas = compute_collision_deltas(pre_states, post_states)

    # Flatten spatial dimensions
    if pre_states.ndim > 2:
        X = pre_states.reshape(-1, 38)
        y = deltas.reshape(-1, 38)
    else:
        X = pre_states.copy()
        y = deltas.copy()

    # Subsample if requested
    if max_samples is not None and X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y


def validate_target_definition(pre_states: np.ndarray,
                             post_states: np.ndarray) -> dict:
    """
    Validate that the target definition makes sense for the data.

    Args:
        pre_states: Pre-collision states, shape (..., 38)
        post_states: Post-collision states, shape (..., 38)

    Returns:
        Validation results dictionary
    """

    results = {
        "target_definition": "collision_increments",
        "validation_status": "pending",
        "issues": [],
        "statistics": {}
    }

    try:
        # Compute deltas
        deltas = compute_collision_deltas(pre_states, post_states)

        # Analyze statistics
        stats = analyze_delta_statistics(deltas)
        results["statistics"] = stats

        # Validation checks
        issues = []

        # Check if deltas are reasonable magnitude
        max_magnitude = stats["global"]["magnitude_max"]
        if max_magnitude > 0.5:  # More than 50% change
            issues.append(f"Very large deltas detected: max magnitude = {max_magnitude:.3f}")

        # Check if deltas are not all near zero
        mean_magnitude = stats["global"]["magnitude_mean"]
        if mean_magnitude < 1e-8:
            issues.append(f"Deltas are nearly zero: mean magnitude = {mean_magnitude:.2e}")

        # Check reconstruction accuracy
        reconstructed = pre_states + deltas
        reconstruction_error = np.mean(np.abs(reconstructed - post_states))

        if reconstruction_error > 1e-10:
            issues.append(f"Reconstruction error: {reconstruction_error:.2e}")
        else:
            results["reconstruction_perfect"] = True

        results["issues"] = issues
        results["validation_status"] = "PASS" if len(issues) == 0 else "WARNINGS"

    except Exception as e:
        results["validation_status"] = "FAIL"
        results["error"] = str(e)

    return results


def main():
    """Demo of collision target definitions."""

    print("=== Collision Target Definition Demo ===")

    # Create synthetic example
    np.random.seed(42)

    # Simulate 1000 collision pairs with realistic LBM values
    n_samples = 1000
    pre_states = np.random.uniform(0.02, 0.35, (n_samples, 38))

    # Small increments (realistic for collision)
    small_deltas = np.random.normal(0, 0.001, (n_samples, 38))
    post_states = pre_states + small_deltas

    # Ensure non-negativity
    post_states = np.maximum(post_states, 0.001)

    print(f"Created synthetic data: {n_samples} collision pairs")
    print(f"Pre-states range: [{np.min(pre_states):.4f}, {np.max(pre_states):.4f}]")
    print(f"Post-states range: [{np.min(post_states):.4f}, {np.max(post_states):.4f}]")

    # Validate target definition
    print("\nValidating target definition...")
    validation = validate_target_definition(pre_states, post_states)

    print(f"Validation status: {validation['validation_status']}")
    if validation.get('issues'):
        for issue in validation['issues']:
            print(f"⚠️  {issue}")

    if 'statistics' in validation:
        stats = validation['statistics']
        print(f"\nDelta statistics:")
        print(f"  Mean magnitude: {stats['global']['magnitude_mean']:.6f}")
        print(f"  Max magnitude: {stats['global']['magnitude_max']:.6f}")
        print(f"  Zero fraction: {stats['global']['zero_fraction']:.4f}")

    # Create training dataset
    print(f"\nCreating training dataset...")
    X, y = create_training_dataset(pre_states, post_states, max_samples=500)

    print(f"Training data: X={X.shape}, y={y.shape}")
    print(f"Target summary: mean={np.mean(y):.6f}, std={np.std(y):.6f}")

    print("\n✅ Collision target definition working correctly")


if __name__ == "__main__":
    main()