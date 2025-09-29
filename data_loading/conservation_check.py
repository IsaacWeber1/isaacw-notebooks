#!/usr/bin/env python3
"""
Conservation law validation for 38-channel LBM data.

For the simplified 38-channel representation, we validate:
1. Total mass conservation: sum of all 38 channels
2. Total momentum conservation: using combined D3Q19 velocity set
3. Conservation between consecutive timesteps

Note: In the simplified approach, we treat f and g as unified,
so we check conservation across all 38 channels together.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from simple_loader import load_combined_sequence, create_collision_pairs, flatten_for_training
from learn_lbm.d3q19 import C_numpy as C


def compute_38_channel_mass(data: np.ndarray) -> np.ndarray:
    """
    Compute total mass for 38-channel data.

    Args:
        data: Shape (..., 38) where last dim has all channels

    Returns:
        Total mass per site: shape (...)
    """
    return np.sum(data, axis=-1)


def compute_38_channel_momentum(data: np.ndarray) -> np.ndarray:
    """
    Compute total momentum for 38-channel data.

    For simplified representation:
    - Channels 0-18: f population with D3Q19 velocity set
    - Channels 19-37: g population with same D3Q19 velocity set

    Args:
        data: Shape (..., 38)

    Returns:
        Total momentum per site: shape (..., 3)
    """

    f_data = data[..., :19]   # f channels
    g_data = data[..., 19:]   # g channels

    # Momentum from f population: sum_i c_i * f_i
    f_momentum = np.einsum('...i,id->...d', f_data, C.astype(data.dtype))

    # Momentum from g population: sum_i c_i * g_i
    g_momentum = np.einsum('...i,id->...d', g_data, C.astype(data.dtype))

    # Total momentum
    total_momentum = f_momentum + g_momentum

    return total_momentum


def check_conservation_single_timestep(data: np.ndarray) -> Dict[str, Any]:
    """
    Check conservation laws for a single timestep.

    Args:
        data: Shape (Nx, Ny, Nz, 38)

    Returns:
        Conservation statistics
    """

    # Compute conserved quantities
    mass = compute_38_channel_mass(data)
    momentum = compute_38_channel_momentum(data)

    # Flatten for statistics
    flat_data = flatten_for_training(data)
    flat_mass = mass.flatten()
    flat_momentum = momentum.reshape(-1, 3)

    results = {
        "mass": {
            "total": float(np.sum(flat_mass)),
            "mean_per_site": float(np.mean(flat_mass)),
            "std_per_site": float(np.std(flat_mass)),
            "min_per_site": float(np.min(flat_mass)),
            "max_per_site": float(np.max(flat_mass)),
            "sites_with_zero_mass": int(np.sum(flat_mass == 0)),
            "sites_with_negative_mass": int(np.sum(flat_mass < 0))
        },
        "momentum": {
            "total": [float(x) for x in np.sum(flat_momentum, axis=0)],
            "magnitude_per_site": {
                "mean": float(np.mean(np.linalg.norm(flat_momentum, axis=1))),
                "std": float(np.std(np.linalg.norm(flat_momentum, axis=1))),
                "max": float(np.max(np.linalg.norm(flat_momentum, axis=1)))
            },
            "components": {
                "x": {"mean": float(np.mean(flat_momentum[:, 0])), "std": float(np.std(flat_momentum[:, 0]))},
                "y": {"mean": float(np.mean(flat_momentum[:, 1])), "std": float(np.std(flat_momentum[:, 1]))},
                "z": {"mean": float(np.mean(flat_momentum[:, 2])), "std": float(np.std(flat_momentum[:, 2]))}
            }
        },
        "data_quality": {
            "total_sites": data.shape[0] * data.shape[1] * data.shape[2],
            "negative_values": int(np.sum(flat_data < 0)),
            "zero_values": int(np.sum(flat_data == 0)),
            "nan_values": int(np.sum(np.isnan(flat_data))),
            "inf_values": int(np.sum(np.isinf(flat_data)))
        }
    }

    return results


def check_conservation_between_timesteps(pre: np.ndarray, post: np.ndarray) -> Dict[str, Any]:
    """
    Check conservation between consecutive timesteps.

    Args:
        pre: Pre-collision state, shape (Nx, Ny, Nz, 38)
        post: Post-collision state, shape (Nx, Ny, Nz, 38)

    Returns:
        Conservation violation statistics
    """

    # Compute conserved quantities for both states
    mass_pre = compute_38_channel_mass(pre)
    mass_post = compute_38_channel_mass(post)

    momentum_pre = compute_38_channel_momentum(pre)
    momentum_post = compute_38_channel_momentum(post)

    # Conservation errors
    mass_error = mass_post - mass_pre
    momentum_error = momentum_post - momentum_pre

    # Flatten for statistics
    flat_mass_error = mass_error.flatten()
    flat_momentum_error = momentum_error.reshape(-1, 3)

    results = {
        "mass_conservation": {
            "total_error": float(np.sum(flat_mass_error)),
            "max_absolute_error": float(np.max(np.abs(flat_mass_error))),
            "mean_absolute_error": float(np.mean(np.abs(flat_mass_error))),
            "rmse": float(np.sqrt(np.mean(flat_mass_error**2))),
            "relative_error": float(np.sum(flat_mass_error) / np.sum(mass_pre.flatten())),
            "sites_with_violations": int(np.sum(np.abs(flat_mass_error) > 1e-10))
        },
        "momentum_conservation": {
            "total_error": [float(x) for x in np.sum(flat_momentum_error, axis=0)],
            "max_absolute_error": [float(x) for x in np.max(np.abs(flat_momentum_error), axis=0)],
            "mean_absolute_error": [float(x) for x in np.mean(np.abs(flat_momentum_error), axis=0)],
            "rmse": [float(x) for x in np.sqrt(np.mean(flat_momentum_error**2, axis=0))],
            "magnitude_rmse": float(np.sqrt(np.mean(np.sum(flat_momentum_error**2, axis=1)))),
            "sites_with_violations": int(np.sum(np.linalg.norm(flat_momentum_error, axis=1) > 1e-10))
        },
        "collision_increment": {
            "mean_change_per_channel": [float(x) for x in np.mean(post.reshape(-1, 38) - pre.reshape(-1, 38), axis=0)],
            "max_change_per_channel": [float(x) for x in np.max(np.abs(post.reshape(-1, 38) - pre.reshape(-1, 38)), axis=0)],
            "total_sites": pre.shape[0] * pre.shape[1] * pre.shape[2]
        }
    }

    return results


def check_sequence_conservation(data_root: str, num_timesteps: int = 10) -> Dict[str, Any]:
    """
    Check conservation across a sequence of timesteps.

    Args:
        data_root: Root directory
        num_timesteps: Number of timesteps to analyze

    Returns:
        Sequence conservation analysis
    """

    print(f"Checking conservation across {num_timesteps} timesteps...")

    # Load sequence
    sequence = load_combined_sequence(data_root, start_idx=0, num_timesteps=num_timesteps)

    results = {
        "metadata": {
            "num_timesteps": len(sequence),
            "grid_shape": sequence[0].shape[:3],
            "total_sites": sequence[0].shape[0] * sequence[0].shape[1] * sequence[0].shape[2]
        },
        "per_timestep": [],
        "temporal_conservation": {},
        "pairwise_conservation": []
    }

    # Analyze each timestep individually
    for t, timestep_data in enumerate(sequence):
        timestep_conservation = check_conservation_single_timestep(timestep_data)
        timestep_conservation["timestep"] = t
        results["per_timestep"].append(timestep_conservation)

    # Analyze pairwise conservation
    for t in range(len(sequence) - 1):
        pair_conservation = check_conservation_between_timesteps(sequence[t], sequence[t+1])
        pair_conservation["timestep_pair"] = [t, t+1]
        results["pairwise_conservation"].append(pair_conservation)

    # Analyze temporal trends
    total_masses = [ts["mass"]["total"] for ts in results["per_timestep"]]
    total_momenta = [ts["momentum"]["total"] for ts in results["per_timestep"]]

    results["temporal_conservation"] = {
        "mass_drift": {
            "initial": total_masses[0],
            "final": total_masses[-1],
            "absolute_change": total_masses[-1] - total_masses[0],
            "relative_change": (total_masses[-1] - total_masses[0]) / total_masses[0],
            "max_fluctuation": float(np.max(total_masses) - np.min(total_masses)),
            "std_fluctuation": float(np.std(total_masses))
        },
        "momentum_drift": {
            "initial": total_momenta[0],
            "final": total_momenta[-1],
            "absolute_change": [total_momenta[-1][i] - total_momenta[0][i] for i in range(3)],
            "magnitude_change": float(np.linalg.norm(
                np.array(total_momenta[-1]) - np.array(total_momenta[0])
            ))
        }
    }

    return results


def main():
    """Main conservation validation routine."""

    data_root = "/Users/owner/Projects/LBM"
    output_dir = Path(__file__).parent.parent / "artifacts"
    output_dir.mkdir(exist_ok=True)

    print("=== Conservation Law Validation ===")

    try:
        # Full sequence analysis
        print("\n1. Analyzing conservation across timestep sequence...")
        conservation_results = check_sequence_conservation(data_root, num_timesteps=10)

        # Collision pair analysis
        print("\n2. Analyzing collision pair conservation...")
        pre, post = create_collision_pairs(
            load_combined_sequence(data_root, start_idx=0, num_timesteps=5)
        )

        collision_conservation = []
        for i in range(pre.shape[0]):
            pair_result = check_conservation_between_timesteps(pre[i], post[i])
            pair_result["pair_index"] = i
            collision_conservation.append(pair_result)

        # Compile final results
        full_results = {
            "analysis_type": "conservation_validation_38_channel",
            "data_source": data_root,
            "sequence_analysis": conservation_results,
            "collision_pair_analysis": collision_conservation,
            "summary": {
                "max_mass_violation": max(
                    pair["mass_conservation"]["max_absolute_error"]
                    for pair in collision_conservation
                ),
                "max_momentum_violation": max(
                    pair["momentum_conservation"]["magnitude_rmse"]
                    for pair in collision_conservation
                ),
                "conservation_quality": "GOOD" if all(
                    pair["mass_conservation"]["max_absolute_error"] < 1e-8 and
                    pair["momentum_conservation"]["magnitude_rmse"] < 1e-8
                    for pair in collision_conservation
                ) else "VIOLATIONS_DETECTED"
            }
        }

        # Save results
        output_file = output_dir / "conservation_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("CONSERVATION ANALYSIS COMPLETE")
        print("="*60)
        print(f"Timesteps analyzed: {conservation_results['metadata']['num_timesteps']}")
        print(f"Collision pairs: {len(collision_conservation)}")
        print(f"Max mass violation: {full_results['summary']['max_mass_violation']:.2e}")
        print(f"Max momentum violation: {full_results['summary']['max_momentum_violation']:.2e}")
        print(f"Conservation quality: {full_results['summary']['conservation_quality']}")
        print(f"Results saved: {output_file}")

    except Exception as e:
        print(f"Conservation check failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())