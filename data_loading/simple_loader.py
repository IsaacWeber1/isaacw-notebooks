#!/usr/bin/env python3
"""
Simplified data loader for 38-channel LBM representation.

Combines f and g populations into single state vectors:
- Input: population_f/ and population_g/ directories
- Output: (num_sites, 38) arrays where channels 0-18 are f, 19-37 are g
- Creates collision pairs from consecutive timesteps
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to import from learn_lbm
sys.path.append(str(Path(__file__).parent.parent.parent))

from learn_lbm.data_loading import list_plotfiles, load_plotfile_array
from learn_lbm.d3q19 import Q


def load_combined_timestep(data_root: str, timestep_idx: int, pattern: str = r"plt1001\d{3}") -> np.ndarray:
    """
    Load and combine f and g populations for a single timestep.

    Args:
        data_root: Root directory containing population_f/ and population_g/
        timestep_idx: Index in the sorted timestep list
        pattern: Regex pattern for timestep directories

    Returns:
        Combined array of shape (Nx, Ny, Nz, 38) where:
        - Channels 0-18: f population
        - Channels 19-37: g population
    """

    # Get file lists
    f_files = list_plotfiles(data_root, "population_f", pattern)
    g_files = list_plotfiles(data_root, "population_g", pattern)

    if timestep_idx >= len(f_files) or timestep_idx >= len(g_files):
        raise IndexError(f"Timestep index {timestep_idx} exceeds available data")

    # Load both populations
    f_data, _ = load_plotfile_array(f_files[timestep_idx], "f")
    g_data, _ = load_plotfile_array(g_files[timestep_idx], "g")

    # Verify shapes match
    if f_data.shape != g_data.shape:
        raise ValueError(f"Shape mismatch: f={f_data.shape}, g={g_data.shape}")

    # Combine along channel dimension
    combined = np.concatenate([f_data, g_data], axis=-1)

    # Should have shape (Nx, Ny, Nz, 38)
    assert combined.shape[-1] == 38, f"Expected 38 channels, got {combined.shape[-1]}"

    return combined


def load_combined_sequence(data_root: str,
                          start_idx: int = 0,
                          num_timesteps: int = 10,
                          pattern: str = r"plt1001\d{3}") -> List[np.ndarray]:
    """
    Load sequence of combined timesteps.

    Args:
        data_root: Root directory
        start_idx: Starting timestep index
        num_timesteps: Number of consecutive timesteps to load
        pattern: Regex pattern for timestep directories

    Returns:
        List of arrays, each shape (Nx, Ny, Nz, 38)
    """

    sequence = []
    for i in range(num_timesteps):
        timestep_data = load_combined_timestep(data_root, start_idx + i, pattern)
        sequence.append(timestep_data)

    return sequence


def create_collision_pairs(sequence: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (pre-collision, post-collision) pairs from timestep sequence.

    Args:
        sequence: List of timestep arrays, each shape (Nx, Ny, Nz, 38)

    Returns:
        pre_collision: Array of shape (T-1, Nx, Ny, Nz, 38)
        post_collision: Array of shape (T-1, Nx, Ny, Nz, 38)

    The collision increment can be computed as: delta = post - pre
    """

    if len(sequence) < 2:
        raise ValueError("Need at least 2 timesteps to create pairs")

    # Stack consecutive pairs
    pre_collision = np.stack(sequence[:-1], axis=0)   # timesteps 0 to T-2
    post_collision = np.stack(sequence[1:], axis=0)   # timesteps 1 to T-1

    return pre_collision, post_collision


def flatten_for_training(data: np.ndarray) -> np.ndarray:
    """
    Flatten spatial dimensions for machine learning.

    Args:
        data: Array of shape (T, Nx, Ny, Nz, 38) or (Nx, Ny, Nz, 38)

    Returns:
        Flattened array of shape (T*Nx*Ny*Nz, 38) or (Nx*Ny*Nz, 38)
    """

    if data.ndim == 5:  # (T, Nx, Ny, Nz, 38)
        T, Nx, Ny, Nz, channels = data.shape
        return data.reshape(T * Nx * Ny * Nz, channels)
    elif data.ndim == 4:  # (Nx, Ny, Nz, 38)
        Nx, Ny, Nz, channels = data.shape
        return data.reshape(Nx * Ny * Nz, channels)
    else:
        raise ValueError(f"Expected 4D or 5D input, got shape {data.shape}")


def load_training_data(data_root: str,
                      start_idx: int = 0,
                      num_timesteps: int = 10,
                      flatten: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline: load sequence and create flattened training pairs.

    Args:
        data_root: Root directory
        start_idx: Starting timestep index
        num_timesteps: Number of timesteps to load
        flatten: Whether to flatten spatial dimensions

    Returns:
        X: Pre-collision states, shape (N, 38) if flatten=True
        y: Post-collision states, shape (N, 38) if flatten=True

    Where N = (num_timesteps-1) * Nx * Ny * Nz for flattened data
    """

    print(f"Loading {num_timesteps} timesteps starting from index {start_idx}...")

    # Load sequence
    sequence = load_combined_sequence(data_root, start_idx, num_timesteps)

    print(f"Loaded sequence shape: {len(sequence)} timesteps of {sequence[0].shape}")

    # Create collision pairs
    pre, post = create_collision_pairs(sequence)

    print(f"Created collision pairs: pre={pre.shape}, post={post.shape}")

    if flatten:
        pre_flat = flatten_for_training(pre)
        post_flat = flatten_for_training(post)

        print(f"Flattened for training: X={pre_flat.shape}, y={post_flat.shape}")

        return pre_flat, post_flat
    else:
        return pre, post


def main():
    """Demo of the simplified loader."""

    data_root = "/Users/owner/Projects/LBM"

    print("=== Simplified 38-Channel LBM Data Loader Demo ===")

    try:
        # Load small sample
        X, y = load_training_data(data_root,
                                 start_idx=0,
                                 num_timesteps=5,
                                 flatten=True)

        print(f"\nSample training data:")
        print(f"X (pre-collision): {X.shape}, dtype={X.dtype}")
        print(f"y (post-collision): {y.shape}, dtype={y.dtype}")

        # Basic statistics
        print(f"\nData ranges:")
        print(f"X: [{X.min():.6f}, {X.max():.6f}], mean={X.mean():.6f}")
        print(f"y: [{y.min():.6f}, {y.max():.6f}], mean={y.mean():.6f}")

        # Check for any obvious issues
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print("WARNING: Found NaN values in data!")

        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            print("WARNING: Found infinite values in data!")

        print("\n✓ Loader working correctly")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())