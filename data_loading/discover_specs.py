#!/usr/bin/env python3
"""
Data discovery script for LBM samples - discovers grid dimensions,
channel layout, and basic properties of the 38-channel representation.

Outputs:
- Grid dimensions (Nx, Ny, Nz)
- Channel count validation (19 f + 19 g = 38)
- Available timesteps range
- Basic data shape information
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import from learn_lbm
sys.path.append(str(Path(__file__).parent.parent.parent))

from learn_lbm.data_loading import list_plotfiles, load_plotfile_array
from learn_lbm.d3q19 import Q


def parse_header_dimensions(header_path: str) -> Tuple[int, int, int]:
    """Parse grid dimensions from AMReX header file."""
    with open(header_path, 'r') as f:
        lines = f.readlines()

    # Look for line with format: ((x_min, y_min, z_min) (x_max, y_max, z_max) ...)
    for line in lines:
        if '((' in line and '))' in line:
            # Extract the bounds
            match = re.search(r'\(\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\)', line)
            if match:
                x_min, y_min, z_min = int(match.group(1)), int(match.group(2)), int(match.group(3))
                x_max, y_max, z_max = int(match.group(4)), int(match.group(5)), int(match.group(6))

                # Grid size is max - min + 1
                nx = x_max - x_min + 1
                ny = y_max - y_min + 1
                nz = z_max - z_min + 1

                return nx, ny, nz

    raise ValueError(f"Could not parse grid dimensions from {header_path}")


def discover_data_specs(data_root: str) -> Dict:
    """Discover complete data specifications for both populations."""

    print("Discovering LBM data specifications...")
    print("=" * 50)

    # Check both populations exist
    pop_f_dir = os.path.join(data_root, "population_f")
    pop_g_dir = os.path.join(data_root, "population_g")

    if not os.path.exists(pop_f_dir):
        raise FileNotFoundError(f"Population f directory not found: {pop_f_dir}")
    if not os.path.exists(pop_g_dir):
        raise FileNotFoundError(f"Population g directory not found: {pop_g_dir}")

    # Discover available timesteps
    pattern = r"plt1001\d{3}"
    f_files = list_plotfiles(data_root, "population_f", pattern)
    g_files = list_plotfiles(data_root, "population_g", pattern)

    print(f"Found {len(f_files)} f-population timesteps")
    print(f"Found {len(g_files)} g-population timesteps")

    # Get grid dimensions from first timestep
    first_f_header = os.path.join(f_files[0], "Header")
    first_g_header = os.path.join(g_files[0], "Header")

    nx_f, ny_f, nz_f = parse_header_dimensions(first_f_header)
    nx_g, ny_g, nz_g = parse_header_dimensions(first_g_header)

    print(f"F-population grid: {nx_f} × {ny_f} × {nz_f}")
    print(f"G-population grid: {nx_g} × {ny_g} × {nz_g}")

    # Verify grids match
    if (nx_f, ny_f, nz_f) != (nx_g, ny_g, nz_g):
        raise ValueError("F and G population grids don't match!")

    nx, ny, nz = nx_f, ny_f, nz_f
    total_sites = nx * ny * nz

    print(f"Unified grid: {nx} × {ny} × {nz} = {total_sites:,} sites")

    # Load one timestep to validate channel structure
    print("\nLoading sample timestep to validate channels...")

    try:
        f_data, f_metadata = load_plotfile_array(f_files[0], "f")
        g_data, g_metadata = load_plotfile_array(g_files[0], "g")

        print(f"F-data shape: {f_data.shape}")
        print(f"G-data shape: {g_data.shape}")

        # Verify shapes match expected
        expected_shape = (nx, ny, nz, Q)
        if f_data.shape != expected_shape:
            raise ValueError(f"F-data shape {f_data.shape} != expected {expected_shape}")
        if g_data.shape != expected_shape:
            raise ValueError(f"G-data shape {g_data.shape} != expected {expected_shape}")

        print(f"✓ Both populations have correct shape: {expected_shape}")

        # Check for combined representation
        combined_channels = f_data.shape[-1] + g_data.shape[-1]
        print(f"Combined channels: {f_data.shape[-1]} + {g_data.shape[-1]} = {combined_channels}")

        if combined_channels != 38:
            raise ValueError(f"Expected 38 total channels, got {combined_channels}")

        print("✓ Total channels = 38 as expected")

    except Exception as e:
        print(f"Error loading sample data: {e}")
        raise

    # Extract timestep range
    f_steps = [int(re.search(r'(\d+)$', os.path.basename(f)).group(1)) for f in f_files]
    g_steps = [int(re.search(r'(\d+)$', os.path.basename(f)).group(1)) for f in g_files]

    min_step_f, max_step_f = min(f_steps), max(f_steps)
    min_step_g, max_step_g = min(g_steps), max(g_steps)

    print(f"\nTimestep ranges:")
    print(f"F-population: {min_step_f} to {max_step_f}")
    print(f"G-population: {min_step_g} to {max_step_g}")

    # Compile results
    specs = {
        "grid_dimensions": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "total_sites": total_sites
        },
        "channels": {
            "f_channels": Q,
            "g_channels": Q,
            "total_channels": 2 * Q,
            "expected_total": 38
        },
        "timesteps": {
            "f_population": {
                "count": len(f_files),
                "range": [min_step_f, max_step_f],
                "files": len(f_files)
            },
            "g_population": {
                "count": len(g_files),
                "range": [min_step_g, max_step_g],
                "files": len(g_files)
            }
        },
        "data_format": {
            "per_timestep_shape": [nx, ny, nz, 38],
            "flattened_shape": [total_sites, 38],
            "dtype": str(f_data.dtype),
            "file_format": "HyperCLaw-V1.1 AMReX/BoxLib"
        },
        "sample_metadata": {
            "f_metadata": f_metadata,
            "g_metadata": g_metadata
        }
    }

    return specs


def main():
    """Main discovery routine."""
    data_root = "/Users/owner/Projects/LBM"

    # Create output directory
    output_dir = Path(__file__).parent.parent / "artifacts"
    output_dir.mkdir(exist_ok=True)

    try:
        specs = discover_data_specs(data_root)

        # Save to JSON
        output_file = output_dir / "data_specs.json"
        with open(output_file, 'w') as f:
            json.dump(specs, f, indent=2, default=str)

        print(f"\n" + "="*50)
        print("DISCOVERY COMPLETE")
        print("="*50)
        print(f"Grid: {specs['grid_dimensions']['nx']}×{specs['grid_dimensions']['ny']}×{specs['grid_dimensions']['nz']}")
        print(f"Sites: {specs['grid_dimensions']['total_sites']:,}")
        print(f"Channels: {specs['channels']['total_channels']}")
        print(f"Timesteps: {specs['timesteps']['f_population']['count']}")
        print(f"Output: {output_file}")

    except Exception as e:
        print(f"Discovery failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())