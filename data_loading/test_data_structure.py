#!/usr/bin/env python3
"""
Basic test of data structure without numpy/yt dependencies.
Tests header parsing and directory structure validation.
"""

import os
import re
import json
from pathlib import Path


def test_directory_structure(data_root: str):
    """Test that expected directories and files exist."""

    print("Testing directory structure...")

    # Check population directories
    pop_f_dir = os.path.join(data_root, "population_f")
    pop_g_dir = os.path.join(data_root, "population_g")

    assert os.path.exists(pop_f_dir), f"population_f directory missing: {pop_f_dir}"
    assert os.path.exists(pop_g_dir), f"population_g directory missing: {pop_g_dir}"

    print(f"✓ Found population directories")

    # Check for plotfile directories
    pattern = r"plt1001\d{3}"

    f_entries = [d for d in os.listdir(pop_f_dir) if re.fullmatch(pattern, d)]
    g_entries = [d for d in os.listdir(pop_g_dir) if re.fullmatch(pattern, d)]

    print(f"✓ Found {len(f_entries)} f-population timesteps")
    print(f"✓ Found {len(g_entries)} g-population timesteps")

    assert len(f_entries) > 0, "No f-population plotfiles found"
    assert len(g_entries) > 0, "No g-population plotfiles found"

    # Check first timestep structure
    first_f = os.path.join(pop_f_dir, f_entries[0])
    first_g = os.path.join(pop_g_dir, g_entries[0])

    f_header = os.path.join(first_f, "Header")
    g_header = os.path.join(first_g, "Header")

    assert os.path.exists(f_header), f"F header missing: {f_header}"
    assert os.path.exists(g_header), f"G header missing: {g_header}"

    print(f"✓ Headers exist for first timesteps")

    return len(f_entries), len(g_entries)


def test_header_parsing(data_root: str):
    """Test header file parsing."""

    print("\nTesting header parsing...")

    # Get first f and g headers
    pop_f_dir = os.path.join(data_root, "population_f")
    pop_g_dir = os.path.join(data_root, "population_g")

    pattern = r"plt1001\d{3}"
    f_entries = sorted([d for d in os.listdir(pop_f_dir) if re.fullmatch(pattern, d)])
    g_entries = sorted([d for d in os.listdir(pop_g_dir) if re.fullmatch(pattern, d)])

    f_header = os.path.join(pop_f_dir, f_entries[0], "Header")
    g_header = os.path.join(pop_g_dir, g_entries[0], "Header")

    def parse_header(header_path, expected_prefix):
        with open(header_path, 'r') as f:
            lines = f.readlines()

        # Check format
        assert lines[0].strip() == "HyperCLaw-V1.1", f"Unexpected format in {header_path}"

        # Check channel count
        num_channels = int(lines[1].strip())
        assert num_channels == 19, f"Expected 19 channels, got {num_channels}"

        # Check channel names
        for i in range(19):
            expected_name = f"{expected_prefix}{i}"
            actual_name = lines[2 + i].strip()
            assert actual_name == expected_name, f"Expected {expected_name}, got {actual_name}"

        # Look for grid dimensions
        for line in lines:
            if '((' in line and '))' in line:
                match = re.search(r'\(\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\)', line)
                if match:
                    x_min, y_min, z_min = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    x_max, y_max, z_max = int(match.group(4)), int(match.group(5)), int(match.group(6))

                    nx = x_max - x_min + 1
                    ny = y_max - y_min + 1
                    nz = z_max - z_min + 1

                    return nx, ny, nz

        raise ValueError(f"Could not parse grid dimensions from {header_path}")

    # Parse both headers
    f_dims = parse_header(f_header, "f")
    g_dims = parse_header(g_header, "g")

    print(f"✓ F-population: 19 channels, grid {f_dims}")
    print(f"✓ G-population: 19 channels, grid {g_dims}")

    # Verify grids match
    assert f_dims == g_dims, f"Grid mismatch: f={f_dims}, g={g_dims}"

    print(f"✓ Grid dimensions match: {f_dims}")

    return f_dims


def test_timestep_sequence(data_root: str):
    """Test timestep sequence and numbering."""

    print("\nTesting timestep sequence...")

    pop_f_dir = os.path.join(data_root, "population_f")
    pattern = r"plt1001\d{3}"

    f_entries = [d for d in os.listdir(pop_f_dir) if re.fullmatch(pattern, d)]

    # Extract timestep numbers
    def extract_timestep(dirname):
        match = re.search(r"(\d+)$", dirname)
        return int(match.group(1)) if match else None

    timesteps = sorted([extract_timestep(entry) for entry in f_entries])

    print(f"✓ Timestep range: {timesteps[0]} to {timesteps[-1]}")
    print(f"✓ Total timesteps: {len(timesteps)}")

    # Check for gaps
    expected_range = list(range(timesteps[0], timesteps[-1] + 1))
    missing = set(expected_range) - set(timesteps)

    if missing:
        print(f"⚠ Missing timesteps: {sorted(missing)}")
    else:
        print(f"✓ No gaps in timestep sequence")

    return timesteps


def create_test_report(data_root: str):
    """Create a test report with findings."""

    output_dir = Path(__file__).parent.parent / "artifacts"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*50)
    print("RUNNING BASIC DATA STRUCTURE TESTS")
    print("="*50)

    results = {
        "test_type": "basic_structure_validation",
        "data_root": data_root,
        "tests": {}
    }

    try:
        # Test directory structure
        f_count, g_count = test_directory_structure(data_root)
        results["tests"]["directory_structure"] = {
            "status": "PASS",
            "f_timesteps": f_count,
            "g_timesteps": g_count
        }

        # Test header parsing
        grid_dims = test_header_parsing(data_root)
        results["tests"]["header_parsing"] = {
            "status": "PASS",
            "grid_dimensions": grid_dims,
            "total_sites": grid_dims[0] * grid_dims[1] * grid_dims[2],
            "channels_per_population": 19,
            "total_channels": 38
        }

        # Test timestep sequence
        timesteps = test_timestep_sequence(data_root)
        results["tests"]["timestep_sequence"] = {
            "status": "PASS",
            "count": len(timesteps),
            "range": [timesteps[0], timesteps[-1]]
        }

        results["overall_status"] = "PASS"
        results["summary"] = {
            "grid_shape": grid_dims,
            "total_sites": grid_dims[0] * grid_dims[1] * grid_dims[2],
            "channels": 38,
            "timesteps": len(timesteps),
            "ml_ready_shape": f"({len(timesteps)-1} * {grid_dims[0] * grid_dims[1] * grid_dims[2]}, 38)"
        }

    except Exception as e:
        results["overall_status"] = "FAIL"
        results["error"] = str(e)
        print(f"\n❌ Test failed: {e}")
        return results

    # Save results
    output_file = output_dir / "basic_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print("BASIC TESTS COMPLETE")
    print("="*50)
    print(f"Grid: {grid_dims[0]}×{grid_dims[1]}×{grid_dims[2]}")
    print(f"Sites: {grid_dims[0] * grid_dims[1] * grid_dims[2]:,}")
    print(f"Channels: 38")
    print(f"Timesteps: {len(timesteps)}")
    print(f"ML shape: ({len(timesteps)-1} * {grid_dims[0] * grid_dims[1] * grid_dims[2]:,}, 38)")
    print(f"Results: {output_file}")

    return results


def main():
    data_root = "/Users/owner/Projects/LBM"

    try:
        results = create_test_report(data_root)
        return 0 if results["overall_status"] == "PASS" else 1
    except Exception as e:
        print(f"Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())