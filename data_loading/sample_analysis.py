#!/usr/bin/env python3
"""
Comprehensive statistical analysis of 38-channel LBM data.

Analyzes:
- Per-channel statistics (min/max/mean/std)
- Data quality (NaN, inf, negative values)
- Channel correlations
- Temporal consistency
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add local imports
sys.path.append(str(Path(__file__).parent))
from simple_loader import load_training_data, load_combined_sequence, flatten_for_training


def analyze_channel_statistics(data: np.ndarray) -> Dict[str, Any]:
    """
    Compute per-channel statistics for 38-channel data.

    Args:
        data: Array of shape (N, 38)

    Returns:
        Dictionary with per-channel stats
    """

    assert data.shape[1] == 38, f"Expected 38 channels, got {data.shape[1]}"

    stats = {
        "per_channel": {},
        "summary": {}
    }

    # Per-channel statistics
    for ch in range(38):
        ch_data = data[:, ch]

        # Determine if this is f or g population
        pop_type = "f" if ch < 19 else "g"
        ch_in_pop = ch if ch < 19 else ch - 19

        channel_name = f"{pop_type}{ch_in_pop}"

        stats["per_channel"][channel_name] = {
            "channel_index": ch,
            "population": pop_type,
            "min": float(np.min(ch_data)),
            "max": float(np.max(ch_data)),
            "mean": float(np.mean(ch_data)),
            "std": float(np.std(ch_data)),
            "median": float(np.median(ch_data)),
            "q25": float(np.percentile(ch_data, 25)),
            "q75": float(np.percentile(ch_data, 75)),
            "zeros": int(np.sum(ch_data == 0)),
            "negatives": int(np.sum(ch_data < 0)),
            "nans": int(np.sum(np.isnan(ch_data))),
            "infs": int(np.sum(np.isinf(ch_data)))
        }

    # Summary statistics
    all_mins = [stats["per_channel"][ch]["min"] for ch in stats["per_channel"]]
    all_maxs = [stats["per_channel"][ch]["max"] for ch in stats["per_channel"]]
    all_means = [stats["per_channel"][ch]["mean"] for ch in stats["per_channel"]]
    all_stds = [stats["per_channel"][ch]["std"] for ch in stats["per_channel"]]

    stats["summary"] = {
        "total_samples": data.shape[0],
        "total_channels": data.shape[1],
        "global_min": float(np.min(all_mins)),
        "global_max": float(np.max(all_maxs)),
        "mean_of_means": float(np.mean(all_means)),
        "std_of_means": float(np.std(all_means)),
        "mean_of_stds": float(np.mean(all_stds)),
        "total_negatives": sum(stats["per_channel"][ch]["negatives"] for ch in stats["per_channel"]),
        "total_nans": sum(stats["per_channel"][ch]["nans"] for ch in stats["per_channel"]),
        "total_infs": sum(stats["per_channel"][ch]["infs"] for ch in stats["per_channel"]),
        "data_health": "GOOD" if (
            sum(stats["per_channel"][ch]["nans"] for ch in stats["per_channel"]) == 0 and
            sum(stats["per_channel"][ch]["infs"] for ch in stats["per_channel"]) == 0
        ) else "ISSUES_DETECTED"
    }

    return stats


def analyze_correlations(data: np.ndarray, max_samples: int = 10000) -> Dict[str, Any]:
    """
    Compute channel correlation matrix.

    Args:
        data: Array of shape (N, 38)
        max_samples: Maximum samples to use for correlation (for speed)

    Returns:
        Correlation analysis results
    """

    # Subsample if data is too large
    if data.shape[0] > max_samples:
        indices = np.random.choice(data.shape[0], max_samples, replace=False)
        sample_data = data[indices]
    else:
        sample_data = data

    # Compute correlation matrix
    corr_matrix = np.corrcoef(sample_data.T)

    # Analyze correlations
    results = {
        "correlation_matrix_shape": corr_matrix.shape,
        "samples_used": sample_data.shape[0],
        "analysis": {}
    }

    # Find strong correlations (excluding diagonal)
    mask = ~np.eye(38, dtype=bool)
    corr_values = corr_matrix[mask]

    results["analysis"] = {
        "max_correlation": float(np.max(corr_values)),
        "min_correlation": float(np.min(corr_values)),
        "mean_correlation": float(np.mean(corr_values)),
        "std_correlation": float(np.std(corr_values)),
        "strong_correlations_count": int(np.sum(np.abs(corr_values) > 0.8)),
        "moderate_correlations_count": int(np.sum((np.abs(corr_values) > 0.5) & (np.abs(corr_values) <= 0.8)))
    }

    # Find strongest cross-population correlations (f vs g channels)
    f_indices = list(range(19))
    g_indices = list(range(19, 38))

    cross_corrs = []
    for f_idx in f_indices:
        for g_idx in g_indices:
            cross_corrs.append(abs(corr_matrix[f_idx, g_idx]))

    results["analysis"]["cross_population"] = {
        "max_f_g_correlation": float(np.max(cross_corrs)),
        "mean_f_g_correlation": float(np.mean(cross_corrs)),
        "strong_f_g_correlations": int(np.sum(np.array(cross_corrs) > 0.5))
    }

    return results


def analyze_temporal_consistency(data_root: str, num_timesteps: int = 10) -> Dict[str, Any]:
    """
    Analyze temporal consistency across timesteps.

    Args:
        data_root: Root directory
        num_timesteps: Number of timesteps to analyze

    Returns:
        Temporal analysis results
    """

    print(f"Analyzing temporal consistency over {num_timesteps} timesteps...")

    # Load sequence without flattening to preserve temporal structure
    sequence = load_combined_sequence(data_root, start_idx=0, num_timesteps=num_timesteps)

    results = {
        "timesteps_analyzed": len(sequence),
        "grid_shape": sequence[0].shape[:3],
        "temporal_stats": {}
    }

    # Compute statistics per timestep
    timestep_stats = []
    for t, timestep_data in enumerate(sequence):
        flat_data = flatten_for_training(timestep_data)

        timestep_stat = {
            "timestep": t,
            "total_mass": float(np.sum(flat_data)),
            "mean_value": float(np.mean(flat_data)),
            "std_value": float(np.std(flat_data)),
            "min_value": float(np.min(flat_data)),
            "max_value": float(np.max(flat_data))
        }
        timestep_stats.append(timestep_stat)

    # Analyze temporal trends
    total_masses = [stat["total_mass"] for stat in timestep_stats]
    mean_values = [stat["mean_value"] for stat in timestep_stats]

    results["temporal_stats"] = {
        "per_timestep": timestep_stats,
        "trends": {
            "mass_drift": {
                "initial_mass": total_masses[0],
                "final_mass": total_masses[-1],
                "relative_change": (total_masses[-1] - total_masses[0]) / total_masses[0],
                "max_fluctuation": float(np.max(total_masses) - np.min(total_masses)),
                "std_fluctuation": float(np.std(total_masses))
            },
            "mean_value_trend": {
                "initial_mean": mean_values[0],
                "final_mean": mean_values[-1],
                "relative_change": (mean_values[-1] - mean_values[0]) / mean_values[0]
            }
        }
    }

    return results


def main():
    """Complete statistical analysis pipeline."""

    data_root = "/Users/owner/Projects/LBM"
    output_dir = Path(__file__).parent.parent / "artifacts"
    output_dir.mkdir(exist_ok=True)

    print("=== Comprehensive LBM Data Analysis ===")

    try:
        # Load sample data for analysis
        print("\n1. Loading training data...")
        X, y = load_training_data(data_root, start_idx=0, num_timesteps=10, flatten=True)

        print(f"Loaded: X={X.shape}, y={y.shape}")

        # Analyze pre-collision states
        print("\n2. Analyzing pre-collision statistics...")
        pre_stats = analyze_channel_statistics(X)

        # Analyze post-collision states
        print("\n3. Analyzing post-collision statistics...")
        post_stats = analyze_channel_statistics(y)

        # Correlation analysis
        print("\n4. Computing correlations...")
        pre_corr = analyze_correlations(X, max_samples=5000)
        post_corr = analyze_correlations(y, max_samples=5000)

        # Temporal consistency
        print("\n5. Analyzing temporal consistency...")
        temporal_stats = analyze_temporal_consistency(data_root, num_timesteps=10)

        # Compile all results
        full_analysis = {
            "metadata": {
                "analysis_type": "comprehensive_38_channel_statistics",
                "data_shape": {
                    "pre_collision": X.shape,
                    "post_collision": y.shape
                },
                "data_source": data_root
            },
            "pre_collision_stats": pre_stats,
            "post_collision_stats": post_stats,
            "correlations": {
                "pre_collision": pre_corr,
                "post_collision": post_corr
            },
            "temporal_analysis": temporal_stats
        }

        # Save results
        output_file = output_dir / "data_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(full_analysis, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Samples analyzed: {X.shape[0]:,}")
        print(f"Channels: {X.shape[1]}")
        print(f"Data health: {pre_stats['summary']['data_health']}")
        print(f"Global range: [{pre_stats['summary']['global_min']:.6f}, {pre_stats['summary']['global_max']:.6f}]")
        print(f"Negatives: {pre_stats['summary']['total_negatives']}")
        print(f"NaN/Inf: {pre_stats['summary']['total_nans']}/{pre_stats['summary']['total_infs']}")
        print(f"Results saved: {output_file}")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())