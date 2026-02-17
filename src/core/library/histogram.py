"""Histogram sampling and computation for activation and weight distributions."""

import torch
import numpy as np

# Configuration for different analysis types
_ACTIVATION_BIN_RANGE = 10000
_ACTIVATION_BIN_RESOLUTION = 1.0  # Integer bins

_WEIGHT_BIN_RANGE = 100
_WEIGHT_BIN_RESOLUTION = 0.01  # Fine-grained bins for weights

# Legacy constants for backward compatibility
_BIN_RANGE = _ACTIVATION_BIN_RANGE
_NUM_BINS = 2 * _BIN_RANGE + 1


def get_histogram_config(analysis_type: str) -> tuple[int, float, int]:
    """Get histogram configuration for analysis type.

    Args:
        analysis_type: "aa" for activations or "wa" for weights

    Returns:
        Tuple of (bin_range, bin_resolution, num_bins)
    """
    if analysis_type == "wa":
        # For weights: finer resolution, smaller range
        bin_range = _WEIGHT_BIN_RANGE
        bin_resolution = _WEIGHT_BIN_RESOLUTION
        num_bins = int(2 * bin_range / bin_resolution) + 1
    else:  # "aa" or default
        # For activations: integer bins, larger range
        bin_range = _ACTIVATION_BIN_RANGE
        bin_resolution = _ACTIVATION_BIN_RESOLUTION
        num_bins = int(2 * bin_range / bin_resolution) + 1

    return bin_range, bin_resolution, num_bins


def sample_and_histogram(
    tensor: torch.Tensor,
    sampling_percent: float,
    hist_counts: np.ndarray,
    bin_range: int,
    bin_resolution: float = 1.0,
) -> tuple[int, float, float]:
    """Sample values from tensor and update histogram counts in-place.

    Args:
        tensor: Tensor to sample from (activations or weights)
        sampling_percent: Percentage of values to sample (0.01-100)
        hist_counts: Numpy array to accumulate histogram counts (modified in-place)
        bin_range: Range for histogram bins (-bin_range to +bin_range)
        bin_resolution: Resolution of bins (1.0 for integer bins, 0.01 for fine-grained)

    Returns:
        Tuple of (num_sampled, sampled_min, sampled_max)
    """
    num_elements = tensor.numel()
    sample_count = max(1, int(num_elements * sampling_percent / 100))
    actual_sampled = min(sample_count, num_elements)

    # Efficient sampling
    flat = tensor.detach().reshape(-1)
    if sample_count < num_elements:
        indices = torch.randint(0, num_elements, (sample_count,), device=flat.device)
        sampled = flat[indices].float().cpu().numpy()
    else:
        sampled = flat.float().cpu().numpy()

    # Get min/max
    if len(sampled) == 0:
        return 0, 0.0, 0.0

    sampled_min, sampled_max = sampled.min(), sampled.max()

    # Update histogram counts with appropriate binning
    # Divide by resolution to get bin indices, then floor
    bin_indices = np.floor(sampled / bin_resolution).astype(np.int64) + int(bin_range / bin_resolution)
    bin_indices = np.clip(bin_indices, 0, len(hist_counts) - 1)
    np.add.at(hist_counts, bin_indices, 1)

    return actual_sampled, float(sampled_min), float(sampled_max)


def compute_histogram(
    hist_counts: np.ndarray,
    bin_range: int,
    data_range: dict,
    activation_counts: dict,
    bin_resolution: float = 1.0,
) -> dict:
    """Compute final histogram result from accumulated counts.

    Args:
        hist_counts: Numpy array of histogram counts
        bin_range: Range for bins
        data_range: Dict with "min" and "max" keys for actual data range
        activation_counts: Dict with "total" and "sampled" keys
        bin_resolution: Resolution of bins (1.0 for integer bins, 0.01 for fine-grained)

    Returns:
        Dictionary with histogram data, or empty dict if no data
    """
    total_sampled = int(hist_counts.sum())

    if total_sampled == 0:
        return {}

    # Find the actual range with non-zero counts to trim output
    nonzero_indices = np.nonzero(hist_counts)[0]
    if len(nonzero_indices) == 0:
        return {}

    first_nonzero = nonzero_indices[0]
    last_nonzero = nonzero_indices[-1]

    # Extract only the relevant portion
    trimmed_counts = hist_counts[first_nonzero : last_nonzero + 1]

    # Convert bin indices back to values
    # Account for bin resolution when converting indices to actual values
    bin_offset = int(bin_range / bin_resolution)
    value_min = (first_nonzero - bin_offset) * bin_resolution
    value_max = (last_nonzero - bin_offset) * bin_resolution

    # Create bin edges and centers
    edges = np.arange(value_min, value_max + 2 * bin_resolution, bin_resolution)
    centers = np.arange(value_min, value_max + bin_resolution, bin_resolution) + bin_resolution / 2

    # Get actual data range
    data_min = data_range["min"] if data_range["min"] != float("inf") else value_min
    data_max = data_range["max"] if data_range["max"] != float("-inf") else value_max

    return {
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "counts": trimmed_counts.tolist(),
        "total_sampled": total_sampled,
        "total_activations": activation_counts["total"],
        "data_range": [float(data_min), float(data_max)],
        "bin_resolution": bin_resolution,
    }
