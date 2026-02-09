"""Histogram sampling and computation for activation distributions."""

import torch
import numpy as np

_BIN_RANGE = 10000
_NUM_BINS = 2 * _BIN_RANGE + 1  # 20001 bins total


def sample_and_histogram(
    tensor: torch.Tensor,
    sampling_percent: float,
    hist_counts: np.ndarray,
    bin_range: int,
) -> tuple[int, float, float]:
    """Sample activations from tensor and update histogram counts in-place.

    Args:
        tensor: Activation tensor to sample from
        sampling_percent: Percentage of values to sample (0.01-100)
        hist_counts: Numpy array to accumulate histogram counts (modified in-place)
        bin_range: Range for histogram bins (-bin_range to +bin_range)

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
    sampled_min, sampled_max = sampled.min(), sampled.max()

    # Update histogram counts
    bin_indices = np.floor(sampled).astype(np.int64) + bin_range
    bin_indices = np.clip(bin_indices, 0, 2 * bin_range)
    np.add.at(hist_counts, bin_indices, 1)

    return actual_sampled, float(sampled_min), float(sampled_max)


def compute_histogram(
    hist_counts: np.ndarray, bin_range: int, data_range: dict, activation_counts: dict
) -> dict:
    """Compute final histogram result from accumulated counts.

    Args:
        hist_counts: Numpy array of histogram counts
        bin_range: Range for bins
        data_range: Dict with "min" and "max" keys for actual data range
        activation_counts: Dict with "total" and "sampled" keys

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
    int_min = first_nonzero - bin_range
    int_max = last_nonzero - bin_range

    # Create bin edges and centers
    edges = np.arange(int_min, int_max + 2, 1.0)
    centers = np.arange(int_min, int_max + 1, 1.0) + 0.5

    # Get actual data range
    data_min = data_range["min"] if data_range["min"] != float("inf") else int_min
    data_max = data_range["max"] if data_range["max"] != float("-inf") else int_max

    return {
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "counts": trimmed_counts.tolist(),
        "total_sampled": total_sampled,
        "total_activations": activation_counts["total"],
        "data_range": [float(data_min), float(data_max)],
    }
