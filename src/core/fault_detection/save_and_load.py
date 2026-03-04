import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

from src.core.library.layers import get_linear_layers


DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


@dataclass
class NeuroWeights:
    """Mean-based checker weights for a single linear layer."""

    checker_row: torch.Tensor  # (in_features,) - mean of each column
    checker_bias: torch.Tensor | None  # scalar - mean of bias if present


@dataclass
class ChecksumWeights:
    """Sum-based ABFT checksums for a single linear layer."""

    col_sums: torch.Tensor
    row_sums: torch.Tensor
    total_sum: torch.Tensor
    bias_sum: torch.Tensor | None


def compute_neuro_weights(model: nn.Module) -> Dict[str, NeuroWeights]:
    """Compute mean-based checker weights for each linear layer.

    Args:
        model: The model to analyze

    Returns:
        Dict mapping layer names to NeuroWeights
    """
    linear_layers = get_linear_layers(model)
    weights = {}

    for name, layer in linear_layers.items():
        W = layer.weight.data
        weights[name] = NeuroWeights(
            checker_row=W.mean(dim=0).clone(),
            checker_bias=layer.bias.data.mean().clone()
            if layer.bias is not None
            else None,
        )

    return weights


def compute_checksum_weights(model: nn.Module) -> Dict[str, ChecksumWeights]:
    """Compute sum-based ABFT checksums for each linear layer.

    Args:
        model: The model to analyze

    Returns:
        Dict mapping layer names to ChecksumWeights
    """
    linear_layers = get_linear_layers(model)
    checksums = {}

    for name, layer in linear_layers.items():
        W = layer.weight.data
        checksums[name] = ChecksumWeights(
            col_sums=W.sum(dim=0).clone(),
            row_sums=W.sum(dim=1).clone(),
            total_sum=W.sum().clone(),
            bias_sum=layer.bias.data.sum().clone() if layer.bias is not None else None,
        )

    return checksums


def save_checker_weights(
    weights: Dict[str, NeuroWeights | ChecksumWeights],
    model_key: str,
    method: str = "neuro",
) -> Path:
    """Save checker weights to disk.

    Args:
        weights: Dict of layer name to weights
        model_key: Model identifier for filename
        method: "neuro" or "checksum"

    Returns:
        Path to saved file
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / f"{method}_weights_{model_key}.pt"

    data = {}
    for name, w in weights.items():
        if isinstance(w, NeuroWeights):
            data[name] = {
                "checker_row": w.checker_row,
                "checker_bias": w.checker_bias,
            }
        else:  # ChecksumWeights
            data[name] = {
                "col_sums": w.col_sums,
                "row_sums": w.row_sums,
                "total_sum": w.total_sum,
                "bias_sum": w.bias_sum,
            }

    torch.save(data, filepath)
    return filepath


def load_checker_weights(
    model_key: str, method: str = "neuro"
) -> Dict[str, NeuroWeights | ChecksumWeights] | None:
    """Load checker weights from disk.

    Args:
        model_key: Model identifier for filename
        method: "neuro" or "checksum"

    Returns:
        Dict of layer name to weights, or None if not found
    """
    filepath = DATA_DIR / f"{method}_weights_{model_key}.pt"
    if not filepath.exists():
        return None

    data = torch.load(filepath, weights_only=True)
    weights = {}

    for name, d in data.items():
        if method == "neuro":
            weights[name] = NeuroWeights(
                checker_row=d["checker_row"],
                checker_bias=d["checker_bias"],
            )
        else:  # checksum
            weights[name] = ChecksumWeights(
                col_sums=d["col_sums"],
                row_sums=d["row_sums"],
                total_sum=d["total_sum"],
                bias_sum=d["bias_sum"],
            )

    return weights


def get_or_compute_checker_weights(
    model: nn.Module,
    model_key: str,
    method: str = "neuro",
    force_recompute: bool = False,
) -> Dict[str, NeuroWeights | ChecksumWeights]:
    """Get checker weights, computing and saving if needed.

    Args:
        model: The model
        model_key: Model identifier
        method: "neuro" or "checksum"
        force_recompute: If True, recompute even if cached

    Returns:
        Dict of layer name to weights
    """
    if not force_recompute:
        cached = load_checker_weights(model_key, method)
        if cached is not None:
            return cached

    if method == "neuro":
        weights = compute_neuro_weights(model)
    else:
        weights = compute_checksum_weights(model)

    save_checker_weights(weights, model_key, method)
    return weights
