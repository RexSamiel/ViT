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


def save_checker_weights(weights: Dict[str, NeuroWeights], model_key: str) -> Path:
    """Save checker weights to disk.

    Args:
        weights: Dict of layer name to weights
        model_key: Model identifier for filename

    Returns:
        Path to saved file
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / f"neuro_weights_{model_key}.pt"

    data = {}
    for name, w in weights.items():
        data[name] = {
            "checker_row": w.checker_row,
            "checker_bias": w.checker_bias,
        }

    torch.save(data, filepath)
    return filepath


def load_checker_weights(model_key: str) -> Dict[str, NeuroWeights] | None:
    """Load checker weights from disk.

    Args:
        model_key: Model identifier for filename

    Returns:
        Dict of layer name to weights, or None if not found
    """
    filepath = DATA_DIR / f"neuro_weights_{model_key}.pt"
    if not filepath.exists():
        return None

    data = torch.load(filepath, weights_only=True)
    weights = {}

    for name, d in data.items():
        weights[name] = NeuroWeights(
            checker_row=d["checker_row"],
            checker_bias=d["checker_bias"],
        )

    return weights


def get_or_compute_checker_weights(
    model: nn.Module,
    model_key: str,
    force_recompute: bool = False,
) -> Dict[str, NeuroWeights]:
    """Get checker weights, computing and saving if needed.

    Args:
        model: The model
        model_key: Model identifier
        force_recompute: If True, recompute even if cached

    Returns:
        Dict of layer name to weights
    """
    if not force_recompute:
        cached = load_checker_weights(model_key)
        if cached is not None:
            return cached

    weights = compute_neuro_weights(model)
    save_checker_weights(weights, model_key)
    return weights
