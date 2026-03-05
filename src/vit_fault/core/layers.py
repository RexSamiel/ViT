"""Layer utilities for traversing and filtering model layers."""

import torch.nn as nn


def get_linear_layers(model: nn.Module, wrapper_class=None) -> dict[str, nn.Linear]:
    """Find all linear layers in a model.

    Args:
        model: PyTorch model
        wrapper_class: Optional wrapper class to unwrap (e.g., NeuroChecker)

    Returns:
        Dict mapping layer names to Linear modules
    """
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name:
            layers[name] = module
        elif wrapper_class and isinstance(module, wrapper_class):
            layers[name] = module.original
    return layers


def filter_layers(layers: dict, pattern: str) -> dict:
    """Filter layers by name pattern.

    Args:
        layers: Dict of layer name -> module
        pattern: Filter pattern ("all", "fc1", "fc2", "qkv", "proj", etc.)

    Returns:
        Filtered dict
    """
    if pattern == "all":
        return layers
    return {n: l for n, l in layers.items() if pattern in n}


def get_layer(model: nn.Module, name: str) -> nn.Module:
    """Get a layer by its dot-separated path.

    Args:
        model: The model
        name: Dot-separated path (e.g., "blocks.0.attn.qkv")

    Returns:
        The module at that path
    """
    parts = name.split(".")
    module = model
    for p in parts:
        module = getattr(module, p)
    return module


def set_layer(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a layer by its dot-separated path.

    Args:
        model: The model
        name: Dot-separated path
        new_module: Replacement module
    """
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)
