"""Layer utilities for traversing and filtering model layers."""

from collections.abc import Mapping
from typing import TypeVar

import torch.nn as nn

W = TypeVar("W", bound=nn.Module)


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


def get_attention_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Find all attention modules in a ViT model.

    Args:
        model: PyTorch model (assumes timm ViT structure)

    Returns:
        Dict mapping module names to Attention modules
    """
    modules = {}
    for name, module in model.named_modules():
        # Check for timm Attention class
        if module.__class__.__name__ == "Attention" and hasattr(module, "qkv"):
            modules[name] = module
    return modules


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


def wrap_layers(
    model: nn.Module,
    wrapper_cls: type[W],
    layers: str = "all",
) -> dict[str, W]:
    """Wrap linear layers with a wrapper class.

    Args:
        model: The model (or model.net)
        wrapper_cls: Wrapper class with __init__(original, name)
        layers: Filter pattern ("all", "fc1", "fc2", "qkv", "proj")

    Returns:
        Dict of wrapped layers {name: wrapper}
    """
    target_layers = filter_layers(get_linear_layers(model), layers)
    wrapped = {}

    for name in target_layers:
        original = get_layer(model, name)
        wrapper = wrapper_cls(original, name)
        set_layer(model, name, wrapper)
        wrapped[name] = wrapper

    return wrapped


def unwrap_layers(model: nn.Module, wrapped: Mapping[str, nn.Module]):
    """Restore original layers from wrappers.

    Args:
        model: The model
        wrapped: Dict of {name: wrapper} from wrap_layers()
    """
    for name, wrapper in wrapped.items():
        if hasattr(wrapper, "original"):
            original = wrapper.original  # type: ignore[union-attr]
            assert isinstance(original, nn.Module)
            set_layer(model, name, original)
