"""Layer discovery, classification, model structure queries, and parameter collection."""

import re
from typing import Any


def extract_block_idx(name: str) -> int | None:
    """Extract block index from module name. Handles ViT, Swin, etc.

    Args:
        name: Module name from model.named_modules()

    Returns:
        Block index (int) or None if not a block
    """
    name_lower = name.lower()

    # Swin-style: layers.X.blocks.Y
    swin_match = re.search(r"layers\.(\d+)\.blocks\.(\d+)", name_lower)
    if swin_match:
        return int(swin_match.group(1)) * 100 + int(swin_match.group(2))

    # Standard ViT: blocks.X
    for pattern in [r"blocks\.(\d+)", r"blocks_(\d+)", r"layer_(\d+)"]:
        match = re.search(pattern, name_lower)
        if match:
            return int(match.group(1))
    return None


def classify_component(name: str, block_idx: int | None, max_block_seen: int) -> str:
    """Classify module into: input, output, block, mha, or mlp.

    Args:
        name: Module name
        block_idx: Block index (from extract_block_idx)
        max_block_seen: Maximum block index seen so far

    Returns:
        Component classification string
    """
    name_lower = name.lower()

    if block_idx is None:
        return "input" if max_block_seen < 0 else "output"

    if "attn" in name_lower:
        return "mha"
    if "mlp" in name_lower:
        return "mlp"
    return "block"


def is_excluded(
    name: str, module_type: str, exclude_patterns: list[str] | None = None
) -> bool:
    """Check if layer should be excluded from distributions.

    Args:
        name: Module name
        module_type: Module class name
        exclude_patterns: List of patterns to exclude (default: None = empty list)

    Returns:
        True if layer should be excluded
    """
    if exclude_patterns is None:
        exclude_patterns = []

    name_lower = name.lower()
    type_lower = module_type.lower()
    return any(
        p.lower() in name_lower or p.lower() in type_lower for p in exclude_patterns
    )


def get_num_blocks(model) -> int:
    """Get total number of transformer blocks in the model."""
    if hasattr(model, "blocks"):
        return len(model.blocks)
    elif hasattr(model, "layers"):
        return sum(len(layer.blocks) for layer in model.layers)
    else:
        raise ValueError("Model does not have blocks or layers[].blocks")


def get_block(model, block_idx):
    """Get a specific transformer block by index, handling different architectures."""
    if hasattr(model, "blocks"):
        return model.blocks[block_idx]  # ViT / BEiT

    elif hasattr(model, "layers"):  # Swin
        for layer in model.layers:
            if block_idx < len(layer.blocks):
                return layer.blocks[block_idx]
            block_idx -= len(layer.blocks)

    raise ValueError("Invalid block_idx or unsupported model architecture.")


ATTENTION_PARAMS: dict[str, list[str]] = {
    "qkv": ["qkv.weight"],
    "proj": ["proj.weight"],
}

MLP_PARAMS: dict[str, list[str]] = {
    "fc1": ["fc1.weight"],
    "fc2": ["fc2.weight"],
}


def collect_attention_params(
    attn_module, sub_component: str | None, block_idx: int
) -> tuple[list[tuple[str, Any]], str | None]:
    """Collect named parameters from an attention module.

    Args:
        attn_module: The attention sub-module (e.g., block.attn)
        sub_component: Sub-component key ("qkv", "proj") or None for all
        block_idx: Block index for parameter naming

    Returns:
        Tuple of (params_list, resolved_sub_component)
        where params_list is [(prefixed_name, param_tensor), ...]
    """
    param_map = ATTENTION_PARAMS
    target_names = param_map.get(sub_component, []) if sub_component else None

    if sub_component is not None and sub_component not in param_map:
        raise ValueError(
            f"Invalid attention sub_component: {sub_component}. "
            f"Choose from: {list(param_map.keys())}"
        )

    available = []
    for name, param in attn_module.named_parameters():
        if target_names is None or name in target_names:
            available.append((f"Block{block_idx}.attn.{name}", param))

    return available, sub_component


def collect_mlp_params(
    mlp_module, sub_component: str | None, block_idx: int
) -> tuple[list[tuple[str, Any]], str | None]:
    """Collect named parameters from an MLP module.

    Args:
        mlp_module: The MLP sub-module (e.g., block.mlp)
        sub_component: Sub-component key ("fc1", "fc2") or None for all
        block_idx: Block index for parameter naming

    Returns:
        Tuple of (params_list, resolved_sub_component)
        where params_list is [(prefixed_name, param_tensor), ...]
    """
    param_map = MLP_PARAMS
    target_names = param_map.get(sub_component, []) if sub_component else None

    if sub_component is not None and sub_component not in param_map:
        raise ValueError(
            f"Invalid MLP sub_component: {sub_component}. "
            f"Choose from: {list(param_map.keys())}"
        )

    available = []
    for name, param in mlp_module.named_parameters():
        if target_names is None or name in target_names:
            available.append((f"Block{block_idx}.mlp.{name}", param))

    return available, sub_component


def collect_norm_params(block, block_idx: int) -> list[tuple[str, Any]]:
    """Collect normalization layer parameters from a block.

    Args:
        block: Transformer block module
        block_idx: Block index for parameter naming

    Returns:
        List of (prefixed_name, param_tensor) tuples
    """
    available = []
    for name, param in block.norm1.named_parameters():
        available.append((f"Block{block_idx}.norm1.{name}", param))
    for name, param in block.norm2.named_parameters():
        available.append((f"Block{block_idx}.norm2.{name}", param))
    return available


def collect_patch_embed_params(model) -> list[tuple[str, Any]]:
    """Collect patch embedding parameters from the model.

    Args:
        model: The full model

    Returns:
        List of (prefixed_name, param_tensor) tuples
    """
    return [
        (f"patch_embed.{name}", param)
        for name, param in model.patch_embed.named_parameters()
    ]


def collect_classifier_params(model) -> list[tuple[str, Any]]:
    """Collect classifier head parameters (norm + head) from the model.

    Args:
        model: The full model

    Returns:
        List of (prefixed_name, param_tensor) tuples
    """
    available = []
    if hasattr(model, "norm") and model.norm is not None:
        for name, param in model.norm.named_parameters():
            available.append((f"norm.{name}", param))
    if hasattr(model, "head") and model.head is not None:
        for name, param in model.head.named_parameters():
            available.append((f"head.{name}", param))
    return available


def get_tensor(
    layer_name: str, model: torch.nn.Module, x: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Runs the model on input x and returns the tensor from the layer named layer_name.

    Example layer_name:
        'blocks.0.attn.qkv'
        'blocks.5.mlp.fc1'
    """
    tensor_holder = {}

    def find_submodule(module, name):
        names = name.split(".")
        sub = module
        for n in names:
            if hasattr(sub, n):
                sub = getattr(sub, n)
            else:
                raise ValueError(f"Module has no submodule '{n}' in path '{name}'")
        return sub

    def hook_fn(module, input, output):
        tensor_holder["output"] = output

    submodule = find_submodule(model, layer_name)
    handle = submodule.register_forward_hook(hook_fn)

    _ = model(x)

    handle.remove()

    return tensor_holder.get("output")
