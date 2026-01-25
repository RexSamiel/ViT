import random
import torch
import numpy as np
from src.utils.formatting import format_fault_injection_info


def get_num_blocks(model):
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


def parse_bit_range(bit_spec):
    """Parse bit range specification into (min_bit, max_bit) tuple."""
    if isinstance(bit_spec, int):
        return bit_spec, bit_spec
    elif isinstance(bit_spec, tuple) and len(bit_spec) == 2:
        return bit_spec
    elif isinstance(bit_spec, str) and "..." in bit_spec:
        parts = bit_spec.split("...")
        return int(parts[0]), int(parts[1])
    else:
        raise ValueError(
            f"Invalid bit specification: {bit_spec}. "
            "Use int, tuple (min, max), or string 'min...max'"
        )


def flip_random_bit(value: torch.Tensor, bit_range=None):
    """
    Flip a random bit in a float32 value's IEEE 754 representation.
    Returns corrupted value, bit index, and binary representations.
    """
    if value.dtype != torch.float32:
        value = value.float()

    if bit_range is None:
        rand_bit = random.randint(0, 31)
    else:
        min_bit, max_bit = parse_bit_range(bit_range)
        rand_bit = random.randint(min_bit, max_bit)

    val_int = value.view(torch.int32)
    mask = torch.tensor(1, dtype=torch.int32, device=value.device) << rand_bit
    corrupted_int = val_int ^ mask

    corrupted_value = corrupted_int.view(torch.float32)
    original_bits = f"{val_int.item():032b}"
    corrupted_bits = f"{corrupted_int.item():032b}"

    return corrupted_value, rand_bit, original_bits, corrupted_bits


def _select_component_type(component_type):
    """Select component type, handling 'all' option."""
    if component_type == "all":
        return random.choice(["attention", "mlp", "norm", "patch_embed", "classifier"])
    return component_type


def _collect_attention_params(attn, sub_component, block_idx):
    """Collect attention parameters for fault injection."""
    attn_params = {
        "qkv": ["qkv.weight"],
        "proj": ["proj.weight"],
    }

    if sub_component is None:
        sub_component = random.choice(list(attn_params.keys()))
    elif sub_component not in attn_params:
        raise ValueError(
            f"Invalid attention sub_component: {sub_component}. "
            f"Choose from: {list(attn_params.keys())}"
        )

    available_params = []
    for name, param in attn.named_parameters():
        if name in attn_params[sub_component]:
            available_params.append((f"Block{block_idx}.attn.{name}", param))

    return available_params, sub_component


def _collect_mlp_params(mlp, sub_component, block_idx):
    """Collect MLP parameters for fault injection."""
    mlp_params = {
        "fc1": ["fc1.weight"],
        "fc2": ["fc2.weight"],
    }

    if sub_component is None:
        sub_component = random.choice(list(mlp_params.keys()))
    elif sub_component not in mlp_params:
        raise ValueError(
            f"Invalid MLP sub_component: {sub_component}. "
            f"Choose from: {list(mlp_params.keys())}"
        )

    available_params = []
    for name, param in mlp.named_parameters():
        if name in mlp_params[sub_component]:
            available_params.append((f"Block{block_idx}.mlp.{name}", param))

    return available_params, sub_component


def _collect_norm_params(block, block_idx):
    """Collect normalization layer parameters."""
    available_params = []
    for name, param in block.norm1.named_parameters():
        available_params.append((f"Block{block_idx}.norm1.{name}", param))
    for name, param in block.norm2.named_parameters():
        available_params.append((f"Block{block_idx}.norm2.{name}", param))
    return available_params


def _collect_patch_embed_params(model):
    """Collect patch embedding parameters."""
    available_params = []
    for name, param in model.patch_embed.named_parameters():
        available_params.append((f"patch_embed.{name}", param))
    return available_params


def _collect_classifier_params(model):
    """Collect classifier head parameters."""
    available_params = []
    if hasattr(model, "norm") and model.norm is not None:
        for name, param in model.norm.named_parameters():
            available_params.append((f"norm.{name}", param))
    if hasattr(model, "head") and model.head is not None:
        for name, param in model.head.named_parameters():
            available_params.append((f"head.{name}", param))
    return available_params


def inject_fault(
    model,
    component_type="attention",
    sub_component=None,
    block_idx=None,
    idx=None,
    bit_range=None,
    verbose=True,
):
    """
    Inject a single-bit fault into a random weight of the specified component.

    Supports targeting specific components (attention, mlp, norm, patch_embed, classifier)
    and sub-components (qkv/proj for attention, fc1/fc2 for mlp).
    """
    total_blocks = get_num_blocks(model)

    if block_idx is None:
        block_idx = random.randint(0, total_blocks - 1)

    block = get_block(model, block_idx)
    component_type = _select_component_type(component_type)

    # Collect parameters based on component type
    if component_type == "attention":
        available_params, sub_component = _collect_attention_params(
            block.attn, sub_component, block_idx
        )
    elif component_type == "mlp":
        available_params, sub_component = _collect_mlp_params(
            block.mlp, sub_component, block_idx
        )
    elif component_type == "norm":
        available_params = _collect_norm_params(block, block_idx)
    elif component_type == "patch_embed":
        available_params = _collect_patch_embed_params(model)
    elif component_type == "classifier":
        available_params = _collect_classifier_params(model)
    else:
        raise ValueError(f"Unknown component_type: {component_type}")

    if not available_params:
        raise ValueError(
            f"No suitable params for component_type: {component_type}, "
            f"sub_component: {sub_component}"
        )

    # Select random parameter and inject fault
    param_full_name, param = random.choice(available_params)
    if idx is None:
        idx = tuple(random.randint(0, s - 1) for s in param.shape)

    original_value = param[idx].clone()
    corrupted_value, bit_flipped, original_bits, corrupted_bits = flip_random_bit(
        original_value, bit_range
    )

    with torch.no_grad():
        param[idx] = corrupted_value

    fault_info = {
        "component_type": component_type,
        "sub_component": sub_component
        if component_type in ["attention", "mlp"]
        else None,
        "block_idx": block_idx,
        "param_name": param_full_name,
        "fault_idx": idx,
        "bit_range": bit_range,
        "bit_flipped": bit_flipped,
        "original_value": original_value.item(),
        "corrupted_value": corrupted_value.item(),
        "original_bits": original_bits,
        "corrupted_bits": corrupted_bits,
    }

    if verbose:
        print(format_fault_injection_info(fault_info))

    return fault_info
