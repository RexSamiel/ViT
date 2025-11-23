import random
import torch
import numpy as np


def get_num_blocks(model):
    if hasattr(model, "blocks"):
        return len(model.blocks)
    elif hasattr(model, "layers"):
        return sum(len(layer.blocks) for layer in model.layers)
    else:
        raise ValueError("Model does not have blocks or layers[].blocks")


def get_block(model, block_idx):
    if hasattr(model, "blocks"):
        return model.blocks[block_idx]  # ViT / BEiT

    elif hasattr(model, "layers"):  # Swin
        for layer in model.layers:
            if block_idx < len(layer.blocks):
                return layer.blocks[block_idx]
            block_idx -= len(layer.blocks)

    raise ValueError("Invalid block_idx or unsupported model architecture.")


def parse_bit_range(bit_spec):
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


def format_ieee754_bits(bits_str: str) -> str:
    bits_str = bits_str.replace("-", "").replace("+", "").zfill(32)
    return (
        f"Sign  Exponent   Mantissa\n {bits_str[0]}    {bits_str[1:9]}  {bits_str[9:]}"
    )


def inject_fault(
    model,
    component_type="attention",
    block_idx=None,
    idx=None,
    bit_range=None,
    verbose=True,
):
    available_params = []

    total_blocks = get_num_blocks(model)

    if block_idx is None:
        block_idx = random.randint(0, total_blocks - 1)

    block = get_block(model, block_idx)

    if component_type == "all":
        component_type = random.choice(
            ["attention", "norm", "mlp", "patch_embed", "classifier"]
        )

    if component_type == "attention":
        attn = block.attn
        for name, param in attn.named_parameters():
            if name == "qkv.weight":
                available_params.append((f"Block{block_idx}.attn.{name}", param))

    elif component_type == "norm":
        for name, param in block.norm1.named_parameters():
            available_params.append((f"Block{block_idx}.norm1.{name}", param))
        for name, param in block.norm2.named_parameters():
            available_params.append((f"Block{block_idx}.norm2.{name}", param))

    elif component_type == "mlp":
        mlp = block.mlp
        for name, param in mlp.named_parameters():
            if name in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
                available_params.append((f"Block{block_idx}.mlp.{name}", param))

    elif component_type == "patch_embed":
        for name, param in model.patch_embed.named_parameters():
            available_params.append((f"patch_embed.{name}", param))

    elif component_type == "classifier":
        if hasattr(model, "norm") and model.norm is not None:
            for name, param in model.norm.named_parameters():
                available_params.append((f"norm.{name}", param))
        if hasattr(model, "head") and model.head is not None:
            for name, param in model.head.named_parameters():
                available_params.append((f"head.{name}", param))

    if not available_params:
        raise ValueError(f"No suitable params for component_type: {component_type}")

    param_full_name, param = random.choice(available_params)
    if idx is None:
        idx = tuple(random.randint(0, s - 1) for s in param.shape)

    print(param_full_name)
    original_value = param[idx].clone()
    corrupted_value, bit_flipped, original_bits, corrupted_bits = flip_random_bit(
        original_value, bit_range
    )

    with torch.no_grad():
        param[idx] = corrupted_value

    fault_info = {
        "component_type": component_type,
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
        print(f"""
Fault Injection Details
{"-" * 80}
Component Type : {component_type}
Block Index    : {block_idx}
Parameter Name : {param_full_name}
Fault Index    : {idx}
Bit Flipped    : {bit_flipped}
Original Value : {original_value.item():.8f}
Corrupted Value: {corrupted_value.item():.8f}

Original Bits:
{format_ieee754_bits(original_bits)}

Corrupted Bits:
{format_ieee754_bits(corrupted_bits)}
{"-" * 80}
""")

    return fault_info
