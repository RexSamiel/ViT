import random
import torch
import numpy as np

def flip_random_bit(value: torch.Tensor) -> tuple[torch.Tensor, int, str, str]:
    """Flip a random bit in a 32-bit float tensor element and return bit info."""
    if value.dtype != torch.float32:
        value = value.float()

    # Pick a random bit
    rand_bit = random.randint(0, 31)

    # View the float as int for bit manipulation
    val_int = value.view(torch.int32)

    # Create mask and flip the bit
    mask_value = np.int32(1 << rand_bit)
    mask = torch.tensor(mask_value, dtype=torch.int32, device=value.device)
    corrupted_int = val_int ^ mask
    corrupted_value = corrupted_int.view(torch.float32)

    # Convert to 32-bit binary strings
    original_bits = f"{val_int.item():032b}"
    corrupted_bits = f"{corrupted_int.item():032b}"

    return corrupted_value, rand_bit, original_bits, corrupted_bits


def inject_fault(model, component_type="attention", block_idx=None, idx=None, verbose=True):
    """Inject a single bit-flip fault into the model and optionally print details."""
    available_params = []

    # Randomize component type if "all" is requested
    if component_type == "all":
        component_type = random.choice(
            ["attention", "norm", "mlp", "patch_embed", "classifier"]
        )

    # --- Collect parameters ---
    if component_type == "attention":
        if block_idx is None:
            block_idx = random.randint(0, len(model.blocks) - 1)
        attn = model.blocks[block_idx].attn
        for name, param in attn.named_parameters():
            if name in ["qkv.weight"]:
                available_params.append((f"Block{block_idx}.attn.{name}", param))

    elif component_type == "norm":
        if block_idx is None:
            block_idx = random.randint(0, len(model.blocks) - 1)
        block = model.blocks[block_idx]
        for name, param in block.norm1.named_parameters():
            available_params.append((f"Block{block_idx}.norm1.{name}", param))
        for name, param in block.norm2.named_parameters():
            available_params.append((f"Block{block_idx}.norm2.{name}", param))

    elif component_type == "mlp":
        if block_idx is None:
            block_idx = random.randint(0, len(model.blocks) - 1)
        mlp = model.blocks[block_idx].mlp
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
        raise ValueError(f"No suitable parameters found for component_type: {component_type}")

    # Pick a random parameter and index
    param_full_name, param = random.choice(available_params)
    if idx is None:
        idx = tuple(random.randint(0, s - 1) for s in param.shape)

    original_value = param[idx].clone()
    corrupted_value, bit_flipped, original_bits, corrupted_bits = flip_random_bit(original_value)

    # Apply the corruption
    with torch.no_grad():
        param[idx] = corrupted_value

    fault_info = {
        "component_type": component_type,
        "block_idx": block_idx,
        "param_name": param_full_name,
        "fault_idx": idx,
        "bit_flipped": bit_flipped,
        "original_value": original_value.item(),
        "corrupted_value": corrupted_value.item(),
        "original_bits": original_bits,
        "corrupted_bits": corrupted_bits,
    }

    # --- Print details ---
    if verbose:
        print("\n🧩 Fault Injection Details")
        print("-" * 80)
        print(f" Component Type : {fault_info['component_type']}")
        print(f" Block Index    : {fault_info['block_idx']}")
        print(f" Parameter Name : {fault_info['param_name']}")
        print(f" Fault Index    : {fault_info['fault_idx']}")
        print(f" Bit Flipped    : {fault_info['bit_flipped']}")
        print(f" Original Value : {fault_info['original_value']:.8f}")
        print(f" Corrupted Value: {fault_info['corrupted_value']:.8f}")
        print(f" Original Bits  : {original_bits}")
        print(f" Corrupted Bits : {corrupted_bits}")
        print(f"Bit position legend: [Sign | Exponent | Mantissa]")
        print("-" * 80)

    return fault_info

