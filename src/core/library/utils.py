"""General-purpose utility functions and bit manipulation."""

import random
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ("true", "1", "yes", "y")


def int_or_none(s: str) -> int | None:
    """Convert string to int or None."""
    if s.lower() == "none":
        return None
    return int(s)


def parse_bit_range(s: str) -> tuple[int, int] | None:
    """Parse bit range string 'START,END' to tuple (for CLI args)."""
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError("bit_range must be START,END")
    return (int(parts[0]), int(parts[1]))


def _parse_bit_spec(bit_spec):
    """Parse bit range specification into (min_bit, max_bit) tuple (internal use).

    Args:
        bit_spec: Can be:
            - int: single bit
            - tuple of (min, max): bit range
            - str "min...max": bit range

    Returns:
        Tuple of (min_bit, max_bit)
    """
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
        min_bit, max_bit = _parse_bit_spec(bit_range)
        rand_bit = random.randint(min_bit, max_bit)

    val_int = value.view(torch.int32)
    mask = torch.tensor(1, dtype=torch.int32, device=value.device) << rand_bit
    corrupted_int = val_int ^ mask

    corrupted_value = corrupted_int.view(torch.float32)
    original_bits = f"{val_int.item():032b}"
    corrupted_bits = f"{corrupted_int.item():032b}"

    return corrupted_value, rand_bit, original_bits, corrupted_bits


def resolve_amp(config) -> bool:
    """Determine if AMP should be used for this model and device.

    Some models (e.g., BEiT) are unstable with FP16, so AMP is disabled for them.
    """
    UNSTABLE_FP16 = ["beit"]
    return (
        config.use_amp
        and config.device.type == "cuda"
        and not any(name in config.model_name.lower() for name in UNSTABLE_FP16)
    )
