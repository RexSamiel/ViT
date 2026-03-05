"""Bit manipulation utilities for fault injection."""

import random
import torch


def flip_random_bit(
    value: torch.Tensor,
    bit_range: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, int, str, str]:
    """Flip a random bit in a float32 value's IEEE 754 representation.

    Args:
        value: Scalar tensor to corrupt
        bit_range: Optional (min, max) bit range, defaults to (0, 31)

    Returns:
        Tuple of (corrupted_value, bit_index, original_bits, corrupted_bits)
    """
    if value.dtype != torch.float32:
        value = value.float()

    if bit_range is None:
        bit = random.randint(0, 31)
    else:
        bit = random.randint(bit_range[0], bit_range[1])

    val_int = value.view(torch.int32)
    mask = torch.tensor(1, dtype=torch.int32, device=value.device) << bit
    corrupted_int = val_int ^ mask
    corrupted_value = corrupted_int.view(torch.float32)

    original_bits = f"{val_int.item():032b}"
    corrupted_bits = f"{corrupted_int.item():032b}"

    return corrupted_value, bit, original_bits, corrupted_bits


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
