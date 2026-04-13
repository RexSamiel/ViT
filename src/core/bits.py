"""Bit manipulation utilities for fault injection."""

import random

import torch


def flip_bit(
    value: torch.Tensor,
    bit: int | None = None,
    bit_range: list[int] | None = None,
) -> tuple[torch.Tensor, int, str, str]:
    """Flip a bit in a float32 value's IEEE 754 representation."""
    if value.dtype != torch.float32:
        value = value.float()

    if bit is None:
        bit = random.choice(bit_range if bit_range is not None else list(range(32)))

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
