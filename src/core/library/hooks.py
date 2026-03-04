"""Forward hook management for activation capture."""

import torch
import torch.nn as nn
from typing import Any, Callable

from src.core.library.layers import extract_block_idx, classify_component


class HookManager:
    def __init__(self):
        self._handles: list = []
        self._max_block_seen: int = -1

    def register(self, model: nn.Module, callback: Callable) -> int:
        """Register forward hooks on all non-root modules.

        Args:
            model: The model to hook
            callback: Called for each activation with signature:
                callback(tensor, name, module_type, component, block_idx) -> None

        Returns:
            Number of modules hooked
        """
        self.remove()
        count = 0

        for name, module in model.named_modules():
            if name == "":
                continue

            block_idx = extract_block_idx(name)

            def make_hook(n: str, b: int | None):
                def hook(mod: nn.Module, inp: Any, out: Any) -> None:
                    if isinstance(out, tuple):
                        out = out[0]
                    if not isinstance(out, torch.Tensor) or out.numel() <= 1:
                        return

                    if b is not None and b > self._max_block_seen:
                        self._max_block_seen = b

                    component = classify_component(n, b, self._max_block_seen)
                    callback(out, n, mod.__class__.__name__, component, b)

                return hook

            handle = module.register_forward_hook(make_hook(name, block_idx))
            self._handles.append(handle)
            count += 1

        return count

    def remove(self) -> None:
        """Remove all registered forward hooks and clear handles."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset_block_tracking(self) -> None:
        """Reset the max_block_seen counter. Call between batches."""
        self._max_block_seen = -1
