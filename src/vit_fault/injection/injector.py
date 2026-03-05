"""Injector - Bit-flip fault injection into model weights."""

import random
import torch
from dataclasses import dataclass

from vit_fault.core.layers import get_linear_layers, filter_layers
from vit_fault.core.bits import flip_random_bit
from vit_fault.detection.checker import NeuroChecker


@dataclass
class Fault:
    """Record of an injected fault."""
    layer: str
    index: tuple
    bit: int
    original: float
    corrupted: float
    original_tensor: torch.Tensor
    layer_ref: torch.nn.Linear


class Injector:
    """Injects bit-flip faults into linear layer weights.

    Example:
        injector = Injector(model, layers="fc1")
        injector.inject(count=1)
        # Run inference...
        injector.print_info()
        injector.restore()
    """

    def __init__(
        self,
        model,
        layers: str = "all",
        bit_range: tuple[int, int] | None = None,
    ):
        """
        Args:
            model: Model instance (vit_fault.Model) or nn.Module
            layers: Layer filter ("all", "fc1", "fc2", "qkv", "proj")
            bit_range: Optional (min, max) bit range for flips
        """
        # Handle both Model wrapper and raw nn.Module
        if hasattr(model, "net"):
            self.model = model.net
        else:
            self.model = model

        self.layer_filter = layers
        self.bit_range = bit_range
        self.faults: list[Fault] = []

    def _get_layers(self) -> dict[str, torch.nn.Linear]:
        """Get target layers, unwrapping NeuroCheckers if present."""
        layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name:
                layers[name] = module
            elif isinstance(module, NeuroChecker):
                layers[name] = module.original
        return filter_layers(layers, self.layer_filter)

    def inject(self, count: int = 1):
        """Inject bit-flip faults into random weights.

        Args:
            count: Number of faults to inject
        """
        layers = self._get_layers()
        if not layers:
            raise ValueError(f"No layers match filter: {self.layer_filter}")

        layer_names = list(layers.keys())

        for _ in range(count):
            name = random.choice(layer_names)
            layer = layers[name]
            weight = layer.weight

            # Random index into weight tensor
            idx = tuple(random.randint(0, s - 1) for s in weight.shape)
            original = weight[idx].clone()

            # Flip random bit
            corrupted, bit, _, _ = flip_random_bit(original, self.bit_range)

            # Apply corruption
            with torch.no_grad():
                weight[idx] = corrupted

            self.faults.append(Fault(
                layer=name,
                index=idx,
                bit=bit,
                original=original.item(),
                corrupted=corrupted.item(),
                original_tensor=original,
                layer_ref=layer,
            ))

    def restore(self):
        """Restore all injected faults to original values."""
        for fault in self.faults:
            with torch.no_grad():
                fault.layer_ref.weight[fault.index] = fault.original_tensor
        self.faults.clear()

    def print_info(self):
        """Print information about injected faults."""
        if not self.faults:
            print("No faults injected")
            return

        print()
        print(f"FAULTS INJECTED: {len(self.faults)}")
        for i, fault in enumerate(self.faults):
            print(f"[{i + 1}] {fault.layer}")
            print(f"    Index: {fault.index}, Bit: {fault.bit}")
            print(f"    {fault.original:.6e} -> {fault.corrupted:.6e}")

    @property
    def count(self) -> int:
        """Number of currently injected faults."""
        return len(self.faults)

    def get_info(self) -> list[dict]:
        """Get fault information as serializable dictionaries."""
        return [
            {
                "layer": f.layer,
                "index": list(f.index),
                "bit": f.bit,
                "original": f.original,
                "corrupted": f.corrupted,
            }
            for f in self.faults
        ]
