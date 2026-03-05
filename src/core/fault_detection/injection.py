"""Fault injection for detection experiments."""

import random
import torch
import torch.nn as nn

from src.core.library.utils import flip_random_bit
from src.core.fault_detection.neuron import NeuroChecker


def get_linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Find linear layers, including inside checker wrappers."""
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name:
            layers[name] = module
        elif isinstance(module, NeuroChecker):
            layers[name] = module.original
    return layers


class Injector:
    """Injects faults into linear layers."""

    def __init__(self):
        self.faults = []

    def inject(
        self,
        model: nn.Module,
        layer_filter: str = "all",
        bit_range=None,
        count: int = 1,
    ):
        """Inject bit-flip faults into random layers.

        Args:
            model: The model to inject faults into
            layer_filter: Filter layers by name
            bit_range: Tuple (start, end) for bit positions
            count: Number of faults to inject
        """
        layers = get_linear_layers(model)

        if layer_filter != "all":
            layers = {n: l for n, l in layers.items() if layer_filter in n}

        if not layers:
            raise ValueError(f"No layers match: {layer_filter}")

        layer_names = list(layers.keys())

        for _ in range(count):
            name = random.choice(layer_names)
            layer = layers[name]
            weight = layer.weight
            idx = tuple(random.randint(0, s - 1) for s in weight.shape)

            original = weight[idx].clone()
            corrupted, bit, _, _ = flip_random_bit(original, bit_range)

            with torch.no_grad():
                weight[idx] = corrupted

            self.faults.append(
                {
                    "layer": name,
                    "layer_ref": layer,
                    "idx": idx,
                    "bit": bit,
                    "original": original.item(),
                    "corrupted": corrupted.item(),
                    "original_tensor": original,
                }
            )

    def restore(self):
        """Restore all faults."""
        for fault in self.faults:
            with torch.no_grad():
                fault["layer_ref"].weight[fault["idx"]] = fault["original_tensor"]
        self.faults.clear()

    def print_info(self):
        """Print fault info."""
        if not self.faults:
            print("No faults injected")
            return

        print()
        print(f"FAULTS INJECTED: {len(self.faults)}")

        for i, fault in enumerate(self.faults):
            print(f"[{i + 1}] {fault['layer']}")
            print(f"    Index: {fault['idx']}, Bit: {fault['bit']}")
            print(f"    {fault['original']:.6e} -> {fault['corrupted']:.6e}")
