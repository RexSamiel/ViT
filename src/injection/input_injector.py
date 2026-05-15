import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from core.bits import flip_bit
from core.layers import filter_layers


@dataclass
class InjectedInputFault:
    layer: str
    position: tuple  # (token_idx, feature_idx) or (feature_idx,) for 2D inputs
    bit: int
    original: float
    corrupted: float


class InputInjector:
    """Injects bit-flip faults into linear layer input activations.

    Fault model: one position (token, feature) is selected per run and corrupted
    for ALL samples across ALL batches in that run — comparable in scope to a
    weight fault. Proportional sampling ensures each activation element has equal
    probability of being selected regardless of layer size.

    Requires layer shapes saved via: python -m cli -m MODEL save --shapes
    """

    def __init__(
        self,
        model,
        layers: str = "all",
        bit_range: list[int] | None = None,
        layer_shapes: dict[str, tuple] | None = None,
    ):
        if hasattr(model, "net"):
            self.model = model.net
        else:
            self.model = model
        self.layer_filter = layers
        self.bit_range = bit_range

        self._layers: dict[str, nn.Linear] = self._get_layers()
        self._layer_shapes: dict[str, tuple] = layer_shapes or {}

        self._layer_names: list[str] = list(self._layers.keys())
        self._layer_sizes: list[int] = [
            self._layers[n].in_features for n in self._layer_names
        ]

        self._cumulative: list[int] = []
        total = 0
        for s in self._layer_sizes:
            self._cumulative.append(total)
            total += s
        self._total = total

        self._hooks: list = []
        self.faults: list[InjectedInputFault] = []

    def _get_layers(self) -> dict[str, nn.Linear]:
        layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name and ".original" not in name:
                layers[name] = module
        return filter_layers(layers, self.layer_filter)

    def _sample_layer_and_feature(self) -> tuple[str, int]:
        """Pick a (layer_name, feature_idx) proportional to in_features."""
        flat = random.randint(0, self._total - 1)
        layer_idx = 0
        for i, cumulative in enumerate(self._cumulative):
            if flat >= cumulative:
                layer_idx = i
        name = self._layer_names[layer_idx]
        feature_idx = flat - self._cumulative[layer_idx]
        return name, feature_idx

    def arm(self, bit_range: list[int] | None = None):
        """Pre-select injection coordinates and register a persistent hook for this run.

        The hook corrupts input[:, token_idx, feature_idx] for every sample in
        every batch — same position, same bit, all samples.
        """
        effective_bit_range = bit_range or self.bit_range

        layer_name, feature_idx = self._sample_layer_and_feature()
        shape = self._layer_shapes.get(layer_name)

        if shape and len(shape) == 2:
            token_idx = random.randint(0, shape[0] - 1)
            position = (token_idx, feature_idx)
        else:
            token_idx = None
            position = (feature_idx,)

        layer = self._layers[layer_name]
        fault_recorded = [False]

        def hook(module, input):
            tensor = input[0]
            if token_idx is not None:
                col = tensor[:, token_idx, feature_idx]
            else:
                col = tensor[:, feature_idx]

            corrupted_val, bit, _, _ = flip_bit(col[0], bit_range=effective_bit_range)
            original_val = col[0].item()

            if token_idx is not None:
                tensor[:, token_idx, feature_idx] = corrupted_val
            else:
                tensor[:, feature_idx] = corrupted_val

            if not fault_recorded[0]:
                self.faults.append(
                    InjectedInputFault(
                        layer=layer_name,
                        position=position,
                        bit=bit,
                        original=original_val,
                        corrupted=corrupted_val.item(),
                    )
                )
                fault_recorded[0] = True

        handle = layer.register_forward_pre_hook(hook)
        self._hooks.append(handle)

    def restore(self):
        """Remove hooks and clear fault records for the next run."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.faults.clear()

    def get_info(self) -> list[dict]:
        return [
            {
                "layer": f.layer,
                "position": list(f.position),
                "bit": f.bit,
                "original": f.original,
                "corrupted": f.corrupted,
            }
            for f in self.faults
        ]

    def print_info(self):
        if not self.faults:
            print("No input faults injected")
            return
        print(f"\nINPUT FAULTS INJECTED: {len(self.faults)}")
        for f in self.faults:
            print(
                f"  {f.layer}  pos={f.position}  bit={f.bit}"
                f"  {f.original:.6e} → {f.corrupted:.6e}"
            )

    @property
    def total_elements(self) -> int:
        return self._total
