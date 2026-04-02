import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from core.bits import flip_bit
from core.layers import filter_layers


@dataclass
class InjectedFault:
    """Record of an injected fault."""

    layer: str
    index: tuple
    bit: int
    original: float
    corrupted: float
    original_tensor: torch.Tensor
    layer_ref: nn.Linear


class WeightPool:
    """Maps a flat index space to (layer, weight_index) for proportional sampling.

    This allows uniform random sampling across all weights in selected layers,
    giving each weight parameter an equal probability of being selected.
    """

    def __init__(self, layers: dict[str, nn.Linear]):
        """Build the weight pool from a dictionary of layers.

        Args:
            layers: Dict mapping layer names to Linear modules
        """
        self.layers = layers
        self.layer_names = list(layers.keys())

        self._cumulative_sizes = []
        self._layer_sizes = []
        total = 0

        for name in self.layer_names:
            layer = layers[name]
            size = layer.weight.numel()
            self._layer_sizes.append(size)
            self._cumulative_sizes.append(total)
            total += size

        self._total_weights = total

    @property
    def total_weights(self) -> int:
        """Total number of weight parameters across all layers."""
        return self._total_weights

    @property
    def total_bits(self) -> int:
        """Total number of bits across all weights (assuming float32)."""
        return self._total_weights * 32

    def flat_to_layer_index(self, flat_idx: int) -> tuple[str, tuple]:
        """Convert a flat index to (layer_name, weight_index).

        Args:
            flat_idx: Index in range [0, total_weights)

        Returns:
            Tuple of (layer_name, index_tuple)
        """
        # Find which layer this index belongs to
        layer_idx = 0
        for i, cum_size in enumerate(self._cumulative_sizes):
            if flat_idx >= cum_size:
                layer_idx = i
            else:
                break
        else:
            if flat_idx >= self._cumulative_sizes[-1]:
                layer_idx = len(self._cumulative_sizes) - 1

        name = self.layer_names[layer_idx]
        layer = self.layers[name]
        local_idx = flat_idx - self._cumulative_sizes[layer_idx]

        # Convert flat local index to tuple index
        weight_shape = layer.weight.shape
        index_tuple = []
        for dim in reversed(weight_shape):
            index_tuple.append(local_idx % dim)
            local_idx //= dim
        index_tuple = tuple(reversed(index_tuple))

        return name, index_tuple

    def sample_indices(
        self, count: int, allow_duplicates: bool = False
    ) -> list[tuple[str, tuple]]:
        """Sample random weight indices.

        Args:
            count: Number of indices to sample
            allow_duplicates: If True, same weight can be selected multiple times

        Returns:
            List of (layer_name, index_tuple) pairs
        """
        if allow_duplicates:
            flat_indices = [
                random.randint(0, self._total_weights - 1) for _ in range(count)
            ]
        else:
            if count > self._total_weights:
                raise ValueError(
                    f"Cannot sample {count} unique indices from {self._total_weights} weights"
                )
            flat_indices = random.sample(range(self._total_weights), count)

        return [self.flat_to_layer_index(idx) for idx in flat_indices]


class Injector:
    """Injects bit-flip faults into linear layer weights with proportional sampling.

    All weights across selected layers are treated as a single pool, ensuring
    each weight has equal probability of being corrupted regardless of layer size.

    Example:
        injector = Injector(model, layers="fc1")

        # Inject by count
        injector.inject(count=5)

        # Or inject by bit error rate
        injector.inject(ber=1e-6)

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
            model: Model instance (with .net attribute) or nn.Module
            layers: Layer filter ("all", "fc1", "fc2", "qkv", "proj", or custom pattern)
            bit_range: Optional (min, max) bit range for flips (0-31 for float32)
        """
        if hasattr(model, "net"):
            self.model = model.net
        else:
            self.model = model

        self.layer_filter = layers
        self.bit_range = bit_range
        self.faults: list[InjectedFault] = []

        self._layers = self._get_layers()
        self._pool = WeightPool(self._layers) if self._layers else None

        # Set by CLI to carry injection parameters across calls
        self.fi_faults: int | None = None
        self.fi_ber: float | None = None

    def _get_layers(self) -> dict[str, nn.Linear]:
        """Get target layers for injection."""
        layers = {}
        for name, module in self.model.named_modules():
            # Handle detection wrappers - check for .original attribute
            if hasattr(module, "original") and isinstance(module.original, nn.Linear):
                layers[name] = module.original
            elif isinstance(module, nn.Linear) and name:
                if ".original" not in name:
                    layers[name] = module
        return filter_layers(layers, self.layer_filter)

    @property
    def total_weights(self) -> int:
        """Total number of weight parameters in selected layers."""
        return self._pool.total_weights if self._pool else 0

    @property
    def total_bits(self) -> int:
        """Total number of bits in selected layer weights."""
        return self._pool.total_bits if self._pool else 0

    @property
    def layer_info(self) -> dict[str, int]:
        """Dictionary of layer names to their weight counts."""
        return {name: layer.weight.numel() for name, layer in self._layers.items()}

    def inject(
        self,
        count: int | None = None,
        ber: float | None = None,
        allow_multi_bit: bool = False,
        allow_same_weight: bool = False,
    ):
        """Inject bit-flip faults into random weights.

        Must specify exactly one of `count` or `ber`.

        Args:
            count: Number of bit flips to inject
            ber: Bit Error Rate - probability per bit (e.g., 1e-6)
            allow_multi_bit: If True with ber mode, allows multiple bits
                            per weight to be flipped independently
            allow_same_weight: If True, the same weight can be selected
                              multiple times (relevant for count mode)

        Raises:
            ValueError: If neither or both count/ber are specified,
                       or if no layers match the filter
        """
        if not self._pool:
            raise ValueError(f"No layers match filter: {self.layer_filter}")

        if (count is None) == (ber is None):
            raise ValueError("Must specify exactly one of 'count' or 'ber'")

        if ber is not None:
            if allow_multi_bit:
                self._inject_ber_multi_bit(ber)
                return
            else:
                # Calculate expected number of faults from BER
                expected_faults = ber * self._pool.total_bits
                count = int(round(expected_faults))
                if count == 0 and ber > 0:
                    count = 1

        if count is not None and count > 0:
            self._inject_count(count, allow_same_weight)

    def _inject_count(self, count: int, allow_duplicates: bool = False):
        """Inject a fixed number of bit flips with proportional sampling."""
        assert self._pool is not None, "No layers available for injection"
        indices = self._pool.sample_indices(count, allow_duplicates)

        for name, idx in indices:
            layer = self._layers[name]
            weight = layer.weight
            original = weight[idx].clone()

            corrupted, bit, _, _ = flip_bit(original, bit_range=self.bit_range)

            with torch.no_grad():
                weight[idx] = corrupted

            self.faults.append(
                InjectedFault(
                    layer=name,
                    index=idx,
                    bit=bit,
                    original=original.item(),
                    corrupted=corrupted.item(),
                    original_tensor=original,
                    layer_ref=layer,
                )
            )

    def _inject_ber_multi_bit(self, ber: float):
        """Inject faults using BER with potential multiple bits per weight.

        Each bit in each weight has independent probability `ber` of being flipped.
        This can result in 0, 1, or multiple bits flipped per weight.
        """
        if self.bit_range is None:
            bit_start, bit_end = 0, 31
        else:
            bit_start, bit_end = self.bit_range

        for name, layer in self._layers.items():
            weight = layer.weight
            flat_weight = weight.view(-1)

            for w_idx in range(flat_weight.numel()):
                for bit in range(bit_start, bit_end + 1):
                    if random.random() < ber:
                        # Convert flat index to tuple index
                        idx_list = []
                        temp_idx = w_idx
                        for dim in reversed(weight.shape):
                            idx_list.append(temp_idx % dim)
                            temp_idx //= dim
                        idx = tuple(reversed(idx_list))

                        original = weight[idx].clone()
                        corrupted, actual_bit, _, _ = flip_bit(original, bit=bit)

                        with torch.no_grad():
                            weight[idx] = corrupted

                        self.faults.append(
                            InjectedFault(
                                layer=name,
                                index=idx,
                                bit=actual_bit,
                                original=original.item(),
                                corrupted=corrupted.item(),
                                original_tensor=original,
                                layer_ref=layer,
                            )
                        )

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
        print(f"Total weights in pool: {self.total_weights:,}")
        print(f"Total bits in pool: {self.total_bits:,}")
        print(f"Effective BER: {len(self.faults) / self.total_bits:.2e}")
        print()

        for i, fault in enumerate(self.faults):
            print(f"[{i + 1}] {fault.layer}")
            print(f"    Index: {fault.index}, Bit: {fault.bit}")
            print(f"    {fault.original:.6e} -> {fault.corrupted:.6e}")

    def print_layer_info(self):
        """Print information about layers in the weight pool."""
        print(f"\nLayer filter: '{self.layer_filter}'")
        print(f"Total layers: {len(self._layers)}")
        print(f"Total weights: {self.total_weights:,}")
        print(f"Total bits: {self.total_bits:,}")
        print()
        for name, count in self.layer_info.items():
            pct = 100 * count / self.total_weights
            print(f"  {name}: {count:,} ({pct:.1f}%)")

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
