"""Fault injection - Injector class with all injection-related methods."""

import random
import torch
from typing import Any

from src.core.library.layers import (
    get_num_blocks,
    get_block,
    collect_attention_params,
    collect_mlp_params,
    collect_norm_params,
    collect_patch_embed_params,
    collect_classifier_params,
    ATTENTION_PARAMS,
    MLP_PARAMS,
)
from src.core.library.utils import flip_random_bit


class Injector:
    """Handles fault injection, restoration, and target parameter collection.

    Encapsulates the full inject/restore lifecycle and parameter targeting logic.
    The manager delegates all injection operations to this class.

    All injection helper functions are methods of this class:
    - random_param_index, inject_at_param, format_fault_info
    - collect_target_params, inject, restore
    """

    def __init__(self):
        self.last_fault_info: dict | None = None

    @staticmethod
    def random_param_index(param: torch.Tensor) -> tuple:
        """Generate a random index into a parameter tensor.

        Args:
            param: Parameter tensor.

        Returns:
            Tuple of indices, one per dimension.
        """
        return tuple(random.randint(0, s - 1) for s in param.shape)

    @staticmethod
    def inject_at_param(
        param: torch.Tensor, idx: tuple, corrupted_value: torch.Tensor
    ) -> torch.Tensor:
        """Write a corrupted value into a parameter tensor at the given index.

        Args:
            param: Parameter tensor to corrupt
            idx: Index tuple into param
            corrupted_value: The corrupted value to write

        Returns:
            The original value (clone) for later restoration
        """
        original = param[idx].clone()
        with torch.no_grad():
            param[idx] = corrupted_value
        return original

    @staticmethod
    def format_fault_info(info: dict) -> str:
        """Format fault injection info for display.

        Args:
            info: Fault info dictionary from Injector.inject()

        Returns:
            Formatted multi-line string
        """
        sep = "=" * 60
        sub = (
            f"Sub-component:  {info['sub_component']}\n"
            if info.get("sub_component")
            else ""
        )
        return (
            f"{sep}\n"
            f"FAULT INJECTION\n"
            f"{sep}\n"
            f"Component:      {info['component_type']}\n"
            f"{sub}"
            f"Block:          {info['block_idx']}\n"
            f"Parameter:      {info['param_name']}\n"
            f"Index:          {info['fault_idx']}\n"
            f"Bit flipped:    {info['bit_flipped']}\n"
            f"Original:       {info['original_value']:.6e}\n"
            f"Corrupted:      {info['corrupted_value']:.6e}\n"
            f"{sep}"
        )

    def collect_target_params(
        self,
        model,
        component_type: str = "attention",
        sub_component: str | None = None,
        block_idx: int | None = None,
    ) -> tuple[list[tuple[str, Any]], str, str | None]:
        """Find and collect parameters to target for fault injection.

        Resolves "all" to a random component type and None block_idx to a random
        block. Uses general parameter collection functions from library/layers.py.

        Args:
            model: The neural network model
            component_type: "attention", "mlp", "norm", "patch_embed", "classifier", or "all"
            sub_component: Optional sub-component (e.g. "qkv", "fc1")
            block_idx: Block index (None for random)

        Returns:
            Tuple of (params_list, resolved_component_type, resolved_sub_component)
            where params_list is [(param_name, param_tensor), ...]
        """
        total_blocks = get_num_blocks(model)

        if block_idx is None:
            block_idx = random.randint(0, total_blocks - 1)
        elif block_idx >= total_blocks:
            raise ValueError(
                f"block_idx {block_idx} out of range "
                f"(model has {total_blocks} blocks, indices 0-{total_blocks - 1})"
            )

        if component_type == "all":
            component_type = random.choice(
                ["attention", "mlp", "norm", "patch_embed", "classifier"]
            )

        block = get_block(model, block_idx)

        if component_type == "attention":
            if sub_component is None:
                sub_component = random.choice(list(ATTENTION_PARAMS.keys()))
            params, sub_component = collect_attention_params(
                block.attn, sub_component, block_idx
            )
        elif component_type == "mlp":
            if sub_component is None:
                sub_component = random.choice(list(MLP_PARAMS.keys()))
            params, sub_component = collect_mlp_params(
                block.mlp, sub_component, block_idx
            )
        elif component_type == "norm":
            params = collect_norm_params(block, block_idx)
        elif component_type == "patch_embed":
            params = collect_patch_embed_params(model)
        elif component_type == "classifier":
            params = collect_classifier_params(model)
        else:
            raise ValueError(f"Unknown component_type: {component_type}")

        if not params:
            raise ValueError(
                f"No suitable params for component_type: {component_type}, "
                f"sub_component: {sub_component}"
            )

        return params, component_type, sub_component

    def inject(self, model, fault_params: dict) -> dict:
        """Inject a single-bit fault into the model.

        Selects a target parameter based on fault_params, flips a random bit,
        and stores fault info for later restoration.

        Args:
            model: The neural network model.
            fault_params: Dict with component, sub_component, block_idx, idx,
                bit_range keys.

        Returns:
            fault_info dictionary with all fault details.
        """
        params, comp, sub = self.collect_target_params(
            model,
            component_type=fault_params.get("component", "all"),
            sub_component=fault_params.get("sub_component"),
            block_idx=fault_params.get("block_idx"),
        )
        param_name, param = random.choice(params)

        idx = fault_params.get("idx")
        if idx is None:
            idx = self.random_param_index(param)

        bit_range = fault_params.get("bit_range")
        original_value = param[idx].clone()
        corrupted_value, bit_flipped, original_bits, corrupted_bits = flip_random_bit(
            original_value, bit_range
        )
        original_tensor = self.inject_at_param(param, idx, corrupted_value)

        fault_info = {
            "component_type": comp,
            "sub_component": sub if comp in ["attention", "mlp"] else None,
            "block_idx": fault_params.get("block_idx"),
            "param_name": param_name,
            "param_ref": param,
            "fault_idx": idx,
            "bit_range": bit_range,
            "bit_flipped": bit_flipped,
            "original_value": original_value.item(),
            "original_tensor": original_tensor,
            "corrupted_value": corrupted_value.item(),
            "original_bits": original_bits,
            "corrupted_bits": corrupted_bits,
        }
        self.last_fault_info = fault_info
        return fault_info

    def restore(self) -> None:
        """Restore the last injected fault to its original value."""
        if self.last_fault_info is not None:
            info = self.last_fault_info
            with torch.no_grad():
                info["param_ref"][info["fault_idx"]] = info["original_tensor"]
            self.last_fault_info = None
