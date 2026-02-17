"""Unified data collection for activation and weight analysis."""

import re
import torch
import torch.nn as nn
import numpy as np

from src.core.library.hooks import HookManager
from src.core.library.layers import extract_block_idx, is_excluded
from src.core.library.histogram import sample_and_histogram, get_histogram_config


class DataCollector:
    """Unified collector for activation and weight data.

    Behaves differently based on analysis_type:
    - "aa": Activation analysis (uses forward hooks)
    - "wa": Weight analysis (iterates parameters)
    """

    # Activation-specific exclude patterns
    ACTIVATION_EXCLUDE_PATTERNS: list[str] = [
        # Uncomment to exclude:
        # "attn_drop",    # Attention weights (0-1 probabilities)
        # "drop",         # Dropout layers
    ]

    def __init__(self, analysis_type: str, sampling_percent: float = 1.0):
        """Initialize data collector.

        Args:
            analysis_type: "aa" for activations or "wa" for weights
            sampling_percent: Percentage to sample (for aa only, 0.01-100)
        """
        if analysis_type not in ["aa", "wa"]:
            raise ValueError(
                f"analysis_type must be 'aa' or 'wa', got '{analysis_type}'"
            )

        self.analysis_type = analysis_type
        self.sampling_percent = sampling_percent if analysis_type == "aa" else 100.0

        # Get histogram configuration for this analysis type
        self.bin_range, self.bin_resolution, self.num_bins = get_histogram_config(
            analysis_type
        )

        # Activation-specific state
        if analysis_type == "aa":
            self._hook_manager = HookManager()
            self.total_samples = 0
            self.total_batches = 0

    def classify_parameter(self, name: str) -> str:
        """Classify parameter into component type (for weight analysis).

        Args:
            name: Parameter name from model.named_parameters()

        Returns:
            Component classification
        """
        name_lower = name.lower()

        if (
            "patch_embed" in name_lower
            or "pos_embed" in name_lower
            or "cls_token" in name_lower
        ):
            return "patch_embed"
        elif "head" in name_lower and ("weight" in name_lower or "bias" in name_lower):
            return "classifier"
        elif "attn" in name_lower or "qkv" in name_lower or "proj" in name_lower:
            # Check if it's not a norm layer with "attn" in the name
            if "norm" not in name_lower:
                return "attention"
        elif "mlp" in name_lower or "fc1" in name_lower or "fc2" in name_lower:
            return "mlp"
        elif "norm" in name_lower or "ln" in name_lower:
            return "norm"

        return "other"

    def extract_block_idx_from_param(self, name: str) -> int | None:
        """Extract block index from parameter name.

        Args:
            name: Parameter name

        Returns:
            Block index or None if not in a block
        """
        name_lower = name.lower()

        # Try blocks.X pattern
        match = re.search(r"blocks\.(\d+)", name_lower)
        if match:
            return int(match.group(1))

        # Try layers.X.blocks.Y pattern (Swin)
        match = re.search(r"layers\.(\d+)\.blocks\.(\d+)", name_lower)
        if match:
            return int(match.group(1)) * 100 + int(match.group(2))

        return None

    def record_activation_callback(
        self,
        tensor: torch.Tensor,
        name: str,
        module_type: str,
        component: str,
        block_idx: int | None,
        # State passed from manager
        data_dict: dict,
        name_to_idx: dict,
        global_stats: dict,
        hist_counts: dict,
        data_range: dict,
        element_counts: dict,
    ) -> None:
        """Callback invoked for activation collection during forward pass.

        Args:
            tensor: Output activation tensor
            name: Module name
            module_type: Module type string
            component: Component classification
            block_idx: Transformer block index
            data_dict: Shared layer data dictionary
            name_to_idx: Shared name-to-index mapping
            global_stats: Shared global statistics
            hist_counts: Shared histogram counts
            data_range: Shared data range tracking
            element_counts: Shared element counts
        """
        num_elements = tensor.numel()

        if name not in name_to_idx:
            name_to_idx[name] = len(name_to_idx)
        idx = name_to_idx[name]

        excluded = is_excluded(name, module_type, self.ACTIVATION_EXCLUDE_PATTERNS)

        if idx in data_dict:
            data = data_dict[idx]
            t_min, t_max = tensor.aminmax()
            data["min"] = min(data["min"], t_min.item())
            data["max"] = max(data["max"], t_max.item())
            data["total_elements"] += num_elements
        else:
            t_min, t_max = tensor.aminmax()
            data_dict[idx] = {
                "name": name,
                "op_type": module_type,
                "component": component,
                "block_idx": block_idx,
                "min": t_min.item(),
                "max": t_max.item(),
                "excluded": excluded,
                "total_elements": num_elements,
                "sampled_elements": 0,
            }

        if excluded:
            return

        element_counts[component]["total"] += num_elements

        data = data_dict[idx]
        global_stats[component]["min"] = min(
            global_stats[component]["min"], data["min"]
        )
        global_stats[component]["max"] = max(
            global_stats[component]["max"], data["max"]
        )

        num_sampled, sampled_min, sampled_max = sample_and_histogram(
            tensor,
            self.sampling_percent,
            hist_counts[component],
            self.bin_range,
            self.bin_resolution,
        )

        element_counts[component]["sampled"] += num_sampled
        data_dict[idx]["sampled_elements"] += num_sampled

        data_range[component]["min"] = min(
            data_range[component]["min"], sampled_min
        )
        data_range[component]["max"] = max(
            data_range[component]["max"], sampled_max
        )

    def record_weight(
        self,
        name: str,
        param: torch.Tensor,
        component: str,
        block_idx: int | None,
        # State passed from manager
        data_dict: dict,
        name_to_idx: dict,
        global_stats: dict,
        hist_counts: dict,
        data_range: dict,
        element_counts: dict,
    ) -> None:
        """Record weight values from a parameter.

        Args:
            name: Parameter name
            param: Parameter tensor
            component: Component classification
            block_idx: Block index if applicable
            data_dict: Shared data dictionary
            name_to_idx: Shared name-to-index mapping
            global_stats: Shared global statistics
            hist_counts: Shared histogram counts
            data_range: Shared data range tracking
            element_counts: Shared element counts
        """
        if name not in name_to_idx:
            name_to_idx[name] = len(name_to_idx)
        idx = name_to_idx[name]

        # Get parameter statistics
        num_elements = param.numel()
        param_flat = param.detach().float().cpu().numpy().ravel()
        param_min = float(param_flat.min())
        param_max = float(param_flat.max())

        # Store parameter info
        data_dict[idx] = {
            "name": name,
            "component": component,
            "block_idx": block_idx,
            "min": param_min,
            "max": param_max,
            "total_elements": num_elements,
            "sampled_elements": num_elements,  # All weights are "sampled"
            "shape": list(param.shape),
        }

        # Update global stats for component and "all"
        for comp in [component, "all"]:
            global_stats[comp]["min"] = min(global_stats[comp]["min"], param_min)
            global_stats[comp]["max"] = max(global_stats[comp]["max"], param_max)

            # Update element counts
            element_counts[comp]["total"] += num_elements
            element_counts[comp]["sampled"] += num_elements

            # Update data range
            data_range[comp]["min"] = min(data_range[comp]["min"], param_min)
            data_range[comp]["max"] = max(data_range[comp]["max"], param_max)

            # Update histogram with proper resolution
            bin_indices = (
                np.floor(param_flat / self.bin_resolution).astype(np.int64)
                + int(self.bin_range / self.bin_resolution)
            )
            bin_indices = np.clip(bin_indices, 0, len(hist_counts[comp]) - 1)
            np.add.at(hist_counts[comp], bin_indices, 1)

    def collect(
        self,
        model: nn.Module,
        batches: tuple | None = None,
        inference_fn=None,
        use_amp: bool = False,
        verbose: bool = True,
        # State dictionaries
        data_dict: dict = None,
        name_to_idx: dict = None,
        global_stats: dict = None,
        hist_counts: dict = None,
        data_range: dict = None,
        element_counts: dict = None,
        num_blocks_ref: list = None,
    ) -> None:
        """Collect data based on analysis type.

        For activations (aa):
            - Requires batches, inference_fn
            - Registers forward hooks and processes batches

        For weights (wa):
            - Only requires model
            - Iterates through named_parameters

        Args:
            model: The model to analyze
            batches: Tuple of (images, labels) batches (aa only)
            inference_fn: Callable(images, use_amp) -> outputs (aa only)
            use_amp: Whether to use AMP (aa only)
            verbose: Whether to print progress
            data_dict: Shared data dictionary
            name_to_idx: Shared name-to-index mapping
            global_stats: Shared global statistics
            hist_counts: Shared histogram counts
            data_range: Shared data range tracking
            element_counts: Shared element counts
            num_blocks_ref: Mutable list containing num_blocks
        """
        if self.analysis_type == "aa":
            self._collect_activations(
                model,
                batches,
                inference_fn,
                use_amp,
                verbose,
                data_dict,
                name_to_idx,
                global_stats,
                hist_counts,
                data_range,
                element_counts,
                num_blocks_ref,
            )
        else:
            self._collect_weights(
                model,
                verbose,
                data_dict,
                name_to_idx,
                global_stats,
                hist_counts,
                data_range,
                element_counts,
                num_blocks_ref,
            )

    def _collect_activations(
        self,
        model: nn.Module,
        batches: tuple,
        inference_fn,
        use_amp: bool,
        verbose: bool,
        data_dict: dict,
        name_to_idx: dict,
        global_stats: dict,
        hist_counts: dict,
        data_range: dict,
        element_counts: dict,
        num_blocks_ref: list,
    ) -> None:
        """Collect activation data using forward hooks."""
        self._hook_manager.remove()

        # Count blocks
        for name, _ in model.named_modules():
            idx = extract_block_idx(name)
            if idx is not None:
                num_blocks_ref[0] = max(num_blocks_ref[0], idx + 1)

        # Create closure callback with state
        def _callback(tensor, name, module_type, component, block_idx):
            self.record_activation_callback(
                tensor,
                name,
                module_type,
                component,
                block_idx,
                data_dict,
                name_to_idx,
                global_stats,
                hist_counts,
                data_range,
                element_counts,
            )

        hook_count = self._hook_manager.register(model, _callback)

        pct_str = (
            f"{self.sampling_percent:.2f}%"
            if self.sampling_percent < 1
            else f"{self.sampling_percent:.1f}%"
        )
        if verbose:
            print(f"Registered hooks on {hook_count} modules (sampling {pct_str})")

        # Process batches
        total_batches = len(batches)
        if verbose:
            print(f"Processing {total_batches} batches")

        with torch.inference_mode():
            for batch_idx, (images, _) in enumerate(batches):
                _ = inference_fn(images, use_amp)
                self.total_samples += images.size(0)
                self.total_batches += 1
                self._hook_manager.reset_block_tracking()

                if verbose and (batch_idx + 1) % max(1, total_batches // 10) == 0:
                    progress = 100 * (batch_idx + 1) / total_batches
                    print(
                        f"  Progress: {progress:.0f}% ({batch_idx + 1}/{total_batches} batches)"
                    )

        # Clean up hooks
        self._hook_manager.remove()

    def _collect_weights(
        self,
        model: nn.Module,
        verbose: bool,
        data_dict: dict,
        name_to_idx: dict,
        global_stats: dict,
        hist_counts: dict,
        data_range: dict,
        element_counts: dict,
        num_blocks_ref: list,
    ) -> None:
        """Collect weight data by iterating parameters."""
        # Count blocks
        for name, _ in model.named_modules():
            block_idx = self.extract_block_idx_from_param(name)
            if block_idx is not None:
                num_blocks_ref[0] = max(num_blocks_ref[0], block_idx + 1)

        if verbose:
            print(f"Analyzing weights from {num_blocks_ref[0]} transformer blocks")

        # Iterate through parameters
        param_count = 0
        for name, param in model.named_parameters():
            component = self.classify_parameter(name)
            block_idx = self.extract_block_idx_from_param(name)

            self.record_weight(
                name,
                param,
                component,
                block_idx,
                data_dict,
                name_to_idx,
                global_stats,
                hist_counts,
                data_range,
                element_counts,
            )
            param_count += 1

        if verbose:
            print(f"Processed {param_count} parameter tensors")

    def get_components(self) -> list[str]:
        """Get list of component types based on analysis type."""
        if self.analysis_type == "aa":
            return ["input", "output", "block", "mha", "mlp"]
        else:  # wa
            return ["attention", "mlp", "norm", "patch_embed", "classifier", "all"]

    def cleanup(self) -> None:
        """Clean up resources (hooks for activation analysis)."""
        if self.analysis_type == "aa":
            self._hook_manager.remove()
