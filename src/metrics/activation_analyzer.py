import torch
import torch.nn as nn
import numpy as np
import re
from typing import Any


# =============================================================================
# EXCLUSION PATTERNS - Add patterns here to exclude layers from JSON output
# =============================================================================
# Layers matching these patterns will be:
#   - Shown in terminal with "[EXCLUDED]" marker
#   - NOT saved to the JSON output file
#   - NOT included in distributions/histograms
#
# Patterns are matched against the full layer name (case-insensitive)
# Examples:
#   "attn_drop"     - excludes layers containing "attn_drop" (attention weights)
#   "softmax"       - excludes softmax layers
#   ".q_norm"       - excludes layers ending with ".q_norm"
#
EXCLUDE_PATTERNS: list[str] = [
    # Add your exclusion patterns here, e.g.:
    # "attn_drop",
    # "softmax",
    # ".q_norm",
    # ".k_norm",
]
# =============================================================================


class ActivationAnalyzer:
    """Captures activation statistics using simple module iteration.

    Instead of complex FX tracing, this uses forward hooks on all modules
    found by recursively iterating through model.named_modules().
    """

    def __init__(self, num_bins: int = 1000, sample_size: int = 100000, deduplicate: bool = False):
        self.num_bins = num_bins
        self.sample_size = sample_size
        self.deduplicate = deduplicate  # Skip modules that output same tensor as children (disabled by default)
        self.reset()

    def reset(self) -> None:
        self.layer_data: dict[int, dict] = {}
        self.block_aggregated: dict[int, dict] = {}

        self.global_stats = {
            "input": {"min": float("inf"), "max": float("-inf")},
            "output": {"min": float("inf"), "max": float("-inf")},
            "block": {"min": float("inf"), "max": float("-inf")},
            "mha": {"min": float("inf"), "max": float("-inf")},
            "mlp": {"min": float("inf"), "max": float("-inf")},
        }

        self._samples = {"input": [], "output": [], "block": [], "mha": [], "mlp": []}
        self._counts = {"input": 0, "output": 0, "block": 0, "mha": 0, "mlp": 0}
        self._hooks: list = []
        self._layer_idx = 0
        self._rng = np.random.default_rng(42)
        self._max_block_seen: int = -1
        self.total_samples = 0
        self.total_batches = 0
        self.num_blocks = 0
        self._module_order: dict[str, int] = {}
        self._seen_storages: set = set()  # For deduplication within a batch

    def _reservoir_sample(self, new_values: np.ndarray, component: str) -> None:
        """Vectorized reservoir sampling for efficiency."""
        samples = self._samples[component]
        count = self._counts[component]
        n_new = len(new_values)

        if n_new == 0:
            return

        # Phase 1: Fill up to sample_size
        if len(samples) < self.sample_size:
            space_left = self.sample_size - len(samples)
            to_add = min(space_left, n_new)
            samples.extend(new_values[:to_add].tolist())
            count += to_add
            new_values = new_values[to_add:]
            n_new = len(new_values)

        # Phase 2: Reservoir sampling for remaining values (vectorized)
        if n_new > 0:
            # Generate all random indices at once
            indices = self._rng.integers(count + 1, count + n_new + 1, size=n_new)
            # Find which values should replace existing samples
            mask = indices < self.sample_size
            replace_indices = indices[mask]
            replace_values = new_values[mask]
            # Apply replacements
            for idx, val in zip(replace_indices, replace_values):
                samples[idx] = float(val)
            count += n_new

        self._counts[component] = count

    def _extract_block_idx(self, name: str) -> int | None:
        """Extract a unique block index from module name.

        For ViT: blocks.0, blocks.1, etc. → simple index
        For Swin: layers.X.blocks.Y → compound index (stage * 100 + block)
                  to ensure uniqueness across stages
        """
        name_lower = name.lower()

        # Check for Swin-style nested structure: layers.X.blocks.Y
        swin_match = re.search(r"layers\.(\d+)\.blocks\.(\d+)", name_lower)
        if swin_match:
            stage_idx = int(swin_match.group(1))
            block_idx = int(swin_match.group(2))
            # Create compound index: stage * 100 + block
            return stage_idx * 100 + block_idx

        # Check for Swin stage-level wrapper: exactly "layers.X" (SwinTransformerStage)
        # Use stage index * 100 + 99 to place after all blocks in that stage
        if re.fullmatch(r"layers\.\d+", name_lower):
            stage_idx = int(name_lower.split(".")[1])
            return stage_idx * 100 + 99  # Stage-level wrapper

        # Check for Swin stage sub-components: layers.X.downsample, layers.X.blocks (Sequential)
        # These are part of a stage but not block-specific
        stage_sub_match = re.search(r"layers\.(\d+)\.(downsample|blocks)(?:\.|$)", name_lower)
        if stage_sub_match and ".blocks." not in name_lower:
            stage_idx = int(stage_sub_match.group(1))
            # downsample components get stage * 100 + 98 (before the stage wrapper)
            return stage_idx * 100 + 98

        # Simple patterns for ViT and other architectures
        patterns = [
            r"blocks_(\d+)",
            r"blocks\.(\d+)",
            r"layer_(\d+)",
            r"encoder_(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, name_lower)
            if match:
                return int(match.group(1))
        return None

    def _classify_component(
        self, name: str, module: nn.Module, block_idx: int | None
    ) -> str:
        """Classify module into component type based on name.

        Simple rule:
        - If "attn" in name → MHA
        - If "mlp" in name → MLP
        - Otherwise → Block (includes norm1, norm2, drop_path, etc.)
        """
        name_lower = name.lower()

        # Before first block = input processing
        if block_idx is None and self._max_block_seen < 0:
            return "input"
        # After blocks = output processing
        if block_idx is None and self._max_block_seen >= 0:
            return "output"

        # Simple classification: only "attn" for MHA, only "mlp" for MLP
        if "attn" in name_lower:
            return "mha"
        if "mlp" in name_lower:
            return "mlp"

        # Everything else in a block is "block" (norm1, norm2, drop_path, etc.)
        return "block"

    def _is_leaf_module(self, module: nn.Module) -> bool:
        """Check if module is a leaf (has no children with parameters)."""
        children = list(module.children())
        return len(children) == 0

    def _should_capture(self, name: str, module: nn.Module) -> bool:
        """Determine if this module should have its activations captured.

        We capture ALL modules except the root module (empty name).
        This matches the simple counting approach in test/layers.py.
        """
        # Skip only the root module (empty name)
        if name == "":
            return False

        # Capture all other modules
        return True

    def _record_activation(
        self,
        tensor: torch.Tensor,
        name: str,
        module_type: str,
        component: str,
        block_idx: int | None,
    ) -> None:
        # Compute min/max on GPU (fast, only transfers 2 scalars)
        val_min = tensor.min().item()
        val_max = tensor.max().item()

        # Check for duplicate tensor (same storage = same data)
        is_duplicate = False
        if self.deduplicate:
            storage_id = tensor.untyped_storage().data_ptr()
            if storage_id in self._seen_storages:
                is_duplicate = True
            else:
                self._seen_storages.add(storage_id)

        idx = self._layer_idx
        self._layer_idx += 1

        # Check if this layer should be excluded based on EXCLUDE_PATTERNS
        name_lower = name.lower()
        module_type_lower = module_type.lower()
        excluded = False

        for pattern in EXCLUDE_PATTERNS:
            pattern_lower = pattern.lower()
            if pattern_lower in name_lower or pattern_lower in module_type_lower:
                excluded = True
                break

        self.layer_data[idx] = {
            "min": val_min,
            "max": val_max,
            "component": component,
            "name": name,
            "op_type": module_type,
            "block_idx": block_idx,
            "shape": list(tensor.shape),
            "is_duplicate": is_duplicate,
            "excluded": excluded,
        }

        # Skip from distribution if excluded or duplicate
        if excluded or is_duplicate:
            return

        self.global_stats[component]["min"] = min(
            self.global_stats[component]["min"], val_min
        )
        self.global_stats[component]["max"] = max(
            self.global_stats[component]["max"], val_max
        )

        if block_idx is not None and component in ("mha", "mlp", "block"):
            if block_idx not in self.block_aggregated:
                self.block_aggregated[block_idx] = {
                    "mha": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
                    "mlp": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
                    "block": {"min": float("inf"), "max": float("-inf"), "last_layer_idx": -1},
                }
            self.block_aggregated[block_idx][component]["min"] = min(
                self.block_aggregated[block_idx][component]["min"], val_min
            )
            self.block_aggregated[block_idx][component]["max"] = max(
                self.block_aggregated[block_idx][component]["max"], val_max
            )
            # Track last layer index for this component in this block
            self.block_aggregated[block_idx][component]["last_layer_idx"] = max(
                self.block_aggregated[block_idx][component]["last_layer_idx"], idx
            )

        # Sample ON GPU first, then transfer only the sample (much faster!)
        flat = tensor.detach().flatten()
        num_elements = flat.numel()
        sample_size = min(10000, num_elements)

        if num_elements > sample_size:
            # Random sampling on GPU using randint (faster than randperm for large tensors)
            indices = torch.randint(0, num_elements, (sample_size,), device=flat.device)
            flat = flat[indices]

        # Only transfer the small sample to CPU (non_blocking for async transfer)
        flat_np = flat.float().cpu().numpy()
        self._reservoir_sample(flat_np, component)

    def register_hooks(self, model: nn.Module) -> int:
        """Register forward hooks on all modules found by iterating recursively."""
        self.remove_hooks()
        self.reset()

        # Disable fused attention to expose intermediate operations
        fused_count = 0
        for module in model.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = False
                fused_count += 1
            if hasattr(module, "use_sdpa"):
                module.use_sdpa = False
                fused_count += 1
        if fused_count > 0:
            print(f"Disabled fused attention on {fused_count} modules")

        # Count blocks
        block_indices = set()
        for name, _ in model.named_modules():
            idx = self._extract_block_idx(name)
            if idx is not None:
                block_indices.add(idx)
        self.num_blocks = len(block_indices)

        # Build module order for deterministic processing
        module_list = []
        for name, module in model.named_modules():
            if self._should_capture(name, module):
                module_list.append((name, module))

        # Store order for later reference
        for order_idx, (name, _) in enumerate(module_list):
            self._module_order[name] = order_idx

        # Register hooks on all captured modules
        count = 0
        for name, module in module_list:
            block_idx = self._extract_block_idx(name)

            def make_hook(n: str, b: int | None):
                def hook(mod: nn.Module, inp: Any, out: Any) -> None:
                    # Handle tuple outputs (some modules return tuples)
                    if isinstance(out, tuple):
                        out = out[0]

                    if not isinstance(out, torch.Tensor):
                        return

                    if out.numel() <= 1:
                        return

                    # Update max block seen for component classification
                    if b is not None and b > self._max_block_seen:
                        self._max_block_seen = b

                    component = self._classify_component(n, mod, b)
                    module_type = mod.__class__.__name__
                    self._record_activation(out, n, module_type, component, b)

                return hook

            handle = module.register_forward_hook(make_hook(name, block_idx))
            self._hooks.append(handle)
            count += 1

        print(
            f"Registered hooks on {count} modules ({self.num_blocks} transformer blocks)"
        )
        return count

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def run_inference(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference - just a regular forward pass since we use hooks."""
        self._layer_idx = 0
        self._max_block_seen = -1
        self._seen_storages.clear()  # Reset deduplication for new batch
        return model(inputs)

    def update(self, batch_size: int) -> None:
        self.total_samples += batch_size
        self.total_batches += 1

    def _compute_histogram(self, component: str) -> dict:
        samples = self._samples[component]
        stats = self.global_stats[component]
        count = self._counts[component]

        if not samples or stats["min"] == float("inf"):
            return {
                "bin_edges": [],
                "counts": [],
                "bin_centers": [],
                "total_values_seen": 0,
            }

        values = np.array(samples)
        val_min, val_max = stats["min"], stats["max"]
        eps = (val_max - val_min) * 1e-6 if val_max > val_min else 1e-6

        counts, edges = np.histogram(
            values, bins=self.num_bins, range=(val_min - eps, val_max + eps)
        )
        centers = (edges[:-1] + edges[1:]) / 2
        scale = count / len(samples) if samples else 1

        return {
            "bin_edges": edges.tolist(),
            "counts": (counts * scale).tolist(),
            "counts_raw": counts.tolist(),
            "bin_centers": centers.tolist(),
            "total_values_seen": count,
            "samples_used": len(samples),
        }

    def _compute_stage_envelopes(self) -> None:
        """For Swin-style models, compute stage envelopes.

        Stage wrappers (block_idx ending in 99) should have min/max that
        encompasses all blocks within that stage.
        """
        # Find all stage indices (those ending in 99)
        stage_indices = [idx for idx in self.block_aggregated.keys() if idx % 100 == 99]

        for stage_idx in stage_indices:
            stage_base = (stage_idx // 100) * 100  # e.g., 99 -> 0, 199 -> 100, 299 -> 200

            # Find all blocks in this stage (indices from stage_base to stage_base + 97)
            blocks_in_stage = [
                idx for idx in self.block_aggregated.keys()
                if stage_base <= idx < stage_base + 98
            ]

            if not blocks_in_stage:
                continue

            # Compute envelope for each component
            for comp in ["mha", "mlp", "block"]:
                comp_min = float("inf")
                comp_max = float("-inf")
                max_layer_idx = -1

                for block_idx in blocks_in_stage:
                    if comp in self.block_aggregated[block_idx]:
                        block_data = self.block_aggregated[block_idx][comp]
                        comp_min = min(comp_min, block_data["min"])
                        comp_max = max(comp_max, block_data["max"])
                        max_layer_idx = max(max_layer_idx, block_data.get("last_layer_idx", -1))

                # Update stage envelope with aggregated values
                if comp_min != float("inf"):
                    if comp not in self.block_aggregated[stage_idx]:
                        self.block_aggregated[stage_idx][comp] = {
                            "min": float("inf"),
                            "max": float("-inf"),
                            "last_layer_idx": -1,
                        }
                    # Envelope includes both the stage's own values AND all inner blocks
                    self.block_aggregated[stage_idx][comp]["min"] = min(
                        self.block_aggregated[stage_idx][comp]["min"], comp_min
                    )
                    self.block_aggregated[stage_idx][comp]["max"] = max(
                        self.block_aggregated[stage_idx][comp]["max"], comp_max
                    )
                    # Use the stage's own last_layer_idx (which should be after all inner blocks)

    def get_results(self) -> dict:
        # Compute stage envelopes for Swin-style models
        self._compute_stage_envelopes()

        layers_by_comp = {
            "input": {},
            "output": {},
            "block": {},
            "mha": {},
            "mlp": {},
        }
        # Only include non-excluded layers in output
        layers_for_json = {}
        for idx, data in self.layer_data.items():
            # Skip excluded layers from JSON output
            if data.get("excluded"):
                continue

            comp = data["component"]
            layers_for_json[idx] = data

            if comp in layers_by_comp:
                layers_by_comp[comp][idx] = {
                    "min": data["min"],
                    "max": data["max"],
                    "name": data["name"],
                    "op_type": data.get("op_type", ""),
                    "block_idx": data["block_idx"],
                }

        block_agg = {}
        for block_idx, comps in self.block_aggregated.items():
            block_agg[str(block_idx)] = {
                c: {
                    "min": v["min"],
                    "max": v["max"],
                    "last_layer_idx": v.get("last_layer_idx", -1),
                }
                for c, v in comps.items()
                if v["min"] != float("inf")
            }

        return {
            "layers": {str(k): v for k, v in layers_for_json.items()},
            "block_aggregated": block_agg,
            "ranges": {
                c: {
                    "layers": {str(k): v for k, v in layers_by_comp[c].items()},
                    "global_min": self.global_stats[c]["min"]
                    if layers_by_comp[c]
                    else None,
                    "global_max": self.global_stats[c]["max"]
                    if layers_by_comp[c]
                    else None,
                }
                for c in ["input", "output", "block", "mha", "mlp"]
            },
            "distributions": {
                c: self._compute_histogram(c)
                for c in ["input", "output", "block", "mha", "mlp"]
            },
            "statistics": {
                "total_samples": self.total_samples,
                "total_batches": self.total_batches,
                "total_layers": len(self.layer_data),
                "num_blocks": self.num_blocks,
                "num_input_layers": len(layers_by_comp["input"]),
                "num_output_layers": len(layers_by_comp["output"]),
                "num_block_layers": len(layers_by_comp["block"]),
                "num_mha_layers": len(layers_by_comp["mha"]),
                "num_mlp_layers": len(layers_by_comp["mlp"]),
                "capture_method": "module_hooks",
            },
        }

    def print_results(self) -> None:
        print("\nActivation Analysis Results:")
        print("-" * 50)
        counts = {"input": 0, "output": 0, "block": 0, "mha": 0, "mlp": 0}
        excluded_counts = 0

        for d in self.layer_data.values():
            comp = d["component"]
            if comp in counts:
                counts[comp] += 1
            if d.get("excluded"):
                excluded_counts += 1

        total = sum(counts.values())

        print(f"\nCapture: module hooks (simple iteration)")
        print(f"Total: {len(self.layer_data)} layers ({self.num_blocks} blocks)")
        print(f"  In distributions: {total - excluded_counts}")
        if excluded_counts > 0:
            print(f"  Excluded (bounded/norm): {excluded_counts}")
        print(f"\n  Input: {counts['input']}")
        print(f"  Output: {counts['output']}")
        print(
            f"  Block: {counts['block']} ({counts['block'] / max(1, self.num_blocks):.1f}/block)"
        )
        print(
            f"  MHA: {counts['mha']} ({counts['mha'] / max(1, self.num_blocks):.1f}/block)"
        )
        print(
            f"  MLP: {counts['mlp']} ({counts['mlp'] / max(1, self.num_blocks):.1f}/block)"
        )

        for c in ["input", "mha", "mlp", "block", "output"]:
            if counts[c] > 0:
                s = self.global_stats[c]
                print(f"\n{c.upper()}: [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"\nSamples: {self.total_samples}")

    def print_layer_ranges(self) -> None:
        print("\nPer-Layer Activation Ranges:")
        print("=" * 130)
        print(
            f"{'Idx':<5} {'Blk':<4} {'Comp':<6} {'Min':>12} {'Max':>12} {'Type':<18} {'Name':<45} {'Status'}"
        )
        print("-" * 130)
        for idx in sorted(self.layer_data.keys()):
            d = self.layer_data[idx]
            name = d["name"]
            op = d.get("op_type", "")[:17]
            if len(name) > 44:
                name = "..." + name[-41:]
            blk = str(d["block_idx"]) if d["block_idx"] is not None else "-"
            status = "[EXCLUDED]" if d.get("excluded") else ""
            print(
                f"{idx:<5} {blk:<4} {d['component']:<6} {d['min']:>12.4f} {d['max']:>12.4f} {op:<18} {name:<45} {status}"
            )
