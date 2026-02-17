"""Detector neurons via weight matrix extension.

Adds detector rows to Linear layer weight matrices. PyTorch automatically
computes detector outputs during the normal forward pass (matmul). A forward
hook captures these values and strips them from the output so downstream
layers see the original shape.

How it works
------------
For a Linear layer with weight W of shape [out_features, in_features]:

1. Append detector rows to W:
   - sum_row = [1, 1, 1, ..., 1]       -> computes sum(input)
   - avg_row = [1/n, 1/n, ..., 1/n]   -> computes mean(input)

2. New weight W' has shape [out_features + 2, in_features]

3. During forward pass:
   - output' = input @ W'.T  (shape: [batch, seq, out_features + 2])
   - Last 2 elements are detector outputs (computed by matmul, not separately!)

4. Hook extracts detector values and returns stripped output to downstream

5. ALSO captures sum/avg/min of the ORIGINAL output (first out_features elements)
   to detect faults in the layer's own weights

Detection capability
--------------------
- **Input-based detectors** (from weight rows): Detect faults that propagate
  FROM upstream layers (changes in input activations)
- **Output-based detectors** (computed in hook): Detect faults IN this layer's
  weights (changes in output activations)

Both are captured and can be compared against fault-free baselines.
"""

import torch
import torch.nn as nn


class DetectorNeurons:
    """Adds detector rows to weight matrix and captures outputs via hook.

    The detector rows are incorporated into the normal matmul, so their
    outputs are computed by PyTorch automatically. The hook:
    1. Extracts detector values from the end of output tensor
    2. Computes aggregates of the original output (for fault-in-layer detection)
    3. Returns stripped output so downstream sees correct shape

    Attributes:
        target_layer: The monitored nn.Linear layer
        original_weight: Saved copy for restoration
        original_bias: Saved copy for restoration
        original_out_features: Original output dimension
        detection_values: Dict with captured values after each forward pass
        hook_handle: Handle for cleanup
    """

    def __init__(self) -> None:
        self.target_layer: nn.Linear | None = None
        self.original_weight: torch.Tensor | None = None
        self.original_bias: torch.Tensor | None = None
        self.original_out_features: int | None = None
        self.detection_values: dict[str, torch.Tensor] | None = None
        self.hook_handle = None

    def add_to_layer(self, layer: nn.Linear) -> None:
        """Add detector rows to weight matrix and register output hook.

        Modifies the layer's weight tensor by appending 2 detector rows:
        - Row 1: all ones (computes sum of input)
        - Row 2: all 1/n (computes mean of input)

        Args:
            layer: nn.Linear layer to monitor
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(layer)}")

        self.target_layer = layer

        # Save originals for restoration
        self.original_weight = layer.weight.data.clone()
        self.original_out_features = layer.out_features
        if layer.bias is not None:
            self.original_bias = layer.bias.data.clone()
        else:
            self.original_bias = None

        out_features, in_features = layer.weight.shape
        device = layer.weight.device
        dtype = layer.weight.dtype

        # Create detector rows
        # sum_row: dot(input, ones) = sum(input)
        sum_row = torch.ones(1, in_features, device=device, dtype=dtype)
        # avg_row: dot(input, ones/n) = mean(input)
        avg_row = torch.ones(1, in_features, device=device, dtype=dtype) / in_features

        # Append detector rows to weight matrix
        # New shape: [out_features + 2, in_features]
        new_weight = torch.cat([layer.weight.data, sum_row, avg_row], dim=0)
        layer.weight = nn.Parameter(new_weight)
        layer.out_features = out_features + 2

        # Extend bias with zeros for detector rows
        if layer.bias is not None:
            padding = torch.zeros(2, device=device, dtype=dtype)
            new_bias = torch.cat([layer.bias.data, padding])
            layer.bias = nn.Parameter(new_bias)

        # Register hook to capture and strip detector values
        self._register_hook()

    def _register_hook(self) -> None:
        """Register forward hook that captures detector values and strips them."""

        # Store original out_features for slicing
        orig_out = self.original_out_features

        def capture_and_strip_hook(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            """Capture detector values and return stripped output.

            Args:
                module: The layer
                input: Tuple of inputs
                output: Full output including detector values at end

            Returns:
                Output with detector values stripped (original shape)
            """
            # Output shape: [batch, seq, out_features + 2] or [batch, out_features + 2]

            if output.dim() == 3:
                # Transformer: [batch, seq_len, features]
                # Original output (first orig_out features) - affected by weight faults
                orig_output = output[:, :, :orig_out]
                # Detector outputs (last 2 features) - computed from input
                det_sum_input = output[:, :, -2].clone()  # sum(input) via weight row
                det_avg_input = output[:, :, -1].clone()  # avg(input) via weight row

                # Also compute aggregates of ORIGINAL output (detects faults in weights)
                det_sum_output = orig_output.sum(dim=-1).clone()
                det_avg_output = orig_output.mean(dim=-1).clone()
                det_min_output = orig_output.min(dim=-1).values.clone()

            elif output.dim() == 2:
                # Standard FC: [batch, features]
                orig_output = output[:, :orig_out]
                det_sum_input = output[:, -2].clone()
                det_avg_input = output[:, -1].clone()

                det_sum_output = orig_output.sum(dim=-1).clone()
                det_avg_output = orig_output.mean(dim=-1).clone()
                det_min_output = orig_output.min(dim=-1).values.clone()
            else:
                # Fallback for other dimensions
                orig_output = output[..., :orig_out]
                det_sum_input = output[..., -2].clone()
                det_avg_input = output[..., -1].clone()

                det_sum_output = orig_output.sum(dim=-1).clone()
                det_avg_output = orig_output.mean(dim=-1).clone()
                det_min_output = orig_output.min(dim=-1).values.clone()

            # Store all detection values
            self.detection_values = {
                # Input-based (from detector weight rows) - detects upstream faults
                "sum_input": det_sum_input,
                "avg_input": det_avg_input,
                # Output-based (computed from original output) - detects faults in this layer
                "sum": det_sum_output,
                "avg": det_avg_output,
                "min": det_min_output,
            }

            # Return stripped output so downstream sees original shape
            return orig_output

        self.hook_handle = self.target_layer.register_forward_hook(capture_and_strip_hook)

    def get_detection_values(self) -> dict[str, torch.Tensor] | None:
        """Return captured detector values.

        Returns:
            Dict with keys:
            - "sum_input", "avg_input": From detector weight rows (detect upstream faults)
            - "sum", "avg", "min": From original output (detect faults in this layer)
            Returns None if no forward pass has occurred yet.
        """
        return self.detection_values

    def is_active(self) -> bool:
        """True when hook is registered and weights are modified."""
        return self.hook_handle is not None

    def remove(self) -> None:
        """Remove hook and restore original weights exactly."""
        # Remove hook first
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

        # Restore original weights
        if self.target_layer is not None and self.original_weight is not None:
            self.target_layer.weight = nn.Parameter(self.original_weight)
            self.target_layer.out_features = self.original_out_features

            if self.original_bias is not None:
                self.target_layer.bias = nn.Parameter(self.original_bias)

        # Clear all references
        self.target_layer = None
        self.original_weight = None
        self.original_bias = None
        self.original_out_features = None
        self.detection_values = None


# ---------------------------------------------------------------------------
# Layer accessor helpers
# ---------------------------------------------------------------------------


def get_qkv_layer(model, block_idx: int) -> nn.Linear:
    """Get the QKV projection layer from a specific block."""
    from src.core.library.layers import get_block
    block = get_block(model, block_idx)
    return block.attn.qkv


def get_fc1_layer(model, block_idx: int) -> nn.Linear:
    """Get the first MLP layer (fc1) from a specific block."""
    from src.core.library.layers import get_block
    block = get_block(model, block_idx)
    return block.mlp.fc1


def get_proj_layer(model, block_idx: int) -> nn.Linear:
    """Get the attention output projection layer from a specific block."""
    from src.core.library.layers import get_block
    block = get_block(model, block_idx)
    return block.attn.proj


def get_fc2_layer(model, block_idx: int) -> nn.Linear:
    """Get the second MLP layer (fc2) from a specific block."""
    from src.core.library.layers import get_block
    block = get_block(model, block_idx)
    return block.mlp.fc2
