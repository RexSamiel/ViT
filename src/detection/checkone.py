from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import DETECTION_DIR
from core.layers import unwrap_layers, wrap_layers


@dataclass
class DetectedFault:
    """Fault detected by checksum comparison."""

    layer: str
    batch: int
    feature: int
    diff: float
    fault_type: str = "weight"


class _Wrapper(nn.Module):
    """ABFT row-sum checksum wrapper for nn.Linear."""

    atol = 1e-5

    def __init__(self, original: nn.Linear, name: str):
        super().__init__()
        self.original = original
        self.name = name

        self.weight_sums = original.weight.data.sum(dim=1).clone()
        self.w_col_sums = original.weight.data.sum(dim=0).clone()

        self.clean_weights: torch.Tensor | None = None
        self.clean_bias: torch.Tensor | None = None

        self.input_mean: float | None = None
        self.input_min: float | None = None
        self.input_max: float | None = None
        self.output_mean: torch.Tensor | None = None

        self.weight_faults: list[tuple[int, int, float]] = []
        self.input_faults: list[tuple[int, int, float]] = []
        self.correction: str | None = None

        self._calibrating = False
        self._cal_min = float("inf")
        self._cal_max = float("-inf")
        self._cal_sum = 0.0
        self._cal_count = 0
        self._cal_out_sum: torch.Tensor | None = None
        self._cal_out_count = 0
        self._cal_weight_check: torch.Tensor | None = None

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        extra_shape: tuple | None = None
        if x.dim() > 3:
            extra_shape = x.shape[1:-1]
            x = x.flatten(1, -2)

        B, N, C = x.shape

        ones_col = torch.ones(1, C, device=x.device, dtype=x.dtype)
        weights_ext = torch.cat([self.original.weight, ones_col], dim=0)

        ones_row = torch.ones(B, 1, C, device=x.device, dtype=x.dtype)
        x_ext = torch.cat([x, ones_row], dim=1)

        out_ext = F.linear(x_ext, weights_ext)

        weight_check = out_ext[:, -1, :-1]
        input_check = out_ext[:, :-1, -1]

        out = out_ext[:, :-1, :-1]
        if self.original.bias is not None:
            out = out + self.original.bias

        if self._calibrating:
            if self._cal_weight_check is None:
                self._cal_weight_check = weight_check.detach()[0].cpu().clone()
            self._cal_min = min(self._cal_min, input_check.min().item())
            self._cal_max = max(self._cal_max, input_check.max().item())
            self._cal_sum += input_check.sum().item()
            self._cal_count += input_check.numel()
            out_sum = out.detach().sum(dim=(0, 1))
            if self._cal_out_sum is None:
                self._cal_out_sum = out_sum
            else:
                self._cal_out_sum += out_sum
            self._cal_out_count += B * N

        self.weight_faults = self._detect_weights(weight_check)
        self.input_faults = self._detect_input(input_check)

        if self.correction is not None:
            out = self._correct(out, x)

        if extra_shape is not None:
            out = out.view(B, *extra_shape, -1)

        if squeeze:
            out = out.squeeze(1)

        return out

    def _detect_weights(
        self, weight_check: torch.Tensor
    ) -> list[tuple[int, int, float]]:
        """Detect weight faults by comparing against saved weight sums."""
        diffs = weight_check - self.weight_sums.unsqueeze(0)
        mask = diffs.abs() > self.atol

        faults = []
        for b in range(weight_check.shape[0]):
            for feat in mask[b].nonzero(as_tuple=False).squeeze(-1).tolist():
                if isinstance(feat, int):
                    faults.append((b, feat, diffs[b, feat].item()))
        return faults

    def _detect_input(self, input_check: torch.Tensor) -> list[tuple[int, int, float]]:
        """Detect input faults by comparing against calibrated range."""
        bad_values = ~torch.isfinite(input_check)

        if self.input_min is not None and self.input_max is not None:
            margin = (self.input_max - self.input_min) * 0.1
            low = self.input_min - margin
            high = self.input_max + margin
            out_of_range = (input_check < low) | (input_check > high)
            bad_values = bad_values | out_of_range

        faults = []
        for b in range(input_check.shape[0]):
            for tok in bad_values[b].nonzero(as_tuple=False).squeeze(-1).tolist():
                if isinstance(tok, int):
                    val = input_check[b, tok].item()
                    faults.append((b, tok, val))
        return faults

    def _correct(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.correction == "zero":
            if self.weight_faults:
                out = self._zero_output_columns(out, self.weight_faults)
            if self.input_faults:
                out = self._zero_output_rows(out, self.input_faults)
        elif self.correction == "subtract":
            if self.weight_faults:
                out = self._subtract_output_columns(out, self.weight_faults)
            if self.input_faults:
                out = self._zero_output_rows(out, self.input_faults)
        elif self.correction == "correct":
            if self.weight_faults:
                if self.clean_weights is not None:
                    out = self._locate_and_fix(out, x, self.weight_faults)
                else:
                    out = self._zero_output_columns(out, self.weight_faults)
            if self.input_faults:
                out = self._zero_output_rows(out, self.input_faults)
        else:  # "rerun"
            if self.weight_faults:
                if self.clean_weights is not None:
                    out = self._rerun_weights_local(out, x, self.weight_faults)
                else:
                    out = self._zero_output_columns(out, self.weight_faults)
            if self.input_faults:
                out = self._rerun_inputs_local(out, x, self.input_faults)
        return out

    def _zero_output_columns(
        self, out: torch.Tensor, faults: list[tuple[int, int, float]]
    ) -> torch.Tensor:
        """Zero out corrupted output columns (one column per faulty weight row)."""
        for b, feat, _ in faults:
            out[b, :, feat] = 0.0
        return out

    def _subtract_output_columns(
        self, out: torch.Tensor, faults: list[tuple[int, int, float]]
    ) -> torch.Tensor:
        """Subtract the weight-sum diff from each corrupted output column"""
        for b, feat, diff in faults:
            out[b, :, feat] -= diff
        return out

    def _locate_and_fix(
        self,
        out: torch.Tensor,
        x: torch.Tensor,
        faults: list[tuple[int, int, float]],
    ) -> torch.Tensor:
        """Exact correction without a full matrix multiply.

        For each faulty output column:
          1. Compare W_faulty[feat] vs W_clean[feat] to find the exact position j
             and fault magnitude delta  (one row subtraction + argmax).
          2. Subtract x[:, :, j] * delta from the faulty column
             (one input column × scalar — no dot product over C_in).
        """
        assert self.clean_weights is not None
        for feat in {feat for _, feat, _ in faults}:
            diff_row = self.original.weight[feat] - self.clean_weights[feat]
            j = int(diff_row.abs().argmax().item())
            delta = diff_row[j].item()
            out[:, :, feat] -= x[:, :, j] * delta
        return out

    def _zero_output_rows(
        self, out: torch.Tensor, faults: list[tuple[int, int, float]]
    ) -> torch.Tensor:
        """Zero out corrupted output rows (one row per faulty input token)."""
        for b, tok, _ in faults:
            out[b, tok, :] = 0.0
        return out

    def _rerun_weights_local(
        self,
        out: torch.Tensor,
        x: torch.Tensor,
        faults: list[tuple[int, int, float]],
    ) -> torch.Tensor:
        """Recompute faulty output features using clean weights."""
        clean_weights = self.clean_weights
        assert clean_weights is not None
        for feat in {feat for _, feat, _ in faults}:
            corrected = F.linear(x, clean_weights[feat : feat + 1, :])
            out[:, :, feat] = corrected.squeeze(-1)
            if self.clean_bias is not None:
                out[:, :, feat] += self.clean_bias[feat]
        return out

    def _rerun_inputs_local(
        self,
        out: torch.Tensor,
        x: torch.Tensor,
        faults: list[tuple[int, int, float]],
    ) -> torch.Tensor:
        """Substitute faulty input tokens with zeros and recompute affected output rows."""
        x_clean = x.clone()
        for b, tok, _ in faults:
            x_clean[b, tok, :] = 0.0
        out_recomputed = F.linear(x_clean, self.original.weight, self.original.bias)
        for b, tok, _ in faults:
            out[b, tok, :] = out_recomputed[b, tok, :]
        return out

    def calibrate_start(self):
        """Start collecting input and output statistics (running stats, no memory buildup)."""
        self._calibrating = True
        self._cal_min = float("inf")
        self._cal_max = float("-inf")
        self._cal_sum = 0.0
        self._cal_count = 0
        self._cal_out_sum = None
        self._cal_out_count = 0
        self._cal_weight_check = None

    def calibrate_end(self):
        """Finish calibration and compute input and output statistics."""
        if self._cal_weight_check is not None:
            self.weight_sums = self._cal_weight_check.to(self.original.weight.device)
        if self._cal_count > 0:
            self.input_min = self._cal_min
            self.input_max = self._cal_max
            self.input_mean = self._cal_sum / self._cal_count
        if self._cal_out_sum is not None and self._cal_out_count > 0:
            self.output_mean = (self._cal_out_sum / self._cal_out_count).cpu()
        self._calibrating = False

    def get_baseline(self, include_weights: bool = False) -> dict:
        """Get baseline data for saving.

        Args:
            include_weights: If True, include full weight matrix for correction.
                             This uses significant memory for large models.
        """
        data = {
            "weight_sums": self.weight_sums.cpu(),
            "w_col_sums": self.w_col_sums.cpu(),
            "input_mean": self.input_mean,
            "input_min": self.input_min,
            "input_max": self.input_max,
        }
        if include_weights:
            data["weights"] = self.original.weight.data.cpu().clone()
            data["bias"] = (
                self.original.bias.data.cpu().clone()
                if self.original.bias is not None
                else None
            )
        return data

    def set_baseline(self, data: dict):
        """Load baseline data."""
        device = self.original.weight.device
        self.weight_sums = data["weight_sums"].to(device)

        if "weights" in data:
            self.clean_weights = data["weights"].to(device)
            if data.get("bias") is not None:
                self.clean_bias = data["bias"].to(device)

        self.input_mean = data.get("input_mean")
        self.input_min = data.get("input_min")
        self.input_max = data.get("input_max")


class CheckOne:
    """ABFT row-sum checksum detector.

    Detects weight faults (static checksums) and input faults (calibrated ranges).

    Example:
        # Save baseline with calibration
        detector = CheckOne(model, layers="fc1")
        detector.calibrate(model)  # Run on training data
        detector.save()
        detector.remove()

        # Later, with faults:
        detector = CheckOne(model, layers="fc1", correction="rerun")
        detector.load()
        outputs = model(images)  # Faults detected and corrected
    """

    name = "checkone"

    def __init__(self, model, layers: str = "all", correction: str | None = None):
        """Wrap model layers with detection.

        Args:
            model: Model instance (with .net/.name) or nn.Module
            layers: Filter ("all", "fc1", "fc2", "qkv", "proj")
            correction: Correction mode — "zero" (zero out faulty outputs),
                        "rerun" (recompute from clean weights/inputs), or None
        """
        if hasattr(model, "net"):
            self.model = model.net
            self.model_name = model.name
        else:
            self.model = model
            self.model_name = "unknown"

        self._model_ref = model  # Keep reference for calibration
        self.wrapped = wrap_layers(self.model, _Wrapper, layers)
        self.correction = correction

        for w in self.wrapped.values():
            w.correction = correction

        print(f"[{self.name}] Wrapped {len(self.wrapped)} layers")

    def calibrate(self, model=None, max_batches: int = 50):
        """Run calibration to collect input statistics.

        Args:
            model: Model instance with dataloader attribute.
                   If None, uses model passed to __init__.
            max_batches: Maximum batches to use for calibration (default 50).
                         More batches = better range estimates but slower.
        """
        if model is None:
            model = self._model_ref

        if not hasattr(model, "dataloader"):
            print("Warning: Model has no dataloader, skipping input calibration")
            return

        for w in self.wrapped.values():
            w.calibrate_start()

        print(f"Calibrating on up to {max_batches} batches...")

        device = next(model.net.parameters()).device
        count = 0
        for images, _ in model.dataloader:
            images = images.to(device, non_blocking=True)
            with torch.inference_mode():
                model.net(images)
            count += 1
            if count >= max_batches:
                break

        print(f"  Calibrated on {count} batches")

        for w in self.wrapped.values():
            w.calibrate_end()

        for name, w in self.wrapped.items():
            if w.input_min is not None:
                print(f"  {name}: input range [{w.input_min:.2f}, {w.input_max:.2f}]")

    def get_faults(self) -> list[DetectedFault]:
        """Get all faults from last forward pass."""
        faults = []
        for name, wrapper in self.wrapped.items():
            for batch, feature, diff in wrapper.weight_faults:
                faults.append(DetectedFault(name, batch, feature, diff, "weight"))
            for batch, token, val in wrapper.input_faults:
                faults.append(DetectedFault(name, batch, token, val, "input"))
        return faults

    def get_weight_faults(self) -> list[DetectedFault]:
        """Get weight faults only."""
        return [f for f in self.get_faults() if f.fault_type == "weight"]

    def get_input_faults(self) -> list[DetectedFault]:
        """Get input faults only."""
        return [f for f in self.get_faults() if f.fault_type == "input"]

    def check(self) -> dict[str, bool]:
        """Check which layers have faults."""
        faults = self.get_faults()
        faulty = {f.layer for f in faults}
        return {name: name in faulty for name in self.wrapped}

    def remove(self):
        """Restore original layers."""
        unwrap_layers(self.model, self.wrapped)
        self.wrapped.clear()

    def save(self, path: Path | str | None = None, include_weights: bool = False):
        """Save baselines for all wrapped layers.

        Args:
            path: Save path (default: data/detection/baseline_{model}.pt)
            include_weights: If True, save full weights for correction.
                             Warning: uses significant memory/disk for large models.
        """
        if path is None:
            DETECTION_DIR.mkdir(parents=True, exist_ok=True)
            path = DETECTION_DIR / f"baseline_{self.model_name}.pt"

        data = {}
        for name, w in self.wrapped.items():
            data[name] = w.get_baseline(include_weights=include_weights)

        torch.save(data, path)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"Saved baseline to {path} ({size_mb:.1f} MB)")

    def load(self, path: Path | str | None = None) -> bool:
        """Load baselines.

        Returns:
            True if loaded, False if file doesn't exist
        """
        if path is None:
            path = DETECTION_DIR / f"baseline_{self.model_name}.pt"

        if not Path(path).exists():
            return False

        data = torch.load(path, weights_only=True)
        for name, wrapper in self.wrapped.items():
            if name in data:
                wrapper.set_baseline(data[name])

        has_input_cal = any(
            self.wrapped[n].input_min is not None for n in self.wrapped if n in data
        )
        has_weights = any(
            self.wrapped[n].clean_weights is not None for n in self.wrapped if n in data
        )

        status = f"input_cal={has_input_cal}, weights={has_weights}"
        print(f"Loaded baseline ({len(data)} layers, {status})")

        if self.correction == "rerun" and not has_weights:
            print(
                "Warning: correction enabled but no weights in baseline. Weight faults cannot be corrected."
            )

        return True

    def print_results(self):
        """Print detection results."""
        weight_faults = self.get_weight_faults()
        input_faults = self.get_input_faults()

        print(f"FAULT DETECTION ({self.name})")
        print(f"Weight faults: {len(weight_faults)}, Input faults: {len(input_faults)}")
        print()

        # Group by layer
        w_by_layer: dict[str, list[DetectedFault]] = {}
        i_by_layer: dict[str, list[DetectedFault]] = {}
        for f in weight_faults:
            w_by_layer.setdefault(f.layer, []).append(f)
        for f in input_faults:
            i_by_layer.setdefault(f.layer, []).append(f)

        faulty_count = 0
        for layer_name in self.wrapped:
            w_faults = w_by_layer.get(layer_name, [])
            i_faults = i_by_layer.get(layer_name, [])

            if w_faults or i_faults:
                faulty_count += 1
                print(f"\033[91m{layer_name}\033[0m")

                if w_faults:
                    groups: dict[tuple[int, float], int] = {}
                    for f in w_faults:
                        key = (f.feature, round(f.diff, 6))
                        groups[key] = groups.get(key, 0) + 1
                    for (feat, diff), count in sorted(groups.items()):
                        print(
                            f"    [W] feature {feat}: diff={diff:.6e} ({count} batches)"
                        )

                if i_faults:
                    tokens = list({f.feature for f in i_faults})
                    print(
                        f"    [I] {len(i_faults)} input faults at tokens: {tokens[:5]}{'...' if len(tokens) > 5 else ''}"
                    )
            else:
                print(f"\033[92m{layer_name}\033[0m")

        print()
        if faulty_count > 0:
            print(f"\033[91mFaults in {faulty_count}/{len(self.wrapped)} layers\033[0m")
        else:
            print(f"\033[92mNo faults ({len(self.wrapped)} layers checked)\033[0m")

    def get_values(self) -> dict:
        """Get results for JSON serialization."""
        faults = self.get_faults()
        weight_faults = [f for f in faults if f.fault_type == "weight"]
        input_faults = [f for f in faults if f.fault_type == "input"]

        return {
            "method": self.name,
            "layers_checked": len(self.wrapped),
            "weight_faults": len({(f.layer, f.feature) for f in weight_faults}),
            "input_faults": len(input_faults),
            "faults": [
                {
                    "layer": f.layer,
                    "batch": f.batch,
                    "feature": f.feature,
                    "diff": f.diff,
                    "type": f.fault_type,
                }
                for f in faults
            ],
        }
