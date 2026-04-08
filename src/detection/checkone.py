import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import calibration_path, weights_path
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

    atol: float = 1e-5
    rtol: float = 0.0
    weights_ext: torch.Tensor

    def __init__(self, original: nn.Linear, name: str):
        super().__init__()
        self.original = original
        self.name = name
        self.C_out = original.weight.shape[0]

        ones_row = torch.ones(
            1,
            original.weight.shape[1],
            dtype=original.weight.dtype,
            device=original.weight.device,
        )
        self.register_buffer(
            "weights_ext",
            torch.cat([original.weight.data.clone(), ones_row], dim=0),
            persistent=False,
        )

        # Golden row sums for weight-fault detection (clean, never mutated by injection)
        self.weight_sums = original.weight.data.sum(dim=1).clone()

        # Per-layer wall-clock timing (enabled by detector when hr --time is set)
        self.elapsed_ms: float = 0.0
        self._timing: bool = False

        # Loaded only when --correction is set
        self.clean_weights: torch.Tensor | None = None
        self.clean_bias: torch.Tensor | None = None

        # Loaded from calibration baseline for input fault detection
        self.input_min: float | None = None
        self.input_max: float | None = None

        self.weight_faults: list[tuple[int, int, float]] = []
        self.input_faults: list[tuple[int, int, float]] = []
        self.correction: str | None = None

        self._calibrating = False
        self._cal_min = float("inf")
        self._cal_max = float("-inf")
        self._cal_sum = 0.0
        self._cal_count = 0
        self._cal_weight_check: torch.Tensor | None = None

    @property
    def weight(self):
        return self.weights_ext[: self.C_out]

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

        _t0 = time.perf_counter() if self._timing else 0.0

        ones_row = torch.ones(B, 1, C, device=x.device, dtype=x.dtype)
        x_ext = torch.cat([x, ones_row], dim=1)

        out_ext = F.linear(x_ext, self.weights_ext)  # [B, N+1, C_out+1]

        weight_check = out_ext[:, -1, : self.C_out]  # ones @ W.T — weight check
        input_check = out_ext[:, :-1, self.C_out]  # x @ ones  — input check

        out = out_ext[:, :-1, : self.C_out]
        if self.original.bias is not None:
            out = out + self.original.bias

        self.weight_faults = self._detect_weights(weight_check)
        self.input_faults = self._detect_input(input_check)

        if self._calibrating:
            self._calibrate_update(weight_check, input_check, out, B, N)

        if getattr(self, "_calibrating_threshold", False):
            self.threshold_calibrate_update(weight_check)

        if self.correction is not None:
            out = self._correct(out, x)

        if extra_shape is not None:
            out = out.view(B, *extra_shape, -1)

        if squeeze:
            out = out.squeeze(1)

        if self._timing:
            self.elapsed_ms += (time.perf_counter() - _t0) * 1000.0

        return out

    def _detect_weights(
        self, weight_check: torch.Tensor
    ) -> list[tuple[int, int, float]]:
        """Detect weight faults by comparing against saved weight sums.

        weight_check is batch-independent (ones @ W.T), so only one batch
        item needs to be checked — all B items carry the same signal.
        """
        actual = weight_check[0]
        mask = ~torch.isclose(actual, self.weight_sums, atol=self.atol, rtol=self.rtol)
        if not mask.any():
            return []
        diffs = actual - self.weight_sums
        return [(0, int(feat), float(diffs[feat])) for feat in mask.nonzero(as_tuple=False).squeeze(-1).tolist()]

    def _detect_input(self, input_check: torch.Tensor) -> list[tuple[int, int, float]]:
        """Detect input faults by comparing against calibrated range.

        No-op if calibration has not been run — avoids extra GPU kernels
        during uncalibrated inference.
        """
        if self.input_min is None or self.input_max is None:
            return []

        margin = (self.input_max - self.input_min) * 0.1
        low = self.input_min - margin
        high = self.input_max + margin
        bad_values = (
            ~torch.isfinite(input_check) | (input_check < low) | (input_check > high)
        )
        if not bad_values.any():
            return []
        indices = bad_values.nonzero(as_tuple=False)
        return [(int(r[0]), int(r[1]), float(input_check[r[0], r[1]])) for r in indices]

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
            diff_row = self.weight[feat] - self.clean_weights[feat]
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
        out_recomputed = F.linear(x_clean, self.weight, self.original.bias)
        for b, tok, _ in faults:
            out[b, tok, :] = out_recomputed[b, tok, :]
        return out

    def _calibrate_update(
        self,
        weight_check: torch.Tensor,
        input_check: torch.Tensor,
        out: torch.Tensor,
        B: int,
        N: int,
    ):
        if self._cal_weight_check is None:
            self._cal_weight_check = weight_check.detach()[0].cpu().clone()
        self._cal_min = min(self._cal_min, input_check.min().item())
        self._cal_max = max(self._cal_max, input_check.max().item())
        self._cal_sum += input_check.sum().item()
        self._cal_count += input_check.numel()

    def calibrate_start(self):
        """Start collecting input statistics for input fault detection."""
        self._calibrating = True
        self._cal_min = float("inf")
        self._cal_max = float("-inf")
        self._cal_sum = 0.0
        self._cal_count = 0
        self._cal_weight_check = None

    def calibrate_end(self):
        """Finish calibration: update weight_sums and input range."""
        if self._cal_weight_check is not None:
            self.weight_sums = self._cal_weight_check.to(self.weights_ext.device)
        if self._cal_count > 0:
            self.input_min = self._cal_min
            self.input_max = self._cal_max
        self._calibrating = False

    def threshold_calibrate_start(self):
        """Start collecting clean-condition detection noise for threshold calibration."""
        self._cal_abs_values: list[float] = []
        self._cal_rel_values: list[float] = []
        self._calibrating_threshold = True

    def threshold_calibrate_update(self, weight_check: torch.Tensor):
        """Record per-batch absolute and relative noise for threshold estimation.

        Compares GEMM-computed row sums against directly-computed row sums
        from weights_ext — independent of self.weight_sums state so input
        calibration running first does not contaminate the noise measurement.
        """
        golden = self.weights_ext[: self.C_out].sum(dim=1)
        diff = (weight_check[0] - golden).abs()
        self._cal_abs_values.append(diff.max().item())
        self._cal_rel_values.append((diff / (golden.abs() + 1e-12)).max().item())

    def threshold_calibrate_end(self, margin: float = 1.0):
        """Set atol and rtol using mean + margin * std of per-batch noise.

        Uses the torch.allclose formula at inference: |diff| > atol + rtol * |signal|.
        atol handles near-zero signals; rtol scales with signal magnitude.
        margin is the number of standard deviations above the mean (same for both methods).
        For CheckOne, rtol ≈ 0 (noise is data-independent); for Checksum, rtol > 0
        for layers where noise scales with input magnitude.
        """
        if not self._cal_abs_values:
            return

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            return mean, std

        abs_mean, abs_std = _mean_std(self._cal_abs_values)
        rel_mean, rel_std = _mean_std(self._cal_rel_values)
        self.atol = abs_mean + margin * abs_std
        self.rtol = rel_mean + margin * rel_std
        self._calibrating_threshold = False

    def get_baseline(self, include_weights: bool = False) -> dict:
        """Get baseline data for saving.

        Always saves: weight_sums (for detection), input_min/max (for input detection).
        With include_weights: full weight matrix (for correction only).
        """
        data: dict = {
            "weight_sums": self.weight_sums.cpu(),
            "input_min": self.input_min,
            "input_max": self.input_max,
            "atol": self.atol,
            "rtol": self.rtol,
        }
        if include_weights:
            data["weights"] = self.weight.data.cpu().clone()
            data["bias"] = (
                self.original.bias.data.cpu().clone()
                if self.original.bias is not None
                else None
            )
        return data

    def set_baseline(self, data: dict, dtype: torch.dtype | None = None):
        """Load baseline data."""
        device = self.weights_ext.device
        target_dtype = dtype or self.weights_ext.dtype
        if "weight_sums" in data:
            self.weight_sums = data["weight_sums"].to(device=device, dtype=target_dtype)
        self.input_min = data.get("input_min")
        self.input_max = data.get("input_max")
        if "atol" in data:
            self.atol = data["atol"]
        if "rtol" in data:
            self.rtol = data["rtol"]
        if "weights" in data and self.correction is not None:
            self.clean_weights = data["weights"].to(device=device, dtype=target_dtype)
            if data.get("bias") is not None:
                self.clean_bias = data["bias"].to(device=device, dtype=target_dtype)


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

    def calibrate(self, model=None, max_batches: int | None = None) -> int:
        """Run calibration to collect input statistics. Returns total samples used."""
        if model is None:
            model = self._model_ref

        if not hasattr(model, "dataloader"):
            print("Warning: Model has no dataloader, skipping input calibration")
            return 0

        for w in self.wrapped.values():
            w.calibrate_start()

        limit_str = f"up to {max_batches} batches" if max_batches is not None else "full dataset"
        print(f"Calibrating inputs on {limit_str}...")

        device = next(model.net.parameters()).device
        total_samples = 0
        for images, _ in model.dataloader:
            images = images.to(device, non_blocking=True)
            with torch.inference_mode():
                model.net(images)
            total_samples += len(images)
            if max_batches is not None and total_samples >= max_batches * len(images):
                break

        for w in self.wrapped.values():
            w.calibrate_end()

        print(f"  Calibrated on {total_samples} samples")
        for name, w in self.wrapped.items():
            if w.input_min is not None:
                print(f"  {name}: input range [{w.input_min:.2f}, {w.input_max:.2f}]")
        return total_samples

    def calibrate_threshold(self, model=None, max_batches: int | None = None, margin: float = 3.0):
        """Calibrate detection threshold from clean data.

        Runs inference on clean model, measures floating-point noise in the detection
        signal, then sets atol = mean + margin*std (3-sigma rule) for each layer.

        Args:
            model: Model instance with dataloader. If None, uses model from __init__.
            max_batches: Batches to use (None = full dataset, recommended).
            margin: Standard deviations above mean (default 3.0 = 3-sigma rule).
        """
        if model is None:
            model = self._model_ref

        if not hasattr(model, "dataloader"):
            print("Warning: Model has no dataloader, skipping threshold calibration")
            return 0

        for w in self.wrapped.values():
            w.threshold_calibrate_start()

        limit_str = f"up to {max_batches} batches" if max_batches is not None else "full dataset"
        print(f"Calibrating threshold on {limit_str} (margin={margin})...")

        device = next(model.net.parameters()).device
        total_samples = 0
        batch_count = 0
        for images, _ in model.dataloader:
            images = images.to(device, non_blocking=True)
            with torch.inference_mode():
                model.net(images)
            total_samples += len(images)
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

        for w in self.wrapped.values():
            w.threshold_calibrate_end(margin=margin)

        print(f"  Calibrated on {total_samples} samples ({batch_count} batches)")
        for name, w in self.wrapped.items():
            abs_vals = getattr(w, "_cal_abs_values", [])
            if abs_vals:
                abs_mean = sum(abs_vals) / len(abs_vals)
                abs_std = (sum((v - abs_mean) ** 2 for v in abs_vals) / len(abs_vals)) ** 0.5
                print(f"  {name}: atol={w.atol:.2e} rtol={w.rtol:.2e}  (abs max={max(abs_vals):.2e} std={abs_std:.2e})")
        return total_samples

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

    def enable_timing(self, enabled: bool = True):
        """Enable or disable per-layer wall-clock timing in all wrapped layers."""
        for w in self.wrapped.values():
            w._timing = enabled
            w.elapsed_ms = 0.0

    def get_layer_times(self) -> dict[str, float]:
        """Return accumulated wall-clock time (ms) per wrapped layer."""
        return {name: w.elapsed_ms for name, w in self.wrapped.items()}

    def remove(self):
        """Restore original layers."""
        unwrap_layers(self.model, self.wrapped)
        self.wrapped.clear()

    def save(self, path: Path | str | None = None, include_weights: bool = False):
        """Save calibration to data/{model}/calibration/checkone.pt (always overwritten).
        If include_weights, also saves to data/{model}/weights/checkone.pt.
        """
        p = Path(path) if path else calibration_path(self.model_name, "checkone")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {name: w.get_baseline(include_weights=False) for name, w in self.wrapped.items()}
        torch.save(data, p)
        print(f"Saved checkone calibration to {p} ({p.stat().st_size / 1024**2:.1f} MB)")

        if include_weights:
            wp = weights_path(self.model_name, "checkone")
            wp.parent.mkdir(parents=True, exist_ok=True)
            wdata = {name: w.get_baseline(include_weights=True) for name, w in self.wrapped.items()}
            torch.save(wdata, wp)
            print(f"Saved checkone weights to {wp} ({wp.stat().st_size / 1024**2:.1f} MB)")

    def load(self, path: Path | str | None = None) -> bool:
        """Load calibration from data/{model}/calibration/checkone.pt.
        Also loads weights from data/{model}/weights/checkone.pt if present.
        Returns True if calibration file was found.
        """
        p = Path(path) if path else calibration_path(self.model_name, "checkone")
        if not p.exists():
            return False

        data = torch.load(p, weights_only=False)
        for name, wrapper in self.wrapped.items():
            if name in data:
                wrapper.set_baseline(data[name])
        print(f"Loaded checkone calibration from {p} ({len(data)} layers)")

        wp = weights_path(self.model_name, "checkone")
        if wp.exists():
            wdata = torch.load(wp, weights_only=False)
            for name, wrapper in self.wrapped.items():
                if name in wdata and "weights" in wdata[name]:
                    wrapper.set_baseline({"weights": wdata[name]["weights"],
                                          "bias": wdata[name].get("bias")})
            print(f"Loaded checkone weights from {wp}")

        has_weights = any(self.wrapped[n].clean_weights is not None for n in self.wrapped)
        for name, w in self.wrapped.items():
            if name in data:
                rtol_str = f" rtol={w.rtol:.2e}" if w.rtol > 0 else ""
                print(f"  {name}: atol={w.atol:.2e}{rtol_str}")

        if self.correction in ("rerun", "correct") and not has_weights:
            print("Warning: correction enabled but no weights found. Run save --weights first.")

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
