from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import calibration_path
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

        # Loaded from calibration baseline for input fault detection
        self.input_min: float | None = None
        self.input_max: float | None = None

        # Pre-allocated ones buffer — reused across forward calls, resized only when B changes
        self._ones_buf: torch.Tensor | None = None

        self.weight_faults: list[tuple[int, int, float]] = []
        self.input_faults: list[tuple[int, int, float]] = []
        self.correction: str | None = None

        self._calibrating = False
        self._cal_min = float("inf")
        self._cal_max = float("-inf")
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

        if self._ones_buf is None or self._ones_buf.shape != (B, 1, C):
            self._ones_buf = x.new_ones(B, 1, C)
        x_ext = torch.cat([x, self._ones_buf], dim=1)

        out_ext = F.linear(x_ext, self.weights_ext)  # [B, N+1, C_out+1]

        weight_check = out_ext[:, -1, : self.C_out]  # ones @ W.T — weight check
        input_check = out_ext[:, :-1, self.C_out]  # x @ ones  — input check
        out = out_ext[:, :-1, : self.C_out]

        if self.original.bias is not None:
            out = out + self.original.bias

        if self._calibrating or getattr(self, "_calibrating_threshold", False):
            self.weight_faults = []
            self.input_faults = []
            if self._calibrating:
                self._calibrate_update(weight_check, input_check)
            if getattr(self, "_calibrating_threshold", False):
                self.threshold_calibrate_update(weight_check)
        else:
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
        """Detect weight faults by comparing each batch item against saved weight sums.

        For persistent weight faults all batch items carry the same signal, but
        checking per-batch also catches transient computation faults that only
        affect a single batch item.
        """
        # weight_sums: [C_out] → broadcast against weight_check: [B, C_out]
        mask = ~torch.isclose(
            weight_check, self.weight_sums.unsqueeze(0), atol=self.atol, rtol=self.rtol
        )
        if not mask.any():
            return []
        diffs = weight_check - self.weight_sums.unsqueeze(0)
        indices = mask.nonzero(as_tuple=False)
        return [(int(r[0]), int(r[1]), float(diffs[r[0], r[1]])) for r in indices]

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
        bad_values = (input_check < low) | (input_check > high)
        if not bad_values.any():
            return []
        indices = bad_values.nonzero(as_tuple=False)
        return [(int(r[0]), int(r[1]), float(input_check[r[0], r[1]])) for r in indices]

    def _correct(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.weight_faults:
            out = self._zero_output_columns(out, self.weight_faults)
        if self.input_faults:
            out = self._zero_output_rows(out, self.input_faults)
        return out

    def _zero_output_columns(
        self, out: torch.Tensor, faults: list[tuple[int, int, float]]
    ) -> torch.Tensor:
        """Zero out corrupted output columns per affected batch item."""
        for b, feat, _ in faults:
            out[b, :, feat] = 0.0
        return out

    def _zero_output_rows(
        self, out: torch.Tensor, faults: list[tuple[int, int, float]]
    ) -> torch.Tensor:
        """Zero out corrupted output rows (one row per faulty input token)."""
        for b, tok, _ in faults:
            out[b, tok, :] = 0.0
        return out

    def _calibrate_update(self, weight_check: torch.Tensor, input_check: torch.Tensor):
        if self._cal_weight_check is None:
            self._cal_weight_check = weight_check.detach()[0].cpu().clone()
        self._cal_min = min(self._cal_min, input_check.min().item())
        self._cal_max = max(self._cal_max, input_check.max().item())
        self._cal_count += input_check.numel()

    def calibrate_start(self):
        """Start collecting input statistics for input fault detection."""
        self._calibrating = True
        self._cal_min = float("inf")
        self._cal_max = float("-inf")
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
        self._cal_abs_buf: list[torch.Tensor] = []
        # Cache golden row sums once — they never change during calibration
        self._cal_golden = self.weights_ext[: self.C_out].sum(dim=1)
        self._calibrating_threshold = True

    def threshold_calibrate_update(self, weight_check: torch.Tensor):
        """Accumulate per-batch noise — stays on GPU, no CPU sync until end."""
        diff = (weight_check - self._cal_golden.unsqueeze(0)).abs()
        self._cal_abs_buf.append(diff.max())

    def threshold_calibrate_end(self):
        """Set atol from maximum observed noise during clean calibration run."""
        if not self._cal_abs_buf:
            return
        self.atol = float(torch.stack(self._cal_abs_buf).max())
        self._calibrating_threshold = False

    def get_baseline(self) -> dict:
        """Get baseline data for saving."""
        return {
            "weight_sums": self.weight_sums.cpu(),
            "input_min": self.input_min,
            "input_max": self.input_max,
            "atol": self.atol,
        }

    def set_baseline(self, data: dict, dtype: torch.dtype | None = None):
        """Load baseline data."""
        device = self.weights_ext.device
        target_dtype = dtype or self.weights_ext.dtype
        if "weight_sums" in data:
            self.weight_sums = data["weight_sums"].to(device=device, dtype=target_dtype)
        if "input_min" in data:
            self.input_min = data["input_min"]
        if "input_max" in data:
            self.input_max = data["input_max"]
        if "atol" in data:
            self.atol = data["atol"]


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

    def calibrate(
        self,
        model=None,
        max_batches: int | None = None,
        inputs: bool = True,
        threshold: bool = False,
    ) -> int:
        """Run one calibration pass collecting input stats and/or threshold noise.

        Combines both into a single dataloader pass when both are requested,
        avoiding the cost of iterating through the dataset twice.

        Args:
            model: Model instance with dataloader. If None, uses model from __init__.
            max_batches: Number of batches to use (None = full dataset).
            inputs: Calibrate input fault detection range.
            threshold: Calibrate detection threshold (atol).
        Returns:
            Total samples used.
        """
        if model is None:
            model = self._model_ref

        if not hasattr(model, "dataloader"):
            print("Warning: Model has no dataloader, skipping calibration")
            return 0

        if inputs:
            for w in self.wrapped.values():
                w.calibrate_start()
        if threshold:
            for w in self.wrapped.values():
                w.threshold_calibrate_start()

        parts = []
        if inputs:
            parts.append("inputs")
        if threshold:
            parts.append("threshold")
        limit_str = (
            f"up to {max_batches} batches"
            if max_batches is not None
            else "full dataset"
        )
        print(f"Calibrating {'+'.join(parts)} on {limit_str}...")

        device = next(model.net.parameters()).device
        total_samples = 0
        batch_count = 0
        report_every = max(1, (max_batches or 100) // 10)
        for images, _ in model.dataloader:
            images = images.to(device, non_blocking=True)
            with torch.inference_mode():
                model.net(images)
            total_samples += len(images)
            batch_count += 1
            if batch_count % report_every == 0:
                limit = max_batches or "?"
                print(f"  [{batch_count}/{limit}] {total_samples} samples", flush=True)
            if max_batches is not None and batch_count >= max_batches:
                break

        if inputs:
            for w in self.wrapped.values():
                w.calibrate_end()
        if threshold:
            for w in self.wrapped.values():
                w.threshold_calibrate_end()

        print(f"  Calibrated on {total_samples} samples ({batch_count} batches)")
        if inputs:
            for name, w in self.wrapped.items():
                if w.input_min is not None:
                    print(
                        f"  {name}: input range [{w.input_min:.2f}, {w.input_max:.2f}]"
                    )
        if threshold:
            for name, w in self.wrapped.items():
                if getattr(w, "_cal_abs_buf", []):
                    print(f"  {name}: atol={w.atol:.2e}")
        return total_samples

    def calibrate_threshold(self, model=None, max_batches: int | None = None):
        """Calibrate detection threshold only. Prefer calibrate(threshold=True) for combined passes."""
        return self.calibrate(model=model, max_batches=max_batches, inputs=False, threshold=True)

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

    def save(self, path: Path | str | None = None, save_calibration: bool = True):
        """Save calibration to data/{model}/calibration/checkone.pt."""
        if save_calibration:
            p = Path(path) if path else calibration_path(self.model_name, "checkone")
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {name: w.get_baseline() for name, w in self.wrapped.items()}
            torch.save(data, p)
            print(f"Saved checkone calibration to {p} ({p.stat().st_size / 1024**2:.1f} MB)")

    def load(self, path: Path | str | None = None, verbose: bool = False) -> bool:
        """Load calibration from data/{model}/calibration/checkone.pt.
        Raises RuntimeError if calibration file is missing or input calibration was not run.
        """
        p = Path(path) if path else calibration_path(self.model_name, "checkone")
        if not p.exists():
            raise RuntimeError(
                f"CheckOne: no calibration file found at {p}.\n"
                f"  Run: save --inputs --threshold  to calibrate first."
            )

        data = torch.load(p, weights_only=False)
        for name, wrapper in self.wrapped.items():
            if name in data:
                wrapper.set_baseline(data[name])
        print(f"Loaded checkone calibration from {p} ({len(data)} layers)")

        missing_input_cal = [n for n, w in self.wrapped.items() if w.input_min is None]
        if missing_input_cal:
            raise RuntimeError(
                f"CheckOne: input calibration missing for {len(missing_input_cal)} layer(s) "
                f"(e.g. {missing_input_cal[0]}).\n"
                f"  Run: save --inputs  to calibrate input detection ranges."
            )

        if verbose:
            for name, w in self.wrapped.items():
                if name in data:
                    print(f"  {name}: atol={w.atol:.2e}")

        return True

    def reset(self):
        """Clear per-run fault lists in all wrappers (call before each run)."""
        for w in self.wrapped.values():
            w.weight_faults = []
            w.input_faults = []

    def start_threshold_calibration(self):
        """Activate threshold calibration hooks (fire on every subsequent net(images))."""
        for w in self.wrapped.values():
            w.threshold_calibrate_start()

    def end_threshold_calibration(self):
        """Finalise threshold calibration and print per-layer tolerances."""
        for w in self.wrapped.values():
            w.threshold_calibrate_end()
        print("CheckOne threshold calibration complete:")
        for name, w in self.wrapped.items():
            if getattr(w, "_cal_abs_buf", []):
                print(f"  {name}: atol={w.atol:.2e}")

    def start_input_calibration(self):
        """Activate input-range calibration hooks."""
        for w in self.wrapped.values():
            w.calibrate_start()

    def end_input_calibration(self):
        """Finalise input-range calibration and print per-layer ranges."""
        for w in self.wrapped.values():
            w.calibrate_end()
        print("CheckOne input calibration complete:")
        for name, w in self.wrapped.items():
            if w.input_min is not None:
                print(f"  {name}: input range [{w.input_min:.2f}, {w.input_max:.2f}]")

    def print_summary(self, layer_stats: dict[str, dict[str, int]]):
        """Print aggregate detection table across all runs."""
        if not layer_stats:
            return
        col = max(len(l) for l in layer_stats) + 2
        print(
            f"\nDetection Summary ({self.name}):\n"
            f"  {'Layer':<{col}}  {'Injected':>8}  {'Detected':>8}  {'Rate':>7}  {'False+':>6}  {'Input Det':>9}\n"
            f"  {'-' * col}  {'--------':>8}  {'--------':>8}  {'-------':>7}  {'------':>6}  {'---------':>9}"
        )
        for layer in sorted(layer_stats):
            s = layer_stats[layer]
            inj = s["injected"]
            det = s["detected"]
            fp = s["false_positive"]
            inp = s.get("input_faults", 0)
            rate = f"{100 * det / inj:.1f}%" if inj else "  n/a"
            print(f"  {layer:<{col}}  {inj:>8}  {det:>8}  {rate:>7}  {fp:>6}  {inp:>9}")

        total_inj = sum(s["injected"] for s in layer_stats.values())
        total_det = sum(s["detected"] for s in layer_stats.values())
        total_fp = sum(s["false_positive"] for s in layer_stats.values())
        total_inp = sum(s.get("input_faults", 0) for s in layer_stats.values())
        total_rate = f"{100 * total_det / total_inj:.1f}%" if total_inj else "  n/a"
        print(
            f"  {'-' * col}  {'--------':>8}  {'--------':>8}  {'-------':>7}  {'------':>6}  {'---------':>9}\n"
            f"  {'TOTAL':<{col}}  {total_inj:>8}  {total_det:>8}  {total_rate:>7}  {total_fp:>6}  {total_inp:>9}"
        )

    def print_timing_summary(self, all_runs: list[dict]):
        """Print timing: total and per-sample mean ± std across runs."""
        timed = [r for r in all_runs if r.get("times_ms")]
        if not timed:
            return
        per_run_ms = [sum(r["times_ms"]) for r in timed]
        per_run_mpb = [ms / len(r["times_ms"]) for ms, r in zip(per_run_ms, timed)]
        total_ms = sum(per_run_ms)
        avg_mpb = sum(per_run_mpb) / len(per_run_mpb)
        std_mpb = (
            (sum((v - avg_mpb) ** 2 for v in per_run_mpb) / (len(per_run_mpb) - 1))
            ** 0.5
            if len(per_run_mpb) > 1
            else 0.0
        )
        print(
            f"\nDetection Timing — {self.name} ({len(timed)} runs):\n"
            f"  Total:         {total_ms:.1f} ms  ({total_ms / 1000:.2f} s)\n"
            f"  Avg per batch: {avg_mpb:.4f} ms  ±{std_mpb:.4f} ms std"
        )

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
