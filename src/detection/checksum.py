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
    """Fault detected by approxABFT checksum comparison."""

    layer: str
    batch: int
    feature: int
    diff: float
    fault_type: str = "col_check"


class _Wrapper(nn.Module):
    """approxABFT row/column checksum wrapper for nn.Linear.

    Extended matrix multiply embeds two golden checksums:
      - Row check: X @ w_col_sums_golden == Y.sum(dim=-1)
        With a weight fault in row i, ALL row checks fail.
      - Col check: X_token_sums @ W_golden^T == Y.sum(dim=1)
        With a weight fault in row i, ONLY column i check fails.

    This lets us detect (row check) and localize (col check) weight faults.
    """

    atol: float = 1e-2
    rtol: float = 0.0
    weights_ext: torch.Tensor

    def __init__(self, original: nn.Linear, name: str):
        super().__init__()
        self.original = original
        self.name = name
        self.C_out = original.weight.shape[0]

        w_col_sums = original.weight.data.sum(dim=0)
        self.register_buffer(
            "weights_ext",
            torch.cat([original.weight.data.clone(), w_col_sums.unsqueeze(0)], dim=0),
            persistent=False,
        )

        self.clean_weights: torch.Tensor | None = None
        self.clean_bias: torch.Tensor | None = None

        self.row_faults: list[tuple[int, int, float]] = []
        self.col_faults: list[tuple[int, int, float]] = []

        self.correction: str | None = None

        self.elapsed_ms: float = 0.0
        self._timing: bool = False

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

        B, N, C_in = x.shape

        _t0 = time.perf_counter() if self._timing else 0.0

        x_token_sums = x.sum(dim=1, keepdim=True)
        x_ext = torch.cat([x, x_token_sums], dim=1)

        out_ext = F.linear(x_ext, self.weights_ext)

        out = out_ext[:, :N, : self.C_out]
        golden_row = out_ext[:, :N, self.C_out]
        actual_col = out_ext[:, N, : self.C_out]

        if self.original.bias is not None:
            out = out + self.original.bias

        actual_row = out_ext[:, :N, : self.C_out].sum(dim=-1)

        self.row_faults = self._detect_rows(actual_row, golden_row)
        self.col_faults = self._detect_cols(actual_col, x_token_sums)

        if getattr(self, "_calibrating_threshold", False):
            self.threshold_calibrate_update(actual_row, golden_row)

        if self.correction is not None:
            out = self._correct(out, x)

        if extra_shape is not None:
            out = out.view(B, *extra_shape, -1)

        if squeeze:
            out = out.squeeze(1)

        if self._timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed_ms += (time.perf_counter() - _t0) * 1000.0

        return out

    def _detect_rows(
        self,
        actual_row: torch.Tensor,
        golden_row: torch.Tensor,
    ) -> list[tuple[int, int, float]]:
        """Row check: compare actual Y row sums vs golden.

        With a single weight fault in row i, EVERY token's row sum is off
        because every output vector Y[b,n,:] now contains a corrupted feature i.
        """
        mask = ~torch.isclose(actual_row, golden_row, atol=self.atol, rtol=self.rtol)
        if not mask.any():
            return []
        diffs = actual_row - golden_row
        indices = mask.nonzero(as_tuple=False)
        return [(int(r[0]), int(r[1]), float(diffs[r[0], r[1]])) for r in indices]

    def _detect_cols(
        self,
        actual_col: torch.Tensor,
        x_token_sums: torch.Tensor,
    ) -> list[tuple[int, int, float]]:
        """Col check: compare actual col sums vs golden.

        With a single weight fault in row i, ONLY feature i's column sum is off.
        Requires clean_weights to compute golden col sums at inference time.
        """
        if self.clean_weights is None:
            return []

        golden_col = F.linear(x_token_sums.squeeze(1), self.clean_weights)

        mask = ~torch.isclose(actual_col, golden_col, atol=self.atol, rtol=self.rtol)
        if not mask.any():
            return []
        diffs = actual_col - golden_col
        indices = mask.nonzero(as_tuple=False)
        return [(int(r[0]), int(r[1]), float(diffs[r[0], r[1]])) for r in indices]

    def _correct(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        faulty_feats = {feat for _, feat, _ in self.col_faults}
        if not faulty_feats:
            return out

        if self.correction == "zero":
            for b, feat, _ in self.col_faults:
                out[b, :, feat] = 0.0
        elif self.correction == "correct":
            if self.clean_weights is not None:
                out = self._locate_and_fix(out, x, faulty_feats)
            else:
                for b, feat, _ in self.col_faults:
                    out[b, :, feat] = 0.0
        else:
            if self.clean_weights is not None:
                for feat in faulty_feats:
                    corrected = F.linear(x, self.clean_weights[feat : feat + 1, :])
                    out[:, :, feat] = corrected.squeeze(-1)
                    if self.clean_bias is not None:
                        out[:, :, feat] += self.clean_bias[feat]
            else:
                for b, feat, _ in self.col_faults:
                    out[b, :, feat] = 0.0
        return out

    def _locate_and_fix(
        self,
        out: torch.Tensor,
        x: torch.Tensor,
        faulty_feats: set[int],
    ) -> torch.Tensor:
        """Exact correction without a full matrix multiply.

        For each faulty output column:
          1. Compare W_faulty[feat] vs W_clean[feat] to find position j
             and fault magnitude delta  (one row subtraction + argmax).
          2. Subtract x[:, :, j] * delta from the faulty column.
        """
        assert self.clean_weights is not None
        for feat in faulty_feats:
            diff_row = self.weight[feat] - self.clean_weights[feat]
            j = int(diff_row.abs().argmax().item())
            delta = diff_row[j].item()
            out[:, :, feat] -= x[:, :, j] * delta
        return out

    def threshold_calibrate_start(self):
        """Start collecting clean-condition row-check noise for threshold calibration."""
        self._cal_abs_values: list[float] = []
        self._cal_rel_values: list[float] = []
        self._calibrating_threshold = True

    def threshold_calibrate_update(
        self, actual_row: torch.Tensor, golden_row: torch.Tensor
    ):
        """Record per-batch absolute and relative noise for threshold estimation."""
        diff = (actual_row - golden_row).abs()
        self._cal_abs_values.append(diff.max().item())
        self._cal_rel_values.append((diff / (golden_row.abs() + 1e-12)).max().item())

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
        """Return baseline data for saving.

        With include_weights: full weight matrix (needed for col-check and correction).
        """
        data: dict = {"atol": self.atol, "rtol": self.rtol}
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
        if "atol" in data:
            self.atol = data["atol"]
        if "rtol" in data:
            self.rtol = data["rtol"]
        if "weights" in data:
            w = data["weights"].to(device=device, dtype=dtype or self.weights_ext.dtype)
            self.clean_weights = w
            if data.get("bias") is not None:
                self.clean_bias = data["bias"].to(
                    device=device, dtype=dtype or self.weights_ext.dtype
                )


class Checksum:
    """approxABFT row/column checksum detector.

    Embeds golden checksums into a single extended matrix multiply —
    no separate calibration pass required.

    Row check alone detects that a weight fault occurred.
    Col check (requires saved weights) localises which output feature is faulty.

    Example:
        # Save baseline (weights needed for col-check and rerun correction)
        detector = Checksum(model, layers="fc1")
        detector.save(include_weights=True)
        detector.remove()

        # Later, with faults:
        detector = Checksum(model, layers="fc1", correction="rerun")
        detector.load()
        outputs = model(images)   # faults detected and corrected
        detector.print_results()
    """

    name = "checksum"

    def __init__(self, model, layers: str = "all", correction: str | None = None):
        """Wrap model layers with approxABFT detection.

        Args:
            model: Model instance (with .net/.name) or plain nn.Module.
            layers: Layer filter ("all", "fc1", "fc2", "qkv", "proj").
            correction: "zero" (zero faulty output column),
                        "rerun" (recompute from clean weights), or None.
        """
        if hasattr(model, "net"):
            self.model = model.net
            self.model_name = model.name
        else:
            self.model = model
            self.model_name = "unknown"

        self._model_ref = model
        self.wrapped = wrap_layers(self.model, _Wrapper, layers)
        self.correction = correction

        for w in self.wrapped.values():
            w.correction = correction

        print(f"[{self.name}] Wrapped {len(self.wrapped)} layers")

    def get_faults(self) -> list[DetectedFault]:
        """Get all faults from the last forward pass."""
        faults = []
        for name, wrapper in self.wrapped.items():
            for batch, tok, diff in wrapper.row_faults:
                faults.append(DetectedFault(name, batch, tok, diff, "row_check"))
            for batch, feat, diff in wrapper.col_faults:
                faults.append(DetectedFault(name, batch, feat, diff, "col_check"))
        return faults

    def get_weight_faults(self) -> list[DetectedFault]:
        """Localised weight faults (column check hits)."""
        return [f for f in self.get_faults() if f.fault_type == "col_check"]

    def get_detection_flags(self) -> list[DetectedFault]:
        """Raw row-check failures (fault present, column not yet identified)."""
        return [f for f in self.get_faults() if f.fault_type == "row_check"]

    def check(self) -> dict[str, bool]:
        """Return {layer_name: fault_detected} for every wrapped layer."""
        return {
            name: bool(w.row_faults or w.col_faults) for name, w in self.wrapped.items()
        }

    def calibrate_threshold(
        self, model=None, max_batches: int | None = None, margin: float = 3.0
    ):
        """Calibrate detection threshold from clean data.

        Runs inference on clean model, measures floating-point noise in the row-check
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
        """Save calibration to data/{model}/calibration/checksum.pt (always overwritten).
        If include_weights, also saves to data/{model}/weights/checksum.pt.
        """
        p = Path(path) if path else calibration_path(self.model_name, "checksum")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {name: w.get_baseline(include_weights=False) for name, w in self.wrapped.items()}
        torch.save(data, p)
        print(f"Saved checksum calibration to {p} ({p.stat().st_size / 1024**2:.1f} MB)")

        if include_weights:
            wp = weights_path(self.model_name, "checksum")
            wp.parent.mkdir(parents=True, exist_ok=True)
            wdata = {name: w.get_baseline(include_weights=True) for name, w in self.wrapped.items()}
            torch.save(wdata, wp)
            print(f"Saved checksum weights to {wp} ({wp.stat().st_size / 1024**2:.1f} MB)")

    def load(self, path: Path | str | None = None) -> bool:
        """Load calibration from data/{model}/calibration/checksum.pt.
        Also loads weights from data/{model}/weights/checksum.pt if present.
        Returns True if calibration file was found.
        """
        p = Path(path) if path else calibration_path(self.model_name, "checksum")
        if not p.exists():
            return False

        data = torch.load(p, weights_only=False)
        for name, wrapper in self.wrapped.items():
            if name in data:
                wrapper.set_baseline(data[name])
        print(f"Loaded checksum calibration from {p} ({len(data)} layers)")

        wp = weights_path(self.model_name, "checksum")
        if wp.exists():
            wdata = torch.load(wp, weights_only=False)
            for name, wrapper in self.wrapped.items():
                if name in wdata and "weights" in wdata[name]:
                    wrapper.set_baseline({"weights": wdata[name]["weights"],
                                          "bias": wdata[name].get("bias")})
            print(f"Loaded checksum weights from {wp}")

        has_weights = any(self.wrapped[n].clean_weights is not None for n in self.wrapped)
        for name, w in self.wrapped.items():
            if name in data:
                rtol_str = f" rtol={w.rtol:.2e}" if w.rtol > 0 else ""
                print(f"  {name}: atol={w.atol:.2e}{rtol_str}")

        if self.correction in ("rerun", "zero", "correct") and not has_weights:
            print("Warning: correction enabled but no weights found. Run save --weights first.")

        return True

    # Printing

    def print_results(self):
        """Print detection and localisation results."""
        faults = self.get_faults()
        col_faults = [f for f in faults if f.fault_type == "col_check"]
        row_faults = [f for f in faults if f.fault_type == "row_check"]

        c_by_layer: dict[str, list[DetectedFault]] = {}
        r_by_layer: dict[str, list[DetectedFault]] = {}
        for f in col_faults:
            c_by_layer.setdefault(f.layer, []).append(f)
        for f in row_faults:
            r_by_layer.setdefault(f.layer, []).append(f)

        print(f"FAULT DETECTION ({self.name})")
        print()

        faulty_count = 0
        for layer_name in self.wrapped:
            c_faults = c_by_layer.get(layer_name, [])
            r_faults = r_by_layer.get(layer_name, [])

            if c_faults or r_faults:
                faulty_count += 1
                print(f"\033[91m{layer_name}\033[0m")

                if r_faults:
                    n_batches = len({f.batch for f in r_faults})
                    n_checks = len(r_faults)
                    print(
                        f"    [ROW] {n_checks} failed row checks across {n_batches} batch(es)"
                    )

                if c_faults:
                    by_feat: dict[int, list[float]] = {}
                    for f in c_faults:
                        by_feat.setdefault(f.feature, []).append(f.diff)
                    for feat, diffs in sorted(by_feat.items()):
                        n_batches = len(diffs)
                        max_diff = max(abs(d) for d in diffs)
                        print(
                            f"    [COL] feature {feat}: {n_batches} batch(es) affected, max diff={max_diff:.3e}"
                        )
                elif r_faults:
                    print(
                        "    [COL] not localised — save with --weights to enable col check"
                    )
            else:
                print(f"\033[92m{layer_name}\033[0m")

        print()
        if faulty_count > 0:
            print(f"\033[91mFaults in {faulty_count}/{len(self.wrapped)} layers\033[0m")
        else:
            print(f"\033[92mNo faults ({len(self.wrapped)} layers checked)\033[0m")

    def get_values(self) -> dict:
        """Return results as a dict (same schema as CheckOne for easy comparison)."""
        faults = self.get_faults()
        col_faults = [f for f in faults if f.fault_type == "col_check"]
        row_faults = [f for f in faults if f.fault_type == "row_check"]

        return {
            "method": self.name,
            "layers_checked": len(self.wrapped),
            "weight_faults": len({(f.layer, f.feature) for f in col_faults}),
            "row_check_failures": len(row_faults),
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
