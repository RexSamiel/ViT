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

    atol:     float = 1e-5   # row-check threshold
    rtol:     float = 0.0
    atol_col: float = 1e-2   # col-check threshold (separate: different BLAS noise profile)
    rtol_col: float = 0.0
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

        x_token_sums = x.sum(dim=1, keepdim=True)
        x_ext = torch.cat([x, x_token_sums], dim=1)

        out_ext = F.linear(x_ext, self.weights_ext)

        out = out_ext[:, :N, : self.C_out]
        golden_row = out_ext[:, :N, self.C_out]
        actual_col = out_ext[:, N, : self.C_out]

        if self.original.bias is not None:
            out = out + self.original.bias

        actual_row = out_ext[:, :N, : self.C_out].sum(dim=-1)
        token_out_sums = out_ext[:, :N, : self.C_out].sum(dim=1)

        if getattr(self, "_calibrating_threshold", False):
            self.row_faults = []
            self.col_faults = []
            self.threshold_calibrate_update(actual_row, golden_row, actual_col, x_token_sums, token_out_sums)
        else:
            self.row_faults = self._detect_rows(actual_row, golden_row)
            if self.correction is not None and self.row_faults and self.clean_weights is not None:
                golden_col = F.linear(x_token_sums.squeeze(1), self.clean_weights)
            else:
                golden_col = token_out_sums
            self.col_faults = self._detect_cols(actual_col, golden_col)

        if self.correction is not None:
            out = self._correct(out, x)

        if extra_shape is not None:
            out = out.view(B, *extra_shape, -1)

        if squeeze:
            out = out.squeeze(1)

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
        idx_cpu = indices.cpu()
        vals_cpu = diffs[indices[:, 0], indices[:, 1]].cpu()
        return list(zip(idx_cpu[:, 0].tolist(), idx_cpu[:, 1].tolist(), vals_cpu.tolist()))

    def _detect_cols(
        self,
        actual_col: torch.Tensor,
        golden_col: torch.Tensor,
    ) -> list[tuple[int, int, float]]:
        """Col check: compare actual col sums vs golden.

        Standard path: golden_col = token_out_sums (free from extended matmul).
        Correction path: golden_col = F.linear(x_token_sums, clean_weights) for
        persistent weight fault localisation.
        """
        mask = ~torch.isclose(actual_col, golden_col, atol=self.atol_col, rtol=self.rtol_col)
        if not mask.any():
            return []
        diffs = actual_col - golden_col
        indices = mask.nonzero(as_tuple=False)
        idx_cpu = indices.cpu()
        vals_cpu = diffs[indices[:, 0], indices[:, 1]].cpu()
        return list(zip(idx_cpu[:, 0].tolist(), idx_cpu[:, 1].tolist(), vals_cpu.tolist()))

    def _correct(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        faulty_feats = {feat for _, feat, _ in self.col_faults}
        if not faulty_feats:
            return out

        # Build per-batch mapping of token -> row residual (error magnitude)
        row_fault_map: dict[int, dict[int, float]] = {}
        for b, tok, diff in self.row_faults:
            row_fault_map.setdefault(b, {})[tok] = diff

        # Build per-batch col fault map: {b: {feat: diff}}
        col_fault_map: dict[int, dict[int, float]] = {}
        for b, feat, diff in self.col_faults:
            col_fault_map.setdefault(b, {})[feat] = diff

        if self.correction == "zero":
            for b, feat_diffs in col_fault_map.items():
                tok_diffs = row_fault_map.get(b, {})
                for feat in feat_diffs:
                    for tok in tok_diffs:
                        out[b, tok, feat] = 0.0

        elif self.correction == "correct":
            # ApproxABFT algebraic correction — no weights required.
            # For each fault at (b, tok, feat):
            #   - Alone in its column (1 faulty token):  subtract col residual
            #   - Alone in its row    (1 faulty feature): subtract row residual
            #   - Multiple faults in both dimensions:    zero the element
            for b, feat_diffs in col_fault_map.items():
                tok_diffs = row_fault_map.get(b, {})
                n_faulty_toks  = len(tok_diffs)
                n_faulty_feats = len(feat_diffs)
                for feat, col_diff in feat_diffs.items():
                    for tok, row_diff in tok_diffs.items():
                        if n_faulty_toks == 1:
                            out[b, tok, feat] -= col_diff
                        elif n_faulty_feats == 1:
                            out[b, tok, feat] -= row_diff
                        else:
                            out[b, tok, feat] = 0.0
        return out

    def threshold_calibrate_start(self):
        """Start collecting clean-condition noise for both row and col checks."""
        self._cal_abs_buf:     list[torch.Tensor] = []
        self._cal_abs_col_buf: list[torch.Tensor] = []
        self._calibrating_threshold = True

    def threshold_calibrate_update(
        self, actual_row: torch.Tensor, golden_row: torch.Tensor,
        actual_col: torch.Tensor, x_token_sums: torch.Tensor, token_out_sums: torch.Tensor,
    ):
        """Accumulate per-batch noise for row and col checks — stays on GPU."""
        self._cal_abs_buf.append((actual_row - golden_row).abs().max())
        self._cal_abs_col_buf.append((actual_col - token_out_sums).abs().max())

    def threshold_calibrate_end(self):
        """Set atol for row and col checks from maximum observed noise during clean calibration run."""
        if not self._cal_abs_buf:
            return
        self.atol = float(torch.stack(self._cal_abs_buf).max())
        if self._cal_abs_col_buf:
            self.atol_col = float(torch.stack(self._cal_abs_col_buf).max())
        self._calibrating_threshold = False

    def get_baseline(self, include_weights: bool = False) -> dict:
        """Return baseline data for saving.

        With include_weights: full weight matrix (needed for col-check and correction).
        """
        data: dict = {"atol": self.atol, "rtol": self.rtol,
                      "atol_col": self.atol_col, "rtol_col": self.rtol_col}
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
        if "atol_col" in data:
            self.atol_col = data["atol_col"]
        if "rtol_col" in data:
            self.rtol_col = data["rtol_col"]
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

    def calibrate_threshold(self, model=None, max_batches: int | None = None):
        """Calibrate detection threshold from clean data.

        Runs inference on clean model, measures floating-point noise in the row and
        col check signals, then sets atol = maximum observed noise for each layer.

        Args:
            model: Model instance with dataloader. If None, uses model from __init__.
            max_batches: Batches to use (None = full dataset, recommended).
        """
        if model is None:
            model = self._model_ref

        if not hasattr(model, "dataloader"):
            print("Warning: Model has no dataloader, skipping threshold calibration")
            return 0

        for w in self.wrapped.values():
            w.threshold_calibrate_start()

        limit_str = (
            f"up to {max_batches} batches"
            if max_batches is not None
            else "full dataset"
        )
        print(f"Calibrating threshold on {limit_str}...")

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

        for w in self.wrapped.values():
            w.threshold_calibrate_end()

        print(f"  Calibrated on {total_samples} samples ({batch_count} batches)")
        for name, w in self.wrapped.items():
            if getattr(w, "_cal_abs_buf", []):
                print(f"  {name}: row atol={w.atol:.2e}  col atol={w.atol_col:.2e}")
        return total_samples

    def remove(self):
        """Restore original layers."""
        unwrap_layers(self.model, self.wrapped)
        self.wrapped.clear()

    def _cal_key(self) -> str:
        """Calibration file key: 'checksum_zero' for zeroing, 'checksum' for detection."""
        return "checksum_zero" if self.correction is not None else "checksum"

    def save(
        self,
        path: Path | str | None = None,
        include_weights: bool = False,
        save_calibration: bool = True,
    ):
        """Save calibration and/or weights.

        save_calibration=True  writes data/{model}/calibration/checksum.pt
                               or checksum_zero.pt depending on correction mode.
        include_weights=True   writes data/{model}/weights/checksum.pt.
        Pass save_calibration=False when saving weights only — avoids overwriting
        a previously calibrated file with default (uncalibrated) values.
        """
        if save_calibration:
            p = Path(path) if path else calibration_path(self.model_name, self._cal_key())
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {
                name: w.get_baseline(include_weights=False)
                for name, w in self.wrapped.items()
            }
            torch.save(data, p)
            print(
                f"Saved checksum calibration to {p} ({p.stat().st_size / 1024**2:.1f} MB)"
            )

        if include_weights:
            wp = weights_path(self.model_name, "checksum")
            wp.parent.mkdir(parents=True, exist_ok=True)
            wdata = {
                name: w.get_baseline(include_weights=True)
                for name, w in self.wrapped.items()
            }
            torch.save(wdata, wp)
            print(
                f"Saved checksum weights to {wp} ({wp.stat().st_size / 1024**2:.1f} MB)"
            )

    def load(self, path: Path | str | None = None, verbose: bool = False) -> bool:
        """Load calibration from data/{model}/calibration/checksum.pt or checksum_zero.pt.
        Zeroing mode loads checksum_zero.pt; detection-only loads checksum.pt.
        Also loads weights from data/{model}/weights/checksum.pt if correction is set.
        Raises RuntimeError if calibration or weights files are missing.
        """
        if path:
            p = Path(path)
        else:
            p = calibration_path(self.model_name, self._cal_key())
            if not p.exists() and self.correction is not None:
                fallback = calibration_path(self.model_name, "checksum")
                if fallback.exists():
                    import warnings
                    warnings.warn(
                        f"Checksum: zeroing calibration not found at {p}. "
                        f"Falling back to detection calibration at {fallback}. "
                        f"Run: save --threshold  with correction mode to get accurate zeroing thresholds."
                    )
                    p = fallback
        if not p.exists():
            raise RuntimeError(
                f"Checksum: no calibration file found at {p}.\n"
                f"  Run: save --threshold  to calibrate first."
            )

        data = torch.load(p, weights_only=False)
        for name, wrapper in self.wrapped.items():
            if name in data:
                wrapper.set_baseline(data[name])
        print(f"Loaded checksum calibration from {p} ({len(data)} layers)")

        if self.correction is not None:
            wp = weights_path(self.model_name, "checksum")
            if not wp.exists():
                raise RuntimeError(
                    f"Checksum: no weights file found at {wp}.\n"
                    f"  Run: save --weights  to save weights for column check."
                )
            wdata = torch.load(wp, weights_only=False)
            for name, wrapper in self.wrapped.items():
                if name in wdata and "weights" in wdata[name]:
                    wrapper.set_baseline(
                        {
                            "weights": wdata[name]["weights"],
                            "bias": wdata[name].get("bias"),
                        }
                    )
            print(f"Loaded checksum weights from {wp}")

        if verbose:
            for name, w in self.wrapped.items():
                if name in data:
                    print(f"  {name}: row atol={w.atol:.2e} rtol={w.rtol:.2e}  col atol={w.atol_col:.2e} rtol={w.rtol_col:.2e}")

        return True

    def reset(self):
        """Clear per-run fault lists in all wrappers (call before each run)."""
        for w in self.wrapped.values():
            w.row_faults = []
            w.col_faults = []

    def start_threshold_calibration(self):
        """Activate threshold calibration hooks (fire on every subsequent net(images))."""
        for w in self.wrapped.values():
            w.threshold_calibrate_start()

    def end_threshold_calibration(self):
        """Finalise threshold calibration and print per-layer tolerances."""
        for w in self.wrapped.values():
            w.threshold_calibrate_end()
        print("Checksum threshold calibration complete:")
        for name, w in self.wrapped.items():
            if getattr(w, "_cal_abs_buf", []):
                print(f"  {name}: row atol={w.atol:.2e}  col atol={w.atol_col:.2e}")

    def print_summary(self, layer_stats: dict[str, dict[str, int]]):
        """Print aggregate detection table across all runs."""
        if not layer_stats:
            return
        col = max(len(l) for l in layer_stats) + 2
        print(
            f"\nDetection Summary ({self.name}):\n"
            f"  {'Layer':<{col}}  {'Injected':>8}  {'Detected':>8}  {'Rate':>7}  {'False+':>6}\n"
            f"  {'-' * col}  {'--------':>8}  {'--------':>8}  {'-------':>7}  {'------':>6}"
        )
        for layer in sorted(layer_stats):
            s = layer_stats[layer]
            inj = s["injected"]
            det = s["detected"]
            fp = s["false_positive"]
            rate = f"{100 * det / inj:.1f}%" if inj else "  n/a"
            print(f"  {layer:<{col}}  {inj:>8}  {det:>8}  {rate:>7}  {fp:>6}")

        total_inj  = sum(s["injected"]       for s in layer_stats.values())
        total_det  = sum(s["detected"]       for s in layer_stats.values())
        total_fp   = sum(s["false_positive"] for s in layer_stats.values())
        total_rate = f"{100 * total_det / total_inj:.1f}%" if total_inj else "  n/a"
        print(
            f"  {'-' * col}  {'--------':>8}  {'--------':>8}  {'-------':>7}  {'------':>6}\n"
            f"  {'TOTAL':<{col}}  {total_inj:>8}  {total_det:>8}  {total_rate:>7}  {total_fp:>6}"
        )

    def print_timing_summary(self, all_runs: list[dict]):
        """Print timing: total and per-sample mean ± std across runs."""
        timed = [r for r in all_runs if r.get("times_ms")]
        if not timed:
            return
        per_run_ms  = [sum(r["times_ms"]) for r in timed]
        per_run_mpb = [ms / len(r["times_ms"]) for ms, r in zip(per_run_ms, timed)]
        total_ms = sum(per_run_ms)
        avg_mpb  = sum(per_run_mpb) / len(per_run_mpb)
        std_mpb  = (
            (sum((v - avg_mpb) ** 2 for v in per_run_mpb) / (len(per_run_mpb) - 1)) ** 0.5
            if len(per_run_mpb) > 1 else 0.0
        )
        print(
            f"\nDetection Timing — {self.name} ({len(timed)} runs):\n"
            f"  Total:         {total_ms:.1f} ms  ({total_ms / 1000:.2f} s)\n"
            f"  Avg per batch: {avg_mpb:.4f} ms  ±{std_mpb:.4f} ms std"
        )

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
            "weight_faults": len({f.layer for f in faults}),
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
