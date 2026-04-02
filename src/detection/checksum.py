from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import DETECTION_DIR
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

    atol = 1e-2
    rtol = 1e-3

    def __init__(self, original: nn.Linear, name: str):
        super().__init__()
        self.original = original
        self.name = name

        self.w_col_sums: torch.Tensor = original.weight.data.sum(dim=0).clone()

        self.clean_weights: torch.Tensor | None = None
        self.clean_bias: torch.Tensor | None = None

        self.row_faults: list[tuple[int, int, float]] = []
        self.col_faults: list[tuple[int, int, float]] = []

        self.correction: str | None = None

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

        B, N, C_in = x.shape
        C_out = self.original.weight.shape[0]

        W_ext = torch.cat(
            [
                self.original.weight,
                self.w_col_sums.unsqueeze(0).to(device=x.device, dtype=x.dtype),
            ],
            dim=0,
        )

        x_token_sums = x.sum(dim=1, keepdim=True)
        x_ext = torch.cat([x, x_token_sums], dim=1)

        out_ext = F.linear(x_ext, W_ext)

        out = out_ext[:, :N, :C_out]
        golden_row = out_ext[:, :N, C_out]
        actual_col = out_ext[:, N, :C_out]

        if self.original.bias is not None:
            out = out + self.original.bias

        actual_row = out_ext[:, :N, :C_out].sum(dim=-1)  # [B, N]

        self.row_faults = self._detect_rows(actual_row, golden_row)
        self.col_faults = self._detect_cols(actual_col, x_token_sums)

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
        diffs = actual_row - golden_row  # [B, N]
        threshold = self.atol + self.rtol * golden_row.abs()
        mask = diffs.abs() > threshold

        faults = []
        for b in range(actual_row.shape[0]):
            for tok in mask[b].nonzero(as_tuple=False).view(-1).tolist():
                if isinstance(tok, int):
                    faults.append((b, tok, diffs[b, tok].item()))
        return faults

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

        golden_col = F.linear(
            x_token_sums.squeeze(1),
            self.clean_weights.to(dtype=x_token_sums.dtype),
        )

        diffs = actual_col - golden_col  # [B, C_out]
        mask = diffs.abs() > self.atol

        faults = []
        for b in range(actual_col.shape[0]):
            for feat in mask[b].nonzero(as_tuple=False).view(-1).tolist():
                if isinstance(feat, int):
                    faults.append((b, feat, diffs[b, feat].item()))
        return faults

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
        else:  # "rerun"
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
            diff_row = self.original.weight[feat] - self.clean_weights[feat]  # [C_in]
            j = int(diff_row.abs().argmax().item())
            delta = diff_row[j].item()
            out[:, :, feat] -= x[:, :, j] * delta
        return out

    def get_baseline(self, include_weights: bool = False) -> dict:
        """Return baseline data for saving."""
        data: dict = {
            "w_col_sums": self.w_col_sums.cpu(),
            "weight_sums": self.original.weight.data.sum(dim=1).cpu(),  # for checkone
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
        self.w_col_sums = data["w_col_sums"].to(device)

        if "weights" in data:
            self.clean_weights = data["weights"].to(device)
            if data.get("bias") is not None:
                self.clean_bias = data["bias"].to(device)


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

    def remove(self):
        """Restore original layers."""
        unwrap_layers(self.model, self.wrapped)
        self.wrapped.clear()

    def save(self, path: Path | str | None = None, include_weights: bool = False):
        """Save baselines for all wrapped layers.

        Args:
            path: Save path (default: data/detection/checksum_{model}.pt).
            include_weights: Save full weight matrix for col-check and rerun correction.
        """
        if path is None:
            DETECTION_DIR.mkdir(parents=True, exist_ok=True)
            path = DETECTION_DIR / f"baseline_{self.model_name}.pt"

        data = {
            name: w.get_baseline(include_weights=include_weights)
            for name, w in self.wrapped.items()
        }

        torch.save(data, path)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"Saved baseline to {path} ({size_mb:.1f} MB)")

    def load(self, path: Path | str | None = None) -> bool:
        """Load baselines.  Returns True if file found."""
        if path is None:
            path = DETECTION_DIR / f"baseline_{self.model_name}.pt"

        if not Path(path).exists():
            return False

        data = torch.load(path, weights_only=True)
        for name, wrapper in self.wrapped.items():
            if name in data:
                wrapper.set_baseline(data[name])

        has_weights = any(
            self.wrapped[n].clean_weights is not None for n in self.wrapped if n in data
        )
        print(f"Loaded baseline ({len(data)} layers, weights={has_weights})")

        if self.correction in ("rerun", "zero") and not has_weights:
            print(
                "Warning: correction enabled but no weights in baseline. "
                "Col-check and rerun correction unavailable — save with include_weights=True."
            )

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
