import torch.nn as nn
import torch
from src.core.library.layers import get_linear_layers
from src.core.fault_detection.neuron import (
    CheckerType,
    wrap_layer,
    unwrap_layer,
)
from src.core.fault_detection.save_and_load import (
    get_or_compute_checker_weights,
    NeuroWeights,
    ChecksumWeights,
)


class FaultDetector:
    """Manages checker neurons across linear layers."""

    def __init__(
        self, model: nn.Module, method: str = "checksum", threshold: float = 1e-6
    ):
        """
        Args:
            model: The model to protect
            method: "neuro" for mean-based, "checksum" for sum-based ABFT
            threshold: Detection threshold for flagging faults
        """
        self.model = model
        self.method = method
        self.threshold = threshold
        self.wrapped: dict[str, CheckerType] = {}
        self.preloaded_weights: dict | None = None

    def load_weights(self, model_key: str, force_recompute: bool = False):
        """Load or compute checker weights from disk.

        Args:
            model_key: Model identifier (e.g., "vit_tiny")
            force_recompute: If True, recompute even if cached on disk
        """
        self.preloaded_weights = get_or_compute_checker_weights(
            self.model, model_key, self.method, force_recompute
        )
        print(f"Loaded {self.method} weights for {len(self.preloaded_weights)} layers")

    def apply(self, layer_filter: str = "all") -> list[str]:
        """Wrap layers with checker neurons."""
        layers = get_linear_layers(self.model)
        names = []

        for name in layers.keys():
            if layer_filter != "all" and layer_filter not in name:
                continue

            if self.preloaded_weights and name in self.preloaded_weights:
                w = self.preloaded_weights[name]
                if isinstance(w, NeuroWeights):
                    layer_weights = {
                        "checker_row": w.checker_row,
                        "checker_bias": w.checker_bias,
                    }
                elif isinstance(w, ChecksumWeights):
                    layer_weights = {
                        "col_sums": w.col_sums,
                        "row_sums": w.row_sums,
                        "total_sum": w.total_sum,
                        "bias_sum": w.bias_sum,
                    }

            self.wrapped[name] = wrap_layer(
                self.model, name, self.method, layer_weights
            )
            names.append(name)

        source = "preloaded" if self.preloaded_weights else "computed"
        print(
            f"Applied {self.method} checkers to {len(names)} layers ({source} weights)"
        )
        return names

    def remove(self):
        """Remove all wrappers."""
        for name in list(self.wrapped.keys()):
            unwrap_layer(self.model, name)
        self.wrapped.clear()

    def print_values(self, detailed: bool = False, relative: bool = True):
        """Print checker neuron value vs expected value for each layer.

        Args:
            detailed: If True and using checksum method, show per-row/col diffs
            relative: If True, use relative difference (diff/expected) for threshold
        """
        print()

        if self.method == "checksum":
            self._print_checksum_values(detailed)
        else:
            self._print_neuro_values(relative)

    def _print_neuro_values(self, relative: bool):
        """Print values for NeuroChecker (mean-based)."""
        header = f"{'Layer':<30} {'Checker':>12} {'Expected':>12} {'Diff':>12}"
        if relative:
            header += f" {'RelDiff':>12}"
        print(header)
        print("-" * (70 + (14 if relative else 0)))

        for name, wrapper in self.wrapped.items():
            c = wrapper.checker_val
            e = wrapper.expected_val
            diff = c - e

            rel_diff = abs(diff / e) if abs(e) > 1e-10 else abs(diff)

            line = f"{name:<30} {c:>12.6f} {e:>12.6f} {diff:>12.6f}"
            if relative:
                line += f" {rel_diff:>12.2e}"

            is_fault = (
                rel_diff > self.threshold if relative else abs(diff) > self.threshold
            )
            if is_fault:
                line += "  <- Fault"

            print(line)

    def _print_checksum_values(self, detailed: bool):
        """Print classical matrix checksum verification results.

        ABFT verifies: actual_output.sum() == expected_output.sum()
        where expected = x @ clean_col_sums + clean_bias_sum
        """
        print("ABFT OUTPUT CHECKSUM VERIFICATION")
        print("=" * 60)

        total_faults = 0

        for name, wrapper in self.wrapped.items():
            print(f"\n[{name}]")

            # Step 1: Output checksum verification (the actual ABFT check)
            output_diff = wrapper.output_checksum_diff
            output_ok = output_diff <= self.threshold

            if output_ok:
                print(f"  Output checksum: PASS (max_diff = {output_diff:.2e})")
                print("  Status: NO FAULT DETECTED")
                continue

            print(f"  Output checksum: FAIL (max_diff = {output_diff:.2e})")
            print(f"  Status: FAULT DETECTED")
            total_faults += 1

            # Step 2: Use weight checksums to locate the fault
            row_diffs = wrapper.row_diffs
            col_diffs = wrapper.col_diffs

            row_fault_mask = torch.abs(row_diffs) > self.threshold
            col_fault_mask = torch.abs(col_diffs) > self.threshold

            faulty_rows = torch.where(row_fault_mask)[0]
            faulty_cols = torch.where(col_fault_mask)[0]

            n_row_faults = len(faulty_rows)
            n_col_faults = len(faulty_cols)

            # Show fault localization from weight checksums
            if n_row_faults > 0:
                print(f"  Faulty rows (output neurons): {n_row_faults}")
                for idx in faulty_rows[:10]:
                    diff = row_diffs[idx].item()
                    print(f"    row[{idx.item()}]: diff = {diff:.6e}")
                if n_row_faults > 10:
                    print(f"    ... and {n_row_faults - 10} more")

            if n_col_faults > 0:
                print(f"  Faulty cols (input connections): {n_col_faults}")
                for idx in faulty_cols[:10]:
                    diff = col_diffs[idx].item()
                    print(f"    col[{idx.item()}]: diff = {diff:.6e}")
                if n_col_faults > 10:
                    print(f"    ... and {n_col_faults - 10} more")

            # Fault coordinates (intersection)
            if n_row_faults > 0 and n_col_faults > 0:
                print(f"  Fault location in W[row, col]:")
                count = 0
                for r in faulty_rows:
                    if count >= 10:
                        break
                    for c in faulty_cols:
                        if count >= 10:
                            break
                        print(f"    W[{r.item()}, {c.item()}]")
                        count += 1
                total_coords = n_row_faults * n_col_faults
                if total_coords > 10:
                    print(f"    ... and {total_coords - 10} more coordinates")

            # Check bias faults
            if (
                wrapper.bias_diff is not None
                and abs(wrapper.bias_diff) > self.threshold
            ):
                print(f"  Bias fault: diff = {wrapper.bias_diff:.6e}")

        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {total_faults} layer(s) with faults detected")

    def get_fault_locations(self) -> dict[str, dict]:
        """Get detailed fault location info for each layer.

        Returns:
            Dict mapping layer name to fault info with coordinates
        """
        results = {}
        for name, wrapper in self.wrapped.items():
            if hasattr(wrapper, "output_checksum_diff"):
                # Checksum method: use output checksum for detection
                has_fault = wrapper.output_checksum_diff > self.threshold

                row_diffs = wrapper.row_diffs
                col_diffs = wrapper.col_diffs

                faulty_rows = torch.where(torch.abs(row_diffs) > self.threshold)[0]
                faulty_cols = torch.where(torch.abs(col_diffs) > self.threshold)[0]

                fault_coords = [
                    (r.item(), c.item()) for r in faulty_rows for c in faulty_cols
                ]

                results[name] = {
                    "has_fault": has_fault,
                    "output_checksum_diff": wrapper.output_checksum_diff,
                    "fault_coords": fault_coords,
                    "faulty_rows": faulty_rows.tolist(),
                    "faulty_cols": faulty_cols.tolist(),
                    "row_diffs": {i.item(): row_diffs[i].item() for i in faulty_rows},
                    "col_diffs": {i.item(): col_diffs[i].item() for i in faulty_cols},
                }
            else:
                # Neuro method
                diff = wrapper.checker_val - wrapper.expected_val
                results[name] = {
                    "diff": diff,
                    "is_fault": abs(diff) > self.threshold,
                }
        return results
