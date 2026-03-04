import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


CHECKER_WEIGHTS_DIR = Path(__file__).parent.parent.parent / "data" / "checker_weights"


class NeuroChecker(nn.Module):
    """Linear layer with mean-based checker neuron."""

    def __init__(
        self,
        original: nn.Linear,
        checker_row: torch.Tensor | None = None,
        checker_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.original = original
        self.out_features = original.out_features

        if checker_row is not None:
            self.checker_row = (
                checker_row.unsqueeze(0) if checker_row.dim() == 1 else checker_row
            )
        else:
            self.checker_row = original.weight.data.mean(dim=0, keepdim=True).clone()

        if checker_bias is not None:
            self.checker_bias = (
                checker_bias.unsqueeze(0) if checker_bias.dim() == 0 else checker_bias
            )
        elif original.bias is not None:
            self.checker_bias = original.bias.data.mean().unsqueeze(0).clone()
        else:
            self.checker_bias = None

        self.checker_val = 0.0
        self.expected_val = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extended_weight = torch.vstack([self.original.weight, self.checker_row])

        if self.original.bias is not None and self.checker_bias is not None:
            extended_bias = torch.cat([self.original.bias, self.checker_bias])
        else:
            extended_bias = None

        extended_out = F.linear(x, extended_weight, extended_bias)

        out = extended_out[..., : self.out_features]
        checker_out = extended_out[..., self.out_features]

        self.checker_val = checker_out.mean().item()
        self.expected_val = out.mean(dim=-1).mean().item()

        return out


class ChecksumChecker(nn.Module):
    """Classical matrix checksum ABFT for linear layer fault detection.

    For y = x @ W.T + b, ABFT verifies:
    - Output row checksum: y.sum(dim=-1) should equal x @ col_sums(W) + sum(b)
    - Output col checksum: y.sum(dim=0) should equal row_sums(W) @ x.sum(dim=0) + b * batch_size

    Weight sums are used to locate faults after detection.
    """

    def __init__(
        self,
        original: nn.Linear,
        col_sums: torch.Tensor | None = None,
        row_sums: torch.Tensor | None = None,
        total_sum: torch.Tensor | None = None,
        bias_sum: torch.Tensor | None = None,
    ):
        super().__init__()
        self.original = original
        self.out_features = original.out_features
        self.in_features = original.in_features

        W = original.weight.data

        self.clean_col_sums = col_sums if col_sums is not None else W.sum(dim=0).clone()
        self.clean_row_sums = row_sums if row_sums is not None else W.sum(dim=1).clone()
        self.clean_total = total_sum if total_sum is not None else W.sum().clone()

        if bias_sum is not None:
            self.clean_bias_sum = bias_sum
        elif original.bias is not None:
            self.clean_bias_sum = original.bias.data.sum().clone()
        else:
            self.clean_bias_sum = None

        self.output_checksum_diff = None
        self.row_diffs = None
        self.col_diffs = None
        self.total_diff = None
        self.bias_diff = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.original.weight, self.original.bias)
        self.original.forward(x)

        expected_row_sums = torch.matmul(x, self.clean_col_sums)
        if self.clean_bias_sum is not None:
            expected_row_sums = expected_row_sums + self.clean_bias_sum

        actual_row_sums = out.sum(dim=-1)

        checksum_diff = actual_row_sums - expected_row_sums
        self.output_checksum_diff = checksum_diff.abs().max().item()

        W = self.original.weight
        current_col_sums = W.sum(dim=0)
        current_row_sums = W.sum(dim=1)

        self.col_diffs = current_col_sums - self.clean_col_sums
        self.row_diffs = current_row_sums - self.clean_row_sums
        self.total_diff = (W.sum() - self.clean_total).item()

        if self.original.bias is not None and self.clean_bias_sum is not None:
            self.bias_diff = (self.original.bias.sum() - self.clean_bias_sum).item()
        else:
            self.bias_diff = None

        return out


LinearChecker = NeuroChecker
CheckerType = NeuroChecker | ChecksumChecker


def wrap_layer(
    model: nn.Module,
    name: str,
    method: str = "neuro",
    preloaded_weights: dict | None = None,
) -> CheckerType:
    """Replace a linear layer with a checker wrapper.

    Args:
        model: The model containing the layer
        name: Dot-separated path to the layer
        method: "neuro" for mean-based, "checksum" for sum-based ABFT
        preloaded_weights: Optional dict with pre-computed weights for this layer
    """
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)

    orig = getattr(parent, parts[-1])

    if method == "checksum":
        if preloaded_weights:
            wrapped = ChecksumChecker(
                orig,
                col_sums=preloaded_weights.get("col_sums"),
                row_sums=preloaded_weights.get("row_sums"),
                total_sum=preloaded_weights.get("total_sum"),
                bias_sum=preloaded_weights.get("bias_sum"),
            )
        else:
            wrapped = ChecksumChecker(orig)
    else:
        if preloaded_weights:
            wrapped = NeuroChecker(
                orig,
                checker_row=preloaded_weights.get("checker_row"),
                checker_bias=preloaded_weights.get("checker_bias"),
            )
        else:
            wrapped = NeuroChecker(orig)

    setattr(parent, parts[-1], wrapped)
    return wrapped


def unwrap_layer(model: nn.Module, name: str):
    """Restore original layer."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)

    wrapped = getattr(parent, parts[-1])
    if isinstance(wrapped, (NeuroChecker, ChecksumChecker)):
        setattr(parent, parts[-1], wrapped.original)


def save_checker_weights(model_name: str, weights: dict[str, torch.Tensor]):
    """Save precomputed checker weights to file."""
    CHECKER_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKER_WEIGHTS_DIR / f"{model_name}_checker.pt"
    torch.save(weights, path)
    print(f"Saved checker weights to {path}")


def load_checker_weights(model_name: str) -> dict[str, torch.Tensor] | None:
    """Load precomputed checker weights from file."""
    path = CHECKER_WEIGHTS_DIR / f"{model_name}_checker.pt"
    if path.exists():
        return torch.load(path)
    return None
