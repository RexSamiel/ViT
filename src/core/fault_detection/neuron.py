import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


CHECKER_WEIGHTS_DIR = Path(__file__).parent.parent.parent / "data" / "checker_weights"


class NeuroChecker(nn.Module):
    """Linear layer with mean checker neuron."""

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


LinearChecker = NeuroChecker


def wrap_layer(
    model: nn.Module,
    name: str,
    preloaded_weights: dict | None = None,
) -> NeuroChecker:
    """Replace a linear layer with a NeuroChecker wrapper.

    Args:
        model: The model containing the layer
        name: Dot-separated path to the layer
        preloaded_weights: Optional dict with pre-computed weights for this layer
    """
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)

    orig = getattr(parent, parts[-1])

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
    if isinstance(wrapped, NeuroChecker):
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
