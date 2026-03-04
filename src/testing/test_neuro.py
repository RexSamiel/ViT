import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st

from src.core.fault_detection.neuron import NeuroChecker

LAYER_DIMS = [
    (192, 576),
    (192, 192),
    (192, 768),
    (768, 192),
]


@st.composite
def vit_linear_input(draw):
    in_features, out_features = draw(st.sampled_from(LAYER_DIMS))

    batch_size = draw(st.integers(min_value=1, max_value=4))
    seq_len = draw(st.sampled_from([197, 50, 14]))

    has_bias = draw(st.booleans())

    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    return {
        "in_features": in_features,
        "out_features": out_features,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "has_bias": has_bias,
        "seed": seed,
    }


@given(params=vit_linear_input())
def test_neurochecker_output(params):
    torch.manual_seed(params["seed"])

    original = nn.Linear(
        params["in_features"],
        params["out_features"],
        bias=params["has_bias"],
    )

    x = torch.randn(params["batch_size"], params["seq_len"], params["in_features"])

    with torch.no_grad():
        expected = original(x).clone()

    checker = NeuroChecker(original)

    with torch.no_grad():
        actual = checker(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6), (
        f"max_diff = {(actual - expected).abs().max().item():.2e}"
    )


if __name__ == "__main__":
    print("Running hypothesis test for NeuroChecker...")
    test_neurochecker_output()
    print("PASSED (100 examples)")
