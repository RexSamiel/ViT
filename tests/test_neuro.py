"""Hypothesis tests for NeuroChecker."""

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st

from vit_fault.detection.checker import NeuroChecker

# ViT layer dimensions (in_features, out_features)
LAYER_DIMS = [
    (192, 576),   # fc1 in vit_tiny MLP
    (192, 192),   # proj in vit_tiny attention
    (192, 768),   # qkv (simplified, actual is 192 -> 576)
    (768, 192),   # fc2 in vit_tiny MLP
]


@st.composite
def vit_linear_input(draw):
    """Generate realistic ViT-like linear layer inputs."""
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
@settings(max_examples=100)
def test_neurochecker_output_matches_original(params):
    """NeuroChecker output should match original Linear layer output."""
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


@given(params=vit_linear_input())
@settings(max_examples=50)
def test_neurochecker_stores_values(params):
    """NeuroChecker should store checker and expected values after forward."""
    torch.manual_seed(params["seed"])

    original = nn.Linear(
        params["in_features"],
        params["out_features"],
        bias=params["has_bias"],
    )
    x = torch.randn(params["batch_size"], params["seq_len"], params["in_features"])

    checker = NeuroChecker(original)

    # Before forward, values should be 0
    assert checker.checker_val == 0.0
    assert checker.expected_val == 0.0

    with torch.no_grad():
        _ = checker(x)

    # After forward, values should be set
    assert isinstance(checker.checker_val, float)
    assert isinstance(checker.expected_val, float)
    assert isinstance(checker.rel_diff, float)


if __name__ == "__main__":
    print("Running hypothesis tests for NeuroChecker...")
    test_neurochecker_output_matches_original()
    print("test_neurochecker_output_matches_original: PASSED")
    test_neurochecker_stores_values()
    print("test_neurochecker_stores_values: PASSED")
