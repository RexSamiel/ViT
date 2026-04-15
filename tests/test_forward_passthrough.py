"""Test that CheckOne and Checksum wrappers produce identical outputs to the
original nn.Linear for arbitrary input shapes drawn from realistic transformer
dimension ranges, verified by Hypothesis."""

import torch
import torch.nn as nn
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


from detection.checkone import _Wrapper as CheckOneWrapper
from detection.checksum import _Wrapper as ChecksumWrapper


# Realistic transformer embedding dims (ViT-tiny/small/base, DeiT, Swin)
EMBED_DIMS = [96, 192, 384, 768]
# MLP hidden dims are typically 4× embed; attention projection dims equal embed
MLP_MULTIPLIERS = [1, 4]


def linear_shapes():
    """Strategy: (B, N, C_in, C_out, bias) drawn from realistic ViT sizes."""
    embed = st.sampled_from(EMBED_DIMS)
    multiplier = st.sampled_from(MLP_MULTIPLIERS)

    @st.composite
    def _shape(draw):
        c_in = draw(embed)
        mult = draw(multiplier)
        c_out = c_in * mult
        b = draw(st.integers(min_value=1, max_value=8))
        # Realistic token counts: ViT 197, Swin 49/196, or small values
        n = draw(st.sampled_from([1, 49, 64, 196, 197]))
        bias = draw(st.booleans())
        return b, n, c_in, c_out, bias

    return _shape()


@given(linear_shapes())
@settings(max_examples=40, deadline=5000)
def test_checkone_matches_linear(params):
    B, N, C_in, C_out, use_bias = params
    torch.manual_seed(0)
    linear = nn.Linear(C_in, C_out, bias=use_bias)
    x = torch.randn(B, N, C_in)

    with torch.no_grad():
        expected = linear(x).clone()

    wrapper = CheckOneWrapper(linear, "test")
    with torch.no_grad():
        actual = wrapper(x)

    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"Max diff: {(actual - expected).abs().max().item():.2e}  shape=({B},{N},{C_in},{C_out}) bias={use_bias}"


@given(linear_shapes())
@settings(max_examples=40, deadline=5000)
def test_checksum_matches_linear(params):
    B, N, C_in, C_out, use_bias = params
    torch.manual_seed(0)
    linear = nn.Linear(C_in, C_out, bias=use_bias)
    x = torch.randn(B, N, C_in)

    with torch.no_grad():
        expected = linear(x).clone()

    wrapper = ChecksumWrapper(linear, "test")
    with torch.no_grad():
        actual = wrapper(x)

    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"Max diff: {(actual - expected).abs().max().item():.2e}  shape=({B},{N},{C_in},{C_out}) bias={use_bias}"


@given(
    b=st.integers(min_value=1, max_value=8),
    c_in=st.sampled_from(EMBED_DIMS),
    use_bias=st.booleans(),
)
@settings(max_examples=20, deadline=5000)
def test_checkone_matches_linear_2d(b, c_in, use_bias):
    """2-D input (no token dim) passthrough."""
    torch.manual_seed(0)
    c_out = c_in * 4
    linear = nn.Linear(c_in, c_out, bias=use_bias)
    x = torch.randn(b, c_in)

    with torch.no_grad():
        expected = linear(x).clone()

    wrapper = CheckOneWrapper(linear, "test")
    with torch.no_grad():
        actual = wrapper(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"Max diff: {(actual - expected).abs().max().item():.2e}"


@given(
    b=st.integers(min_value=1, max_value=8),
    c_in=st.sampled_from(EMBED_DIMS),
    use_bias=st.booleans(),
)
@settings(max_examples=20, deadline=5000)
def test_checksum_matches_linear_2d(b, c_in, use_bias):
    """2-D input (no token dim) passthrough."""
    torch.manual_seed(0)
    c_out = c_in * 4
    linear = nn.Linear(c_in, c_out, bias=use_bias)
    x = torch.randn(b, c_in)

    with torch.no_grad():
        expected = linear(x).clone()

    wrapper = ChecksumWrapper(linear, "test")
    with torch.no_grad():
        actual = wrapper(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"Max diff: {(actual - expected).abs().max().item():.2e}"
