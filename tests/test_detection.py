"""Test ABFT checksum detection on linear layers."""

import torch
import torch.nn as nn
import pytest

from detection import CheckOne
from detection.checkone import _Wrapper
from core.bits import flip_bit


class TestWrapper:
    """Tests for the CheckOne _Wrapper class."""

    def test_output_matches_original(self):
        """Wrapper output should match original Linear layer output."""
        torch.manual_seed(42)
        original = nn.Linear(192, 768, bias=False)
        x = torch.randn(2, 10, 192)

        with torch.no_grad():
            expected = original(x).clone()

        wrapper = _Wrapper(original, "test")

        with torch.no_grad():
            actual = wrapper(x)

        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_detects_fault_in_correct_feature(self):
        """Wrapper should detect fault in the correct output feature."""
        torch.manual_seed(42)
        linear = nn.Linear(192, 768, bias=False)
        x = torch.randn(2, 10, 192)

        wrapper = _Wrapper(linear, "test_layer")

        # Inject fault
        row, col = 100, 50
        original_val = linear.weight.data[row, col].clone()
        corrupted, bit, _, _ = flip_bit(linear.weight.data[row, col], bit=25)
        linear.weight.data[row, col] = corrupted

        with torch.no_grad():
            wrapper(x)

        faults = wrapper.detect()
        faulty_features = {f[1] for f in faults}

        assert row in faulty_features, f"Expected fault in feature {row}, got {faulty_features}"

    def test_no_false_positives_without_fault(self):
        """Wrapper should not detect faults when none are injected."""
        torch.manual_seed(42)
        linear = nn.Linear(192, 768, bias=False)
        x = torch.randn(2, 10, 192)

        wrapper = _Wrapper(linear, "test_layer")

        with torch.no_grad():
            wrapper(x)

        faults = wrapper.detect()
        assert len(faults) == 0, f"Expected no faults, got {len(faults)}"


class TestCheckOne:
    """Tests for the CheckOne detector class."""

    def test_wraps_all_linear_layers(self):
        """CheckOne should wrap all linear layers in the model."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        class ModelWrapper:
            def __init__(self):
                self.net = model
                self.name = "test_model"

        detector = CheckOne(ModelWrapper(), layers="all")
        assert len(detector.wrapped) == 2

    def test_detects_fault_in_model(self):
        """CheckOne should detect faults in wrapped model."""

        class SimpleViTBlock(nn.Module):
            def __init__(self, dim=192):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim * 4, bias=False)
                self.fc2 = nn.Linear(dim * 4, dim, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        class SimpleViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([SimpleViTBlock() for _ in range(2)])

            def forward(self, x):
                for block in self.blocks:
                    x = x + block(x)
                return x

        class ModelWrapper:
            def __init__(self):
                self.net = SimpleViT()
                self.name = "simple_vit"

        model = ModelWrapper()
        x = torch.randn(2, 10, 192)

        detector = CheckOne(model, layers="all")

        # Inject fault into blocks.0.fc1
        fc1 = model.net.blocks[0].fc1
        row, col = 200, 100
        corrupted, _, _, _ = flip_bit(fc1.weight.data[row, col], bit=28)
        fc1.weight.data[row, col] = corrupted

        with torch.no_grad():
            model.net(x)

        faults = detector.get_faults()
        faulty_layers = {f.layer for f in faults}

        assert "blocks.0.fc1" in faulty_layers

    def test_get_values_returns_dict(self):
        """get_values should return fault info as dict."""
        model = nn.Sequential(nn.Linear(64, 128, bias=False))

        class ModelWrapper:
            def __init__(self):
                self.net = model
                self.name = "test_model"

        detector = CheckOne(ModelWrapper(), layers="all")

        # Inject fault
        model[0].weight.data[50, 30] *= 100

        x = torch.randn(4, 8, 64)
        with torch.no_grad():
            model(x)

        values = detector.get_values()
        assert isinstance(values, dict)
        assert "layers_checked" in values
        assert "faults_detected" in values
        assert "method" in values
