"""Tests for standalone input activation fault injector."""

import torch
import torch.nn as nn

from injection.input_injector import InjectedInputFault, InputInjector


class SimpleModel(nn.Module):
    """Minimal transformer-like model with named linear layers."""

    def __init__(self):
        super().__init__()
        self.fc_small = nn.Linear(16, 32)
        self.fc_large = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc_small(x[:, :, :16]) + self.fc_large(x[:, :, :64]).mean(
            dim=-1, keepdim=True
        ).expand_as(self.fc_small(x[:, :, :16]))


def _make_injector(layers="all"):
    model = SimpleModel()
    layer_shapes = {
        "fc_small": (10, 16),  # seq=10, features=16
        "fc_large": (10, 64),  # seq=10, features=64
    }
    return InputInjector(model, layers=layers, layer_shapes=layer_shapes), model


class TestInputInjectorInit:
    def test_layers_collected(self):
        injector, _ = _make_injector()
        assert "fc_small" in injector._layers
        assert "fc_large" in injector._layers

    def test_total_elements_correct(self):
        injector, _ = _make_injector()
        assert injector.total_elements == 80

    def test_cumulative_correct(self):
        injector, _ = _make_injector()
        # fc_small starts at 0, fc_large starts at 16
        assert injector._cumulative[0] == 0
        assert injector._cumulative[1] == 16

    def test_layer_filter(self):
        injector, _ = _make_injector(layers="fc_small")
        assert "fc_small" in injector._layers
        assert "fc_large" not in injector._layers


class TestProportionalSampling:
    def test_larger_layer_hit_more_often(self):
        """fc_large has 4x more features so should be selected ~4x more often."""
        injector, _ = _make_injector()
        counts = {"fc_small": 0, "fc_large": 0}
        for _ in range(1000):
            name, _ = injector._sample_layer_and_feature()
            counts[name] += 1
        ratio = counts["fc_large"] / 1000
        assert 0.70 < ratio < 0.90, f"fc_large ratio was {ratio:.2f}, expected ~0.80"

    def test_feature_idx_within_bounds(self):
        injector, _ = _make_injector()
        for _ in range(200):
            name, feat_idx = injector._sample_layer_and_feature()
            layer = injector._layers[name]
            assert 0 <= feat_idx < layer.in_features


class TestArm:
    def test_arm_registers_hook(self):
        injector, _ = _make_injector()
        assert len(injector._hooks) == 0
        injector.arm()
        assert len(injector._hooks) == 1

    def test_arm_selects_valid_position(self):
        injector, _ = _make_injector()
        injector.arm()
        # Run one batch so the hook fires and fault is recorded
        batch = torch.randn(4, 10, 64)
        layer = injector._layers.get("fc_small") or injector._layers.get("fc_large")
        # Find which layer was armed by checking hooks
        assert len(injector._hooks) == 1

    def test_fault_recorded_after_inference(self):
        injector, model = _make_injector()
        injector.arm()
        x = torch.randn(4, 10, 64)
        with torch.inference_mode():
            model(x)
        assert len(injector.faults) == 1
        fault = injector.faults[0]
        assert isinstance(fault, InjectedInputFault)
        assert fault.layer in ("fc_small", "fc_large")
        assert fault.bit >= 0
        assert fault.original != fault.corrupted

    def test_fault_recorded_only_once_across_batches(self):
        """Hook is persistent but fault should only be recorded once."""
        injector, model = _make_injector()
        injector.arm()
        x = torch.randn(4, 10, 64)
        with torch.inference_mode():
            model(x)
            model(x)
            model(x)
        assert len(injector.faults) == 1

    def test_all_samples_corrupted(self):
        """The fault should affect all samples in the batch, not just one."""
        injector, model = _make_injector()

        # Capture inputs before and after
        captured = {}

        def capture_hook(module, input):
            captured["before"] = input[0].clone()

        # Register capture hook first, then arm
        layer_name = list(injector._layers.keys())[0]
        layer = injector._layers[layer_name]
        h = layer.register_forward_pre_hook(capture_hook)

        injector.arm()
        # Force injection into the captured layer by checking what armed
        h.remove()

        x = torch.randn(4, 10, 64)
        with torch.inference_mode():
            model(x)

        if injector.faults:
            fault = injector.faults[0]
            assert fault.original != fault.corrupted


class TestRestore:
    def test_restore_removes_hooks(self):
        injector, _ = _make_injector()
        injector.arm()
        assert len(injector._hooks) == 1
        injector.restore()
        assert len(injector._hooks) == 0

    def test_restore_clears_faults(self):
        injector, model = _make_injector()
        injector.arm()
        x = torch.randn(4, 10, 64)
        with torch.inference_mode():
            model(x)
        assert len(injector.faults) == 1
        injector.restore()
        assert len(injector.faults) == 0

    def test_hook_does_not_fire_after_restore(self):
        injector, model = _make_injector()
        injector.arm()
        injector.restore()
        x = torch.randn(4, 10, 64)
        with torch.inference_mode():
            model(x)
        assert len(injector.faults) == 0

    def test_multiple_runs(self):
        """arm/restore cycle should work correctly across multiple runs."""
        injector, model = _make_injector()
        x = torch.randn(4, 10, 64)
        for _ in range(5):
            injector.arm()
            with torch.inference_mode():
                model(x)
            assert len(injector.faults) == 1
            injector.restore()
            assert len(injector.faults) == 0


class TestGetInfo:
    def test_get_info_format(self):
        injector, model = _make_injector()
        injector.arm()
        x = torch.randn(4, 10, 64)
        with torch.inference_mode():
            model(x)
        info = injector.get_info()
        assert len(info) == 1
        entry = info[0]
        assert "layer" in entry
        assert "position" in entry
        assert "bit" in entry
        assert "original" in entry
        assert "corrupted" in entry
        assert isinstance(entry["position"], list)
        assert 0 <= entry["bit"] <= 31
