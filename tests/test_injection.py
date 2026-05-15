"""Test fault injection."""

import pytest
import torch
import torch.nn as nn

from core.bits import flip_bit
from injection import InjectedFault, Injector


class TestFlipBit:
    """Tests for the flip_bit function."""

    def test_flips_single_bit(self):
        val = torch.Tensor(1.0)
        corrupted, bit, orig_bits, new_bits = flip_bit(val, bit=0)

        diff = int(orig_bits, 2) ^ int(new_bits, 2)
        assert bin(diff).count("1") == 1

    def test_respects_bit_range(self):
        val = torch.Tensor(1.0)

        for _ in range(50):
            _, bit, _, _ = flip_bit(val, bit_range=(20, 25))
            assert 20 <= bit <= 25


class TestInjector:
    def _make_model(self):

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        return Model()

    def test_injects_specified_count(self):
        model = self._make_model()

        injector = Injector(model, layers="all")
        injector.inject(count=5)

        assert injector.count == 5
        assert len(injector.faults) == 5

    def test_restore_reverts_weights(self):
        model = self._make_model()
        original_fc1 = model.fc1.weight.data.clone()
        original_fc2 = model.fc2.weight.data.clone()

        injector = Injector(model, layers="all")
        injector.inject(count=10)

        fc1_changed = not torch.equal(model.fc1.weight.data, original_fc1)
        fc2_changed = not torch.equal(model.fc2.weight.data, original_fc2)
        assert fc1_changed or fc2_changed

        injector.restore()

        assert torch.allclose(model.fc1.weight.data, original_fc1)
        assert torch.allclose(model.fc2.weight.data, original_fc2)

    def test_layer_filtering(self):
        model = self._make_model()

        # Only inject into fc1
        injector = Injector(model, layers="fc1")
        injector.inject(count=5)

        # All faults should be in fc1
        for fault in injector.faults:
            assert "fc1" in fault.layer

    def test_ber_mode(self):
        """Injector should work with BER mode."""
        model = self._make_model()

        injector = Injector(model, layers="all")
        injector.inject(ber=1e-4)

        # Should have injected some faults
        assert injector.count > 0

    def test_fault_info(self):
        """get_info should return serializable fault info."""
        model = self._make_model()

        injector = Injector(model, layers="all")
        injector.inject(count=3)

        info = injector.get_info()
        assert len(info) == 3
        assert all(isinstance(f, dict) for f in info)
        assert all("layer" in f and "bit" in f for f in info)
