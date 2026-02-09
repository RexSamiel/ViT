from pathlib import Path
import torch

LOGITS_DIR = Path("logits")


class FaultFreeLogits:
    def __init__(self, model_key):
        self.logits_dir = LOGITS_DIR
        self.filename = self.logits_dir / f"ff_logits_{model_key}.pt"
        self.data = None
        self._gpu_cache = None  # Cache logits on GPU to avoid repeated transfers
        self._cached_device = None
        self.load()

    def load(self):
        if self.filename.exists():
            self.data = torch.load(self.filename, weights_only=False)
            print(f"✓ Fault-free logits loaded from {self.filename}")
        else:
            print(f"x Fault-free logits not found. Run with --mode faultfree --save_logits true first.")

    def save(self, logits, labels):
        self.logits_dir.mkdir(exist_ok=True)
        torch.save(
            {"logits": torch.cat(logits), "labels": torch.cat(labels)},
            self.filename,
        )
        print(f"✓ Fault-free logits saved to {self.filename}")

    def _ensure_gpu_cache(self, device):
        """Load logits to GPU once and cache them."""
        if self._gpu_cache is None or self._cached_device != device:
            self._gpu_cache = self.data["logits"].to(device)
            self._cached_device = device

    def get_batch(self, batch_idx, batch_size, actual_size, device):
        if self.data is None:
            raise RuntimeError(
                "Fault-free logits required for SDC computation. "
                "Run: python script.py --model <model> --faultfree --logits"
            )

        # Use GPU-cached logits to avoid repeated CPU->GPU transfers
        self._ensure_gpu_cache(device)

        start = batch_idx * batch_size
        end = start + actual_size
        return self._gpu_cache[start:end]

    @property
    def available(self):
        return self.data is not None
