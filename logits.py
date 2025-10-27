class FaultFreeLogits:
    def __init__(self, model_key):
        self.filename = f"ff_logits_{model_key}.pt"
        self.data = None
        self.load()

    def load(self):
        if Path(self.filename).exists():
            self.data = torch.load(self.filename, weights_only=False)
            print(f"✓ Fault-free logits loaded from {self.filename}")
        else:
            print(
                f"x Fault-free logits not found. Run with --faultfree --logits first."
            )

    def save(self, logits, labels):
        torch.save(
            {"logits": torch.cat(logits), "labels": torch.cat(labels)},
            self.filename,
        )
        print(f"✓ Fault-free logits saved to {self.filename}")

    def get_batch(self, batch_idx, batch_size, actual_size, device):
        if self.data is None:
            raise RuntimeError(
                "Fault-free logits required for SDC computation. "
                "Run: python script.py --model <model> --faultfree --logits"
            )

        start = batch_idx * batch_size
        end = start + actual_size
        return self.data["logits"][start:end].to(device)

    @property
    def available(self):
        return self.data is not None
