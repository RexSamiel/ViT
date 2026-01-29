import torch


class AccuracyMetrics:
    """Handles top-1 and top-5 accuracy computation and aggregation."""

    def __init__(self):
        self.reset()
        self.reset_aggregation()

    # ==================== Single Run Methods ====================

    def reset(self) -> None:
        """Clear data for new run."""
        self.total_samples = 0
        self.top1_correct = 0
        self.top5_correct = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Update accuracy with a batch of predictions."""
        batch_size = labels.size(0)

        # Top-1 accuracy
        predictions = outputs.argmax(dim=1)
        self.top1_correct += (predictions == labels).sum().item()

        # Top-5 accuracy
        top5_preds = outputs.topk(5, dim=1)[1]
        top5_match = (labels.unsqueeze(1) == top5_preds).any(dim=1)
        self.top5_correct += top5_match.sum().item()

        self.total_samples += batch_size

    def get_results(self) -> dict:
        """Return accuracy metrics for current run."""
        if self.total_samples == 0:
            return {"top1_acc": 0.0, "top5_acc": 0.0, "samples": 0}

        return {
            "top1_acc": 100 * self.top1_correct / self.total_samples,
            "top5_acc": 100 * self.top5_correct / self.total_samples,
            "samples": self.total_samples,
        }

    def print_results(self) -> None:
        """Print accuracy results for current run."""
        r = self.get_results()
        print(f"  Top-1 Accuracy: {r['top1_acc']:.2f}%")
        print(f"  Top-5 Accuracy: {r['top5_acc']:.2f}%")
        print(f"  Samples: {r['samples']}")

    # ==================== Aggregation Methods ====================

    def reset_aggregation(self) -> None:
        """Clear aggregation data."""
        self.n_runs = 0
        self.avg_top1 = 0.0
        self.avg_top5 = 0.0
        self.worst_top1 = 100.0
        self.worst_top5 = 100.0
        self.worst_top1_nonzero = 100.0
        self.worst_top5_nonzero = 100.0

    def aggregate(self, run_results: dict) -> None:
        """Aggregate results from a single run."""
        self.n_runs += 1

        top1 = run_results.get("top1_acc", 0.0)
        top5 = run_results.get("top5_acc", 0.0)

        # Running average
        self.avg_top1 = (self.avg_top1 * (self.n_runs - 1) + top1) / self.n_runs
        self.avg_top5 = (self.avg_top5 * (self.n_runs - 1) + top5) / self.n_runs

        # Worst case tracking
        self.worst_top1 = min(self.worst_top1, top1)
        self.worst_top5 = min(self.worst_top5, top5)

        if top1 > 0.0:
            self.worst_top1_nonzero = min(self.worst_top1_nonzero, top1)
        if top5 > 0.0:
            self.worst_top5_nonzero = min(self.worst_top5_nonzero, top5)

    def get_summary(self) -> dict:
        """Return aggregated accuracy summary."""
        return {
            "avg_top1_acc": self.avg_top1,
            "avg_top5_acc": self.avg_top5,
            "worst_top1_acc": self.worst_top1 if self.n_runs > 0 else None,
            "worst_top5_acc": self.worst_top5 if self.n_runs > 0 else None,
            "worst_top1_nonzero": self.worst_top1_nonzero if self.n_runs > 0 else None,
            "worst_top5_nonzero": self.worst_top5_nonzero if self.n_runs > 0 else None,
        }

    def print_summary(self) -> None:
        """Print aggregated accuracy summary."""
        print("Accuracy Metrics:")
        print(f"  Average Top-1 Accuracy:       {self.avg_top1:.2f}%")
        print(f"  Average Top-5 Accuracy:       {self.avg_top5:.2f}%")

        if self.worst_top1 > 0.0:
            print(f"  Worst Top-1 Accuracy:         {self.worst_top1:.2f}%")
        else:
            print(f"  Worst Top-1 (non-zero):       {self.worst_top1_nonzero:.2f}%")

        if self.worst_top5 > 0.0:
            print(f"  Worst Top-5 Accuracy:         {self.worst_top5:.2f}%")
        else:
            print(f"  Worst Top-5 (non-zero):       {self.worst_top5_nonzero:.2f}%")
