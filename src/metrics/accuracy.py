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

        # Identify samples where ALL logits are NaN (catastrophic failure)
        all_nan_mask = torch.isnan(outputs).all(dim=1)

        # Handle partial NaN: replace individual NaN values with -inf
        # so they don't interfere with argmax/topk on valid values
        nan_mask = torch.isnan(outputs)
        if nan_mask.any():
            outputs = outputs.clone()
            outputs[nan_mask] = float("-inf")

        # Top-1 accuracy
        predictions = outputs.argmax(dim=1)
        # For all-NaN samples, set prediction to invalid class 1001
        # (ImageNet has classes 0-999, so 1001 never matches any label)
        if all_nan_mask.any():
            predictions = predictions.clone()
            predictions[all_nan_mask] = 1001
        self.top1_correct += (predictions == labels).sum().item()

        # Top-5 accuracy
        top5_preds = outputs.topk(5, dim=1)[1]
        # For all-NaN samples, set all top-5 predictions to invalid class 1001
        if all_nan_mask.any():
            top5_preds = top5_preds.clone()
            top5_preds[all_nan_mask] = 1001
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
        # Welford's online variance (M2 = sum of squared differences from mean)
        self.m2_top1 = 0.0
        self.m2_top5 = 0.0
        self.worst_top1 = 100.0
        self.worst_top5 = 100.0
        self.best_top1 = 0.0
        self.best_top5 = 0.0
        self.worst_top1_nonzero = 100.0
        self.worst_top5_nonzero = 100.0

    def aggregate(self, run_results: dict) -> None:
        """Aggregate results from a single run using Welford's online algorithm."""
        self.n_runs += 1

        top1 = run_results.get("top1_acc", 0.0)
        top5 = run_results.get("top5_acc", 0.0)

        # Welford's online algorithm for mean and variance
        delta_top1 = top1 - self.avg_top1
        delta_top5 = top5 - self.avg_top5

        self.avg_top1 += delta_top1 / self.n_runs
        self.avg_top5 += delta_top5 / self.n_runs

        delta2_top1 = top1 - self.avg_top1
        delta2_top5 = top5 - self.avg_top5

        self.m2_top1 += delta_top1 * delta2_top1
        self.m2_top5 += delta_top5 * delta2_top5

        # Best/worst case tracking
        self.worst_top1 = min(self.worst_top1, top1)
        self.worst_top5 = min(self.worst_top5, top5)
        self.best_top1 = max(self.best_top1, top1)
        self.best_top5 = max(self.best_top5, top5)

        if top1 > 0.0:
            self.worst_top1_nonzero = min(self.worst_top1_nonzero, top1)
        if top5 > 0.0:
            self.worst_top5_nonzero = min(self.worst_top5_nonzero, top5)

    def _std(self, m2: float) -> float:
        """Compute standard deviation from M2 (Welford's algorithm)."""
        if self.n_runs < 2:
            return 0.0
        return (m2 / (self.n_runs - 1)) ** 0.5

    def get_summary(self) -> dict:
        """Return aggregated accuracy summary."""
        return {
            "avg_top1_acc": self.avg_top1,
            "avg_top5_acc": self.avg_top5,
            "std_top1_acc": self._std(self.m2_top1),
            "std_top5_acc": self._std(self.m2_top5),
            "best_top1_acc": self.best_top1 if self.n_runs > 0 else None,
            "best_top5_acc": self.best_top5 if self.n_runs > 0 else None,
            "worst_top1_acc": self.worst_top1 if self.n_runs > 0 else None,
            "worst_top5_acc": self.worst_top5 if self.n_runs > 0 else None,
            "worst_top1_nonzero": self.worst_top1_nonzero if self.n_runs > 0 else None,
            "worst_top5_nonzero": self.worst_top5_nonzero if self.n_runs > 0 else None,
        }

    def print_summary(self) -> None:
        """Print aggregated accuracy summary."""
        std_top1 = self._std(self.m2_top1)
        std_top5 = self._std(self.m2_top5)

        print("Accuracy Metrics:")
        print(f"  Average Top-1 Accuracy:       {self.avg_top1:.2f}% ± {std_top1:.2f}%")
        print(f"  Average Top-5 Accuracy:       {self.avg_top5:.2f}% ± {std_top5:.2f}%")
        print(f"  Best Top-1 Accuracy:          {self.best_top1:.2f}%")
        print(f"  Best Top-5 Accuracy:          {self.best_top5:.2f}%")

        if self.worst_top1 > 0.0:
            print(f"  Worst Top-1 Accuracy:         {self.worst_top1:.2f}%")
        else:
            print(f"  Worst Top-1 (non-zero):       {self.worst_top1_nonzero:.2f}%")

        if self.worst_top5 > 0.0:
            print(f"  Worst Top-5 Accuracy:         {self.worst_top5:.2f}%")
        else:
            print(f"  Worst Top-5 (non-zero):       {self.worst_top5_nonzero:.2f}%")
