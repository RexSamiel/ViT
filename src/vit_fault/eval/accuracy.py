"""Accuracy computation with multi-run aggregation."""

import torch


class AccuracyTracker:
    """Tracks accuracy per-run and aggregates across multiple runs."""

    def __init__(self):
        self.reset()
        self.reset_aggregation()

    def reset(self):
        """Reset per-run counters."""
        self.total_samples = 0
        self.top1_correct = 0
        self.top5_correct = 0

    def update_batch(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Process a batch and accumulate accuracy counts."""
        batch_size = labels.size(0)

        nan_mask = torch.isnan(outputs).all(dim=1)
        if nan_mask.any():
            outputs = outputs.clone()
            outputs[torch.isnan(outputs)] = float("-inf")
            labels = labels.clone()
            labels[nan_mask] = -1  # Invalid label

        # Top-1 accuracy
        predictions = outputs.argmax(dim=1)
        if nan_mask.any():
            predictions[nan_mask] = 1001  # Invalid class
        self.top1_correct += (predictions == labels).sum().item()

        # Top-5 accuracy
        top5_preds = outputs.topk(5, dim=1)[1]
        if nan_mask.any():
            top5_preds[nan_mask] = 1001
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

    # Multi-run aggregation (Welford's algorithm)

    def reset_aggregation(self):
        """Reset multi-run aggregation state."""
        self.n_runs = 0
        self.avg_top1 = 0.0
        self.avg_top5 = 0.0
        self.m2_top1 = 0.0
        self.m2_top5 = 0.0
        self.worst_top1 = 100.0
        self.worst_top5 = 100.0
        self.best_top1 = 0.0
        self.best_top5 = 0.0

    def aggregate_run(self, run_results: dict):
        """Aggregate a run's accuracy into multi-run statistics."""
        top1 = run_results.get("top1_acc", 0.0)
        top5 = run_results.get("top5_acc", 0.0)

        self.n_runs += 1

        # Welford's online algorithm
        delta_top1 = top1 - self.avg_top1
        delta_top5 = top5 - self.avg_top5

        self.avg_top1 += delta_top1 / self.n_runs
        self.avg_top5 += delta_top5 / self.n_runs

        delta2_top1 = top1 - self.avg_top1
        delta2_top5 = top5 - self.avg_top5

        self.m2_top1 += delta_top1 * delta2_top1
        self.m2_top5 += delta_top5 * delta2_top5

        # Track extremes
        self.worst_top1 = min(self.worst_top1, top1)
        self.worst_top5 = min(self.worst_top5, top5)
        self.best_top1 = max(self.best_top1, top1)
        self.best_top5 = max(self.best_top5, top5)

    def get_summary(self) -> dict:
        """Return aggregated accuracy summary."""
        std_top1 = (self.m2_top1 / (self.n_runs - 1)) ** 0.5 if self.n_runs > 1 else 0.0
        std_top5 = (self.m2_top5 / (self.n_runs - 1)) ** 0.5 if self.n_runs > 1 else 0.0

        return {
            "avg_top1": self.avg_top1,
            "avg_top5": self.avg_top5,
            "std_top1": std_top1,
            "std_top5": std_top5,
            "best_top1": self.best_top1,
            "best_top5": self.best_top5,
            "worst_top1": self.worst_top1,
            "worst_top5": self.worst_top5,
            "n_runs": self.n_runs,
        }

    def print_summary(self):
        """Print aggregated accuracy summary."""
        s = self.get_summary()
        print(f"Accuracy ({s['n_runs']} runs):")
        print(f"  Top-1: {s['avg_top1']:.2f}% ± {s['std_top1']:.2f}%")
        print(f"  Top-5: {s['avg_top5']:.2f}% ± {s['std_top5']:.2f}%")
        print(f"  Best:  {s['best_top1']:.2f}% / {s['best_top5']:.2f}%")
        print(f"  Worst: {s['worst_top1']:.2f}% / {s['worst_top5']:.2f}%")
