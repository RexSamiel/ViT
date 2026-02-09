"""Accuracy computation - AccuracyTracker with all accuracy-related methods."""

import torch


class AccuracyTracker:
    """Tracks per-run accuracy counts and multi-run aggregation via Welford's algorithm.

    All accuracy computation functions are methods of this class:
    - Batch-level: compute_batch_accuracy, compute_accuracy_results
    - Helpers: prepare_labels, align_batch_sizes, compute_std
    - Aggregation: aggregate_accuracy
    """

    def __init__(self):
        self.reset()
        self.reset_aggregation()

    # Static helper methods

    @staticmethod
    def compute_batch_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> dict:
        """Compute top-1 and top-5 accuracy for a single batch.

        Args:
            outputs: Model output logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]

        Returns:
            Dictionary with keys:
            - top1_correct: int, number of correct top-1 predictions
            - top5_correct: int, number of correct top-5 predictions
            - batch_size: int, number of samples in batch
        """
        batch_size = labels.size(0)

        # Handle NaN values
        all_nan_mask = torch.isnan(outputs).all(dim=1)

        nan_mask = torch.isnan(outputs)
        if nan_mask.any():
            outputs = outputs.clone()
            outputs[nan_mask] = float("-inf")

        # Top-1 accuracy
        predictions = outputs.argmax(dim=1)
        if all_nan_mask.any():
            predictions = predictions.clone()
            predictions[all_nan_mask] = 1001  # Invalid class
        top1_correct = (predictions == labels).sum().item()

        # Top-5 accuracy
        top5_preds = outputs.topk(5, dim=1)[1]
        if all_nan_mask.any():
            top5_preds = top5_preds.clone()
            top5_preds[all_nan_mask] = 1001  # Invalid class
        top5_match = (labels.unsqueeze(1) == top5_preds).any(dim=1)
        top5_correct = top5_match.sum().item()

        return {
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
            "batch_size": batch_size,
        }

    @staticmethod
    def compute_accuracy_results(top1_correct: int, top5_correct: int, total_samples: int) -> dict:
        """Compute accuracy percentages from accumulated counts.

        Args:
            top1_correct: Total number of correct top-1 predictions
            top5_correct: Total number of correct top-5 predictions
            total_samples: Total number of samples

        Returns:
            Dictionary with top1_acc, top5_acc, samples
        """
        if total_samples == 0:
            return {"top1_acc": 0.0, "top5_acc": 0.0, "samples": 0}

        return {
            "top1_acc": 100 * top1_correct / total_samples,
            "top5_acc": 100 * top5_correct / total_samples,
            "samples": total_samples,
        }

    @staticmethod
    def aggregate_accuracy(
        avg_top1: float,
        avg_top5: float,
        m2_top1: float,
        m2_top5: float,
        n_runs: int,
        new_top1: float,
        new_top5: float,
    ) -> dict:
        """Welford's online algorithm for incremental mean/variance.

        Args:
            avg_top1: Current average top-1 accuracy
            avg_top5: Current average top-5 accuracy
            m2_top1: Current M2 for top-1 (sum of squared differences)
            m2_top5: Current M2 for top-5 (sum of squared differences)
            n_runs: Number of runs processed so far
            new_top1: New top-1 accuracy to add
            new_top5: New top-5 accuracy to add

        Returns:
            Dictionary with updated avg_top1, avg_top5, m2_top1, m2_top5
        """
        n_runs += 1

        delta_top1 = new_top1 - avg_top1
        delta_top5 = new_top5 - avg_top5

        avg_top1 += delta_top1 / n_runs
        avg_top5 += delta_top5 / n_runs

        delta2_top1 = new_top1 - avg_top1
        delta2_top5 = new_top5 - avg_top5

        m2_top1 += delta_top1 * delta2_top1
        m2_top5 += delta_top5 * delta2_top5

        return {
            "avg_top1": avg_top1,
            "avg_top5": avg_top5,
            "m2_top1": m2_top1,
            "m2_top5": m2_top5,
        }

    @staticmethod
    def compute_std(m2: float, n_runs: int) -> float:
        """Compute standard deviation from M2 (Welford's algorithm).

        Args:
            m2: M2 value (sum of squared differences from mean)
            n_runs: Number of runs

        Returns:
            Standard deviation, or 0.0 if n_runs < 2
        """
        if n_runs < 2:
            return 0.0
        return (m2 / (n_runs - 1)) ** 0.5

    @staticmethod
    def prepare_labels(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Replace labels for NaN outputs with -1 so they don't count as correct."""
        nan_mask = torch.isnan(outputs).all(dim=1)
        if nan_mask.any():
            labels = labels.clone()
            labels[nan_mask] = -1
        return labels

    @staticmethod
    def align_batch_sizes(faulty: torch.Tensor, faultfree: torch.Tensor) -> tuple:
        """Align batch sizes between two tensors by truncating to the smaller."""
        min_size = min(faulty.size(0), faultfree.size(0))
        return faulty[:min_size], faultfree[:min_size]

    # Per-run state

    def reset(self) -> None:
        """Reset per-run counters."""
        self.total_samples = 0
        self.top1_correct = 0
        self.top5_correct = 0

    def update_batch(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Process a batch and accumulate accuracy counts."""
        labels_clean = self.prepare_labels(outputs, labels)
        batch_acc = self.compute_batch_accuracy(outputs, labels_clean)
        self.top1_correct += batch_acc["top1_correct"]
        self.top5_correct += batch_acc["top5_correct"]
        self.total_samples += batch_acc["batch_size"]

    def get_results(self) -> dict:
        """Return accuracy metrics for current run."""
        return self.compute_accuracy_results(
            self.top1_correct, self.top5_correct, self.total_samples
        )

    # Multi-run aggregation state

    def reset_aggregation(self) -> None:
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
        self.worst_top1_nonzero = 100.0
        self.worst_top5_nonzero = 100.0

    def aggregate_run(self, run_results: dict) -> None:
        """Aggregate a run's accuracy into multi-run statistics."""
        top1 = run_results.get("top1_acc", 0.0)
        top5 = run_results.get("top5_acc", 0.0)

        agg = self.aggregate_accuracy(
            self.avg_top1, self.avg_top5,
            self.m2_top1, self.m2_top5,
            self.n_runs, top1, top5,
        )

        self.avg_top1 = agg["avg_top1"]
        self.avg_top5 = agg["avg_top5"]
        self.m2_top1 = agg["m2_top1"]
        self.m2_top5 = agg["m2_top5"]

        self.worst_top1 = min(self.worst_top1, top1)
        self.worst_top5 = min(self.worst_top5, top5)
        self.best_top1 = max(self.best_top1, top1)
        self.best_top5 = max(self.best_top5, top5)

        if top1 > 0.0:
            self.worst_top1_nonzero = min(self.worst_top1_nonzero, top1)
        if top5 > 0.0:
            self.worst_top5_nonzero = min(self.worst_top5_nonzero, top5)

        self.n_runs += 1

    def get_summary(self) -> dict:
        """Return aggregated accuracy summary."""
        return {
            "avg_top1_acc": self.avg_top1,
            "avg_top5_acc": self.avg_top5,
            "std_top1_acc": self.compute_std(self.m2_top1, self.n_runs),
            "std_top5_acc": self.compute_std(self.m2_top5, self.n_runs),
            "best_top1_acc": self.best_top1 if self.n_runs > 0 else None,
            "best_top5_acc": self.best_top5 if self.n_runs > 0 else None,
            "worst_top1_acc": self.worst_top1 if self.n_runs > 0 else None,
            "worst_top5_acc": self.worst_top5 if self.n_runs > 0 else None,
            "worst_top1_nonzero": self.worst_top1_nonzero if self.n_runs > 0 else None,
            "worst_top5_nonzero": self.worst_top5_nonzero if self.n_runs > 0 else None,
        }

    def print_summary(self) -> None:
        """Print aggregated accuracy summary."""
        s = self.get_summary()

        worst_top1_line = (
            f"  Worst Top-1 Accuracy:         {s['worst_top1_acc']:.2f}%"
            if s['worst_top1_acc'] > 0.0
            else f"  Worst Top-1 (non-zero):       {s['worst_top1_nonzero']:.2f}%"
        )
        worst_top5_line = (
            f"  Worst Top-5 Accuracy:         {s['worst_top5_acc']:.2f}%"
            if s['worst_top5_acc'] > 0.0
            else f"  Worst Top-5 (non-zero):       {s['worst_top5_nonzero']:.2f}%"
        )

        print(
            f"Accuracy Metrics:\n"
            f"  Average Top-1 Accuracy:       {s['avg_top1_acc']:.2f}% +/- {s['std_top1_acc']:.2f}%\n"
            f"  Average Top-5 Accuracy:       {s['avg_top5_acc']:.2f}% +/- {s['std_top5_acc']:.2f}%\n"
            f"  Best Top-1 Accuracy:          {s['best_top1_acc']:.2f}%\n"
            f"  Best Top-5 Accuracy:          {s['best_top5_acc']:.2f}%\n"
            f"{worst_top1_line}\n"
            f"{worst_top5_line}"
        )
