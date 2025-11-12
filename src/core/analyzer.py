# analyze.py
import math


class RunAnalyzer:
    """Streaming / incremental analyzer for repeated ViT runs."""

    def __init__(self, msdc_threshold: float = 1e6):
        self.n_runs = 0
        # Running averages
        self.avg_top1 = 0.0
        self.avg_top5 = 0.0
        self.avg_logit_sdc = 0.0
        self.avg_msdc = 0.0
        self.msdc_counted = 0
        self.msdc_skipped = 0
        self.msdc_threshold = msdc_threshold

        # Risk counters
        self.high_risk = 0
        self.medium_risk = 0
        self.safe = 0

    def update(self, run_result: dict):
        """Update averages and risk counts after a single run."""
        self.n_runs += 1

        # Top-1 and Top-5
        self.avg_top1 = (
            self.avg_top1 * (self.n_runs - 1) + run_result.get("top1_acc", 0.0)
        ) / self.n_runs
        self.avg_top5 = (
            self.avg_top5 * (self.n_runs - 1) + run_result.get("top5_acc", 0.0)
        ) / self.n_runs

        # Logit SDC
        self.avg_logit_sdc = (
            self.avg_logit_sdc * (self.n_runs - 1)
            + run_result.get("logit_sdc_rate", 0.0)
        ) / self.n_runs

        # MSDC handling
        msdc = run_result.get("msdc_avg", None)
        if msdc is None or math.isnan(msdc) or msdc > self.msdc_threshold:
            self.msdc_skipped += 1
        else:
            self.msdc_counted += 1
            self.avg_msdc = (
                self.avg_msdc * (self.msdc_counted - 1) + msdc
            ) / self.msdc_counted

        # Risk categories
        pred_sdc = run_result.get("pred_sdc_rate", 0.0)
        pred_top5_sdc = run_result.get("pred_top5_sdc_rate", 0.0)
        if pred_sdc > 0.0:
            self.high_risk += 1
        elif pred_top5_sdc > 0.0:
            self.medium_risk += 1
        else:
            self.safe += 1

    def print_summary(self):
        """Print the current summary of all runs so far."""
        print("\n===== ANALYSIS OF MULTI-RUN EXPERIMENT =====")
        print(f"Total runs: {self.n_runs}")
        print(f"Average Top-1 Accuracy: {self.avg_top1:.2f}%")
        print(f"Average Top-5 Accuracy: {self.avg_top5:.2f}%")
        print(f"Average Logit SDC Rate: {self.avg_logit_sdc:.2f}%")

        if self.msdc_counted > 0:
            print(f"Average MSDC (counted runs): {self.avg_msdc:.6f}")
            print(f"Runs skipped for MSDC: {self.msdc_skipped}")
        else:
            print("No valid MSDC values could be averaged (all runs skipped).")

        print("\nRisk categories:")
        print(
            f"High risk runs (top-1 changed): {100 * self.high_risk / self.n_runs:.2f}% ({self.high_risk})"
        )
        print(
            f"Medium risk runs (top-5 changed): {100 * self.medium_risk / self.n_runs:.2f}% ({self.medium_risk})"
        )
        print(
            f"Safe runs (no top-1/top-5 changes): {100 * self.safe / self.n_runs:.2f}% ({self.safe})"
        )
        print("============================================\n")
