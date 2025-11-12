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
        self.avg_pred_sdc = 0.0
        self.avg_pred_top5_sdc = 0.0
        self.msdc_counted = 0
        self.msdc_skipped = 0
        self.msdc_threshold = msdc_threshold
        # Risk counters
        self.high_risk = 0  # Top-1 prediction changed significantly
        self.medium_risk = 0  # Top-5 changed (but top-1 didn't)
        self.safe = 0  # No significant prediction changes

        # Thresholds for "significant" changes (for risk categorization)
        self.top1_threshold = 0.1  # 0.1% top-1 change threshold
        self.top5_threshold = 0.1  # 0.1% top-5 change threshold

    def update(self, run_result: dict):
        """Update averages and risk counts after a single run."""
        self.n_runs += 1

        # Top-1 and Top-5 accuracy
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

        # Prediction SDCs (averages across runs)
        self.avg_pred_sdc = (
            self.avg_pred_sdc * (self.n_runs - 1) + run_result.get("pred_sdc_rate", 0.0)
        ) / self.n_runs

        self.avg_pred_top5_sdc = (
            self.avg_pred_top5_sdc * (self.n_runs - 1)
            + run_result.get("pred_top5_sdc_rate", 0.0)
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

        # Risk categories with thresholds
        pred_sdc = run_result.get("pred_sdc_rate", 0.0)
        pred_top5_sdc = run_result.get("pred_top5_sdc_rate", 0.0)

        # High risk: Top-1 prediction changed significantly
        # Medium risk: Top-5 set changed significantly (but top-1 didn't)
        # Safe: No significant changes

        if pred_sdc > self.top1_threshold:
            self.high_risk += 1
        elif pred_top5_sdc > self.top5_threshold:
            self.medium_risk += 1
        else:
            self.safe += 1

    def print_summary(self):
        """Print the current summary of all runs so far."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("ANALYSIS OF MULTI-RUN EXPERIMENT")
        print("=" * 60)
        print(f"Total runs: {summary['total_runs']}")
        print(f"\nAccuracy Metrics:")
        print(f"  Average Top-1 Accuracy: {summary['avg_top1_acc']:.2f}%")
        print(f"  Average Top-5 Accuracy: {summary['avg_top5_acc']:.2f}%")

        print(f"\nSDC Metrics:")
        print(f"  Average Logit SDC Rate:       {summary['avg_logit_sdc']:.2f}%")
        print(f"  Average Top-1 Pred Change:    {summary['avg_pred_sdc']:.2f}%")
        print(f"  Average Top-5 Set Change:     {summary['avg_pred_top5_sdc']:.2f}%")

        if summary["msdc_counted_runs"] > 0:
            print(f"  Average MSDC (counted):       {summary['avg_msdc']:.6f}")
            if summary["msdc_skipped_runs"] > 0:
                print(f"  Runs skipped for MSDC:        {summary['msdc_skipped_runs']}")
        else:
            print("  No valid MSDC values (all runs skipped)")

        print(f"\nRisk Categories:")
        print(
            f"  High risk (top-1 changed):    {summary['high_risk_pct']:>6.2f}% ({summary['high_risk_count']} runs)"
        )
        print(
            f"  Medium risk (top-5 changed):  {summary['medium_risk_pct']:>6.2f}% ({summary['medium_risk_count']} runs)"
        )
        print(
            f"  Safe (minimal changes):       {summary['safe_pct']:>6.2f}% ({summary['safe_count']} runs)"
        )
        print("=" * 60 + "\n")

    def get_summary(self) -> dict[str, float | int | None]:
        """Return the current summary as a dictionary for JSON export or further analysis."""
        return {
            "total_runs": self.n_runs,
            "avg_top1_acc": self.avg_top1,
            "avg_top5_acc": self.avg_top5,
            "avg_logit_sdc": self.avg_logit_sdc,
            "avg_pred_sdc": self.avg_pred_sdc,
            "avg_pred_top5_sdc": self.avg_pred_top5_sdc,
            "avg_msdc": self.avg_msdc if self.msdc_counted > 0 else None,
            "msdc_counted_runs": self.msdc_counted,
            "msdc_skipped_runs": self.msdc_skipped,
            "high_risk_pct": 100 * self.high_risk / self.n_runs if self.n_runs else 0.0,
            "high_risk_count": self.high_risk,
            "medium_risk_pct": 100 * self.medium_risk / self.n_runs
            if self.n_runs
            else 0.0,
            "medium_risk_count": self.medium_risk,
            "safe_pct": 100 * self.safe / self.n_runs if self.n_runs else 0.0,
            "safe_count": self.safe,
        }
