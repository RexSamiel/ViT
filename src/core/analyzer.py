import math


class RunAnalyzer:
    def __init__(self, msdc_threshold: float = 1e6):
        self.n_runs = 0

        # Accuracy metrics
        self.avg_top1 = 0.0
        self.avg_top5 = 0.0
        self.worst_top1 = 100.0
        self.worst_top5 = 100.0
        self.worst_top1_nonzero = 100.0
        self.worst_top5_nonzero = 100.0

        # Logit SDC metrics
        self.avg_logit_sdc = 0.0

        # MSDC metrics
        self.avg_msdc = 0.0
        self.worst_msdc = 0.0
        self.msdc_counted = 0
        self.msdc_skipped = 0
        self.msdc_threshold = msdc_threshold

        # Prediction SDC metrics
        self.avg_pred_sdc = 0.0
        self.avg_pred_top5_sdc = 0.0

        # Risk categories
        self.high_risk = 0
        self.medium_risk = 0
        self.safe = 0

    def update(self, run_result: dict):
        """Update the analyzer with results from a single run."""

        self.n_runs += 1

        # Accuracy Metrics
        top1_acc = run_result.get("top1_acc", 0.0)
        top5_acc = run_result.get("top5_acc", 0.0)

        self.avg_top1 = (self.avg_top1 * (self.n_runs - 1) + top1_acc) / self.n_runs
        self.avg_top5 = (self.avg_top5 * (self.n_runs - 1) + top5_acc) / self.n_runs

        self.worst_top1 = min(self.worst_top1, top1_acc)
        self.worst_top5 = min(self.worst_top5, top5_acc)

        # Track worst non-zero accuracies
        if top1_acc > 0.0:
            self.worst_top1_nonzero = min(self.worst_top1_nonzero, top1_acc)
        if top5_acc > 0.0:
            self.worst_top5_nonzero = min(self.worst_top5_nonzero, top5_acc)

        # Logit SDC Metrics
        logit_sdc = run_result.get("logit_sdc_rate", 0.0)

        self.avg_logit_sdc = (
            self.avg_logit_sdc * (self.n_runs - 1) + logit_sdc
        ) / self.n_runs

        #  MSDC Metrics
        msdc = run_result.get("msdc_avg", None)

        if msdc is None or math.isnan(msdc) or msdc > self.msdc_threshold:
            self.msdc_skipped += 1
        else:
            self.msdc_counted += 1
            self.avg_msdc = (
                self.avg_msdc * (self.msdc_counted - 1) + msdc
            ) / self.msdc_counted
            self.worst_msdc = max(self.worst_msdc, msdc)

        #  Prediction SDC Metrics
        pred_sdc = run_result.get("pred_sdc_rate", 0.0)
        pred_top5_sdc = run_result.get("pred_top5_sdc_rate", 0.0)

        self.avg_pred_sdc = (
            self.avg_pred_sdc * (self.n_runs - 1) + pred_sdc
        ) / self.n_runs
        self.avg_pred_top5_sdc = (
            self.avg_pred_top5_sdc * (self.n_runs - 1) + pred_top5_sdc
        ) / self.n_runs

        # Risk Categories
        if pred_sdc > 0.0:
            self.high_risk += 1
        elif pred_top5_sdc > 0.0:
            self.medium_risk += 1
        else:
            self.safe += 1

    def get_summary(self) -> dict[str, float | int | None]:
        return {
            # Run info
            "total_runs": self.n_runs,
            # Accuracy metrics
            "avg_top1_acc": self.avg_top1,
            "avg_top5_acc": self.avg_top5,
            "worst_top1_acc": self.worst_top1 if self.n_runs > 0 else None,
            "worst_top5_acc": self.worst_top5 if self.n_runs > 0 else None,
            "worst_top1_nonzero": self.worst_top1_nonzero if self.n_runs > 0 else None,
            "worst_top5_nonzero": self.worst_top5_nonzero if self.n_runs > 0 else None,
            # Logit SDC metrics
            "avg_logit_sdc": self.avg_logit_sdc,
            # MSDC metrics
            "avg_msdc": self.avg_msdc if self.msdc_counted > 0 else None,
            "worst_msdc": self.worst_msdc if self.msdc_counted > 0 else None,
            "msdc_counted_runs": self.msdc_counted,
            "msdc_skipped_runs": self.msdc_skipped,
            # Prediction SDC metrics
            "avg_pred_sdc": self.avg_pred_sdc,
            "avg_pred_top5_sdc": self.avg_pred_top5_sdc,
            # Risk categories
            "high_risk_count": self.high_risk,
            "high_risk_pct": 100 * self.high_risk / self.n_runs if self.n_runs else 0.0,
            "medium_risk_count": self.medium_risk,
            "medium_risk_pct": 100 * self.medium_risk / self.n_runs
            if self.n_runs
            else 0.0,
            "safe_count": self.safe,
            "safe_pct": 100 * self.safe / self.n_runs if self.n_runs else 0.0,
        }

    def print_summary(self):
        high_risk_pct = 100 * self.high_risk / self.n_runs if self.n_runs else 0.0
        medium_risk_pct = 100 * self.medium_risk / self.n_runs if self.n_runs else 0.0
        safe_pct = 100 * self.safe / self.n_runs if self.n_runs else 0.0

        output = (
            "\n============================================================\n"
            "ANALYSIS OF MULTI-RUN EXPERIMENT\n"
            "============================================================\n"
            f"Total runs: {self.n_runs}\n"
            "\nAccuracy Metrics:\n"
            f"  Average Top-1 Accuracy:       {self.avg_top1:.2f}%\n"
            f"  Average Top-5 Accuracy:       {self.avg_top5:.2f}%\n"
            f"  Worst Top-1 Accuracy:         {self.worst_top1:.2f}%\n"
            f"  Worst Top-5 Accuracy:         {self.worst_top5:.2f}%\n"
            f"  Worst Top-1 (non-zero):       {self.worst_top1_nonzero:.2f}%\n"
            f"  Worst Top-5 (non-zero):       {self.worst_top5_nonzero:.2f}%\n"
            "\nLogit SDC Metrics:\n"
            f"  Average Logit SDC Rate:       {self.avg_logit_sdc:.2f}%\n"
        )

        # MSDC section
        if self.msdc_counted > 0:
            output += (
                "\nMSDC Metrics:\n"
                f"  Average MSDC:                 {self.avg_msdc:.6f}\n"
                f"  Worst MSDC:                   {self.worst_msdc:.6f}\n"
            )
            if self.msdc_skipped > 0:
                output += f"  Runs skipped for MSDC:        {self.msdc_skipped}\n"
        else:
            output += "\nMSDC Metrics:\n  No valid MSDC values (all runs skipped)\n"

        output += (
            "\nPrediction SDC Metrics:\n"
            f"  Average Top-1 Pred Change:    {self.avg_pred_sdc:.2f}%\n"
            f"  Average Top-5 Set Change:     {self.avg_pred_top5_sdc:.2f}%\n"
        )

        # Risk categories
        output += (
            "\nRisk Categories:\n"
            f"  High risk (top-1 changed):    {high_risk_pct:>6.2f}% "
            f"({self.high_risk} runs)\n"
            f"  Medium risk (top-5 changed):  {medium_risk_pct:>6.2f}% "
            f"({self.medium_risk} runs)\n"
            f"  Safe (no changes):            {safe_pct:>6.2f}% "
            f"({self.safe} runs)\n"
            "============================================================\n"
        )

        print(output)

