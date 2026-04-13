"""SDC (Silent Data Corruption) metrics computation.

Metrics computed:
- Logit SDC Rate: Percentage of logits with >0.1% relative change
- MSDC: Mean magnitude of logit changes (absolute)
- Threshold SDC: % of samples where MAX logit change exceeds threshold (1-50%)
- Critical SDC: Prediction changed AND original prediction's logit was corrupted
"""

import torch


class SDCTracker:
    """Tracks comprehensive SDC metrics per-run and aggregates across multiple runs."""

    THRESHOLDS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50]

    def __init__(self):
        self.reset()
        self.reset_aggregation()

    def reset(self):
        """Reset per-run accumulators."""
        self.sdc_rates: list[torch.Tensor] = []
        self.sdc_magnitudes: list[torch.Tensor] = []
        self.threshold_counts: dict[float, list[torch.Tensor]] = {
            t: [] for t in self.THRESHOLDS
        }
        self.critical_top1_rates: list[float] = []
        self.critical_top5_rates: list[float] = []
        self.total_batches = 0
        self.total_samples = 0
        self.crash_samples = 0

    def update_batch(self, faulty: torch.Tensor, faultfree: torch.Tensor):
        """Compute and accumulate SDC metrics for a batch."""
        batch_size = min(faulty.size(0), faultfree.size(0))
        faulty = faulty[:batch_size]
        faultfree = faultfree[:batch_size]

        self.total_batches += 1

        crash_mask = ~torch.isfinite(faulty).all(dim=1)
        has_valid = (~crash_mask).any().item()

        self.total_samples += faulty.size(0)
        self.crash_samples += crash_mask.sum().item()

        diff = faultfree - faulty
        relative_per_logit = diff.abs() / (faultfree.abs() + 1e-10)
        sdc_rate = (relative_per_logit > 0.001).float().mean(dim=1)
        sdc_rate[crash_mask] = 1.0
        self.sdc_rates.append(sdc_rate.cpu())

        if has_valid:
            valid_faulty = faulty[~crash_mask]
            valid_ff = faultfree[~crash_mask]
            valid_diff = valid_ff - valid_faulty

            sdc_magnitude = valid_diff.abs().median(dim=1).values
            self.sdc_magnitudes.append(sdc_magnitude.cpu())

            mean_relative_change = valid_diff.abs().mean(dim=1) / (valid_ff.abs().mean(dim=1) + 1e-10)
            for threshold in self.THRESHOLDS:
                exceeds = (mean_relative_change >= threshold).float()
                self.threshold_counts[threshold].append(exceeds.cpu())

            pred_faulty = valid_faulty.argmax(dim=1)
            pred_ff = valid_ff.argmax(dim=1)
            crit_top1 = (pred_faulty != pred_ff).float().mean().item()

            top5_faulty = valid_faulty.topk(5, dim=1)[1]
            top5_ff = valid_ff.topk(5, dim=1)[1]

            ff_sorted, _ = top5_ff.sort(dim=1)
            faulty_sorted, _ = top5_faulty.sort(dim=1)
            crit_top5 = (ff_sorted != faulty_sorted).any(dim=1).float().mean().item()

            self.critical_top1_rates.append(crit_top1)
            self.critical_top5_rates.append(crit_top5)

        elif crash_mask.all():
            self.critical_top1_rates.append(1.0)
            self.critical_top5_rates.append(1.0)

    def get_results(self) -> dict[str, float | int]:
        """Return summarized SDC metrics for current run."""
        crash_rate = (
            (self.crash_samples / self.total_samples * 100)
            if self.total_samples
            else 0.0
        )
        results: dict[str, float | int] = {
            "batches": self.total_batches,
            "crash_rate": crash_rate,
            "crash_samples": self.crash_samples,
            "total_samples": self.total_samples,
        }

        if self.sdc_rates:
            results["logit_sdc_rate"] = torch.cat(self.sdc_rates).mean().item() * 100
        else:
            results["logit_sdc_rate"] = 0.0

        if self.sdc_magnitudes:
            results["msdc"] = torch.cat(self.sdc_magnitudes).median().item()
        else:
            results["msdc"] = 0.0

        for threshold in self.THRESHOLDS:
            key = f"sdc_{int(threshold * 100)}pct"
            if self.threshold_counts[threshold]:
                results[key] = (
                    torch.cat(self.threshold_counts[threshold]).mean().item() * 100
                )
            else:
                results[key] = 0.0

        if self.critical_top1_rates:
            results["critical_top1_sdc_rate"] = (
                sum(self.critical_top1_rates) / len(self.critical_top1_rates) * 100
            )
        else:
            results["critical_top1_sdc_rate"] = 0.0

        if self.critical_top5_rates:
            results["critical_top5_sdc_rate"] = (
                sum(self.critical_top5_rates) / len(self.critical_top5_rates) * 100
            )
        else:
            results["critical_top5_sdc_rate"] = 0.0

        return results

    # Multi-run aggregation

    def reset_aggregation(self):
        """Reset multi-run aggregation state."""
        self.n_runs = 0
        self.avg_sdc_rate = 0.0
        self.avg_msdc = 0.0
        self.avg_critical_top1 = 0.0
        self.avg_critical_top5 = 0.0
        self.m2_critical_top1 = 0.0
        self.m2_critical_top5 = 0.0
        self.avg_thresholds = {t: 0.0 for t in self.THRESHOLDS}
        self.high_risk = 0
        self.medium_risk = 0
        self.safe = 0
        self.total_crash_samples = 0
        self.total_eval_samples = 0

    def aggregate_run(self, run_results: dict):
        """Aggregate a run's SDC results into multi-run statistics."""
        self.n_runs += 1

        sdc = run_results.get("logit_sdc_rate", 0.0)
        self.avg_sdc_rate = (self.avg_sdc_rate * (self.n_runs - 1) + sdc) / self.n_runs

        msdc = run_results.get("msdc", 0.0)
        self.avg_msdc = (self.avg_msdc * (self.n_runs - 1) + msdc) / self.n_runs

        for threshold in self.THRESHOLDS:
            key = f"sdc_{int(threshold * 100)}pct"
            val = run_results.get(key, 0.0)
            self.avg_thresholds[threshold] = (
                self.avg_thresholds[threshold] * (self.n_runs - 1) + val
            ) / self.n_runs

        crit_top1 = run_results.get("critical_top1_sdc_rate", 0.0)
        crit_top5 = run_results.get("critical_top5_sdc_rate", 0.0)

        delta_top1 = crit_top1 - self.avg_critical_top1
        delta_top5 = crit_top5 - self.avg_critical_top5

        self.avg_critical_top1 += delta_top1 / self.n_runs
        self.avg_critical_top5 += delta_top5 / self.n_runs

        delta2_top1 = crit_top1 - self.avg_critical_top1
        delta2_top5 = crit_top5 - self.avg_critical_top5

        self.m2_critical_top1 += delta_top1 * delta2_top1
        self.m2_critical_top5 += delta_top5 * delta2_top5

        # Risk categories
        if crit_top1 > 0.0:
            self.high_risk += 1
        elif crit_top5 > 0.0:
            self.medium_risk += 1
        else:
            self.safe += 1

        self.total_crash_samples += run_results.get("crash_samples", 0)
        self.total_eval_samples  += run_results.get("total_samples", 0)

    def get_summary(self) -> dict:
        """Return aggregated SDC summary."""
        std_top1 = (
            (self.m2_critical_top1 / (self.n_runs - 1)) ** 0.5
            if self.n_runs > 1
            else 0.0
        )
        std_top5 = (
            (self.m2_critical_top5 / (self.n_runs - 1)) ** 0.5
            if self.n_runs > 1
            else 0.0
        )

        summary = {
            "avg_sdc_rate": self.avg_sdc_rate,
            "avg_msdc": self.avg_msdc,
            "avg_critical_top1": self.avg_critical_top1,
            "avg_critical_top5": self.avg_critical_top5,
            "std_critical_top1": std_top1,
            "std_critical_top5": std_top5,
            "high_risk": self.high_risk,
            "medium_risk": self.medium_risk,
            "safe": self.safe,
            "n_runs": self.n_runs,
            "total_crash_samples": self.total_crash_samples,
            "total_eval_samples": self.total_eval_samples,
        }

        # Add threshold averages
        for threshold in self.THRESHOLDS:
            key = f"avg_sdc_{int(threshold * 100)}pct"
            summary[key] = self.avg_thresholds[threshold]

        return summary

    def print_summary(self):
        """Print aggregated SDC summary."""
        s = self.get_summary()
        print(f"\nSDC Metrics ({s['n_runs']} runs):")
        if s["total_crash_samples"] > 0:
            pct = 100.0 * s["total_crash_samples"] / s["total_eval_samples"] if s["total_eval_samples"] else 0.0
            print(f"  Crashes:        {s['total_crash_samples']} samples ({pct:.1f}% of all samples)")
        print(f"  Logit SDC Rate: {s['avg_sdc_rate']:.2f}%")
        print(f"  MSDC (median):  {s['avg_msdc']:.6f}")
        print()
        print(f"  Threshold-based SDC:")
        for threshold in self.THRESHOLDS:
            key = f"avg_sdc_{int(threshold * 100)}pct"
            print(f"    ≥{int(threshold * 100):2d}%: {s[key]:.2f}%")
        print()
        print(f"  Critical SDC:")
        print(
            f"    Top-1: {s['avg_critical_top1']:.2f}% ± {s['std_critical_top1']:.2f}%"
        )
        print(
            f"    Top-5: {s['avg_critical_top5']:.2f}% ± {s['std_critical_top5']:.2f}%"
        )
        print()
        print(f"  Risk Categories:")
        print(
            f"    High risk:   {s['high_risk']} ({100 * s['high_risk'] / s['n_runs']:.1f}%)"
        )
        print(
            f"    Medium risk: {s['medium_risk']} ({100 * s['medium_risk'] / s['n_runs']:.1f}%)"
        )
        print(f"    Safe:        {s['safe']} ({100 * s['safe'] / s['n_runs']:.1f}%)")
