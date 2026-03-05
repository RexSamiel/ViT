"""SDC (Silent Data Corruption) metrics computation."""

import torch


class SDCTracker:
    """Tracks SDC metrics per-run and aggregates across multiple runs."""

    def __init__(self):
        self.reset()
        self.reset_aggregation()

    def reset(self):
        """Reset per-run accumulators."""
        self.sdc_rates = []
        self.critical_top1_rates = []
        self.critical_top5_rates = []
        self.total_batches = 0

    def update_batch(self, faulty: torch.Tensor, faultfree: torch.Tensor):
        """Compute and accumulate SDC metrics for a batch."""
        self.total_batches += 1

        # Handle NaN samples
        nan_mask = torch.isnan(faulty).all(dim=1)
        has_valid = (~nan_mask).any().item()

        # Logit SDC rate (NaN counts as changed)
        diff = faultfree - faulty
        sdc_rate = (diff != 0).float().mean(dim=1)
        self.sdc_rates.append(sdc_rate.cpu())

        if has_valid:
            valid_faulty = faulty[~nan_mask]
            valid_ff = faultfree[~nan_mask]

            # Critical Top-1: prediction changed AND original logit corrupted
            pred_faulty = valid_faulty.argmax(dim=1)
            pred_ff = valid_ff.argmax(dim=1)
            pred_changed = pred_faulty != pred_ff

            idx = torch.arange(valid_faulty.size(0), device=faulty.device)
            logit_changed = valid_faulty[idx, pred_ff] != valid_ff[idx, pred_ff]
            crit_top1 = (logit_changed & pred_changed).float().mean().item()

            # Critical Top-5: top-5 set changed AND original logits corrupted
            top5_faulty = valid_faulty.topk(5, dim=1)[1]
            top5_ff = valid_ff.topk(5, dim=1)[1]

            ff_sorted, _ = top5_ff.sort(dim=1)
            faulty_sorted, _ = top5_faulty.sort(dim=1)
            set_changed = (ff_sorted != faulty_sorted).any(dim=1)

            ff_logits = torch.gather(valid_ff, 1, top5_ff)
            faulty_logits = torch.gather(valid_faulty, 1, top5_ff)
            logits_changed = (ff_logits != faulty_logits).any(dim=1)
            crit_top5 = (logits_changed & set_changed).float().mean().item()

            # Adjust for NaN samples (NaN = 100% critical)
            if nan_mask.any():
                num_nan = nan_mask.sum().item()
                total = nan_mask.size(0)
                num_valid = total - num_nan
                crit_top1 = (crit_top1 * num_valid + num_nan) / total
                crit_top5 = (crit_top5 * num_valid + num_nan) / total

            self.critical_top1_rates.append(crit_top1)
            self.critical_top5_rates.append(crit_top5)
        elif nan_mask.all():
            self.critical_top1_rates.append(1.0)
            self.critical_top5_rates.append(1.0)

    def get_results(self) -> dict:
        """Return summarized SDC metrics for current run."""
        sdc_rate = torch.cat(self.sdc_rates).mean().item() * 100 if self.sdc_rates else 0.0

        crit_top1 = (
            sum(self.critical_top1_rates) / len(self.critical_top1_rates) * 100
            if self.critical_top1_rates else 0.0
        )
        crit_top5 = (
            sum(self.critical_top5_rates) / len(self.critical_top5_rates) * 100
            if self.critical_top5_rates else 0.0
        )

        return {
            "logit_sdc_rate": sdc_rate,
            "critical_top1_sdc_rate": crit_top1,
            "critical_top5_sdc_rate": crit_top5,
            "batches": self.total_batches,
        }

    # Multi-run aggregation

    def reset_aggregation(self):
        """Reset multi-run aggregation state."""
        self.n_runs = 0
        self.avg_sdc_rate = 0.0
        self.avg_critical_top1 = 0.0
        self.avg_critical_top5 = 0.0
        self.m2_critical_top1 = 0.0
        self.m2_critical_top5 = 0.0
        self.high_risk = 0
        self.medium_risk = 0
        self.safe = 0

    def aggregate_run(self, run_results: dict):
        """Aggregate a run's SDC results into multi-run statistics."""
        self.n_runs += 1

        # Simple average for SDC rate
        sdc = run_results.get("logit_sdc_rate", 0.0)
        self.avg_sdc_rate = (self.avg_sdc_rate * (self.n_runs - 1) + sdc) / self.n_runs

        # Welford's for critical SDC
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

    def get_summary(self) -> dict:
        """Return aggregated SDC summary."""
        std_top1 = (self.m2_critical_top1 / (self.n_runs - 1)) ** 0.5 if self.n_runs > 1 else 0.0
        std_top5 = (self.m2_critical_top5 / (self.n_runs - 1)) ** 0.5 if self.n_runs > 1 else 0.0

        return {
            "avg_sdc_rate": self.avg_sdc_rate,
            "avg_critical_top1": self.avg_critical_top1,
            "avg_critical_top5": self.avg_critical_top5,
            "std_critical_top1": std_top1,
            "std_critical_top5": std_top5,
            "high_risk": self.high_risk,
            "medium_risk": self.medium_risk,
            "safe": self.safe,
            "n_runs": self.n_runs,
        }

    def print_summary(self):
        """Print aggregated SDC summary."""
        s = self.get_summary()
        print(f"SDC Metrics ({s['n_runs']} runs):")
        print(f"  Logit SDC Rate: {s['avg_sdc_rate']:.2f}%")
        print(f"  Critical Top-1: {s['avg_critical_top1']:.2f}% ± {s['std_critical_top1']:.2f}%")
        print(f"  Critical Top-5: {s['avg_critical_top5']:.2f}% ± {s['std_critical_top5']:.2f}%")
        print(f"Risk Categories:")
        print(f"  High risk:   {s['high_risk']} runs ({100*s['high_risk']/s['n_runs']:.1f}%)")
        print(f"  Medium risk: {s['medium_risk']} runs ({100*s['medium_risk']/s['n_runs']:.1f}%)")
        print(f"  Safe:        {s['safe']} runs ({100*s['safe']/s['n_runs']:.1f}%)")
