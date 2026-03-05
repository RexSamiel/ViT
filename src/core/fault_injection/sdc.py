"""SDC (Silent Data Corruption) metrics - SDCTracker with all SDC-related methods."""

import math
import torch

from src.core.fault_injection.accuracy import AccuracyTracker


_THRESHOLD_PCTS = [1, 5, 10, 15, 20, 25, 50]


class SDCTracker:
    """Tracks per-run SDC metrics and multi-run aggregation.

    All SDC computation functions are methods of this class:
    - Per-sample: logit_sdc_rate, sdc_magnitude, relative_sdc
    - Critical: critical_top1, critical_top5, nan_adjustment
    - Batch-level: compute_batch_sdc, summarize_sdc
    """

    def __init__(self):
        self.reset()
        self.reset_aggregation()

    # Static computation methods

    @staticmethod
    def logit_sdc_rate(faulty: torch.Tensor, faultfree: torch.Tensor) -> torch.Tensor:
        """Percentage of logits that changed per sample. NaN counts as changed.

        Args:
            faulty: Faulty logits [batch_size, num_classes]
            faultfree: Fault-free logits [batch_size, num_classes]

        Returns:
            1D tensor of per-sample SDC rates [batch_size]
        """
        diff = faultfree - faulty
        return (diff != 0).float().mean(dim=1)

    @staticmethod
    def sdc_magnitude(faulty: torch.Tensor, faultfree: torch.Tensor) -> torch.Tensor:
        """Mean absolute difference per sample.

        Args:
            faulty: Faulty logits [batch_size, num_classes]
            faultfree: Fault-free logits [batch_size, num_classes]

        Returns:
            1D tensor of per-sample magnitudes [batch_size]
        """
        diff = faultfree - faulty
        return diff.abs().mean(dim=1)

    @staticmethod
    def relative_sdc(faulty: torch.Tensor, faultfree: torch.Tensor) -> dict:
        """Compute relative SDC at multiple thresholds.

        Args:
            faulty: Faulty logits [batch_size, num_classes]
            faultfree: Fault-free logits [batch_size, num_classes]

        Returns:
            Dictionary with keys "sdc_1", "sdc_5", ..., "sdc_50" - each a 1D tensor
            of per-sample flags. Returns empty dict if no valid nonzero samples.
        """
        diff = faultfree - faulty
        abs_diff_mean = diff.abs().mean(dim=1)
        abs_ff_mean = faultfree.abs().mean(dim=1)

        nonzero = abs_ff_mean != 0
        if not nonzero.any():
            return {}

        relative = abs_diff_mean[nonzero] / abs_ff_mean[nonzero]

        levels = torch.stack(
            [
                (relative >= 0.01).float(),
                (relative >= 0.05).float(),
                (relative >= 0.10).float(),
                (relative >= 0.15).float(),
                (relative >= 0.20).float(),
                (relative >= 0.25).float(),
                (relative >= 0.50).float(),
            ],
            dim=0,
        )

        return {
            "sdc_1": levels[0],
            "sdc_5": levels[1],
            "sdc_10": levels[2],
            "sdc_15": levels[3],
            "sdc_20": levels[4],
            "sdc_25": levels[5],
            "sdc_50": levels[6],
        }

    @staticmethod
    def critical_top1(faulty: torch.Tensor, faultfree: torch.Tensor) -> float:
        """Rate of samples where top-1 prediction changed AND the original top-1 logit was corrupted.

        Args:
            faulty: Faulty logits [batch_size, num_classes]
            faultfree: Fault-free logits [batch_size, num_classes]

        Returns:
            Float in range [0, 1] representing the rate
        """
        pred_faulty = faulty.argmax(dim=1)
        pred_ff = faultfree.argmax(dim=1)
        pred_changed = pred_faulty != pred_ff

        idx = torch.arange(faulty.size(0), device=faulty.device)
        logit_changed = faulty[idx, pred_ff] != faultfree[idx, pred_ff]

        return (logit_changed & pred_changed).float().mean().item()

    @staticmethod
    def critical_top5(faulty: torch.Tensor, faultfree: torch.Tensor) -> float:
        """Rate of samples where top-5 set changed AND original top-5 logits were corrupted.

        Args:
            faulty: Faulty logits [batch_size, num_classes]
            faultfree: Fault-free logits [batch_size, num_classes]

        Returns:
            Float in range [0, 1] representing the rate
        """
        top5_faulty = faulty.topk(5, dim=1)[1]
        top5_ff = faultfree.topk(5, dim=1)[1]

        # Set comparison via sorting
        ff_sorted, _ = top5_ff.sort(dim=1)
        faulty_sorted, _ = top5_faulty.sort(dim=1)
        set_changed = (ff_sorted != faulty_sorted).any(dim=1)

        # Logit comparison
        ff_logits = torch.gather(faultfree, 1, top5_ff)
        faulty_logits = torch.gather(faulty, 1, top5_ff)
        logits_changed = (ff_logits != faulty_logits).any(dim=1)

        return (logits_changed & set_changed).float().mean().item()

    @staticmethod
    def nan_adjustment(
        critical_top1_rate: float,
        critical_top5_rate: float,
        nan_mask: torch.Tensor,
        has_valid: bool,
    ) -> tuple[float, float]:
        """Adjust critical SDC rates for NaN samples (NaN = 100% critical SDC).

        Args:
            critical_top1_rate: Current critical top-1 SDC rate for valid samples
            critical_top5_rate: Current critical top-5 SDC rate for valid samples
            nan_mask: Boolean mask [batch_size] indicating NaN samples
            has_valid: Whether there are any valid (non-NaN) samples

        Returns:
            Tuple of (adjusted_crit_top1, adjusted_crit_top5)
        """
        num_nan = nan_mask.sum().item()
        total = nan_mask.size(0)

        if has_valid:
            num_valid = total - num_nan
            adjusted_top1 = (critical_top1_rate * num_valid + num_nan) / total
            adjusted_top5 = (critical_top5_rate * num_valid + num_nan) / total
            return adjusted_top1, adjusted_top5
        else:
            return 1.0, 1.0

    @classmethod
    def compute_batch_sdc(cls, faulty: torch.Tensor, faultfree: torch.Tensor) -> dict:
        """Compute all SDC metrics for a single batch.

        Returns dict with keys:
            sdc_rate: tensor of per-sample SDC rates
            magnitude: tensor of per-sample magnitudes (or None if all NaN)
            relative_levels: dict of threshold tensors (or None)
            critical_top1: float (or None)
            critical_top5: float (or None)
            nan_mask: boolean tensor
            has_valid: bool
            has_nan: bool
            all_nan: bool
        """
        nan_mask = torch.isnan(faulty).all(dim=1)
        has_valid = (~nan_mask).any().item()
        has_nan = nan_mask.any().item()
        all_nan = nan_mask.all().item()

        # Logit SDC rate (always computed, NaN counts as changed)
        sdc_rate_val = cls.logit_sdc_rate(faulty, faultfree)

        result = {
            "sdc_rate": sdc_rate_val.detach().cpu(),
            "magnitude": None,
            "relative_levels": None,
            "critical_top1": None,
            "critical_top5": None,
            "nan_mask": nan_mask,
            "has_valid": has_valid,
            "has_nan": has_nan,
            "all_nan": all_nan,
        }

        if has_valid:
            valid_faulty = faulty[~nan_mask]
            valid_ff = faultfree[~nan_mask]

            result["magnitude"] = (
                cls.sdc_magnitude(valid_faulty, valid_ff).detach().cpu()
            )

            rel = cls.relative_sdc(valid_faulty, valid_ff)
            if rel:
                result["relative_levels"] = {
                    k: v.detach().cpu() for k, v in rel.items()
                }

            crit1 = cls.critical_top1(valid_faulty, valid_ff)
            crit5 = cls.critical_top5(valid_faulty, valid_ff)

            if has_nan:
                crit1, crit5 = cls.nan_adjustment(crit1, crit5, nan_mask, has_valid)

            result["critical_top1"] = crit1
            result["critical_top5"] = crit5
        elif all_nan:
            result["critical_top1"] = 1.0
            result["critical_top5"] = 1.0

        return result

    @staticmethod
    def summarize_sdc(
        sdc_rates: list,
        sdc_magnitudes: list,
        sdc_levels: dict[str, list],
        critical_top1_rates: list[float],
        critical_top5_rates: list[float],
        batches_all_nan: int,
        batches_partial_nan: int,
        total_batches: int,
    ) -> dict:
        """Summarize accumulated per-batch SDC data into final metrics."""
        results = {}

        # Logit SDC rate
        results["logit_sdc_rate"] = (
            100 * torch.cat(sdc_rates).mean().item() if sdc_rates else 0.0
        )

        # SDC magnitude
        results["msdc_avg"] = (
            torch.cat(sdc_magnitudes).mean().item() if sdc_magnitudes else float("nan")
        )

        # Critical SDC
        if critical_top1_rates:
            results["critical_top1_sdc_rate"] = (
                100 * sum(critical_top1_rates) / len(critical_top1_rates)
            )
            results["critical_top5_sdc_rate"] = (
                100 * sum(critical_top5_rates) / len(critical_top5_rates)
            )
        else:
            results["critical_top1_sdc_rate"] = 0.0
            results["critical_top5_sdc_rate"] = 0.0

        # Relative SDC thresholds
        threshold_keys = [
            "sdc_1",
            "sdc_5",
            "sdc_10",
            "sdc_15",
            "sdc_20",
            "sdc_25",
            "sdc_50",
        ]
        pct_keys = [
            "sdc_1pct",
            "sdc_5pct",
            "sdc_10pct",
            "sdc_15pct",
            "sdc_20pct",
            "sdc_25pct",
            "sdc_50pct",
        ]

        if sdc_levels.get("sdc_1"):
            for tkey, pkey in zip(threshold_keys, pct_keys):
                results[pkey] = 100 * torch.cat(sdc_levels[tkey]).mean().item()
        else:
            for pkey in pct_keys:
                results[pkey] = float("nan")

        results["batches_all_nan"] = batches_all_nan
        results["batches_partial_nan"] = batches_partial_nan
        results["total_batches"] = total_batches

        return results

    # Per-run state

    def reset(self) -> None:
        """Reset per-run SDC accumulators."""
        self.sdc_rates: list[torch.Tensor] = []
        self.sdc_magnitudes: list[torch.Tensor] = []

        self.sdc_levels: dict[str, list[torch.Tensor]] = {
            f"sdc_{pct}": [] for pct in _THRESHOLD_PCTS
        }

        self.critical_top1_rates: list[float] = []
        self.critical_top5_rates: list[float] = []

        self.batches_all_nan = 0
        self.batches_partial_nan = 0
        self.total_batches = 0

    def update_batch(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> None:
        """Compute and accumulate SDC metrics for a single batch."""
        self.total_batches += 1

        batch_sdc = self.compute_batch_sdc(faulty, faultfree)

        if batch_sdc["all_nan"]:
            self.batches_all_nan += 1
        elif batch_sdc["has_nan"]:
            self.batches_partial_nan += 1

        self.sdc_rates.append(batch_sdc["sdc_rate"])

        if batch_sdc["magnitude"] is not None:
            self.sdc_magnitudes.append(batch_sdc["magnitude"])

        if batch_sdc["relative_levels"] is not None:
            for key in self.sdc_levels:
                self.sdc_levels[key].append(batch_sdc["relative_levels"][key])

        if batch_sdc["critical_top1"] is not None:
            self.critical_top1_rates.append(batch_sdc["critical_top1"])
            self.critical_top5_rates.append(batch_sdc["critical_top5"])

    def get_results(self) -> dict:
        """Return summarized SDC metrics for current run."""
        return self.summarize_sdc(
            self.sdc_rates,
            self.sdc_magnitudes,
            self.sdc_levels,
            self.critical_top1_rates,
            self.critical_top5_rates,
            self.batches_all_nan,
            self.batches_partial_nan,
            self.total_batches,
        )

    # Multi-run aggregation state

    def reset_aggregation(self) -> None:
        """Reset multi-run SDC aggregation state."""
        self.n_runs = 0

        self.avg_logit_sdc = 0.0

        # Relative SDC (with counted tracking)
        self._avg_pct = {pct: 0.0 for pct in _THRESHOLD_PCTS}
        self._pct_counted = {pct: 0 for pct in _THRESHOLD_PCTS}

        # MSDC
        self.avg_msdc = 0.0
        self.worst_msdc = 0.0
        self.msdc_counted = 0

        # Critical SDC (with Welford's)
        self.avg_critical_top1 = 0.0
        self.avg_critical_top5 = 0.0
        self.m2_critical_top1 = 0.0
        self.m2_critical_top5 = 0.0

        # Risk categories
        self.high_risk = 0
        self.medium_risk = 0
        self.safe = 0

        # NaN totals
        self.total_batches_all_nan = 0
        self.total_batches_partial_nan = 0
        self.total_batches_all = 0

    def aggregate_run(self, run_results: dict) -> None:
        """Aggregate a run's SDC results into multi-run statistics."""
        self.n_runs += 1

        # Logit SDC
        logit_sdc = run_results.get("logit_sdc_rate", 0.0)
        self.avg_logit_sdc = (
            self.avg_logit_sdc * (self.n_runs - 1) + logit_sdc
        ) / self.n_runs

        # Relative SDC thresholds
        for pct in _THRESHOLD_PCTS:
            val = run_results.get(f"sdc_{pct}pct", 0.0)
            if not math.isnan(val):
                self._pct_counted[pct] += 1
                cnt = self._pct_counted[pct]
                self._avg_pct[pct] = (self._avg_pct[pct] * (cnt - 1) + val) / cnt

        # MSDC
        msdc = run_results.get("msdc_avg", None)
        if msdc is not None and not math.isnan(msdc):
            self.msdc_counted += 1
            self.avg_msdc = (
                self.avg_msdc * (self.msdc_counted - 1) + msdc
            ) / self.msdc_counted
            self.worst_msdc = max(self.worst_msdc, msdc)

        # Critical SDC (Welford's)
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

        # NaN tracking
        self.total_batches_all_nan += run_results.get("batches_all_nan", 0)
        self.total_batches_partial_nan += run_results.get("batches_partial_nan", 0)
        self.total_batches_all += run_results.get("total_batches", 0)

    def get_summary(self) -> dict:
        """Return aggregated SDC summary."""
        summary = {
            "total_runs": self.n_runs,
            "avg_logit_sdc": self.avg_logit_sdc,
        }

        for pct in _THRESHOLD_PCTS:
            summary[f"avg_sdc_{pct}pct"] = self._avg_pct[pct]

        summary.update(
            {
                "avg_msdc": self.avg_msdc if self.msdc_counted > 0 else None,
                "worst_msdc": self.worst_msdc if self.msdc_counted > 0 else None,
                "msdc_counted_runs": self.msdc_counted,
                "avg_critical_top1_sdc": self.avg_critical_top1,
                "std_critical_top1_sdc": AccuracyTracker.compute_std(
                    self.m2_critical_top1, self.n_runs
                ),
                "avg_critical_top5_sdc": self.avg_critical_top5,
                "std_critical_top5_sdc": AccuracyTracker.compute_std(
                    self.m2_critical_top5, self.n_runs
                ),
                "high_risk_count": self.high_risk,
                "high_risk_pct": 100 * self.high_risk / self.n_runs
                if self.n_runs
                else 0.0,
                "medium_risk_count": self.medium_risk,
                "medium_risk_pct": 100 * self.medium_risk / self.n_runs
                if self.n_runs
                else 0.0,
                "safe_count": self.safe,
                "safe_pct": 100 * self.safe / self.n_runs if self.n_runs else 0.0,
                "batches_all_nan": self.total_batches_all_nan,
                "batches_partial_nan": self.total_batches_partial_nan,
                "total_batches": self.total_batches_all,
            }
        )

        return summary

    def print_summary(self) -> None:
        """Print aggregated SDC summary."""
        s = self.get_summary()

        thresholds = "\n".join(
            f"  Average SDC >= {p}%:{' ' * (12 - len(str(p)))}{s[f'avg_sdc_{p}pct']:.2f}%"
            for p in _THRESHOLD_PCTS
        )
        print(
            f"Logit SDC Metrics:\n"
            f"  Average Logit SDC Rate:       {s['avg_logit_sdc']:.2f}%\n"
            f"{thresholds}"
        )

        if s.get("avg_msdc") is not None:
            print(
                f"\nMSDC Metrics:\n"
                f"  Average MSDC:                 {s['avg_msdc']:.6f}\n"
                f"  Worst MSDC:                   {s['worst_msdc']:.6f}"
            )
        else:
            print("\nMSDC Metrics: No valid values")

        print(
            f"\nCritical SDC Metrics:\n"
            f"  Average Critical Top-1 SDC:   {s['avg_critical_top1_sdc']:.2f}% +/- {s['std_critical_top1_sdc']:.2f}%\n"
            f"  Average Critical Top-5 SDC:   {s['avg_critical_top5_sdc']:.2f}% +/- {s['std_critical_top5_sdc']:.2f}%"
        )

        if s.get("total_runs", 0) > 0:
            print(
                f"\nRisk Categories:\n"
                f"  High risk (top-1 changed):    {s['high_risk_pct']:>6.2f}% ({s['high_risk_count']} runs)\n"
                f"  Medium risk (top-5 changed):  {s['medium_risk_pct']:>6.2f}% ({s['medium_risk_count']} runs)\n"
                f"  Safe (no changes):            {s['safe_pct']:>6.2f}% ({s['safe_count']} runs)"
            )

        if s.get("batches_all_nan", 0) > 0 or s.get("batches_partial_nan", 0) > 0:
            print(
                f"\nNaN Batch Statistics:\n"
                f"  Batches all NaN:              {s['batches_all_nan']}/{s['total_batches']}\n"
                f"  Batches partial NaN:          {s['batches_partial_nan']}/{s['total_batches']}"
            )
