import math
import torch


class SDCMetrics:
    """Handles all SDC (Silent Data Corruption) metrics computation and aggregation."""

    def __init__(self):
        self.reset()
        self.reset_aggregation()

    # ==================== Single Run Methods ====================

    def reset(self) -> None:
        """Clear data for new run."""
        # Logit SDC
        self.sdc_rates: list[torch.Tensor] = []

        # SDC magnitude
        self.sdc_magnitudes: list[torch.Tensor] = []

        # Relative SDC thresholds
        self.sdc_1_levels: list[torch.Tensor] = []
        self.sdc_5_levels: list[torch.Tensor] = []
        self.sdc_10_levels: list[torch.Tensor] = []
        self.sdc_15_levels: list[torch.Tensor] = []
        self.sdc_20_levels: list[torch.Tensor] = []
        self.sdc_25_levels: list[torch.Tensor] = []
        self.sdc_50_levels: list[torch.Tensor] = []

        # Critical SDC
        self.critical_top1_rates: list[float] = []
        self.critical_top5_rates: list[float] = []

        # NaN tracking
        self.batches_all_nan = 0
        self.batches_partial_nan = 0
        self.total_batches = 0

    def update(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> None:
        """Update SDC metrics with faulty and fault-free logits."""
        self.total_batches += 1

        # Identify NaN samples
        nan_mask = torch.isnan(faulty).all(dim=1)
        has_valid = (~nan_mask).any()
        has_nan = nan_mask.any()

        # Track NaN batches
        if nan_mask.all():
            self.batches_all_nan += 1
        elif has_nan:
            self.batches_partial_nan += 1

        # Logit SDC rate (all samples, NaN = 100% changed)
        self._logit_sdc_rate(faulty, faultfree)

        # Other metrics only for valid samples
        if has_valid:
            valid_faulty = faulty[~nan_mask]
            valid_ff = faultfree[~nan_mask]

            self._sdc_magnitude(valid_faulty, valid_ff)
            self._relative_sdc(valid_faulty, valid_ff)
            self.critical_top1_rates.append(self._critical_top1(valid_faulty, valid_ff))
            self.critical_top5_rates.append(self._critical_top5(valid_faulty, valid_ff))

        # NaN adjustment for critical SDC
        if has_nan:
            self._nan_adjustment(nan_mask, has_valid)

    # -------------------- Internal Metric Methods --------------------

    def _logit_sdc_rate(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> None:
        """Percentage of logits that changed. NaN counts as changed."""
        diff = faultfree - faulty
        sdc_rate = (diff != 0).float().mean(dim=1)
        self.sdc_rates.append(sdc_rate.detach().cpu())

    def _sdc_magnitude(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> None:
        """Mean absolute difference between faulty and faultfree logits."""
        diff = faultfree - faulty
        magnitude = diff.abs().mean(dim=1)
        self.sdc_magnitudes.append(magnitude.detach().cpu())

    def _relative_sdc(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> None:
        """Percentage of samples where relative change >= threshold."""
        diff = faultfree - faulty
        abs_diff_mean = diff.abs().mean(dim=1)
        abs_ff_mean = faultfree.abs().mean(dim=1)

        # Exclude zero division
        nonzero = abs_ff_mean != 0
        if not nonzero.any():
            return

        relative = abs_diff_mean[nonzero] / abs_ff_mean[nonzero]

        levels = (
            torch.stack(
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
            .detach()
            .cpu()
        )

        self.sdc_1_levels.append(levels[0])
        self.sdc_5_levels.append(levels[1])
        self.sdc_10_levels.append(levels[2])
        self.sdc_15_levels.append(levels[3])
        self.sdc_20_levels.append(levels[4])
        self.sdc_25_levels.append(levels[5])
        self.sdc_50_levels.append(levels[6])

    def _critical_top1(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> float:
        """Samples where faultfree top-1 logit changed AND prediction changed."""
        pred_faulty = faulty.argmax(dim=1)
        pred_ff = faultfree.argmax(dim=1)
        pred_changed = pred_faulty != pred_ff

        idx = torch.arange(faulty.size(0), device=faulty.device)
        logit_changed = faulty[idx, pred_ff] != faultfree[idx, pred_ff]

        return (logit_changed & pred_changed).float().mean().item()

    def _critical_top5(self, faulty: torch.Tensor, faultfree: torch.Tensor) -> float:
        """Samples where any faultfree top-5 logit changed AND top-5 set changed."""
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

    def _nan_adjustment(self, nan_mask: torch.Tensor, has_valid: bool) -> None:
        """NaN = catastrophic failure = 100% critical SDC."""
        num_nan = nan_mask.sum().item()
        total = nan_mask.size(0)

        if has_valid:
            num_valid = total - num_nan
            self.critical_top1_rates[-1] = (
                self.critical_top1_rates[-1] * num_valid + num_nan
            ) / total
            self.critical_top5_rates[-1] = (
                self.critical_top5_rates[-1] * num_valid + num_nan
            ) / total
        else:
            self.critical_top1_rates.append(1.0)
            self.critical_top5_rates.append(1.0)

    # -------------------- Results Methods --------------------

    def get_results(self) -> dict:
        """Return SDC metrics for current run."""
        results = {}

        # Logit SDC rate
        if self.sdc_rates:
            results["logit_sdc_rate"] = 100 * torch.cat(self.sdc_rates).mean().item()
        else:
            results["logit_sdc_rate"] = 0.0

        # SDC magnitude
        if self.sdc_magnitudes:
            results["msdc_avg"] = torch.cat(self.sdc_magnitudes).mean().item()
        else:
            results["msdc_avg"] = float("nan")

        # Critical SDC
        if self.critical_top1_rates:
            results["critical_top1_sdc_rate"] = (
                100 * sum(self.critical_top1_rates) / len(self.critical_top1_rates)
            )
            results["critical_top5_sdc_rate"] = (
                100 * sum(self.critical_top5_rates) / len(self.critical_top5_rates)
            )
        else:
            results["critical_top1_sdc_rate"] = 0.0
            results["critical_top5_sdc_rate"] = 0.0

        # Relative SDC thresholds
        if self.sdc_1_levels:
            results["sdc_1pct"] = 100 * torch.cat(self.sdc_1_levels).mean().item()
            results["sdc_5pct"] = 100 * torch.cat(self.sdc_5_levels).mean().item()
            results["sdc_10pct"] = 100 * torch.cat(self.sdc_10_levels).mean().item()
            results["sdc_15pct"] = 100 * torch.cat(self.sdc_15_levels).mean().item()
            results["sdc_20pct"] = 100 * torch.cat(self.sdc_20_levels).mean().item()
            results["sdc_25pct"] = 100 * torch.cat(self.sdc_25_levels).mean().item()
            results["sdc_50pct"] = 100 * torch.cat(self.sdc_50_levels).mean().item()
        else:
            for k in [
                "sdc_1pct",
                "sdc_5pct",
                "sdc_10pct",
                "sdc_15pct",
                "sdc_20pct",
                "sdc_25pct",
                "sdc_50pct",
            ]:
                results[k] = float("nan")

        # NaN statistics
        results["batches_all_nan"] = self.batches_all_nan
        results["batches_partial_nan"] = self.batches_partial_nan
        results["total_batches"] = self.total_batches

        return results

    def print_results(self) -> None:
        """Print SDC results for current run."""
        r = self.get_results()

        print("SDC Metrics:")
        print(f"  Logit SDC Rate:          {r['logit_sdc_rate']:.2f}%")

        if not math.isnan(r["msdc_avg"]):
            print(f"  MSDC Average:            {r['msdc_avg']:.6f}")

        print(f"  Critical Top-1 SDC:      {r['critical_top1_sdc_rate']:.2f}%")
        print(f"  Critical Top-5 SDC:      {r['critical_top5_sdc_rate']:.2f}%")

        print("Relative SDC Thresholds:")
        for pct in [1, 5, 10, 15, 20, 25, 50]:
            val = r[f"sdc_{pct}pct"]
            if not math.isnan(val):
                print(f"  SDC >= {pct}%:              {val:.2f}%")

        if r["batches_all_nan"] > 0 or r["batches_partial_nan"] > 0:
            print("NaN Statistics:")
            print(
                f"  Batches all NaN:         {r['batches_all_nan']}/{r['total_batches']}"
            )
            print(
                f"  Batches partial NaN:     {r['batches_partial_nan']}/{r['total_batches']}"
            )

    # ==================== Aggregation Methods ====================

    def reset_aggregation(self) -> None:
        """Clear aggregation data."""
        self.n_runs = 0

        # Logit SDC
        self.avg_logit_sdc = 0.0

        # Relative SDC (with counted tracking)
        self.avg_sdc_1pct = 0.0
        self.sdc_1pct_counted = 0
        self.avg_sdc_5pct = 0.0
        self.sdc_5pct_counted = 0
        self.avg_sdc_10pct = 0.0
        self.sdc_10pct_counted = 0
        self.avg_sdc_15pct = 0.0
        self.sdc_15pct_counted = 0
        self.avg_sdc_20pct = 0.0
        self.sdc_20pct_counted = 0
        self.avg_sdc_25pct = 0.0
        self.sdc_25pct_counted = 0
        self.avg_sdc_50pct = 0.0
        self.sdc_50pct_counted = 0

        # MSDC
        self.avg_msdc = 0.0
        self.worst_msdc = 0.0
        self.msdc_counted = 0

        # Critical SDC (with Welford's online variance)
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

    def aggregate(self, run_results: dict) -> None:
        """Aggregate results from a single run."""
        self.n_runs += 1

        # Logit SDC
        logit_sdc = run_results.get("logit_sdc_rate", 0.0)
        self.avg_logit_sdc = (
            self.avg_logit_sdc * (self.n_runs - 1) + logit_sdc
        ) / self.n_runs

        # Relative SDC thresholds
        for pct, attr_avg, attr_cnt in [
            (1, "avg_sdc_1pct", "sdc_1pct_counted"),
            (5, "avg_sdc_5pct", "sdc_5pct_counted"),
            (10, "avg_sdc_10pct", "sdc_10pct_counted"),
            (15, "avg_sdc_15pct", "sdc_15pct_counted"),
            (20, "avg_sdc_20pct", "sdc_20pct_counted"),
            (25, "avg_sdc_25pct", "sdc_25pct_counted"),
            (50, "avg_sdc_50pct", "sdc_50pct_counted"),
        ]:
            val = run_results.get(f"sdc_{pct}pct", 0.0)
            if not math.isnan(val):
                cnt = getattr(self, attr_cnt) + 1
                setattr(self, attr_cnt, cnt)
                avg = getattr(self, attr_avg)
                setattr(self, attr_avg, (avg * (cnt - 1) + val) / cnt)

        # MSDC
        msdc = run_results.get("msdc_avg", None)
        if msdc is not None and not math.isnan(msdc):
            self.msdc_counted += 1
            self.avg_msdc = (
                self.avg_msdc * (self.msdc_counted - 1) + msdc
            ) / self.msdc_counted
            self.worst_msdc = max(self.worst_msdc, msdc)

        # Critical SDC (Welford's online algorithm for mean and variance)
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

        # NaN totals
        self.total_batches_all_nan += run_results.get("batches_all_nan", 0)
        self.total_batches_partial_nan += run_results.get("batches_partial_nan", 0)
        self.total_batches_all += run_results.get("total_batches", 0)

    def _std(self, m2: float) -> float:
        """Compute standard deviation from M2 (Welford's algorithm)."""
        if self.n_runs < 2:
            return 0.0
        return (m2 / (self.n_runs - 1)) ** 0.5

    def get_summary(self) -> dict:
        """Return aggregated SDC summary."""
        return {
            "total_runs": self.n_runs,
            "avg_logit_sdc": self.avg_logit_sdc,
            "avg_sdc_1pct": self.avg_sdc_1pct,
            "avg_sdc_5pct": self.avg_sdc_5pct,
            "avg_sdc_10pct": self.avg_sdc_10pct,
            "avg_sdc_15pct": self.avg_sdc_15pct,
            "avg_sdc_20pct": self.avg_sdc_20pct,
            "avg_sdc_25pct": self.avg_sdc_25pct,
            "avg_sdc_50pct": self.avg_sdc_50pct,
            "avg_msdc": self.avg_msdc if self.msdc_counted > 0 else None,
            "worst_msdc": self.worst_msdc if self.msdc_counted > 0 else None,
            "msdc_counted_runs": self.msdc_counted,
            "avg_critical_top1_sdc": self.avg_critical_top1,
            "std_critical_top1_sdc": self._std(self.m2_critical_top1),
            "avg_critical_top5_sdc": self.avg_critical_top5,
            "std_critical_top5_sdc": self._std(self.m2_critical_top5),
            "high_risk_count": self.high_risk,
            "high_risk_pct": 100 * self.high_risk / self.n_runs if self.n_runs else 0.0,
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

    def print_summary(self) -> None:
        """Print aggregated SDC summary."""
        print("Logit SDC Metrics:")
        print(f"  Average Logit SDC Rate:       {self.avg_logit_sdc:.2f}%")
        print(f"  Average SDC >= 1%:            {self.avg_sdc_1pct:.2f}%")
        print(f"  Average SDC >= 5%:            {self.avg_sdc_5pct:.2f}%")
        print(f"  Average SDC >= 10%:           {self.avg_sdc_10pct:.2f}%")
        print(f"  Average SDC >= 15%:           {self.avg_sdc_15pct:.2f}%")
        print(f"  Average SDC >= 20%:           {self.avg_sdc_20pct:.2f}%")
        print(f"  Average SDC >= 25%:           {self.avg_sdc_25pct:.2f}%")
        print(f"  Average SDC >= 50%:           {self.avg_sdc_50pct:.2f}%")

        if self.msdc_counted > 0:
            print("\nMSDC Metrics:")
            print(f"  Average MSDC:                 {self.avg_msdc:.6f}")
            print(f"  Worst MSDC:                   {self.worst_msdc:.6f}")
        else:
            print("\nMSDC Metrics: No valid values")

        std_top1 = self._std(self.m2_critical_top1)
        std_top5 = self._std(self.m2_critical_top5)

        print("\nCritical SDC Metrics:")
        print(f"  Average Critical Top-1 SDC:   {self.avg_critical_top1:.2f}% ± {std_top1:.2f}%")
        print(f"  Average Critical Top-5 SDC:   {self.avg_critical_top5:.2f}% ± {std_top5:.2f}%")

        high_pct = 100 * self.high_risk / self.n_runs if self.n_runs else 0.0
        med_pct = 100 * self.medium_risk / self.n_runs if self.n_runs else 0.0
        safe_pct = 100 * self.safe / self.n_runs if self.n_runs else 0.0

        print("\nRisk Categories:")
        print(
            f"  High risk (top-1 changed):    {high_pct:>6.2f}% ({self.high_risk} runs)"
        )
        print(
            f"  Medium risk (top-5 changed):  {med_pct:>6.2f}% ({self.medium_risk} runs)"
        )
        print(f"  Safe (no changes):            {safe_pct:>6.2f}% ({self.safe} runs)")

        if self.total_batches_all_nan > 0 or self.total_batches_partial_nan > 0:
            print("\nNaN Batch Statistics:")
            print(
                f"  Batches all NaN:              {self.total_batches_all_nan}/{self.total_batches_all}"
            )
            print(
                f"  Batches partial NaN:          {self.total_batches_partial_nan}/{self.total_batches_all}"
            )
