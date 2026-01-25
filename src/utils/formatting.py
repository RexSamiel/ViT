"""Centralized formatting utilities for metrics display and output."""

from typing import Any


# Metric display configuration - single source of truth for all metrics
METRIC_CONFIGS = {
    # Accuracy metrics
    "samples": {"label": "Samples", "format": "{:.0f}"},
    "top1_acc": {"label": "Top-1 Accuracy", "format": "{:.2f}%"},
    "top5_acc": {"label": "Top-5 Accuracy", "format": "{:.2f}%"},
    "avg_top1_acc": {"label": "Average Top-1 Accuracy", "format": "{:.2f}%"},
    "avg_top5_acc": {"label": "Average Top-5 Accuracy", "format": "{:.2f}%"},
    "worst_top1_acc": {"label": "Worst Top-1 Accuracy", "format": "{:.2f}%"},
    "worst_top5_acc": {"label": "Worst Top-5 Accuracy", "format": "{:.2f}%"},
    "worst_top1_nonzero": {"label": "Worst Top-1 Accuracy (non-zero)", "format": "{:.2f}%"},
    "worst_top5_nonzero": {"label": "Worst Top-5 Accuracy (non-zero)", "format": "{:.2f}%"},

    # SDC metrics
    "logit_sdc_rate": {"label": "Logit SDC Rate", "format": "{:.2f}%"},
    "avg_logit_sdc": {"label": "Average Logit SDC Rate", "format": "{:.2f}%"},
    "sdc_1pct": {"label": "SDC ≥1%", "format": "{:.2f}%"},
    "sdc_5pct": {"label": "SDC ≥5%", "format": "{:.2f}%"},
    "sdc_10pct": {"label": "SDC ≥10%", "format": "{:.2f}%"},
    "sdc_15pct": {"label": "SDC ≥15%", "format": "{:.2f}%"},
    "sdc_20pct": {"label": "SDC ≥20%", "format": "{:.2f}%"},
    "sdc_25pct": {"label": "SDC ≥25%", "format": "{:.2f}%"},
    "sdc_50pct": {"label": "SDC ≥50%", "format": "{:.2f}%"},
    "avg_sdc_1pct": {"label": "Average SDC ≥1%", "format": "{:.2f}%"},
    "avg_sdc_5pct": {"label": "Average SDC ≥5%", "format": "{:.2f}%"},
    "avg_sdc_10pct": {"label": "Average SDC ≥10%", "format": "{:.2f}%"},
    "avg_sdc_15pct": {"label": "Average SDC ≥15%", "format": "{:.2f}%"},
    "avg_sdc_20pct": {"label": "Average SDC ≥20%", "format": "{:.2f}%"},
    "avg_sdc_25pct": {"label": "Average SDC ≥25%", "format": "{:.2f}%"},
    "avg_sdc_50pct": {"label": "Average SDC ≥50%", "format": "{:.2f}%"},

    # MSDC metrics
    "msdc_avg": {"label": "MSDC Average", "format": "{:.6f}"},
    "avg_msdc": {"label": "Average MSDC", "format": "{:.6f}"},
    "worst_msdc": {"label": "Worst MSDC", "format": "{:.6f}"},

    # Critical SDC metrics
    "critical_top1_sdc_rate": {"label": "Critical Top-1 SDC", "format": "{:.2f}%"},
    "critical_top5_sdc_rate": {"label": "Critical Top-5 SDC", "format": "{:.2f}%"},
    "avg_critical_top1_sdc": {"label": "Average Critical Top-1 SDC", "format": "{:.2f}%"},
    "avg_critical_top5_sdc": {"label": "Average Critical Top-5 SDC", "format": "{:.2f}%"},
}


def format_metric(key: str, value: Any) -> str:
    """Format a metric value according to its configured format."""
    if key not in METRIC_CONFIGS:
        return f"{value}"

    config = METRIC_CONFIGS[key]
    try:
        return config["format"].format(value)
    except (ValueError, TypeError):
        return str(value)


def format_metric_line(key: str, value: Any, label_width: int = 30) -> str:
    """Format a single metric line with label and value."""
    if key not in METRIC_CONFIGS:
        label = key.replace("_", " ").title()
    else:
        label = METRIC_CONFIGS[key]["label"]

    formatted_value = format_metric(key, value)
    return f"{label + ':':<{label_width}} {formatted_value}"


def print_run_results(results: dict[str, float], model_key: str, model_name: str) -> None:
    """Print results from a single run."""
    print("\n" + "=" * 50)
    print(f"RESULTS for {model_key} ({model_name})")
    print("=" * 50)

    # Always print basic accuracy metrics
    if "samples" in results:
        print(format_metric_line("samples", results["samples"]))
    if "top1_acc" in results:
        print(format_metric_line("top1_acc", results["top1_acc"]))
    if "top5_acc" in results:
        print(format_metric_line("top5_acc", results["top5_acc"]))

    # Print SDC metrics if available
    if results.get("logit_sdc_rate", 0.0) > 0:
        print(format_metric_line("logit_sdc_rate", results["logit_sdc_rate"]))
        print(format_metric_line("msdc_avg", results["msdc_avg"]))
        print(format_metric_line("sdc_1pct", results["sdc_1pct"]))
        print(format_metric_line("sdc_5pct", results["sdc_5pct"]))
        print(format_metric_line("sdc_10pct", results["sdc_10pct"]))
        print(format_metric_line("sdc_15pct", results["sdc_15pct"]))
        print(format_metric_line("sdc_20pct", results["sdc_20pct"]))
        print(format_metric_line("sdc_25pct", results["sdc_25pct"]))
        print(format_metric_line("sdc_50pct", results["sdc_50pct"]))
        print(format_metric_line("critical_top1_sdc_rate", results["critical_top1_sdc_rate"]))
        print(format_metric_line("critical_top5_sdc_rate", results["critical_top5_sdc_rate"]))

    print("=" * 50 + "\n")


def print_multi_run_summary(summary: dict[str, Any]) -> None:
    """Print summary statistics from multiple runs."""
    output = []
    output.append("=" * 60)
    output.append("ANALYSIS OF MULTI-RUN EXPERIMENT")
    output.append(f"Total runs: {summary['total_runs']}")
    output.append("")

    # Accuracy Metrics
    output.append("Accuracy Metrics:")
    output.append("  " + format_metric_line("avg_top1_acc", summary["avg_top1_acc"], 28))
    output.append("  " + format_metric_line("avg_top5_acc", summary["avg_top5_acc"], 28))

    # Worst accuracies
    if summary["worst_top1_acc"] > 0.0:
        output.append("  " + format_metric_line("worst_top1_acc", summary["worst_top1_acc"], 28))
    else:
        output.append("  " + format_metric_line("worst_top1_nonzero", summary["worst_top1_nonzero"], 28))

    if summary["worst_top5_acc"] > 0.0:
        output.append("  " + format_metric_line("worst_top5_acc", summary["worst_top5_acc"], 28))
    else:
        output.append("  " + format_metric_line("worst_top5_nonzero", summary["worst_top5_nonzero"], 28))

    output.append("")

    # Logit SDC Metrics
    output.append("Logit SDC Metrics:")
    output.append("  " + format_metric_line("avg_logit_sdc", summary["avg_logit_sdc"], 28))
    output.append("  " + format_metric_line("avg_sdc_1pct", summary["avg_sdc_1pct"], 28))
    output.append("  " + format_metric_line("avg_sdc_5pct", summary["avg_sdc_5pct"], 28))
    output.append("  " + format_metric_line("avg_sdc_10pct", summary["avg_sdc_10pct"], 28))
    output.append("  " + format_metric_line("avg_sdc_15pct", summary["avg_sdc_15pct"], 28))
    output.append("  " + format_metric_line("avg_sdc_20pct", summary["avg_sdc_20pct"], 28))
    output.append("  " + format_metric_line("avg_sdc_25pct", summary["avg_sdc_25pct"], 28))
    output.append("  " + format_metric_line("avg_sdc_50pct", summary["avg_sdc_50pct"], 28))
    output.append("")

    # MSDC Metrics
    msdc_counted = summary["msdc_counted_runs"]

    output.append("MSDC Metrics:")
    if msdc_counted > 0:
        output.append("  " + format_metric_line("avg_msdc", summary["avg_msdc"], 28))
        output.append("  " + format_metric_line("worst_msdc", summary["worst_msdc"], 28))
    else:
        output.append("  No valid MSDC values")
    output.append("")

    # Critical SDC Metrics
    output.append("Critical SDC Metrics:")
    output.append("  " + format_metric_line("avg_critical_top1_sdc", summary["avg_critical_top1_sdc"], 28))
    output.append("  " + format_metric_line("avg_critical_top5_sdc", summary["avg_critical_top5_sdc"], 28))
    output.append("")

    # Risk Categories
    output.append("Risk Categories:")
    high_risk_pct = summary["high_risk_pct"]
    high_risk_count = summary["high_risk_count"]
    medium_risk_pct = summary["medium_risk_pct"]
    medium_risk_count = summary["medium_risk_count"]
    safe_pct = summary["safe_pct"]
    safe_count = summary["safe_count"]

    output.append(f"  High risk (top-1 changed):    {high_risk_pct:>6.2f}% ({high_risk_count} runs)")
    output.append(f"  Medium risk (top-5 changed):  {medium_risk_pct:>6.2f}% ({medium_risk_count} runs)")
    output.append(f"  Safe (no changes):            {safe_pct:>6.2f}% ({safe_count} runs)")
    output.append("=" * 60)

    print("\n".join(output))


def format_fault_injection_info(fault_info: dict[str, Any]) -> str:
    """Format fault injection details for display."""
    sub_comp_str = ""
    if fault_info.get("sub_component"):
        sub_comp_str = f"\nSub-Component  : {fault_info['sub_component']}"

    original_bits = fault_info["original_bits"]
    corrupted_bits = fault_info["corrupted_bits"]

    # Format IEEE 754 bits
    def format_ieee754(bits_str: str) -> str:
        bits_str = bits_str.replace("-", "").replace("+", "").zfill(32)
        return f"Sign  Exponent   Mantissa\n {bits_str[0]}    {bits_str[1:9]}  {bits_str[9:]}"

    return f"""
Fault Injection Details
{"-" * 80}
Component Type : {fault_info['component_type']}{sub_comp_str}
Block Index    : {fault_info['block_idx']}
Parameter Name : {fault_info['param_name']}
Fault Index    : {fault_info['fault_idx']}
Bit Flipped    : {fault_info['bit_flipped']}
Original Value : {fault_info['original_value']:.8f}
Corrupted Value: {fault_info['corrupted_value']:.8f}

Original Bits:
{format_ieee754(original_bits)}

Corrupted Bits:
{format_ieee754(corrupted_bits)}
{"-" * 80}
"""
