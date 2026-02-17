"""Main entry point for Vision Transformer Fault Injection Framework."""

import argparse
import random

from src.config.settings import Config
from src.core.model import ModelRunner
from src.core import parameter_analysis, fault_injection
from src.core import fault_detection
from src.core.fault_detection.tracker import DetectionTracker
from src.core.library.layers import get_num_blocks
from src.core.library.ui import SUPPORTED_MODELS, print_supported_models
from src.core.library.utils import set_seed, str_to_bool, int_or_none, parse_bit_range


def parse_args():
    """Parse command line arguments with mode-specific subparsers."""
    # Main parser with shared arguments
    parser = argparse.ArgumentParser(
        description="""Vision Transformer Fault Injection & Analysis Framework

Analyze Vision Transformer models through fault injection experiments
and activation distribution analysis.

Modes:
  fi    Fault injection - inject bit-flips into model weights and measure impact
  pa    Parameter analysis - analyze activation or weight distributions
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key to run (e.g., vit_base, deit_small)",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default=None,
        help="Batch size (int or 'None' to use full batches)",
    )
    parser.add_argument(
        "--max_batches",
        type=str,
        default=None,
        help="Max batches (int or 'None' for all batches)",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="true",
        help="Print verbose output (default: true)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/data/new_runs)",
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Analysis mode",
    )

    fi_parser = subparsers.add_parser(
        "fi",
        help="Fault injection mode",
        description="""Fault Injection Mode

Inject single-bit faults into model parameters and measure accuracy
degradation and Silent Data Corruption (SDC) metrics.

Workflow:
  1. Run with --condition faultfree first to establish baseline
  2. Run with --condition faulty to inject faults and compare
  3. Use --repeat N for statistical significance

Detection:
  Add --detection <type> to monitor layer input sums during the run.
  On a faultfree run the baseline is captured and saved.
  On a faulty run the current values are compared against the saved baseline.
  --detection choices: none, qkv, proj, fc1, fc2, all  (default: none)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fi_parser.add_argument(
        "--condition",
        type=str,
        choices=["faultfree", "faulty"],
        default="faultfree",
        help="Run condition: faultfree (baseline) or faulty (inject faults)",
    )
    fi_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of fault injection runs (default: 1)",
    )
    fi_parser.add_argument(
        "--info",
        type=str,
        default="false",
        help="Show per-run info in multi-run mode (default: false)",
    )
    fi_parser.add_argument(
        "--save_logits",
        type=str,
        default="false",
        help="Save fault-free logits for analysis (default: false)",
    )
    fi_parser.add_argument(
        "--component",
        type=str,
        default="all",
        choices=["mlp", "norm", "attention", "patch_embed", "classifier", "all"],
        help="Component type to target for fault injection (default: all)",
    )
    fi_parser.add_argument(
        "--sub_component",
        type=str,
        default=None,
        help="Specific sub-component name (e.g., fc1, fc2 for MLP)",
    )
    fi_parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Specific parameter index within component",
    )
    fi_parser.add_argument(
        "--block_idx",
        type=int,
        default=None,
        help="Specific transformer block index to target",
    )
    fi_parser.add_argument(
        "--bit_range",
        type=str,
        default=None,
        help="Bit range to target",
    )
    fi_parser.add_argument(
        "--detection",
        type=str,
        default="none",
        choices=["none", "qkv", "proj", "fc1", "fc2", "all"],
        help=(
            "Enable fault detection on the specified layer type(s). "
            "On a faultfree run the baseline is saved; on a faulty run the "
            "baseline is loaded and results are compared. (default: none)"
        ),
    )
    fi_parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help=(
            "Relative difference threshold for fault detection. "
            "A layer is flagged when |current - baseline| / |baseline| exceeds "
            "this value. (default: 0.1)"
        ),
    )

    # Parameter Analysis mode
    pa_parser = subparsers.add_parser(
        "pa",
        help="Parameter analysis mode",
        description="""Parameter Analysis Mode

Analyze value distributions across model parameters.

Types:
  aa    Activation analysis - profile runtime activation distributions (requires data)
  wa    Weight analysis - analyze static weight distributions (no data needed)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pa_parser.add_argument(
        "--type",
        type=str,
        choices=["aa", "wa"],
        required=True,
        help="Analysis type: aa (activations) or wa (weights)",
    )
    pa_parser.add_argument(
        "--sampling",
        type=float,
        default=1.0,
        help="Percentage of activations to sample (aa only, default: 1.0%%, min: 0.01%%)",
    )
    pa_parser.add_argument(
        "--details",
        type=str,
        default="false",
        help="Print per-parameter/layer details (default: false)",
    )

    return parser.parse_args()


def run_fault_injection(
    args, runner: ModelRunner, verbose: bool, show_info: bool, seed: int, output_dir: str
) -> None:
    """Run fault injection workflow, optionally with integrated fault detection.

    Steps:
      1. If detection is enabled, set up detection neurons on the model
      2. If faulty mode, load baseline accuracy for comparison
      3. Execute single or multiple runs (detection values are captured inline)
      4. On faultfree + detection: save captured baseline values to disk
      5. On faulty + detection: compare against saved baseline and print table

    Args:
        args: Parsed command line arguments
        runner: ModelRunner instance
        verbose: Whether to print output
        show_info: Whether to show per-run info in multi-run mode
        seed: Random seed for reproducibility
        output_dir: Directory to write result files
    """
    save_logits = str_to_bool(args.save_logits)
    detection = args.detection
    threshold = args.threshold
    use_detection = detection != "none"

    fault_params = {
        "component": args.component,
        "sub_component": args.sub_component,
        "idx": args.idx,
        "block_idx": args.block_idx,
        "bit_range": parse_bit_range(args.bit_range),
        "repeat": args.repeat,
    }

    total_blocks = get_num_blocks(runner.model)

    # --- Detection setup (faultfree: capture baseline inline) ---
    # For a faultfree run with detection we let the existing run_single loop
    # handle the forward passes, but we attach hooks beforehand and peel off
    # detection values afterwards.  For a faulty run we attach hooks so that
    # detection happens in the same forward pass as fault injection.

    detection_neurons: dict = {}
    detection_tracker: DetectionTracker | None = None

    if use_detection:
        detection_neurons = fault_detection.add_detection(
            runner.model, detection, total_blocks
        )
        detection_tracker = DetectionTracker(threshold=threshold)
        if verbose:
            print(
                f"Detection enabled: monitoring {len(detection_neurons)} layers "
                f"(type={detection}, threshold={threshold})"
            )

    # Load baseline accuracy if running faulty mode
    base_accuracy = None
    if args.condition == "faulty":
        base_accuracy = fault_injection.load_base_accuracy(args.model)
        if base_accuracy and verbose:
            print(
                f"Loaded base accuracy: Top-1={base_accuracy['top1']:.2f}%, "
                f"Top-5={base_accuracy['top5']:.2f}%"
            )

    # --- Inference runs ---
    if args.repeat <= 1:
        # Single run — run_single executes the forward loop internally.
        # We need access to the per-batch hook values so we wrap the loop here
        # when detection is active rather than delegating to run_single.
        if use_detection:
            results = _run_single_with_detection(
                runner,
                args.condition,
                save_logits,
                fault_params if args.condition == "faulty" else None,
                detection_neurons,
                detection_tracker,
                verbose,
            )
        else:
            results = fault_injection.run_single(
                runner,
                mode=args.condition,
                save_logits=save_logits,
                fault_params=fault_params if args.condition == "faulty" else None,
                verbose=verbose,
            )

        if args.condition == "faultfree":
            fault_injection.save_base_accuracy(args.model, results)
    else:
        # Multiple runs — detection on multi-run uses the last run's tracker
        # to show a representative comparison.  We accumulate across all runs.
        summary = _run_multiple_with_detection(
            runner,
            args.repeat,
            fault_params,
            detection_neurons if use_detection else {},
            detection_tracker,
            verbose=verbose,
            show_info=show_info,
        )
        fault_injection.save_summary(
            summary=summary,
            config=runner.config,
            model_key=args.model,
            mode=args.condition,
            fault_config=fault_params,
            base_accuracy=base_accuracy,
            seed=seed,
            total_blocks=total_blocks,
            output_dir=output_dir,
        )

    # --- Post-run detection processing ---
    if use_detection:
        fault_detection.remove_detection(detection_neurons)

        if args.condition == "faultfree":
            # Save the baseline means captured during this run.
            _save_detection_baseline(
                detection_tracker, detection, args.model, verbose
            )
        else:
            # Compare against saved baseline and print results.
            results = fault_detection.detect_faults(
                runner,
                detection=detection,
                model_key=args.model,
                threshold=threshold,
                current_tracker=detection_tracker,
                verbose=verbose,
            )
            fault_detection.print_results(results, threshold)


# ---------------------------------------------------------------------------
# Detection-aware single-run wrapper
# ---------------------------------------------------------------------------


def _run_single_with_detection(
    runner: ModelRunner,
    mode: str,
    save_logits: bool,
    fault_params: dict | None,
    detection_neurons: dict,
    detection_tracker: DetectionTracker,
    verbose: bool,
) -> dict:
    """Execute one evaluation pass with detection hooks already attached.

    Replicates the core logic of :func:`fault_injection.run_single` but calls
    :func:`fault_detection.update_tracker` after every batch so that detection
    values are captured in the same forward pass as the accuracy metrics.

    Args:
        runner: ModelRunner instance.
        mode: "faultfree" or "faulty".
        save_logits: Whether to buffer logits for saving.
        fault_params: Fault injection parameters dict (faulty mode only).
        detection_neurons: Active detection neuron dict.
        detection_tracker: Tracker accumulating detection values.
        verbose: Print progress.

    Returns:
        Results dict from the fault injection run.
    """
    from src.core.fault_injection.manager import FaultInjection
    from src.core.fault_injection.injection import Injector
    from src.core.fault_injection.accuracy import AccuracyTracker
    import time
    import torch
    from src.core.library.utils import resolve_amp

    fi = FaultInjection()
    use_amp = resolve_amp(runner.config)

    fi.injector.restore()
    fi.reset()

    fault_info = None
    if mode == "faulty" and fault_params:
        fault_info = fi.injector.inject(runner.model, fault_params)
        if verbose:
            print(Injector.format_fault_info(fault_info))

    batches = runner.get_batches()
    compute_sdc = mode == "faulty" and runner.ff_logits.available
    logits_buffer, labels_buffer = [], []

    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(batches):
            outputs = runner.inference(images, use_amp)
            fi.process_batch(outputs, labels, batch_idx, runner, compute_sdc)

            # Capture detection values from this batch.
            fault_detection.update_tracker(detection_neurons, detection_tracker)

            if save_logits:
                logits_buffer.append(outputs.cpu())
                labels_buffer.append(labels.cpu())

    runtime = time.perf_counter() - start_time

    if save_logits and logits_buffer:
        runner.ff_logits.save(logits_buffer, labels_buffer)

    results = fi.get_results(compute_sdc, fault_info, runtime, mode)

    if verbose:
        fi.print_run_results(results, runtime, runner.config.model_name)

    return results


def _run_multiple_with_detection(
    runner: ModelRunner,
    n_runs: int,
    fault_params: dict,
    detection_neurons: dict,
    detection_tracker: DetectionTracker | None,
    verbose: bool,
    show_info: bool,
) -> dict:
    """Execute multiple fault-injection runs, accumulating detection across all.

    Detection values are accumulated across all runs so the tracker reflects
    the average behaviour over all injected-fault scenarios.

    Args:
        runner: ModelRunner instance.
        n_runs: Number of runs.
        fault_params: Fault injection parameters dict.
        detection_neurons: Active detection neurons (may be empty dict).
        detection_tracker: Tracker to accumulate into (may be None).
        verbose: Print summary.
        show_info: Print per-run info.

    Returns:
        Aggregated summary dict.
    """
    use_detection = bool(detection_neurons) and detection_tracker is not None

    if not use_detection:
        # Fast path: no detection overhead.
        return fault_injection.run_multiple(
            runner,
            n_runs,
            fault_params,
            verbose=verbose,
            show_info=show_info,
        )

    from src.core.fault_injection.manager import FaultInjection
    import time

    fi = FaultInjection()
    total_start = time.perf_counter()

    for i in range(n_runs):
        if verbose and show_info:
            print(f"Run {i + 1}/{n_runs}\n{'-' * 60}")

        results = _run_single_with_detection(
            runner,
            mode="faulty",
            save_logits=False,
            fault_params=fault_params,
            detection_neurons=detection_neurons,
            detection_tracker=detection_tracker,
            verbose=show_info,
        )
        fi.aggregate_run(results)

    total_runtime = time.perf_counter() - total_start

    if verbose:
        fi.print_summary(n_runs, total_runtime)

    return fi.get_summary(total_runtime)


# ---------------------------------------------------------------------------
# Baseline persistence helper
# ---------------------------------------------------------------------------


def _save_detection_baseline(
    tracker: DetectionTracker,
    detection: str,
    model_key: str,
    verbose: bool,
) -> None:
    """Persist per-layer detection means (sum, avg, min) via DetectionBaseline.

    Args:
        tracker: Populated tracker from a faultfree run.
        detection: Detection config string used as part of the filename.
        model_key: Model identifier.
        verbose: Print confirmation.
    """
    from src.core.fault_detection.baseline import DetectionBaseline

    means = tracker.get_means()
    if not means:
        if verbose:
            print("No detection values captured; baseline not saved.")
        return

    # means is already dict[str, dict[str, float]] - pass directly.
    baseline = DetectionBaseline(model_key=model_key, detection=detection)
    baseline.save(means)

    if verbose:
        print(f"Detection baseline saved ({len(means)} layers).")


# ---------------------------------------------------------------------------
# Parameter analysis workflow
# ---------------------------------------------------------------------------


def run_parameter_analysis(args, runner: ModelRunner, verbose: bool, output_dir: str) -> None:
    """Run parameter analysis workflow.

    Steps:
      1. Run activation or weight analysis based on --type
      2. Print details if requested
      3. Save results

    Args:
        args: Parsed command line arguments
        runner: ModelRunner instance
        verbose: Whether to print output
        output_dir: Directory to write result files
    """
    show_details = str_to_bool(args.details)

    results, analyzer = parameter_analysis.run(
        runner,
        analysis_type=args.type,
        sampling_percent=args.sampling,
        verbose=verbose,
    )

    if show_details:
        analyzer.print_details()

    parameter_analysis.save_results(results, runner.config, args.model, args.type, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Main entry point.

    Workflow:
      1. Parse arguments with mode-specific validation
      2. Validate model selection
      3. Configure settings (batch size, max batches)
      4. Set reproducibility seed
      5. Load model via ModelRunner
      6. Route to selected mode workflow
    """
    args = parse_args()

    if args.model not in SUPPORTED_MODELS:
        print(f"Error: '{args.model}' not supported.")
        print_supported_models()
        return

    config = Config()
    config.model_key = args.model
    config.model_name = SUPPORTED_MODELS[args.model]
    if args.batch_size is not None:
        config.batch_size = int_or_none(args.batch_size)
    if args.max_batches is not None:
        config.max_batches = int_or_none(args.max_batches)

    verbose = str_to_bool(args.verbose)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    set_seed(seed)
    if verbose:
        print(f"Random seed: {seed}")

    runner = ModelRunner(config, verbose)

    output_dir = args.output_dir

    if args.mode == "fi":
        show_info = str_to_bool(args.info)
        run_fault_injection(args, runner, verbose, show_info, seed, output_dir)
    elif args.mode == "pa":
        run_parameter_analysis(args, runner, verbose, output_dir)


if __name__ == "__main__":
    main()
