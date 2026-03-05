import argparse
import random

from src.config.settings import Config
from src.core.model import ModelRunner
from src.core import parameter_analysis, fault_injection, fault_detection
from src.core.library.layers import get_num_blocks
from src.core.library.ui import SUPPORTED_MODELS, print_supported_models
from src.core.library.utils import set_seed, str_to_bool, int_or_none, parse_bit_range


def parse_args():
    """Parse command line arguments with mode-specific subparsers."""
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
  Add --detection <type> to enable fault detection on the specified layer type(s).
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
        help=("Enable fault detection on the specified layer type(s). (default: none)"),
    )
    fi_parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help=("Relative difference threshold for fault detection. (default: 0.1)"),
    )

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

    fd_parser = subparsers.add_parser(
        "fd",
        help="Fault detection mode",
        description="""Fault Detection Mode

Test fault detection using checker neurons. Wraps linear layers with
extra neurons that compute expected average outputs based on precomputed
average weights. Compares checker neuron output with actual output average
to detect faults.

Workflow:
  1. Run without injection first to see baseline detection values
  2. Run with --inject to inject a fault and see if detection catches it
  3. Adjust --threshold to tune detection sensitivity
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fd_parser.add_argument(
        "--inject",
        type=str,
        default="false",
        help="Inject a fault before running detection (default: false)",
    )
    fd_parser.add_argument(
        "--layer_filter",
        type=str,
        default="all",
        choices=["all", "qkv", "proj", "fc1", "fc2"],
        help="Which layers to apply detection to (default: all)",
    )
    fd_parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Difference threshold for fault detection (default: 1e-3)",
    )
    fd_parser.add_argument(
        "--bit_range",
        type=str,
        default=None,
        help="Bit range to target for injection (e.g., '0,7' or '24,31')",
    )
    fd_parser.add_argument(
        "--fault_count",
        type=int,
        default=1,
        help="Number of faults to inject (default: 1)",
    )
    fd_parser.add_argument(
        "--relative",
        type=str,
        default="false",
        help="Use relative difference for threshold (default: false)",
    )
    fd_parser.add_argument(
        "--load_weights",
        type=str,
        default="true",
        help="Load pre-saved checker weights from disk (default: true)",
    )
    fd_parser.add_argument(
        "--recompute",
        type=str,
        default="false",
        help="Force recompute weights even if cached (default: false)",
    )

    return parser.parse_args()


def run_fault_injection(
    args,
    runner: ModelRunner,
    verbose: bool,
    show_info: bool,
    seed: int,
    output_dir: str,
) -> None:
    """Run fault injection workflow.

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

    if use_detection:
        if verbose:
            print(
                f"Detection requested: type={detection}, threshold={threshold} "
                f"(NOT IMPLEMENTED - implement in fault_detection module)"
            )

    base_accuracy = None
    if args.condition == "faulty":
        base_accuracy = fault_injection.load_base_accuracy(args.model)
        if base_accuracy and verbose:
            print(
                f"Loaded base accuracy: Top-1={base_accuracy['top1']:.2f}%, "
                f"Top-5={base_accuracy['top5']:.2f}%"
            )

    if args.repeat <= 1:
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
        summary = fault_injection.run_multiple(
            runner,
            args.repeat,
            fault_params,
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


def run_parameter_analysis(
    args, runner: ModelRunner, verbose: bool, output_dir: str
) -> None:
    """Run parameter analysis workflow.

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

    parameter_analysis.save_results(
        results, runner.config, args.model, args.type, output_dir
    )


def run_fault_detection(args, runner: ModelRunner, verbose: bool) -> None:
    """Run fault detection workflow."""
    import torch

    do_inject = str_to_bool(args.inject)
    layer_filter = args.layer_filter
    bit_range = parse_bit_range(args.bit_range)
    threshold = args.threshold
    fault_count = args.fault_count
    relative = str_to_bool(args.relative)
    load_weights = str_to_bool(args.load_weights)
    recompute = str_to_bool(args.recompute)

    detector = fault_detection.FaultDetector(runner.model, threshold=threshold)

    if load_weights:
        detector.load_weights(args.model, force_recompute=recompute)

    detector.apply(layer_filter)

    batches = runner.get_batches()
    if not batches:
        print("No batches available")
        return

    images, _ = batches[0]

    # Inject faults if requested
    injector = None
    if do_inject:
        injector = fault_detection.Injector()
        injector.inject(runner.model, layer_filter, bit_range, count=fault_count)
        injector.print_info()

    # Run inference
    with torch.inference_mode():
        _ = runner.model(images)

    # Print results
    detector.print_values(relative=relative)

    # Cleanup
    if injector:
        injector.restore()
    detector.remove()


def main():
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
    elif args.mode == "fd":
        run_fault_detection(args, runner, verbose)


if __name__ == "__main__":
    main()
