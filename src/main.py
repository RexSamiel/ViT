"""Main entry point for Vision Transformer Fault Injection Framework."""

import argparse
import random

from src.config.settings import Config
from src.core.model import ModelRunner
from src.core import activation, fault_injection
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
  aa    Activation analysis - profile activation distributions across layers
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared arguments (before subparsers)
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

    # Subparsers for modes
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Analysis mode",
    )

    # Fault Injection mode
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
        help="Bit range to target (e.g., '0-7' for sign/exponent, '8-31' for mantissa)",
    )

    # Activation Analysis mode
    aa_parser = subparsers.add_parser(
        "aa",
        help="Activation analysis mode",
        description="""Activation Analysis Mode

Profile activation value distributions across all model layers.
Useful for understanding model behavior and identifying outlier layers.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    aa_parser.add_argument(
        "--sampling",
        type=float,
        default=1.0,
        help="Percentage of activations to sample per layer (default: 1.0%%, min: 0.01%%)",
    )

    return parser.parse_args()


def run_fault_injection(
    args, runner: ModelRunner, verbose: bool, show_info: bool, seed: int
) -> None:
    """Run fault injection workflow.

    Steps:
      1. If faulty mode, load baseline accuracy for comparison
      2. If single run, execute once and optionally save baseline
      3. If multiple runs, aggregate results and save summary

    Args:
        args: Parsed command line arguments
        runner: ModelRunner instance
        verbose: Whether to print output
        show_info: Whether to show per-run info in multi-run mode
        seed: Random seed for reproducibility
    """
    save_logits = str_to_bool(args.save_logits)

    fault_params = {
        "component": args.component,
        "sub_component": args.sub_component,
        "idx": args.idx,
        "block_idx": args.block_idx,
        "bit_range": parse_bit_range(args.bit_range),
        "repeat": args.repeat,
    }

    total_blocks = get_num_blocks(runner.model)

    # Load baseline accuracy if running faulty mode
    base_accuracy = None
    if args.condition == "faulty":
        base_accuracy = fault_injection.load_base_accuracy(args.model)
        if base_accuracy and verbose:
            print(
                f"Loaded base accuracy: Top-1={base_accuracy['top1']:.2f}%, Top-5={base_accuracy['top5']:.2f}%"
            )

    # Execute fault injection
    if args.repeat <= 1:
        # Single run
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
        # Multiple runs with aggregation
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
        )


def run_activation_analysis(args, runner: ModelRunner, verbose: bool) -> None:
    """Run activation analysis workflow.

    Steps:
      1. Profile activations once and save results
      2. Print layer details if verbose

    Args:
        args: Parsed command line arguments
        runner: ModelRunner instance
        verbose: Whether to print output
    """
    results, analyzer = activation.run(
        runner,
        sampling_percent=args.sampling,
        verbose=verbose,
    )

    if verbose:
        analyzer.print_layer_ranges()

    activation.save_results(results, runner.config, args.model)


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

    # Validate model
    if args.model not in SUPPORTED_MODELS:
        print(f"Error: '{args.model}' not supported.")
        print_supported_models()
        return

    # Configure settings
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

    if args.mode == "fi":
        show_info = str_to_bool(args.info)
        run_fault_injection(args, runner, verbose, show_info, seed)
    elif args.mode == "aa":
        run_activation_analysis(args, runner, verbose)


if __name__ == "__main__":
    main()
