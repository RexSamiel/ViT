"""Command-line interface for vit_fault."""

import argparse
import random

from vit_fault import Model, Detector, Injector, evaluate
from vit_fault.core.bits import set_seed
from vit_fault.core.model import SUPPORTED_MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="ViT Fault Injection and Detection Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to use",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=100,
        help="Batch size (default: 100)",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=1,
        help="Max batches to process (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random)",
    )

    # Detection options
    parser.add_argument(
        "--detect",
        type=str,
        default=None,
        choices=["all", "fc1", "fc2", "qkv", "proj"],
        help="Apply fault detection to layers",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Detection threshold (default: 0.1)",
    )

    # Injection options
    parser.add_argument(
        "--inject",
        type=str,
        default=None,
        choices=["all", "fc1", "fc2", "qkv", "proj"],
        help="Inject faults into layers",
    )
    parser.add_argument(
        "--faults",
        type=int,
        default=1,
        help="Number of faults to inject (default: 1)",
    )
    parser.add_argument(
        "--bit_range",
        type=str,
        default=None,
        help="Bit range for injection (e.g., '0,31')",
    )

    # Multi-run options
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of runs (default: 1)",
    )

    # Baseline/setup options
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compute and save fault-free logits (required for SDC metrics)",
    )

    # Results output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    return parser.parse_args()


def parse_bit_range(s: str) -> tuple[int, int] | None:
    if not s:
        return None
    parts = s.split(",")
    return (int(parts[0]), int(parts[1]))


def main():
    args = parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    set_seed(seed)
    print(f"Seed: {seed}")

    from vit_fault.core.model import ModelConfig

    config = ModelConfig(
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    model = Model(args.model, config=config)

    if args.baseline:
        model.save_baseline()
        print("\nBaseline saved. Run again without --baseline to inject faults.")
        return

    detector = None
    if args.detect:
        detector = Detector(model, layers=args.detect, threshold=args.threshold)

    injector = None
    if args.inject:
        bit_range = parse_bit_range(args.bit_range)
        injector = Injector(model, layers=args.inject, bit_range=bit_range)

    all_results = []

    for run in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- Run {run + 1}/{args.repeat} ---")

        if injector:
            injector.inject(count=args.faults)
            injector.print_info()

        results = evaluate(model, detector)
        results.print()

        if detector:
            detector.print_results()

        if args.output:
            run_data = {
                "run": run + 1,
                "seed": seed,
                "config": {
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "max_batches": args.max_batches,
                    "detect_layers": args.detect,
                    "inject_layers": args.inject,
                    "threshold": args.threshold,
                    "num_faults": args.faults,
                    "bit_range": args.bit_range,
                },
                "top1_acc": results.top1,
                "top5_acc": results.top5,
                "samples": results.samples,
            }
            if results.sdc_rate is not None:
                run_data["logit_sdc"] = results.sdc_rate
                run_data["critical_top1_sdc"] = results.critical_top1
                run_data["critical_top5_sdc"] = results.critical_top5
            if results.faults_detected > 0:
                run_data["faults_detected"] = results.faults_detected
                run_data["faulty_layers"] = results.faulty_layers
            if detector:
                run_data["detection_values"] = detector.get_values()
            if injector:
                run_data["injection_info"] = injector.get_info()
            all_results.append(run_data)

        if injector:
            injector.restore()

    if args.output:
        import json
        from pathlib import Path

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
