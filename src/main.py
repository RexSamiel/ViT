import argparse
import json
import os
import time
import torch

from src.core.runner import Runner
from src.config.settings import Config
from src.utils.helper import SUPPORTED_MODELS, print_supported_models
from src.fault_injector.fault_injection import inject_fault
from src.core.analyzer import RunAnalyzer
import datetime


class ExperimentManager:
    """Manages multiple experimental runs with model state management."""

    def __init__(self, config: Config, n_runs: int, mode: str, verbose: bool):
        self.config = config
        self.n_runs = n_runs
        self.mode = mode.lower()
        self.verbose = verbose
        self.runner = Runner(config, verbose=verbose)
        # Clone tensors individually instead of deep copy - more memory efficient
        self.original_state = {
            k: v.clone() for k, v in self.runner.evaluator.model.state_dict().items()
        }

    def reset_model(self):
        """Restore model to original (fault-free) state before each run."""
        self.runner.evaluator.model.load_state_dict(self.original_state)
        self.runner.evaluator.clear_cache()
        torch.cuda.empty_cache()

    def run_single(
        self,
        run_id: int,
        idx: int | None = None,
        block_idx: int | None = None,
        bit_range: tuple[int, int] | None = None,
        component: str | None = None,
        sub_component: str | None = None,
    ) -> dict:
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f" Running {self.mode} run #{run_id + 1}/{self.n_runs}")
            print(f"{'=' * 60}")

        start_time = time.perf_counter()

        fault_info = None
        if self.mode == "faulty":
            if component is None:
                component = "all"

            fault_info = inject_fault(
                self.runner.evaluator.model,
                component_type=component,
                sub_component=sub_component,
                idx=idx,
                block_idx=block_idx,
                bit_range=bit_range,
                verbose=self.verbose,
            )

        results = self.runner.run(
            compute_metrics=True,
            save_logits=False,
            verbose=self.verbose,
            compute_sdc=(self.mode == "faulty"),
        )
        end_time = time.perf_counter()
        runtime = round(end_time - start_time, 2)

        results_dict = results if results else {}
        results_dict.update(
            {
                "run_id": run_id + 1,
                "mode": self.mode,
                "fault_info": fault_info,
                "runtime_sec": runtime,
            }
        )

        return results_dict

    def run_all(
        self,
        idx=None,
        block_idx=None,
        bit_range=None,
        component=None,
        sub_component=None,
    ):
        """
        Execute all runs sequentially, resetting the model between each run.
        Returns analyzer with aggregated results and total runtime.
        """
        analyzer = RunAnalyzer()
        total_start = time.perf_counter()

        for i in range(self.n_runs):
            self.reset_model()
            run_result = self.run_single(
                i,
                idx=idx,
                block_idx=block_idx,
                bit_range=bit_range,
                component=component,
                sub_component=sub_component,
            )
            analyzer.update(run_result)

        total_end = time.perf_counter()
        total_runtime = round(total_end - total_start, 2)

        print(f"\nAll {self.n_runs} {self.mode} runs completed.")
        print(f"Total repeated runtime: {total_runtime:.2f} seconds\n")

        analyzer.print_summary()
        return analyzer, total_runtime


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision Transformer Fault Injection Framework"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["faultfree", "faulty"], default="faultfree"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", type=str, default="false")
    parser.add_argument(
        "--idx", type=int, default=None, help="Parameter index to inject fault"
    )
    parser.add_argument(
        "--block_idx", type=int, default=None, help="Block index for fault injection"
    )
    parser.add_argument(
        "--bit_range",
        type=str,
        default=None,
        help="Bit range in form START,END (e.g., 0,31)",
    )
    parser.add_argument(
        "--save_logits",
        type=str,
        default="false",
        help="Save fault-free logits (only for faultfree mode)",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="all",
        choices=["mlp", "norm", "attention", "patch_embed", "classifier", "all"],
        help="Component type for fault injection",
    )
    parser.add_argument(
        "--sub_component",
        type=str,
        default=None,
        help="Sub-component: 'qkv'/'proj' for attention, 'fc1'/'fc2' for mlp",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size from config"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Override max batches from config",
    )
    return parser.parse_args()


def parse_bit_range(bit_range_str):
    """Parse bit range string into tuple."""
    if not bit_range_str:
        return None
    parts = bit_range_str.split(",")
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    else:
        raise ValueError("bit_range must be START,END")


def str_to_bool(s):
    """Convert string argument to boolean."""
    return s.lower() in ("true", "1", "yes", "y")


def setup_config(args):
    """Setup configuration with command line argument overrides."""
    config = Config()

    if args.model not in SUPPORTED_MODELS:
        print(f"Error: '{args.model}' is not supported.")
        print_supported_models()
        return None

    config.model_key = args.model
    config.model_name = SUPPORTED_MODELS[args.model]

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.max_batches is not None:
        config.max_batches = args.max_batches

    return config


def save_experiment_summary(args, analyzer, config):
    """Save experiment summary to JSON file."""
    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", f"summary_{args.model}_{args.mode}.json")

    # Load existing results
    existing_results = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
        except (json.JSONDecodeError, IOError):
            existing_results = []

    # Create new summary
    new_summary = analyzer.get_summary()
    new_summary["timestamp"] = datetime.datetime.now().isoformat()

    # Calculate total samples
    samples_count = "all"
    if config.max_batches is not None:
        samples_count = config.batch_size * config.max_batches

    new_summary["config"] = {
        "model": args.model,
        "mode": args.mode,
        "repeat": args.repeat,
        "samples_per_run": samples_count,
        "batch_size": config.batch_size,
        "max_batches": config.max_batches,
        "idx": args.idx,
        "block_idx": args.block_idx,
        "bit_range": parse_bit_range(args.bit_range),
        "component": args.component,
        "sub_component": args.sub_component,
    }

    existing_results.append(new_summary)

    with open(summary_path, "w") as f:
        json.dump(existing_results, f, indent=2)

    print(f"Summary appended to: {summary_path} (total experiments: {len(existing_results)})")


def run_single_experiment(args, config, verbose, save_logits, bit_range):
    """Execute a single experimental run."""
    runner = Runner(config, verbose=True)

    fault_info = None
    if args.mode == "faulty":
        fault_info = inject_fault(
            runner.evaluator.model,
            component_type=args.component,
            sub_component=args.sub_component,
            idx=args.idx,
            block_idx=args.block_idx,
            bit_range=bit_range,
            verbose=True,
        )

    runner.run(
        compute_metrics=True,
        save_logits=save_logits,
        verbose=True,
        compute_sdc=(args.mode == "faulty"),
    )


def run_multi_experiment(args, config, verbose, bit_range):
    """Execute multiple experimental runs."""
    manager = ExperimentManager(config, args.repeat, args.mode, verbose=verbose)
    analyzer, total_runtime = manager.run_all(
        idx=args.idx,
        block_idx=args.block_idx,
        bit_range=bit_range,
        component=args.component,
        sub_component=args.sub_component,
    )
    save_experiment_summary(args, analyzer, config)


def main():
    args = parse_arguments()
    bit_range = parse_bit_range(args.bit_range)
    verbose = str_to_bool(args.verbose)
    save_logits = str_to_bool(args.save_logits)

    config = setup_config(args)
    if config is None:
        return

    if args.repeat <= 1:
        run_single_experiment(args, config, verbose, save_logits, bit_range)
    else:
        run_multi_experiment(args, config, verbose, bit_range)


if __name__ == "__main__":
    main()

