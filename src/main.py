import argparse
import copy
import json
import os
import time
import torch

from src.core.runner import Runner
from src.config.settings import Config
from src.utils.helper import SUPPORTED_MODELS, print_supported_models
from src.fault_injector.fault_injection import inject_fault
from src.core.analyzer import RunAnalyzer


class ExperimentManager:
    """Manages multiple runs efficiently without saving to file."""

    def __init__(self, config: Config, n_runs: int, mode: str, verbose: bool):
        self.config = config
        self.n_runs = n_runs
        self.mode = mode.lower()
        self.verbose = verbose
        self.runner = Runner(config, verbose=verbose)  # Pass verbose to runner
        self.original_state = copy.deepcopy(self.runner.evaluator.model.state_dict())

    def reset_model(self):
        """Restore model to original (fault-free) state before each run."""
        self.runner.evaluator.model.load_state_dict(self.original_state)
        torch.cuda.empty_cache()

    def run_single(
        self,
        run_id: int,
        idx: int | None = None,
        block_idx: int | None = None,
        bit_range: tuple[int, int] | None = None,
    ) -> dict:
        """Run a single iteration, optionally injecting a fault."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"⚡ Running {self.mode} run #{run_id + 1}/{self.n_runs}")
            print(f"{'=' * 60}")

        start_time = time.perf_counter()

        fault_info = None
        if self.mode == "faulty":
            fault_info = inject_fault(
                self.runner.evaluator.model,
                component_type="attention",
                idx=idx,
                block_idx=block_idx,
                bit_range=bit_range,
                verbose=self.verbose,
            )
            if self.verbose:
                print("✓ Fault injection applied for this run.\n")

        # Run with compute_sdc enabled for faulty mode
        # In verbose mode, let runner print the full results
        results = self.runner.run(
            compute_metrics=True,
            save_logits=False,
            verbose=self.verbose,  # Let runner print if verbose
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

    def run_all(self, idx=None, block_idx=None, bit_range=None):
        """Execute all runs sequentially."""
        analyzer = RunAnalyzer()
        total_start = time.perf_counter()

        for i in range(self.n_runs):
            self.reset_model()
            run_result = self.run_single(
                i, idx=idx, block_idx=block_idx, bit_range=bit_range
            )
            analyzer.update(run_result)

        total_end = time.perf_counter()
        total_runtime = round(total_end - total_start, 2)

        print(f"\n✅ All {self.n_runs} {self.mode} runs completed.")
        print(f"🕒 Total repeated runtime: {total_runtime:.2f} seconds\n")

        analyzer.print_summary()
        return analyzer, total_runtime


def main():
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

    args = parser.parse_args()

    # Parse bit range (e.g., "1,25" -> (1,25))
    bit_range = None
    if args.bit_range:
        parts = args.bit_range.split(",")
        if len(parts) == 2:
            bit_range = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError("bit_range must be START,END")

    verbose = args.verbose.lower() in ("true", "1", "yes", "y")
    save_logits = args.save_logits.lower() in ("true", "1", "yes", "y")
    config = Config()

    # Validate model
    if args.model not in SUPPORTED_MODELS:
        print(f"Error: '{args.model}' is not supported.")
        print_supported_models()
        return

    config.model_key = args.model
    config.model_name = SUPPORTED_MODELS[args.model]

    # ========== SINGLE RUN ==========
    if args.repeat <= 1:
        runner = Runner(config, verbose=True)

        fault_info = None
        if args.mode == "faulty":
            fault_info = inject_fault(
                runner.evaluator.model,
                component_type="attention",
                idx=args.idx,
                block_idx=args.block_idx,
                bit_range=bit_range,
                verbose=True,  # Always show fault info in single run
            )
            print("✓ Fault injection applied.\n")

        # Run evaluation
        runner.run(
            compute_metrics=True,
            save_logits=save_logits,
            verbose=True,  # Print results in single run
            compute_sdc=(args.mode == "faulty"),
        )

    # ========== MULTI-RUN ==========
    else:
        manager = ExperimentManager(config, args.repeat, args.mode, verbose=verbose)
        analyzer, total_runtime = manager.run_all(
            idx=args.idx, block_idx=args.block_idx, bit_range=bit_range
        )

        # Save summary to file
        os.makedirs("results", exist_ok=True)
        summary_path = os.path.join("results", f"summary_{args.model}_{args.mode}.json")
        with open(summary_path, "w") as f:
            json.dump(analyzer.get_summary(), f, indent=2)
        print(f"📄 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
