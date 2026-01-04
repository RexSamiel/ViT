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
    def __init__(self, config: Config, n_runs: int, mode: str, verbose: bool):
        self.config = config
        self.n_runs = n_runs
        self.mode = mode.lower()
        self.verbose = verbose
        self.runner = Runner(config, verbose=verbose)
        self.original_state = copy.deepcopy(self.runner.evaluator.model.state_dict())

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
        """Execute all runs sequentially."""
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

    args = parser.parse_args()

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

    # Override config if arguments are provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.max_batches is not None:
        config.max_batches = args.max_batches

    if args.model not in SUPPORTED_MODELS:
        print(f"Error: '{args.model}' is not supported.")
        print_supported_models()
        return

    config.model_key = args.model
    config.model_name = SUPPORTED_MODELS[args.model]

    if args.repeat <= 1:
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

    else:
        manager = ExperimentManager(config, args.repeat, args.mode, verbose=verbose)
        analyzer, total_runtime = manager.run_all(
            idx=args.idx,
            block_idx=args.block_idx,
            bit_range=bit_range,
            component=args.component,
            sub_component=args.sub_component,
        )

        # Save summary to file (append mode)
        os.makedirs("results", exist_ok=True)
        summary_path = os.path.join("results", f"summary_{args.model}_{args.mode}.json")

        existing_results = []
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    existing_results = json.load(f)
                    if not isinstance(existing_results, list):
                        existing_results = [existing_results]
            except (json.JSONDecodeError, IOError):
                existing_results = []

        import datetime

        new_summary = analyzer.get_summary()
        new_summary["timestamp"] = datetime.datetime.now().isoformat()

        # Calculate total samples safely for the summary
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
            "bit_range": bit_range,
            "component": args.component,
            "sub_component": args.sub_component,
        }
        existing_results.append(new_summary)

        with open(summary_path, "w") as f:
            json.dump(existing_results, f, indent=2)
        print(
            f"Summary appended to: {summary_path} (total experiments: {len(existing_results)})"
        )


if __name__ == "__main__":
    main()

