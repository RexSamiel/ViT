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

        # DEBUG: Store some reference weights to verify reset
        self.debug_weights = {}
        if hasattr(self.runner.evaluator.model, "blocks"):
            # Store a few weights from block 1 for verification
            self.debug_weights["block1_qkv_sample"] = (
                self.runner.evaluator.model.blocks[1].attn.qkv.weight[0, 0].item()
            )

    def reset_model(self):
        """Restore model to original (fault-free) state before each run."""
        # DEBUG: Check if model has accumulated faults
        if self.debug_weights and hasattr(self.runner.evaluator.model, "blocks"):
            current_val = (
                self.runner.evaluator.model.blocks[1].attn.qkv.weight[0, 0].item()
            )
            if abs(current_val - self.debug_weights["block1_qkv_sample"]) > 1e-6:
                print(
                    f"DEBUG: Model was modified (weight changed from "
                    f"{self.debug_weights['block1_qkv_sample']:.8f} to {current_val:.8f})"
                )

        self.runner.evaluator.model.load_state_dict(self.original_state)
        torch.cuda.empty_cache()

        # DEBUG: Verify reset worked
        if self.debug_weights and hasattr(self.runner.evaluator.model, "blocks"):
            after_reset = (
                self.runner.evaluator.model.blocks[1].attn.qkv.weight[0, 0].item()
            )
            if abs(after_reset - self.debug_weights["block1_qkv_sample"]) > 1e-9:
                print(
                    f"DEBUG: Reset FAILED! Weight is {after_reset:.8f}, "
                    f"should be {self.debug_weights['block1_qkv_sample']:.8f}"
                )
            elif self.verbose:
                print(f"✓ Model reset verified (weight restored to {after_reset:.8f})")

    def run_single(
        self,
        run_id: int,
        idx: int | None = None,
        block_idx: int | None = None,
        bit_range: tuple[int, int] | None = None,
        component: str | None = None,
    ) -> dict:
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f" Running {self.mode} run #{run_id + 1}/{self.n_runs}")
            print(f"{'=' * 60}")

        start_time = time.perf_counter()

        fault_info = None
        if self.mode == "faulty":
            # Ensure component is not None for the inject_fault call (for type safety)
            if component is None:
                raise ValueError(
                    "Cannot run in faulty mode: component type is missing."
                )

            fault_info = inject_fault(
                self.runner.evaluator.model,
                component_type=component,
                idx=idx,
                block_idx=block_idx,
                bit_range=bit_range,
                verbose=self.verbose,
            )
            if self.verbose:
                print("Fault injection applied for this run.")

                # DEBUG: Verify the fault was actually applied
                if fault_info:
                    param_name = fault_info["param_name"]
                    fault_idx = fault_info["fault_idx"]
                    expected_value = fault_info["corrupted_value"]

                    # Navigate to the parameter
                    parts = param_name.split(".")
                    obj = self.runner.evaluator.model
                    for part in parts:
                        if part.startswith("Block"):
                            block_num = int(part.replace("Block", ""))
                            obj = obj.blocks[block_num]
                        else:
                            obj = getattr(obj, part)

                    # Handle both Tuple indices (Coordinate) and Integer indices (Flat)
                    if isinstance(fault_idx, (tuple, list)):
                        # If index is (row, col), use it directly on the original tensor
                        actual_value = obj[tuple(fault_idx)].item()
                    else:
                        # If index is a flat integer, use view(-1)
                        actual_value = obj.view(-1)[fault_idx].item()

                    if abs(actual_value - expected_value) > 1e-9:
                        print(f"❌ DEBUG: Fault injection verification FAILED!")
                        print(
                            f"   Expected: {expected_value:.8f}, Got: {actual_value:.8f}"
                        )
                    else:
                        print(
                            f"✓ Fault verified: {param_name}[{fault_idx}] = {actual_value:.8f}"
                        )

                # DEBUG: Check if model output actually changes with this fault
                # Get a sample from the first batch
                batches = self.runner.evaluator.cached_batches(
                    self.config.batch_size,
                    torch.float32,
                    self.config.device,
                    1,  # Just first batch for testing
                )
                if batches:
                    test_images, _ = batches[0]
                    with torch.no_grad():
                        test_output = self.runner.evaluator.model(
                            test_images[:5]
                        )  # First 5 samples
                    print(
                        f"✓ Test forward pass: output shape {test_output.shape}, "
                        f"mean={test_output.mean().item():.4f}, std={test_output.std().item():.4f}"
                    )
                print()

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

    def run_all(self, idx=None, block_idx=None, bit_range=None, component=None):
        """Execute all runs sequentially."""
        analyzer = RunAnalyzer()
        total_start = time.perf_counter()

        # DEBUG: Track unique fault locations
        unique_faults = set()

        for i in range(self.n_runs):
            self.reset_model()
            run_result = self.run_single(
                i,
                idx=idx,
                block_idx=block_idx,
                bit_range=bit_range,
                component=component,
            )
            analyzer.update(run_result)

            # DEBUG: Track fault diversity
            if run_result.get("fault_info"):
                fault_key = (
                    run_result["fault_info"]["param_name"],
                    run_result["fault_info"]["fault_idx"],
                    run_result["fault_info"]["bit_flipped"],
                )
                unique_faults.add(fault_key)

        total_end = time.perf_counter()
        total_runtime = round(total_end - total_start, 2)

        print(f"\nAll {self.n_runs} {self.mode} runs completed.")
        print(f"Total repeated runtime: {total_runtime:.2f} seconds")
        print(
            f"DEBUG: {len(unique_faults)} unique fault locations (out of {self.n_runs} runs)"
        )
        if len(unique_faults) < self.n_runs:
            print(f" WARNING: Some faults were duplicated!")
        print()

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
    parser.add_argument("--debug", type=str, default="false", help="Enable debug mode")
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
        "--batch_size", type=int, default=None, help="Override batch size from config"
    )
    parser.add_argument(
        "--max_batches", type=int, default=None, help="Override max batches from config"
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
    debug = args.debug.lower() in ("true", "1", "yes", "y")
    save_logits = args.save_logits.lower() in ("true", "1", "yes", "y")

    # Enable verbose if debug is on
    if debug:
        verbose = True

    config = Config()

    # 🆕 OVERRIDE CONFIG IF ARGUMENTS ARE PROVIDED
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
                component_type=args.component.lower(),
                idx=args.idx,
                block_idx=args.block_idx,
                bit_range=bit_range,
                verbose=True,
            )
            print("Fault injection applied.\n")

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
            component=args.component.lower(),
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
            "component": args.component.lower(),
        }
        existing_results.append(new_summary)

        with open(summary_path, "w") as f:
            json.dump(existing_results, f, indent=2)
        print(
            f"Summary appended to: {summary_path} (total experiments: {len(existing_results)})"
        )


if __name__ == "__main__":
    main()
