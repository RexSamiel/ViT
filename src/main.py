import argparse
import datetime
import json
import os
import random
import time
import torch

from src.config.settings import Config
from src.core.model import ModelEvaluator
from src.metrics import AccuracyMetrics, SDCMetrics, ActivationAnalyzer
from src.fault_injector.fault_injection import inject_fault, get_num_blocks
from src.utils.helper import SUPPORTED_MODELS, print_supported_models


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_labels(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Replace labels for NaN outputs with -1 so they don't count as correct."""
    nan_mask = torch.isnan(outputs).all(dim=1)
    if nan_mask.any():
        labels = labels.clone()
        labels[nan_mask] = -1
    return labels


def align_batch_sizes(faulty: torch.Tensor, faultfree: torch.Tensor) -> tuple:
    """Align batch sizes between faulty and fault-free tensors."""
    min_size = min(faulty.size(0), faultfree.size(0))
    return faulty[:min_size], faultfree[:min_size]


class RunManager:
    """Manages single and multi-run fault injection experiments."""

    def __init__(self, config: Config, verbose: bool = True, show_info: bool = True):
        self.config = config
        self.verbose = verbose
        self.show_info = show_info  # Controls per-run output in multi-run mode
        self.evaluator = ModelEvaluator(config, verbose)

        # Track last fault for efficient reset (only restore corrupted parameter)
        self.last_fault_info = None

        # Metrics
        self.accuracy = AccuracyMetrics()
        self.sdc = SDCMetrics()

        # AMP config
        unstable_fp16 = ["beit"]
        self.use_amp = (
            config.use_amp
            and config.device.type == "cuda"
            and not any(name in config.model_name.lower() for name in unstable_fp16)
        )

    def reset(self) -> None:
        """Reset model and metrics for a new run."""
        # Only restore the corrupted parameter (much faster than full state_dict)
        if self.last_fault_info is not None:
            info = self.last_fault_info
            param = info["param_ref"]
            idx = info["fault_idx"]
            original = info["original_tensor"]
            with torch.no_grad():
                param[idx] = original
            self.last_fault_info = None

        # Note: Do NOT clear batch cache here - batches should persist between runs
        self.accuracy.reset()
        self.sdc.reset()

    def run_single(
        self,
        mode: str = "faultfree",
        save_logits: bool = False,
        fault_params: dict = None,
        show_results: bool = None,
    ) -> dict:
        """Execute a single evaluation run."""
        self.accuracy.reset()
        self.sdc.reset()

        # Determine whether to show results (default to verbose if not specified)
        if show_results is None:
            show_results = self.verbose

        # Inject fault if in faulty mode
        fault_info = None
        if mode == "faulty" and fault_params:
            fault_info = inject_fault(
                self.evaluator.model,
                component_type=fault_params.get("component", "all"),
                sub_component=fault_params.get("sub_component"),
                idx=fault_params.get("idx"),
                block_idx=fault_params.get("block_idx"),
                bit_range=fault_params.get("bit_range"),
                verbose=show_results,
            )
            # Store for efficient reset (only restore this parameter)
            self.last_fault_info = fault_info

        # Run evaluation
        batches = self.evaluator.get_batches()
        logits_buffer, labels_buffer = [], []
        compute_sdc = mode == "faulty" and self.evaluator.ff_logits.available

        start_time = time.perf_counter()

        with torch.inference_mode():
            for batch_idx, (images, labels) in enumerate(batches):
                outputs = self.evaluator.inference(images, self.use_amp)

                # Prepare labels (handle NaN)
                labels_clean = prepare_labels(outputs, labels)

                # Update accuracy
                self.accuracy.update(outputs, labels_clean)

                # Update SDC if applicable
                if compute_sdc:
                    ff_batch = self.evaluator.ff_logits.get_batch(
                        batch_idx,
                        self.config.batch_size,
                        outputs.size(0),
                        self.config.device,
                    )
                    faulty, faultfree = align_batch_sizes(outputs, ff_batch)
                    self.sdc.update(faulty, faultfree)

                # Save logits if requested
                if save_logits:
                    logits_buffer.append(outputs.cpu())
                    labels_buffer.append(labels.cpu())

        runtime = time.perf_counter() - start_time

        # Save logits
        if save_logits and logits_buffer:
            self.evaluator.ff_logits.save(logits_buffer, labels_buffer)

        # Compile results
        results = {
            **self.accuracy.get_results(),
            **(self.sdc.get_results() if compute_sdc else {}),
            "fault_info": fault_info,
            "runtime_sec": round(runtime, 2),
            "mode": mode,
        }

        if show_results:
            self._print_run_results(results, runtime)

        return results

    def run_multiple(
        self,
        n_runs: int,
        fault_params: dict = None,
    ) -> dict:
        """Execute multiple fault injection runs and aggregate results."""
        total_start = time.perf_counter()

        for i in range(n_runs):
            if self.verbose and self.show_info:
                print(f"\n{'=' * 60}")
                print(f" Run {i + 1}/{n_runs}")
                print(f"{'=' * 60}")

            self.reset()
            results = self.run_single(
                mode="faulty",
                fault_params=fault_params,
                show_results=self.show_info,
            )

            # Aggregate results
            self.accuracy.aggregate(results)
            self.sdc.aggregate(results)

        total_runtime = time.perf_counter() - total_start

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f" SUMMARY: {n_runs} runs completed in {total_runtime:.2f}s")
            print(f"{'=' * 60}\n")
            self.accuracy.print_summary()
            print()
            self.sdc.print_summary()
            print("=" * 60)

        return {
            **self.accuracy.get_summary(),
            **self.sdc.get_summary(),
            "total_runtime": round(total_runtime, 2),
        }

    def _print_run_results(self, results: dict, runtime: float) -> None:
        """Print results for a single run."""
        print(f"\nResults for {self.config.model_name}:")
        print("-" * 40)
        self.accuracy.print_results()

        if "logit_sdc_rate" in results:
            print()
            self.sdc.print_results()

        print(f"\nRuntime: {runtime:.2f}s")


class ActivationRunManager:
    """Manages activation analysis runs for transformer models."""

    def __init__(self, config: Config, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.evaluator = ModelEvaluator(config, verbose)

        # Activation analyzer
        self.analyzer = ActivationAnalyzer()

        # AMP config
        unstable_fp16 = ["beit"]
        self.use_amp = (
            config.use_amp
            and config.device.type == "cuda"
            and not any(name in config.model_name.lower() for name in unstable_fp16)
        )

    def run(self) -> dict:
        """Execute activation analysis on the model."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f" Activation Analysis: {self.config.model_name}")
            print(f"{'=' * 60}")

        # Register hooks
        hook_count = self.analyzer.register_hooks(self.evaluator.model)
        if self.verbose:
            print(f"Registered {hook_count} activation hooks")

        # Get batches
        batches = self.evaluator.get_batches()
        total_batches = len(batches)

        if self.verbose:
            print(f"Processing {total_batches} batches ({self.config.batch_size} samples each)")
            print("-" * 40)

        start_time = time.perf_counter()

        with torch.inference_mode():
            for batch_idx, (images, _) in enumerate(batches):
                # Images already on device from cached_batches
                # Run inference - hooks capture activations automatically
                _ = self.evaluator.inference(images, self.use_amp)

                # Update statistics
                self.analyzer.update(images.size(0))

                if self.verbose and (batch_idx + 1) % max(1, total_batches // 10) == 0:
                    progress = 100 * (batch_idx + 1) / total_batches
                    print(f"  Progress: {progress:.0f}% ({batch_idx + 1}/{total_batches} batches)")

        runtime = time.perf_counter() - start_time

        # Remove hooks
        self.analyzer.remove_hooks()

        # Get results
        results = self.analyzer.get_results()
        results["runtime_sec"] = round(runtime, 2)
        results["model"] = self.config.model_name

        if self.verbose:
            print("-" * 40)
            self.analyzer.print_results()
            print(f"\nRuntime: {runtime:.2f}s")
            print("=" * 60)

        return results

    def print_layer_details(self) -> None:
        """Print detailed per-layer activation ranges."""
        self.analyzer.print_layer_ranges()


# ==================== CLI ====================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision Transformer Fault Injection Framework"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["faultfree", "faulty"], default="faultfree")
    parser.add_argument("--metrics", type=str, default="fi",
                        choices=["fi", "fault_injection", "aa", "activation_analyzer"],
                        help="Metrics mode: fi/fault_injection for fault injection, aa/activation_analyzer for activation analysis")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", type=str, default="true")
    parser.add_argument("--info", type=str, default="false",
                        help="Show per-run info in multi-run mode (default: false)")
    parser.add_argument("--save_logits", type=str, default="false")
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--block_idx", type=int, default=None)
    parser.add_argument("--bit_range", type=str, default=None)
    parser.add_argument("--component", type=str, default="all",
                        choices=["mlp", "norm", "attention", "patch_embed", "classifier", "all"])
    parser.add_argument("--sub_component", type=str, default=None)
    parser.add_argument("--batch_size", type=str, default=None,
                        help="Batch size (int or 'None' to use full batches)")
    parser.add_argument("--max_batches", type=str, default=None,
                        help="Max batches (int or 'None' for all batches)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random)")
    return parser.parse_args()


def parse_bit_range(s: str) -> tuple[int, int] | None:
    """Parse bit range string 'START,END' to tuple."""
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError("bit_range must be START,END")
    return (int(parts[0]), int(parts[1]))


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ("true", "1", "yes", "y")


def int_or_none(s: str) -> int | None:
    """Convert string to int or None."""
    if s.lower() == "none":
        return None
    return int(s)


def get_run_filename(args, config: Config) -> str:
    """Generate filename with date and sample count."""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    samples = config.batch_size * (config.max_batches or 500)  # default 500 if None
    return f"{args.model}_{args.mode}_{samples}samples_{date_str}.json"


def load_base_accuracy(model_key: str) -> dict | None:
    """Load base accuracy from faultfree run if available."""
    base_path = f"results/base_accuracy/{model_key}.json"
    if os.path.exists(base_path):
        try:
            with open(base_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_base_accuracy(model_key: str, results: dict) -> None:
    """Save base accuracy from faultfree run."""
    os.makedirs("results/base_accuracy", exist_ok=True)
    path = f"results/base_accuracy/{model_key}.json"

    base_acc = {
        "top1": results.get("top1_acc", 0.0),
        "top5": results.get("top5_acc", 0.0),
        "samples": results.get("samples", 0),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open(path, "w") as f:
        json.dump(base_acc, f, indent=2)

    print(f"Base accuracy saved to: {path}")


def save_summary(
    args,
    summary: dict,
    config: Config,
    base_accuracy: dict = None,
    seed: int = None,
    total_blocks: int = None,
) -> None:
    """Save experiment summary to JSON."""
    # Save to new_runs folder with descriptive filename
    os.makedirs("results/new_runs", exist_ok=True)
    filename = get_run_filename(args, config)
    path = f"results/new_runs/{filename}"

    # Load existing if file exists (append mode)
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
        except (json.JSONDecodeError, IOError):
            existing = []

    # Calculate samples per run
    samples_per_run = config.batch_size * (config.max_batches or 500)

    # Add new summary
    summary["timestamp"] = datetime.datetime.now().isoformat()
    summary["config"] = {
        "model": args.model,
        "model_name": config.model_name,
        "mode": args.mode,
        "repeat": args.repeat,
        "samples_per_run": samples_per_run,
        "batch_size": config.batch_size,
        "max_batches": config.max_batches,
        "component": args.component,
        "sub_component": args.sub_component,
        "block_idx": args.block_idx,
        "idx": args.idx,
        "bit_range": parse_bit_range(args.bit_range),
        "total_blocks": total_blocks,
    }

    # Reproducibility info
    summary["reproducibility"] = {
        "seed": seed,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(config.device),
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    # Include base accuracy if available
    if base_accuracy:
        summary["base_accuracy"] = base_accuracy

    existing.append(summary)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Summary saved to: {path}")


def save_activation_results(args, results: dict, config: Config) -> None:
    """Save activation analysis results to JSON."""
    # Save to new_runs folder with descriptive filename
    os.makedirs("results/new_runs", exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    samples = config.batch_size * (config.max_batches or 500)
    filename = f"activations_{args.model}_{samples}samples_{date_str}.json"
    path = f"results/new_runs/{filename}"

    # Add metadata
    results["timestamp"] = datetime.datetime.now().isoformat()
    results["config"] = {
        "model": args.model,
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "max_batches": config.max_batches,
        "samples": samples,
    }

    # Convert layer indices to strings for JSON compatibility
    ranges = results.get("ranges", {})
    for component in ["block", "mha", "mlp"]:
        if component in ranges and "layers" in ranges[component]:
            layers = ranges[component]["layers"]
            ranges[component]["layers"] = {str(k): v for k, v in layers.items()}

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nActivation results saved to: {path}")


def run_fault_injection(args, config: Config, verbose: bool, show_info: bool, seed: int) -> None:
    """Run fault injection mode."""
    save_logits = str_to_bool(args.save_logits)

    # Fault parameters
    fault_params = {
        "component": args.component,
        "sub_component": args.sub_component,
        "idx": args.idx,
        "block_idx": args.block_idx,
        "bit_range": parse_bit_range(args.bit_range),
    }

    # Create manager and run
    manager = RunManager(config, verbose=verbose, show_info=show_info)

    # Get total blocks for metadata
    total_blocks = get_num_blocks(manager.evaluator.model)

    # Load base accuracy for faulty runs
    base_accuracy = None
    if args.mode == "faulty":
        base_accuracy = load_base_accuracy(args.model)
        if base_accuracy and verbose:
            print(f"Loaded base accuracy: Top-1={base_accuracy['top1']:.2f}%, Top-5={base_accuracy['top5']:.2f}%")

    if args.repeat <= 1:
        results = manager.run_single(
            mode=args.mode,
            save_logits=save_logits,
            fault_params=fault_params if args.mode == "faulty" else None,
        )
        # Save base accuracy for faultfree single runs
        if args.mode == "faultfree":
            save_base_accuracy(args.model, results)
    else:
        summary = manager.run_multiple(args.repeat, fault_params)
        save_summary(args, summary, config, base_accuracy, seed=seed, total_blocks=total_blocks)


def run_activation_analysis(args, config: Config, verbose: bool) -> None:
    """Run activation analysis mode."""
    manager = ActivationRunManager(config, verbose=verbose)

    # Run analysis
    results = manager.run()

    # Print detailed layer info if verbose
    if verbose:
        manager.print_layer_details()

    # Save results
    save_activation_results(args, results, config)


def main():
    args = parse_args()

    # Validate model
    if args.model not in SUPPORTED_MODELS:
        print(f"Error: '{args.model}' not supported.")
        print_supported_models()
        return

    # Setup config
    config = Config()
    config.model_key = args.model
    config.model_name = SUPPORTED_MODELS[args.model]
    if args.batch_size is not None:
        config.batch_size = int_or_none(args.batch_size)
    if args.max_batches is not None:
        config.max_batches = int_or_none(args.max_batches)

    verbose = str_to_bool(args.verbose)
    show_info = str_to_bool(args.info)

    # Set random seed for reproducibility
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    set_seed(seed)
    if verbose:
        print(f"Random seed: {seed}")

    # Route based on metrics mode
    if args.metrics in ("aa", "activation_analyzer"):
        run_activation_analysis(args, config, verbose)
    else:
        run_fault_injection(args, config, verbose, show_info, seed)


if __name__ == "__main__":
    main()
