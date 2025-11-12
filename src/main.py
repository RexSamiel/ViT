import argparse
import copy
import torch
from src.core.runner import Runner
from src.config.settings import Config
from src.utils.helper import SUPPORTED_MODELS, print_supported_models
from src.fault_injector.fault_injection import inject_fault
from src.core.analyzer import RunAnalyzer
import json
import os


class ExperimentManager:
    """Manages multiple runs efficiently without saving to file."""

    def __init__(self, config: Config, n_runs: int, mode: str, verbose: bool):
        self.config = config
        self.n_runs = n_runs
        self.mode = mode.lower()
        self.verbose = verbose
        self.runner = Runner(config)
        self.original_state = copy.deepcopy(self.runner.evaluator.model.state_dict())

    def reset_model(self):
        self.runner.evaluator.model.load_state_dict(self.original_state)
        torch.cuda.empty_cache()

    def run_single(
        self,
        run_id: int,
        idx: int | None = None,
        block_idx: int | None = None,
        bit_range: tuple[int, int] | None = None,
    ) -> dict:
        if self.verbose:
            print(f"\n⚡ Running {self.mode} run #{run_id + 1}/{self.n_runs}")

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

        results = self.runner.run(compute_metrics=True, save_logits=False)
        results_dict = results if results else {}
        results_dict["run_id"] = run_id + 1
        results_dict["mode"] = self.mode
        results_dict["fault_info"] = fault_info
        return results_dict


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
        "--idx", type=int, default=None, help="Index of parameter to inject fault"
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

    args = parser.parse_args()

    bit_range = None
    if args.bit_range:
        parts = args.bit_range.split(",")
        if len(parts) == 2:
            bit_range = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError("bit_range must be in START,END format (e.g., 0,31)")

    verbose = args.verbose.lower() in ("true", "1", "yes", "y")
    config = Config()

    if args.model not in SUPPORTED_MODELS:
        print(f"Error: '{args.model}' is not a supported model.")
        print_supported_models()
        return
    config.model_key = args.model
    config.model_name = SUPPORTED_MODELS[args.model]

    # SINGLE RUN
    if args.repeat <= 1:
        runner = Runner(config, verbose=True)
        runner.run(compute_metrics=True, save_logits=False)

    # MULTI-RUN
    else:
        manager = ExperimentManager(config, args.repeat, args.mode, verbose=verbose)
        analyzer = RunAnalyzer()

        for i in range(args.repeat):
            manager.reset_model()
            run_result = manager.run_single(
                i,
                idx=args.idx,
                block_idx=args.block_idx,
                bit_range=bit_range,  # use parsed tuple
            )
            analyzer.update(run_result)

        print(f"\nCompleted {args.repeat} runs in {args.mode} mode.")
        analyzer.print_summary()

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        summary_file = os.path.join(
            results_dir,
            f"summary_{config.model_key}_{args.mode}_repeat{args.repeat}.json",
        )

        with open(summary_file, "w") as f:
            json.dump(analyzer.get_summary(), f, indent=2)

        print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
