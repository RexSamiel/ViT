"""CLI argument parsing for the ViT fault injection framework.

Supports chained subcommands:
    python -m cli -m vit_tiny fi --layers qkv --faults 10 hr --detect attn
    python -m cli -m vit_tiny fi --layers qkv --ber 1e-6 eval
    python -m cli -m vit_tiny hr --detect fc1

Orchestration logic lives in main.py.
"""

import argparse
import random
import sys

from core.bits import set_seed
from core.config import SUPPORTED_MODELS


class ChainedArgumentParser:
    """Parser that supports chained subcommands like: fi --faults 10 hr --detect attn"""

    def __init__(self):
        self.global_parser = self._create_global_parser()
        self.subparsers = {
            "fi":   self._create_fi_parser(),
            "hr":   self._create_hr_parser(),
            "pa":   self._create_pa_parser(),
            "save": self._create_save_parser(),
        }

    def _create_global_parser(self):
        parser = argparse.ArgumentParser(
            description="ViT Fault Injection Framework",
            add_help=True,
        )
        parser.add_argument("--model", "-m", type=str, required=True,
                            choices=list(SUPPORTED_MODELS.keys()), help="Model to use")
        parser.add_argument("--batch_size", "-b", type=int, default=100)
        parser.add_argument("--max_batches", type=int, default=1)
        parser.add_argument("--repeat", "-r", type=int, default=1,
                            help="Number of experiment repetitions")
        parser.add_argument("--data", type=str, default="val", choices=["train", "val"])
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--warmup", "-w", type=int, default=0,
                            help="Number of silent inference passes before timing starts")
        parser.add_argument("--info", action="store_true",
                            help="Verbose output: per-run results, fault lists, layer info")
        parser.add_argument("--time", action="store_true",
                            help="Show overall system execution time and total inference time")
        parser.add_argument("--output", "-o", type=str, default=None,
                            help="Output JSON file for results")
        return parser

    def _create_fi_parser(self):
        """Fault Injection command."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--layers", type=str, default="all",
                            choices=["all", "fc1", "fc2", "qkv", "proj"],
                            help="Layers to inject faults into")
        parser.add_argument("--faults", type=int, default=None,
                            help="Number of faults to inject")
        parser.add_argument("--ber", type=float, default=None,
                            help="Bit error rate (alternative to --faults)")
        parser.add_argument("--bit_range", type=str, default=None,
                            help="Bit range for flips, e.g., '20,31' or '0,31^30' to exclude bit 30")
        parser.add_argument("--fault_seed", type=int, default=None,
                            help="Seed for fault injection RNG. Same value across runs injects identical faults.")
        parser.add_argument("--time", action="store_true",
                            help="Show per-run and aggregate inference timing breakdown")
        parser.add_argument("--component", type=str, default=None,
                            choices=["qkv", "proj", "fc1", "fc2"],
                            help="Sub-component label for output metadata")
        parser.add_argument("--block", type=int, default=None,
                            help="Transformer block index for output metadata (0-based)")
        parser.add_argument("--layer_prefix", type=str, default=None,
                            help="Filter layers by exact name prefix, e.g. 'layers.2.blocks.3'")
        return parser

    def _create_hr_parser(self):
        """Hardware Resilience command."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--detect", type=str, default=None,
                            choices=["all", "fc1", "fc2", "qkv", "proj", "attn"],
                            help="Layers to apply detection")
        parser.add_argument("--method", type=str, default="checkone",
                            choices=["checkone", "checksum", "baseline"],
                            help="Detection method")
        parser.add_argument("--correction", type=str, default=None,
                            choices=["zero", "correct"],
                            help="Correction mode")
        parser.add_argument("--time", action="store_true",
                            help="Show per-layer detection timing")
        return parser

    def _create_pa_parser(self):
        """Parameter Analysis command."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--type", type=str, default="activations",
                            choices=["activations", "weights", "both"])
        parser.add_argument("--output", "-o", type=str, default=None)
        return parser

    def _create_save_parser(self):
        """Save baseline data command."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--logits", action="store_true",
                            help="Save fault-free logits")
        parser.add_argument("--inputs", action="store_true",
                            help="Run input calibration (needed for checkone input fault detection)")
        parser.add_argument("--weights", action="store_true",
                            help="Include full weights (needed for col-check localisation and rerun correction)")
        parser.add_argument("--threshold", action="store_true",
                            help="Calibrate detection threshold from clean data and save for both methods")
        parser.add_argument("--layers", type=str, default="all",
                            choices=["all", "fc1", "fc2", "qkv", "proj", "attn"])
        return parser

    def parse(self, argv=None):
        """Parse arguments, returning (global_args, commands_list)."""
        if argv is None:
            argv = sys.argv[1:]

        subcommand_names = set(self.subparsers.keys())

        # Split global args from first subcommand onwards
        first_cmd_idx = next(
            (i for i, arg in enumerate(argv) if arg in subcommand_names), None
        )
        if first_cmd_idx is not None:
            global_argv, rest = argv[:first_cmd_idx], argv[first_cmd_idx:]
        else:
            global_argv, rest = argv, []

        global_args = self.global_parser.parse_args(global_argv)

        commands = []
        while rest:
            cmd_name, rest = rest[0], rest[1:]
            if cmd_name not in self.subparsers:
                raise ValueError(f"Unknown command: {cmd_name}")

            next_cmd_idx = next(
                (i for i, arg in enumerate(rest) if arg in subcommand_names), None
            )
            if next_cmd_idx is not None:
                cmd_argv, rest = rest[:next_cmd_idx], rest[next_cmd_idx:]
            else:
                cmd_argv, rest = rest, []

            commands.append((cmd_name, self.subparsers[cmd_name].parse_args(cmd_argv)))

        return global_args, commands


def parse_bit_range(s: str) -> list[int] | None:
    """Parse a bit range string into an explicit list of bit indices.

    Syntax:
        "lo,hi"          — all bits from lo to hi inclusive
        "lo,hi^b1^b2"    — same range but excluding bits b1, b2, ...

    Examples:
        "23,29"         → [23, 24, 25, 26, 27, 28, 29]
        "0,31^30"       → [0..29, 31]
        "0,31^28^29^30" → [0..27, 31]
    """
    if not s:
        return None
    parts = s.split("^")
    lo, hi = (int(x) for x in parts[0].split(","))
    exclude = {int(x) for x in parts[1:]}
    return [b for b in range(lo, hi + 1) if b not in exclude]


def main():
    import time as _time
    _script_start = _time.perf_counter()

    parser = ChainedArgumentParser()
    try:
        global_args, commands = parser.parse()
    except SystemExit:
        return

    # Seed
    seed = global_args.seed if global_args.seed is not None else random.randint(0, 2**32 - 1)
    global_args.seed = seed
    set_seed(seed)
    if global_args.info:
        print(f"Seed: {seed}")

    # Delegate to orchestrator
    from main import run
    run(global_args, commands)

    # Global timing
    if global_args.time:
        elapsed = _time.perf_counter() - _script_start
        print(f"\n{'=' * 60}")
        print(f"TIMING SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total script execution: {elapsed * 1000:.1f} ms  ({elapsed:.3f} s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
