"""CLI for ViT fault injection framework.

Supports chained commands:
    python -m cli -m vit_tiny fi --layers qkv --faults 10 hr --detect attn
    python -m cli -m vit_tiny fi --layers qkv --ber 1e-6 eval
    python -m cli -m vit_tiny hr --detect fc1

Output format (when --output is given with -r > 1):
    A JSON list with one aggregated entry per run-configuration, compatible
    with results/plot.py. Each entry has avg_*/std_* metric fields plus
    config and base_accuracy nested objects.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

from core.bits import set_seed
from core.config import SUPPORTED_MODELS


class ChainedArgumentParser:
    """Parser that supports chained subcommands like: fi --faults 10 hr --detect attn"""

    def __init__(self):
        self.global_parser = self._create_global_parser()
        self.subparsers = {
            "fi": self._create_fi_parser(),
            "hr": self._create_hr_parser(),
            "pa": self._create_pa_parser(),
            "save": self._create_save_parser(),
            "eval": self._create_eval_parser(),
        }

    def _create_global_parser(self):
        parser = argparse.ArgumentParser(
            description="ViT Fault Injection Framework",
            add_help=True,
        )
        parser.add_argument(
            "--model",
            "-m",
            type=str,
            required=True,
            choices=list(SUPPORTED_MODELS.keys()),
            help="Model to use",
        )
        parser.add_argument("--batch_size", "-b", type=int, default=100)
        parser.add_argument("--max_batches", type=int, default=1)
        parser.add_argument(
            "--repeat",
            "-r",
            type=int,
            default=1,
            help="Number of experiment repetitions",
        )
        parser.add_argument("--data", type=str, default="val", choices=["train", "val"])
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument(
            "--warmup",
            "-w",
            type=int,
            default=0,
            help="Number of silent inference passes before timing starts",
        )
        parser.add_argument(
            "--info",
            action="store_true",
            help="Verbose output: per-run results, fault lists, layer info",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Output JSON file for results",
        )
        return parser

    def _create_fi_parser(self):
        """Fault Injection command."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--layers",
            type=str,
            default="all",
            choices=["all", "fc1", "fc2", "qkv", "proj"],
            help="Layers to inject faults into",
        )
        parser.add_argument(
            "--faults", type=int, default=None, help="Number of faults to inject"
        )
        parser.add_argument(
            "--ber",
            type=float,
            default=None,
            help="Bit error rate (alternative to --faults)",
        )
        parser.add_argument(
            "--bit_range",
            type=str,
            default=None,
            help="Bit range for flips, e.g., '20,31'",
        )
        parser.add_argument(
            "--component",
            type=str,
            default=None,
            choices=["qkv", "proj", "fc1", "fc2"],
            help="Sub-component label for output metadata (e.g. qkv, fc1)",
        )
        parser.add_argument(
            "--block",
            type=int,
            default=None,
            help="Transformer block index for output metadata (0-based)",
        )
        return parser

    def _create_hr_parser(self):
        """Hardware Resilience command."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--detect",
            type=str,
            default=None,
            choices=["all", "fc1", "fc2", "qkv", "proj", "attn"],
            help="Layers to apply detection",
        )
        parser.add_argument(
            "--method",
            type=str,
            default="checkone",
            choices=["checkone", "checksum"],
            help="Detection method",
        )
        parser.add_argument(
            "--correction",
            type=str,
            default=None,
            choices=["zero", "rerun", "subtract", "correct"],
            help="Correction mode: 'zero' zeroes faulty outputs, 'rerun' recomputes from clean baseline, 'subtract' subtracts the weight-sum diff, 'correct' finds exact fault position and subtracts per-token error",
        )
        return parser

    def _create_pa_parser(self):
        """Parameter Analysis command."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--type",
            type=str,
            default="activations",
            choices=["activations", "weights", "both"],
        )
        parser.add_argument("--output", "-o", type=str, default=None)
        return parser

    def _create_save_parser(self):
        """Save baseline data command."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--logits", action="store_true", help="Save fault-free logits"
        )
        parser.add_argument(
            "--inputs",
            action="store_true",
            help="Run input calibration (needed for checkone input fault detection)",
        )
        parser.add_argument(
            "--weights",
            action="store_true",
            help="Include full weights (needed for col-check localisation and rerun correction)",
        )
        parser.add_argument(
            "--layers",
            type=str,
            default="all",
            choices=["all", "fc1", "fc2", "qkv", "proj", "attn"],
        )
        return parser

    def _create_eval_parser(self):
        """Basic evaluation command."""
        parser = argparse.ArgumentParser(add_help=False)
        return parser

    def parse(self, argv=None):
        """Parse arguments, returning (global_args, commands_list)."""
        if argv is None:
            argv = sys.argv[1:]

        subcommand_names = set(self.subparsers.keys())
        first_cmd_idx = None
        for i, arg in enumerate(argv):
            if arg in subcommand_names:
                first_cmd_idx = i
                break

        if first_cmd_idx is not None:
            global_argv = argv[:first_cmd_idx]
            rest = argv[first_cmd_idx:]
        else:
            global_argv = argv
            rest = []

        global_args = self.global_parser.parse_args(global_argv)

        commands = []
        while rest:
            cmd_name = rest[0]
            rest = rest[1:]

            if cmd_name not in self.subparsers:
                raise ValueError(f"Unknown command: {cmd_name}")

            # Find where next command starts
            next_cmd_idx = None
            for i, arg in enumerate(rest):
                if arg in subcommand_names:
                    next_cmd_idx = i
                    break

            if next_cmd_idx is not None:
                cmd_argv = rest[:next_cmd_idx]
                rest = rest[next_cmd_idx:]
            else:
                cmd_argv = rest
                rest = []

            cmd_args = self.subparsers[cmd_name].parse_args(cmd_argv)
            commands.append((cmd_name, cmd_args))

        return global_args, commands


def parse_bit_range(s: str) -> tuple[int, int] | None:
    if not s:
        return None
    parts = s.split(",")
    return (int(parts[0]), int(parts[1]))


def run_commands(global_args, commands):
    """Execute the command chain."""
    from core.model import Model, ModelConfig

    # Setup model
    config = ModelConfig(
        batch_size=global_args.batch_size,
        max_batches=global_args.max_batches,
        use_train=(global_args.data == "train"),
    )
    model = Model(global_args.model, config=config)

    if global_args.warmup > 0:
        from eval.metrics import evaluate
        print(f"Warming up ({global_args.warmup} pass{'es' if global_args.warmup > 1 else ''})...")
        for _ in range(global_args.warmup):
            evaluate(model)
        print("Warmup done.")

    injector = None
    fi_args = None
    action_cmd = None
    action_args = None

    for cmd_name, cmd_args in commands:
        if cmd_name == "fi":
            fi_args = cmd_args
            injector = setup_injection(model, cmd_args, verbose=global_args.info)
        elif cmd_name == "hr":
            action_cmd = "hr"
            action_args = cmd_args
        elif cmd_name == "eval":
            if action_cmd != "hr":  # hr already implies evaluation ��� don't overwrite it
                action_cmd = "eval"
                action_args = cmd_args
        elif cmd_name == "pa":
            run_pa(model, cmd_args, global_args.model)
            return
        elif cmd_name == "save":
            run_save(model, cmd_args)
            return

    if action_cmd or injector:
        run_experiment(model, global_args, injector, action_cmd, action_args, fi_args)
    elif injector and not action_cmd:
        # Only fi specified - do a test injection
        if injector.fi_ber is not None:
            injector.inject(ber=injector.fi_ber)
        else:
            injector.inject(count=injector.fi_faults or 1)
        injector.print_info()
        injector.restore()


def setup_injection(model, args, verbose: bool = False):
    """Setup fault injector from fi command args."""
    from injection import Injector

    bit_range = parse_bit_range(args.bit_range)
    injector = Injector(model, layers=args.layers, bit_range=bit_range)

    # Store injection params for later use by hr/eval
    injector.fi_faults = args.faults
    injector.fi_ber = args.ber

    if verbose:
        injector.print_layer_info()
    return injector


def run_experiment(model, global_args, injector, action_cmd, action_args, fi_args=None):
    """Run experiment with repeats and aggregate results."""
    from detection import CheckOne
    from eval.metrics import evaluate

    repeat = global_args.repeat
    all_results = []

    # Setup detector if hr mode
    detector = None
    if action_cmd == "hr" and action_args and action_args.detect:
        from detection import Checksum

        correction = getattr(action_args, "correction", None)
        method = getattr(action_args, "method", "checkone")
        detector_cls = Checksum if method == "checksum" else CheckOne
        detector = detector_cls(model, layers=action_args.detect, correction=correction)
        # Always try to load baseline (needed for input detection and correction)
        if not detector.load():
            print(
                "Warning: No baseline found. Run 'save --detect' first for input fault detection."
            )

    verbose = global_args.info

    # Per-layer detection tracking: layer -> {injected, detected, false_positive}
    layer_stats: dict[str, dict[str, int]] = {}

    def _record_layer(layer: str, injected: bool, detected: bool):
        if layer not in layer_stats:
            layer_stats[layer] = {"injected": 0, "detected": 0, "false_positive": 0}
        if injected:
            layer_stats[layer]["injected"] += 1
            if detected:
                layer_stats[layer]["detected"] += 1
        elif detected:
            layer_stats[layer]["false_positive"] += 1

    for run in range(repeat):
        if verbose and repeat > 1:
            print(f"\n{'=' * 60}")
            print(f"RUN {run + 1}/{repeat}")
            print(f"{'=' * 60}")

        # Inject faults
        if injector:
            if injector.fi_ber is not None:
                injector.inject(ber=injector.fi_ber)
            else:
                injector.inject(count=injector.fi_faults or 1)
            if verbose:
                injector.print_info()

        # Evaluate
        results = evaluate(model, detector)

        # Print per-run results: always for single run, only with --info for multiple
        if verbose or repeat == 1:
            results.print()
            if detector:
                detector.print_results()

        # Accumulate per-layer detection stats
        if injector or detector:
            injected_layers = {f["layer"] for f in injector.get_info()} if injector else set()
            detected_layers = {f.layer for f in detector.get_faults()} if detector else set()
            all_layers = injected_layers | detected_layers
            for layer in all_layers:
                _record_layer(layer, layer in injected_layers, layer in detected_layers)

        # Collect results for aggregation and output
        run_data = results.to_dict()
        run_data["run"] = run + 1
        run_data["seed"] = global_args.seed
        if detector:
            run_data["detection"] = detector.get_values()
        if injector:
            run_data["faults"] = injector.get_info()
        all_results.append(run_data)

        # Restore for next run
        if injector:
            injector.restore()

    # Print aggregate stats if multiple runs
    if repeat > 1 and all_results:
        print(f"\n{'=' * 60}")
        print(f"AGGREGATE RESULTS ({repeat} runs)")
        print(f"{'=' * 60}")

        # Accuracy stats
        top1s = [r["top1_acc"] for r in all_results]
        top5s = [r["top5_acc"] for r in all_results]
        print(
            f"Top-1: {sum(top1s) / len(top1s):.2f}% (min={min(top1s):.2f}, max={max(top1s):.2f})"
        )
        print(
            f"Top-5: {sum(top5s) / len(top5s):.2f}% (min={min(top5s):.2f}, max={max(top5s):.2f})"
        )

        def _stats(key):
            vals = [r[key] for r in all_results if r.get(key) is not None]
            if not vals:
                return 0.0, 0.0, 0.0
            return sum(vals) / len(vals), min(vals), max(vals)

        # SDC stats if available
        if all_results[0].get("sdc_rate") is not None:
            avg_sdc, min_sdc, max_sdc = _stats("sdc_rate")
            avg_msdc, min_msdc, max_msdc = _stats("msdc")
            print()
            print(f"SDC Rate: {avg_sdc:.2f}%  (min={min_sdc:.2f}, max={max_sdc:.2f})")
            print(f"MSDC:     {avg_msdc:.6f}  (min={min_msdc:.6f}, max={max_msdc:.6f})")
            print()
            print(f"Threshold SDC:")
            for pct in [1, 5, 10, 15, 20, 25, 50]:
                avg_t, min_t, max_t = _stats(f"sdc_{pct}pct")
                print(f"  ≥{pct:2d}%: {avg_t:.2f}%  (min={min_t:.2f}, max={max_t:.2f})")
            print()
            avg_c1, min_c1, max_c1 = _stats("critical_top1")
            avg_c5, min_c5, max_c5 = _stats("critical_top5")
            print(f"Critical Top-1: {avg_c1:.2f}%  (min={min_c1:.2f}, max={max_c1:.2f})")
            print(f"Critical Top-5: {avg_c5:.2f}%  (min={min_c5:.2f}, max={max_c5:.2f})")

        # Timing
        avg_mps, min_mps, max_mps = _stats("ms_per_sample")
        total_s = sum(r.get("elapsed_s", 0.0) for r in all_results)
        print()
        print(f"Time: {avg_mps:.3f} ms/sample avg  (min={min_mps:.3f}, max={max_mps:.3f})  |  {total_s:.1f} s total")

        # Detection summary
        if layer_stats:
            print()
            print(f"Detection Summary:")
            col = max(len(l) for l in layer_stats) + 2
            print(f"  {'Layer':<{col}}  {'Injected':>8}  {'Detected':>8}  {'Rate':>7}  {'False+':>6}")
            print(f"  {'-' * col}  {'--------':>8}  {'--------':>8}  {'-------':>7}  {'------':>6}")
            for layer in sorted(layer_stats):
                s = layer_stats[layer]
                inj = s["injected"]
                det = s["detected"]
                fp  = s["false_positive"]
                rate = f"{100 * det / inj:.1f}%" if inj else "  n/a"
                print(f"  {layer:<{col}}  {inj:>8}  {det:>8}  {rate:>7}  {fp:>6}")

    # Save results
    if global_args.output and all_results:
        Path(global_args.output).parent.mkdir(parents=True, exist_ok=True)
        summary = _build_summary(all_results, global_args, fi_args)
        # Append to existing file if it already contains a list
        output_path = Path(global_args.output)
        existing = []
        if output_path.exists():
            try:
                with open(output_path) as f:
                    loaded = json.load(f)
                    existing = loaded if isinstance(loaded, list) else [loaded]
            except (json.JSONDecodeError, OSError):
                pass
        existing.append(summary)
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\nSaved results to {global_args.output}")


def _build_summary(all_results: list[dict], global_args, fi_args) -> dict:
    """Build a plot-compatible aggregated summary from per-run result dicts."""
    n = len(all_results)

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals, m):
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0

    top1s = [r["top1_acc"] for r in all_results]
    top5s = [r["top5_acc"] for r in all_results]
    avg_top1 = _mean(top1s)
    avg_top5 = _mean(top5s)

    mps_vals = [r["ms_per_sample"] for r in all_results if r.get("ms_per_sample") is not None]
    avg_mps = _mean(mps_vals)

    summary: dict = {
        "total_runs": n,
        "avg_top1_acc": avg_top1,
        "std_top1_acc": _std(top1s, avg_top1),
        "avg_top5_acc": avg_top5,
        "std_top5_acc": _std(top5s, avg_top5),
        "avg_ms_per_sample": avg_mps,
        "std_ms_per_sample": _std(mps_vals, avg_mps),
    }

    # SDC metrics (present only when fault-free logits were available)
    if all_results[0].get("sdc_rate") is not None:
        sdc_rates = [r["sdc_rate"] for r in all_results]
        msdcs = [r["msdc"] for r in all_results if r.get("msdc") is not None]
        crit1s = [r["critical_top1"] for r in all_results if r.get("critical_top1") is not None]
        crit5s = [r["critical_top5"] for r in all_results if r.get("critical_top5") is not None]

        avg_logit_sdc = _mean(sdc_rates)
        avg_msdc = _mean(msdcs)
        avg_crit1 = _mean(crit1s)
        avg_crit5 = _mean(crit5s)

        summary["avg_logit_sdc"] = avg_logit_sdc
        summary["std_logit_sdc"] = _std(sdc_rates, avg_logit_sdc)
        summary["avg_msdc"] = avg_msdc
        summary["avg_critical_top1_sdc"] = avg_crit1
        summary["std_critical_top1_sdc"] = _std(crit1s, avg_crit1)
        summary["avg_critical_top5_sdc"] = avg_crit5
        summary["std_critical_top5_sdc"] = _std(crit5s, avg_crit5)

        for pct in [1, 5, 10, 15, 20, 25, 50]:
            key = f"sdc_{pct}pct"
            vals = [r[key] for r in all_results if r.get(key) is not None]
            summary[f"avg_{key}"] = _mean(vals)

        # Risk categories (mirrors SDCTracker logic)
        high_risk = sum(1 for r in all_results if (r.get("critical_top1") or 0.0) > 0.0)
        medium_risk = sum(
            1 for r in all_results
            if (r.get("critical_top1") or 0.0) == 0.0 and (r.get("critical_top5") or 0.0) > 0.0
        )
        safe = n - high_risk - medium_risk
        summary["high_risk_count"] = high_risk
        summary["high_risk_pct"] = 100.0 * high_risk / n if n else 0.0
        summary["medium_risk_count"] = medium_risk
        summary["medium_risk_pct"] = 100.0 * medium_risk / n if n else 0.0
        summary["safe_count"] = safe
        summary["safe_pct"] = 100.0 * safe / n if n else 0.0

    # Config — sub_component and block come from fi --component / --block
    sub_component = getattr(fi_args, "component", None) if fi_args else None
    block_idx = getattr(fi_args, "block", None) if fi_args else None
    component = (
        "attention" if sub_component in ("qkv", "proj")
        else "mlp" if sub_component in ("fc1", "fc2")
        else None
    )
    summary["config"] = {
        "model": global_args.model,
        "sub_component": sub_component,
        "block_idx": block_idx,
        "component": component,
    }

    return summary


def run_pa(model, args, model_name):
    """Parameter Analysis mode."""
    from analysis import ActivationAnalyzer, WeightAnalyzer

    output_base = args.output or f"results/{model_name}"

    if args.type in ["activations", "both"]:
        print("\n=== Activation Analysis ===")
        analyzer = ActivationAnalyzer(model)
        analyzer.run()
        analyzer.save(f"{output_base}_activations.json")
        analyzer.remove()

    if args.type in ["weights", "both"]:
        print("\n=== Weight Analysis ===")
        analyzer = WeightAnalyzer(model)
        analyzer.run()
        analyzer.save(f"{output_base}_weights.json")


def run_save(model, args):
    """Save baseline data.

    Produces a single shared baseline file usable by both checkone and checksum.
    The file contains weight checksums (always), calibrated input ranges (--inputs),
    and full weights (--weights).
    """
    from detection import CheckOne

    saved = False

    if args.logits:
        print("\nSaving Fault-Free Logits")
        model.save_baseline()
        saved = True

    if args.inputs or args.weights:
        print("\nSaving Detection Baseline")
        # CheckOne wrapper computes both weight_sums and w_col_sums (for checksum)
        # and handles input calibration — one pass covers both methods.
        detector = CheckOne(model, layers=args.layers)
        if args.inputs:
            detector.calibrate(model)
        detector.save(include_weights=args.weights)
        detector.remove()
        saved = True

    if not saved:
        print(
            "Use --logits to save fault-free logits.\n"
            "Use --inputs to calibrate input ranges (checkone input detection).\n"
            "Use --weights to store full weights (col-check and rerun correction).\n"
            "Example: save --inputs --weights"
        )


def main():
    parser = ChainedArgumentParser()

    try:
        global_args, commands = parser.parse()
    except SystemExit:
        return

    seed = (
        global_args.seed
        if global_args.seed is not None
        else random.randint(0, 2**32 - 1)
    )
    global_args.seed = seed
    set_seed(seed)
    if global_args.info:
        print(f"Seed: {seed}")

    if not commands:
        from core.model import Model, ModelConfig

        config = ModelConfig(
            batch_size=global_args.batch_size,
            max_batches=global_args.max_batches,
            use_train=(global_args.data == "train"),
        )
        Model(global_args.model, config=config)
        print("Model loaded. Add 'eval' to evaluate, or 'fi ... eval' to inject and evaluate.")
    else:
        run_commands(global_args, commands)

    print("\nDone.")


if __name__ == "__main__":
    main()
