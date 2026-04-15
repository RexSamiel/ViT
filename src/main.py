"""Main experiment orchestration.

Single unified workflow — every mode (fi, hr, save, pa) activates only the
modules it needs and skips the rest. Read top-to-bottom to understand what
happens in any run.

    SETUP   — model + data, detector.wrap(model), injector.index(layers), accumulators
    WARMUP  — silent GPU passes to stabilise clock
    RUNS    — [inject → batch loop → collect] × repeat
    PRINT   — each active module prints its own summary
    SAVE    — each active module saves its own data

Constraints that must be respected:
  • Detector wraps layers before injector indexes them (injector.refresh_layers() after)
  • Threshold calibration hooks start AFTER warmup so warmup data is excluded
  • In save mode the detector is created fresh in SAVE (not during SETUP/RUNS)
    so that calibrate_threshold() can calibrate both CheckOne and Checksum cleanly

Entry point: run(global_args, commands)
"""

# test
import json
import math
import random
import statistics as _stlib
import time
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def run(global_args, commands):

    fi_args, hr_args, save_args, pa_args = _parse_commands(commands)

    # ── SETUP ─────────────────────────────────────────────────────────────────
    from core.config import ModelConfig
    from core.model import Model

    config = ModelConfig(
        batch_size=global_args.batch_size,
        max_batches=global_args.max_batches,
        use_train=(global_args.data == "train"),
    )
    model = Model(global_args.model, config=config)
    net = model.net
    device = next(net.parameters()).device

    # Pre-load all batches to GPU only for repeated inference (fi/hr modes).
    # Save and pa use model.dataloader directly — one batch at a time — so
    # they never need everything in GPU memory at once.
    batches = list(model.get_batches()) if (not save_args and not pa_args) else []

    if not commands:
        print("Model loaded. Add a subcommand (fi, hr, save, pa) to run an experiment.")
        return

    # Detector wraps model layers — must happen before injector indexes them
    detector = None
    if hr_args and hr_args.detect:
        from detection import Baseline, CheckOne, Checksum

        _METHODS = {"checkone": CheckOne, "checksum": Checksum, "baseline": Baseline}
        method = getattr(hr_args, "method", "checkone")
        correction = getattr(hr_args, "correction", None)
        detector = _METHODS.get(method, CheckOne)(
            model, layers=hr_args.detect, correction=correction
        )
        detector.load(verbose=global_args.info)

    # Injector indexes layer weights — must come after detector wraps them
    injector = None
    if fi_args:
        from cli import parse_bit_range
        from injection import Injector

        bit_range = parse_bit_range(fi_args.bit_range) if fi_args.bit_range else None
        injector = Injector(model, layers=fi_args.layers, bit_range=bit_range)
        injector.fi_faults = fi_args.faults
        injector.fi_ber = fi_args.ber
        if detector:
            injector.refresh_layers()  # re-index after detector wrapped layers
        if global_args.info:
            injector.print_layer_info()

    # Fault-free logits for SDC comparison (not used in save/pa modes)
    ff_logits = (
        model.ff_logits
        if (model.ff_logits.available and not save_args and not pa_args)
        else None
    )

    # Accumulators — active only in evaluation modes
    from eval.accuracy import AccuracyTracker
    from eval.sdc import SDCTracker

    acc = AccuracyTracker() if (not save_args and not pa_args) else None
    sdc = SDCTracker() if ff_logits else None

    # ── WARMUP ────────────────────────────────────────────────────────────────
    if global_args.warmup > 0:
        n = global_args.warmup
        print(f"Warming up ({n} pass{'es' if n > 1 else ''})...")
        for _ in range(n):
            for images, _ in batches:
                with torch.inference_mode():
                    net(images)
        print("Warmup done.")

    # ── RUNS ──────────────────────────────────────────────────────────────────
    all_runs: list[dict] = []
    layer_stats: dict[str, dict[str, int]] = {}  # detection accuracy per layer

    for run_idx in range(global_args.repeat):
        if global_args.info and global_args.repeat > 1:
            print(f"\n{'=' * 60}\nRUN {run_idx + 1}/{global_args.repeat}\n{'=' * 60}")

        # Inject faults
        if injector:
            if fi_args is not None and fi_args.fault_seed is not None:
                random.seed(fi_args.fault_seed + run_idx)
            if injector.fi_ber is not None:
                injector.inject(ber=injector.fi_ber)
            else:
                injector.inject(count=injector.fi_faults or 1)
            if global_args.info:
                injector.print_info()

        # Reset per-run accumulators
        if acc:
            acc.reset()
        if sdc:
            sdc.reset()
        if detector:
            detector.reset()

        times_ms: list[float] = []

        # ── MODEL RUNS ────────────────────────────────────────────────────────
        for batch_idx, (images, labels) in enumerate(batches):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                outputs = net(images)  # detector hooks fire here when active
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000)

            if acc:
                acc.update_batch(outputs, labels)
            if sdc and ff_logits:
                sdc.update_batch(
                    outputs, ff_logits.get_batch(batch_idx, len(images), device)
                )

        # ── COLLECT RUN RESULTS ───────────────────────────────────────────────
        run_data: dict = {"run": run_idx + 1, "times_ms": times_ms}

        if acc:
            acc_result = acc.get_results()
            acc.aggregate_run(acc_result)
            run_data["accuracy"] = acc_result

        if sdc:
            sdc_result = sdc.get_results()
            sdc.aggregate_run(sdc_result)
            run_data["sdc"] = sdc_result

        if detector:
            run_data["detection"] = detector.get_values()

            # Track injection vs detection accuracy per layer
            injected_layers = (
                {f["layer"] for f in injector.get_info()} if injector else set()
            )

            # Separate weight faults from input faults when detector supports it
            if hasattr(detector, "get_input_faults"):
                weight_detected = {f.layer for f in detector.get_weight_faults()}
                input_detected = {f.layer for f in detector.get_input_faults()}
            else:
                weight_detected = {f.layer for f in detector.get_faults()}
                input_detected = set()

            for layer in injected_layers | weight_detected | input_detected:
                s = layer_stats.setdefault(
                    layer,
                    {
                        "injected": 0,
                        "detected": 0,
                        "false_positive": 0,
                        "input_faults": 0,
                    },
                )
                if layer in injected_layers:
                    s["injected"] += 1
                    if layer in weight_detected:
                        s["detected"] += 1
                elif layer in weight_detected:
                    s["false_positive"] += 1
                if layer in input_detected:
                    s["input_faults"] += 1

            # Verbose per-run fault detail
            if global_args.info:
                detector.print_results()

        if injector:
            run_data["faults"] = injector.get_info()
            injector.restore()

        all_runs.append(run_data)

        if global_args.repeat > 1 and not global_args.info:
            end = "\n" if run_idx + 1 == global_args.repeat else "\r"
            print(f"  [{run_idx + 1}/{global_args.repeat}]", end=end, flush=True)

    # ── PRINT ─────────────────────────────────────────────────────────────────
    if not save_args and not pa_args and all_runs:
        n_samples = all_runs[-1].get("accuracy", {}).get("samples", "?")
        print(
            f"\n{'-' * 60}\n"
            f"RESULTS  |  model={global_args.model}  samples={n_samples}  runs={global_args.repeat}\n"
            f"{'-' * 60}"
        )
        if acc:
            acc.print_summary()
        if sdc:
            sdc.print_summary()
        if detector and global_args.repeat == 1:
            detector.print_results()
        if detector and layer_stats:
            detector.print_summary(layer_stats)
        if detector and getattr(hr_args, "time", False):
            detector.print_timing_summary(all_runs)
        _print_timing(all_runs, fi_args, global_args)

    # ── PA ────────────────────────────────────────────────────────────────────
    if pa_args:
        from analysis import ActivationAnalyzer, WeightAnalyzer

        output_base = pa_args.output or f"results/{global_args.model}"

        if pa_args.type in ["activations", "both"]:
            print("\n=== Activation Analysis ===")
            analyzer = ActivationAnalyzer(model)
            analyzer.run()
            analyzer.save(f"{output_base}_activations.json")
            analyzer.remove()

        if pa_args.type in ["weights", "both"]:
            print("\n=== Weight Analysis ===")
            wa = WeightAnalyzer(model)
            wa.run()
            wa.save(f"{output_base}_weights.json")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    if save_args:
        if save_args.logits:
            # Collect logits lazily — one batch at a time, never all in GPU at once
            logits_buf: list[torch.Tensor] = []
            labels_buf: list[torch.Tensor] = []
            max_batches = global_args.max_batches
            print(f"Collecting fault-free logits ({max_batches or 'all'} batches)...")
            for batch_idx, (images, labels) in enumerate(model.dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images = images.to(device, non_blocking=True)
                with torch.inference_mode():
                    outputs = net(images)
                logits_buf.append(outputs.cpu())
                labels_buf.append(labels.cpu())
            n = sum(b.shape[0] for b in logits_buf)
            model.ff_logits.save(logits_buf, labels_buf, n)

        if save_args.threshold or save_args.inputs or save_args.weights:
            from detection import CheckOne, Checksum

            max_batches = global_args.max_batches  # None = full dataset

            do_calib = save_args.inputs or save_args.threshold

            # Calibrate and save CheckOne (single pass for both inputs+threshold)
            print("\nCalibrating CheckOne...")
            co = CheckOne(model, layers=save_args.layers)
            if do_calib:
                co.calibrate(
                    model,
                    max_batches=max_batches,
                    inputs=save_args.inputs,
                    threshold=save_args.threshold,
                    margin=save_args.margin,
                )
            co.save(include_weights=save_args.weights, save_calibration=do_calib)
            co.remove()

            # Calibrate and save Checksum
            print("\nCalibrating Checksum...")
            cs = Checksum(model, layers=save_args.layers)
            if save_args.threshold:
                cs.calibrate_threshold(
                    model, max_batches=max_batches, margin=save_args.margin
                )
            cs.save(
                include_weights=save_args.weights, save_calibration=save_args.threshold
            )
            cs.remove()

    if global_args.output and all_runs:
        _save_json(global_args.output, all_runs, global_args, fi_args, acc, sdc)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_commands(commands):
    """Extract per-subcommand arg namespaces from the command chain."""
    fi_args = hr_args = save_args = pa_args = None
    for cmd_name, cmd_args in commands:
        if cmd_name == "fi":
            fi_args = cmd_args
        elif cmd_name == "hr":
            hr_args = cmd_args
        elif cmd_name == "save":
            save_args = cmd_args
        elif cmd_name == "pa":
            pa_args = cmd_args
    return fi_args, hr_args, save_args, pa_args


def _print_timing(all_runs: list[dict], fi_args, global_args):
    """Print forward-pass timing summary if --time is set."""
    if not (
        global_args.time or (fi_args is not None and getattr(fi_args, "time", False))
    ):
        return
    all_batch_ms = [t for r in all_runs for t in r.get("times_ms", [])]
    per_run_mps = [
        sum(r["times_ms"]) / (r.get("accuracy", {}).get("samples", 1) or 1)
        for r in all_runs
        if r.get("times_ms")
    ]
    total_ms = sum(all_batch_ms)
    avg_mps = sum(per_run_mps) / len(per_run_mps) if per_run_mps else 0.0
    print(
        f"\nInference Timing ({len(all_runs)} runs):\n"
        f"  Total forward: {total_ms:.1f} ms  ({total_ms / 1000:.3f} s)\n"
        f"  ms/sample: avg={avg_mps:.4f}  "
        f"min={min(per_run_mps):.4f}  max={max(per_run_mps):.4f}"
    )
    if all_batch_ms:
        print(
            f"  Per-batch: avg={_stlib.mean(all_batch_ms):.2f} ms  "
            f"min={min(all_batch_ms):.2f}  max={max(all_batch_ms):.2f}"
        )


def _save_json(output_path: str, all_runs: list[dict], global_args, fi_args, acc, sdc):
    """Append an aggregated summary to the output JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list = []
    if path.exists():
        try:
            with open(path) as f:
                loaded = json.load(f)
                existing = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, OSError):
            pass

    existing.append(_build_json_summary(all_runs, global_args, fi_args, acc, sdc))

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved results to {output_path}")


def _build_json_summary(all_runs: list[dict], global_args, fi_args, acc, sdc) -> dict:
    """Build the plot-compatible aggregated summary dict."""
    n: int = len(all_runs)
    summary: dict[str, object] = {"total_runs": n}

    # Accuracy
    if acc:
        s = acc.get_summary()
        summary.update(
            {
                "avg_top1_acc": s["avg_top1"],
                "std_top1_acc": s["std_top1"],
                "avg_top5_acc": s["avg_top5"],
                "std_top5_acc": s["std_top5"],
            }
        )

    # Timing
    per_run_mps = [
        sum(r["times_ms"]) / (r.get("accuracy", {}).get("samples", 1) or 1)
        for r in all_runs
        if r.get("times_ms")
    ]
    if per_run_mps:
        avg_mps = sum(per_run_mps) / len(per_run_mps)
        std_mps = (
            math.sqrt(
                sum((v - avg_mps) ** 2 for v in per_run_mps) / (len(per_run_mps) - 1)
            )
            if len(per_run_mps) > 1
            else 0.0
        )
        summary["avg_ms_per_sample"] = avg_mps
        summary["std_ms_per_sample"] = std_mps

    # SDC
    if sdc:
        s = sdc.get_summary()
        sdc_vals = [r["sdc"]["logit_sdc_rate"] for r in all_runs if r.get("sdc")]
        avg_logit_sdc = s["avg_sdc_rate"]
        std_logit_sdc = (
            math.sqrt(
                sum((v - avg_logit_sdc) ** 2 for v in sdc_vals) / (len(sdc_vals) - 1)
            )
            if len(sdc_vals) > 1
            else 0.0
        )
        summary.update(
            {
                "avg_logit_sdc": avg_logit_sdc,
                "std_logit_sdc": std_logit_sdc,
                "avg_msdc": s["avg_msdc"],
                "avg_critical_top1_sdc": s["avg_critical_top1"],
                "std_critical_top1_sdc": s["std_critical_top1"],
                "avg_critical_top5_sdc": s["avg_critical_top5"],
                "std_critical_top5_sdc": s["std_critical_top5"],
                "high_risk_count": s["high_risk"],
                "high_risk_pct": 100.0 * s["high_risk"] / n if n else 0.0,
                "medium_risk_count": s["medium_risk"],
                "medium_risk_pct": 100.0 * s["medium_risk"] / n if n else 0.0,
                "safe_count": s["safe"],
                "safe_pct": 100.0 * s["safe"] / n if n else 0.0,
            }
        )
        for pct in [1, 5, 10, 15, 20, 25, 50]:
            key = f"avg_sdc_{pct}pct"
            summary[key] = s.get(key, 0.0)

    # Config metadata
    sub_component = getattr(fi_args, "component", None) if fi_args else None
    block_idx = getattr(fi_args, "block", None) if fi_args else None
    component = (
        "attention"
        if sub_component in ("qkv", "proj")
        else "mlp"
        if sub_component in ("fc1", "fc2")
        else None
    )
    summary["config"] = {
        "model": global_args.model,
        "sub_component": sub_component,
        "block_idx": block_idx,
        "component": component,
    }

    return summary
