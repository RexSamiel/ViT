#!/usr/bin/env python3
"""Trace model forward pass - shows all operations with input/output shapes."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn


def trace_forward(model, input_shape=(1, 3, 224, 224), max_ops=None):
    """Trace forward pass showing all operations."""

    if hasattr(model, "net"):
        net = model.net
        device = model.config.device if hasattr(model, "config") else "cpu"
    else:
        net = model
        try:
            device = next(net.parameters()).device
        except StopIteration:
            device = "cpu"

    traces = []
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            inp_shape = inp[0].shape if isinstance(inp, tuple) and len(inp) > 0 else "?"
            out_shape = out.shape if hasattr(out, "shape") else "?"
            traces.append({
                "name": name,
                "type": module.__class__.__name__,
                "input": inp_shape,
                "output": out_shape,
            })
        return hook

    # Register hooks on all modules
    for name, module in net.named_modules():
        if name:  # Skip root module
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Run forward pass
    dummy = torch.randn(*input_shape, device=device)
    with torch.inference_mode():
        _ = net(dummy)

    # Remove hooks
    for h in hooks:
        h.remove()

    return traces


def print_traces(traces, max_ops=None, filter_type=None):
    """Print traces in a readable format."""

    if filter_type:
        traces = [t for t in traces if filter_type.lower() in t["type"].lower()]

    if max_ops:
        traces = traces[:max_ops]

    print(f"\n{'Layer Name':<45} {'Type':<20} {'Input Shape':<25} {'Output Shape':<25}")
    print("=" * 115)

    for t in traces:
        inp = str(tuple(t["input"])) if hasattr(t["input"], "__iter__") else str(t["input"])
        out = str(tuple(t["output"])) if hasattr(t["output"], "__iter__") else str(t["output"])
        print(f"{t['name']:<45} {t['type']:<20} {inp:<25} {out:<25}")

    print("=" * 115)
    print(f"Total operations: {len(traces)}")


def print_summary(traces):
    """Print summary of operation types."""
    from collections import Counter

    types = Counter(t["type"] for t in traces)

    print("\nOperation Summary:")
    print("-" * 40)
    for op_type, count in types.most_common():
        print(f"  {op_type:<25} {count:>5}")
    print("-" * 40)
    print(f"  {'Total':<25} {len(traces):>5}")


def main():
    parser = argparse.ArgumentParser(description="Trace model forward pass")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="vit_tiny",
        help="Model name (default: vit_tiny)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max operations to show",
    )
    parser.add_argument(
        "--filter", "-f",
        type=str,
        default=None,
        help="Filter by operation type (e.g., 'linear', 'norm')",
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show summary of operation types",
    )
    parser.add_argument(
        "--linear-only", "-l",
        action="store_true",
        help="Show only Linear layers",
    )
    args = parser.parse_args()

    import timm
    from core.model import SUPPORTED_MODELS

    # Load model directly (no data needed for tracing)
    model_name = SUPPORTED_MODELS.get(args.model, args.model)
    print(f"Loading model: {model_name}")
    net = timm.create_model(model_name, pretrained=False)
    net.eval()

    print(f"Tracing {args.model}...")
    traces = trace_forward(net)

    filter_type = "linear" if args.linear_only else args.filter
    print_traces(traces, max_ops=args.max, filter_type=filter_type)

    if args.summary:
        print_summary(traces)


if __name__ == "__main__":
    main()
