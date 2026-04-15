"""Debug script to verify SDC calculations."""
import torch
import sys
sys.path.insert(0, "src")

from core.model import Model, ModelConfig
from injection import Injector


def main():
    # Load model and get fault-free logits
    config = ModelConfig(batch_size=10, max_batches=1)
    model = Model("vit_tiny", config=config)

    # Get one batch
    batches = model.get_batches()
    images, labels = batches[0]

    # Get fault-free output
    with torch.inference_mode():
        ff_logits = model.net(images).clone()

    print("=== Fault-Free Logit Statistics ===")
    print(f"Logits shape: {ff_logits.shape}")
    print(f"Mean absolute logit: {ff_logits.abs().mean().item():.4f}")
    print(f"Max logit: {ff_logits.max().item():.4f}")
    print(f"Min logit: {ff_logits.min().item():.4f}")

    mean_abs = ff_logits.abs().mean(dim=1)
    print(f"\nPer-sample mean |logit|: min={mean_abs.min():.4f}, max={mean_abs.max():.4f}, mean={mean_abs.mean():.4f}")

    # What MSDC would trigger thresholds?
    print(f"\nTo trigger 1% threshold, MSDC needs to be >= {0.01 * mean_abs.mean().item():.4f}")
    print(f"To trigger 5% threshold, MSDC needs to be >= {0.05 * mean_abs.mean().item():.4f}")

    # Now inject faults and measure
    print("\n=== Injecting 10 faults (random bits) ===")
    injector = Injector(model, layers="fc1")
    injector.inject(count=10)

    with torch.inference_mode():
        faulty_logits = model.net(images)

    diff = ff_logits - faulty_logits
    print(f"\nDiff statistics:")
    print(f"  Mean |diff|: {diff.abs().mean().item():.6f}")
    print(f"  Max |diff|: {diff.abs().max().item():.6f}")
    print(f"  Non-zero diffs: {(diff != 0).sum().item()} / {diff.numel()}")

    # Compute relative change as done in SDC tracker
    relative_change = diff.abs().mean(dim=1) / (ff_logits.abs().mean(dim=1) + 1e-10)
    print(f"\nRelative change per sample:")
    print(f"  Min: {relative_change.min().item()*100:.4f}%")
    print(f"  Max: {relative_change.max().item()*100:.4f}%")
    print(f"  Mean: {relative_change.mean().item()*100:.4f}%")

    # Count samples exceeding thresholds
    for thresh in [0.01, 0.05, 0.10]:
        pct = (relative_change >= thresh).float().mean().item() * 100
        print(f"  Samples >= {thresh*100:.0f}%: {pct:.1f}%")

    # Check critical SDC
    pred_ff = ff_logits.argmax(dim=1)
    pred_faulty = faulty_logits.argmax(dim=1)
    pred_changed = (pred_ff != pred_faulty)
    print(f"\nPrediction changes: {pred_changed.sum().item()} / {len(pred_changed)}")

    # Check if original prediction's logit was corrupted
    idx = torch.arange(len(pred_ff), device=ff_logits.device)
    orig_logit_ff = ff_logits[idx, pred_ff]
    orig_logit_faulty = faulty_logits[idx, pred_ff]
    logit_changed = orig_logit_ff != orig_logit_faulty
    print(f"Original logit changed: {logit_changed.sum().item()} / {len(logit_changed)}")

    critical = pred_changed & logit_changed
    print(f"Critical SDC (both): {critical.sum().item()} / {len(critical)}")

    injector.restore()

    # Test with more aggressive faults (high bits)
    print("\n=== Injecting 10 faults in high bits (23-31) ===")
    injector2 = Injector(model, layers="fc1", bit_range=(23, 31))
    injector2.inject(count=10)

    with torch.inference_mode():
        faulty_logits2 = model.net(images)

    diff2 = ff_logits - faulty_logits2
    print(f"\nDiff statistics (high bits):")
    print(f"  Mean |diff|: {diff2.abs().mean().item():.6f}")
    print(f"  Max |diff|: {diff2.abs().max().item():.6f}")

    relative_change2 = diff2.abs().mean(dim=1) / (ff_logits.abs().mean(dim=1) + 1e-10)
    print(f"\nRelative change per sample:")
    print(f"  Min: {relative_change2.min().item()*100:.4f}%")
    print(f"  Max: {relative_change2.max().item()*100:.4f}%")
    print(f"  Mean: {relative_change2.mean().item()*100:.4f}%")

    for thresh in [0.01, 0.05, 0.10]:
        pct = (relative_change2 >= thresh).float().mean().item() * 100
        print(f"  Samples >= {thresh*100:.0f}%: {pct:.1f}%")

    pred_faulty2 = faulty_logits2.argmax(dim=1)
    pred_changed2 = (pred_ff != pred_faulty2)
    print(f"\nPrediction changes: {pred_changed2.sum().item()} / {len(pred_changed2)}")

    injector2.restore()


if __name__ == "__main__":
    main()
