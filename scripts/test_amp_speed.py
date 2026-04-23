"""Quick AMP vs float32 speed comparison."""
import time
import torch
import timm

MODEL = "vit_tiny_patch16_224"
BATCH_SIZE = 100
N_BATCHES = 10
WARMUP = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = timm.create_model(MODEL, pretrained=False).to(device).eval()
dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)

def benchmark(use_amp: bool) -> float:
    # Warmup
    for _ in range(WARMUP):
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_BATCHES):
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_BATCHES * 1000

fp32 = benchmark(use_amp=False)
amp  = benchmark(use_amp=True)

print(f"float32: {fp32:.1f} ms/batch")
print(f"AMP:     {amp:.1f} ms/batch")
print(f"Speedup: {fp32/amp:.2f}x")
