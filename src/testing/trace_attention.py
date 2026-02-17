"""Trace tensors through ViT attention - healthy vs faulty comparison.

Outputs two files:
  - trace_healthy.txt : Normal forward pass (no fault)
  - trace_faulty.txt  : Forward pass with fault injected

Run:
    python src/testing/trace_attention.py
"""

import torch
import timm
from torchvision import transforms
from PIL import Image
import os

LAYER = 0  # Which transformer block
TARGET = "k"  # "q", "k", or "v"
FAULT_ROW = 10  # Row within Q/K/V section
FAULT_COL = 5  # Column in weight matrix
FAULT_VALUE = float("nan")  # Value to inject (try: float("nan"), 1e38, 999.0)
IMAGE_PATH = "/home/samiel/Documents/thesis/old/ignore/nvm/imagenet1k/imagenet-val/n01440764/ILSVRC2012_val_00000293.JPEG"


def fmt_tensor(t):
    """Format tensor as aligned matrix string (FULL tensor)."""
    if t.dim() == 1:
        t = t.unsqueeze(0)
    t = t.detach().cpu().float()
    rows, cols = t.shape
    lines = []
    for i in range(rows):
        row_vals = []
        for j in range(cols):
            v = t[i, j].item()
            if abs(v) < 1e-3 or abs(v) > 1e4:
                row_vals.append(f"{v:12.4e}")
            else:
                row_vals.append(f"{v:12.6f}")
        lines.append(" ".join(row_vals))
    return "\n".join(lines)


def write_section(f, title, tensor, row_idx=0, col_idx=0):
    """Write tensor section with CLS, row, and column."""
    f.write(f"\n{'='*60}\n{title}\n{'='*60}\n")
    f.write(f"Shape: {tuple(tensor.shape)}\n")
    f.write(f"NaN count: {torch.isnan(tensor).sum().item()}\n")
    valid = tensor[~torch.isnan(tensor)]
    if valid.numel() > 0:
        f.write(f"Min: {valid.min().item():.4e}, Max: {valid.max().item():.4e}\n\n")
    else:
        f.write(f"Min: N/A (all NaN), Max: N/A (all NaN)\n\n")

    t = tensor[0] if tensor.dim() == 3 else tensor
    f.write("Full tensor:\n")
    f.write(fmt_tensor(t) + "\n\n")

    if t.dim() == 2 and t.shape[0] > 1:
        f.write(f"CLS token (row 0):\n{fmt_tensor(t[0])}\n\n")
    if row_idx < t.shape[0]:
        f.write(f"Row {row_idx}:\n{fmt_tensor(t[row_idx])}\n\n")
    if t.dim() == 2 and col_idx < t.shape[1]:
        f.write(f"Column {col_idx}:\n{fmt_tensor(t[:, col_idx])}\n\n")


def print_structure(model):
    """Print transformer structure."""
    print("\n" + "="*70)
    print("VIT TRANSFORMER STRUCTURE")
    print("="*70)
    print(f"\nModel: vit_tiny_patch16_224")
    print(f"  Embed dim: 192, Heads: 3, Blocks: {len(model.blocks)}")
    print("\n--- BLOCK STRUCTURE ---")
    print("  norm1 -> attn(qkv -> q,k,v -> scores -> softmax -> out -> proj) -> residual")
    print("  norm2 -> mlp(fc1 -> gelu -> fc2) -> residual")
    print("\n--- ATTENTION STEPS ---")
    print("  1. qkv = linear(x)        [B,197,576]")
    print("  2. q,k,v = split(qkv)     [B,197,192] each")
    print("  3. scores = q @ k.T       [B,3,197,197]")
    print("  4. weights = softmax      [B,3,197,197]")
    print("  5. out = weights @ v      [B,197,192]")
    print("  6. out = proj(out)        [B,197,192]")
    print("="*70)


def run_attention(model, x, block, attn, inject_fault, fault_row, fault_col, fault_value, offset, embed_dim):
    """Run attention and return intermediate tensors."""
    qkv_weight = attn.qkv.weight
    actual_row = offset + fault_row
    original_value = qkv_weight[actual_row, fault_col].item()

    with torch.no_grad():
        # Pre-processing
        x_embed = model.patch_embed(x)
        cls_token = model.cls_token.expand(x_embed.shape[0], -1, -1)
        x_embed = torch.cat([cls_token, x_embed], dim=1)
        x_embed = x_embed + model.pos_embed
        x_embed = model.pos_drop(x_embed)

        for i in range(LAYER):
            x_embed = model.blocks[i](x_embed)

        x_normed = block.norm1(x_embed)

        # Inject fault if requested
        if inject_fault:
            qkv_weight[actual_row, fault_col] = fault_value

        # QKV projection
        qkv = attn.qkv(x_normed)
        B, N, _ = qkv.shape
        qkv_split = qkv.reshape(B, N, 3, embed_dim)
        q, k, v = qkv_split[:, :, 0], qkv_split[:, :, 1], qkv_split[:, :, 2]

        # Attention
        num_heads = attn.num_heads
        head_dim = embed_dim // num_heads
        q_heads = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k_heads = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v_heads = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        scale = head_dim ** -0.5
        attn_scores = (q_heads @ k_heads.transpose(-2, -1)) * scale
        attn_weights = attn_scores.softmax(dim=-1)
        attn_out = (attn_weights @ v_heads).transpose(1, 2).reshape(B, N, embed_dim)
        attn_out = attn.proj(attn_out)
        x_after_attn = x_embed + attn_out

        # Restore weight
        if inject_fault:
            qkv_weight[actual_row, fault_col] = original_value

    target_tensor = {"q": q, "k": k, "v": v}[TARGET]
    weight_section = qkv_weight[offset:offset + embed_dim, :].clone()

    return {
        "input": x_normed,
        "weight": weight_section,
        "target_output": target_tensor,
        "attn_scores": attn_scores,
        "attn_weights": attn_weights,
        "final_output": x_after_attn,
        "original_value": original_value,
    }


def main():
    print("Loading model and image...")
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    model.eval()
    print_structure(model)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(IMAGE_PATH).convert("RGB")
    x = transform(img).unsqueeze(0)

    block = model.blocks[LAYER]
    attn = block.attn
    embed_dim = attn.qkv.weight.shape[1]
    offset = {"q": 0, "k": embed_dim, "v": 2 * embed_dim}[TARGET]

    print(f"\n--- SETTINGS ---")
    print(f"Layer: {LAYER}, Target: {TARGET.upper()}")
    print(f"Fault: weight[{FAULT_ROW}, {FAULT_COL}] = {FAULT_VALUE}")

    # Run healthy
    print("\n--- RUNNING HEALTHY ---")
    healthy = run_attention(model, x, block, attn, False, FAULT_ROW, FAULT_COL, FAULT_VALUE, offset, embed_dim)
    print(f"  Output NaN count: {torch.isnan(healthy['final_output']).sum().item()}")

    # Run faulty
    print("\n--- RUNNING FAULTY ---")
    faulty = run_attention(model, x, block, attn, True, FAULT_ROW, FAULT_COL, FAULT_VALUE, offset, embed_dim)
    print(f"  Output NaN count: {torch.isnan(faulty['final_output']).sum().item()}")

    # Write healthy file
    with open("trace_healthy.txt", "w") as f:
        f.write("ViT Attention Trace - HEALTHY (no fault)\n")
        f.write(f"Layer: {LAYER}, Target: {TARGET.upper()}\n")
        write_section(f, "INPUT TO ATTENTION", healthy["input"], FAULT_ROW, FAULT_COL)
        write_section(f, f"{TARGET.upper()} WEIGHT TENSOR", healthy["weight"], FAULT_ROW, FAULT_COL)
        write_section(f, f"OUTPUT AFTER INPUT @ {TARGET.upper()}_WEIGHT", healthy["target_output"], FAULT_ROW, FAULT_COL)
        write_section(f, "OUTPUT AFTER FULL ATTENTION BLOCK", healthy["final_output"], FAULT_ROW, FAULT_COL)

    # Write faulty file
    with open("trace_faulty.txt", "w") as f:
        f.write("ViT Attention Trace - FAULTY\n")
        f.write(f"Layer: {LAYER}, Target: {TARGET.upper()}\n")
        f.write(f"Fault: weight[{FAULT_ROW}, {FAULT_COL}] = {FAULT_VALUE}\n")
        f.write(f"Original value: {faulty['original_value']:.6e}\n")
        write_section(f, "INPUT TO ATTENTION", faulty["input"], FAULT_ROW, FAULT_COL)
        write_section(f, f"{TARGET.upper()} WEIGHT TENSOR (with fault)", faulty["weight"], FAULT_ROW, FAULT_COL)
        write_section(f, f"OUTPUT AFTER INPUT @ {TARGET.upper()}_WEIGHT", faulty["target_output"], FAULT_ROW, FAULT_COL)
        write_section(f, "OUTPUT AFTER FULL ATTENTION BLOCK", faulty["final_output"], FAULT_ROW, FAULT_COL)

    print(f"\nOutput files:")
    print(f"  trace_healthy.txt : {os.path.getsize('trace_healthy.txt') / 1024:.1f} KB")
    print(f"  trace_faulty.txt  : {os.path.getsize('trace_faulty.txt') / 1024:.1f} KB")


if __name__ == "__main__":
    main()
