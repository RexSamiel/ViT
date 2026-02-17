"""Simple test script to trace fault propagation through ViT attention.

Inject a fault into Q/K/V weights and see the tensors at each step:
1. After Q/K/V projection (input @ weights)
2. Before softmax (attention scores = Q @ K.T)
3. After softmax (attention weights)
4. Output of attention block

Usage:
    python src/testing/testing_for_nan.py --target k --bit 30 --row 10 --col 5
    python src/testing/testing_for_nan.py --target k --bit 30 --find-nan  # Find a position that creates NaN
"""

import argparse
import torch
import timm


def will_create_nan(value: float, bit: int) -> bool:
    """Check if flipping this bit will create NaN."""
    val_int = torch.tensor(value).view(torch.int32).item()
    mask = 1 << bit
    corrupted_int = val_int ^ mask
    corrupted = torch.tensor(corrupted_int, dtype=torch.int32).view(torch.float32).item()
    return torch.isnan(torch.tensor(corrupted)).item()


def find_nan_position(weight: torch.Tensor, bit: int, target_offset: int, embed_dim: int) -> tuple[int, int, int] | None:
    """Find a position in the weight tensor that will create NaN when bit is flipped.

    If bit=-1, search all exponent bits (23-30) to find any that creates NaN.
    Returns (row, col, bit) or None.
    """
    bits_to_try = [bit] if bit >= 0 else list(range(23, 31))

    for b in bits_to_try:
        for row in range(embed_dim):
            for col in range(embed_dim):
                actual_row = target_offset + row
                if will_create_nan(weight[actual_row, col].item(), b):
                    return row, col, b
    return None


def flip_bit(tensor: torch.Tensor, row: int, col: int, bit: int) -> tuple[float, float]:
    """Flip a specific bit in the weight tensor. Returns (original, corrupted) values.

    If bit=-1, inject NaN directly.
    """
    original = tensor[row, col].item()

    if bit == -1:
        # Inject NaN directly
        corrupted = float('nan')
    else:
        val_int = tensor[row, col].view(torch.int32)
        mask = torch.tensor(1 << bit, dtype=torch.int32, device=tensor.device)
        corrupted_int = val_int ^ mask
        corrupted = corrupted_int.view(torch.float32).item()

    with torch.no_grad():
        tensor[row, col] = corrupted

    return original, corrupted


def print_tensor(name: str, t: torch.Tensor, max_elements: int = 50):
    """Print tensor with NaN/Inf detection."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Shape: {tuple(t.shape)}")

    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    total = t.numel()

    print(f"NaN: {nan_count}/{total}, Inf: {inf_count}/{total}")

    if nan_count == 0 and inf_count == 0:
        print(f"Min: {t.min().item():.4e}, Max: {t.max().item():.4e}")

    # Print sample of the tensor
    flat = t.flatten()
    if len(flat) <= max_elements:
        print(f"Values:\n{t}")
    else:
        print(f"First {max_elements} values: {flat[:max_elements].tolist()}")

        # Show where NaNs are if present
        if nan_count > 0:
            nan_indices = torch.nonzero(torch.isnan(t.flatten()), as_tuple=False).squeeze()
            if nan_indices.numel() > 0:
                indices = nan_indices[:10].tolist() if nan_indices.numel() > 10 else nan_indices.tolist()
                print(f"NaN at flat indices (first 10): {indices}")


def run_test(layer: int, target: str, bit: int, row: int, col: int, print_all: bool, find_nan: bool):
    """Run the fault injection test.

    Args:
        layer: Which transformer block (0-11 for vit_tiny)
        target: Which weight to inject into ('q', 'k', 'v')
        bit: Which bit to flip (0-31)
        row: Row in weight matrix
        col: Column in weight matrix
        print_all: Print all intermediate tensors
        find_nan: Auto-find a position that creates NaN
    """
    print(f"\nLoading vit_tiny_patch16_224...")
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    model.eval()

    # Get the attention module
    block = model.blocks[layer]
    attn = block.attn

    # QKV weights are combined: shape [3*embed_dim, embed_dim] = [576, 192] for vit_tiny
    # Q weights: rows 0-191
    # K weights: rows 192-383
    # V weights: rows 384-575
    qkv_weight = attn.qkv.weight
    embed_dim = qkv_weight.shape[1]  # 192 for vit_tiny

    print(f"\nQKV weight shape: {tuple(qkv_weight.shape)}")
    print(f"Embed dim: {embed_dim}")
    print(f"Q weights: rows 0-{embed_dim-1}")
    print(f"K weights: rows {embed_dim}-{2*embed_dim-1}")
    print(f"V weights: rows {2*embed_dim}-{3*embed_dim-1}")

    # Calculate target offset
    target_offset = {'q': 0, 'k': embed_dim, 'v': 2 * embed_dim}[target]
    weight_name = target.upper()

    # Find NaN position if requested
    if find_nan:
        search_bit = bit if bit >= 0 else -1
        if search_bit >= 0:
            print(f"\nSearching for position that creates NaN with bit {bit} flip...")
        else:
            print(f"\nSearching for ANY position that creates NaN (trying bits 23-30)...")

        result = find_nan_position(qkv_weight, search_bit, target_offset, embed_dim)
        if result is None:
            print(f"No NaN-creating position found in {weight_name} weights.")
            print("The pretrained weights may not have values that create NaN with single bit flips.")
            print("Trying to create synthetic NaN instead...")
            # Just manually create a NaN for demonstration
            row, col = 0, 0
            bit = -1  # Signal to inject NaN directly
        else:
            row, col, bit = result
            print(f"Found: row={row}, col={col}, bit={bit}")

    # Calculate actual row based on target
    actual_row = target_offset + row

    # Validate indices
    max_row = embed_dim - 1
    max_col = embed_dim - 1
    if row > max_row:
        print(f"Warning: row {row} > max {max_row}, clamping")
        row = max_row
        actual_row = target_offset + row
    if col > max_col:
        print(f"Warning: col {col} > max {max_col}, clamping")
        col = max_col

    print(f"\n--- Injecting fault ---")
    print(f"Target: {weight_name} weights in layer {layer}")
    print(f"Position: weight[{actual_row}, {col}] (row {row} within {weight_name})")
    if bit == -1:
        print(f"Injection: Direct NaN (no natural bit flip creates NaN)")
    else:
        print(f"Bit: {bit}")

    # Inject the fault
    original, corrupted = flip_bit(qkv_weight, actual_row, col, bit)
    print(f"Original value: {original:.6e}")
    print(f"Corrupted value: {corrupted:.6e}")
    print(f"Is NaN: {torch.isnan(torch.tensor(corrupted)).item()}")
    print(f"Is Inf: {torch.isinf(torch.tensor(corrupted)).item()}")

    # Create dummy input - single image
    # ViT tiny: 224x224 -> 14x14 patches = 196 patches + 1 CLS = 197 tokens
    x = torch.randn(1, 3, 224, 224)

    print(f"\nInput image shape: {tuple(x.shape)}")

    # Manual forward pass through attention to capture intermediates
    # First, go through patch embed and blocks before target layer
    with torch.no_grad():
        # Patch embed
        x = model.patch_embed(x)

        # Add CLS token
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add position embedding
        x = x + model.pos_embed
        x = model.pos_drop(x)

        # Go through blocks before target
        for i in range(layer):
            x = model.blocks[i](x)

        print(f"\nInput to layer {layer} attention:")
        print(f"  Shape: {tuple(x.shape)}")  # [1, 197, 192]

        # Now manually compute attention to see intermediates
        B, N, C = x.shape  # batch, num_tokens, channels

        # Norm1
        x_normed = block.norm1(x)
        if print_all:
            print_tensor("After norm1", x_normed)

        # QKV projection - this uses the faulty weights!
        qkv = attn.qkv(x_normed)  # [B, N, 3*C]
        print_tensor("After QKV projection (contains fault)", qkv)

        # Reshape and split into Q, K, V
        qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        print_tensor(f"Q tensor (target={target})", q)
        print_tensor(f"K tensor (target={target})", k)
        print_tensor(f"V tensor (target={target})", v)

        # Attention scores = Q @ K.T
        scale = (C // attn.num_heads) ** -0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        print_tensor("Attention scores BEFORE softmax (Q @ K.T)", attn_scores)

        # Softmax
        attn_weights = attn_scores.softmax(dim=-1)
        print_tensor("Attention weights AFTER softmax", attn_weights)

        # Apply dropout (none in eval) and multiply by V
        attn_output = attn_weights @ v
        print_tensor("Attention output (weights @ V)", attn_output)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        # Output projection
        attn_output = attn.proj(attn_output)
        attn_output = attn.proj_drop(attn_output)
        print_tensor("After output projection", attn_output)

        # Add residual
        x_after_attn = x + attn_output
        print_tensor("After attention block (with residual)", x_after_attn)

        # Check CLS token specifically
        cls_after = x_after_attn[0, 0, :]  # First batch, first token (CLS)
        print_tensor("CLS token after attention", cls_after)


def main():
    parser = argparse.ArgumentParser(description="Test fault injection in ViT attention")
    parser.add_argument("--layer", type=int, default=0, help="Transformer block (0-11)")
    parser.add_argument("--target", type=str, default="k", choices=["q", "k", "v"],
                        help="Which weight to inject into: q, k, or v")
    parser.add_argument("--bit", type=int, default=30, help="Bit to flip (0-31)")
    parser.add_argument("--row", type=int, default=10, help="Row in Q/K/V weight matrix")
    parser.add_argument("--col", type=int, default=5, help="Column in weight matrix")
    parser.add_argument("--print-all", action="store_true", help="Print all intermediate tensors")
    parser.add_argument("--find-nan", action="store_true",
                        help="Auto-find a position that creates NaN when bit is flipped")

    args = parser.parse_args()

    run_test(args.layer, args.target, args.bit, args.row, args.col, args.print_all, args.find_nan)


if __name__ == "__main__":
    main()
