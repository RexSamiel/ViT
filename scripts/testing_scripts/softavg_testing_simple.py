import random

import torch

torch.set_printoptions(sci_mode=False, precision=2, linewidth=200)
torch.manual_seed(42)

# N=2 tokens, C_in=3 features, C_out=3 outputs
x = torch.randint(1, 10, (2, 3)).float()

inject_fault = True
fault_row, fault_col = 1, 1


def flip_bit(W: torch.Tensor, row: int, col: int) -> torch.Tensor:
    tensor = W.clone()
    tensor_int = tensor.view(torch.int32)
    tensor_int[row, col] ^= 1 << 31
    return tensor


# Weight matrix [C_out=3, C_in=3]
W = torch.rand(3, 3)
W_golden_row_sums = W.sum(dim=1).clone()

if inject_fault:
    print(f"W[{fault_row},{fault_col}] before: {W[fault_row, fault_col]:.4f}")
    W = flip_bit(W, fault_row, fault_col)
    print(f"W[{fault_row},{fault_col}] after:  {W[fault_row, fault_col]:.4f}\n")

# Build W_ext: append ones row to W  →  W.T gets ones column on the right
W_ones = torch.ones(1, W.shape[1])
W_ext = torch.cat([W, W_ones], dim=0)  # [C_out+1, C_in]

# Build x_ext: append ones row to x
x_ones = torch.ones(1, x.shape[1])
x_ext = torch.cat([x, x_ones], dim=0)  # [N+1, C_in]

# Single extended matmul
out_ext = x_ext @ W_ext.T  # [N+1, C_out+1]

# Extract blocks
N, C_out = x.shape[0], W.shape[0]
out = out_ext[:N, :C_out]  # normal output
weight_check = out_ext[N, :C_out]  # ones @ W = W.sum(dim=1)  — weight check
input_check = out_ext[:N, C_out]  # x @ ones = x.sum(dim=1)  — input check

# --- PRINT ---
print("x with ones row at bottom (x_ext):\n", x_ext)
print("\nW.T with ones column at right, row sums as last row:")
W_T_ext = torch.cat([W_ext.T, W_ext.sum(dim=1, keepdim=True).T], dim=0)
print(W_T_ext)

print("\nout_ext  [N+1, C_out+1]:\n", out_ext)
print("\nnormal output:\n", out)
print("\nweight_check :", weight_check)
print("golden sums  :", W_golden_row_sums)
print("diff         :", weight_check - W_golden_row_sums)
print("\ninput_check  :", input_check)
print("x.sum(dim=1) :", x.sum(dim=1))
