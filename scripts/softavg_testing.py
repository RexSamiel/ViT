import torch
import torch.nn.functional as F
import random
import timm

torch.set_printoptions(sci_mode=False, precision=8, linewidth=200)

MODEL_NAME = "vit_base_patch16_224"

EMBED_DIM = 768  # embedding dimension
NUM_HEADS = 12  # number of attention heads
HEAD_DIM = 64  # EMBED_DIM // NUM_HEADS
SEQ_LEN = 197  # (224/16)^2 + 1 CLS token = 196 + 1

fault_target = "None"

FAULT_BIT = 0
FAULT_VALUE = 50

CHECKSUM_METHOD = "mean"

print(f"Loading {MODEL_NAME}")
model = timm.create_model(MODEL_NAME, pretrained=True)
model.eval()

qkv_weight = model.blocks[1].attn.qkv.weight.data.clone()
Q_full = qkv_weight[:EMBED_DIM, :]  # (768, 768)
K_full = qkv_weight[EMBED_DIM : 2 * EMBED_DIM, :]  # (768, 768)

head_idx = 0
Q_weights = Q_full[head_idx * HEAD_DIM : (head_idx + 1) * HEAD_DIM, :]  # (64, 768)
K_weights = K_full[head_idx * HEAD_DIM : (head_idx + 1) * HEAD_DIM, :]  # (64, 768)

x = torch.randn(EMBED_DIM, SEQ_LEN) * 0.1  # Scale similar to normalized embeddings


def reduce(tensor, dim, keepdim=False):
    """Apply checksum reduction (mean or sum) based on CHECKSUM_METHOD."""
    if CHECKSUM_METHOD == "sum":
        return tensor.sum(dim=dim, keepdim=keepdim)
    else:
        return tensor.mean(dim=dim, keepdim=keepdim)


def flip_bit(
    weight: torch.Tensor, row=None, col=None, bit=None, value=None
) -> torch.Tensor:
    """Inject a fault into a weight tensor.

    If value is set, directly assigns that value. Otherwise flips the specified bit.
    """
    tensor = weight.clone()
    if row is None:
        row = 1
    if col is None:
        col = 1
    if bit is None:
        bit = 31

    original = tensor[row, col].item()

    if value is not None:
        tensor[row, col] = value
        print(f"Fault at [{row}, {col}]: {original:.6f} -> {value}")
    else:
        tensor_int = tensor.view(torch.int32)
        tensor_int[row, col] ^= 1 << bit
        faulty = tensor.view(torch.float32)[row, col].item()
        print(f"Bit-flip at [{row}, {col}], bit {bit}: {original:.6f} -> {faulty:.6e}")

    return tensor


# ---- Q weights ----
Q_check = reduce(Q_weights, dim=0, keepdim=True)
if fault_target == "Q":
    Q_weights = flip_bit(Q_weights, row=3, col=3, bit=FAULT_BIT, value=FAULT_VALUE)

Q_weights_faulty = torch.cat([Q_weights, Q_check], dim=0)

Q = Q_weights_faulty @ x


# ---- K weights ----
K_check = reduce(K_weights, dim=0, keepdim=True)

if fault_target == "K":
    K_weights = flip_bit(K_weights, row=32, col=384, bit=FAULT_BIT, value=FAULT_VALUE)

K_weights_check = torch.cat([K_weights, K_check], dim=0)

K = K_weights_check @ x


# ---- Attention core ----
QK = Q @ K.T

# Scale by 1/sqrt(head_dim) like real attention
scale = HEAD_DIM**-0.5
QK = QK * scale

QK_inner = QK[:-1, :-1]


row_sums = reduce(QK_inner, dim=1, keepdim=True)
col_sums = reduce(QK_inner, dim=0, keepdim=True)

row_check = QK[:-1, -1]
col_check = QK[-1, :-1]

col_compare = torch.cat([row_sums, row_check.view(-1, 1)], dim=1)
row_compare = torch.cat([col_sums.T, col_check.view(-1, 1)], dim=1)


QK_softmax = F.softmax(QK_inner, dim=1)
QK_softmax_col_sums = reduce(QK_softmax, dim=0)


row_check_softmax = F.softmax(row_check, dim=0)
row_sums_softmax = F.softmax(row_sums, dim=1)

col_check_softmax = F.softmax(col_check, dim=0)
col_sums_softmax = F.softmax(col_sums, dim=1)

row_check_softmax = F.softmax(row_check, dim=0)
row_sums_softmax = F.softmax(row_sums, dim=0)


print("Input x:\n", x)

print("Q:\n", Q)
print("K:\n", K)
print("\nQK = Q @ K.T:\n", QK)

print("\nRow sums vs checksum row (side by side):")
print(row_compare)

print("\nColumn sums vs checksum column (side by side):")
print(col_compare)

print("\nSoftmaxed QK")
print(QK_softmax)


print("\n \033[31mSoftmax mean \033[0m\n", QK_softmax_col_sums)
print("\ncol mean / col check softmax \n", col_sums_softmax, "\n", col_check_softmax)

print(
    "\nrowmean / rowcheck softmax\n",
    row_sums_softmax,
    "\n",
    row_check_softmax.view(-1, 1),
)
