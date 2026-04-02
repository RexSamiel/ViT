import torch

torch.manual_seed(0)

x = torch.rand(3, 4)

Wq = torch.rand(4, 4)
Wk = torch.rand(4, 4)
Wv = torch.rand(4, 4)

Q = x @ Wq.T
K = x @ Wk.T
V = x @ Wv.T

N = Q.shape[0]
d_v = V.shape[1]
online_output = torch.zeros(N, d_v)

for i in range(N):
    max_val = -float("inf")
    sum_exp = 0.0
    row_output = torch.zeros(d_v)

    for j in range(N):
        score = Q[i] @ K[j]
        if score > max_val:
            sum_exp = sum_exp * torch.exp(max_val - score) + 1.0
            max_val = score
        else:
            sum_exp += torch.exp(score - max_val)
        weight = torch.exp(score - max_val) / sum_exp
        row_output += weight * V[j]

    online_output[i] = row_output

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)
print("\nOnline Attention Output:\n", online_output)
