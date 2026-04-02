import torch

x_math = torch.rand(4, 5)
x_torch = x_math.T
x_torch_2 = torch.rand(5, 4)
W = torch.rand(6, 4)
W_2 = torch.rand(6, 4)
math = W @ x_math
torch = x_torch @ W.T

print(torch)
