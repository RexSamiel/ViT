import torch

X = torch.ones(2, 3, 3)
W = torch.ones(2, 3, 3)

W[1, 1, 1] = 30.0

Y = torch.bmm(X, W)

print("Input X:")
print(X)

print("\nWeight W:")
print(W)

print("\nOutput Y = X @ W:")
print(Y)

Y2 = torch.bmm(Y, W)
print("\nY2 = Y @ W:")
print(Y2)
