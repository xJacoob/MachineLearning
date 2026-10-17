import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('mask', (torch.tril(torch.ones(block_size, block_size))))

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        wei = (Q @ K.transpose(-2, -1)) * (K.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        V = self.value(x)
        out = wei @ V
        return out, wei


x = torch.randn(4, 8, 32)
head = Head(n_embd=32, head_size=16, block_size=8)
out, wei = head(x)
print(out.shape)  # torch.Size([4, 8, 16])
print(wei[0])  # nad przekątną same zera