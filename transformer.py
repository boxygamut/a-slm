import torch
import torch.nn as nn
import math


class MLP(nn.Module): # Add gated projection and use SwiGLU
    def __init__(self, hidden_size):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        return self.net(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_kv_heads = 4, num_q_heads = 12, window_size = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.head_size = hidden_size // num_q_heads
        self.num_q_heads = num_q_heads

        self.window_size = window_size

        self.queries = nn.Linear(
            hidden_size,
            num_q_heads * self.head_size,
            bias = False
        )

        self.key = nn.Linear(
            hidden_size,
            num_kv_heads * self.head_size,
            bias = False
        )

        self.val = nn.Linear(
            hidden_size,
            num_kv_heads * self.head_size,
            bias = False
        )

        self.output_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias = False
        )

        self.rms_norm = nn.RMSNorm(hidden_size)

    def create_mask(self, T, device):
        positions = torch.arange(T, device = device)

        i = positions[:, None]
        j = positions[None, :]

        allowed = j <= i

        if self.window_size is not None:
            allowed = allowed & (j >= i - self.window_size + 1)

        mask = torch.zeros(T, T, device = device)
        mask = mask.masked_fill(~allowed, float("-inf"))

        return mask

    def forward(self, x):
        B, T, C = x.shape

        x = self.rms_norm(x)
        
        q = (
            self.queries(x)
            .reshape(B, T, self.num_q_heads, self.head_size) # [B, T, 12, 128] 1536 split into 12 and 128
            .permute(0, 2, 1, 3)
        )

        k = (
            self.key(x)
            .reshape(B, T, self.num_kv_heads, self.head_size) # [B, T, 4, 128]
            .permute(0, 2, 1, 3) # [B, 4, T, 128]
        )
    
        v = (
            self.val(x)
            .reshape(B, T, self.num_kv_heads, self.head_size)
            .permute(0, 2, 1, 3)
        )

        num_repeats = self.num_q_heads // self.num_kv_heads

        k = k.repeat_interleave(num_repeats, dim=1)
        v = v.repeat_interleave(num_repeats, dim=1) # [B, 12, T, 128] -> lines up with Q heads

        mask = self.create_mask(T, device = x.device)

        w = q @ k.transpose(-2, -1) # softmax(Q*K_transposed / sqrt(head_size))*val
        w = w / math.sqrt(self.head_size)
        w = w + mask
        out = torch.softmax(w, dim = -1) @ v

        out = (out
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B, T, self.hidden_size) # [B, 12, T, 128] -> [B, T, 12, 128] -> [B, T, 1536]
        )
        
        out = self.output_proj(out)
        return out