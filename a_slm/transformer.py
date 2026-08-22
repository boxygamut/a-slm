import torch
import torch.nn as nn
import math


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias = False
        )

        self.down_proj = nn.Linear(
            intermediate_size, 
            hidden_size,
            bias = False
        )

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias = False
        )

        self.activation = nn.SiLU()

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)

        swish_gate = self.activation(gate) * up
        
        return self.down_proj(swish_gate)

def get_rope_frequencies(head_size, theta, device = None):
    pair_indices = torch.arange(
        head_size // 2,
        device = device,
        dtype = torch.float32
    )

    return theta ** (-2 * pair_indices / head_size)

def apply_rope(x, theta = 10000.0):
    B, H, T, D = x.shape # shape post permute in the GQA section

    freqs = get_rope_frequencies(
        D,
        theta,
        device = x.device
    )

    positions = torch.arange(
        T,
        device = x.device,
        dtype = torch.float32
    )

    angles = positions[:, None] * freqs[None, :] # Column positions * row freqs

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_pairs = x.reshape(B, H, T, D // 2, 2)
    
    x_even = x_pairs[..., 0] # [B, H, T, D / 2]
    x_odd = x_pairs[..., 1] # Same as above

    cos = cos[None, None, ...] # [T, D / 2] -> [1, 1, T, D / 2]
    sin = sin[None, None, ...]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack((rotated_even, rotated_odd), dim = -1).reshape(B, H, T, D)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_kv_heads = 4, num_q_heads = 12, window_size = None, rope_theta = 10000.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.head_size = hidden_size // num_q_heads
        self.num_q_heads = num_q_heads

        self.window_size = window_size
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(
            hidden_size,
            num_q_heads * self.head_size,
            bias = False
        )

        self.k_proj = nn.Linear(
            hidden_size,
            num_kv_heads * self.head_size,
            bias = False
        )

        self.v_proj = nn.Linear(
            hidden_size,
            num_kv_heads * self.head_size,
            bias = False
        )

        self.output_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias = False
        )

        self.q_norm = nn.RMSNorm(self.head_size)
        self.k_norm = nn.RMSNorm(self.head_size)

    def create_mask(self, T, device):
        positions = torch.arange(
            T, 
            device = device,
            dtype = torch.float32
        )

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
        
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_q_heads, self.head_size) # [B, T, 12, 128] 1536 split into 12 and 128
            .permute(0, 2, 1, 3)
        )

        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_kv_heads, self.head_size) # [B, T, 4, 128]
            .permute(0, 2, 1, 3) # [B, 4, T, 128]
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, self.rope_theta)
        k = apply_rope(k, self.rope_theta)
    
        v = (
            self.v_proj(x)
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

class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_q_heads, num_kv_heads, window_size = None, rope_theta = 10000.0):
        super().__init__()
        
        self.attention_block = GroupedQueryAttention(
            hidden_size = hidden_size,
            num_q_heads = num_q_heads,
            num_kv_heads = num_kv_heads,
            window_size = window_size,
            rope_theta = rope_theta
        )

        self.attention_norm = nn.RMSNorm(hidden_size) # could have been put inside GQA, norms before attention is called
        self.mlp_norm = nn.RMSNorm(hidden_size)

        self.mlp = SwiGLU(
            hidden_size = hidden_size,
            intermediate_size = intermediate_size
        )

    def forward(self, x):
        x = x + self.attention_block(
            self.attention_norm(x)
        )

        x = x + self.mlp(
            self.mlp_norm(x) # double residual connections with norm
        )

        return x

class Transformer(nn.Module):
    def __init__(self, num_l4g_blocks, hidden_size, intermediate_size, num_q_heads, num_kv_heads, window_size, vocab_size, rope_theta=10000.0):
        super().__init__()
        
        self.d_blocks = nn.ModuleList()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        for i in range(num_l4g_blocks * 5):

            loop_window_size = window_size
            
            if (i + 1) % 5 == 0: loop_window_size = None
            
            self.d_blocks.append(DecoderBlock(
                hidden_size = hidden_size,
                intermediate_size = intermediate_size,
                num_q_heads = num_q_heads,
                num_kv_heads = num_kv_heads,
                window_size = loop_window_size,
                rope_theta = rope_theta
            ))

        self.final_norm = nn.RMSNorm(hidden_size)

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias = False
        )

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x):
        x = self.token_embedding(x)
        
        for block in self.d_blocks:
            x = block(x)

        x = self.final_norm(x)

        return self.lm_head(x)