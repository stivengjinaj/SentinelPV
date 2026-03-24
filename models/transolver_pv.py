import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from helpers.helpers import modulate


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., -1:])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class SelfAttention(nn.Module):
    """
    Standard multi-head self-attention for a regular token sequence.
    Replaces Physics_Attention_Irregular_Mesh.
    Input/output: (B, N_tokens, dim)
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim    = dim_head * heads
        self.heads   = heads
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out  = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # x : (B, N_tokens, dim)
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)       # three tensors of (B, N, inner_dim)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.1 if self.training else 0.0
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross-attention: field tokens (queries) <- sensor observations (keys/values)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim    = dim_head * heads
        project_out  = not (heads == 1 and dim_head == dim)
        self.heads   = heads
        self.to_q    = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv   = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out  = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, s):
        q       = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k, v    = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      self.to_kv(s).chunk(2, dim=-1))
        out     = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

def make_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """
    Build a fixed 2D sinusoidal positional embedding.
    Works for any embed_dim by padding to the nearest multiple of 4
    then slicing back to embed_dim.
    Returns: (1, grid_h * grid_w, embed_dim)
    """
    # Pad up to nearest multiple of 4 if needed
    pad      = (4 - embed_dim % 4) % 4   # 0 if already divisible
    dim_work = embed_dim + pad            # 374 -> 376

    half = dim_work // 2                 # 188

    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)

    freq = torch.arange(half // 2, dtype=torch.float32)
    freq = 1.0 / (10000 ** (freq / (half // 2)))

    row_enc = torch.outer(rows, freq)    # (H, half//2)
    col_enc = torch.outer(cols, freq)    # (W, half//2)

    row_embed = torch.cat([torch.sin(row_enc), torch.cos(row_enc)], dim=-1)  # (H, half)
    col_embed = torch.cat([torch.sin(col_enc), torch.cos(col_enc)], dim=-1)  # (W, half)

    row_embed = row_embed.unsqueeze(1).expand(-1, grid_w, -1)  # (H, W, half)
    col_embed = col_embed.unsqueeze(0).expand(grid_h, -1, -1)  # (H, W, half)

    pos_embed = torch.cat([row_embed, col_embed], dim=-1)      # (H, W, dim_work)
    pos_embed = pos_embed.reshape(1, grid_h * grid_w, dim_work)

    # Slice back to the requested embed_dim
    return pos_embed[:, :, :embed_dim]                         # (1, H*W, embed_dim)

class TransformerGrid(nn.Module):
    """
    Identical structure to Transformer but uses SelfAttention
    instead of Physics_Attention_Irregular_Mesh.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers_x = nn.ModuleList()
        cond_dim = dim // 4

        for _ in range(depth):
            self.layers_x.append(nn.ModuleList([
                CrossAttention(dim, heads=heads // 2, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True)),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True)),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))

        for i in range(depth):
            nn.init.zeros_(self.layers_x[i][4][1].weight)
            nn.init.zeros_(self.layers_x[i][4][1].bias)
            nn.init.zeros_(self.layers_x[i][7][1].weight)
            nn.init.zeros_(self.layers_x[i][7][1].bias)

    def forward(self, x, mu, s):
        # x : (B, N_tokens, dim)   mu : (B, cond_dim)   s : (B, S, dim)
        mu_bc = mu.unsqueeze(1) if mu.dim() == 2 else mu

        for (cosattn, ff1, attn, ff2,
             adaLN1, norm1, norm2,
             adaLN2, norm3, norm4) in self.layers_x:

            shift1, scale1, gate1, shift2, scale2, gate2 = adaLN1(mu_bc).chunk(6, dim=-1)
            x = x + gate1 * cosattn(modulate(norm1(x), shift1, scale1), s)
            x = x + gate2 * ff1(modulate(norm2(x), shift2, scale2))

            shift3, scale3, gate3, shift4, scale4, gate4 = adaLN2(mu_bc).chunk(6, dim=-1)
            x = x + gate3 * attn(modulate(norm3(x), shift3, scale3))
            x = x + gate4 * ff2(modulate(norm4(x), shift4, scale4))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        cond_dim = hidden_size // 4
        self.norm_final       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp              = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, out_channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_size, bias=True))

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.mlp[2].weight, 0)
        nn.init.constant_(self.mlp[2].bias,   0)

    def forward(self, x, mu):
        mu_bc        = mu.unsqueeze(1) if mu.dim() == 2 else mu
        shift, scale = self.adaLN_modulation(mu_bc).chunk(2, dim=-1)
        x            = modulate(self.norm_final(x), shift, scale)
        return self.mlp(x)


class IrradianceModelGrid(nn.Module):
    """
    DiT-style flow-matching model for irradiance on a regular grid.

    The H×W grid is flattened to H*W tokens. Each token is one grid cell.
    Cells with no panel assigned carry irradiance=0 and mask=0.

    Input per token : noisy irradiance (1) + mask (1) = 2 channels
                      + 2D sincos positional embedding (embed_dim)
    Sensor context  : same as irregular model — pos (2) + irradiance (1) = 3
    Output          : velocity field (1) per token
    """

    def __init__(
        self,
        grid_h:    int   = 34,
        grid_w:    int   = 34,
        in_chans:  int   = 2,      # noisy irradiance + mask
        out_dim:   int   = 1,
        n_layers:  int   = 12,
        n_hidden:  int   = 374,
        n_head:    int   = 8,
        dim_head:  int   = 64,
        space_dim: int   = 2,
        fun_dim:   int   = 1,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.__name__ = 'IrradiancePhySenseGrid'

        self.grid_h   = grid_h
        self.grid_w   = grid_w
        self.n_tokens = grid_h * grid_w   # 34*34 = 1156
        cond_dim      = n_hidden // 4

        # ── Token input projection ────────────────────────────────────────────
        # Each grid cell has in_chans=2 values (noisy irradiance + mask).
        # Project to n_hidden.
        self.token_proj = nn.Sequential(
            nn.LayerNorm(in_chans),
            nn.Linear(in_chans, n_hidden),
            nn.LayerNorm(n_hidden),
        )

        # ── 2D positional embedding (fixed, not learnable) ────────────────────
        # Registered as a buffer so it moves to GPU with .to(device) but is
        # not treated as a trainable parameter.
        pos_embed = make_2d_sincos_pos_embed(n_hidden, grid_h, grid_w)
        self.register_buffer('pos_embed', pos_embed)   # (1, H*W, n_hidden)

        # ── Sensor encoder (unchanged from irregular model) ───────────────────
        sensor_in = space_dim + fun_dim                # 2 + 1 = 3
        self.sensor_encoder   = nn.Sequential(
            nn.Linear(sensor_in, n_hidden), nn.LayerNorm(n_hidden)
        )
        self.sensor_encoder_2 = nn.Sequential(
            nn.Linear(sensor_in, cond_dim), nn.LayerNorm(cond_dim)
        )

        # ── Timestep embedder (unchanged) ─────────────────────────────────────
        self.t_embedder = TimestepEmbedder(cond_dim, frequency_embedding_size=cond_dim)

        # ── Transformer (regular grid version) ───────────────────────────────
        self.transformer = TransformerGrid(
            n_hidden, n_layers, n_head, dim_head, n_hidden, dropout
        )

        # ── Output head (unchanged) ───────────────────────────────────────────
        self.mlp_head    = FinalLayer(n_hidden, out_channels=out_dim)
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))

        # ── Store grid index for loss masking ─────────────────────────────────
        self.space_dim = space_dim
        self.fun_dim   = fun_dim

    def forward(self, batch):
        """
        batch keys:
            grid   : (B, 1, H, W)  noisy irradiance grid
            mask   : (B, 1, H, W)  1 where panel exists
            y_flat : (B, N, 1)     true irradiance at each panel (for loss)
            pos    : (B, N, 2)     panel coordinates (for sensor sampling)
        """
        grid   = batch['grid']    # (B, 1, H, W)
        mask   = batch['mask']    # (B, 1, H, W)
        y_flat = batch['y_flat']  # (B, N, 1)
        pos    = batch['pos']     # (B, N, 2)

        B  = grid.shape[0]
        N  = y_flat.shape[1]
        device = grid.device

        # ── Flow interpolation on the grid ────────────────────────────────────
        u    = torch.randn(B, device=device)
        t    = torch.sigmoid(u)
        t_bc = t.view(B, 1, 1, 1)                 # broadcast over (1, H, W)

        noise  = torch.randn_like(grid)
        grid_t = t_bc * grid + (1. - t_bc) * noise  # (B, 1, H, W)
        # Target velocity — same formula as irregular model
        # We only care about loss at cells that have a real panel (mask=1)
        target_grid = grid - noise                   # (B, 1, H, W)

        # ── Build token sequence ──────────────────────────────────────────────
        # Flatten (B, 1, H, W) and (B, 1, H, W) to (B, H*W, 2)
        grid_flat = grid_t.permute(0, 2, 3, 1).reshape(B, self.n_tokens, 1)  # (B, T, 1)
        mask_flat = mask.permute(0, 2, 3, 1).reshape(B, self.n_tokens, 1)    # (B, T, 1)
        tokens    = torch.cat([grid_flat, mask_flat], dim=-1)                 # (B, T, 2)

        # Project to hidden dim and add positional embedding
        fx = self.token_proj(tokens)           # (B, T, n_hidden)
        fx = fx + self.pos_embed               # (B, T, n_hidden)  broadcast over B
        fx = fx + self.placeholder[None, None, :]

        # ── Random sensor subset (same as irregular model) ────────────────────
        n_sensors   = random.randint(10, min(200, N))
        idx         = torch.randperm(N, device=device)[:n_sensors]
        s_pos       = pos[:, idx, :]           # (B, S, 2)
        s_y         = y_flat[:, idx, :]        # (B, S, 1)  ground truth at sensors

        sensor_feat = torch.cat([s_pos, s_y], dim=-1)   # (B, S, 3)
        s    = self.sensor_encoder(sensor_feat)          # (B, S, n_hidden)
        s2   = self.sensor_encoder_2(sensor_feat)        # (B, S, cond_dim)

        t_emb = self.t_embedder(t) + s2.mean(dim=1)     # (B, cond_dim)

        # ── Transformer ───────────────────────────────────────────────────────
        x_out = self.transformer(fx, t_emb, s)           # (B, T, n_hidden)
        pred  = self.mlp_head(x_out, t_emb)              # (B, T, 1)

        # ── Loss: only on cells that have a real panel ─────────────────────────
        # Reshape target to (B, T, 1) and apply mask
        target_flat = target_grid.permute(0, 2, 3, 1).reshape(B, self.n_tokens, 1)
        mask_flat_  = mask_flat.detach()                 # (B, T, 1)

        # MSE only over real panel cells
        loss = (((pred - target_flat) ** 2) * mask_flat_).sum() / (mask_flat_.sum() + 1e-8)
        return loss