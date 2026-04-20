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


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        self.in_project_x     = nn.Linear(dim, inner_dim)
        self.in_project_fx    = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q   = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k   = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v   = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)
        x_mid  = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3)

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm    = slice_weights.sum(2)                                             # B H G
        slice_token   = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token   = slice_token / (slice_norm[..., None] + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


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


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, slice_num, dropout=0.):
        super().__init__()
        self.layers_x = nn.ModuleList()
        cond_dim = dim // 4

        for _ in range(depth):
            self.layers_x.append(nn.ModuleList([
                CrossAttention(dim, heads=heads // 2, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                Physics_Attention_Irregular_Mesh(dim, heads=heads, dim_head=dim_head,
                                                 dropout=dropout, slice_num=slice_num),
                FeedForward(dim, mlp_dim, dropout=dropout),
                nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True)),  # adaLN cross
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True)),  # adaLN self
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))

        # Zero-init adaLN output projections
        for i in range(depth):
            nn.init.zeros_(self.layers_x[i][4][1].weight)
            nn.init.zeros_(self.layers_x[i][4][1].bias)
            nn.init.zeros_(self.layers_x[i][7][1].weight)
            nn.init.zeros_(self.layers_x[i][7][1].bias)

    def forward(self, x, mu, s):
        mu_bc = mu.unsqueeze(1) if mu.dim() == 2 else mu  # (B, 1, cond_dim)

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


class IrradianceModel(nn.Module):
    """
    Transolver-based flow-matching model for irradiance field reconstruction.

    Input per point: pos (2) + irradiance_t (1) + ref-grid distances (ref²)
    Sensor context:  pos (2) + irradiance_observed (1)
    Output:          velocity field (1) — predicts (X1 - X0)
    """

    def __init__(
        self,
        space_dim:  int = 2,
        fun_dim:    int = 1,
        out_dim:    int = 1,
        n_layers:   int = 12,
        n_hidden:   int = 374,
        n_head:     int = 8,
        slice_num:  int = 32,
        ref:        int = 4,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.__name__ = 'IrradiancePhySense'
        self.ref       = ref
        self.space_dim = space_dim
        self.fun_dim   = fun_dim
        self.n_hidden  = n_hidden
        cond_dim       = n_hidden // 4

        in_dim     = space_dim + fun_dim + ref * ref       # 2 + 1 + 16 = 19
        sensor_in  = space_dim + fun_dim                   # 2 + 1 = 3

        self.preprocess = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, n_hidden),
            nn.LayerNorm(n_hidden),
        )

        self.sensor_encoder   = nn.Sequential(nn.Linear(sensor_in, n_hidden),  nn.LayerNorm(n_hidden))
        self.sensor_encoder_2 = nn.Sequential(nn.Linear(sensor_in, cond_dim),  nn.LayerNorm(cond_dim))

        self.t_embedder = TimestepEmbedder(cond_dim, frequency_embedding_size=cond_dim)

        # small MLP to embed scalar mean sun height into cond_dim
        self.sun_embedder = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.transformer = Transformer(n_hidden, n_layers, n_head, n_head, n_hidden, slice_num, dropout)
        self.mlp_head    = FinalLayer(n_hidden, out_channels=out_dim)
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))


    def _ref_grid_distances(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute distances from each point to a ref×ref reference grid."""
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        B = pos.shape[0]
        device = pos.device
        gc = torch.linspace(0, 1, self.ref, device=device)
        gx, gy = torch.meshgrid(gc, gc, indexing='ij')
        grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2)   # (1, ref^2, 2)
        grid = grid.expand(B, -1, -1)                             # (B, ref^2, 2)
        diff = pos.unsqueeze(2) - grid.unsqueeze(1)               # (B, N, ref^2, 2)
        return (diff ** 2).sum(-1).sqrt() + 1e-8                  # (B, N, ref^2)


    def forward(self, batch: dict) -> torch.Tensor:
        pos   = batch['pos']    # (B, N, 2)
        y     = batch['y']      # (B, N, 1)
        h_sun = batch['h_sun']  # (B, N, 1)

        B, N, _ = y.shape
        device  = y.device

        # Flow interpolation on irradiance only
        u      = torch.randn(B, device=device)
        t      = torch.sigmoid(u)
        t_bc   = t.view(B, 1, 1)
        noise  = torch.randn_like(y)
        y_t    = t_bc * y + (1. - t_bc) * noise
        target = y - noise                                  # (B, N, 1)

        if pos.dim() == 2:
            pos = pos.unsqueeze(0).expand(B, -1, -1)

        ref_dists = self._ref_grid_distances(pos)           # (B, N, 16)

        # Field tokens: 19 dims, no h_sun here
        x  = torch.cat([pos, y_t, ref_dists], dim=-1)      # (B, N, 19)
        fx = self.preprocess(x) + self.placeholder[None, None, :]

        # Sensor features: 3 dims, no h_sun here
        n_sensors   = random.randint(10, min(200, N))
        idx         = torch.randperm(N, device=device)[:n_sensors]
        s_pos       = pos[:, idx, :]
        s_y         = y[:, idx, :]
        sensor_feat = torch.cat([s_pos, s_y], dim=-1)      # (B, S, 3)
        s    = self.sensor_encoder(sensor_feat)
        s2   = self.sensor_encoder_2(sensor_feat)

        # Sun height as global conditioning signal
        # Mean over panels: nearly constant across N but varies across timesteps
        h_sun_mean = h_sun.mean(dim=1)                     # (B, 1)
        h_sun_emb  = self.sun_embedder(h_sun_mean)         # (B, cond_dim)

        t_emb = self.t_embedder(t) + s2.mean(dim=1) + h_sun_emb   # (B, cond_dim)

        x_out = self.transformer(fx, t_emb, s)
        pred  = self.mlp_head(x_out, t_emb)                # (B, N, 1)

        return F.mse_loss(pred, target)


    @torch.no_grad()
    def sample(
        self,
        pos:            torch.Tensor,    # (N, 2)
        y_full:         torch.Tensor,    # (N, 1)
        h_sun_full:     torch.Tensor,    # (N, 1)
        sensor_indices: torch.Tensor,    # (S,)
        n_steps:        int = 5,
        n_samples:      int = 1,
    ):
        device = pos.device
        pos_bc = pos.unsqueeze(0)                                    # (1, N, 2)
        ref_d  = self._ref_grid_distances(pos_bc)                    # (1, N, 16)

        # Sensor features: irradiance only
        s_pos = pos[sensor_indices].unsqueeze(0)                     # (1, S, 2)
        s_y   = y_full[sensor_indices].unsqueeze(0)                  # (1, S, 1)
        sensor_feat = torch.cat([s_pos, s_y], dim=-1)                # (1, S, 3)
        s    = self.sensor_encoder(sensor_feat)
        s2   = self.sensor_encoder_2(sensor_feat)

        # Global sun height conditioning — computed once, reused every ODE step
        h_sun_mean = h_sun_full.mean(dim=0, keepdim=True)            # (1, 1)
        h_sun_emb  = self.sun_embedder(h_sun_mean.unsqueeze(0))      # (1, cond_dim)

        pred_acc = torch.zeros(len(pos), 1, device=device)

        for _ in range(n_samples):
            z  = torch.randn(1, len(pos), 1, device=device)
            dt = 1.0 / n_steps

            for i in range(n_steps):
                t_val = torch.tensor([i / n_steps], device=device, dtype=torch.float32)
                x     = torch.cat([pos_bc, z, ref_d], dim=-1)        # (1, N, 19)
                fx    = self.preprocess(x) + self.placeholder[None, None, :]
                t_emb = self.t_embedder(t_val) + s2.mean(dim=1) + h_sun_emb.squeeze(0)
                x_out = self.transformer(fx, t_emb, s)
                vel   = self.mlp_head(x_out, t_emb)
                z     = z + vel * dt

            pred_acc += z.squeeze(0)

        pred = (pred_acc / n_samples).squeeze(-1)
        pred = pred.clamp(min=0.0)

        rel_loss = (
            torch.norm(y_full.squeeze(-1) - pred) /
            (torch.norm(y_full) + 1e-8)
        ).item()

        return pred, rel_loss