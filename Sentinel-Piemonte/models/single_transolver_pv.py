import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange, repeat
import math
import random
import torch.nn.functional as F
from typing import Optional


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., -1:])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Physics_Attention_Irregular_Mesh(nn.Module):
    """
    Physics-aware attention for irregular points (scattered PV panels).
    Uses slicing mechanism to efficiently handle unstructured spatial data.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C - B: batch, N: number of panels, C: channels
        B, N, C = x.shape

        ### (1) Slice: Cluster points into slice_num groups
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token, dropout_p=0.1 if self.training else 0.0)

        ### (3) Deslice: Reconstruct full resolution
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross-attention between field tokens and sensor observations."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, s):
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        kv = self.to_kv(s).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, slice_num, dropout=0.):
        super().__init__()
        self.layers_x = nn.ModuleList([])
        for _ in range(depth):
            self.layers_x.append(nn.ModuleList([
                CrossAttention(dim, heads=heads // 2, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                Physics_Attention_Irregular_Mesh(dim, heads=heads, dim_head=dim_head,
                                                     dropout=dropout, slice_num=slice_num),
                FeedForward(dim, mlp_dim, dropout=dropout),
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim//4, 6 * dim, bias=True)
                ),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim//4, 6 * dim, bias=True)
                ),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))

        for i in range(depth):
            nn.init.zeros_(self.layers_x[i][4][1].weight)
            nn.init.zeros_(self.layers_x[i][4][1].bias)
            nn.init.zeros_(self.layers_x[i][7][1].weight)
            nn.init.zeros_(self.layers_x[i][7][1].bias)

    def forward(self, x, mu, s):
        for cosattn, ff1, attn, ff2, adaLN_modulation_mu1, norm1, norm2, adaLN_modulation_mu2, norm3, norm4 in self.layers_x:

            shift_msa_mu_c, scale_msa_mu_c, gate_msa_mu_c, shift_mlp_mu_c, scale_mlp_mu_c, gate_mlp_mu_c = adaLN_modulation_mu1(mu).chunk(6, dim=-1)
            x_cosattn = cosattn(modulate(norm1(x), shift_msa_mu_c, scale_msa_mu_c), s)
            x = x + gate_msa_mu_c * x_cosattn
            x = x + gate_mlp_mu_c * ff1(modulate(norm2(x), shift_mlp_mu_c, scale_mlp_mu_c))

            shift_msa_mu, scale_msa_mu, gate_msa_mu, shift_mlp_mu, scale_mlp_mu, gate_mlp_mu = adaLN_modulation_mu2(mu).chunk(6, dim=-1)
            x_attn = attn(modulate(norm3(x), shift_msa_mu, scale_msa_mu))
            x = x + gate_msa_mu * x_attn
            x = x + gate_mlp_mu * ff2(modulate(norm4(x), shift_mlp_mu, scale_mlp_mu))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_channels, bias=True),
        )
        self.adaLN_modulation_mu = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size//4, 2 * hidden_size, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation_mu[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_mu[-1].bias, 0)
        nn.init.constant_(self.mlp[2].weight, 0)
        nn.init.constant_(self.mlp[2].bias, 0)

    def forward(self, x, mu):
        shift_mu, scale_mu = self.adaLN_modulation_mu(mu).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift_mu, scale_mu)
        x = self.mlp(x)
        return x


class Model(nn.Module):
    """
    Transolver backbone adapted for Photovoltaic (PV) panels.

    Handles scattered PV panels across a region with irregular spatial distribution.
    Uses Physics-aware attention (Physics_Attention_Irregular_Mesh) for efficient
    processing of unstructured point cloud data.

    Input channels: 2 (e.g., irradiance + temperature, or any two measured quantities)
    """
    def __init__(self,
                 space_dim=2,
                 n_layers=12,
                 n_hidden=374,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=2,
                 out_dim=1,
                 slice_num=32,
                 ref=4,
                 unified_pos=True,
                 args=None
                 ):
        super(Model, self).__init__()
        self.__name__ = 'Transolver_PV'
        self.ref = ref
        self.unified_pos = unified_pos
        self.space_dim = space_dim
        self.fun_dim = fun_dim
        self.n_hidden = n_hidden

        # Input: pos (2D: lat, lon) + 2 channels
        # + reference grid distances
        if self.unified_pos:
            self.preprocess = nn.Sequential(
                nn.LayerNorm(self.space_dim + self.fun_dim + self.ref * self.ref),
                nn.Linear(self.space_dim + self.fun_dim + self.ref * self.ref, n_hidden),
                nn.LayerNorm(n_hidden),
            )

        # Sensor encoder: position (2D) + 2 input channels
        self.sensor_encoder = nn.Sequential(
            nn.Linear(self.space_dim + self.fun_dim, n_hidden),
            nn.LayerNorm(n_hidden),
        )

        self.sensor_encoder_2 = nn.Sequential(
            nn.Linear(self.space_dim + self.fun_dim, n_hidden // 4),
            nn.LayerNorm(n_hidden // 4),
        )

        self.t_embedder = TimestepEmbedder(n_hidden // 4, frequency_embedding_size=n_hidden // 4)

        self.transformer = Transformer(n_hidden, n_layers, n_head, n_head, n_hidden, slice_num, dropout)
        self.mlp_head = FinalLayer(n_hidden, out_channels=out_dim)
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))


    def get_grid(self, my_pos):
        """
        Compute distance-based reference grid for 2D positions (lat, lon).

        Args:
            my_pos: (B, N, 2) - positions of PV panels (latitude, longitude)

        Returns:
            pos: (B, N, ref*ref) - distances to reference grid points
        """
        batchsize = my_pos.shape[0]
        device = my_pos.device

        # Create 2D reference grid in lat-lon space
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref * self.ref, 2)

        # Compute distances from panels to reference grid points
        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)
        ).reshape(batchsize, my_pos.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):
        pos = data.pos  # (N, 2)
        y = data.y      # (N, 1) - PV Power Output

        num_points = y.size(0)
        num_samples = random.randint(10, 200)
        random_indices = torch.randperm(num_points)[:num_samples]
        sampled_y = y[random_indices]

        device = pos.device
        noise = torch.randn_like(y)

        u = torch.normal(mean=0.0, std=1.0, size=(1,)).to(device)
        t = torch.sigmoid(u)
        t_tmp = t.unsqueeze(-1).repeat(y.shape[0], 1)
        
        # Path from noise to data
        y_t = t_tmp * y + (1. - t_tmp) * noise
        target = y - noise

        x = torch.concat((pos, y_t), dim=-1).unsqueeze(0)  # Shape: (1, N, 3)

        if self.unified_pos:
            new_pos = self.get_grid(pos.unsqueeze(0))
            x = torch.cat((x, new_pos), dim=-1)

        fx = self.preprocess(x) + self.placeholder[None, None, :]
        t_emb = self.t_embedder(t).squeeze()

        # Cross-Attention
        sensor_feature = torch.concat((pos[random_indices], sampled_y), dim=-1).unsqueeze(0)
        s = self.sensor_encoder(sensor_feature)
        s_2 = self.sensor_encoder_2(sensor_feature)
        t_emb = t_emb + s_2.mean(dim=1).squeeze()

        x_out = self.transformer(fx, t_emb, s)
        out = self.mlp_head(x_out, t_emb)[0]

        loss = nn.MSELoss(reduction='none')(out, target).mean(dim=0)

        return loss

    def sample(self, data, return_pred=False, seed=1, sensor_number=15):
        """
        Inference: reconstruct field from sparse sensor observations.

        Args:
            data: Object with attributes:
                - pos: (N, 2) - PV panel positions
                - y: (N, out_dim) - observed field values at sensor locations
            sensor_number: Number of sensors to use

        Returns:
            relative_loss: Reconstruction error metric
            pred: (optional) Reconstructed field
        """

        pos = data.pos
        y = data.y

        num_points = y.size(0)
        num_samples = sensor_number

        torch.manual_seed(seed)
        random_indices = torch.randperm(num_points)[:num_samples]
        sampled_y = y[random_indices] * 1
        xyz = pos[random_indices]

        device = pos.device

        freq = 1
        pred = 0

        for _ in range(freq):
            z = torch.randn_like(y)

            N = 5  # Number of diffusion steps
            dt = (1. / N)

            for i in range(N):
                x = torch.concat((pos, z), dim=-1).unsqueeze(0)  # (1, N, space_dim + fun_dim)
                t = (torch.ones((1)) * i / N).to(device).unsqueeze(-1).repeat(y.shape[0], 1)

                if self.unified_pos:
                    new_pos = self.get_grid(pos.unsqueeze(0))
                    x = torch.cat((x, new_pos), dim=-1)

                fx = self.preprocess(x)
                fx = fx + self.placeholder[None, None, :]

                t_emb = self.t_embedder(t).squeeze()

                sensor_feature = torch.concat((xyz, sampled_y), dim=-1).unsqueeze(0)
                s_2 = self.sensor_encoder_2(sensor_feature)
                t_emb = t_emb + s_2.mean(dim=1).squeeze()

                s = self.sensor_encoder(sensor_feature)

                x = self.transformer(fx, t_emb, s)
                out = self.mlp_head(x, t_emb)[0]

                z = z + out * dt

            pred += z

        pred /= freq
        relative_loss = torch.norm(y - pred, 2, 0) / torch.norm(y, 2, 0)

        if return_pred:
            return relative_loss, pred, xyz

        return relative_loss
