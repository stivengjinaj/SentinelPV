import torch
import numpy as np
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.utils.checkpoint as checkpoint
import math

import torch.nn.functional as F

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

# classes

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
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Vis_Embedder(nn.Module):
    def __init__(self, in_dim,  hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.emb = nn.Linear(in_dim, hidden_size)
        self.mlp = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = self.emb(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, s):
        
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        
        kv = self.to_kv(s).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
    
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)    


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers_x = nn.ModuleList([])
        self.layers_c = nn.ModuleList([])
        for _ in range(depth):
            self.layers_x.append(nn.ModuleList([
                CrossAttention(dim, heads = heads // 2, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
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
    def __init__(self, hidden_size, patch_height, patch_width, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, patch_height * patch_width * out_channels, bias=True),
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
    def __init__(self, dim=374, depth=8, heads=8, mlp_dim=374, dim_head = 32, ref = 4, dropout = 0., emb_dropout = 0.,args=None):
        super().__init__()

        image_height, image_width = args.model.image_size
        patch_height, patch_width = args.model.patch_size
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.out_channels = args.model.out_channels
        self.patch_num_height = image_height // patch_height
        self.patch_num_width = image_width // patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        in_channels = args.model.in_channels
        self.ref = ref
        self.pos = self.get_grid([image_height, image_width], 'cuda')
        patch_dim = (in_channels + self.ref*self.ref) * self.patch_height * self.patch_width

        self.t_embedder = TimestepEmbedder(dim // 4, frequency_embedding_size=dim // 4)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.sensor_encoder = nn.Sequential(
            nn.Linear(1 + self.ref*self.ref, dim),
            nn.LayerNorm(dim),
        )
    
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = FinalLayer(mlp_dim, patch_height=patch_height, patch_width=patch_width, out_channels=self.out_channels)

    def forward(self, x, t, row_indices, col_indices, sparse_sensor_field):
        
        # sparse_sensor_index: (M, 2) sparse_sensor_field: (M, 1)

        B, H, W, C = x.shape
        grid = self.pos.repeat(B,1,1,1)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)  

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        x = self.dropout(x)

        t = self.t_embedder(t)[:,None,:]
        t = t.repeat(1, n, 1)
        
        
        sensor_pos = grid[:,row_indices, col_indices, :]
        
        sensor_feature = torch.concat([sensor_pos, sparse_sensor_field], dim=-1)
        
        s = self.sensor_encoder(sensor_feature)
        
        x = self.transformer(x, t, s)
        b,l,ch = x.shape
        x = self.mlp_head(x, t).reshape(b, self.patch_num_height, self.patch_num_width, self.patch_height, self.patch_width, self.out_channels).permute(0,1,3,2,4,5).contiguous()
        x = x.reshape(b, self.patch_num_height * self.patch_height, self.patch_num_width * self.patch_width, self.out_channels)
        return x

    def get_grid(self, shape, device):
        size_x, size_y = shape[0], shape[1]
        batchsize = 1
        gridx = torch.tensor(np.linspace(0, 2.7, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 2


        gridx = torch.tensor(np.linspace(0, 2.7, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos


if __name__ == '__main__':
    x = torch.randn([8,64,64,1]).cuda()
    mu = torch.randn([1]).cuda()
    mu = mu[0]
    f = torch.randn([64,64]).cuda()
    net = Model().cuda()
    print(net(x, f).shape)