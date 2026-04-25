import torch.nn as nn

class TemporalEncoder(nn.Module):
    """
    Compress T_in historical irradiance values per panel into a single vector.
    Uses a small 1-D conv + pooling so it's permutation-invariant to nothing
    but respects temporal order.
    """
    def __init__(self, t_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x_seq):
        # x_seq: (B, N, T_in)
        return self.net(x_seq)