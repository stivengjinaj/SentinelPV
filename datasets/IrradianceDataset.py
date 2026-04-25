import torch
import numpy as np
from torch.utils.data import Dataset


class IrradianceDataset(Dataset):
    def __init__(
        self,
        irradiance_path: str,
        coords_path:     str,
        t_in:            int = 12,
        t_out:           int = 24,
    ):
        irradiance = np.load(irradiance_path).astype(np.float32)  # (T, N)
        coords     = np.load(coords_path).astype(np.float32)      # (N, 2)

        self.t_in  = t_in
        self.t_out = t_out

        coords_min  = coords.min(axis=0, keepdims=True)
        coords_max  = coords.max(axis=0, keepdims=True)
        self.pos_tensor = torch.tensor(
            (coords - coords_min) / (coords_max - coords_min + 1e-8),
            dtype=torch.float32,
        )

        irr = torch.tensor(irradiance, dtype=torch.float32)
        self.y_min = float(irr.min())
        self.y_max = float(irr.max())
        self.irr   = (irr - self.y_min) / (self.y_max - self.y_min + 1e-8)  # (T, N)

        # Valid starting indices. Need t_in history and t_out future
        self.valid_starts = list(range(t_in, len(self.irr) - t_out + 1))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        x_seq = self.irr[start - self.t_in : start]         # (T_in, N)
        y_seq = self.irr[start : start + self.t_out]        # (T_out, N)
        return {
            'pos':   self.pos_tensor,                        # (N, 2)
            'x_seq': x_seq.T,                               # (N, T_in)
            'y_seq': y_seq.T,                               # (N, T_out)
        }