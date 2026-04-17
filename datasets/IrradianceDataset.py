import torch
import numpy as np
from torch.utils.data import Dataset


class IrradianceDataset(Dataset):
    """
    Dataset for irradiance-only PhySense training.

    Expects:
        irradiance_path : .npy of shape (T, N) — one scalar per panel per timestep
        coords_path     : .npy of shape (N, 2) — (lat, lon) for each panel

    Returns per sample:
        pos : (N, 2)  float32 — panel coordinates (same for every sample)
        y   : (N, 1)  float32 — irradiance at each panel at time t
    """

    def __init__(self, irradiance_path: str, sun_height_path: str, coords_path: str):
        irradiance = np.load(irradiance_path)  # (T, N)
        coords     = np.load(coords_path) # (N, 2)
        h_sun      = np.load(sun_height_path) # (T, N)

        # Normalise coordinates to [0, 1] so the reference grid makes sense
        coords_min = coords.min(axis=0, keepdims=True)
        coords_max = coords.max(axis=0, keepdims=True)
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)

        self.pos_tensor = torch.tensor(coords_norm, dtype=torch.float32)   # (N, 2)
        self.y_data     = torch.tensor(irradiance,  dtype=torch.float32)   # (T, N)

        # Normalise irradiance to [0, 1] using global statistics
        self.y_min = float(self.y_data.min())
        self.y_max = float(self.y_data.max())
        self.y_data = (self.y_data - self.y_min) / (self.y_max - self.y_min + 1e-8)

        self.h_sun_data = torch.tensor(h_sun, dtype=torch.float32)         # (T, N)
        self.h_min = float(self.h_sun_data.min())
        self.h_max = float(self.h_sun_data.max())
        self.h_sun_data = (self.h_sun_data - self.h_min) / (self.h_max - self.h_min + 1e-8)

    def __len__(self) -> int:
        return len(self.y_data)

    def __getitem__(self, idx: int):
        y = self.y_data[idx].unsqueeze(-1)      # (N, 1)
        h = self.h_sun_data[idx].unsqueeze(-1)  # (N, 1)
        
        return {
            'pos': self.pos_tensor,             # (N, 2)
            'y': y,                             # (N, 1)
            'h_sun': h                          # (N, 1)
        }