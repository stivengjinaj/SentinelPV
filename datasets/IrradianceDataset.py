import torch
import numpy as np
from torch.utils.data import Dataset


class IrradianceDatasetGrid(Dataset):
    """
    Irradiance dataset that represents the field as a dense (H, W) grid image.

    Each of the N panels is assigned to the nearest cell in a GRID_H x GRID_W
    grid (computed by the prepare_grid.py script). Cells with multiple panels
    assigned get the average of their values. Empty cells stay 0.

    Expects:
        irradiance_path   : (T, N)  float32  raw irradiance W/m2
        coords_norm_path  : (N, 2)  float32  coordinates already in [0, 1]
                            (saved by prepare_grid.py as coords_norm.npy)
        grid_indices_path : (N, 2)  int      (row, col) cell per panel
                            (saved by prepare_grid.py as grid_indices.npy)

    Returns per sample:
        grid   : (1, H, W)  float32  normalised irradiance image
        mask   : (1, H, W)  float32  1 where a real panel exists, 0 elsewhere
        y_flat : (N, 1)     float32  normalised irradiance at each panel
                                     (kept for sensor sampling and loss)
        pos    : (N, 2)     float32  normalised panel coordinates
    """

    def __init__(
        self,
        irradiance_path:   str,
        coords_path:  str,
        grid_indices_path: str,
        grid_h:            int,
        grid_w:            int,
    ):

        irradiance   = np.load(irradiance_path).astype(np.float32)    # (T, N)
        coords_norm  = np.load(coords_path).astype(np.float32)   # (N, 2)
        grid_indices = np.load(grid_indices_path).astype(np.int64)    # (N, 2)

        T, N = irradiance.shape
        print(T)
        print(N)
        assert coords_norm.shape  == (N, 2),          f"coords_norm shape mismatch: {coords_norm.shape}"
        assert grid_indices.shape == (N, 2),          f"grid_indices shape mismatch: {grid_indices.shape}"
        assert coords_norm.min() >= 0.0 - 1e-6,      "coords_norm must be in [0, 1] — run prepare_grid.py first"
        assert coords_norm.max() <= 1.0 + 1e-6,      "coords_norm must be in [0, 1] — run prepare_grid.py first"

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.N      = N
        self.T      = T

        # Normalise irradiance to [0, 1]
        self.y_min = float(irradiance.min())
        self.y_max = float(irradiance.max())
        irr_norm   = (irradiance - self.y_min) / (self.y_max - self.y_min + 1e-8)

        # Store as (T, N) float32 numpy array — converted to tensor per sample
        # to avoid holding a (8760, 1149) float32 tensor in GPU memory
        self._irr_norm = irr_norm                                      # (T, N) numpy

        # Stored as a tensor — same for every sample, never changes
        self.pos_tensor = torch.from_numpy(coords_norm)                # (N, 2)

        # Precompute flat cell indices
        # flat_idx[i] = which grid cell panel i belongs to, as a 1-D index
        # into a flattened (H*W,) array
        rows     = grid_indices[:, 0]                                  # (N,) int64
        cols     = grid_indices[:, 1]                                  # (N,) int64
        flat     = rows * grid_w + cols                                # (N,) int64

        self.flat_idx = torch.from_numpy(flat).long()                  # (N,)
        self.ones_n   = torch.ones(N, dtype=torch.float32)             # (N,)

        # Precompute static mask
        # mask[cell] = 1 if at least one panel maps to that cell
        mask_flat = torch.zeros(grid_h * grid_w, dtype=torch.float32)
        mask_flat.index_add_(0, self.flat_idx, self.ones_n)
        mask_flat = (mask_flat > 0).float()
        self.mask = mask_flat.reshape(1, grid_h, grid_w)               # (1, H, W)


    def _to_grid(self, y_1d: torch.Tensor) -> torch.Tensor:
        """
        Convert a 1-D panel value tensor to a (1, H, W) grid image.

        y_1d : (N,) float32  — exactly one dimension, values in [0, 1]

        Cells with multiple panels assigned get the mean.
        Cells with no panel stay 0.
        """
        # Defensive check — should always be (N,) here
        assert y_1d.dim() == 1 and y_1d.shape[0] == self.N, (
            f"_to_grid expects shape ({self.N},), got {tuple(y_1d.shape)}"
        )

        H, W = self.grid_h, self.grid_w

        grid_sum   = torch.zeros(H * W, dtype=torch.float32)
        grid_count = torch.zeros(H * W, dtype=torch.float32)

        # flat_idx and ones_n are CPU tensors; y_1d must also be CPU here
        grid_sum.index_add_(0, self.flat_idx, y_1d)
        grid_count.index_add_(0, self.flat_idx, self.ones_n)

        # Average where count > 0
        filled = grid_count > 0
        grid_sum[filled] = grid_sum[filled] / grid_count[filled]

        return grid_sum.reshape(1, H, W)   # (1, H, W)

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, idx: int) -> dict:
        # Load one timestep as a plain 1-D numpy slice, convert to tensor
        y_np  = self._irr_norm[idx]                    # (N,)  numpy float32
        y_1d  = torch.from_numpy(y_np.copy())          # (N,)  tensor, .copy() avoids
                                                        #        non-writable array issues

        grid  = self._to_grid(y_1d)                    # (1, H, W)

        return {
            'grid':   grid,                            # (1, H, W)
            'mask':   self.mask,                       # (1, H, W)  static
            'y_flat': y_1d.unsqueeze(-1),              # (N, 1)
            'pos':    self.pos_tensor,                 # (N, 2)
        }