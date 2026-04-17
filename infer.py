"""
Production inference: reconstruct full network irradiance
              from live sentinel panel readings.

After Stage 2 we have:
    - checkpoints/irradiance_stage1_final.pth   (trained model)
    - results/sentinel_panels.npy               (15 optimal panel coords)
    - datasets/coords.npy                       (all 1149 panel coords)
    - datasets/panel_ids.npy                    (panel IDs, same order)

Call get_full_irradiance() once per timestep, passing
the live irradiance readings from your 15 sentinel panels.

Usage (standalone test against a known timestep):
    To get IRRAD_MIN and IRRAD_MAX run: 
        python infer.py --constants  
    python infer.py --nc path/to/pvgis_data.nc --timestep 1000
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from models.transolver_pv import IrradianceModel

STAGE1_CKPT     = "training_history/train_pvgis2005_2022_30sentinels/irradiance_stage1_final.pth"
SENTINEL_PATH   = "training_history/train_pvgis2005_2022_30sentinels/sentinel_panels.npy"
COORDS_PATH     = "training_history/train_pvgis2005_2022_30sentinels/dataset/coords.npy"
PANEL_IDS_PATH  = "training_history/train_pvgis2005_2022_30sentinels/dataset/panel_ids.npy"


IRRAD_MIN = 0.0       # W/m2
IRRAD_MAX = 1108.0    # W/m2


class IrradianceReconstructor:
    """
    Production wrapper around the trained IrradianceModel.

    Instantiate once at service startup, then call predict() on every
    new timestep. All tensors stay on the chosen device; only numpy
    arrays cross the boundary.

    Parameters
    ----------
    ckpt_path       : path to irradiance_stage1_final.pth
    sentinel_path   : path to sentinel_panels.npy  (S, 2) lat/lon
    coords_path     : path to coords.npy           (N, 2) lat/lon
    panel_ids_path  : path to panel_ids.npy        (N,)   string IDs
    irrad_min       : irradiance min used at training time (W/m2)
    irrad_max       : irradiance max used at training time (W/m2)
    device          : 'cuda', 'cpu', or None (auto)
    n_steps         : ODE integration steps (5 is fast, 20 is more accurate)
    n_samples       : number of stochastic samples to average (1 = fastest)
    """

    def __init__(
        self,
        ckpt_path:      str  = STAGE1_CKPT,
        sentinel_path:  str  = SENTINEL_PATH,
        coords_path:    str  = COORDS_PATH,
        panel_ids_path: str  = PANEL_IDS_PATH,
        irrad_min:      float = IRRAD_MIN,
        irrad_max:      float = IRRAD_MAX,
        device:         str  = None,
        n_steps:        int  = 5,
        n_samples:      int  = 1,
    ):
        self.irrad_min = irrad_min
        self.irrad_max = irrad_max
        self.n_steps   = n_steps
        self.n_samples = n_samples

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"[IrradianceReconstructor] device = {self.device}")

        # Load all 1149 panel coordinates
        raw_coords  = np.load(coords_path).astype(np.float32)   # (N, 2) lat/lon
        self.panel_ids = np.load(panel_ids_path)                 # (N,)  strings
        N = len(raw_coords)

        # Normalise coordinates to [0, 1] -- same transform as IrradianceDataset
        self._coords_min = raw_coords.min(axis=0)
        self._coords_max = raw_coords.max(axis=0)
        coords_norm = self._normalise_coords(raw_coords)         # (N, 2)
        self.all_pos = torch.tensor(coords_norm, device=self.device)  # (N, 2)

        # Match sentinel lat/lon to panel indices
        sentinel_latlon = np.load(sentinel_path).astype(np.float32)  # (S, 2)
        self.sentinel_indices = self._match_sentinels(
            sentinel_latlon, raw_coords
        )                                                         # (S,) int
        S = len(self.sentinel_indices)
        print(f"[IrradianceReconstructor] {N} panels, {S} sentinels")
        print(f"[IrradianceReconstructor] sentinel panel IDs: "
              f"{self.panel_ids[self.sentinel_indices].tolist()}")

        # Model
        self.model = IrradianceModel(
            space_dim=2, fun_dim=1, out_dim=1,
            n_layers=12, n_hidden=374, slice_num=32,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        total = sum(p.numel() for p in self.model.parameters())
        print(f"[IrradianceReconstructor] model loaded  ({total:,} params)")

    
    # Prediction
    def predict(self, sentinel_readings: dict) -> dict:
        """
        Reconstruct irradiance for all 1149 panels from sentinel readings.

        Parameters
        ----------
        sentinel_readings : dict mapping panel_id (str) -> irradiance (float, W/m2)
                            Must contain all sentinel panel IDs.
                            Extra keys are silently ignored.

        Returns
        -------
        dict with keys:
            'panel_ids'    : (N,)   str   -- all panel IDs in fixed order
            'irradiance'   : (N,)   float -- reconstructed W/m2 for every panel
            'sentinel_ids' : (S,)   str   -- which panels were used as sensors
            'sentinel_irradiance' : (S,) float -- the raw readings you passed in
        """
        # Build sentinel irradiance vector in the correct panel order
        sentinel_ids = self.panel_ids[self.sentinel_indices]
        s_irrad_raw  = np.array(
            [sentinel_readings[pid] for pid in sentinel_ids],
            dtype=np.float32,
        )                                                        # (S,) W/m2

        # Normalise sentinel irradiance to [0, 1]
        s_irrad_norm = self._normalise_irrad(s_irrad_raw)       # (S,)

        # Run ODE reconstruction
        irrad_norm = self._reconstruct(s_irrad_norm)            # (N,)

        # De-normalise back to W/m2
        irrad_wm2 = self._denormalise_irrad(irrad_norm)         # (N,)

        return {
            "panel_ids":           self.panel_ids,
            "irradiance":          irrad_wm2,
            "sentinel_ids":        sentinel_ids,
            "sentinel_irradiance": s_irrad_raw,
        }

  
    def _normalise_coords(self, coords: np.ndarray) -> np.ndarray:
        """Map lat/lon to [0, 1] using the training-time min/max."""
        return (coords - self._coords_min) / (
            self._coords_max - self._coords_min + 1e-8
        )

    def _normalise_irrad(self, x: np.ndarray) -> np.ndarray:
        return (x - self.irrad_min) / (self.irrad_max - self.irrad_min + 1e-8)

    def _denormalise_irrad(self, x: np.ndarray) -> np.ndarray:
        return x * (self.irrad_max - self.irrad_min) + self.irrad_min

    def _match_sentinels(
        self,
        sentinel_latlon: np.ndarray,   # (S, 2)
        all_latlon:      np.ndarray,   # (N, 2)
    ) -> np.ndarray:
        """Return the panel index in all_latlon closest to each sentinel."""
        diffs   = all_latlon[None, :, :] - sentinel_latlon[:, None, :]  # (S, N, 2)
        dists   = np.linalg.norm(diffs, axis=-1)                         # (S, N)
        indices = dists.argmin(axis=-1)                                  # (S,)
        # Warn if any sentinel is unexpectedly far from a known panel
        max_dist = dists[np.arange(len(indices)), indices].max()
        if max_dist > 0.01:
            print(f"  WARNING: largest sentinel-to-panel distance = "
                  f"{max_dist:.5f} deg. Check sentinel_panels.npy alignment.")
        return indices

    @torch.no_grad()
    def _reconstruct(
        self,
        s_irrad_norm: np.ndarray,   # (S,)
        s_hsun_norm:  np.ndarray,   # (S,)
    ) -> np.ndarray:                # returns (N,) irradiance only

        pos      = self.all_pos  # (N, 2)
        h_sun_bc = self.all_hsun.unsqueeze(0) # (1, N, 1) — preloaded at init
        pos_bc   = pos.unsqueeze(0)
        ref_d    = self.model._ref_grid_distances(pos_bc)

        s_idx = torch.tensor(self.sentinel_indices, device=self.device)
        s_pos = pos[s_idx].unsqueeze(0)
        s_y   = torch.tensor(s_irrad_norm, dtype=torch.float32,
                            device=self.device).unsqueeze(0).unsqueeze(-1)  # (1,S,1)
        s_h   = torch.tensor(s_hsun_norm,  dtype=torch.float32,
                            device=self.device).unsqueeze(0).unsqueeze(-1)  # (1,S,1)

        sensor_feat = torch.cat([s_pos, s_y, s_h], dim=-1)
        s    = self.model.sensor_encoder(sensor_feat)
        s2   = self.model.sensor_encoder_2(sensor_feat)

        pred_acc = torch.zeros(len(pos), 1, device=self.device)

        for _ in range(self.n_samples):
            z  = torch.randn(1, len(pos), 1, device=self.device)
            dt = 1.0 / self.n_steps
            for i in range(self.n_steps):
                t_val = torch.tensor([i / self.n_steps], device=self.device)
                x     = torch.cat([pos_bc, z, h_sun_bc, ref_d], dim=-1)  # (1, N, 20)
                fx    = self.model.preprocess(x) + self.model.placeholder[None, None, :]
                t_emb = self.model.t_embedder(t_val) + s2.mean(dim=1)
                x_out = self.model.transformer(fx, t_emb, s)
                vel   = self.model.mlp_head(x_out, t_emb)
                z     = z + vel * dt
            pred_acc += z.squeeze(0)

        pred = (pred_acc / self.n_samples).squeeze(-1)
        return pred.clamp(min=0.0).cpu().numpy()


def print_norm_constants(irrad_path: str, coords_path: str):
    """
    Run this once after prepare_data.py to get the constants you need
    to hardcode in IRRAD_MIN / IRRAD_MAX above.

    Usage:
        from infer import print_norm_constants
        print_norm_constants("datasets/irradiance_train.npy",
                             "datasets/coords.npy")
    """
    irr    = np.load(irrad_path).astype(np.float32)
    coords = np.load(coords_path).astype(np.float32)

    print("=" * 45)
    print("  Normalisation constants for infer.py")
    print("=" * 45)
    print(f"  IRRAD_MIN = {float(irr.min()):.6f}")
    print(f"  IRRAD_MAX = {float(irr.max()):.6f}")
    print()
    print(f"  coords_min = {coords.min(axis=0).tolist()}")
    print(f"  coords_max = {coords.max(axis=0).tolist()}")
    print("=" * 45)
    print("  Paste IRRAD_MIN and IRRAD_MAX into infer.py")


def _test(nc_path: str, timestep: int):
    """
    Smoke test: load one real timestep from the NetCDF, pretend only the
    sentinel panels are observed, reconstruct, and measure the error.
    """
    import xarray as xr

    print(f"\nLoading test data from: {nc_path}  (timestep={timestep})")
    ds     = xr.open_dataset(nc_path)
    irr_nc = ds["solar_irradiance_poa"].values          # (N, T)
    y_true = irr_nc[:, timestep].astype(np.float32)    # (N,)
    ds.close()

    rec = IrradianceReconstructor()

    # Build the sentinel readings dict from true values
    sentinel_ids  = rec.panel_ids[rec.sentinel_indices]
    readings      = {pid: float(y_true[i])
                     for pid, i in zip(sentinel_ids, rec.sentinel_indices)}

    result   = rec.predict(readings)
    y_pred   = result["irradiance"]                     # (N,) W/m2

    rel_l2   = (np.linalg.norm(y_true - y_pred)
                / (np.linalg.norm(y_true) + 1e-8))

    print(f"\n  Timestep  : {timestep}")
    print(f"  True  mean: {y_true.mean():.2f} W/m2")
    print(f"  Pred  mean: {y_pred.mean():.2f} W/m2")
    print(f"  Relative L2 error: {rel_l2:.4f}  ({rel_l2*100:.2f}%)")
    print()
    print("  Per-panel sample (first 10):")
    print(f"  {'ID':<12}  {'True':>8}  {'Pred':>8}  {'Err':>8}")
    print(f"  {'-'*44}")
    for i in range(min(10, len(y_true))):
        pid  = rec.panel_ids[i]
        err  = abs(y_true[i] - y_pred[i])
        print(f"  {pid:<12}  {y_true[i]:>8.2f}  {y_pred[i]:>8.2f}  {err:>8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc",        default=None,
                        help="NetCDF path for the standalone test")
    parser.add_argument("--timestep",  type=int, default=1000,
                        help="Timestep index to test against  (default: 1000)")
    parser.add_argument("--constants", action="store_true",
                        help="Print normalisation constants and exit")
    args = parser.parse_args()

    if args.constants:
        print_norm_constants("datasets/irradiance_train.npy",
                             "datasets/coords.npy")
    elif args.nc:
        _test(args.nc, args.timestep)
    else:
        parser.print_help()