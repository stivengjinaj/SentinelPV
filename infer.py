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
import numpy as np
import torch
import torch.nn as nn

from models.transolver_pv import IrradianceModel

STAGE1_CKPT    = "training_history/train_pvgis2005_2022_30sentinels_sun/irradiance_stage1_final.pth"
SENTINEL_PATH  = "training_history/train_pvgis2005_2022_30sentinels_sun/sentinel_panels.npy"
COORDS_PATH    = "training_history/train_pvgis2005_2022_30sentinels_sun/dataset/coords.npy"
SUN_PATH       = "training_history/train_pvgis2005_2022_30sentinels_sun/dataset/sun_height_val.npy"
PANEL_IDS_PATH = "training_history/train_pvgis2005_2022_30sentinels_sun/dataset/panel_ids.npy"

IRRAD_MIN = 0.0
IRRAD_MAX = 1108.010010
SUN_MIN   = 0.0
SUN_MAX   = 68.989998


class IrradianceReconstructor:
    def __init__(
        self,
        ckpt_path:      str   = STAGE1_CKPT,
        sentinel_path:  str   = SENTINEL_PATH,
        coords_path:    str   = COORDS_PATH,
        sun_path:       str   = SUN_PATH,
        panel_ids_path: str   = PANEL_IDS_PATH,
        irrad_min:      float = IRRAD_MIN,
        irrad_max:      float = IRRAD_MAX,
        sun_min:        float = SUN_MIN,
        sun_max:        float = SUN_MAX,
        device:         str   = None,
        n_steps:        int   = 5,
        n_samples:      int   = 1,
    ):
        self.irrad_min = irrad_min
        self.irrad_max = irrad_max
        self.sun_min   = sun_min
        self.sun_max   = sun_max
        self.n_steps   = n_steps
        self.n_samples = n_samples

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"[IrradianceReconstructor] device = {self.device}")

        # Panel coordinates
        raw_coords     = np.load(coords_path).astype(np.float32)    # (N, 2)
        self.panel_ids = np.load(panel_ids_path)                     # (N,)
        N = len(raw_coords)

        self._coords_min = raw_coords.min(axis=0)
        self._coords_max = raw_coords.max(axis=0)
        coords_norm      = self._normalise_coords(raw_coords)        # (N, 2)
        self.all_pos     = torch.tensor(
            coords_norm, dtype=torch.float32, device=self.device
        )                                                             # (N, 2)

        self._all_hsun_raw = np.load(sun_path).astype(np.float32)   # (T, N)
        print(f"[IrradianceReconstructor] sun height array: "
              f"{self._all_hsun_raw.shape}  (T={self._all_hsun_raw.shape[0]}, N={N})")

        # Sentinel matching
        sentinel_latlon       = np.load(sentinel_path).astype(np.float32)  # (S, 2)
        self.sentinel_indices = self._match_sentinels(sentinel_latlon, raw_coords)
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
        print(f"[IrradianceReconstructor] model loaded ({total:,} params)")


    def predict(self, sentinel_readings: dict, timestep: int) -> dict:
        """
        Reconstruct irradiance for all N panels from sentinel readings.

        Parameters
        ----------
        sentinel_readings : dict  panel_id (str) -> irradiance (float, W/m²)
                            Must contain every sentinel panel ID.
        timestep          : int   index into the sun-height array (0-based).
                            Used to look up the current sun elevation for
                            all panels.

        Returns
        -------
        dict:
            'panel_ids'           : (N,) str
            'irradiance'          : (N,) float  W/m²
            'sentinel_ids'        : (S,) str
            'sentinel_irradiance' : (S,) float  W/m²  (the values you passed in)
        """
        sentinel_ids = self.panel_ids[self.sentinel_indices]

        # Sentinel irradiance in fixed panel order
        s_irrad_raw  = np.array(
            [sentinel_readings[pid] for pid in sentinel_ids],
            dtype=np.float32,
        )                                                             # (S,)

        hsun_full_raw = self._all_hsun_raw[timestep]                 # (N,)

        # Normalise
        s_irrad_norm  = self._normalise_irrad(s_irrad_raw)           # (S,)
        hsun_full_norm = self._normalise_sun(hsun_full_raw)           # (N,)

        s_hsun_norm   = hsun_full_norm[self.sentinel_indices]        # (S,)

        irrad_norm = self._reconstruct(
            s_irrad_norm, s_hsun_norm, hsun_full_norm
        )                                                             # (N,)

        irrad_wm2 = self._denormalise_irrad(irrad_norm)              # (N,)

        return {
            "panel_ids":           self.panel_ids,
            "irradiance":          irrad_wm2,
            "sentinel_ids":        sentinel_ids,
            "sentinel_irradiance": s_irrad_raw,
        }

    def _normalise_coords(self, coords: np.ndarray) -> np.ndarray:
        return (coords - self._coords_min) / (
            self._coords_max - self._coords_min + 1e-8
        )

    def _normalise_irrad(self, x: np.ndarray) -> np.ndarray:
        return (x - self.irrad_min) / (self.irrad_max - self.irrad_min + 1e-8)

    def _denormalise_irrad(self, x: np.ndarray) -> np.ndarray:
        return x * (self.irrad_max - self.irrad_min) + self.irrad_min

    def _normalise_sun(self, x: np.ndarray) -> np.ndarray:
        return (x - self.sun_min) / (self.sun_max - self.sun_min + 1e-8)

    def _match_sentinels(
        self,
        sentinel_latlon: np.ndarray,   # (S, 2)
        all_latlon:      np.ndarray,   # (N, 2)
    ) -> np.ndarray:
        diffs   = all_latlon[None] - sentinel_latlon[:, None]        # (S, N, 2)
        dists   = np.linalg.norm(diffs, axis=-1)                      # (S, N)
        indices = dists.argmin(axis=-1)                               # (S,)
        max_dist = dists[np.arange(len(indices)), indices].max()
        if max_dist > 0.01:
            print(f"  WARNING: largest sentinel-to-panel distance = "
                  f"{max_dist:.5f} deg. Check sentinel_panels.npy alignment.")
        return indices

    @torch.no_grad()
    def _reconstruct(
        self,
        s_irrad_norm:  np.ndarray,   # (S,)  sentinel irradiance, normalised
        s_hsun_norm:   np.ndarray,   # (S,)  sentinel sun heights, normalised
        hsun_full_norm: np.ndarray,  # (N,)  all-panel sun heights, normalised
    ) -> np.ndarray:                 # (N,)  irradiance, normalised
        """
        Two distinct roles for sun height:
          - hsun_full_norm → appended to every field token inside the ODE loop
          - s_hsun_norm    → part of the sensor feature fed to cross-attention
        """
        pos    = self.all_pos                                         # (N, 2)
        pos_bc = pos.unsqueeze(0)                                     # (1, N, 2)
        ref_d  = self.model._ref_grid_distances(pos_bc)               # (1, N, 16)

        # FIX 5: convert full-field sun heights to a (1, N, 1) tensor on device
        h_sun_tensor = torch.tensor(
            hsun_full_norm, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(-1)                                  # (1, N, 1)

        # Sentinel sensor features
        s_idx = torch.tensor(self.sentinel_indices, device=self.device)
        s_pos = pos[s_idx].unsqueeze(0)                               # (1, S, 2)
        s_y   = torch.tensor(
            s_irrad_norm, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(-1)                                  # (1, S, 1)
        s_h   = torch.tensor(
            s_hsun_norm, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(-1)                                  # (1, S, 1)

        # Encode sensor context once — does not change across ODE steps
        sensor_feat = torch.cat([s_pos, s_y, s_h], dim=-1)           # (1, S, 4)
        s  = self.model.sensor_encoder(sensor_feat)                   # (1, S, H)
        s2 = self.model.sensor_encoder_2(sensor_feat)                 # (1, S, H/4)

        pred_acc = torch.zeros(len(pos), 1, device=self.device)

        for _ in range(self.n_samples):
            z  = torch.randn(1, len(pos), 1, device=self.device)      # (1, N, 1)
            dt = 1.0 / self.n_steps

            for i in range(self.n_steps):
                t_val = torch.tensor(
                    [i / self.n_steps],
                    dtype=torch.float32, device=self.device,
                )
                # h_sun_tensor provides static physical context at each ODE step
                x     = torch.cat([pos_bc, z, h_sun_tensor, ref_d], dim=-1)  # (1, N, 20)
                fx    = self.model.preprocess(x) + self.model.placeholder[None, None, :]
                t_emb = self.model.t_embedder(t_val) + s2.mean(dim=1)
                x_out = self.model.transformer(fx, t_emb, s)
                vel   = self.model.mlp_head(x_out, t_emb)             # (1, N, 1)
                z     = z + vel * dt

            pred_acc += z.squeeze(0)                                  # (N, 1)

        pred = (pred_acc / self.n_samples).squeeze(-1)                # (N,)
        return pred.clamp(min=0.0).cpu().numpy()


def print_norm_constants(irrad_path: str, sun_path: str, coords_path: str):
    """Print training-time normalisation constants to hardcode above."""
    irr    = np.load(irrad_path).astype(np.float32)
    sun    = np.load(sun_path).astype(np.float32)
    coords = np.load(coords_path).astype(np.float32)

    print("=" * 45)
    print("  Normalisation constants for infer.py")
    print("=" * 45)
    print(f"  IRRAD_MIN = {float(irr.min()):.6f}")
    print(f"  IRRAD_MAX = {float(irr.max()):.6f}")
    print(f"  SUN_MIN   = {float(sun.min()):.6f}")
    print(f"  SUN_MAX   = {float(sun.max()):.6f}")
    print()
    print(f"  coords_min = {coords.min(axis=0).tolist()}")
    print(f"  coords_max = {coords.max(axis=0).tolist()}")
    print("=" * 45)


def _test(nc_path: str, timestep: int):
    """
    Smoke test against a known timestep from the NetCDF.
    Feeds true sentinel irradiance values and measures reconstruction error.
    """
    import xarray as xr

    print(f"\nLoading test data from: {nc_path}  (timestep={timestep})")
    ds     = xr.open_dataset(nc_path)
    irr_nc = ds["solar_irradiance_poa"].values          # (N, T)
    y_true = irr_nc[:, timestep].astype(np.float32)     # (N,)
    ds.close()

    rec = IrradianceReconstructor()

    sentinel_ids = rec.panel_ids[rec.sentinel_indices]
    readings     = {
        pid: float(y_true[idx])
        for pid, idx in zip(sentinel_ids, rec.sentinel_indices)
    }

    result  = rec.predict(readings, timestep=timestep)
    y_pred  = result["irradiance"]
    rel_l2  = (np.linalg.norm(y_true - y_pred)
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
        pid = rec.panel_ids[i]
        err = abs(y_true[i] - y_pred[i])
        print(f"  {pid:<12}  {y_true[i]:>8.2f}  {y_pred[i]:>8.2f}  {err:>8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc",        default=None)
    parser.add_argument("--timestep",  type=int, default=1000)
    parser.add_argument("--constants", action="store_true")
    args = parser.parse_args()

    if args.constants:
        print_norm_constants(
            irrad_path  = "datasets/irradiance_train.npy",
            sun_path    = SUN_PATH,
            coords_path = COORDS_PATH,
        )
    elif args.nc:
        _test(args.nc, args.timestep)
    else:
        parser.print_help()