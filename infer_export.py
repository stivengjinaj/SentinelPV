import argparse
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from models.transolver_pv import IrradianceModel

TRAINING_PATH   = "training_history/train_pvgis2005_2022_15sentinels"
STAGE1_CKPT     = f"{TRAINING_PATH}/irradiance_stage1_final.pth"
SENTINEL_PATH   = f"{TRAINING_PATH}/sentinel_panels.npy"
COORDS_PATH     = f"{TRAINING_PATH}/dataset/coords.npy"
PANEL_IDS_PATH  = f"{TRAINING_PATH}/dataset/panel_ids.npy"

IRRAD_MIN = 0.0       # W/m2
IRRAD_MAX = 1108.010010    # W/m2

class IrradianceReconstructor:
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

        raw_coords  = np.load(coords_path).astype(np.float32)
        self.panel_ids = np.load(panel_ids_path)
        N = len(raw_coords)

        self._coords_min = raw_coords.min(axis=0)
        self._coords_max = raw_coords.max(axis=0)
        coords_norm = self._normalise_coords(raw_coords)
        self.all_pos = torch.tensor(coords_norm, device=self.device)

        sentinel_latlon = np.load(sentinel_path).astype(np.float32)
        self.sentinel_indices = self._match_sentinels(sentinel_latlon, raw_coords)
        S = len(self.sentinel_indices)
        print(f"[IrradianceReconstructor] {N} panels, {S} sentinels")

        self.model = IrradianceModel(
            space_dim=2, fun_dim=1, out_dim=1,
            n_layers=12, n_hidden=374, slice_num=32,
        ).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def predict(self, sentinel_readings: dict) -> dict:
        sentinel_ids = self.panel_ids[self.sentinel_indices]
        s_irrad_raw  = np.array([sentinel_readings[pid] for pid in sentinel_ids], dtype=np.float32)
        s_irrad_norm = self._normalise_irrad(s_irrad_raw)
        irrad_norm = self._reconstruct(s_irrad_norm)
        irrad_wm2 = self._denormalise_irrad(irrad_norm)

        return {
            "panel_ids":           self.panel_ids,
            "irradiance":          irrad_wm2,
            "sentinel_ids":        sentinel_ids,
            "sentinel_irradiance": s_irrad_raw,
        }

    def _normalise_coords(self, coords: np.ndarray) -> np.ndarray:
        return (coords - self._coords_min) / (self._coords_max - self._coords_min + 1e-8)

    def _normalise_irrad(self, x: np.ndarray) -> np.ndarray:
        return (x - self.irrad_min) / (self.irrad_max - self.irrad_min + 1e-8)

    def _denormalise_irrad(self, x: np.ndarray) -> np.ndarray:
        return x * (self.irrad_max - self.irrad_min) + self.irrad_min

    def _match_sentinels(self, sentinel_latlon: np.ndarray, all_latlon: np.ndarray) -> np.ndarray:
        diffs   = all_latlon[None, :, :] - sentinel_latlon[:, None, :]
        dists   = np.linalg.norm(diffs, axis=-1)
        indices = dists.argmin(axis=-1)
        return indices

    @torch.no_grad()
    def _reconstruct(self, s_irrad_norm: np.ndarray) -> np.ndarray:
        pos = self.all_pos
        s_idx = torch.tensor(self.sentinel_indices, device=self.device)
        s_pos = pos[s_idx]
        s_y = torch.tensor(s_irrad_norm, dtype=torch.float32, device=self.device).unsqueeze(-1)

        sensor_feat = torch.cat([s_pos, s_y], dim=-1).unsqueeze(0)
        s  = self.model.sensor_encoder(sensor_feat)
        s2 = self.model.sensor_encoder_2(sensor_feat)

        pos_bc = pos.unsqueeze(0)
        ref_d  = self.model._ref_grid_distances(pos_bc)
        pred_acc = torch.zeros(len(pos), 1, device=self.device, dtype=torch.float32)

        for _ in range(self.n_samples):
            z  = torch.randn(1, len(pos), 1, device=self.device)
            dt = 1.0 / self.n_steps
            for step in range(self.n_steps):
                t_val = torch.tensor([step / self.n_steps], device=self.device, dtype=torch.float32)
                x     = torch.cat([pos_bc, z, ref_d], dim=-1)
                fx    = self.model.preprocess(x) + self.model.placeholder[None, None, :]
                t_emb = self.model.t_embedder(t_val) + s2.mean(dim=1)
                x_out = self.model.transformer(fx, t_emb, s)
                vel   = self.model.mlp_head(x_out, t_emb)
                z     = z + vel * dt
            pred_acc += z.squeeze(0)

        pred = (pred_acc / self.n_samples).squeeze(-1)
        return pred.clamp(min=0.0).cpu().numpy()

def print_norm_constants(irrad_path: str, coords_path: str):
    irr    = np.load(irrad_path).astype(np.float32)
    coords = np.load(coords_path).astype(np.float32)
    print("=" * 45)
    print(f"  IRRAD_MIN = {float(irr.min()):.6f}")
    print(f"  IRRAD_MAX = {float(irr.max()):.6f}")
    print("=" * 45)

def _test(nc_path: str, timestep: int, export: bool = False):
    import xarray as xr
    print(f"\nLoading test data from: {nc_path} (timestep={timestep})")
    ds = xr.open_dataset(nc_path)
    
    # Check for correct variable name
    var_name = "solar_irradiance_poa" if "solar_irradiance_poa" in ds.variables else "G(h)"
    irr_nc = ds[var_name].values
    
    y_true = irr_nc[:, timestep].astype(np.float32)
    ds.close()

    rec = IrradianceReconstructor()
    sentinel_ids = rec.panel_ids[rec.sentinel_indices]
    readings = {pid: float(y_true[i]) for pid, i in zip(sentinel_ids, rec.sentinel_indices)}

    result = rec.predict(readings)
    y_pred = result["irradiance"]

    rel_l2 = (np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-8))
    print(f"  Relative L2 error: {rel_l2*100:.2f}%")

    if export:
        export_df = pd.DataFrame({
            "panel_id": rec.panel_ids,
            "ENERGIA": y_true,                # Ground Truth
            "P_ac_theoretical_kW": y_pred     # Prediction
        })
        # Map IDs to categorise into bins later in Professor's notebook
        export_df.to_csv(f"reconstruction_results_{timestep}.csv", index=False)
        print(f"\n[EXPORT] Results saved to reconstruction_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc",        default=None, help="NetCDF path")
    parser.add_argument("--timestep",  type=int, default=1000, help="Timestep index")
    parser.add_argument("--constants", action="store_true", help="Print normalization constants")
    parser.add_argument("--export",    action="store_true", help="Export results to CSV for binning study")
    args = parser.parse_args()

    if args.constants:
        print_norm_constants(f"{TRAINING_PATH}/dataset/irradiance_train.npy", f"{TRAINING_PATH}/dataset/coords.npy")
    elif args.nc:
        _test(args.nc, args.timestep, export=args.export)
    else:
        parser.print_help()