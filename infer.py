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

import numpy as np
import torch
from models.transolver_pv import IrradianceModel

SAVE_PATH     = "training_history/train_pvgis2005_15sentinels_temporal"
STAGE1_CKPT   = f"{SAVE_PATH}/irradiance_stage1_final.pth"
SENTINEL_PATH = f"{SAVE_PATH}/sentinel_panels.npy"
COORDS_PATH   = f"{SAVE_PATH}/dataset/coords.npy"
PANEL_IDS_PATH= f"{SAVE_PATH}/dataset/panel_ids.npy"
IRRAD_MIN     = 0.0
IRRAD_MAX     = 1094.010010
T_IN          = 12
T_OUT         = 24


class IrradianceReconstructor:
    def __init__(
        self,
        ckpt_path:      str   = STAGE1_CKPT,
        sentinel_path:  str   = SENTINEL_PATH,
        coords_path:    str   = COORDS_PATH,
        panel_ids_path: str   = PANEL_IDS_PATH,
        irrad_min:      float = IRRAD_MIN,
        irrad_max:      float = IRRAD_MAX,
        t_in:           int   = T_IN,
        t_out:          int   = T_OUT,
        device:         str   = None,
        n_steps:        int   = 5,
        n_samples:      int   = 1,
    ):
        self.irrad_min = irrad_min
        self.irrad_max = irrad_max
        self.t_in      = t_in
        self.t_out     = t_out
        self.n_steps   = n_steps
        self.n_samples = n_samples

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        raw_coords     = np.load(coords_path).astype(np.float32)
        self.panel_ids = np.load(panel_ids_path)
        N = len(raw_coords)

        self._coords_min = raw_coords.min(axis=0)
        self._coords_max = raw_coords.max(axis=0)
        coords_norm      = self._norm_coords(raw_coords)
        self.all_pos     = torch.tensor(coords_norm, dtype=torch.float32, device=self.device)

        sentinel_latlon       = np.load(sentinel_path).astype(np.float32)
        self.sentinel_indices = self._match_sentinels(sentinel_latlon, raw_coords)
        S = len(self.sentinel_indices)
        print(f"[Reconstructor] {N} panels | {S} sentinels | t_in={t_in} | t_out={t_out}")

        self.model = IrradianceModel(
            space_dim=2, fun_dim=1, out_dim=t_out, t_in=t_in,
            n_layers=12, n_hidden=374, slice_num=32,
        ).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def predict(self, sentinel_histories: dict) -> dict:
        """
        Parameters
        ----------
        sentinel_histories : dict  panel_id -> list or array of T_in irradiance
                             values in W/m2, ordered oldest to most recent.

        Returns
        -------
        dict:
            'panel_ids'  : (N,)
            'forecast'   : (N, T_out)  W/m2, one row per panel, 24 future steps
            'sentinel_ids': (S,)
        """
        sentinel_ids = self.panel_ids[self.sentinel_indices]

        # Build (N, T_in) history matrix — non-sentinel panels get zeros
        # because the temporal encoder sees all panels, but cross-attention
        # only uses sentinel features
        N = len(self.all_pos)
        x_seq_np = np.zeros((N, self.t_in), dtype=np.float32)
        for pid, idx in zip(sentinel_ids, self.sentinel_indices):
            raw              = np.array(sentinel_histories[pid], dtype=np.float32)
            x_seq_np[idx]    = self._norm_irrad(raw)

        x_seq = torch.tensor(x_seq_np, device=self.device)   # (N, T_in)

        forecast_norm = self._reconstruct(x_seq)              # (N, T_out)
        forecast_wm2  = self._denorm_irrad(forecast_norm)     # (N, T_out)

        return {
            "panel_ids":   self.panel_ids,
            "forecast":    forecast_wm2,
            "sentinel_ids": sentinel_ids,
        }

    @torch.no_grad()
    def _reconstruct(self, x_seq: torch.Tensor) -> np.ndarray:
        pos    = self.all_pos
        pos_bc = pos.unsqueeze(0)
        ref_d  = self.model._ref_grid_distances(pos_bc)

        x_seq_bc = x_seq.unsqueeze(0)                        # (1, N, T_in)
        hist_emb = self.model.temporal_encoder(x_seq_bc)     # (1, N, n_hidden)

        s_idx = torch.tensor(self.sentinel_indices, device=self.device)
        s_pos = pos[s_idx].unsqueeze(0)                       # (1, S, 2)
        s_seq = x_seq[s_idx].unsqueeze(0)                    # (1, S, T_in)
        sensor_feat = torch.cat([s_pos, s_seq], dim=-1)
        s    = self.model.sensor_encoder(sensor_feat)
        s2   = self.model.sensor_encoder_2(sensor_feat)

        pred_acc = torch.zeros(len(pos), self.t_out, device=self.device)

        for _ in range(self.n_samples):
            z  = torch.randn(1, len(pos), self.t_out, device=self.device)
            dt = 1.0 / self.n_steps
            for i in range(self.n_steps):
                t_val = torch.tensor([i / self.n_steps], dtype=torch.float32, device=self.device)
                x_raw = torch.cat([pos_bc, z, ref_d], dim=-1)
                fx    = self.model.preprocess(x_raw) + self.model.placeholder[None, None, :]
                fx    = fx + hist_emb
                t_emb = self.model.t_embedder(t_val) + s2.mean(dim=1)
                x_out = self.model.transformer(fx, t_emb, s)
                vel   = self.model.mlp_head(x_out, t_emb)
                z     = z + vel * dt
            pred_acc += z.squeeze(0)

        pred = (pred_acc / self.n_samples).clamp(min=0.0)
        return pred.cpu().numpy()

    def _norm_coords(self, c):
        return (c - self._coords_min) / (self._coords_max - self._coords_min + 1e-8)

    def _norm_irrad(self, x):
        return (x - self.irrad_min) / (self.irrad_max - self.irrad_min + 1e-8)

    def _denorm_irrad(self, x):
        return x * (self.irrad_max - self.irrad_min) + self.irrad_min

    def _match_sentinels(self, sentinel_latlon, all_latlon):
        diffs   = all_latlon[None] - sentinel_latlon[:, None]
        dists   = np.linalg.norm(diffs, axis=-1)
        indices = dists.argmin(axis=-1)
        max_dist = dists[np.arange(len(indices)), indices].max()
        if max_dist > 0.01:
            print(f"  WARNING: sentinel-to-panel max distance = {max_dist:.5f} deg")
        return indices


def _test(irrad_path: str, start_timestep: int):
    irr = np.load(irrad_path).astype(np.float32)   # (T, N)
    rec = IrradianceReconstructor()

    sentinel_ids = rec.panel_ids[rec.sentinel_indices]
    histories    = {
        pid: irr[start_timestep - T_IN : start_timestep, idx].tolist()
        for pid, idx in zip(sentinel_ids, rec.sentinel_indices)
    }

    result   = rec.predict(histories)
    forecast = result["forecast"]                  # (N, T_out)
    y_true   = irr[start_timestep : start_timestep + T_OUT]  # (T_out, N)

    errors = []
    for h in range(T_OUT):
        rel = np.linalg.norm(y_true[h] - forecast[:, h]) / (np.linalg.norm(y_true[h]) + 1e-8)
        errors.append(rel)
        print(f"  h+{h+1:02d}  rel-L2: {rel:.4f}")

    print(f"\n  Mean rel-L2 over {T_OUT}h: {np.mean(errors):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--irrad",     default="datasets/irradiance_train.npy")
    parser.add_argument("--timestep",  type=int, default=500)
    args = parser.parse_args()
    _test(args.irrad, args.timestep)