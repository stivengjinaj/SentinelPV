"""
Call get_full_irradiance() once per timestep, passing
the live irradiance readings from your 15 sentinel panels.

Usage (standalone test against known timesteps):
    To get IRRAD_MIN and IRRAD_MAX run:
        python infer.py --constants
    python infer_global.py --nc path/to/pvgis_data.nc
    python infer_global.py --nc path/to/pvgis_data.nc --n_timesteps 500
    python infer_global.py --nc path/to/pvgis_data.nc --timestep 1000
    python infer_global.py --nc path/to/pvgis_data.nc --threshold 50.0
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from models.transolver_pv import IrradianceModel

STAGE1_CKPT     = "training_history/train_pvgis2005_15sentinels/irradiance_stage1_final.pth"
SENTINEL_PATH   = "training_history/train_pvgis2005_15sentinels/sentinel_panels.npy"
COORDS_PATH     = "training_history/train_pvgis2005_15sentinels/dataset/coords.npy"
PANEL_IDS_PATH  = "training_history/train_pvgis2005_15sentinels/dataset/panel_ids.npy"

IRRAD_MIN = 0.0
IRRAD_MAX = 1108.010010


class IrradianceReconstructor:
    """
    Production wrapper around the trained IrradianceModel.

    Instantiate once at service startup, then call predict() on every
    new timestep. All tensors stay on the chosen device; only numpy
    arrays cross the boundary.
    """

    def __init__(
        self,
        ckpt_path:      str   = STAGE1_CKPT,
        sentinel_path:  str   = SENTINEL_PATH,
        coords_path:    str   = COORDS_PATH,
        panel_ids_path: str   = PANEL_IDS_PATH,
        irrad_min:      float = IRRAD_MIN,
        irrad_max:      float = IRRAD_MAX,
        device:         str   = None,
        n_steps:        int   = 5,
        n_samples:      int   = 1,
    ):
        self.irrad_min = irrad_min
        self.irrad_max = irrad_max
        self.n_steps   = n_steps
        self.n_samples = n_samples

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"[IrradianceReconstructor] device = {self.device}")

        raw_coords     = np.load(coords_path).astype(np.float32)
        self.panel_ids = np.load(panel_ids_path)
        N = len(raw_coords)

        self._coords_min = raw_coords.min(axis=0)
        self._coords_max = raw_coords.max(axis=0)
        coords_norm      = self._normalise_coords(raw_coords)
        self.all_pos     = torch.tensor(coords_norm, device=self.device)

        sentinel_latlon        = np.load(sentinel_path).astype(np.float32)
        self.sentinel_indices  = self._match_sentinels(sentinel_latlon, raw_coords)
        S = len(self.sentinel_indices)
        print(f"[IrradianceReconstructor] {N} panels, {S} sentinels")
        print(f"[IrradianceReconstructor] sentinel panel IDs: "
              f"{self.panel_ids[self.sentinel_indices].tolist()}")

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

    def predict(self, sentinel_readings: dict, force_zero: bool = False) -> dict:
        """
        force_zero : if True, skip the ODE entirely and return a zero field.
                    Call this when you know it's nighttime (e.g. from sun
                    elevation angle or when all sentinel readings are 0).
        """
        sentinel_ids = self.panel_ids[self.sentinel_indices]
        s_irrad_raw  = np.array(
            [sentinel_readings[pid] for pid in sentinel_ids], dtype=np.float32
        )

        # Auto-detect night from sentinel readings if not forced
        if force_zero or s_irrad_raw.mean() < 1.0:   # all sentinels are ~0
            return {
                "panel_ids":           self.panel_ids,
                "irradiance":          np.zeros(len(self.panel_ids), dtype=np.float32),
                "sentinel_ids":        sentinel_ids,
                "sentinel_irradiance": s_irrad_raw,
            }

        s_irrad_norm = self._normalise_irrad(s_irrad_raw)
        irrad_norm   = self._reconstruct(s_irrad_norm)
        irrad_wm2    = self._denormalise_irrad(irrad_norm)

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

    def _match_sentinels(self, sentinel_latlon, all_latlon):
        diffs   = all_latlon[None] - sentinel_latlon[:, None]
        dists   = np.linalg.norm(diffs, axis=-1)
        indices = dists.argmin(axis=-1)
        max_d   = dists[np.arange(len(indices)), indices].max()
        if max_d > 0.01:
            print(f"  WARNING: largest sentinel-to-panel distance = "
                  f"{max_d:.5f} deg")
        return indices

    @torch.no_grad()
    def _reconstruct(self, s_irrad_norm: np.ndarray) -> np.ndarray:
        pos   = self.all_pos
        s_idx = torch.tensor(self.sentinel_indices, device=self.device)
        s_pos = pos[s_idx]
        s_y   = torch.tensor(
            s_irrad_norm, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        sensor_feat = torch.cat([s_pos, s_y], dim=-1).unsqueeze(0)
        s  = self.model.sensor_encoder(sensor_feat)
        s2 = self.model.sensor_encoder_2(sensor_feat)

        pos_bc = pos.unsqueeze(0)
        ref_d  = self.model._ref_grid_distances(pos_bc)

        pred_acc = torch.zeros(len(pos), 1, device=self.device)

        for _ in range(self.n_samples):
            z  = torch.randn(1, len(pos), 1, device=self.device)
            dt = 1.0 / self.n_steps

            for step in range(self.n_steps):
                t_val = torch.tensor(
                    [step / self.n_steps],
                    device=self.device, dtype=torch.float32,
                )
                x     = torch.cat([pos_bc, z, ref_d], dim=-1)
                fx    = self.model.preprocess(x) + self.model.placeholder[None, None, :]
                t_emb = self.model.t_embedder(t_val) + s2.mean(dim=1)
                x_out = self.model.transformer(fx, t_emb, s)
                vel   = self.model.mlp_head(x_out, t_emb)
                z     = z + vel * dt

            pred_acc += z.squeeze(0)

        pred = (pred_acc / self.n_samples).squeeze(-1)
        pred = pred.clamp(min=0.0)
        return pred.cpu().numpy()


def _relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-8)
    )


def _test_single(
    rec:       "IrradianceReconstructor",
    irr_nc:    np.ndarray,
    timestep:  int,
    threshold: float,
):
    y_true = irr_nc[:, timestep].astype(np.float32)

    sentinel_ids = rec.panel_ids[rec.sentinel_indices]
    readings     = {pid: float(y_true[i])
                    for pid, i in zip(sentinel_ids, rec.sentinel_indices)}

    result    = rec.predict(readings)
    y_pred    = result["irradiance"]
    mean_true = float(y_true.mean())
    rel_l2    = _relative_l2(y_true, y_pred)
    is_day    = threshold == 0.0 or mean_true >= threshold

    print(f"\n  Timestep        : {timestep}")
    print(f"  True  mean      : {mean_true:.2f} W/m2")
    print(f"  Pred  mean      : {float(y_pred.mean()):.2f} W/m2")
    if threshold > 0.0:
        print(f"  Daytime?        : {'yes' if is_day else 'NO (near-zero field, error metric unreliable)'}")
    print(f"  Relative L2     : {rel_l2:.4f}  ({rel_l2*100:.2f}%)"
          + ("" if is_day else "  [excluded from aggregate]"))
    print()
    print(f"  {'ID':<12}  {'True':>8}  {'Pred':>8}  {'Err':>8}")
    print(f"  {'-'*44}")
    for i in range(min(10, len(y_true))):
        pid = rec.panel_ids[i]
        print(f"  {pid:<12}  {y_true[i]:>8.2f}  {y_pred[i]:>8.2f}  "
              f"  {abs(y_true[i]-y_pred[i]):>8.2f}")


def _test_aggregate(
    rec:         "IrradianceReconstructor",
    irr_nc:      np.ndarray,
    n_timesteps: int,
    threshold:   float,
    seed:        int = 42,
):
    T   = irr_nc.shape[1]
    rng = np.random.default_rng(seed)
    indices = rng.choice(T, size=min(n_timesteps, T), replace=False)
    indices.sort()

    errors_day   = []
    errors_night = []
    skipped      = 0

    threshold_label = f"{threshold:.0f} W/m2" if threshold > 0.0 else "disabled (all timesteps counted)"
    print(f"\n  Evaluating {len(indices)} timesteps  "
          f"(daytime threshold: {threshold_label})\n")

    for k, t in enumerate(indices):
        y_true = irr_nc[:, t].astype(np.float32)
        mean_t = float(y_true.mean())

        sentinel_ids = rec.panel_ids[rec.sentinel_indices]
        readings     = {pid: float(y_true[i])
                        for pid, i in zip(sentinel_ids, rec.sentinel_indices)}

        try:
            result = rec.predict(readings)
            y_pred = result["irradiance"]
            rel_l2 = _relative_l2(y_true, y_pred)

            if threshold == 0.0 or mean_t >= threshold:
                errors_day.append(rel_l2)
            else:
                errors_night.append(rel_l2)

        except Exception as e:
            print(f"  [WARN] timestep {t} failed: {e}")
            skipped += 1
            continue

        if (k + 1) % 50 == 0 or (k + 1) == len(indices):
            day_str = f"{np.mean(errors_day)*100:.2f}%" if errors_day else "n/a"
            print(f"  [{k+1:>{len(str(len(indices)))}}/{len(indices)}]  "
                  f"{'all' if threshold == 0.0 else 'daytime'} so far: {day_str}  "
                  f"({'all' if threshold == 0.0 else 'day'}={len(errors_day)}"
                  + (f", night={len(errors_night)}" if threshold > 0.0 else "")
                  + f", skip={skipped})")

    print("\n" + "=" * 55)
    print("  AGGREGATE RESULTS")
    print("=" * 55)
    print(f"  Total timesteps evaluated : {len(indices)}")
    print(f"  Skipped (error)           : {skipped}")

    if threshold > 0.0:
        print(f"  Daytime  (mean ≥ {threshold:.0f} W/m2) : {len(errors_day)}")
        print(f"  Nighttime (mean <  {threshold:.0f} W/m2): {len(errors_night)}")
        section_label = "Daytime Relative L2"
    else:
        print(f"  Threshold: disabled — all timesteps counted")
        section_label = "Overall Relative L2"

    if errors_day:
        arr = np.array(errors_day)
        print(f"\n  ── {section_label} ──────────────────────────")
        print(f"     Mean   : {arr.mean()*100:.2f}%")
        print(f"     Median : {np.median(arr)*100:.2f}%")
        print(f"     P90    : {np.percentile(arr, 90)*100:.2f}%")
        print(f"     Min    : {arr.min()*100:.2f}%")
        print(f"     Max    : {arr.max()*100:.2f}%")
    else:
        print(f"\n  No {'daytime ' if threshold > 0.0 else ''}timesteps found in sample.")

    if threshold > 0.0 and errors_night:
        arr_n = np.array(errors_night)
        print(f"\n  ── Night/near-zero Relative L2 (informational) ──")
        print(f"     Mean   : {arr_n.mean()*100:.2f}%")
        print(f"     Median : {np.median(arr_n)*100:.2f}%")
    print("=" * 55)


def print_norm_constants(irrad_path: str, coords_path: str):
    irr    = np.load(irrad_path).astype(np.float32)
    coords = np.load(coords_path).astype(np.float32)
    print("=" * 45)
    print("  Normalisation constants for infer.py")
    print("=" * 45)
    print(f"  IRRAD_MIN = {float(irr.min()):.6f}")
    print(f"  IRRAD_MAX = {float(irr.max()):.6f}")
    print(f"  coords_min = {coords.min(axis=0).tolist()}")
    print(f"  coords_max = {coords.max(axis=0).tolist()}")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc",          default=None,
                        help="NetCDF path for the standalone test")
    parser.add_argument("--const_npy",   default="training_history/train_pvgis2005_15sentinels/dataset/irradiance_train.npy",
                        help="NPY file with stage 1 results")
    parser.add_argument("--timestep",    type=int,   default=None,
                        help="Single timestep index to test against")
    parser.add_argument("--n_timesteps", type=int,   default=None,
                        help="Number of random timesteps for aggregate eval "
                             "(default: all). Ignored if --timestep is set.")
    parser.add_argument("--threshold",   type=float, default=0.0,
                        help="Daytime filter threshold in W/m2. Timesteps with "
                             "mean irradiance below this are reported separately "
                             "and excluded from the headline metric. "
                             "Default: 0.0 (disabled — all timesteps counted).")
    parser.add_argument("--constants",   action="store_true",
                        help="Print normalisation constants and exit")
    args = parser.parse_args()

    if args.constants:
        print_norm_constants(args.const_npy, COORDS_PATH)

    elif args.nc:
        import xarray as xr
        print(f"\nLoading: {args.nc}")
        ds     = xr.open_dataset(args.nc)
        irr_nc = ds["solar_irradiance_poa"].values
        ds.close()
        print(f"Dataset shape: {irr_nc.shape}  (N panels, T timesteps)")

        rec = IrradianceReconstructor()

        if args.timestep is not None:
            _test_single(rec, irr_nc, args.timestep, args.threshold)
        else:
            T = irr_nc.shape[1]
            n = args.n_timesteps if args.n_timesteps is not None else T
            _test_aggregate(rec, irr_nc, n_timesteps=n, threshold=args.threshold)

    else:
        parser.print_help()