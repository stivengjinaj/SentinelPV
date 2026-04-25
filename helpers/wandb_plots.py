import numpy as np
import torch
import wandb
import matplotlib

from datasets.IrradianceDataset import IrradianceDataset
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

@torch.no_grad()
def evaluate(
    model:     torch.nn.Module,
    dataset:   IrradianceDataset,
    device:    torch.device,
    n_samples: int = 50,
    n_steps:   int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true, all_pred = [], []

    pos = dataset.pos_tensor.to(device)
    N   = pos.shape[0]
    T   = len(dataset)

    indices = np.random.choice(T, size=min(n_samples, T), replace=False)

    for idx in indices:
        sample  = dataset[idx]
        x_seq   = sample['x_seq'].to(device)   # (N, T_in)
        y_seq   = sample['y_seq'].to(device)   # (N, T_out)

        sensor_idx = torch.randperm(N, device=device)[:15]

        pred_norm, _ = model.sample(
            pos            = pos,
            x_seq_full     = x_seq,
            y_seq_full     = y_seq,
            sensor_indices = sensor_idx,
            n_steps        = n_steps,
            n_samples      = 1,
        )

        all_true.append(y_seq.cpu().numpy().flatten())
        all_pred.append(pred_norm.cpu().numpy().flatten())

    return np.concatenate(all_true), np.concatenate(all_pred)


def log_nae_scalars(
    y_true_norm: np.ndarray,   # (M*N,) in [0, 1]
    y_pred_norm: np.ndarray,
    irrad_range: float,        # y_max - y_min from dataset, acts as normaliser
    step:        int,
) -> None:
    # NAE computed in normalised space — irrad_range cancels out in the ratio
    # so the percentage is still physically meaningful relative to the data range
    nae    = 100.0 * np.abs(y_pred_norm - y_true_norm) / (irrad_range + 1e-8)
    y_true = y_true_norm.flatten()
    nae    = nae.flatten()
    n_bins = 6
    edges  = np.linspace(y_true.min(), y_true.max(), n_bins + 1)

    log_dict = {
        "nae/mean_overall":   float(np.mean(nae)),
        "nae/median_overall": float(np.median(nae)),
        "nae/p25_overall":    float(np.percentile(nae, 25)),
        "nae/p75_overall":    float(np.percentile(nae, 75)),
    }

    for i in range(n_bins):
        mask    = (y_true >= edges[i]) & (y_true < edges[i + 1])
        if i == n_bins - 1:
            mask = (y_true >= edges[i]) & (y_true <= edges[i + 1])
        bin_nae = nae[mask]
        label   = f"{edges[i]:.3f}-{edges[i+1]:.3f}"
        if len(bin_nae) > 0:
            log_dict[f"nae_bins/mean_{label}"]   = float(np.mean(bin_nae))
            log_dict[f"nae_bins/median_{label}"] = float(np.median(bin_nae))
            log_dict[f"nae_bins/p25_{label}"]    = float(np.percentile(bin_nae, 25))
            log_dict[f"nae_bins/p75_{label}"]    = float(np.percentile(bin_nae, 75))
            log_dict[f"nae_bins/count_{label}"]  = int(mask.sum())

    wandb.log(log_dict, step=step)