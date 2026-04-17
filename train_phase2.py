import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from datasets.IrradianceDataset import IrradianceDataset
from models.transolver_pv import IrradianceModel

NUM_SENTINELS = 100
EPOCHS        = 50
LR            = 0.25
BATCH_SIZE    = 64
STAGE1_CKPT   = "./training_history/train_pvgis2005_2022_100sentinels/irradiance_stage1_final.pth"
IRRAD_PATH    = "datasets/irradiance_train.npy"
COORDS_PATH   = "datasets/coords.npy"
RESULTS_DIR   = "./training_history/train_pvgis2005_2022_100sentinels"

WANDB_PROJECT  = "physense-irradiance"
WANDB_ENTITY   = "stivengjinaj-politecnico-di-torino"
WANDB_RUN_NAME = "sentinelpv-stage2"

TELEGRAM_TOKEN = "8647539434:AAGQ4Ik9OVVEd0Z0QhlDBHpAyTjnrIUmTms"
TELEGRAM_CHAT_ID = "6694449067"


def tg_notify(msg: str):
    if not TELEGRAM_TOKEN:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception:
        pass


# Differentiable sampling: Inverse distance weighting
def sample_idw(
    query_pos:   torch.Tensor,   # (B, S, 2)
    grid_pos:    torch.Tensor,   # (B, N, 2)
    grid_values: torch.Tensor,   # (B, N, 1)
    power: float = 2.0,
    eps:   float = 1e-6,
) -> torch.Tensor:
    """
    Inverse-Distance Weighting interpolation.
    Estimates irradiance at floating sentinel positions as a weighted average
    of neighbouring panel values. Fully differentiable w.r.t. query_pos.

    Returns: sampled (B, S, 1)
    """
    dist    = torch.cdist(query_pos, grid_pos)             # (B, S, N)
    weights = 1.0 / (dist ** power + eps)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalise
    return torch.bmm(weights, grid_values)                 # (B, S, 1)


# Flow-matching loss at given sentinel positions
def flow_loss_for_sentinels(
    model:        IrradianceModel,
    batch:        dict,
    sentinel_pos: torch.Tensor,
    device:       torch.device,
) -> torch.Tensor:

    pos   = batch['pos'].to(device)    # (B, N, 2)
    y     = batch['y'].to(device)      # (B, N, 1)
    h_sun = batch['h_sun'].to(device)  # (B, N, 1)
    B, N, _ = y.shape

    # Flow interpolation on irradiance only
    u       = torch.randn(B, device=device)
    t       = torch.sigmoid(u)
    t_bc    = t.view(B, 1, 1)
    noise   = torch.randn_like(y)
    y_t     = t_bc * y + (1. - t_bc) * noise
    target  = y - noise                               # (B, N, 1)

    ref_d = model._ref_grid_distances(pos)
    x     = torch.cat([pos, y_t, h_sun, ref_d], dim=-1)   # (B, N, 20)
    fx    = model.preprocess(x) + model.placeholder[None, None, :]

    s_pos_batch = sentinel_pos.unsqueeze(0).expand(B, -1, -1)
    s_y  = sample_idw(s_pos_batch, pos, y)             # (B, S, 1)
    s_h  = sample_idw(s_pos_batch, pos, h_sun)         # (B, S, 1)

    sensor_feat = torch.cat([s_pos_batch, s_y, s_h], dim=-1)  # (B, S, 4)
    s    = model.sensor_encoder(sensor_feat)
    s2   = model.sensor_encoder_2(sensor_feat)

    t_emb = model.t_embedder(t) + s2.mean(dim=1)
    x_out = model.transformer(fx, t_emb, s)
    pred  = model.mlp_head(x_out, t_emb) # (B, N, 1)

    return nn.functional.mse_loss(pred, target)   # irradiance loss only


# Projection: snap each sentinel to the nearest real panel
def project_to_grid(sentinel_pos: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    For each sentinel, find the nearest panel in the grid and return that coord.
    sentinel_pos : (S, 2)   grid : (N, 2)   — both on same device
    """
    with torch.no_grad():
        dists   = torch.cdist(sentinel_pos, grid)   # (S, N)
        nearest = dists.argmin(dim=-1)              # (S,)
        return grid[nearest]                        # (S, 2)


def optimise():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = IrradianceDataset(IRRAD_PATH, COORDS_PATH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True)
    N  = dataset.pos_tensor.shape[0]
    T  = len(dataset)
    print(f"Dataset: {T} timesteps | {N} panels")

    all_pos = dataset.pos_tensor.to(device)   # (N, 2) normalised

    # Frozen Stage 1 model
    model = IrradianceModel(
        space_dim=2, fun_dim=1, out_dim=1,
        n_layers=12, n_hidden=374, slice_num=32,
    ).to(device)
    model.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Stage 1 checkpoint loaded: {STAGE1_CKPT}")

    # Initialise sentinels at random panel positions
    init_idx     = torch.randperm(N)[:NUM_SENTINELS]
    sentinel_pos = nn.Parameter(all_pos[init_idx].clone())   # (S, 2)
    optimizer    = optim.Adam([sentinel_pos], lr=LR)
    scaler       = torch.amp.GradScaler('cuda')

    wandb.init(
        project = WANDB_PROJECT,
        entity  = WANDB_ENTITY,
        name    = WANDB_RUN_NAME,
        config  = {
            "num_sentinels":    NUM_SENTINELS,
            "epochs":           EPOCHS,
            "learning_rate":    LR,
            "batch_size":       BATCH_SIZE,
            "stage1_ckpt":      STAGE1_CKPT,
            "n_panels":         N,
            "n_timesteps":      T,
            "idw_power":        2.0,
        },
        tags = ["stage2", "sentinel-optimisation", "pgd"],
    )

    raw_coords = np.load(COORDS_PATH)
    init_table = wandb.Table(columns=["sentinel_id", "lat", "lon"])
    init_idx_cpu = init_idx.cpu().numpy()
    for i, idx in enumerate(init_idx_cpu):
        ll = raw_coords[idx]
        init_table.add_data(i + 1, float(ll[0]), float(ll[1]))
    wandb.log({"sentinels/initial_positions": init_table})

    tg_notify(f"Stage 2 started — {NUM_SENTINELS} sentinels.")
    print(f"\nStarting Stage 2: optimising {NUM_SENTINELS} sentinel positions...\n")

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = flow_loss_for_sentinels(model, batch, sentinel_pos, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_([sentinel_pos], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                snapped = project_to_grid(sentinel_pos.data, all_pos)
                sentinel_pos.data.copy_(snapped)

            epoch_loss  += loss.item()
            global_step += 1

            # Per-step
            wandb.log({
                "opt/step_loss":  loss.item(),
                "opt/grad_norm":  grad_norm.item(),
            }, step=global_step)

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch:>3}/{EPOCHS} | Flow loss: {avg:.6f}")

        # Per-epoch
        wandb.log({
            "opt/epoch_loss": avg,
            "epoch":          epoch,
        }, step=global_step)

        if epoch % 10 == 0:
            tg_notify(f"Stage 2 epoch {epoch}/{EPOCHS} | loss: {avg:.6f}")

            # Resolve current normalised positions back to lat/lon
            pos_np   = dataset.pos_tensor.numpy()
            cur_norm = sentinel_pos.detach().cpu().numpy()
            dists_np = np.linalg.norm(pos_np[None] - cur_norm[:, None], axis=-1)
            cur_idx  = dists_np.argmin(axis=-1)
            cur_ll   = raw_coords[cur_idx]

            scatter_table = wandb.Table(columns=["sentinel_id", "lat", "lon", "epoch"])
            for i, ll in enumerate(cur_ll):
                scatter_table.add_data(i + 1, float(ll[0]), float(ll[1]), epoch)
            wandb.log({f"sentinels/positions_ep{epoch}": scatter_table}, step=global_step)

    # Results
    pos_np    = dataset.pos_tensor.numpy()
    final_pos = sentinel_pos.detach().cpu().numpy()
    dists_np  = np.linalg.norm(pos_np[None] - final_pos[:, None], axis=-1)
    best_idx  = dists_np.argmin(axis=-1)
    latlon    = raw_coords[best_idx]

    out_path = os.path.join(RESULTS_DIR, "sentinel_panels.npy")
    np.save(out_path, latlon)

    final_table = wandb.Table(columns=["sentinel_id", "panel_index", "lat", "lon"])
    for i, (ll, idx) in enumerate(zip(latlon, best_idx)):
        final_table.add_data(i + 1, int(idx), float(ll[0]), float(ll[1]))
    wandb.log({"sentinels/final_positions": final_table})
    wandb.summary["final_opt_loss"]  = avg
    wandb.summary["num_sentinels"]   = NUM_SENTINELS
    wandb.summary["result_path"]     = out_path
    wandb.save(out_path)

    print("\n" + "=" * 55)
    print(f"  Optimisation complete — {NUM_SENTINELS} sentinel panels found")
    print("=" * 55)
    for i, (ll, idx) in enumerate(zip(latlon, best_idx)):
        print(f"  Sentinel {i+1:>2}: panel #{idx}  lat={ll[0]:.5f}  lon={ll[1]:.5f}")
    print(f"\n  Saved to: {out_path}")

    tg_notify(f"Stage 2 complete. Sentinels saved to {out_path}.")
    wandb.finish()
    return latlon


if __name__ == "__main__":
    optimise()