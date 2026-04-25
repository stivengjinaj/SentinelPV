import os
import time

import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.IrradianceDataset import IrradianceDataset
from models.transolver_pv import IrradianceModel

EPOCHS      = 50
BATCH_SIZE  = 128
LR          = 3e-4
SAVE_DIR    = "./training_history/train_pvgis2005_20022_30sentinels_50epochs"
IRRAD_PATH  = "./training_history/train_pvgis2005_20022_30sentinels_50epochs/dataset/irradiance_train.npy"
COORDS_PATH = "./training_history/train_pvgis2005_20022_30sentinels_50epochs/dataset/coords.npy"

WANDB_PROJECT  = "physense-irradiance"
WANDB_ENTITY   = "stivengjinaj-politecnico-di-torino"
WANDB_RUN_NAME = "train_pvgis2005_2022_30sentinels_50epochs-stage1"

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


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = IrradianceDataset(IRRAD_PATH, COORDS_PATH)

    mean_irr     = dataset.y_data.mean(dim=1).squeeze()          # (T,)
    bins         = torch.quantile(mean_irr, torch.linspace(0, 1, 5))
    bin_ids      = torch.bucketize(mean_irr, bins[1:-1])          # 4 buckets: 0,1,2,3
    class_counts = torch.bincount(bin_ids)
    weights      = 1.0 / class_counts[bin_ids].float()
    sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=4, pin_memory=True)
    
    N = dataset.pos_tensor.shape[0]
    T = len(dataset)
    print(f"Dataset: {T} timesteps | {N} panels")

    model = IrradianceModel(
        space_dim = 2,
        fun_dim   = 1,
        out_dim   = 1,
        n_layers  = 12,
        n_hidden  = 374,
        slice_num = 32,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    run = wandb.init(
        project = WANDB_PROJECT,
        entity  = WANDB_ENTITY,
        name    = WANDB_RUN_NAME,
        config  = {
            # training
            "epochs":           EPOCHS,
            "batch_size":       BATCH_SIZE,
            "learning_rate":    LR,
            "lr_warmup_epochs": 10,
            "grad_clip":        1.0,
            # model
            "n_layers":         12,
            "n_hidden":         374,
            "slice_num":        32,
            "ref_grid":         4,
            "fun_dim":          1,
            "space_dim":        2,
            # data
            "n_panels":         N,
            "n_timesteps":      T,
        },
        tags = ["stage1", "flow-matching", "irradiance"],
    )
    

    wandb.watch(model, log="gradients", log_freq=100)

    # Optimiser & scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)
    cosine    = CosineAnnealingLR(optimizer, T_max=EPOCHS - 10, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])
    scaler    = torch.amp.GradScaler('cuda')

    tg_notify("Stage 1 training started.")
    print("Starting Stage 1 training...")

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        skipped    = 0
        t0         = time.time()

        for batch in loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss = model(batch_gpu)

            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            global_step += 1

            # Per-step log: loss and gradient norm
            wandb.log({
                "train/step_loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
            }, step=global_step)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        n_valid  = max(len(loader) - skipped, 1)
        avg_loss = epoch_loss / n_valid
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:>4}/{EPOCHS} | Loss: {avg_loss:.5f} | "
              f"LR: {current_lr:.2e} | Skipped: {skipped} | Time: {elapsed:.1f}s")

        wandb.log({
            "train/epoch_loss":       avg_loss,
            "train/learning_rate":    current_lr,
            "train/skipped_batches":  skipped,
            "train/epoch_time_s":     elapsed,
            "epoch":                  epoch,
        }, step=global_step)

        # Checkpoint every 50 epochs
        if epoch % 50 == 0:
            ckpt = os.path.join(SAVE_DIR, f"irradiance_stage1_ep{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            wandb.save(ckpt)
            tg_notify(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.5f} | Time: {elapsed:.1f}s")
            print(f"  Checkpoint saved: {ckpt}")

    final_ckpt = os.path.join(SAVE_DIR, "irradiance_stage1_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    wandb.save(final_ckpt)

    wandb.summary["final_train_loss"] = avg_loss
    wandb.summary["total_epochs"]     = EPOCHS
    wandb.summary["total_steps"]      = global_step

    tg_notify(f"Stage 1 complete. Final loss: {avg_loss:.5f}")
    print(f"\nDone. Final checkpoint: {final_ckpt}")
    wandb.finish()


if __name__ == "__main__":
    train()