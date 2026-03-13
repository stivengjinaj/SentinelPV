import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import requests

from models.single_transolver_pv import Model
from datasets.SingleSolarDataset import SingleSolarDataset 

EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
SAVE_DIR = "./checkpoints"

TELEGRAM_TOKEN = "8647539434:AAGQ4Ik9OVVEd0Z0QhlDBHpAyTjnrIUmTms"
TELEGRAM_CHAT_ID = "6694449067"

def tg_notify(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception:
        pass

def train():
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")

        train_dataset = SingleSolarDataset(
            y_path="datasets/power_train.npy", 
            coords_path="datasets/coords.npy"
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        model = Model(
            space_dim=2, 
            out_dim=1, 
            fun_dim=1,      
            n_layers=12,    
            n_hidden=374,   
            slice_num=32    
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        scaler = torch.amp.GradScaler('cuda')
        # autocast_ctx = torch.amp.autocast('cuda')

        print("Starting Flow Matching Training...")

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()

            for batch in train_loader:
                optimizer.zero_grad()
                
                batch_gpu = {k: v.to(device) for k, v in batch.items()}

                with torch.amp.autocast('cuda'):
                    loss = model(batch_gpu) 

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            epoch_time = time.time() - start_time

            print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.5f} | Time: {epoch_time:.2f}s")

            if epoch % 50 == 0:
                msg = f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.5f} | Time: {epoch_time:.2f}s"
                tg_notify(msg)
                checkpoint_path = os.path.join(SAVE_DIR, f"physense_transolver_ep{epoch}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        print("Training Completed")
        tg_notify("Training Completed")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"There was an error:\n{error_details}")
        tg_notify(f"There was an error: {e}")

def main():
    train()

if __name__ == "__main__":
    main()