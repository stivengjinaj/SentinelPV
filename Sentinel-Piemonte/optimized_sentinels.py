import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial import KDTree
import tqdm

from model import Model
from dataset import PiedmontSolarDataset

NUM_SENTINELS = 10
EPOCHS = 50
LEARNING_RATE = 1e-2
STAGE_1_CHECKPOINT = "./checkpoints/physense_transolver_ep300.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load 1,149 coordinates and the power plants
all_coords = np.load("coords.npy")                     # Shape (1149, 2)
plant_coords = np.load("actual_power_plants.npy")      # to be defined

# Snap the continuous coordinates to real power plants
tree = KDTree(plant_coords)

train_dataset = PiedmontSolarDataset("irradiance_train.npy", "weather_train.npy", "coords.npy")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Differentiable sampling (Replaces Bilinear Interpolation)
def sample_field_idw(query_pos, grid_pos, grid_values, power=2.0, eps=1e-6):
    """
    Inverse Distance Weighting: Estimates the irradiance/weather for the 
    floating sentinel by calculating the weighted average of nearby points.
    """
    # query_pos: (B, Num_Sentinels, 2), grid_pos: (B, 1149, 2), grid_values: (B, 1149, 1)
    dist = torch.cdist(query_pos, grid_pos) # (B, Num_Sentinels, 1149)
    weights = 1.0 / (dist ** power + eps)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Multiply weights by grid values to get the sampled values at the query pos
    sampled_values = torch.bmm(weights, grid_values)
    return sampled_values

model = Model(space_dim=2, out_dim=1, n_layers=12, n_hidden=374, slice_num=32)
model.load_state_dict(torch.load(STAGE_1_CHECKPOINT, map_location=device))
model.eval()

# Freeze all main model Parameters
for param in model.parameters():
    param.requires_grad = False

# Random initialization on valid power plants
init_indices = np.random.choice(len(plant_coords), NUM_SENTINELS, replace=False)
init_sentinels = torch.tensor(plant_coords[init_indices], dtype=torch.float32).to(device)

# Make sentinel coordinates trainable
sentinel_pos = nn.Parameter(init_sentinels, requires_grad=True)

# Optimizer updates only the sentinel coordinates
optimizer = optim.Adam([sentinel_pos], lr=LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda')

print(f"Starting Stage 2: Searching the {NUM_SENTINELS} best sentinels...")

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    
    for batch_data in tqdm.tqdm(train_loader):
        pos = batch_data.pos.to(device)         # (B, 1149, 2)
        y = batch_data.y.to(device)             # (B, 1149, 1)
        weather = batch_data.weather.to(device) # (B, 1149, 1)
        
        batch_size = pos.shape[0]
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Noise to Target
            noise = torch.randn_like(y)
            u = torch.normal(mean=0.0, std=1.0, size=(1,)).to(device)
            t = torch.sigmoid(u).unsqueeze(-1).repeat(batch_size, y.shape[1], 1)
            
            y_t = t * y + (1. - t) * noise
            target = y - noise
            
            # Differentiable Sampling for Sentinels
            sentinel_batch_pos = sentinel_pos.unsqueeze(0).repeat(batch_size, 1, 1)
            sampled_y = sample_field_idw(sentinel_batch_pos, pos, y)
            
            field_combined = torch.cat((y_t, weather), dim=-1)
            x = torch.concat((pos, field_combined), dim=-1) # (B, 1149, 4)
            fx = model.preprocess(x) + model.placeholder[None, None, :]
            t_emb = model.t_embedder(t[:, 0, 0]).squeeze()
            
            sensor_feature = torch.concat((sentinel_batch_pos, sampled_y), dim=-1)
            s = model.sensor_encoder(sensor_feature)
            s_2 = model.sensor_encoder_2(sensor_feature)
            t_emb = t_emb + s_2.mean(dim=1).squeeze()
            
            x_out = model.transformer(fx, t_emb, s)
            predict_v = model.mlp_head(x_out, t_emb)[0]
            
            loss = nn.MSELoss()(predict_v, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Project gradiend descend
        with torch.no_grad():
            cpu_pos = sentinel_pos.cpu().numpy()
            distances, indices = tree.query(cpu_pos)
            snapped_pos = plant_coords[indices]
            # Override the floating tensor with the snapped physical locations
            sentinel_pos.copy_(torch.tensor(snapped_pos).to(device))

        epoch_loss += loss.item()

    print(f"Epoch {epoch}/{EPOCHS} | Optimization Loss: {epoch_loss/len(train_loader):.5f}")

print("\n OPTIMIZATION COMPLETE!")
print("Here are your best Sentinel Power Plant Coordinates:")
final_sentinels = sentinel_pos.detach().cpu().numpy()
for i, coords in enumerate(final_sentinels):
    print(f"Sentinel {i+1}: Lat {coords[0]:.4f}, Lon {coords[1]:.4f}")