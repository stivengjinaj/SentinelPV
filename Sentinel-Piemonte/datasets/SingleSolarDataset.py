import torch
import numpy as np
from torch.utils.data import Dataset

class SingleSolarDataset(Dataset):
    def __init__(self, y_path, coords_path):
        self.y_data = np.load(y_path)
        self.coords = np.load(coords_path)
        
        self.pos_tensor = torch.tensor(self.coords).float()

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        y = torch.tensor(self.y_data[idx]).float()
        
        return {
            'pos': self.pos_tensor,
            'y': y
        }