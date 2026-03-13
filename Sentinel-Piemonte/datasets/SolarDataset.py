import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SolarDataset(Dataset):
    def __init__(self, pvgis_csv, meteo_csv):
        # pvgis_csv: rows=dates, cols=irradiance for 1149 points
        # meteo_csv: rows=dates, cols=temp for 1149 points
        # coords_csv: index=ID, cols=[lat, lon]
        self.pvgis = pd.read_csv(pvgis_csv, index_col=0)
        self.meteo = pd.read_csv(meteo_csv, index_col=0)
        
        self.coords = torch.tensor(pd.read_csv("coords_1149.csv")[['lat', 'lon']].values).float()

    def __len__(self):
        return len(self.pvgis)

    def __getitem__(self, idx):
        y = torch.tensor(self.pvgis.iloc[idx].values).float().unsqueeze(-1)
        weather = torch.tensor(self.meteo.iloc[idx].values).float().unsqueeze(-1)
        return {'pos': self.coords, 'y': y, 'weather': weather}

input_data = SolarDataset("pvgis_irradiance.csv", "meteo_temp.csv")
loader = DataLoader(input_data, batch_size=32, shuffle=True)