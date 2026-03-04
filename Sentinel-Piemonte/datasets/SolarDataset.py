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
        # (Irradiance)
        y = torch.tensor(self.pvgis.iloc[idx].values).float().unsqueeze(-1)
        
        # (Temperature)
        weather = torch.tensor(self.meteo.iloc[idx].values).float().unsqueeze(-1)
        
        class DataPacket:
            def __init__(self, pos, y, weather):
                self.pos = pos        # (1149, 2)
                self.y = y            # (1149, 1) - Target for Stage 1
                self.weather = weather # (1149, 1) - Input Channel 2
        
        return DataPacket(self.coords, y, weather)

input_data = SolarDataset("pvgis_irradiance.csv", "meteo_temp.csv")
loader = DataLoader(input_data, batch_size=32, shuffle=True)