from torch.utils.data import Dataset
import torch
import numpy as np
import xarray as xr

COORD = "coord"
TARGET = "target"

class SphericalDataset(Dataset):
    def __init__(self, grd_file_path: str):
        ds = xr.open_dataset(grd_file_path)
    
        # 1. Extract 1D arrays
        lats_1d = torch.tensor(ds['y'].values, dtype=torch.float32) 
        lons_1d = torch.tensor(ds['x'].values, dtype=torch.float32)
        
        # 2. Create a 2D meshgrid for every (lat, lon) pair
        # indexing='ij' ensures shape is [len(lats_1d), len(lons_1d)] matching 'z'
        grid_lat, grid_lon = torch.meshgrid(lats_1d, lons_1d, indexing='ij')
        
        # 3. Convert to Radians for mathematical consistency
        self.coords = torch.stack([
            grid_lon.flatten() * (np.pi / 180.0), # Longitude in [-pi, pi]
            grid_lat.flatten() * (np.pi / 180.0)  # Latitude in [-pi/2, pi/2]
        ], dim=-1)
        
        # 4. Flatten the 2D elevation matrix
        elevation = torch.tensor(ds['z'].values, dtype=torch.float32).flatten()
        
        # 5. Normalize and store original bounds for evaluation metrics!
        self.elevation_min = elevation.min()
        self.elevation_max = elevation.max()
        self.elevations = 2 * ((elevation - self.elevation_min) / 
                               (self.elevation_max - self.elevation_min)) - 1

    def __len__(self):
        return len(self.elevations)

    def __getitem__(self, idx: int):
        # Return coordinate tensor and target value
        return {COORD: self.coords[idx], TARGET: self.elevations[idx]}