import torch
import numpy as np
import xarray as xr

def angular_encoding(ds: xr.Dataset):
    
    # 1. Extract 1D arrays
    lats_1d = torch.tensor(ds['y'].values, dtype=torch.float32) 
    lons_1d = torch.tensor(ds['x'].values, dtype=torch.float32)
    
    # 2. Create a 2D meshgrid for every (lat, lon) pair
    # indexing='ij' ensures shape is [len(lats_1d), len(lons_1d)] matching 'z'
    grid_lat, grid_lon = torch.meshgrid(lats_1d, lons_1d, indexing='ij')
    
    # 3. Convert to Radians for mathematical consistency
    coords = torch.stack([
        grid_lon.flatten() * (np.pi / 180.0), # Longitude in [-pi, pi]
        grid_lat.flatten() * (np.pi / 180.0)  # Latitude in [-pi/2, pi/2]
    ], dim=-1)
    
    # 4. Flatten the 2D target values matrix
    target = torch.tensor(ds['z'].values, dtype=torch.float32).flatten()
    
    # 5. Normalize and store original bounds for evaluation metrics!
    target_min = target.min()
    target_max = target.max()
    targets = 2 * ((target - target_min) / 
                            (target_max - target_min)) - 1
    # Reshape from [N] to [N, 1]
    targets = targets.unsqueeze(-1)
    
    return coords, targets, target_min, target_max
    
def cartesian_encoding(ds: xr.Dataset):
    coords, targets, target_min, target_max = angular_encoding(ds)
    
    # Convert (lat, lon) to Cartesian (x, y, z)
    lat = coords[:, 1]  # Latitude in radians
    lon = coords[:, 0]  # Longitude in radians
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    
    coords_cartesian = torch.stack([x, y, z], dim=-1)
    
    return coords_cartesian, targets, target_min, target_max