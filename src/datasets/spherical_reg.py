from torch.utils.data import Dataset
import torch
import numpy as np
import xarray as xr
from typing import Literal
from config.constants import COORD, TARGET, CE_TYPES
from coord_encodings import angular_encoding, cartesian_encoding

CE_MAPPING = {
    'angular': angular_encoding,
    'cartesian': cartesian_encoding,
    
}

class SphericalDataset(Dataset):
    def __init__(self, grd_file_path: str, coordinate_encoding: CE_TYPES):
        ds = xr.open_dataset(grd_file_path)
        
        encoding_fn = CE_MAPPING.get(coordinate_encoding)
        self.coords, self.targets, self.target_min, self.target_max = encoding_fn(ds)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        # Return coordinate tensor and target value
        return {COORD: self.coords[idx], TARGET: self.targets[idx]}