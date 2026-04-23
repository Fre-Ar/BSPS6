"""
Unified spherical regression dataset.

One class for ETOPO1 / ERA5 / CMB / HDRI. The dataset contract is defined
by `src/datasets/preprocessing/common.py`: an xarray.Dataset with ds['y']
(lats deg), ds['x'] (lons deg), ds['z'] signal of shape (H, W) or (H, W, 3).

The coordinate encoding (angular / cartesian / SH / spherical-RFF) is
applied once at load time; we keep the full flat tensors in memory, which
is fine at 512x1024 = 524,288 samples, even SH at L_max=32 is ~2.3 GB.
"""
from __future__ import annotations

from typing import Optional

import torch
import xarray as xr
from torch.utils.data import Dataset

from config.constants import COORD, TARGET, CE_TYPES
from .coord_encodings import (
    angular_encoding,
    cartesian_encoding,
    spherical_harmonics_encoding,
    spherical_rff_encoding,
)


# Map CE_TYPES -> encoding fn.
_CE_MAPPING = {
    'angular':             angular_encoding,
    'cartesian':           cartesian_encoding,
    'spherical-harmonics': spherical_harmonics_encoding,
    'spherical-rff':       spherical_rff_encoding,
}


def _resolve_encoding(name: str):
    if name in _CE_MAPPING:
        return _CE_MAPPING[name]
    raise ValueError(f"Unknown coordinate encoding '{name}'. "
                     f"Available: {list(_CE_MAPPING)}")


class SphericalDataset(Dataset):
    """Flat (N, D_coord) <-> (N, C) mapping, already encoded.

    Extra per-encoding hyperparameters (e.g. L_max for SH; num_features,
    sigma, seed for RFF) are passed via `encoding_kwargs` and forwarded to
    the encoding function. They're also stashed on `self.encoding_kwargs`
    so downstream code can reconstruct the encoding deterministically.
    """

    def __init__(
        self,
        grd_file_path: str,
        coordinate_encoding: CE_TYPES = 'angular',
        encoding_kwargs: Optional[dict] = None,
    ):
        ds = xr.open_dataset(grd_file_path)
        encoding_kwargs = dict(encoding_kwargs or {})

        encoding_fn = _resolve_encoding(coordinate_encoding)
        self.coords, self.targets, self.target_min, self.target_max = \
            encoding_fn(ds, **encoding_kwargs)

        # Metadata
        self.num_channels: int = int(self.targets.shape[-1])  # 1 or 3
        self.coord_dim: int = int(self.coords.shape[-1])
        self.height: int = int(ds.sizes.get('y', ds['y'].size))
        self.width:  int = int(ds.sizes.get('x', ds['x'].size))
        self.file_path: str = grd_file_path
        self.coordinate_encoding: str = coordinate_encoding
        self.encoding_kwargs: dict = encoding_kwargs

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        return {COORD: self.coords[idx], TARGET: self.targets[idx]}