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
import numpy as np
import xarray as xr
from torch.utils.data import Dataset

from config.constants import COORD, TARGET, CE_TYPES
from .coord_encodings import (
    angular_encoding,
    cartesian_encoding,
    spherical_harmonics_encoding,
    compute_coords,
)


# Map CE_TYPES -> encoding fn.
_CE_MAPPING = {
    'angular':             angular_encoding,
    'cartesian':           cartesian_encoding,
    'spherical-harmonics': spherical_harmonics_encoding,
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
        held_out_file_path: Optional[str] = None,
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
        
        # Per-axis lat/lon arrays in degrees (length H and W respectively).
        # Used by breakdown code to recover per-pixel latitude for polar /
        # equatorial band masks.
        self.lats_deg: np.ndarray = np.asarray(ds['y'].values)
        self.lons_deg: np.ndarray = np.asarray(ds['x'].values)
        
        self.file_path: str = grd_file_path
        self.held_out_file_path: Optional[str] = held_out_file_path
        self.coordinate_encoding: str = coordinate_encoding
        self.encoding_kwargs: dict = encoding_kwargs
        
        # Lazy-initialized held-out cache
        self._held_out_eval: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._held_out_lats_deg: Optional[np.ndarray] = None
        self._held_out_lons_deg: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        return {COORD: self.coords[idx], TARGET: self.targets[idx]}
    
    
    # ----- Held-out evaluation -----
    def make_held_out_eval(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (coords, targets) for the (H-1)×(W-1) half-pixel-offset grid.

        Per preregistration §3.4 (v0.4), ground-truth values at the offset
        positions are read directly from a SEPARATELY preprocessed
        `*_511x1023_held_out.nc` file (path supplied via
        `held_out_file_path` at construction). That file is produced by the
        per-dataset preprocessor by interpolating the BASE SOURCE onto the
        511×1023 grid — never from the 512×1024 training file. The
        training-grid bilinear interpolation that earlier drafts used is
        retired.

        Coords use the same coordinate encoding and the same kwargs
        (including SRFF seed) as the training data, so the held-out coords
        inhabit the identical feature space the model trained on — only the
        spatial positions differ.

        Targets are normalized to [-1, 1] using the BASE-SOURCE `target_min` /
        `target_max` (written to the training file's NetCDF attrs by the
        preprocessor and carried through `SphericalDataset.__init__`), not
        the held-out signal's own min/max. The same reference is used during
        training, so the model's outputs and the held-out targets are on the
        same scale and both fall within [-1, 1] regardless of which extremes
        each spatial sub-sample happens to capture.

        Cached on first call; subsequent calls are O(1).
        """
        if self._held_out_eval is not None:
            return self._held_out_eval

        if not self.held_out_file_path:
            raise RuntimeError(
                "SphericalDataset.make_held_out_eval() needs a "
                "held_out_file_path (the 511×1023 file produced by the "
                "per-dataset preprocessor). Pass --held_out_path to main.py "
                "or set DATASET_CONFIG[<dataset>]['held_out_path'] and "
                "re-run preprocessing with "
                "`python -m datasets.preprocess --dataset <name>`."
            )

        with xr.open_dataset(self.held_out_file_path) as ds_held:
            lats_held = np.asarray(ds_held['y'].values, dtype=np.float32)
            lons_held = np.asarray(ds_held['x'].values, dtype=np.float32)
            z_held = np.asarray(ds_held['z'].values, dtype=np.float32)

        # Sanity checks: the file must be exactly the (H-1)×(W-1) shape we
        # expect, and consistent with the training file's channel count.
        expected_h, expected_w = self.height - 1, self.width - 1
        if z_held.shape[0] != expected_h or z_held.shape[1] != expected_w:
            raise RuntimeError(
                f"held-out file {self.held_out_file_path} has shape "
                f"{z_held.shape[:2]} but the training file is "
                f"{self.height}×{self.width}, so we expected "
                f"({expected_h}, {expected_w})."
            )

        if z_held.ndim == 2:
            z_held = z_held[..., None]
        elif z_held.ndim != 3:
            raise ValueError(f"Unsupported held-out target shape {z_held.shape}.")
        Hh, Wh, C = z_held.shape
        assert C == self.num_channels, (
                f"held-out channels {C} != training channels "
                f"{self.num_channels} (file: {self.held_out_file_path})"
            )

        # Encode the offset positions using the SAME encoding pipeline as
        # training. For SRFF, this re-uses the same ω draw (same seed).
        coords_held = compute_coords(
            self.coordinate_encoding,
            lats_held, lons_held,
            **self.encoding_kwargs,
        )

        # Normalize targets using TRAINING min/max (not held-out min/max).
        targets_held = torch.from_numpy(z_held.reshape(Hh * Wh, C))
        denom = (self.target_max - self.target_min).clamp(min=1e-8)
        targets_held = 2.0 * ((targets_held - self.target_min) / denom) - 1.0

        # Stash the offset lat/lon arrays so per-band breakdown code can
        # reconstruct the spatial mapping without re-opening the file.
        self._held_out_lats_deg = lats_held
        self._held_out_lons_deg = lons_held
        self._held_out_eval = (coords_held, targets_held)
        return self._held_out_eval

    @property
    def held_out_height(self) -> int:
        return self.height - 1

    @property
    def held_out_width(self) -> int:
        return self.width - 1
