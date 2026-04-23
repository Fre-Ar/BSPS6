"""
Shared helpers for dataset preprocessors.

All pre-processors in this submodule produce a standardized NetCDF file:

    ds['y']  : 1D latitudes in degrees, length = BENCH_LAT   (default 512)
    ds['x']  : 1D longitudes in degrees, length = BENCH_LON  (default 1024)
    ds['z']  : 2D array of shape (BENCH_LAT, BENCH_LON) for scalar signals,
               or 3D array of shape (BENCH_LAT, BENCH_LON, 3) for RGB.
"""
from __future__ import annotations

import os
from typing import Optional
import numpy as np
import xarray as xr


def _standard_grid(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the canonical equirectangular grid.

    Latitudes: n_lat points in [-90, 90], endpoints inclusive.
    Longitudes: n_lon points in [-180, 180), endpoints *exclusive on the right*
        (we drop the +180 duplicate of -180 to avoid degenerate samples).
    """
    lats = np.linspace(-90.0, 90.0, n_lat, dtype=np.float32)
    lons = np.linspace(-180.0, 180.0, n_lon, endpoint=False, dtype=np.float32)
    return lats, lons

def save_standardized(
    out_path: str,
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    signal: np.ndarray,
    extra_attrs: Optional[dict] = None,
) -> None:
    """
    Persist a signal to disk as a standardized NetCDF.

    Parameters
    ----------
    out_path : str
        Destination .nc file.
    lats_deg : (H,) float array of latitudes in degrees.
    lons_deg : (W,) float array of longitudes in degrees.
    signal   : (H, W) scalar or (H, W, 3) RGB array (float32 recommended).
    extra_attrs : optional dict of metadata attached to ds.attrs (source,
        timestamp, tone-mapping params, etc.).
    """
    signal = np.asarray(signal, dtype=np.float32)
    H, W = len(lats_deg), len(lons_deg)

    if signal.shape[:2] != (H, W):
        raise ValueError(
            f"signal first two dims {signal.shape[:2]} do not match "
            f"(len(lats), len(lons)) = ({H}, {W})"
        )

    if signal.ndim == 2:
        ds = xr.Dataset(
            data_vars={'z': (('y', 'x'), signal)},
            coords={'y': lats_deg.astype(np.float32),
                    'x': lons_deg.astype(np.float32)},
        )
    elif signal.ndim == 3 and signal.shape[2] == 3:
        ds = xr.Dataset(
            data_vars={'z': (('y', 'x', 'c'), signal)},
            coords={'y': lats_deg.astype(np.float32),
                    'x': lons_deg.astype(np.float32),
                    'c': np.array([0, 1, 2], dtype=np.int32)},
        )
    else:
        raise ValueError(
            f"Unsupported signal shape {signal.shape}; expected (H,W) or (H,W,3)."
        )

    if extra_attrs:
        ds.attrs.update({k: str(v) for k, v in extra_attrs.items()})

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # Small engine-agnostic encoding: fall back to default if netCDF4 unavailable.
    ds.to_netcdf(out_path)


def sanity_check_standardized(path: str) -> None:
    """Raise if a saved standardized file doesn't satisfy the pipeline contract."""
    ds = xr.open_dataset(path)
    assert 'x' in ds.coords and 'y' in ds.coords, "missing x/y coords"
    assert 'z' in ds.data_vars, "missing z data variable"
    z = ds['z'].values
    assert z.ndim in (2, 3), f"z must be 2D or 3D, got {z.ndim}D"
    if z.ndim == 3:
        assert z.shape[2] == 3, f"3D z must have 3 channels, got {z.shape[2]}"
    lats = ds['y'].values
    lons = ds['x'].values
    assert lats.min() >= -90.0 and lats.max() <= 90.0, "latitudes out of range"
    assert lons.min() >= -180.0 and lons.max() <= 180.0, "longitudes out of range"
    print(f"[sanity-check] {path}: z={z.shape} "
          f"lat in [{lats.min():.2f},{lats.max():.2f}] "
          f"lon in [{lons.min():.2f},{lons.max():.2f}] "
          f"val in [{float(z.min()):.4g},{float(z.max()):.4g}]")