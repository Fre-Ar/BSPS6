"""
ETOPO1 pre-processor.

Reads the NOAA ETOPO1 ice-surface GRD file and down-samples it to
the standard 512x1024 equirectangular grid used across the benchmark.
"""
from __future__ import annotations
 
import numpy as np
import xarray as xr
 
from .common import (
    _standard_grid, _held_out_grid,
    save_standardized, sanity_check_standardized,
)


def _interp_etopo1_to_grid(ds, lat_name, lon_name, new_lats, new_lons, label):
    """Bilinearly interpolate the (already-oriented) ETOPO1 source onto
    (new_lats, new_lons). Fails loudly on any NaN cells."""
    src_lat_min = float(ds[lat_name].min()); src_lat_max = float(ds[lat_name].max())
    src_lon_min = float(ds[lon_name].min()); src_lon_max = float(ds[lon_name].max())
    print(f"[ETOPO1] {label}: interpolating to {len(new_lats)}x{len(new_lons)} "
          f"(extrapolation enabled at edges)...")
    ds_ds = ds.interp(
        {lat_name: new_lats, lon_name: new_lons},
        method='linear',
        kwargs={'fill_value': 'extrapolate'},
    )
    signal = ds_ds['z'].values.astype(np.float32)
    n_nan = int(np.isnan(signal).sum())
    if n_nan:
        raise RuntimeError(
            f"[ETOPO1] {label}: {n_nan} NaN pixels after interpolation "
            f"({100*n_nan/signal.size:.3f}% of grid). "
            f"Source dims: lat[{src_lat_min},{src_lat_max}], "
            f"lon[{src_lon_min},{src_lon_max}]. "
            f"Target dims: lat[{new_lats[0]:.4f},{new_lats[-1]:.4f}], "
            f"lon[{new_lons[0]:.4f},{new_lons[-1]:.4f}]."
        )
    return signal
 
 
def preprocess_etopo1(
    input_filepath: str,
    output_filepath: str,
    held_out_filepath: str | None = None,
    n_lat: int = 512,
    n_lon: int = 1024,
) -> None:
    """Pre-process ETOPO1 onto the standard (n_lat × n_lon) training grid,
    and, if `held_out_filepath` is given, ALSO onto the half-pixel-offset
    ((n_lat-1) × (n_lon-1)) grid — both interpolated directly from the
    NOAA source, never from each other."""
    
    print(f"[ETOPO1] loading {input_filepath} ...")
    ds = xr.open_dataset(input_filepath)
 
    # ETOPO1 grd: longitude='x', latitude='y'. Some copies use 'lat'/'lon'.
    lat_name = 'y' if 'y' in ds.dims else 'lat'
    lon_name = 'x' if 'x' in ds.dims else 'lon'
 
    # ETOPO1 .grd files from NOAA typically store latitudes in DESCENDING
    # order (+90 -> -90). xarray.interp wants the source coord monotonically
    # ascending to reliably avoid NaN fall-throughs, so reindex defensively.
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.reindex({lat_name: ds[lat_name][::-1]})
    if ds[lon_name].values[0] > ds[lon_name].values[-1]:
        ds = ds.reindex({lon_name: ds[lon_name][::-1]})
 
    src_lat_min = float(ds[lat_name].min()); src_lat_max = float(ds[lat_name].max())
    src_lon_min = float(ds[lon_name].min()); src_lon_max = float(ds[lon_name].max())
    print(f"[ETOPO1] source lat in [{src_lat_min:.4f},{src_lat_max:.4f}], "
          f"lon in [{src_lon_min:.4f},{src_lon_max:.4f}]")
    
    # Source-derived normalization reference: SphericalDataset uses these so
    # both training and held-out targets normalize to [-1, 1] consistently
    # (see preregistration §3.4).
    source_z = np.asarray(ds['z'].values, dtype=np.float64)
    target_min = np.array([float(np.nanmin(source_z))], dtype=np.float32)
    target_max = np.array([float(np.nanmax(source_z))], dtype=np.float32)
    print(f"[ETOPO1] source signal in [{target_min[0]:.4g}, {target_max[0]:.4g}]")
 
    # ---- 1. Standard training grid ----
    new_lats, new_lons = _standard_grid(n_lat, n_lon)
    signal = _interp_etopo1_to_grid(ds, lat_name, lon_name,
                                    new_lats, new_lons, label='train')
    save_standardized(
        output_filepath,
        lats_deg=new_lats,
        lons_deg=new_lons,
        signal=signal,
        extra_attrs={
            'source': 'NOAA ETOPO1 ice surface, gmt4',
            'units': 'metres (elevation)',
            'preprocess': (
                f'ascending reindex, bilinear interp to {n_lat}x{n_lon}, '
                f'linear extrapolation enabled at edges'
            ),
        },
        target_min=target_min,
        target_max=target_max,
    )
    sanity_check_standardized(output_filepath)
    print(f"[ETOPO1] wrote training file {output_filepath}")

    # ---- 2. Half-pixel-offset held-out grid (from same source) ----
    if held_out_filepath:
        held_lats, held_lons = _held_out_grid(n_lat, n_lon)
        signal_held = _interp_etopo1_to_grid(
            ds, lat_name, lon_name, held_lats, held_lons, label='held_out',
        )
        save_standardized(
            held_out_filepath,
            lats_deg=held_lats,
            lons_deg=held_lons,
            signal=signal_held,
            extra_attrs={
                'source': 'NOAA ETOPO1 ice surface, gmt4',
                'units': 'metres (elevation)',
                'preprocess': (
                    f'ascending reindex, bilinear interp to '
                    f'{n_lat-1}x{n_lon-1} half-pixel-offset grid from SOURCE'
                ),
                'grid_role': 'held_out',
                'train_shape': f'{n_lat}x{n_lon}',
            },
            target_min=target_min,
            target_max=target_max,
        )
        sanity_check_standardized(held_out_filepath)
        print(f"[ETOPO1] wrote held-out file {held_out_filepath}")