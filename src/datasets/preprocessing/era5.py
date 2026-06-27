"""
ERA5 pre-processor.

Reads a CDS-downloaded ERA5 single-levels NetCDF containing 2m temperature
(variable short-name 't2m' or standard-name '2m_temperature') and re-samples
the field onto the standard 512x1024 equirectangular grid.

Assumptions about the raw file (as delivered by the CDS web UI for a single
hour of reanalysis):
  - dims include ('latitude', 'longitude'); may also include 'time' and
    'valid_time' (CDS-Beta format). If a time dim of size > 1 exists, we take
    the first time slice.
  - latitudes descend from +90 to -90 (CDS convention). We flip if needed.
  - longitudes are in [0, 360). We roll to [-180, 180) to match our grid.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .common import (
    _standard_grid, _held_out_grid,
    save_standardized, sanity_check_standardized,
)

_T2M_CANDIDATES = ('t2m', '2m_temperature', 'air_temperature_2m', 't2')


def _pick_variable(ds: xr.Dataset) -> str:
    for name in _T2M_CANDIDATES:
        if name in ds.data_vars:
            return name
    # Fall back: first 2D-or-higher float variable.
    for name, var in ds.data_vars.items():
        if var.ndim >= 2 and np.issubdtype(var.dtype, np.floating):
            return name
    raise KeyError(f"No 2m-temperature variable found. "
                   f"Tried {_T2M_CANDIDATES}, available: {list(ds.data_vars)}")


def _rename_dims(ds: xr.Dataset) -> xr.Dataset:
    """Normalize coordinate names to ('lat', 'lon')."""
    rename = {}
    for cand_lat in ('latitude', 'lat', 'y'):
        if cand_lat in ds.dims:
            rename[cand_lat] = 'lat'
            break
    for cand_lon in ('longitude', 'lon', 'x'):
        if cand_lon in ds.dims:
            rename[cand_lon] = 'lon'
            break
    if rename:
        ds = ds.rename(rename)
    return ds

def _interp_era5_to_grid(da, new_lats, new_lons, label):
    """Bilinearly interpolate the (already-oriented) ERA5 DataArray onto
    (new_lats, new_lons). Returns a 2D float32 signal."""
    print(f"[ERA5] {label}: interpolating to {len(new_lats)}x{len(new_lons)} ...")
    da_ds = da.interp(lat=new_lats, lon=new_lons, method='linear')
    signal = da_ds.values.astype(np.float32)
    if signal.ndim != 2:
        raise RuntimeError(f"[ERA5] {label}: expected 2D signal, got {signal.shape}")
    return signal

def preprocess_era5(
    input_filepath: str,
    output_filepath: str,
    held_out_filepath: str | None = None,
    n_lat: int = 512,
    n_lon: int = 1024,
    time_index: int = 0,
) -> None:
    """Pre-process ERA5 onto the standard (n_lat × n_lon) training grid,
    and, if `held_out_filepath` is given, ALSO onto the half-pixel-offset
    ((n_lat-1) × (n_lon-1)) grid — both interpolated directly from the
    CDS source NetCDF, never from each other."""
    
    print(f"[ERA5] loading {input_filepath} ...")
    ds = xr.open_dataset(input_filepath)
    ds = _rename_dims(ds)

    var_name = _pick_variable(ds)
    print(f"[ERA5] selected variable: {var_name}")

    da = ds[var_name]

    # Handle extra leading dims (time / valid_time / number / step).
    for extra in list(da.dims):
        if extra in ('lat', 'lon'):
            continue
        if da.sizes[extra] > 1:
            print(f"[ERA5] reducing dim '{extra}' (size {da.sizes[extra]}) "
                  f"by isel index={time_index}")
        da = da.isel({extra: min(time_index, da.sizes[extra] - 1)})

    # Ensure ascending latitude (ERA5 comes 90..-90).
    if da['lat'].values[0] > da['lat'].values[-1]:
        da = da.reindex(lat=da['lat'][::-1])

    # Convert longitudes from [0, 360) to [-180, 180) if necessary.
    lons = da['lon'].values
    if lons.max() > 180.0:
        da = da.assign_coords(lon=(((lons + 180.0) % 360.0) - 180.0))
        da = da.sortby('lon')
        
    # Source-derived normalization reference (preregistration §3.4).
    source_z = np.asarray(da.values, dtype=np.float64)
    target_min = np.array([float(np.nanmin(source_z))], dtype=np.float32)
    target_max = np.array([float(np.nanmax(source_z))], dtype=np.float32)
    print(f"[ERA5] source signal in [{target_min[0]:.4g}, {target_max[0]:.4g}]")

    # ---- 1. Standard training grid ----
    new_lats, new_lons = _standard_grid(n_lat, n_lon)
    signal = _interp_era5_to_grid(da, new_lats, new_lons, label='train')
    save_standardized(
        output_filepath,
        lats_deg=new_lats,
        lons_deg=new_lons,
        signal=signal,
        extra_attrs={
            'source': 'ECMWF ERA5 single-levels reanalysis (via Copernicus CDS)',
            'variable': var_name,
            'units': 'Kelvin (for 2m temperature)',
            'preprocess': f'bilinear interp to {n_lat}x{n_lon}, '
                          f'lon rolled to [-180,180), lat ascending',
            'time_index_used': str(time_index),
            'grid_role': 'train',
        },
        target_min=target_min,
        target_max=target_max,
    )
    sanity_check_standardized(output_filepath)
    print(f"[ERA5] wrote training file {output_filepath}")
    
    # ---- 2. Half-pixel-offset held-out grid (from same source) ----
    if held_out_filepath:
        held_lats, held_lons = _held_out_grid(n_lat, n_lon)
        signal_held = _interp_era5_to_grid(da, held_lats, held_lons,
                                           label='held_out')
        save_standardized(
            held_out_filepath,
            lats_deg=held_lats,
            lons_deg=held_lons,
            signal=signal_held,
            extra_attrs={
                'source': 'ECMWF ERA5 single-levels reanalysis (via Copernicus CDS)',
                'variable': var_name,
                'units': 'Kelvin (for 2m temperature)',
                'preprocess': (
                    f'bilinear interp to {n_lat-1}x{n_lon-1} half-pixel-offset '
                    f'grid from SOURCE'
                ),
                'time_index_used': str(time_index),
                'grid_role': 'held_out',
                'train_shape': f'{n_lat}x{n_lon}',
            },
            target_min=target_min,
            target_max=target_max,
        )
        sanity_check_standardized(held_out_filepath)
        print(f"[ERA5] wrote held-out file {held_out_filepath}")
