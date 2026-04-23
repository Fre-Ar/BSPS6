"""
ETOPO1 pre-processor.

Reads the NOAA ETOPO1 ice-surface GRD file and down-samples it to
the standard 512x1024 equirectangular grid used across the benchmark.
"""
from __future__ import annotations
 
import numpy as np
import xarray as xr
 
from .common import _standard_grid, save_standardized, sanity_check_standardized
 
 
def preprocess_etopo1(
    input_filepath: str,
    output_filepath: str,
    n_lat: int = 512,
    n_lon: int = 1024,
) -> None:
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
 
    # Diagnostic: print source bounds so any future mismatch is obvious.
    src_lat_min = float(ds[lat_name].min()); src_lat_max = float(ds[lat_name].max())
    src_lon_min = float(ds[lon_name].min()); src_lon_max = float(ds[lon_name].max())
    print(f"[ETOPO1] source lat in [{src_lat_min:.4f},{src_lat_max:.4f}], "
          f"lon in [{src_lon_min:.4f},{src_lon_max:.4f}]")
 
    # Build the canonical target grid.
    new_lats, new_lons = _standard_grid(n_lat, n_lon)
 
    # The target grid hits lat=-90 and lat=+90 exactly. Source files are
    # often off by a few ULPs (e.g. -89.99999999999997) due to float drift
    # in the upstream pipeline; even a ~3e-15 overshoot makes xr.interp
    # silently NaN-fill the whole pole row. So we unconditionally enable
    # scipy's linear extrapolation — a no-op for strictly-interior points
    # and a smooth edge extension otherwise.
    print(f"[ETOPO1] interpolating to {n_lat}x{n_lon} "
          f"(this may take a minute, extrapolation enabled at edges)...")
    ds_ds = ds.interp(
        {lat_name: new_lats, lon_name: new_lons},
        method='linear',
        kwargs={'fill_value': 'extrapolate'},
    )
 
    signal = ds_ds['z'].values.astype(np.float32)  # (H, W)
 
    # Hard fail-fast: if *any* NaN remains, surface the diagnosis loudly
    # instead of silently writing junk.
    n_nan = int(np.isnan(signal).sum())
    if n_nan:
        raise RuntimeError(
            f"[ETOPO1] {n_nan} NaN pixels after interpolation "
            f"({100*n_nan/signal.size:.3f}% of grid). "
            f"Source dims: lat[{src_lat_min},{src_lat_max}], "
            f"lon[{src_lon_min},{src_lon_max}]. "
            f"Target dims: lat[-90,90], lon[-180,180). "
            f"Check coordinate conventions of the source file."
        )
 
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
    )
    sanity_check_standardized(output_filepath)
    print(f"[ETOPO1] wrote {output_filepath}")