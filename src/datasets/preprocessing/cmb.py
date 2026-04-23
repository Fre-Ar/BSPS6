"""
Planck CMB (SMICA, Full mission) pre-processor.

Reads the Planck Legacy Archive FITS file:

    COM_CMB_IQU-smica_2048_R3.00_full.fits

which is a HEALPix full-sky map at Nside=2048 containing Stokes parameters
I, Q, U in Kelvin_CMB. We use only I (temperature fluctuations), down-grade
the HEALPix Nside if desired, then bilinearly interpolate to the standard
512x1024 equirectangular grid.

Notes
-----
* The file is ~2GB on disk; healpy.read_map memory-maps it, so reading only
  I is cheap. We apply astropy-style masking (any UNSEEN pixels become NaN
  before interpolation).
* The Planck SMICA CMB map is in HEALPix RING-ordered by default. We force
  `nest=False` when reading to be explicit.
* Nside=2048 has 5e7 pixels; interpolating onto 512x1024 lat/lon is cheap.
  We optionally `ud_grade` down to Nside=512 first to save memory and to
  smooth out sub-resolution noise before the equirectangular projection.
"""
from __future__ import annotations

import numpy as np
import healpy as hp

from .common import _standard_grid, save_standardized, sanity_check_standardized


def preprocess_cmb(
    input_filepath: str,
    output_filepath: str,
    n_lat: int = 512,
    n_lon: int = 1024,
    intermediate_nside: int | None = 512,
) -> None:
    """
    Parameters
    ----------
    input_filepath : path to COM_CMB_IQU-smica_2048_R3.00_full.fits
    output_filepath : path for the standardized 512x1024 NetCDF
    intermediate_nside : if set, ud_grade the HEALPix map to this Nside
        before equirect projection. None keeps the native Nside=2048.
        Default 512 (sufficient to capture all angular scales resolvable
        on a 512x1024 equirect grid, ~Nyquist).
    """
    print(f"[CMB] loading {input_filepath} ...")
    # field=0 -> intensity/temperature (Stokes I); 1 -> Q; 2 -> U.
    cmb_I = hp.read_map(input_filepath, field=0, nest=False)

    cmb_I = np.asarray(cmb_I, dtype=np.float64)
    # Replace UNSEEN sentinel (-1.6375e30) with NaN.
    UNSEEN = hp.UNSEEN if hasattr(hp, 'UNSEEN') else -1.6375e30
    cmb_I = np.where(cmb_I <= 0.5 * UNSEEN, np.nan, cmb_I)

    native_nside = hp.get_nside(cmb_I)
    print(f"[CMB] native Nside = {native_nside} ({12*native_nside**2} pixels)")

    if intermediate_nside is not None and intermediate_nside < native_nside:
        print(f"[CMB] ud_grade to Nside={intermediate_nside}")
        cmb_I = hp.ud_grade(cmb_I, nside_out=intermediate_nside, order_in='RING')

    # Build target lat/lon grid (degrees -> radians conversion for healpy).
    new_lats, new_lons = _standard_grid(n_lat, n_lon)
    grid_lat, grid_lon = np.meshgrid(new_lats, new_lons, indexing='ij')  # (H, W) each

    # healpy uses (theta, phi) in radians. theta = colatitude in [0, pi] (north
    # pole -> 0), phi = longitude in [0, 2*pi).
    theta = np.deg2rad(90.0 - grid_lat)   # colatitude
    phi   = np.deg2rad(grid_lon % 360.0)  # longitude in [0, 2pi)

    print(f"[CMB] interpolating to {n_lat}x{n_lon} equirectangular grid ...")
    values = hp.get_interp_val(cmb_I, theta.ravel(), phi.ravel(), nest=False)
    signal = values.reshape(n_lat, n_lon).astype(np.float32)

    # Any residual NaN (shouldn't happen for full-sky SMICA, but be safe):
    n_nan = int(np.isnan(signal).sum())
    if n_nan:
        print(f"[CMB] warning: {n_nan} NaN pixels after interp; filling with mean")
        signal = np.where(np.isnan(signal), np.nanmean(signal), signal)

    save_standardized(
        output_filepath,
        lats_deg=new_lats,
        lons_deg=new_lons,
        signal=signal,
        extra_attrs={
            'source': 'Planck 2018 SMICA Full-mission CMB I-Stokes (ESA PLA)',
            'units': 'K_CMB (Kelvin, CMB thermodynamic temperature)',
            'native_nside': str(native_nside),
            'intermediate_nside': str(intermediate_nside),
            'preprocess': (
                f'read Stokes I from FITS, ud_grade to Nside={intermediate_nside}, '
                f'bilinear HEALPix->equirect interp on {n_lat}x{n_lon}'
            ),
        },
    )
    sanity_check_standardized(output_filepath)
    print(f"[CMB] wrote {output_filepath}")


