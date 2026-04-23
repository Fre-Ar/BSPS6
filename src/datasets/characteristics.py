"""
Dataset characterization along four axes:
    A. Spectral complexity       -> angular power spectrum C_l
    B. Effective bandwidth       -> L_95 = smallest degree capturing 95% energy
    C. Isotropy                  -> CV of latitudinal variance
    D. Dynamic range & gradient  -> max-min, mean gradient, P99 gradient

For scalar signals (ETOPO1, CMB, ERA5) the computation is identical to
before. For RGB signals (360° panoramas) we report TWO views:
    * 'luminance'   : standard Rec.709 Y = 0.2126 R + 0.7152 G + 0.0722 B
                      (treated as the primary / comparable scalar metric)
    * 'per_channel' : R, G, B separately, so per-channel spectral and
                      isotropy structure is available for the appendix

The returned dict is shaped:
    {
      'kind': 'scalar' | 'rgb',
      'luminance': { ...same keys as scalar result... },        # rgb only
      'per_channel': {'R': {...}, 'G': {...}, 'B': {...}},      # rgb only
      '<metric>': <value>                                       # scalar only
    }
"""
from __future__ import annotations

from typing import Union

import numpy as np
import xarray as xr
import pyshtools as pysh
import matplotlib.pyplot as plt


# Rec.709 / sRGB luminance weights (linear-light assumption — fine for our
# tone-mapped LDR images in [0,1]).
REC709_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


# ---------------------------------------------------------------------------
# Scalar characterization (the original function, refactored to take an array)
# ---------------------------------------------------------------------------
def _characterize_scalar(data_2d: np.ndarray, lats: np.ndarray) -> dict:
    """Core scalar characterization. `data_2d` must be (H, W), `lats` in deg."""
    results: dict = {}

    # --- A & B: power spectrum, L_95 ---------------------------------------
    grid = pysh.SHGrid.from_array(data_2d, grid='DH')
    coeffs = grid.expand()
    power_spectrum = coeffs.spectrum()
    degrees = coeffs.degrees()

    # L_95 is computed on the AC spectrum only: exclude l=0, which is just
    # the squared global mean ("DC"). Without this, any signal with non-zero
    # mean (ERA5 ~285 K, HDRI ~0.5) has C_0 dwarf the rest and L_95 collapses
    # to 0 — meaningless. CMB literature follows the same convention of
    # reporting spectra from l=1 upward.
    ac_spectrum = power_spectrum[1:]
    total_ac = float(np.sum(ac_spectrum))
    cumulative_ac = np.cumsum(ac_spectrum)
    # argmax returns the first True index; add 1 to map back to l (since
    # ac_spectrum[i] corresponds to l = i+1).
    l_95 = int(np.argmax(cumulative_ac >= 0.95 * total_ac)) + 1

    results['degrees'] = degrees
    results['power_spectrum'] = power_spectrum
    results['L_95'] = l_95
    results['total_power'] = float(np.sum(power_spectrum))
    results['total_ac_power'] = total_ac
    results['dc_power'] = float(power_spectrum[0])

    # --- C: isotropy (latitudinal variance CV) -----------------------------
    lat_variances = np.var(data_2d, axis=1)
    results['lats'] = lats
    results['lat_variances'] = lat_variances
    mean_var = float(np.mean(lat_variances))
    results['isotropy_cv'] = (float(np.std(lat_variances)) / mean_var) if mean_var > 0 else np.nan

    # --- D: dynamic range + gradient ---------------------------------------
    results['dynamic_range'] = float(np.max(data_2d) - np.min(data_2d))
    results['min_val'] = float(np.min(data_2d))
    results['max_val'] = float(np.max(data_2d))

    dy, dx = np.gradient(data_2d)
    cos_lats = np.clip(np.cos(np.radians(lats)), 1e-5, 1.0)
    dx_scaled = dx / cos_lats[:, None]
    grad_mag = np.sqrt(dx_scaled ** 2 + dy ** 2)
    results['mean_gradient'] = float(np.mean(grad_mag))
    results['p99_gradient'] = float(np.percentile(grad_mag, 99))

    return results


def _to_luminance(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB in [0, 1] -> (H, W) luminance."""
    return np.tensordot(rgb.astype(np.float64), REC709_WEIGHTS, axes=([2], [0]))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def characterize_spherical_dataset(
    filepath: str,
    var_name: str = 'z',
    lat_name: str = 'y',
    lon_name: str = 'x',
) -> dict:
    print(f"Loading dataset: {filepath} ...")
    ds = xr.open_dataset(filepath)

    data = np.asarray(ds[var_name].values)
    lats = ds[lat_name].values if lat_name in ds else ds['lat'].values
    lons = ds[lon_name].values if lon_name in ds else ds['lon'].values   # noqa: F841
    print(f"Data shape = {data.shape}")

    if data.ndim == 2:
        print("Computing scalar characteristics ...")
        res = _characterize_scalar(data, lats)
        res['kind'] = 'scalar'
        return res

    if data.ndim == 3 and data.shape[-1] == 3:
        print("Computing RGB characteristics (luminance + per-channel) ...")
        Y = _to_luminance(data)
        res: dict = {'kind': 'rgb'}
        print("  - luminance ...")
        res['luminance'] = _characterize_scalar(Y, lats)
        res['per_channel'] = {}
        for i, name in enumerate(('R', 'G', 'B')):
            print(f"  - channel {name} ...")
            res['per_channel'][name] = _characterize_scalar(data[..., i], lats)
        return res

    raise ValueError(f"Unsupported data shape {data.shape} (expected (H,W) or (H,W,3)).")


# ---------------------------------------------------------------------------
# Pretty-printing / plotting
# ---------------------------------------------------------------------------
def _print_scalar_summary(results: dict, header: str = '') -> None:
    if header:
        print(f"\n--- {header} ---")
    print(f"Dynamic Range:      {results['dynamic_range']:.4g} "
          f"(Min: {results['min_val']:.4g}, Max: {results['max_val']:.4g})")
    print(f"Mean Gradient:      {results['mean_gradient']:.4g} per pixel")
    print(f"99th %ile Gradient: {results['p99_gradient']:.4g} per pixel (Sharpness)")
    print(f"Isotropy CV:        {results['isotropy_cv']:.4f} (Higher = more anisotropic)")
    print(f"Effective Bandwidth (L_95%): Degree {results['L_95']}")


def print_and_plot_results(results: dict) -> None:
    kind = results.get('kind', 'scalar')

    print("\n" + "=" * 50)
    print(f" DATASET CHARACTERISTICS SUMMARY  [{kind}] ")
    print("=" * 50)

    if kind == 'scalar':
        _print_scalar_summary(results)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(results['degrees'], results['power_spectrum'], color='blue', label='C_l')
        ax1.axvline(x=results['L_95'], color='red', linestyle='--', label=f'L_95 = {results["L_95"]}')
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_title("Spherical Harmonic Power Spectrum")
        ax1.set_xlabel("Degree (l)"); ax1.set_ylabel("Power (C_l)")
        ax1.grid(True, which="both", ls="--", alpha=0.5); ax1.legend()

        ax2.plot(results['lat_variances'], results['lats'], color='green')
        ax2.set_title("Latitudinal Variance (Anisotropy Check)")
        ax2.set_xlabel("Variance across Longitudes"); ax2.set_ylabel("Latitude (Degrees)")
        ax2.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        return

    # kind == 'rgb'
    _print_scalar_summary(results['luminance'], header='LUMINANCE (Rec.709)')
    for name in ('R', 'G', 'B'):
        _print_scalar_summary(results['per_channel'][name], header=f'CHANNEL {name}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'luminance': 'black', 'R': 'red', 'G': 'green', 'B': 'blue'}
    for label, res in [('luminance', results['luminance'])] + [
            (ch, results['per_channel'][ch]) for ch in ('R', 'G', 'B')]:
        ax1.plot(res['degrees'], res['power_spectrum'],
                 color=colors[label], label=f'{label} (L95={res["L_95"]})',
                 alpha=0.9 if label == 'luminance' else 0.6)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_title("Spherical Harmonic Power Spectrum (RGB)")
    ax1.set_xlabel("Degree (l)"); ax1.set_ylabel("Power (C_l)")
    ax1.grid(True, which="both", ls="--", alpha=0.5); ax1.legend()

    lum = results['luminance']
    ax2.plot(lum['lat_variances'], lum['lats'], color='black', label='luminance')
    for ch in ('R', 'G', 'B'):
        r = results['per_channel'][ch]
        ax2.plot(r['lat_variances'], r['lats'], color=colors[ch], alpha=0.6, label=ch)
    ax2.set_title("Latitudinal Variance (Anisotropy, RGB)")
    ax2.set_xlabel("Variance across Longitudes"); ax2.set_ylabel("Latitude (Degrees)")
    ax2.grid(True, ls="--", alpha=0.5); ax2.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    
def _characterize_spherical_dataset_old(filepath: str, var_name='z', lat_name='y', lon_name='x'):
    print(f"Loading dataset: {filepath}...")
    ds = xr.open_dataset(filepath)
    
    # Extract the 2D data array and 1D coordinate arrays
    data = ds[var_name].values
    lats = ds[lat_name].values if lat_name in ds else ds['lat'].values
    lons = ds[lon_name].values if lon_name in ds else ds['lon'].values
    
    print("Computing characteristics...")
    results = {}

    # ---------------------------------------------------------
    # A & B: Spectral Complexity and Effective Bandwidth
    # ---------------------------------------------------------
    # Create a Spherical Harmonic Grid object.
    # 'DH' (Driscoll and Healy) is the standard type for equally spaced lat/lon grids.
    grid = pysh.SHGrid.from_array(data, grid='DH')
    
    # Expand the spatial grid into Spherical Harmonic coefficients
    coeffs = grid.expand()
    
    # Get the power per degree l (C_l)
    power_spectrum = coeffs.spectrum()
    degrees = coeffs.degrees()
    
    # Calculate Effective Bandwidth (L_95%)
    total_power = np.sum(power_spectrum)
    cumulative_power = np.cumsum(power_spectrum)
    l_95 = np.argmax(cumulative_power >= 0.95 * total_power)
    
    results['degrees'] = degrees
    results['power_spectrum'] = power_spectrum
    results['L_95'] = l_95
    results['total_power'] = total_power

    # ---------------------------------------------------------
    # C: Isotropy (Latitudinal Variance)
    # ---------------------------------------------------------
    # Calculate the variance of the signal along each latitude band
    # axis=1 computes variance across the longitudes (x-axis)
    lat_variances = np.var(data, axis=1)
    results['lats'] = lats
    results['lat_variances'] = lat_variances
    
    # For a single summary metric, we can use the Coefficient of Variation of the variances.
    # High CV means highly anisotropic (variance changes wildly depending on latitude).
    results['isotropy_cv'] = np.std(lat_variances) / np.mean(lat_variances)

    # ---------------------------------------------------------
    # D: Dynamic Range and Spatial Gradient
    # ---------------------------------------------------------
    results['dynamic_range'] = np.max(data) - np.min(data)
    results['min_val'] = np.min(data)
    results['max_val'] = np.max(data)
    
    # Calculate 2D gradients (dy, dx)
    dy, dx = np.gradient(data)
    
    # Correct the longitudinal gradient (dx) for converging meridians
    # lats are in degrees, convert to radians
    lats_rad = np.radians(lats)
    cos_lats = np.cos(lats_rad)
    # Clip to avoid division by zero exactly at the poles (-90, 90)
    cos_lats = np.clip(cos_lats, 1e-5, 1.0) 
    
    # Broadcast cos_lats to match 2D data shape
    cos_lats_2d = cos_lats[:, np.newaxis]
    
    # Scale longitudinal gradient 
    dx_scaled = dx / cos_lats_2d
    
    # Calculate gradient magnitude vector
    grad_magnitude = np.sqrt(dx_scaled**2 + dy**2)
    
    results['mean_gradient'] = np.mean(grad_magnitude)
    results['p99_gradient'] = np.percentile(grad_magnitude, 99)

    return results

def _print_and_plot_results_old(results):
    # Print the summary table
    print("\n" + "="*50)
    print(" DATASET CHARACTERISTICS SUMMARY ")
    print("="*50)
    print(f"Dynamic Range:      {results['dynamic_range']:.2f} (Min: {results['min_val']:.2f}, Max: {results['max_val']:.2f})")
    print(f"Mean Gradient:      {results['mean_gradient']:.2f} per pixel")
    print(f"99th %ile Gradient: {results['p99_gradient']:.2f} per pixel (Sharpness)")
    print("-" * 50)
    print(f"Isotropy CV:        {results['isotropy_cv']:.4f} (Higher = more anisotropic)")
    print("-" * 50)
    print(f"Effective Bandwidth (L_95%): Degree {results['L_95']}")
    print("="*50 + "\n")

    # Plot 1: Power Spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results['degrees'], results['power_spectrum'], color='blue', label='Power (C_l)')
    ax1.axvline(x=results['L_95'], color='red', linestyle='--', label=f'L_95% = {results["L_95"]}')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title("Spherical Harmonic Power Spectrum")
    ax1.set_xlabel("Degree (l)")
    ax1.set_ylabel("Power (C_l)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # Plot 2: Latitudinal Variance (Isotropy)
    # We plot latitude on the Y-axis to visually match a map
    ax2.plot(results['lat_variances'], results['lats'], color='green')
    ax2.set_title("Latitudinal Variance (Anisotropy Check)")
    ax2.set_xlabel("Variance across Longitudes")
    ax2.set_ylabel("Latitude (Degrees)")
    ax2.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


