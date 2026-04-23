"""
Coordinate encodings for signals on S^2.

Each encoding function accepts the standardized xarray.Dataset produced by
the preprocessing pipeline (ds['y']=lats deg, ds['x']=lons deg, ds['z']=
signal of shape (H, W) or (H, W, C)) and returns:

    coords      : (N, D_coord) float32 tensor
    targets     : (N, C)       float32 tensor in [-1, 1] (per-channel min-max)
    target_min  : (C,)         float32 tensor (original per-channel min)
    target_max  : (C,)         float32 tensor (original per-channel max)

where N = H*W, and C = 1 for scalar signals or 3 for RGB.

Encodings implemented:
    angular(ds)                                  -> D_coord = 2
    cartesian(ds)                                -> D_coord = 3
    spherical_harmonics(ds, L_max)               -> D_coord = (L_max+1)**2
    spherical_rff(ds, num_features, sigma, seed) -> D_coord = 2 * num_features

Convention notes:
    * Latitude in [-90, +90] deg; colatitude = 90 - lat.
    * Longitude in [-180, +180) deg.
    * SH uses REAL, 4π-orthonormal Y_lm with no Condon-Shortley phase:
        ∫_S² Y_lm(θ,φ) Y_l'm'(θ,φ) sinθ dθ dφ = δ_ll' δ_mm'
      Concretely:
        Y_l^0(θ,φ)    =          Ȳ_l^0(cosθ)
        Y_l^m(θ,φ)    = √2 ·    Ȳ_l^m(cosθ) · cos(mφ)        for m > 0
        Y_l^{-m}(θ,φ) = √2 ·    Ȳ_l^m(cosθ) · sin(mφ)        for m > 0
      with Ȳ_l^m = √((2l+1)(l-m)! / (4π (l+m)!)) · P_l^m.
    * Spherical-RFF applies RFF to the CARTESIAN embedding (x,y,z) of the
      sphere (not to (λ,φ)). This approximates a Gaussian kernel in R^3
      restricted to S^2 — hence "approximately geodesic-aware".
"""
from __future__ import annotations

import numpy as np
import torch
import xarray as xr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_targets(ds: xr.Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten ds['z'] to (N, C), compute per-channel min/max, return
    (targets_normalized_in_[-1,1], target_min, target_max).
    """
    z = np.asarray(ds['z'].values, dtype=np.float32)
    if z.ndim == 2:
        z = z[..., None]   # (H, W, 1)
    elif z.ndim != 3:
        raise ValueError(f"Unsupported target shape {z.shape}; expected (H,W) or (H,W,C).")
    C = z.shape[-1]
    H, W = z.shape[:2]

    target = torch.from_numpy(z.reshape(H * W, C))                     # (N, C)
    target_min = target.amin(dim=0)                                    # (C,)
    target_max = target.amax(dim=0)                                    # (C,)
    denom = (target_max - target_min).clamp(min=1e-8)
    targets = 2.0 * ((target - target_min) / denom) - 1.0              # (N, C)
    return targets, target_min, target_max


def _lat_lon_meshgrid_radians(ds: xr.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (grid_lat_rad, grid_lon_rad), each (H, W) float32."""
    lats_deg = torch.tensor(ds['y'].values, dtype=torch.float32)
    lons_deg = torch.tensor(ds['x'].values, dtype=torch.float32)
    grid_lat_deg, grid_lon_deg = torch.meshgrid(lats_deg, lons_deg, indexing='ij')
    deg2rad = float(np.pi / 180.0)
    return grid_lat_deg * deg2rad, grid_lon_deg * deg2rad


def _cartesian_grid(ds: xr.Dataset) -> torch.Tensor:
    """Unit-sphere Cartesian embedding (N, 3) float32."""
    grid_lat, grid_lon = _lat_lon_meshgrid_radians(ds)
    cos_lat = torch.cos(grid_lat)
    x = cos_lat * torch.cos(grid_lon)
    y = cos_lat * torch.sin(grid_lon)
    z = torch.sin(grid_lat)
    return torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)


# ---------------------------------------------------------------------------
# Normalized associated Legendre functions (geodesy / 4π-real convention).
#
# Computes Ȳ_l^m(x) = √((2l+1)(l-m)! / (4π(l+m)!)) · P_l^m(x)
# for all 0 ≤ m ≤ l ≤ L_max, at input points x ∈ [-1, 1] (x = cos θ = sin φ).
#
# Uses the standard stable 3-term recurrence seeded by sectoral harmonics:
#     Ȳ_0^0(x)     = 1 / √(4π)
#     Ȳ_m^m(x)    = -√((2m+1)/(2m)) · √(1-x²) · Ȳ_{m-1}^{m-1}(x)    [m ≥ 1]
#     Ȳ_{m+1}^m(x) = √(2m+3) · x · Ȳ_m^m(x)
#     Ȳ_l^m(x)    = a_lm · x · Ȳ_{l-1}^m(x) − b_lm · Ȳ_{l-2}^m(x)   [l ≥ m+2]
# with a_lm = √((2l+1)(2l-1) / ((l+m)(l-m))),
#      b_lm = √((2l+1)(l-m-1)(l+m-1) / ((2l-3)(l+m)(l-m))).
#
# Storage: flat triangular, index (l, m) -> l*(l+1)//2 + m, for 0 ≤ m ≤ l.
# ---------------------------------------------------------------------------
def _normalized_associated_legendre(L_max: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    num_entries = (L_max + 1) * (L_max + 2) // 2
    out = np.zeros((N, num_entries), dtype=np.float64)

    # √(1 - x²) is |sin θ| on our domain — clamp against tiny negatives.
    sqrt_one_minus_x2 = np.sqrt(np.clip(1.0 - x * x, 0.0, None))

    inv_sqrt_4pi = 1.0 / np.sqrt(4.0 * np.pi)
    out[:, 0] = inv_sqrt_4pi

    # Sectoral: Ȳ_m^m for m = 1..L_max
    for m in range(1, L_max + 1):
        prev_idx = (m - 1) * m // 2 + (m - 1)
        idx = m * (m + 1) // 2 + m
        out[:, idx] = (
            -np.sqrt((2.0 * m + 1.0) / (2.0 * m))
            * sqrt_one_minus_x2 * out[:, prev_idx]
        )

    # First sub-diagonal: Ȳ_{m+1}^m, m = 0..L_max-1
    for m in range(0, L_max):
        idx_m = m * (m + 1) // 2 + m
        idx_m1 = (m + 1) * (m + 2) // 2 + m
        out[:, idx_m1] = np.sqrt(2.0 * m + 3.0) * x * out[:, idx_m]

    # Main recurrence: l = m+2..L_max
    for m in range(0, L_max - 1):
        for l in range(m + 2, L_max + 1):
            a = np.sqrt((2.0 * l + 1.0) * (2.0 * l - 1.0)
                        / ((l + m) * (l - m)))
            b = np.sqrt((2.0 * l + 1.0) * (l - m - 1.0) * (l + m - 1.0)
                        / ((2.0 * l - 3.0) * (l + m) * (l - m)))
            out[:, l * (l + 1) // 2 + m] = (
                a * x * out[:, (l - 1) * l // 2 + m]
                - b * out[:, (l - 2) * (l - 1) // 2 + m]
            )

    return out


def _sh_features(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    L_max: int,
) -> np.ndarray:
    """Compute (H, W, (L_max+1)²) real 4π-orthonormal SH features on a lat/lon grid.

    Feature ordering (flat index): for l=0..L_max, m=−l..+l — so feature 0 is Y_0^0,
    features 1..3 are Y_1^{-1}, Y_1^0, Y_1^1, etc.

    Efficient because lat and lon factor: Ȳ_l^m depends only on sin(lat), and
    the longitudinal part is cos(mφ) or sin(mφ). We compute each factor once
    and outer-product them; O(H · L²) + O(W · L) ops rather than O(H W L²).
    """
    lats_rad = np.deg2rad(np.asarray(lats_deg, dtype=np.float64))
    lons_rad = np.deg2rad(np.asarray(lons_deg, dtype=np.float64))
    H, W = len(lats_rad), len(lons_rad)

    # Ȳ_l^m(sin lat) for all (l, m) with 0 ≤ m ≤ l, at each of the H latitudes.
    plm_table = _normalized_associated_legendre(L_max, np.sin(lats_rad))   # (H, (L+1)(L+2)/2)

    # cos(m φ), sin(m φ) for m = 0..L_max at each of W longitudes.
    m_arr = np.arange(L_max + 1, dtype=np.float64)
    m_phi = m_arr[:, None] * lons_rad[None, :]                             # (L_max+1, W)
    cos_mp = np.cos(m_phi)                                                 # (L_max+1, W)
    sin_mp = np.sin(m_phi)                                                 # (L_max+1, W)

    num_features = (L_max + 1) ** 2
    sh = np.empty((H, W, num_features), dtype=np.float32)
    sqrt2 = np.sqrt(2.0)

    feat_idx = 0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            plm_col = plm_table[:, l * (l + 1) // 2 + abs(m)]              # (H,)
            if m == 0:
                # Y_l^0 = Ȳ_l^0 (no √2 factor)
                sh[:, :, feat_idx] = plm_col[:, None].astype(np.float32)
            elif m > 0:
                # Y_l^m = √2 · Ȳ_l^m · cos(mφ)
                sh[:, :, feat_idx] = (
                    sqrt2 * plm_col[:, None] * cos_mp[m, None, :]
                ).astype(np.float32)
            else:
                # Y_l^{-|m|} = √2 · Ȳ_l^{|m|} · sin(|m|φ)
                sh[:, :, feat_idx] = (
                    sqrt2 * plm_col[:, None] * sin_mp[-m, None, :]
                ).astype(np.float32)
            feat_idx += 1

    return sh


# ---------------------------------------------------------------------------
# Public encoding entry points
# ---------------------------------------------------------------------------

def angular_encoding(ds: xr.Dataset):
    """Naive (longitude, latitude) in radians. Input dim = 2."""
    grid_lat, grid_lon = _lat_lon_meshgrid_radians(ds)
    coords = torch.stack([grid_lon.flatten(),    # longitude in [-π, π)
                          grid_lat.flatten()],   # latitude  in [-π/2, π/2]
                         dim=-1)                 # (N, 2)

    targets, target_min, target_max = _extract_targets(ds)
    return coords, targets, target_min, target_max


def cartesian_encoding(ds: xr.Dataset):
    """Unit-sphere embedding (x, y, z) in R^3. Input dim = 3."""
    coords = _cartesian_grid(ds)                 # (N, 3)
    targets, target_min, target_max = _extract_targets(ds)
    return coords, targets, target_min, target_max


def spherical_harmonics_encoding(ds: xr.Dataset, L_max: int = 32):
    """
    Real 4π-orthonormal spherical harmonic features up to degree L_max.

    Input dim = (L_max + 1)². At L_max=32 that's 1089 features per sample.
    Bandlimited by construction: any signal whose SH power beyond L_max is
    significant cannot be represented regardless of network capacity.
    """
    if L_max < 0:
        raise ValueError(f"L_max must be ≥ 0, got {L_max}.")
    lats_deg = np.asarray(ds['y'].values)
    lons_deg = np.asarray(ds['x'].values)
    sh = _sh_features(lats_deg, lons_deg, L_max)                # (H, W, (L+1)²)
    H, W, D = sh.shape
    coords = torch.from_numpy(sh.reshape(H * W, D))             # (N, D)

    targets, target_min, target_max = _extract_targets(ds)
    return coords, targets, target_min, target_max


def spherical_rff_encoding(
    ds: xr.Dataset,
    num_features: int = 128,
    sigma: float = 8.0,
    seed: int = 42,
):
    """
    Spherical Random Fourier Features: apply RFF to Cartesian (x, y, z).

    ω_i ~ N(0, σ² I₃) drawn ONCE at construction (seeded), then each sample
    is encoded as

        [cos(ω_1 · x), …, cos(ω_m · x), sin(ω_1 · x), …, sin(ω_m · x)]

    Input dim = 2 · num_features. At m=128 that's 256 features per sample.
    Approximates a Gaussian kernel exp(-|x-x'|² / (2σ⁻²)) in R³, which on
    the sphere is monotonically related to geodesic distance — hence
    "approximately geodesic-aware".
    """
    if num_features <= 0:
        raise ValueError(f"num_features must be > 0, got {num_features}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    x_cart = _cartesian_grid(ds)                                # (N, 3)

    # Draw ω once with a reproducible generator. float32 throughout to avoid
    # matmul dtype promotion downstream.
    g = torch.Generator().manual_seed(int(seed))
    omegas = torch.randn(num_features, 3, generator=g,
                         dtype=torch.float32) * float(sigma)    # (m, 3)

    proj = x_cart @ omegas.T                                    # (N, m)
    coords = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 2m)

    targets, target_min, target_max = _extract_targets(ds)
    return coords, targets, target_min, target_max


# ---------------------------------------------------------------------------
# Encoding-dim lookup
# ---------------------------------------------------------------------------
def coord_encoding_dim(name: str, **kwargs) -> int:
    """Return the flat coordinate-feature dim for a given encoding + kwargs."""
    if name == 'angular':
        return 2
    if name == 'cartesian':
        return 3
    if name == 'spherical-harmonics':
        L_max = int(kwargs.get('L_max', 32))
        return (L_max + 1) ** 2
    if name == 'spherical-rff':
        m = int(kwargs.get('num_features', 128))
        return 2 * m
    raise ValueError(f"Unknown encoding '{name}'.")