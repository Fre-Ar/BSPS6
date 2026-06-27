"""
Integration test for the standardized-dataset pipeline + coordinate encodings.

Each per-dataset test is skipped if the corresponding pre-processed file is
missing, so you can iteratively bring datasets online. Encoding tests run
against the first available dataset; they don't require all four.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.constants import DATASET_CHOICES, DATASET_CONFIG, CE_CHOICES   
from datasets.spherical_reg import SphericalDataset                        
from datasets.coord_encodings import (                                     
    _normalized_associated_legendre,
    _sh_features,
    coord_encoding_dim,
)


# Per-encoding default kwargs used by the integration loop. Kept small where
# possible so SH doesn't balloon to gigabytes during local testing.
_ENCODING_KWARGS_FOR_TEST = {
    'angular':             {},
    'cartesian':           {},
    'spherical-harmonics': {'L_max': 8},     # (8+1)^2 = 81 features
    'spherical-rff':       {'num_features': 32, 'sigma': 8.0, 'seed': 0},
}


def _file_exists(dataset: str) -> bool:
    return os.path.exists(DATASET_CONFIG[dataset]['path'])


def _check_ds(ds_path: str, expected_channels: int) -> None:
    ds = xr.open_dataset(ds_path)
    assert 'y' in ds.coords and 'x' in ds.coords, 'missing y/x coords'
    assert 'z' in ds.data_vars, 'missing z data var'
    lats = ds['y'].values
    lons = ds['x'].values
    assert lats.shape == (512,), f'lats shape={lats.shape}, expected (512,)'
    assert lons.shape == (1024,), f'lons shape={lons.shape}, expected (1024,)'
    assert lats.min() >= -90 and lats.max() <= 90
    assert lons.min() >= -180 and lons.max() <= 180
    z = ds['z'].values
    if expected_channels == 1:
        assert z.shape == (512, 1024), f'z shape={z.shape}'
    else:
        assert z.shape == (512, 1024, expected_channels), f'z shape={z.shape}'
    assert np.all(np.isfinite(z)), 'non-finite values in z'


def _check_dataset_loader(dataset: str, encoding: str) -> None:
    path = DATASET_CONFIG[dataset]['path']
    held_out_path = DATASET_CONFIG[dataset].get('held_out_path', None)
    if held_out_path and not os.path.exists(held_out_path):
        held_out_path = None
    C = DATASET_CONFIG[dataset]['out_features']
    kwargs = _ENCODING_KWARGS_FOR_TEST[encoding]
    sd = SphericalDataset(path, coordinate_encoding=encoding,
                          encoding_kwargs=kwargs,
                          held_out_file_path=held_out_path)
    N = len(sd)
    assert N == 512 * 1024, f'N = {N}, expected {512*1024}'

    sample = sd[0]
    coord = sample['coord']; target = sample['target']
    assert target.shape == (C,), f'{dataset}/{encoding}: target shape {target.shape}'
    assert float(target.min()) >= -1.01 and float(target.max()) <= 1.01, \
        f'{dataset}/{encoding}: target out of [-1,1] range'

    expected_dim = coord_encoding_dim(encoding, **kwargs)
    assert coord.shape == (expected_dim,), (
        f'{encoding} coord shape {coord.shape}, expected ({expected_dim},)'
    )

    if encoding == 'cartesian':
        norm_sq = float((sd.coords ** 2).sum(-1).mean())
        assert abs(norm_sq - 1.0) < 1e-4, \
            f'cartesian coords not on unit sphere (mean |r|^2 = {norm_sq})'

    if encoding == 'spherical-rff':
        m = kwargs['num_features']
        cos_part = sd.coords[:, :m]
        sin_part = sd.coords[:, m:]
        # cos² + sin² = 1 for each frequency-pair, sample-wise.
        unit_circle_err = (cos_part ** 2 + sin_part ** 2 - 1.0).abs().max().item()
        assert unit_circle_err < 1e-5, (
            f'spherical-rff: cos²+sin² should equal 1, got max err {unit_circle_err}'
        )
        assert float(cos_part.abs().max()) <= 1.0 + 1e-6
        assert float(sin_part.abs().max()) <= 1.0 + 1e-6

    if encoding == 'spherical-harmonics':
        # Feature 0 is Y_0^0 = 1/sqrt(4π) — a literal constant across the sphere.
        y00 = float(sd.coords[:, 0].std())
        assert y00 < 1e-5, f'SH Y_00 should be constant; std = {y00}'
        expected_y00 = 1.0 / np.sqrt(4.0 * np.pi)
        mean_y00 = float(sd.coords[:, 0].mean())
        assert abs(mean_y00 - expected_y00) < 1e-4, (
            f'SH Y_00 = {mean_y00}, expected {expected_y00}'
        )
        
    # ----- Held-out half-pixel-offset eval  -----
    if held_out_path is None:
        print(f'  OK {dataset}/{encoding}: N={N}, C={C}, coord={tuple(coord.shape)} '
              f'(skipped held-out: {DATASET_CONFIG[dataset].get("held_out_path")} '
              f'missing — run `python -m datasets.preprocess --dataset {dataset}` '
              f'to generate it)')
        return

    coords_held, targets_held = sd.make_held_out_eval()
    expected_N_held = sd.held_out_height * sd.held_out_width
    assert coords_held.shape == (expected_N_held, expected_dim), (
        f'{dataset}/{encoding}: held-out coord shape {coords_held.shape}, '
        f'expected ({expected_N_held}, {expected_dim})'
    )
    assert targets_held.shape == (expected_N_held, C), (
        f'{dataset}/{encoding}: held-out target shape {targets_held.shape}, '
        f'expected ({expected_N_held}, {C})'
    )
    
    # Held-out targets come from a separately preprocessed 511×1023 file
    # (sampled directly from the base source). Both training and held-out
    # files normalize against the BASE-SOURCE per-channel min/max stored as
    # NetCDF attrs by the preprocessor, so every sub-sample falls within
    # [-1, 1] regardless of which extremes it captured.
    assert float(targets_held.min()) >= -1.01, (
        f'{dataset}/{encoding}: held-out target min '
        f'{float(targets_held.min())} outside [-1, 1] — check that the '
        f'preprocessor wrote source-derived target_min/target_max attrs '
        f'and that both files use the same reference.'
    )
    assert float(targets_held.max()) <= 1.01, (
        f'{dataset}/{encoding}: held-out target max '
        f'{float(targets_held.max())} outside [-1, 1] — same diagnosis as above.'
    )
    # Caching: a second call should return the same tensors (same id).
    coords_held_2, targets_held_2 = sd.make_held_out_eval()
    assert coords_held_2 is coords_held, 'held-out coords not cached'
    assert targets_held_2 is targets_held, 'held-out targets not cached'


    print(f'  OK {dataset}/{encoding}: N={N}, C={C}, coord={tuple(coord.shape)}, '
          f'target range=[{float(sd.targets.min()):.3f},{float(sd.targets.max()):.3f}]')
    print(f'       held-out: N_held={expected_N_held}, '
          f'coord={tuple(coords_held.shape)}, target range='
          f'[{float(targets_held.min()):.3f},{float(targets_held.max()):.3f}]')



# ---------------------------------------------------------------------------
# SH self-tests — don't require any preprocessed dataset.
# ---------------------------------------------------------------------------
def test_sh_y00_constant_and_correct() -> None:
    print('\n[SH] Y_00 constancy and value ...')
    x = np.linspace(-1.0, 1.0, 17)
    plm = _normalized_associated_legendre(L_max=0, x=x)
    expected = 1.0 / np.sqrt(4.0 * np.pi)
    assert plm.shape == (17, 1)
    assert np.allclose(plm[:, 0], expected, atol=1e-12), \
        f'Y_00 != 1/sqrt(4π); got {plm[:, 0]}'
    print(f'  OK Y_00 = {expected:.6f} at all x')


def test_sh_orthonormality_on_gauss_legendre() -> None:
    """
    Verify ∫ Y_lm Y_l'm' dΩ = δ_ll' δ_mm' numerically on a Gauss-Legendre
    lat grid (exact quadrature in cos(θ) for low L_max), plus uniform
    longitude quadrature. Small L_max so the test is fast.
    """
    print('\n[SH] orthonormality on Gauss-Legendre grid, L_max=4 ...')
    L_max = 4
    n_lat = 16  # > L_max + 1 for exact Gauss–Legendre in cos(θ)
    n_lon = 2 * (L_max + 1) + 2  # > 2 L_max for exact trig quadrature

    x_nodes, w_x = np.polynomial.legendre.leggauss(n_lat)
    lats_deg = np.degrees(np.arcsin(x_nodes)).astype(np.float64)
    lons_deg = np.linspace(-180.0, 180.0, n_lon, endpoint=False).astype(np.float64)

    sh = _sh_features(lats_deg, lons_deg, L_max)     # (H, W, (L+1)^2)
    H, W, D = sh.shape
    # dΩ = sin(θ) dθ dφ = dx dφ. Integrand weights: w_x[i] * dφ.
    d_phi = 2.0 * np.pi / W
    weights = (w_x[:, None] * d_phi * np.ones((1, W))).astype(np.float64)  # (H, W)
    flat_sh = sh.reshape(H * W, D).astype(np.float64)
    flat_w = weights.reshape(H * W)
    G = (flat_sh.T * flat_w) @ flat_sh
    I = np.eye(D)
    err = float(np.abs(G - I).max())
    print(f'  max|Gram - I| = {err:.3e} (expect < 1e-6 for exact quadrature)')
    assert err < 1e-6, f'SH non-orthonormal: max|Gram - I| = {err}'


def main() -> None:
    # ------------------ Encoding self-tests (no data required) -----------------
    print('== SH correctness self-tests ==')
    test_sh_y00_constant_and_correct()
    test_sh_orthonormality_on_gauss_legendre()

    # ------------------ Schema + loader over all datasets ----------------------
    print('\n== Standardized-NetCDF schema checks ==')
    for ds in DATASET_CHOICES:
        if not _file_exists(ds):
            print(f'  SKIP {ds}: {DATASET_CONFIG[ds]["path"]} not found')
            continue
        _check_ds(DATASET_CONFIG[ds]['path'], DATASET_CONFIG[ds]['out_features'])
        print(f'  OK   {ds}: schema valid')

    print('\n== SphericalDataset loader + coord encodings ==')
    for ds in DATASET_CHOICES:
        if not _file_exists(ds):
            continue
        for enc in CE_CHOICES:
            _check_dataset_loader(ds, enc)

    print('\nAll available datasets passed integration checks.')


if __name__ == '__main__':
    main()