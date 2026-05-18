"""
Unit tests for src/metrics/breakdowns.py
the polar/equatorial/per-channel PSNR breakdown function.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics.breakdowns import (                                       
    compute_psnr_breakdowns,
    _flat_lats,
    _psnr_from_mse,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def test_flat_lats_row_major() -> None:
    """_flat_lats expands (H,) → (H*W,) so all W positions in row h share lat[h]."""
    print('\n[breakdowns] _flat_lats row-major ordering ...')
    lats = np.array([10.0, 20.0, 30.0], dtype=np.float32)   # H = 3
    flat = _flat_lats(lats, width=4)                         # (12,)
    expected = torch.tensor(
        [10.0, 10.0, 10.0, 10.0,
         20.0, 20.0, 20.0, 20.0,
         30.0, 30.0, 30.0, 30.0],
        dtype=torch.float32,
    )
    assert torch.allclose(flat, expected), f'got {flat}'
    print('  OK row-major expansion correct.')


def test_psnr_from_mse_safe_zero() -> None:
    """MSE == 0 yields a large finite PSNR (not +inf)."""
    print('\n[breakdowns] _psnr_from_mse handles zero MSE ...')
    p = _psnr_from_mse(torch.tensor(0.0))
    assert math.isfinite(p), f'PSNR should be finite, got {p}'
    assert p > 100.0, f'expected very high PSNR for zero MSE, got {p:.3f}'
    print(f'  OK PSNR(MSE=0) = {p:.2f} dB (clamped at 1e-20).')


def test_psnr_from_mse_known_values() -> None:
    """A known MSE produces the expected PSNR (within float slop)."""
    # MSE = 0.01 → PSNR = -10 log10(0.01) = 20 dB
    p = _psnr_from_mse(torch.tensor(0.01))
    assert abs(p - 20.0) < 1e-4, f'expected 20 dB, got {p}'
    # MSE = 0.001 → PSNR = 30 dB
    p = _psnr_from_mse(torch.tensor(0.001))
    assert abs(p - 30.0) < 1e-4, f'expected 30 dB, got {p}'


def test_psnr_from_mse_nan_propagates() -> None:
    """NaN MSE produces NaN PSNR (caller can interpret as a failed run)."""
    p = _psnr_from_mse(torch.tensor(float('nan')))
    assert math.isnan(p), f'expected NaN, got {p}'


# ---------------------------------------------------------------------------
# Band masking and overall consistency
# ---------------------------------------------------------------------------
def test_overall_matches_manual_mse() -> None:
    """The 'overall' key equals the naive MSE-based PSNR on the full grid."""
    print('\n[breakdowns] overall PSNR matches manual computation ...')
    torch.manual_seed(0)
    H, W, C = 8, 16, 1
    lats_deg = np.linspace(-90, 90, H, dtype=np.float32)
    preds = torch.randn(H * W, C)
    targets = preds + 0.1 * torch.randn(H * W, C)   # MSE ≈ 0.01

    out = compute_psnr_breakdowns(preds, targets, lats_deg, W)
    manual = _psnr_from_mse(((preds - targets) ** 2).mean())
    assert abs(out['overall'] - manual) < 1e-6, (
        f"overall={out['overall']} vs manual={manual}"
    )
    print(f"  OK overall = {out['overall']:.4f} dB matches manual.")


def test_polar_equatorial_band_masking() -> None:
    """Polar band picks only |lat|>60°; equatorial picks only |lat|<30°."""
    print('\n[breakdowns] polar/equatorial band masking ...')
    # Construct a grid where polar rows have zero error and equatorial rows
    # have a known error. Result: polar PSNR very high, equatorial PSNR
    # matches the equatorial MSE.
    H, W, C = 10, 4, 1
    lats_deg = np.linspace(-85.0, 85.0, H, dtype=np.float32)  # full lat sweep
    # Equatorial rows: |lat| < 30 — for this H, roughly rows 4 and 5.
    # Polar rows: |lat| > 60 — roughly rows 0–1 and 8–9.
    targets = torch.zeros(H * W, C)
    preds = torch.zeros(H * W, C)
    eq_err = 0.1                              # per-element error in equatorial
    for h in range(H):
        if abs(lats_deg[h]) < 30.0:
            preds[h * W:(h + 1) * W] = eq_err
        # polar rows stay zero → MSE=0 → very high PSNR
    out = compute_psnr_breakdowns(preds, targets, lats_deg, W)

    # Polar PSNR should be very high (zero error in the polar mask).
    assert out['polar'] > 100.0, f"polar={out['polar']}: should be near-inf"
    # Equatorial PSNR: MSE = eq_err² = 0.01 → 20 dB.
    assert abs(out['equatorial'] - 20.0) < 1e-3, (
        f"equatorial={out['equatorial']}: expected ~20 dB"
    )
    print(f"  OK polar={out['polar']:.1f} (zero error), "
          f"equatorial={out['equatorial']:.3f} (eq_err=0.1 → ~20 dB).")


def test_band_empty_returns_nan() -> None:
    """If a band has no pixels, its PSNR is NaN (not a crash)."""
    print('\n[breakdowns] empty band → NaN ...')
    # Latitudes all in [-20, 20] — no polar pixels (|lat|>60 is empty).
    H, W = 5, 4
    lats_deg = np.linspace(-20.0, 20.0, H, dtype=np.float32)
    preds = torch.zeros(H * W, 1)
    targets = torch.zeros(H * W, 1)
    out = compute_psnr_breakdowns(preds, targets, lats_deg, W)
    assert math.isnan(out['polar']), f"expected polar=NaN, got {out['polar']}"
    assert not math.isnan(out['equatorial']), "equatorial should not be NaN"
    print(f"  OK polar=NaN (correct), equatorial={out['equatorial']:.1f}.")


# ---------------------------------------------------------------------------
# Per-channel breakdown
# ---------------------------------------------------------------------------
def test_per_channel_rgb() -> None:
    """C=3 produces channel_r / channel_g / channel_b matching per-channel MSE."""
    print('\n[breakdowns] per-channel RGB breakdown ...')
    H, W = 4, 4
    lats_deg = np.linspace(-45.0, 45.0, H, dtype=np.float32)
    targets = torch.zeros(H * W, 3)
    preds = torch.zeros(H * W, 3)
    # Different per-channel errors.
    preds[:, 0] = 0.1   # R: MSE = 0.01 → 20 dB
    preds[:, 1] = 0.01  # G: MSE = 1e-4 → 40 dB
    preds[:, 2] = 0.001 # B: MSE = 1e-6 → 60 dB
    out = compute_psnr_breakdowns(preds, targets, lats_deg, W)
    assert 'channel_r' in out and 'channel_g' in out and 'channel_b' in out
    assert abs(out['channel_r'] - 20.0) < 1e-3, f"R PSNR={out['channel_r']}"
    assert abs(out['channel_g'] - 40.0) < 1e-3, f"G PSNR={out['channel_g']}"
    assert abs(out['channel_b'] - 60.0) < 1e-3, f"B PSNR={out['channel_b']}"
    print(f"  OK R={out['channel_r']:.2f}  G={out['channel_g']:.2f}  "
          f"B={out['channel_b']:.2f}.")


def test_scalar_omits_channel_breakdown() -> None:
    """C=1 (scalar) produces no channel_* keys (collapses to overall)."""
    print('\n[breakdowns] scalar signal omits channel breakdown ...')
    H, W = 4, 4
    lats_deg = np.linspace(-45.0, 45.0, H, dtype=np.float32)
    preds = torch.zeros(H * W, 1)
    targets = torch.zeros(H * W, 1) + 0.1
    out = compute_psnr_breakdowns(preds, targets, lats_deg, W)
    channel_keys = [k for k in out if k.startswith('channel_')]
    assert channel_keys == [], f"expected no channel keys, got {channel_keys}"
    print(f"  OK keys = {sorted(out.keys())} (no channel_*).")


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------
def test_mismatched_shape_raises() -> None:
    """preds and targets with different shapes → ValueError."""
    print('\n[breakdowns] shape mismatch raises ValueError ...')
    H, W = 4, 4
    lats_deg = np.linspace(-45.0, 45.0, H, dtype=np.float32)
    preds = torch.zeros(H * W, 1)
    targets = torch.zeros(H * W, 3)
    try:
        compute_psnr_breakdowns(preds, targets, lats_deg, W)
    except ValueError as e:
        print(f"  OK raised: {e}")
        return
    raise AssertionError("expected ValueError, got none")


def test_wrong_N_raises() -> None:
    """N != H*W → ValueError."""
    print('\n[breakdowns] N mismatch raises ValueError ...')
    H, W = 4, 4
    lats_deg = np.linspace(-45.0, 45.0, H, dtype=np.float32)
    preds = torch.zeros(H * W + 1, 1)   # one too many
    targets = torch.zeros(H * W + 1, 1)
    try:
        compute_psnr_breakdowns(preds, targets, lats_deg, W)
    except ValueError as e:
        print(f"  OK raised: {e}")
        return
    raise AssertionError("expected ValueError, got none")


def main() -> None:
    print('== breakdowns internal helpers ==')
    test_flat_lats_row_major()
    test_psnr_from_mse_safe_zero()
    test_psnr_from_mse_known_values()
    test_psnr_from_mse_nan_propagates()

    print('\n== overall + band masking ==')
    test_overall_matches_manual_mse()
    test_polar_equatorial_band_masking()
    test_band_empty_returns_nan()

    print('\n== per-channel breakdown ==')
    test_per_channel_rgb()
    test_scalar_omits_channel_breakdown()

    print('\n== shape validation ==')
    test_mismatched_shape_raises()
    test_wrong_N_raises()

    print('\nAll breakdown tests passed.')


if __name__ == '__main__':
    main()
