"""
PSNR breakdowns by spatial band and channel.

Logs overall PSNR plus polar / equatorial / per-channel
breakdowns for BOTH reconstruction (training pixels) and held-out (half-pixel
offset grid) metrics. These split a single PSNR scalar into the regions where
the coordinate encoding is expected to matter most:

  * Polar band (|φ| > polar_thresh_deg): angular encoding's singularity zone.
  * Equatorial band (|φ| < equatorial_thresh_deg): the "easy" reference region.
  * Per-channel (RGB only): chromatic spectral heterogeneity in HDRI datasets.

All inputs are flat (N, C) tensors in row-major (h, w) order — the same order
SphericalDataset emits. The per-pixel latitude is recovered as
    lat_per_pixel[i] = lats_deg[i // W]
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _flat_lats(lats_deg: np.ndarray, width: int) -> torch.Tensor:
    """Expand (H,) lats to a (H*W,) per-pixel latitude vector in row-major
    (h, w) order. All W positions in row h share `lats_deg[h]`."""
    lats_t = torch.as_tensor(np.asarray(lats_deg), dtype=torch.float32)
    return lats_t.repeat_interleave(int(width))


def _psnr_from_mse(mse: torch.Tensor) -> float:
    """Numerically safe -10*log10(mse) → float.

    Clamps MSE at 1e-20 so MSE==0 yields a large finite value (~200 dB)
    rather than +inf. NaN propagates unchanged — the caller can interpret
    NaN as a failed run.
    """
    mse_clamped = mse.clamp(min=1e-20)
    return float((-10.0 * torch.log10(mse_clamped)).item())


def compute_psnr_breakdowns(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lats_deg: np.ndarray,
    width: int,
    polar_thresh_deg: float = 60.0,
    equatorial_thresh_deg: float = 30.0,
    channel_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Overall, polar, equatorial, and per-channel PSNR (dB).

    Inputs
    ------
    preds, targets : (N, C) float tensors, row-major (h, w) order, N = H * W.
    lats_deg       : (H,) array of pixel-center latitudes in degrees.
    width          : W = number of longitude pixels.
    polar_thresh_deg, equatorial_thresh_deg :
                     band thresholds. Defaults match preregistration H1.
    channel_names  : optional list of length C; defaults to ['r','g','b']
                     for C=3 and ['c0', ...] otherwise. Used only as the
                     suffix in the returned dict keys.

    Returns
    -------
    dict[str, float]
        Keys: 'overall', 'polar', 'equatorial', and (if C > 1) 'channel_<NAME>'
        for each channel. Bands with no pixels yield NaN rather than raising.
    """
    if preds.shape != targets.shape:
        raise ValueError(f"preds {tuple(preds.shape)} != targets {tuple(targets.shape)}")
    N, C = preds.shape
    H = int(np.asarray(lats_deg).shape[0])
    W = int(width)
    if N != H * W:
        raise ValueError(
            f"N={N} does not match len(lats_deg) * width = {H} * {W} = {H * W}"
        )

    sq_err = (preds - targets).pow(2)                       # (N, C)
    lat_per_pixel = _flat_lats(lats_deg, W)                 # (N,)

    out: dict[str, float] = {}
    out['overall'] = _psnr_from_mse(sq_err.mean())

    polar_mask = lat_per_pixel.abs() > float(polar_thresh_deg)
    out['polar'] = (
        _psnr_from_mse(sq_err[polar_mask].mean())
        if polar_mask.any() else float('nan')
    )

    equatorial_mask = lat_per_pixel.abs() < float(equatorial_thresh_deg)
    out['equatorial'] = (
        _psnr_from_mse(sq_err[equatorial_mask].mean())
        if equatorial_mask.any() else float('nan')
    )

    if C > 1:
        if channel_names is None:
            channel_names = ['r', 'g', 'b'] if C == 3 else [f'c{i}' for i in range(C)]
        if len(channel_names) != C:
            raise ValueError(
                f"channel_names has {len(channel_names)} entries, expected {C}"
            )
        for c, name in enumerate(channel_names):
            out[f'channel_{name}'] = _psnr_from_mse(sq_err[:, c].mean())

    return out
