"""Render equirectangular previews of every preprocessed dataset.

Produces:
    figures/preview_<dataset>.png       (5 individual PNGs)
    figures/datasets_glance.png         (5-in-a-row montage, Figure 1)

The montage is the drop-in artifact for the paper's Figure 1
(five-dataset at-a-glance). The individual PNGs are provided in case
the paper wants a \\subfloat / \\subcaptionbox composition instead of
the pre-composed montage.

Notes on rendering:
  * Every preprocessed *_512x1024.nc file stores the signal as `ds['z']`
    with dims (y, x) for scalar datasets and (y, x, c) for RGB. Targets
    are normalized to [-1, 1] using source-derived per-channel bounds
    (see src/datasets/preprocessing/common.py), so the display range
    is consistent across every file.
  * Scalar datasets use dataset-appropriate diverging or sequential
    colormaps; RGB datasets are shifted from [-1, 1] to [0, 1] and
    gamma-corrected for display (the source .exr files are HDR, so a
    linear display would clip the sky).
  * No colorbars in the montage — this is a "here is what the signals
    look like" figure, not a "compare their magnitudes" figure. The
    per-dataset spectral characterization lives in Table 2 and
    Appendix B.

Usage:
    PYTHONPATH=src python scripts/make_dataset_previews.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Dataset registry: (short_key, path_constant_name, display_title, cmap, is_rgb)
# ---------------------------------------------------------------------------
# The 5 datasets, in the same left-to-right order used in the paper's
# Table 2. Colormaps are picked to be semantically appropriate:
#   * CMB    — diverging around 0 (temperature fluctuations).
#   * ERA5   — cool-warm; absolute temperatures are not zero-centered.
#   * ETOPO1 — diverging around 0 (bathymetry negative, terrain positive).
#   * HDRI-* — RGB, cmap ignored.
# Fields: (short_key, path_constant_name, display_title, cmap, is_rgb,
#          imshow_origin, flip_lon)
#
# * imshow_origin: whether row 0 of the file goes at the TOP ('upper') or
#   BOTTOM ('lower') of the rendered image. Every dataset labels row 0
#   as latitude -90 (south) via the standard grid in common.py, but the
#   HDRI preprocessor pairs those row-0 latitudes with the EXR's top
#   row (sky), so origin='upper' is required to put sky at the top of
#   the preview. All other datasets have row-0 pixel data that matches
#   the row-0 latitude label, so origin='lower' puts North at the top.
# * flip_lon: horizontal mirror. Left as False for all datasets by
#   default; toggle for CMB if the healpy phi convention produces a
#   left-right-mirrored image relative to conventional galactic maps.
_DATASETS = (
    ('cmb',        'CMB_DATA_PATH',        'CMB (Planck)', 'RdBu_r',   False, 'lower', True),
    ('era5',       'ERA5_DATA_PATH',       'ERA5 2m temp', 'coolwarm', False, 'lower', False),
    ('etopo1',     'ELEVATION_DATA_PATH',  'ETOPO1',       'terrain',  False, 'lower', False),
    ('hdri_sky',   'HDRI_SKY_DATA_PATH',   'HDRI-sky',     None,       True,  'upper', False),
    ('hdri_urban', 'HDRI_URBAN_DATA_PATH', 'HDRI-urban',   None,       True,  'upper', False),
)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _load_signal(nc_path: str) -> tuple[np.ndarray,
                                        Optional[np.ndarray],
                                        Optional[np.ndarray]]:
    """Load the (H, W) or (H, W, 3) signal from a preprocessed NetCDF.

    Returns (z, target_min, target_max):
      * `z` — the raw normalized-[-1, 1] array. RGB signals keep their
        (H, W, 3) shape; scalar signals return (H, W).
      * `target_min`, `target_max` — per-channel arrays of length C
        holding the source-signal bounds used for the [-1, 1]
        normalization, read from the NetCDF's `target_min_json` /
        `target_max_json` attrs (see src/datasets/preprocessing/common.py).
        `None` for files that predate the per-channel normalization
        convention.

    Display orientation is handled by the per-dataset `origin` and
    `flip_lon` flags in _DATASETS, not derived from the file — the
    HDRI preprocessor stores pixel data with row 0 = sky (image
    convention) but pairs it with the standard latitude grid that
    labels row 0 as latitude -90 (south), so auto-detection from the
    latitude coordinate alone would be wrong for those files.
    """
    import json as _json
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    try:
        z = np.array(ds['z'].values, dtype=np.float32)
        target_min: Optional[np.ndarray] = None
        target_max: Optional[np.ndarray] = None
        if 'target_min_json' in ds.attrs and 'target_max_json' in ds.attrs:
            try:
                target_min = np.asarray(
                    _json.loads(ds.attrs['target_min_json']), dtype=np.float64,
                )
                target_max = np.asarray(
                    _json.loads(ds.attrs['target_max_json']), dtype=np.float64,
                )
            except (ValueError, TypeError):
                pass
    finally:
        ds.close()
    return z, target_min, target_max


def _prepare_for_display(z: np.ndarray, is_rgb: bool,
                         target_min: Optional[np.ndarray],
                         target_max: Optional[np.ndarray]) -> np.ndarray:
    """Convert a normalized-[-1, 1] array into a form imshow can render.

    For RGB (HDR panoramas): the HDRI preprocessor already applies a
    Reinhard + sRGB-gamma tone-map to the raw EXR *before* storing
    values (see `_tonemap` in src/datasets/preprocessing/hdri.py), so
    the on-disk values are already display-ready LDR. We invert the
    per-channel [-1, 1] normalization to recover those LDR values,
    then apply a per-luminance percentile stretch (set the darkest 1%
    to pure black and the brightest 1% to pure white with a linear
    ramp between) so the two HDRIs in the paper — a bright sky
    panorama and a dark urban panorama — both use the display's full
    dynamic range instead of clustering in whichever half of it their
    scene happens to prefer.

    For scalar: return unchanged; matplotlib's `vmin`/`vmax` handles
    the colormap range in _imshow_signal().
    """
    if is_rgb:
        if target_min is not None and target_max is not None:
            span = np.asarray(target_max - target_min,
                              dtype=np.float32).reshape((1, 1, -1))
            base = np.asarray(target_min,
                              dtype=np.float32).reshape((1, 1, -1))
            v = (z + 1.0) * 0.5 * span + base                   # → preprocessor LDR
        else:
            v = (z + 1.0) * 0.5                                 # fallback: [-1,1] → [0,1]
        v = np.clip(v, 0.0, 1.0)
        # Per-image auto-levels: affine stretch so the 1st percentile of
        # luminance lands at 0 and the 99th at 1. Applied to each RGB
        # channel with the same shift and scale, so it darkens the whole
        # image AND amplifies inter-channel chroma at the same time.
        # We blend a fraction of that stretched image against the
        # original — `strength` at 0 leaves the (usually bright, low-
        # chroma) preprocessor output untouched, and 1 gives the full
        # (darker, high-chroma) auto-levelled version. Blending at the
        # pre-clip stage darkens luminance and lifts saturation in the
        # same proportion, which is what we want.
        lum = 0.2126 * v[..., 0] + 0.7152 * v[..., 1] + 0.0722 * v[..., 2]
        p_low  = float(np.percentile(lum,  1.0))
        p_high = float(np.percentile(lum, 99.0))
        if p_high > p_low + 1e-6:
            v_stretched = (v - p_low) / (p_high - p_low)
            strength = 0.65
            v = v * (1.0 - strength) + v_stretched * strength
        return np.clip(v, 0.0, 1.0)
    return z


def _imshow_signal(ax, z: np.ndarray, cmap: str | None, is_rgb: bool,
                   title: str, origin: str, flip_lon: bool) -> None:
    """Shared imshow logic used by both the single and montage renderers.

    * `origin` is set per-dataset (see _DATASETS): 'lower' when row 0 of
      the file corresponds to the South hemisphere in the pixel data,
      'upper' when row 0 already sits at the top of the image (HDRIs).
      With the correct choice, North always appears at the top of the
      rendered image.
    * `flip_lon` mirrors the image horizontally; useful for CMB whose
      HEALPix source uses a phi convention that can render mirrored
      relative to conventional galactic Mollweide maps.
    * Diverging colormaps (CMB, ETOPO1) centre at zero with vmin/vmax
      set from the 99.5th percentile of |z|, not from max(|z|). Using
      the percentile prevents a handful of outlier pixels from
      dominating the colormap and washing everything else out.
    * ERA5 uses a non-diverging colormap and matplotlib's autoscaling,
      since absolute temperatures are not zero-centred.
    """
    if flip_lon:
        z = z[..., :, ::-1] if is_rgb else z[:, ::-1]
    if is_rgb:
        ax.imshow(z, interpolation='nearest', origin=origin)
    else:
        symmetric = title.startswith(('CMB', 'ETOPO1'))
        if symmetric:
            m = float(np.nanpercentile(np.abs(z), 99.5))
            if not np.isfinite(m) or m <= 0:
                m = float(np.nanmax(np.abs(z)))
            ax.imshow(z, cmap=cmap, vmin=-m, vmax=m,
                      interpolation='nearest', origin=origin)
        else:
            ax.imshow(z, cmap=cmap, interpolation='nearest', origin=origin)


def _render_single(z: np.ndarray, title: str, cmap: str | None,
                   is_rgb: bool, origin: str, flip_lon: bool,
                   out_path: str) -> None:
    """Render a single equirectangular preview to `out_path`."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
    _imshow_signal(ax, z, cmap, is_rgb, title, origin, flip_lon)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def _render_montage(images, out_path: str) -> None:
    """Render all datasets side-by-side in a single figure.

    5 subplots in a 1×5 row. IEEE conference `figure*` spans full text
    width (~7.1"); at 5 columns each subplot is ~1.4" wide by 0.7" tall
    (2:1 equirectangular aspect ratio) — small but legible.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 1.6), dpi=200)
    if n == 1:
        axes = [axes]
    for ax, (z, title, cmap, is_rgb, origin, flip_lon) in zip(axes, images):
        _imshow_signal(ax, z, cmap, is_rgb, title, origin, flip_lon)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out_dir', default='figures',
                   help='Directory to write the PNGs into.')
    p.add_argument('--skip_individual', action='store_true',
                   help='Skip the per-dataset PNGs; only produce the montage.')
    args = p.parse_args()

    # Late import so `--help` still works without the code path deps installed.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from config import constants                                     # noqa: E402

    entries = []
    for key, path_attr, title, cmap, is_rgb, origin, flip_lon in _DATASETS:
        nc_path = getattr(constants, path_attr)
        if not os.path.exists(nc_path):
            print(f'[make_dataset_previews] SKIP {key}: {nc_path} not found. '
                  'Run the preprocessor first (see SETUP.md §4).',
                  file=sys.stderr)
            continue
        print(f'[make_dataset_previews] loading {key} ...')
        z, tmin, tmax = _load_signal(nc_path)
        z_disp = _prepare_for_display(z, is_rgb, tmin, tmax)
        entries.append((z_disp, title, cmap, is_rgb, origin, flip_lon, key))

    if not entries:
        print('[make_dataset_previews] no datasets found; nothing to render.',
              file=sys.stderr)
        return 2

    if not args.skip_individual:
        for z, title, cmap, is_rgb, origin, flip_lon, key in entries:
            out_path = os.path.join(args.out_dir, f'preview_{key}.png')
            _render_single(z, title, cmap, is_rgb, origin, flip_lon, out_path)
            print(f'[make_dataset_previews] wrote {out_path}')

    montage_path = os.path.join(args.out_dir, 'datasets_glance.png')
    _render_montage(
        [(z, title, cmap, is_rgb, origin, flip_lon)
         for z, title, cmap, is_rgb, origin, flip_lon, _ in entries],
        montage_path,
    )
    print(f'[make_dataset_previews] wrote {montage_path} '
          f'({len(entries)} datasets)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
