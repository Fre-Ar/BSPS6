"""
Poly Haven HDRI (equirectangular 360° panorama) pre-processor.

Reads an EXR file (linear-light radiance, float16 or float32), applies a
deterministic exposure + Reinhard + gamma tone-map to produce LDR RGB in
[0, 1], resizes to the standard 512x1024 equirectangular grid, and saves as
a standardized NetCDF with ds['z'] of shape (H, W, 3).

Tone-mapping pipeline:

    1. x = clip(x, 0, +inf)                # drop negative spikes
    2. x = x * exposure                    # exposure=1.0 default (neutral)
    3. x = reinhard(x) = x / (1 + x)       # per-channel
    4. x = x ** (1 / gamma)                # gamma=2.2 default
    5. x = clip(x, 0, 1)
"""
from __future__ import annotations

import os
import numpy as np

from .common import (
    _standard_grid, _held_out_grid,
    save_standardized, sanity_check_standardized,
)

def _read_exr(path: str) -> np.ndarray:
    """Read an EXR as (H, W, 3) float32 linear RGB.
 
    Tries, in order:
      1. OpenCV (`cv2.imread`) — smallest, most reliable on macOS/Linux.
         Needs OPENCV_IO_ENABLE_OPENEXR=1 in the environment BEFORE cv2 is
         imported (the cv2 wheel respects it). We set it here defensively.
      2. Modern OpenEXR v3 Python API (`OpenEXR.File`), introduced 2023.
      3. Legacy OpenEXR v1/v2 API (`OpenEXR.InputFile` + `Imath`).
      4. imageio.v3 (last, because EXR support there is flaky in recent
         versions — the FreeImage plugin no longer ships a usable binary
         on macOS ARM).
    """
    errors: dict[str, str] = {}
 
    # --- 1. OpenCV --------------------------------------------------------
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2  # type: ignore
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(
                "cv2.imread returned None. "
                "This build of OpenCV may lack OpenEXR; try `pip install OpenEXR`."
            )
        img = img.astype(np.float32)
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., [2, 1, 0]]        # BGR -> RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img
    except Exception as e:  # noqa: BLE001
        errors['cv2'] = repr(e)
 
    # --- 2. Modern OpenEXR v3 (OpenEXR.File) ------------------------------
    try:
        import OpenEXR  # type: ignore
        if hasattr(OpenEXR, "File"):
            with OpenEXR.File(path) as f:     # type: ignore[attr-defined]
                parts = f.parts
                # Choose the first part that has R/G/B channels.
                channels = parts[0].channels()
                names = [c for c in ('R', 'G', 'B') if c in channels]
                if len(names) < 3:
                    raise RuntimeError(
                        f"EXR missing R/G/B channels: found {list(channels)}"
                    )
                planes = [np.asarray(channels[c].pixels, dtype=np.float32)
                          for c in names]
                return np.stack(planes, axis=-1)
        raise AttributeError("no OpenEXR.File (v3 API unavailable); falling back")
    except Exception as e:  # noqa: BLE001
        errors['OpenEXR(v3)'] = repr(e)
 
    # --- 3. Legacy OpenEXR v1/v2 (OpenEXR.InputFile + Imath) --------------
    try:
        import OpenEXR  # type: ignore
        import Imath    # type: ignore
        exr = OpenEXR.InputFile(path)
        hdr = exr.header()
        dw = hdr['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = hdr['channels'].keys()
        names = [c for c in ('R', 'G', 'B') if c in channels]
        if len(names) < 3:
            raise RuntimeError(f"EXR missing RGB, channels={list(channels)}")
        raw = [np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w)
               for c in names]
        return np.stack(raw, axis=-1)
    except Exception as e:  # noqa: BLE001
        errors['OpenEXR(legacy)'] = repr(e)
 
    # --- 4. imageio (last-resort; often fails for .exr today) -------------
    try:
        import imageio.v3 as iio  # type: ignore
        img = iio.imread(path)
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img
    except Exception as e:  # noqa: BLE001
        errors['imageio'] = repr(e)
 
    lines = [f"All EXR readers failed for {path}:"]
    for k, v in errors.items():
        lines.append(f"    {k}: {v}")
    lines.append("")
    lines.append("Install one of:")
    lines.append("  pip install opencv-python        # recommended, smallest")
    lines.append("  pip install OpenEXR              # modern v3 API")
    lines.append("  pip install OpenEXR Imath        # legacy API")
    raise RuntimeError("\n".join(lines))


def _resample_equirect_at(
    img: np.ndarray,
    target_lats_deg: np.ndarray,
    target_lons_deg: np.ndarray,
) -> np.ndarray:
    """
    Sample (H_src, W_src, 3) equirectangular `img` at arbitrary (lat, lon)
    coordinates via bilinear interpolation. Returns
    (len(target_lats_deg), len(target_lons_deg), 3).

    Lat convention: endpoint-inclusive [-90, +90] → source pixel rows
    [0, H_src - 1] (north pole at row 0 for the typical Poly Haven export).
    Lon convention: endpoint-exclusive [-180, +180) → source pixel columns
    [0, W_src), with 'wrap' edge mode handling the longitude seam.
    """
    from scipy.ndimage import map_coordinates

    H, W, C = img.shape
    # Source-pixel mapping for the target lats and lons (degrees).
    v_coords = (np.asarray(target_lats_deg, dtype=np.float32) + 90.0) / 180.0 * (H - 1)
    u_coords = (np.asarray(target_lons_deg, dtype=np.float32) + 180.0) / 360.0 * W
    vv, uu = np.meshgrid(v_coords, u_coords, indexing='ij')

    out = np.empty((len(target_lats_deg), len(target_lons_deg), C), dtype=np.float32)
    for c in range(C):
        out[..., c] = map_coordinates(img[..., c], [vv, uu], order=1, mode='wrap')
    return out


def _tonemap(
    rgb_linear: np.ndarray,
    exposure: float = 1.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Deterministic Reinhard + gamma tone-map."""
    x = np.clip(rgb_linear, 0.0, None)
    x = x * float(exposure)
    x = x / (1.0 + x)                         # Reinhard, per channel
    x = np.power(np.clip(x, 0.0, 1.0), 1.0 / float(gamma))
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def preprocess_hdri(
    input_filepath: str,
    output_filepath: str,
    held_out_filepath: str | None = None,
    n_lat: int = 512,
    n_lon: int = 1024,
    exposure: float = 1.0,
    gamma: float = 2.2,
) -> None:
    """Pre-process the EXR onto the standard (n_lat × n_lon) training grid,
    and, if `held_out_filepath` is given, ALSO onto the half-pixel-offset
    ((n_lat-1) × (n_lon-1)) grid — both sampled directly from the EXR,
    never from each other."""
    
    print(f"[HDRI] loading {input_filepath} ...")
    rgb = _read_exr(input_filepath)
    print(f"[HDRI] EXR shape={rgb.shape} dtype={rgb.dtype} "
          f"range=[{rgb.min():.4g}, {rgb.max():.4g}]")

    # Sanity: EXR from Poly Haven is typically 2:1 equirectangular.
    H, W = rgb.shape[:2]
    if not (0.99 * W <= 2 * H <= 1.01 * W):
        print(f"[HDRI] warning: source aspect {W}/{H} != 2:1. "
              f"We're assuming equirectangular; visual distortion possible.")
        
    # Source-derived per-channel normalization reference. 
    # Computed on the tonemapped FULL EXR (point-wise op, so the
    # per-pixel min/max are well-defined even though the train/held-out
    # files apply tonemap after resampling — bilinear resampling preserves
    # the value range, and tonemap is monotonic, so the sub-sample values
    # always lie within the source [target_min, target_max]).
    ldr_source = _tonemap(rgb, exposure=exposure, gamma=gamma)  # (H, W, 3)
    flat_ldr = ldr_source.reshape(-1, 3)
    target_min = flat_ldr.min(axis=0).astype(np.float32)
    target_max = flat_ldr.max(axis=0).astype(np.float32)
    print(f"[HDRI] source LDR per-channel range: "
          f"min={target_min.tolist()}, max={target_max.tolist()}")

    # ---- 1. Standard training grid ----
    new_lats, new_lons = _standard_grid(n_lat, n_lon)
    print(f"[HDRI] train: resampling to {n_lat}x{n_lon}")
    rgb_train = _resample_equirect_at(rgb, new_lats, new_lons)
    print(f"[HDRI] tone-mapping (exposure={exposure}, gamma={gamma})")
    ldr_train = _tonemap(rgb_train, exposure=exposure, gamma=gamma)
    
    save_standardized(
        output_filepath,
        lats_deg=new_lats,
        lons_deg=new_lons,
        signal=ldr_train,  # (H, W, 3)
        extra_attrs={
            'source': f'Poly Haven HDRI, EXR 2K: {os.path.basename(input_filepath)}',
            'units': 'LDR RGB in [0, 1]',
            'tone_map': f'clip>=0, *exposure={exposure}, Reinhard x/(1+x), gamma={gamma}',
            'preprocess': f'bilinear resample to {n_lat}x{n_lon}, tone-map to LDR',
            'grid_role': 'train',
        },
        target_min=target_min,
        target_max=target_max,
    )
    sanity_check_standardized(output_filepath)
    print(f"[HDRI] wrote training file {output_filepath}")

    # ---- 2. Half-pixel-offset held-out grid (sampled from same EXR) ----
    if held_out_filepath:
        held_lats, held_lons = _held_out_grid(n_lat, n_lon)
        print(f"[HDRI] held_out: resampling to "
              f"{len(held_lats)}x{len(held_lons)}")
        rgb_held = _resample_equirect_at(rgb, held_lats, held_lons)
        ldr_held = _tonemap(rgb_held, exposure=exposure, gamma=gamma)
        save_standardized(
            held_out_filepath,
            lats_deg=held_lats,
            lons_deg=held_lons,
            signal=ldr_held,
            extra_attrs={
                'source': f'Poly Haven HDRI, EXR 2K: {os.path.basename(input_filepath)}',
                'units': 'LDR RGB in [0, 1]',
                'tone_map': f'clip>=0, *exposure={exposure}, Reinhard x/(1+x), gamma={gamma}',
                'preprocess': (
                    f'bilinear resample to {n_lat-1}x{n_lon-1} half-pixel-offset '
                    f'grid from SOURCE EXR, tone-map to LDR'
                ),
                'grid_role': 'held_out',
                'train_shape': f'{n_lat}x{n_lon}',
            },
            target_min=target_min,
            target_max=target_max,
        )
        sanity_check_standardized(held_out_filepath)
        print(f"[HDRI] wrote held-out file {held_out_filepath}")

