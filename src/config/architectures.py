"""The locked grid of (activation × positional-encoding) Coordinate-MLP cells
plus the standalone Fourier Coordinate-KAN row.

This file replaces the previous (architecture × coordinate-encoding) grid.
The new structure mirrors INR-Bench's own factor decomposition for Coordinate
models (Li et al. 2025, Table III): for the MLP family, the design dimensions
are *activation function* and *positional encoding*; Coordinate-KANs are
reported as a separate row family without positional encoding.

Concretely, this benchmark uses
  * 3 activations: ReLU, ScaledSine, Gaussian
  * 5 PE cells:    None+angular, None+cartesian, RFF, SH, FKAN
  * 1 KAN row:     Fourier Coordinate-KAN (no PE)
giving 3 × 5 + 1 = 16 cells per dataset.

Each cell exposes:
  * a unique `cell_key` (the canonical run identifier in runs.csv);
  * a list of CLI flag overrides for `src/main.py`;
  * a human-readable display name for plots/reports;
  * an INR-Bench Table III baseline number where directly comparable.
"""
from __future__ import annotations

from typing import Any


# ----- Common MLP shape (shared across all 15 MLP cells) --------------------
# Matches INR-Bench Coordinate-MLP defaults (Appendix "Network Settings":
# 6-layer MLP, hidden width 256).
_MLP_SHAPE: dict[str, Any] = {
    'arch':            'mlp',
    'mlp_num_layers':  6,
    'mlp_layer_width': 256,
}


# ----- Activations (n = 3) --------------------------------------------------
# Names are taken from INR-Bench Table III rows. Values not specified by
# INR-Bench (e.g. trainable-scale defaults for ScaledSine) follow our
# opts.py defaults.
ACTIVATIONS: dict[str, dict[str, Any]] = {
    'relu': {
        'act': 'relu',
    },
    'scaled_sine': {
        'act':     'scaled-sine',
        'sine_w0': 30.0,
        'sine_w':  30.0,
    },
    'gaussian': {
        'act':        'gaussian',
        'gaussian_a': 0.1,
    },
}


# ----- PE cells (n = 5) -----------------------------------------------------
# Each PE cell pins (ce, pe, encoding hyperparams). The first two cells
# (`none_angular` and `none_cartesian`) play the role of INR-Bench's
# "Identity" PE row on the sphere — they feed raw coordinates to the MLP,
# with two choices of coordinate system. The remaining three are proper
# positional encodings.
PE_CELLS: dict[str, dict[str, Any]] = {
    # No PE, raw (lat, lon) in radians. INR-Bench-Id analog on (λ, φ).
    'none_angular': {
        'ce': 'angular',
        'pe': 'None',
    },
    # No PE, raw (x, y, z) on the unit sphere. INR-Bench-Id analog on R^3.
    'none_cartesian': {
        'ce': 'cartesian',
        'pe': 'None',
    },
    # RFF PE applied to Cartesian inputs. Equivalent in spirit to
    # INR-Bench's RFF row, lifted to the sphere via (x, y, z) instead of
    # 2D (u, v) to sidestep polar singularities.
    'rff': {
        'ce':            'cartesian',
        'pe':            'RFF',
        'ffn_scale':     10.0,   # σ; matches INR-Bench Appendix.
        'mapping_input': 32,     # L = 32; matches INR-Bench Appendix.
    },
    # Spherical-harmonic features. SH must operate on the angular base
    # coordinates by construction (Y_l^m is a function of (θ, φ)).
    'sh': {
        'ce':      'spherical-harmonics',
        'pe':      'None',
        'sh_lmax': 32,           # PE output dim (L_max+1)^2 = 1089.
    },
    # FKAN PE — faithful implementation (single Fourier feature map with
    # trainable a, b per (d, ω), Ω = 1024). Applied to Cartesian inputs
    # (D = 3 → PE output 2 · 3 · 1024 = 6144).
    # INR-Bench Appendix: "For FKAN positional encoding, the maximum
    # frequency threshold Ω is set to 1024."
    'fkan': {
        'ce':    'cartesian',
        'pe':    'FKAN',
        'omega': 32,
    },
}


# ----- Standalone Fourier Coordinate-KAN row --------------------------------
# INR-Bench Coordinate-KANs are evaluated without an additional positional
# encoding. Layer count and width match
# INR-Bench Appendix: "All Coordinate-KANs are 6-layer KAN networks with a
# width of 64." Fed Cartesian (x, y, z) so the KAN basis isn't asked to
# absorb angular coordinate singularities.
KAN_ROW: dict[str, dict[str, Any]] = {
    'fourier_kan': {
        'arch':             'kan',
        'act':              'fourier',
        'ce':               'cartesian',
        'pe':               'None',
        'kan_num_layers':   6,
        'kan_layer_width':  64,
        'input_grid_size':  8,
        'hidden_grid_size': 8,
        'output_grid_size': 8,
    },
}


# ----- INR-Bench published baselines (preregistration §3.2) -----------------
# Image Regression column of Table III (Li et al. 2025), where directly
# comparable. Soft references for pilot sanity-checks only — Euclidean
# 2D image regression is not strictly identical to spherical regression,
# so we do NOT use these as pass/fail targets.
INR_BENCH_BASELINES: dict[str, float | None] = {
    # ReLU row
    'relu__none_angular':     None,
    'relu__none_cartesian':   None,
    'relu__rff':              33.65,  # ReLU + RFF, Table III (prereg §3.2).
    'relu__sh':               None,
    'relu__fkan':             None,
    # ScaledSine row
    'scaled_sine__none_angular':   44.44,  # ScaledSine + Id., Table III.
    'scaled_sine__none_cartesian': None,
    'scaled_sine__rff':            None,
    'scaled_sine__sh':             None,
    'scaled_sine__fkan':           None,
    # Gaussian row
    'gaussian__none_angular':   None,
    'gaussian__none_cartesian': None,
    'gaussian__rff':            None,
    'gaussian__sh':             None,
    'gaussian__fkan':           34.70,  # Gaussian + FKAN, Table III.
    # KAN row
    'fourier_kan':              33.56,  # Fourier Coordinate-KAN, Table III.
}

# ----- Display names --------------------------------------------------------
_ACT_DISPLAY = {
    'relu':         'ReLU',
    'scaled_sine':  'ScaledSine',
    'gaussian':     'Gaussian',
}
_PE_DISPLAY = {
    'none_angular':   'Id. (angular)',
    'none_cartesian': 'Id. (cartesian)',
    'rff':            'RFF',
    'sh':             'SH',
    'fkan':           'FKAN',
}


def display_name(cell_key: str) -> str:
    """Human-readable name for plots / reports."""
    if cell_key in KAN_ROW:
        return 'Fourier KAN'
    act_key, pe_key = cell_key.split('__')
    return f'{_ACT_DISPLAY[act_key]} + {_PE_DISPLAY[pe_key]}'



# ----- Public helpers -------------------------------------------------------
def cell_keys() -> tuple[str, ...]:
    """The canonical iteration order for the 16 cells.

    Order: all MLP cells (activation outer, PE inner), then the KAN row.
    """
    mlp_keys = [
        f'{act_key}__{pe_key}'
        for act_key in ACTIVATIONS
        for pe_key in PE_CELLS
    ]
    return tuple(mlp_keys + list(KAN_ROW.keys()))


def is_kan_cell(cell_key: str) -> bool:
    """True if the cell is the standalone Coordinate-KAN row."""
    return cell_key in KAN_ROW


def cell_config(cell_key: str) -> dict[str, Any]:
    """Return the merged flag-override dict for `cell_key`.

    For MLP cells this is `_MLP_SHAPE ∪ ACTIVATIONS[act] ∪ PE_CELLS[pe]`;
    for the KAN row it is `KAN_ROW[cell_key]` directly.
    """
    if cell_key in KAN_ROW:
        return dict(KAN_ROW[cell_key])
    if '__' not in cell_key:
        raise ValueError(
            f"Unknown cell_key '{cell_key}'. "
            f"Available: {list(cell_keys())}"
        )
    act_key, pe_key = cell_key.split('__', 1)
    if act_key not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{act_key}' in cell_key '{cell_key}'. "
            f"Available: {sorted(ACTIVATIONS)}"
        )
    if pe_key not in PE_CELLS:
        raise ValueError(
            f"Unknown PE cell '{pe_key}' in cell_key '{cell_key}'. "
            f"Available: {sorted(PE_CELLS)}"
        )
    return {**_MLP_SHAPE, **ACTIVATIONS[act_key], **PE_CELLS[pe_key]}


def cell_cli_args(cell_key: str) -> list[str]:
    """Return CLI args (alternating --flag value pairs) for `cell_key`.

    Designed to be spliced into `subprocess.run([..., 'main.py', *args])`
    by the launcher.
    """
    args: list[str] = []
    for flag, value in cell_config(cell_key).items():
        args.extend([f'--{flag}', str(value)])
    return args


def inr_bench_baseline(cell_key: str) -> float | None:
    """INR-Bench published image-regression PSNR (dB) for `cell_key`,
    or None if no directly comparable number is tabulated."""
    return INR_BENCH_BASELINES.get(cell_key)


# ----- Backward-compat shims ------------------------------------------------
# Older code paths (and tests) referenced the previous ARCHITECTURES /
# architecture_keys / architecture_cli_args names. They now resolve to the
# new cell terminology so we don't have to flush every import site at once.
ARCHITECTURES = {k: cell_config(k) for k in cell_keys()}
DISPLAY_NAMES = {k: display_name(k) for k in cell_keys()}


def architecture_keys() -> tuple[str, ...]:
    """Alias for `cell_keys()` (preserves the old import surface)."""
    return cell_keys()


def architecture_cli_args(cell_key: str) -> list[str]:
    """Alias for `cell_cli_args(cell_key)` (preserves the old import surface)."""
    return cell_cli_args(cell_key)
