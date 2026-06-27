"""The locked grid of (activation × positional-encoding) Coordinate-MLP cells.

The benchmark is a 2-dimensional factorization over Coordinate-MLPs, mirroring
INR-Bench's (Li et al. 2025, Table III) own decomposition. The two design
dimensions are:

  * activation function: ReLU, ScaledSine, Gaussian.
  * positional encoding: Identity on angular coordinates, Identity on
    Cartesian coordinates, RFF, Spherical Harmonics, FKAN.

Crossing them gives every cell in the grid. Each cell exposes:

  * a unique `cell_key` (the canonical run identifier in runs.csv),
  * a list of CLI flag overrides for `src/main.py`,
  * a human-readable display name for plots and reports,
  * an INR-Bench Table III baseline number where directly comparable.
"""
from __future__ import annotations

from typing import Any


# ----- Common MLP shape (shared across all cells) ---------------------------
# Matches INR-Bench Coordinate-MLP defaults (Appendix "Network Settings":
# 6-layer MLP, hidden width 256).
_MLP_SHAPE: dict[str, Any] = {
    'arch':            'mlp',
    'mlp_num_layers':  6,
    'mlp_layer_width': 256,
}


# ----- Activations ----------------------------------------------------------
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


# ----- PE cells -------------------------------------------------------------
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
    # FKAN PE — single Fourier feature mapping with trainable per-(d, ω)
    # coefficients (a_{iω}, b_{iω}); see INR-Bench Eq. 6/7 and
    # `src/models/encodings/fkan_encoding.py`. Applied to Cartesian inputs
    # (D = 3). Ω is matched to the spectral reach of SH (L_max = 32) and
    # RFF (L = 32) so the three frequency-aware PEs share one scale — see
    # preregistration §3.3 "On FKAN Ω".
    'fkan': {
        'ce':    'cartesian',
        'pe':    'FKAN',
        'omega': 32,            # PE output dim 2 · D · Ω = 192.
    },
}


# ----- INR-Bench published baselines (preregistration §3.2) -----------------
# Image Regression column of Table III (Li et al. 2025), where directly
# comparable. Soft references for pilot sanity-checks only — Euclidean
# 2D image regression is not strictly identical to spherical regression,
# so we do NOT use these as pass/fail targets. Filled where the value has
# been cross-referenced against the paper; the rest left as None.
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
    act_key, pe_key = cell_key.split('__')
    return f'{_ACT_DISPLAY[act_key]} + {_PE_DISPLAY[pe_key]}'



# ----- Public helpers -------------------------------------------------------
def cell_keys() -> tuple[str, ...]:
    """The canonical iteration order: activation outer, PE inner."""
    return tuple(
        f'{act_key}__{pe_key}'
        for act_key in ACTIVATIONS
        for pe_key in PE_CELLS
    )

def cell_config(cell_key: str) -> dict[str, Any]:
    """Return the merged flag-override dict for `cell_key`.

    Composed as `_MLP_SHAPE ∪ ACTIVATIONS[act] ∪ PE_CELLS[pe]`.
    """
    if '__' not in cell_key:
        raise ValueError(
            f"Unknown cell_key '{cell_key}'. Available: {list(cell_keys())}"
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


# Derived constants (handy for tests and reports — they always stay in sync
# with cell_keys() and cell_config()).
ARCHITECTURES = {k: cell_config(k) for k in cell_keys()}
DISPLAY_NAMES = {k: display_name(k) for k in cell_keys()}
