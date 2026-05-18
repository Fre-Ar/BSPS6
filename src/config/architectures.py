"""
The 4 locked architecture configurations.

Each entry maps a short config-key to a dict of CLI flag overrides. Keys are
the long-form `--flag` names (without the leading `--`); values are passed
through `str(...)` when generating CLI args.
"""
from __future__ import annotations

from typing import Any


# ----- Locked architecture configurations -----------------------------------
# CLI flag → value. The exact flag names match those defined in
# `src/config/opts.py`. Only flags that differ from opts.py defaults need to
# be present here, but we include the locked values explicitly so the row
# in runs.csv (which records them) is unambiguous and the configs are
# self-documenting.

ARCHITECTURES: dict[str, dict[str, Any]] = {
    # ScaledSine MLP
    # INR-Bench Table III "ScaledSine + Id." = 44.44 dB on image regression.
    'scaled_sine_mlp': {
        'arch':              'mlp',
        'act':               'scaled-sine',
        'pe':                'None',
        'mlp_num_layers':    6,
        'mlp_layer_width':   256,
        'sine_w0':           30.0,
        'sine_w':            30.0,
    },

    # ReLU + RFF MLP
    # INR-Bench Table III "ReLU + RFF" = 33.65 dB on image regression.
    'relu_rff_mlp': {
        'arch':              'mlp',
        'act':               'relu',
        'pe':                'RFF',
        'mlp_num_layers':    6,
        'mlp_layer_width':   256,
        'ffn_scale':         10.0,
        'mapping_input':     32,
    },

    # Fourier KAN
    # INR-Bench Table III "Fourier" (Coordinate-KAN, Eq. 21) = 33.56 dB.
    # Selected over B-Spline KAN (21.99 dB) for better PSNR, ~half the
    # memory, and ~60% the wall-clock per INR-Bench's reported timings.
    'fourier_kan': {
        'arch':              'kan',
        'act':               'fourier',
        'kan_num_layers':    6,
        'kan_layer_width':   64,
    },

    # Gaussian + FKAN 
    # INR-Bench Table III "Gaussian + FKAN" = 34.70 dB on image regression.
    # INR-Bench's "FKAN PE" is dispatched through their `arch="kamp"` branch:
    # the leading Fourier-basis KAN front-end *is* the FKAN PE (per their
    # Eq. 6); the MLP back-end is the regressor. `act` is unused for kamp
    # but is set to 'gaussian' so it round-trips cleanly through runs.csv.
    'gaussian_fkan': {
        'arch':              'kamp',
        'act':               'gaussian',
        'kan_act':           'fourier',
        'mlp_act':           'gaussian',
        'mlp_num_layers':    6,
        'mlp_layer_width':   256,
        'kan_num_layers':    6,
        'kan_layer_width':   64,
        'gaussian_a':        0.1,
    },
}


# ----- INR-Bench published baselines  -----------------
# For sanity-checking pilot runs: "ScaledSine + Id. on Euclidean image
# regression got 44.44 dB; we should get >35 dB on a sphere-as-image baseline,
# else the pipeline is broken." Used as soft references, never as targets we
# pre-commit to matching.

INR_BENCH_BASELINES: dict[str, float] = {
    'scaled_sine_mlp':  44.44,
    'relu_rff_mlp':     33.65,
    'fourier_kan':      33.56,
    'gaussian_fkan':    34.70,
}


# ----- Display names --------------------------------------------------------
DISPLAY_NAMES: dict[str, str] = {
    'scaled_sine_mlp': 'ScaledSine MLP',
    'relu_rff_mlp':    'ReLU + RFF MLP',
    'fourier_kan':     'Fourier KAN',
    'gaussian_fkan':   'Gaussian + FKAN',
}


# ----- Public helpers -------------------------------------------------------
def architecture_keys() -> tuple[str, ...]:
    """The canonical iteration order for the 4 architectures."""
    return tuple(ARCHITECTURES.keys())


def architecture_cli_args(arch_key: str) -> list[str]:
    """Return CLI args (alternating --flag value pairs) for `arch_key`.

    Designed to be spliced into a `subprocess.run([..., 'main.py', *args])`
    call by the launcher (Task 19).
    """
    if arch_key not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{arch_key}'. "
            f"Available: {sorted(ARCHITECTURES)}"
        )
    args: list[str] = []
    for flag, value in ARCHITECTURES[arch_key].items():
        args.extend([f'--{flag}', str(value)])
    return args


def display_name(arch_key: str) -> str:
    """Human-readable name for plotting / reporting."""
    if arch_key not in DISPLAY_NAMES:
        raise ValueError(f"Unknown architecture '{arch_key}'.")
    return DISPLAY_NAMES[arch_key]


def inr_bench_baseline(arch_key: str) -> float:
    """INR-Bench published image-regression PSNR (dB) for `arch_key`."""
    if arch_key not in INR_BENCH_BASELINES:
        raise ValueError(f"No INR-Bench baseline for '{arch_key}'.")
    return INR_BENCH_BASELINES[arch_key]
