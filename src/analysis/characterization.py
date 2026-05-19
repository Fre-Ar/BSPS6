"""
Dataset characterization metrics — locked values.

These are the features the H4 regression uses to predict PSNR. Re-running
`src/datasets/characteristics.py` should reproduce these to within float error;
if they drift, this file must be amended.

For RGB datasets, characterization is computed on Rec.709 luminance for the
H4 feature vector. Per-channel values are tabulated below for the deferred
H7 chromatic analysis (preregistration scope-out) but are NOT used in H4.
"""
from __future__ import annotations


# (L_95, CV, P99_norm)
DATASET_METRICS: dict[str, dict[str, float]] = {
    'cmb': {
        'L_95':     236.0,
        'CV':       0.162,
        'P99_norm': 0.116,
    },
    'era5': {
        'L_95':     13.0,
        'CV':       1.202,
        'P99_norm': 0.048,
    },
    'etopo1': {
        'L_95':     45.0,
        'CV':       0.654,
        'P99_norm': 0.129,
    },
    'hdri_sky': {
        'L_95':     31.0,        # luminance L_95
        'CV':       0.656,
        'P99_norm': 0.144,
    },
    'hdri_urban': {
        'L_95':     97.0,        # luminance L_95
        'CV':       1.124,
        'P99_norm': 0.218,
    },
}

# Per-channel L_95 for the RGB datasets — reference only (H7, out of scope here).
RGB_PER_CHANNEL_L95: dict[str, dict[str, float]] = {
    'hdri_sky':   {'r': 40.0, 'g': 29.0, 'b': 16.0},
    'hdri_urban': {'r': 95.0, 'g': 98.0, 'b': 134.0},
}

FEATURE_NAMES: tuple[str, ...] = ('L_95', 'CV', 'P99_norm')


def feature_vector(dataset: str) -> list[float]:
    """The (L_95, CV, P99_norm) row used by H4's regression for `dataset`."""
    if dataset not in DATASET_METRICS:
        raise KeyError(f"No characterization for dataset '{dataset}'.")
    m = DATASET_METRICS[dataset]
    return [m[f] for f in FEATURE_NAMES]
