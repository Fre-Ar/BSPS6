"""CLI argument parser for `src/main.py`.

Every flag here is wired into something downstream. Flags inherited from
INR-Bench that this benchmark does not exercise (KAN architectures and
basis functions; NeRF positional encoding; activations outside the
{ReLU, ScaledSine, Gaussian} trio; coordinate encodings
outside the five PE cells) have been removed.
"""

import argparse
from config.constants import (
    ELEVATION_DATA_PATH, CE_CHOICES, DATASET_CHOICES, DATASET_CONFIG,
)


def get_opts():
    parser = argparse.ArgumentParser()
    # ---- MLP shape ---------------------------------------------------------
    parser.add_argument('--mlp_num_layers', type=int, default=6,
                        help='Number of MLP layers (incl. output).')
    parser.add_argument('--mlp_layer_width', type=int, default=256,
                        help='Hidden width per MLP layer.')
    
    # ---- Activation --------------------------------------------------------
    parser.add_argument('--act', type=str, default='scaled-sine',
                        choices=['relu', 'scaled-sine', 'gaussian'],
                        help='MLP activation function (preregistration §3.2).')
    parser.add_argument('--sine_w0', type=float, default=30.0,
                        help='First-layer ω for ScaledSine init (SIREN-style).')
    parser.add_argument('--sine_w', type=float, default=30.0,
                        help='Hidden-layer ω for ScaledSine activations.')
    parser.add_argument('--gaussian_a', type=float, default=0.1,
                        help='Parameter a for the Gaussian activation exp(-x²/(2a²)).')

    # ---- Coordinate encoding (input feature representation) ----------------
    parser.add_argument('--ce', type=str, default='angular',
                        choices=CE_CHOICES,
                        help='Coordinate encoding fed to the (optional) PE / MLP.')
    parser.add_argument('--sh_lmax', type=int, default=32,
                        help='Max SH degree for --ce spherical-harmonics. '
                             'PE output dim becomes (L_max+1)^2.')
    
    # ---- Positional encoding ----------------------------------------------
    parser.add_argument('--pe', type=str, default='None',
                        choices=['None', 'RFF', 'FKAN'],
                        help='Positional encoding applied to the coord-encoded input.')
    # RFF PE hyperparameters (INR-Bench Appendix).
    parser.add_argument('--ffn_scale', type=float, default=10.0,
                        help='σ for RFF Gaussian projection (INR-Bench RFF).')
    parser.add_argument('--mapping_input', type=int, default=32,
                        help='L for RFF (output dim = 2L).')
    # FKAN PE hyperparameters (Li et al. 2025 Eq. 6/7).
    parser.add_argument('--omega', type=int, default=32,
                        help='Maximum frequency threshold Ω for FKAN PE '
                             '(integer frequencies 1..Ω per coord). '
                             'PE output dim is 2 · in_features · Ω.')

    # ---- Task shape (auto-set from --dataset; rarely overridden) ----------
    parser.add_argument('--in_features', type=int, default=2,
                        help='Input dim to the MLP. Auto-set from --ce.')
    parser.add_argument('--out_features', type=int, default=1,
                        help='Output dim (1 for scalar signals, 3 for RGB). '
                             'Auto-set from --dataset.')

    # ---- Training ---------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='Batch size (INR-Bench default).')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Initial learning rate (INR-Bench default).')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (INR-Bench default).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed (preregistration §3.5).')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20,
                        help='Validation cadence (INR-Bench default).')

    # ---- Dataset routing --------------------------------------------------
    parser.add_argument('--dataset', type=str, default='etopo1',
                        choices=DATASET_CHOICES,
                        help='Which pre-processed benchmark dataset to train on. '
                             'Auto-sets data_path / held_out_path / out_features.')
    parser.add_argument('--data_path', type=str, default=ELEVATION_DATA_PATH,
                        help='Path to the pre-processed training .nc file. '
                             'Overrides --dataset.')
    parser.add_argument('--held_out_path', type=str, default='',
                        help='Path to the 511×1023 held-out .nc file '
                             '(preregistration §3.4). Empty disables held-out '
                             'evaluation. Auto-resolved from --dataset.')

    # ---- Logging / outputs ------------------------------------------------
    parser.add_argument('--runs_csv', type=str, default='results/runs.csv',
                        help='Path to append the per-run results row '
                             '(preregistration §3.6). Empty string disables.')
    parser.add_argument('--save_dir', type=str, default='logs/image_regression',
                        help='TensorBoard / checkpoint log dir.')
    parser.add_argument('--save_vis', default=True, action='store_true',
                        help='Save per-validation prediction snapshots.')
    parser.add_argument('--vis_every', type=int, default=200,
                        help='Snapshot cadence (epochs) when --save_vis.')
    parser.add_argument('--exp_name', type=str, default='None',
                        help='Optional human-readable experiment label.')

    # ---- Resolve dataset → paths / out_features ---------------------------
    import sys as _sys
    hparams = parser.parse_args()
    explicit = set(a.split('=')[0] for a in _sys.argv[1:] if a.startswith('--'))
    cfg = DATASET_CONFIG[hparams.dataset]
    if '--data_path' not in explicit:
        hparams.data_path = cfg['path']
    if '--out_features' not in explicit:
        hparams.out_features = cfg['out_features']
    if '--held_out_path' not in explicit:
        hparams.held_out_path = cfg.get('held_out_path', '')

    # ---- Resolve coord-encoding kwargs + auto-set in_features -------------
    from datasets.coord_encodings import coord_encoding_dim
    ce_resolved = hparams.ce if hparams.ce != 'None' else 'angular'
    ce_kwargs = _encoding_kwargs_from_hparams(hparams, ce_resolved)
    if '--in_features' not in explicit:
        hparams.in_features = coord_encoding_dim(ce_resolved, **ce_kwargs)
    hparams.ce_resolved = ce_resolved
    hparams.encoding_kwargs = ce_kwargs
    return hparams


def _encoding_kwargs_from_hparams(hparams, ce_name: str) -> dict:
    """Pull per-encoding hyperparameters off hparams into a plain dict."""
    if ce_name == 'spherical-harmonics':
        return {'L_max': hparams.sh_lmax}
    return {}
