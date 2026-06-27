"""
Unit tests for src/config/architectures.py — verify the locked
(activation × PE) grid + KAN row match preregistration §3.2 / §3.3 exactly.

Run from repo root:
    PYTHONPATH=src python tests/test_architectures.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.architectures import (                                       # noqa: E402
    ACTIVATIONS, PE_CELLS, DISPLAY_NAMES, INR_BENCH_BASELINES,
    cell_keys, cell_config, cell_cli_args, display_name
)


EXPECTED_ACTS = ('relu', 'scaled_sine', 'gaussian')
EXPECTED_PE_CELLS = (
    'none_angular', 'none_cartesian', 'rff', 'sh', 'fkan',
)
EXPECTED_N_CELLS = len(EXPECTED_ACTS) * len(EXPECTED_PE_CELLS)


def test_grid_shape() -> None:
    """Grid is the full cross product of activations × PE cells."""
    print('\n[archs] grid shape ...')
    assert tuple(ACTIVATIONS.keys()) == EXPECTED_ACTS
    assert tuple(PE_CELLS.keys()) == EXPECTED_PE_CELLS
    keys = cell_keys()
    assert len(keys) == EXPECTED_N_CELLS, (
        f'expected {EXPECTED_N_CELLS} cells, got {len(keys)}'
    )
    print(f'  OK {EXPECTED_N_CELLS} cells = '
          f'{len(EXPECTED_ACTS)} × {len(EXPECTED_PE_CELLS)}.')


def test_mlp_shape_locked_per_activation() -> None:
    """Each activation cell yields the locked MLP shape (6 × 256, arch=mlp)."""
    print('\n[archs] MLP shape locked per activation ...')
    for act in EXPECTED_ACTS:
        cfg = cell_config(f'{act}__none_angular')
        assert cfg['arch'] == 'mlp', cfg
        assert cfg['mlp_num_layers'] == 6, cfg
        assert cfg['mlp_layer_width'] == 256, cfg
    print('  OK all 3 activations use arch=mlp at 6 × 256.')


def test_pe_cells_have_expected_ce_pe_pairs() -> None:
    """Each PE cell pins the right (ce, pe) combination."""
    print('\n[archs] PE cells have correct (ce, pe) pairs ...')
    expected = {
        'none_angular':   ('angular',             'None'),
        'none_cartesian': ('cartesian',           'None'),
        'rff':            ('cartesian',           'RFF'),
        'sh':             ('spherical-harmonics', 'None'),
        'fkan':           ('cartesian',           'FKAN'),
    }
    for pe_key, (ce, pe) in expected.items():
        cfg = PE_CELLS[pe_key]
        assert cfg['ce'] == ce, f'{pe_key}: ce={cfg["ce"]} vs {ce}'
        assert cfg['pe'] == pe, f'{pe_key}: pe={cfg["pe"]} vs {pe}'
    print('  OK all PE cells correctly configured.')


def test_fkan_locked_values() -> None:
    """FKAN PE PE Ω is locked."""
    print('\n[archs] FKAN Ω = 32 ...')
    assert PE_CELLS['fkan']['omega'] == 32
    print('  OK FKAN Ω = 32.')


def test_sh_locked_values() -> None:
    """SH PE L_max is locked."""
    print('\n[archs] SH L_max = 32 ...')
    print('\n[archs] SH L_max locked ...')
    lmax = PE_CELLS['sh']['sh_lmax']
    assert lmax == 32, f'expected sh_lmax=32, got {lmax}'
    print(f'  OK SH L_max = {lmax}, input dim = {(lmax+1) ** 2}.')


def test_rff_locked_values() -> None:
    """RFF σ and L (mapping_input) are locked."""
    print('\n[archs] RFF locked values ...')
    assert PE_CELLS['rff']['ffn_scale'] == 10.0
    assert PE_CELLS['rff']['mapping_input'] == 32
    print('  OK RFF σ = 10, mapping_input = 32.')

def test_inr_bench_baselines_confirmed() -> None:
    """The handful of confirmed INR-Bench Table III baselines are wired in."""
    print('\n[archs] INR-Bench baselines ...')
    assert INR_BENCH_BASELINES['scaled_sine__none_angular'] == 44.44
    assert INR_BENCH_BASELINES['relu__rff'] == 33.65
    assert INR_BENCH_BASELINES['gaussian__fkan'] == 34.70
    assert INR_BENCH_BASELINES['fourier_kan'] == 33.56
    print('  OK confirmed Table III rows wired.')


def test_cli_args_well_formed() -> None:
    """CLI args alternate --flag/value and parse cleanly."""
    print('\n[archs] CLI args well-formed ...')
    for k in cell_keys():
        args = cell_cli_args(k)
        assert len(args) % 2 == 0, f'{k}: odd arg count {len(args)}'
        flags = args[0::2]
        values = args[1::2]
        for f in flags:
            assert f.startswith('--'), f'{k}: malformed flag {f!r}'
        assert len(set(flags)) == len(flags), f'{k}: duplicate flag in {flags}'
        for v in values:
            assert isinstance(v, str) and v != '', f'{k}: empty value'
    print(f'  OK all {len(cell_keys())} cells CLI-well-formed.')


def test_unknown_cell_raises() -> None:
    """cell_cli_args / cell_config raise on unknown key."""
    print('\n[archs] unknown key raises ValueError ...')
    for fn in (cell_cli_args, cell_config):
        try:
            fn('not_a_real_cell')
        except ValueError as e:
            print(f"  OK {fn.__name__} raised: {e}")
            continue
        raise AssertionError(f'{fn.__name__} should have raised')


def test_display_names_are_unique() -> None:
    """Every cell has a non-empty, distinct display name."""
    print('\n[archs] display names ...')
    names = [display_name(k) for k in cell_keys()]
    for k, n in zip(cell_keys(), names):
        assert n, f'{k}: empty display name'
    assert len(set(names)) == len(names), 'duplicate display names'
    assert DISPLAY_NAMES['gaussian__fkan'] == 'Gaussian + FKAN'
    print(f'  OK {len(names)} distinct display names.')


def main() -> None:
    print('== Architecture configurations ==')
    test_grid_shape()
    test_mlp_shape_locked_per_activation()
    test_pe_cells_have_expected_ce_pe_pairs()
    test_fkan_locked_values()
    test_sh_locked_values()
    test_rff_locked_values()
    test_inr_bench_baselines_confirmed()

    print('\n== CLI args generation ==')
    test_cli_args_well_formed()
    test_unknown_cell_raises()
    test_display_names_are_unique()

    print('\nAll architecture-config tests passed.')


if __name__ == '__main__':
    main()
