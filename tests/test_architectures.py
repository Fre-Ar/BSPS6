"""
Unit tests for src/config/architectures.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.architectures import (                                       # noqa: E402
    ARCHITECTURES,
    INR_BENCH_BASELINES,
    DISPLAY_NAMES,
    architecture_keys,
    architecture_cli_args,
    display_name,
    inr_bench_baseline,
)


EXPECTED_KEYS = ('scaled_sine_mlp', 'relu_rff_mlp', 'fourier_kan', 'gaussian_fkan')


def test_exactly_four_architectures() -> None:
    """The preregistration locks exactly 4 architectures, no more, no less."""
    print('\n[archs] exactly 4 architectures ...')
    assert tuple(ARCHITECTURES.keys()) == EXPECTED_KEYS, (
        f'expected {EXPECTED_KEYS}, got {tuple(ARCHITECTURES.keys())}'
    )
    assert tuple(architecture_keys()) == EXPECTED_KEYS
    print(f'  OK 4 archs in canonical order: {EXPECTED_KEYS}.')


def test_scaled_sine_mlp_locked_values() -> None:
    """ScaledSine MLP — INR-Bench Table III 'ScaledSine + Id.' = 44.44 dB."""
    cfg = ARCHITECTURES['scaled_sine_mlp']
    assert cfg['arch'] == 'mlp'
    assert cfg['act'] == 'scaled-sine'
    assert cfg['pe'] == 'None'
    assert cfg['mlp_num_layers'] == 6
    assert cfg['mlp_layer_width'] == 256
    assert cfg['sine_w0'] == 30.0
    assert INR_BENCH_BASELINES['scaled_sine_mlp'] == 44.44
    print('  OK ScaledSine MLP locked.')


def test_relu_rff_mlp_locked_values() -> None:
    """ReLU + RFF MLP — INR-Bench Table III 'ReLU + RFF' = 33.65 dB."""
    cfg = ARCHITECTURES['relu_rff_mlp']
    assert cfg['arch'] == 'mlp'
    assert cfg['act'] == 'relu'
    assert cfg['pe'] == 'RFF'
    assert cfg['mlp_num_layers'] == 6
    assert cfg['mlp_layer_width'] == 256
    assert cfg['ffn_scale'] == 10.0
    assert cfg['mapping_input'] == 32
    assert INR_BENCH_BASELINES['relu_rff_mlp'] == 33.65
    print('  OK ReLU + RFF MLP locked.')


def test_fourier_kan_locked_values() -> None:
    """Fourier KAN — INR-Bench Table III 'Fourier' = 33.56 dB.
    Chosen over B-Spline KAN (21.99 dB) for better PSNR + lower memory."""
    cfg = ARCHITECTURES['fourier_kan']
    assert cfg['arch'] == 'kan'
    assert cfg['act'] == 'fourier'
    assert cfg['kan_num_layers'] == 6
    assert cfg['kan_layer_width'] == 64
    assert INR_BENCH_BASELINES['fourier_kan'] == 33.56
    print('  OK Fourier KAN locked.')


def test_gaussian_fkan_locked_values() -> None:
    """Gaussian + FKAN — INR-Bench Table III 'Gaussian + FKAN' = 34.70 dB.
    Implemented as arch=kamp (KAN-then-MLP), kan_act=fourier (FKAN PE per
    INR-Bench Eq. 6), mlp_act=gaussian."""
    cfg = ARCHITECTURES['gaussian_fkan']
    assert cfg['arch'] == 'kamp'
    assert cfg['kan_act'] == 'fourier'
    assert cfg['mlp_act'] == 'gaussian'
    assert cfg['gaussian_a'] == 0.1
    assert cfg['mlp_num_layers'] == 6
    assert cfg['mlp_layer_width'] == 256
    assert cfg['kan_num_layers'] == 6
    assert cfg['kan_layer_width'] == 64
    assert INR_BENCH_BASELINES['gaussian_fkan'] == 34.70
    print('  OK Gaussian + FKAN locked.')


def test_cli_args_well_formed() -> None:
    """CLI args alternate --flag/value and parse cleanly."""
    print('\n[archs] CLI args well-formed ...')
    for k in EXPECTED_KEYS:
        args = architecture_cli_args(k)
        assert len(args) % 2 == 0, f'{k}: odd arg count {len(args)}'
        flags = args[0::2]
        values = args[1::2]
        for f in flags:
            assert f.startswith('--'), f'{k}: malformed flag {f!r}'
        # No duplicate flags within an architecture.
        assert len(set(flags)) == len(flags), f'{k}: duplicate flag in {flags}'
        # All values are non-empty strings.
        for v in values:
            assert isinstance(v, str) and v != '', f'{k}: empty value'
        print(f'  OK {k}: {len(flags)} flags, {len(values)} values.')


def test_cli_args_match_dict_contents() -> None:
    """architecture_cli_args is a faithful translation of the dict."""
    for k in EXPECTED_KEYS:
        args = architecture_cli_args(k)
        flags = args[0::2]
        values = args[1::2]
        for flag, value, (expect_k, expect_v) in zip(
            flags, values, ARCHITECTURES[k].items(),
        ):
            assert flag == f'--{expect_k}', f'flag mismatch in {k}: {flag}'
            assert value == str(expect_v), f'value mismatch in {k}: {value}'


def test_unknown_architecture_raises() -> None:
    """architecture_cli_args / display_name / baseline raise on unknown key."""
    print('\n[archs] unknown key raises ValueError ...')
    for fn in (architecture_cli_args, display_name, inr_bench_baseline):
        try:
            fn('not_a_real_arch')
        except ValueError as e:
            print(f"  OK {fn.__name__} raised: {e}")
            continue
        raise AssertionError(f'{fn.__name__} should have raised')


def test_display_names_complete() -> None:
    """Every architecture has a display name."""
    print('\n[archs] display names defined for all archs ...')
    for k in EXPECTED_KEYS:
        name = display_name(k)
        assert name and ' ' in name, f'{k}: bad display name {name!r}'
    print(f'  OK display names: {[DISPLAY_NAMES[k] for k in EXPECTED_KEYS]}.')


def main() -> None:
    print('== Architecture configurations ==')
    test_exactly_four_architectures()
    test_scaled_sine_mlp_locked_values()
    test_relu_rff_mlp_locked_values()
    test_fourier_kan_locked_values()
    test_gaussian_fkan_locked_values()

    print('\n== CLI args generation ==')
    test_cli_args_well_formed()
    test_cli_args_match_dict_contents()
    test_unknown_architecture_raises()
    test_display_names_complete()

    print('\nAll architecture-config tests passed.')


if __name__ == '__main__':
    main()
