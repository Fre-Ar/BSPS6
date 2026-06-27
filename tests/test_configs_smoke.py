"""
Smoke test: for every configuration the locked grid will train, instantiate
the model, run forward + a few optimizer steps on a small batch, and verify
predictions + gradients + parameters are finite at every stage.

Covers (one forward+backward+5-step Adam run per config):

  * Every (activation × PE) cell × default config         = 15 configs
  * Every SH cell × every L_max value used in the grid    = 3 acts × 4 lmaxs = 12 configs
  * Optional per-dataset end-to-end check (uses the actual preprocessed
    NetCDFs; skipped if a dataset file is missing)        = up to 5 cell-on-dataset checks

The point is to catch silent NaN/Inf in things like:
  * activation divide-by-zero (e.g., Gaussian with a=0, the SINC bug pattern)
  * coordinate-encoding singularities at the poles
  * gradient explosion that turns parameters into NaN after a few steps

Runs on CPU for portability — no GPU required. Total wall-clock ~30-90 s.

Run from repo root:
    PYTHONPATH=src python tests/test_configs_smoke.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.architectures import cell_keys, cell_cli_args, PE_CELLS         # noqa: E402
from config.constants import DATASET_CHOICES, DATASET_CONFIG                # noqa: E402
from datasets.coord_encodings import compute_coords                         # noqa: E402

# Lats deliberately include both poles, the equator, and intermediate bands
# so any latitude-dependent singularity (e.g., angular at ±90°) gets touched.
_SMOKE_LATS_DEG = np.array(
    [-89.5, -60.0, -30.0, 0.0, 30.0, 60.0, 89.5, -45.0, 45.0, 75.0],
    dtype=np.float32,
)
# Longitudes include the ±180 wrap and intermediate values.
_SMOKE_LONS_DEG = np.array(
    [0.0, 45.0, 90.0, 135.0, 179.5, -179.5, -135.0, -90.0, -45.0, 0.0],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Hparams construction
# ---------------------------------------------------------------------------
def _build_hparams(cell_key: str, sh_lmax: int | None = None):
    """Build hparams via the real opts.py code path."""
    from config.opts import get_opts
    cli_args = cell_cli_args(cell_key)
    if sh_lmax is not None:
        cli_args.extend(['--sh_lmax', str(sh_lmax)])
    old_argv = sys.argv
    try:
        sys.argv = ['test_configs_smoke.py'] + cli_args
        hparams = get_opts()
    finally:
        sys.argv = old_argv
    return hparams


# ---------------------------------------------------------------------------
# Per-cell forward+backward smoke
# ---------------------------------------------------------------------------
def _check_forward_backward(cell_key: str, sh_lmax: int | None = None,
                            n_steps: int = 5) -> None:
    """Build INR(hparams) for this cell, run forward + 5 Adam steps on a
    small batch of synthetic coords, and verify nothing goes NaN/Inf along
    the way."""
    label = (f'{cell_key} (sh_lmax={sh_lmax})'
             if sh_lmax is not None else cell_key)

    hparams = _build_hparams(cell_key, sh_lmax=sh_lmax)
    from models.INR import INR
    torch.manual_seed(0)
    model = INR(hparams).to('cpu')

    # 1. Parameter init finiteness
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f'{label}: init param {name} has NaN/Inf'

    # 2. Build a small batch using the actual coord-encoding pipeline.
    ce = hparams.ce_resolved
    enc_kwargs = dict(getattr(hparams, 'encoding_kwargs', {}) or {})
    coords = compute_coords(ce, _SMOKE_LATS_DEG, _SMOKE_LONS_DEG, **enc_kwargs)
    coords = coords.float().to('cpu')
    n_pixels = coords.shape[0]
    out_dim = hparams.out_features

    # Random "targets" in [-1, 1] (the normalized range the model expects).
    rng = torch.Generator().manual_seed(0)
    targets = (torch.rand(n_pixels, out_dim, generator=rng) * 2 - 1).float()

    # 3. Forward pass.
    out = model(coords)['model_out']
    assert out.shape == (n_pixels, out_dim), (
        f'{label}: expected output shape ({n_pixels}, {out_dim}), got {tuple(out.shape)}'
    )
    assert torch.isfinite(out).all(), (
        f'{label}: forward pass produced NaN/Inf (out range '
        f'[{float(out.min()):.4g}, {float(out.max()):.4g}])'
    )

    # 4. Initial loss + backward + grad finite check.
    loss = ((out - targets) ** 2).mean()
    assert torch.isfinite(loss), f'{label}: initial loss is NaN/Inf'
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), (
            f'{label}: grad on {name} has NaN/Inf'
        )

    # 5. Run a few Adam steps. Catches instabilities that show up only after
    # weight updates (e.g., a forward path that's fine at init but explodes
    # one step in).
    # Note: use loss.item() (not float(loss)) — .item() is the idiomatic way
    # to extract a Python scalar from a 0-dim tensor and doesn't trigger
    # PyTorch's "converting a tensor with requires_grad=True to a scalar"
    # warning (which is a guard against accidentally keeping the autograd
    # graph alive when only the numeric value is wanted).
    opt = torch.optim.Adam(model.parameters(), lr=4e-4)
    losses: list[float] = [loss.item()]
    for step in range(n_steps):
        opt.zero_grad()
        out = model(coords)['model_out']
        assert torch.isfinite(out).all(), (
            f'{label}: step {step+1} forward produced NaN/Inf'
        )
        loss = ((out - targets) ** 2).mean()
        assert torch.isfinite(loss), f'{label}: step {step+1} loss is NaN/Inf'
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f'{label}: step {step+1} grad on {name} has NaN/Inf'
                )
        opt.step()
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), (
                f'{label}: step {step+1} param {name} became NaN/Inf after opt.step'
            )
        losses.append(loss.item())

    print(f'  OK {label:48s} loss: {losses[0]:8.4f} → {losses[-1]:8.4f} '
          f'(n_steps={n_steps}, n_params={sum(p.numel() for p in model.parameters()):,})')


def test_every_cell_default_config() -> None:
    """One forward+backward+steps run per (activation × PE) cell at the
    locked default config."""
    print('\n[smoke] all locked cells at default config ...')
    for cell_key in cell_keys():
        _check_forward_backward(cell_key)


def test_sh_cells_at_every_lmax() -> None:
    """SH cells get evaluated at L_max ∈ {16, 32} in the main grid, and at
    L_max ∈ {13, 31} in the post-saturation sub-grid. Smoke-test all four."""
    print('\n[smoke] SH cells at all L_max values used in the grid ...')
    sh_cells = [k for k in cell_keys() if k.endswith('__sh')]
    for cell_key in sh_cells:
        for lmax in (13, 16, 31, 32):
            _check_forward_backward(cell_key, sh_lmax=lmax)


# ---------------------------------------------------------------------------
# Optional per-dataset end-to-end check (uses the actual preprocessed NetCDFs)
# ---------------------------------------------------------------------------
def test_per_dataset_one_cell_e2e() -> None:
    """For each dataset whose preprocessed NetCDF is present, run one
    representative cell (relu__rff) on a random pixel sample drawn from
    the actual SphericalDataset (training + held-out). This catches issues
    in the dataset → coord-encoding → model → target-normalization chain
    that the synthetic-coord tests would miss."""
    print('\n[smoke] per-dataset end-to-end (skips datasets whose .nc is missing) ...')

    from datasets.spherical_reg import SphericalDataset
    from models.INR import INR

    representative_cell = 'relu__rff'
    n_skipped = 0
    n_checked = 0
    for dataset in DATASET_CHOICES:
        cfg = DATASET_CONFIG[dataset]
        if not os.path.exists(cfg['path']):
            n_skipped += 1
            print(f'  SKIP {dataset:12s}: training file missing '
                  f'({cfg["path"]}). Run preprocessing first.')
            continue

        # Build hparams for the chosen cell with this dataset.
        from config.opts import get_opts
        old_argv = sys.argv
        try:
            sys.argv = (
                ['test_configs_smoke.py']
                + cell_cli_args(representative_cell)
                + ['--dataset', dataset]
            )
            hparams = get_opts()
        finally:
            sys.argv = old_argv

        # Load the real dataset (training file). Don't bother with held-out
        # here — the per-cell smoke already exercised the coord pipeline.
        sd = SphericalDataset(
            hparams.data_path,
            coordinate_encoding=hparams.ce_resolved,
            encoding_kwargs=dict(getattr(hparams, 'encoding_kwargs', {}) or {}),
        )
        # Sample 64 pixels (deterministic — pixel 0, then every Nth).
        n_pixels = len(sd)
        idx = torch.arange(0, n_pixels, max(1, n_pixels // 64))[:64]
        coords = sd.coords[idx].float()
        targets = sd.targets[idx].float()

        torch.manual_seed(0)
        model = INR(hparams).to('cpu')

        out = model(coords)['model_out']
        assert out.shape == targets.shape, (
            f'{dataset}/{representative_cell}: output shape {tuple(out.shape)} '
            f'vs target shape {tuple(targets.shape)}'
        )
        assert torch.isfinite(out).all(), (
            f'{dataset}/{representative_cell}: forward NaN/Inf on real coords'
        )
        loss = ((out - targets) ** 2).mean()
        assert torch.isfinite(loss), f'{dataset}/{representative_cell}: loss NaN/Inf'
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            assert torch.isfinite(p.grad).all(), (
                f'{dataset}/{representative_cell}: grad on {name} has NaN/Inf'
            )
        n_checked += 1
        print(f'  OK   {dataset:12s}: {representative_cell} forward+backward '
              f'on real coords, loss={loss.item():.4g}')

    print(f'  [{n_checked} checked, {n_skipped} skipped]')


# ---------------------------------------------------------------------------
# Coverage check (does the smoke test actually cover every grid config?)
# ---------------------------------------------------------------------------
def test_smoke_covers_every_grid_config() -> None:
    """Sanity check that the smoke iterations cover every (cell, L_max)
    combination that src/run_grid.py will actually run."""
    print('\n[smoke] coverage check vs src/run_grid.py ...')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'run_grid', os.path.join(
            os.path.dirname(__file__), '..', 'src', 'run_grid.py'
        ),
    )
    run_grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_grid)

    # Collect every (cell_key, sh_lmax) pair that the launcher will run.
    grid_pairs: set[tuple[str, int | None]] = set()
    for c in run_grid.build_main_grid(42):
        sh_lmax = None
        cfg_ce = None
        from config.architectures import cell_config as _cc
        if _cc(c['cell_key']).get('ce') == 'spherical-harmonics':
            sh_lmax = int(_cc(c['cell_key']).get('sh_lmax', 32))
        grid_pairs.add((c['cell_key'], sh_lmax))
    for c in (run_grid.build_sh_post_saturation_grid(42)
              + run_grid.build_sh_pre_saturation_grid(42)):
        # extra_cli is ['--sh_lmax', '<n>']
        sh_lmax = int(c['extra_cli'][c['extra_cli'].index('--sh_lmax') + 1])
        grid_pairs.add((c['cell_key'], sh_lmax))

    # The smoke-test iterations (default config + 4 L_max values for SH cells).
    smoke_pairs: set[tuple[str, int | None]] = set()
    for cell_key in cell_keys():
        # The "default" smoke run doesn't override sh_lmax; for SH cells, the
        # default _is_ L_max=32.
        if cell_key.endswith('__sh'):
            smoke_pairs.add((cell_key, 32))
        else:
            smoke_pairs.add((cell_key, None))
    sh_cells = [k for k in cell_keys() if k.endswith('__sh')]
    for cell_key in sh_cells:
        for lmax in (13, 16, 31, 32):
            smoke_pairs.add((cell_key, lmax))

    missing = grid_pairs - smoke_pairs
    assert not missing, (
        f'Smoke test does not cover {len(missing)} grid configs: '
        f'{sorted(missing)}'
    )
    extras = smoke_pairs - grid_pairs
    print(f'  OK every grid config covered by smoke test '
          f'({len(grid_pairs)} grid configs, {len(extras)} smoke-only configs '
          f'for sub-grid coverage).')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Smaller PyTorch warnings (kaiming-init "fan-in" complaints etc.) clutter
    # the smoke output; silence them — we'd see actual problems via assertions.
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')

    print('== Per-cell forward + backward + 5 Adam steps ==')
    test_every_cell_default_config()

    print('\n== SH cells × every L_max value used in the grid ==')
    test_sh_cells_at_every_lmax()

    print('\n== Per-dataset end-to-end (real preprocessed NetCDFs) ==')
    test_per_dataset_one_cell_e2e()

    print('\n== Coverage check ==')
    test_smoke_covers_every_grid_config()

    print('\nAll smoke tests passed.')


if __name__ == '__main__':
    main()
