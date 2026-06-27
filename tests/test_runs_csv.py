"""
Unit tests for src/callbacks/runs_csv.py — the per-run CSV logging callback.

We don't spin up a real Trainer here; we exercise the callback directly with
hand-built stub objects, so the tests are fast and don't need GPU/torch-cuda.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from callbacks.runs_csv import (                                         # noqa: E402
    RunsCSVLogger,
    RUNS_CSV_SCHEMA,
    append_row,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _HParams:
    """Minimal stand-in for `pl_module.hparams` — exposes attributes via
    plain Python attribute access, matching PL's `AttributeDict` API."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubModule:
    """Minimal stand-in for the LightningModule. Carries:
       - hparams: the namespace the logger reads from
       - reconstruction_*, held_out_* metrics: set as needed
       - parameters(): yields a tensor-like object with .numel()
    """
    def __init__(self, hparams: _HParams, num_params: int = 0):
        self.hparams = hparams
        self._num_params = num_params

    def parameters(self):
        # The logger calls sum(p.numel() for p in pl_module.parameters()).
        class _P:
            def __init__(self, n): self._n = n
            def numel(self): return self._n
        # Return one fake parameter with the requested total count.
        yield _P(self._num_params)


class _StubLogger:
    def __init__(self, log_dir: str = ''):
        self.log_dir = log_dir


class _StubTrainer:
    def __init__(self, current_epoch: int = 0, log_dir: str = ''):
        self.current_epoch = current_epoch
        self.logger = _StubLogger(log_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _baseline_hparams(**overrides) -> _HParams:
    base = dict(
        # identifiers
        dataset='etopo1', ce='angular', ce_resolved='angular',
        act='scaled-sine', pe='None', seed=42,
        encoding_kwargs={},
        # hyperparameters
        lr=4e-4, batch_size=8192, num_epochs=1000,
        mlp_num_layers=6, mlp_layer_width=256,
        sine_w0=30.0, gaussian_a=0.1,
        ffn_scale=10.0, mapping_input=32, omega=32,
        sh_lmax=32,
    )
    base.update(overrides)
    return _HParams(**base)


def _read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_schema_columns_unique_and_nonempty() -> None:
    """Sanity: schema has no duplicates and no empty strings."""
    print('\n[runs_csv] schema integrity ...')
    assert len(RUNS_CSV_SCHEMA) == len(set(RUNS_CSV_SCHEMA)), 'duplicate columns'
    assert all(c.strip() for c in RUNS_CSV_SCHEMA), 'empty column name'
    print(f'  OK {len(RUNS_CSV_SCHEMA)} unique columns.')


def test_append_row_writes_header_and_row() -> None:
    """First append writes header; the row's values are recoverable."""
    print('\n[runs_csv] append_row writes header on empty file ...')
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        append_row(csv_path, {'dataset': 'etopo1', 'ce': 'angular', 'seed': 42})
        rows = _read_rows(csv_path)
        assert len(rows) == 1
        assert rows[0]['dataset'] == 'etopo1'
        assert rows[0]['ce'] == 'angular'
        assert rows[0]['seed'] == '42'
        # Schema columns missing from `row` are written as empty strings.
        assert rows[0]['act'] == ''
        print('  OK header + one row recovered.')


def test_append_row_appends_without_re_header() -> None:
    """Second append does NOT rewrite the header."""
    print('\n[runs_csv] append_row appends without rewriting header ...')
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        append_row(csv_path, {'dataset': 'etopo1', 'seed': 42})
        append_row(csv_path, {'dataset': 'cmb', 'seed': 42})
        rows = _read_rows(csv_path)
        assert len(rows) == 2, f'expected 2 rows, got {len(rows)}'
        assert rows[0]['dataset'] == 'etopo1'
        assert rows[1]['dataset'] == 'cmb'
        # Header should appear exactly once: count lines manually.
        with open(csv_path) as f:
            header = f.readline().strip()
            assert header.startswith('dataset,ce,'), f'unexpected header {header!r}'
            non_header = [line for line in f if line.strip()]
            assert len(non_header) == 2, 'extra header(s) written'
        print('  OK 2 rows, single header.')


def test_logger_writes_completed_row_with_metrics() -> None:
    """The callback writes a `completed` row populated with all metrics."""
    print('\n[runs_csv] RunsCSVLogger writes completed row + metrics ...')
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        cb = RunsCSVLogger(csv_path)

        hp = _baseline_hparams()
        mod = _StubModule(hp, num_params=123456)
        # Populate end-of-training metrics as ImgRegCoordSystem.on_fit_end would.
        mod.reconstruction_psnr = 35.5
        mod.reconstruction_psnr_polar = 30.1
        mod.reconstruction_psnr_equatorial = 40.7
        mod.held_out_psnr = 33.2
        mod.held_out_psnr_polar = 28.0
        mod.held_out_psnr_equatorial = 38.5

        tr = _StubTrainer(current_epoch=999, log_dir='/tmp/log_dir')

        cb.on_train_start(tr, mod)
        cb.on_fit_end(tr, mod)

        rows = _read_rows(csv_path)
        assert len(rows) == 1, f'expected 1 row, got {len(rows)}'
        r = rows[0]
        assert r['status'] == 'completed'
        assert r['dataset'] == 'etopo1'
        assert r['act'] == 'scaled-sine'
        assert r['seed'] == '42'
        assert float(r['reconstruction_psnr']) == 35.5
        assert float(r['held_out_psnr']) == 33.2
        assert int(r['parameter_count']) == 123456
        assert int(r['epochs_run']) == 999
        # Scalar dataset → channel columns are empty (NaN-like).
        assert r['reconstruction_psnr_r'] in ('', 'nan'), \
            f"expected blank/nan for scalar dataset, got {r['reconstruction_psnr_r']!r}"
        print(f"  OK status={r['status']}, recon={r['reconstruction_psnr']} dB, "
              f"held={r['held_out_psnr']} dB.")


def test_logger_handles_rgb_channel_metrics() -> None:
    """For an RGB dataset, channel_r/g/b columns are populated."""
    print('\n[runs_csv] RGB channel metrics propagate ...')
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        cb = RunsCSVLogger(csv_path)

        hp = _baseline_hparams(dataset='hdri_urban')
        mod = _StubModule(hp, num_params=42)
        mod.reconstruction_psnr = 30.0
        mod.reconstruction_psnr_polar = 28.0
        mod.reconstruction_psnr_equatorial = 32.0
        mod.reconstruction_psnr_r = 31.5
        mod.reconstruction_psnr_g = 29.7
        mod.reconstruction_psnr_b = 28.9
        mod.held_out_psnr = 27.0
        mod.held_out_psnr_polar = 25.0
        mod.held_out_psnr_equatorial = 29.0
        mod.held_out_psnr_r = 28.0
        mod.held_out_psnr_g = 26.5
        mod.held_out_psnr_b = 25.5

        tr = _StubTrainer(current_epoch=1000, log_dir='/tmp/x')
        cb.on_train_start(tr, mod)
        cb.on_fit_end(tr, mod)

        r = _read_rows(csv_path)[0]
        assert float(r['reconstruction_psnr_r']) == 31.5
        assert float(r['reconstruction_psnr_g']) == 29.7
        assert float(r['reconstruction_psnr_b']) == 28.9
        assert float(r['held_out_psnr_b']) == 25.5
        print(f"  OK RGB channels: R={r['reconstruction_psnr_r']}, "
              f"G={r['reconstruction_psnr_g']}, B={r['reconstruction_psnr_b']}.")


def test_logger_missing_metrics_become_nan() -> None:
    """If the LightningModule has no metric attrs (e.g. held-out eval crashed),
    those columns are NaN rather than crashing the callback."""
    print('\n[runs_csv] missing metric attrs → NaN, no crash ...')
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        cb = RunsCSVLogger(csv_path)
        hp = _baseline_hparams()
        mod = _StubModule(hp, num_params=1)
        # No reconstruction_*/held_out_* attributes set on `mod`.
        tr = _StubTrainer(current_epoch=0, log_dir='')
        cb.on_train_start(tr, mod)
        cb.on_fit_end(tr, mod)

        r = _read_rows(csv_path)[0]
        # NaN renders as 'nan' (Python str(float('nan'))) — the analysis
        # script reads with pandas.read_csv which interprets 'nan' as NaN.
        for k in ('reconstruction_psnr', 'held_out_psnr',
                  'reconstruction_psnr_polar', 'held_out_psnr_equatorial'):
            assert r[k] == 'nan', f'expected NaN for {k}, got {r[k]!r}'
        assert r['status'] == 'completed'
        print('  OK missing metrics → "nan" cells, status=completed.')


def test_logger_on_exception_records_failure() -> None:
    """An exception during training writes a row with status=oom or
    status=error_<ExceptionType>."""
    print('\n[runs_csv] on_exception records failure status ...')

    # ---- OOM case ----
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        cb = RunsCSVLogger(csv_path)
        hp = _baseline_hparams()
        mod = _StubModule(hp, num_params=1)
        tr = _StubTrainer(current_epoch=0)
        cb.on_train_start(tr, mod)
        # Simulate an OOM error. We use a RuntimeError with the substring
        # 'CUDA out of memory' since `torch.cuda.OutOfMemoryError` may not
        # exist on CPU-only builds.
        err = RuntimeError('CUDA out of memory. Tried to allocate 2.28 GiB')
        cb.on_exception(tr, mod, err)
        r = _read_rows(csv_path)[0]
        assert r['status'] == 'oom', f'expected status=oom, got {r["status"]}'
        assert 'CUDA out of memory' in r['mitigation_note']
        print(f"  OK OOM:    status={r['status']}, note='{r['mitigation_note'][:40]}...'")

    # ---- Generic error case ----
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        cb = RunsCSVLogger(csv_path)
        hp = _baseline_hparams()
        mod = _StubModule(hp, num_params=1)
        tr = _StubTrainer(current_epoch=0)
        cb.on_train_start(tr, mod)
        cb.on_exception(tr, mod, ValueError('something went wrong'))
        r = _read_rows(csv_path)[0]
        assert r['status'] == 'error_ValueError', \
            f"expected error_ValueError, got {r['status']}"
        print(f"  OK ValueError: status={r['status']}.")


def test_logger_idempotent_write() -> None:
    """Calling on_fit_end twice writes only one row (idempotency)."""
    print('\n[runs_csv] double on_fit_end writes only one row ...')
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        cb = RunsCSVLogger(csv_path)
        hp = _baseline_hparams()
        mod = _StubModule(hp, num_params=1)
        mod.reconstruction_psnr = 10.0
        mod.held_out_psnr = 8.0
        tr = _StubTrainer(current_epoch=1000)

        cb.on_train_start(tr, mod)
        cb.on_fit_end(tr, mod)
        cb.on_fit_end(tr, mod)  # second call: must be a no-op

        rows = _read_rows(csv_path)
        assert len(rows) == 1, f'expected 1 row, got {len(rows)}'
        print('  OK exactly one row written despite two on_fit_end calls.')


def test_logger_encoding_kwargs_json() -> None:
    """encoding_kwargs is serialized as canonical JSON for analysis grouping."""
    print('\n[runs_csv] encoding_kwargs_json canonical encoding ...')
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        cb = RunsCSVLogger(csv_path)
        hp = _baseline_hparams(
            ce='spherical-harmonics',
            ce_resolved='spherical-harmonics',
            encoding_kwargs={'L_max': 32},
        )
        mod = _StubModule(hp, num_params=1)
        tr = _StubTrainer(current_epoch=0)
        cb.on_train_start(tr, mod)
        cb.on_fit_end(tr, mod)
        r = _read_rows(csv_path)[0]
        parsed = json.loads(r['encoding_kwargs_json'])
        assert parsed == {'L_max': 32}
        print(f"  OK encoding_kwargs_json = {r['encoding_kwargs_json']}.")


def main() -> None:
    print('== runs.csv schema ==')
    test_schema_columns_unique_and_nonempty()

    print('\n== append_row primitive ==')
    test_append_row_writes_header_and_row()
    test_append_row_appends_without_re_header()

    print('\n== RunsCSVLogger callback ==')
    test_logger_writes_completed_row_with_metrics()
    test_logger_handles_rgb_channel_metrics()
    test_logger_missing_metrics_become_nan()
    test_logger_on_exception_records_failure()
    test_logger_idempotent_write()
    test_logger_encoding_kwargs_json()

    print('\nAll runs_csv tests passed.')


if __name__ == '__main__':
    main()
