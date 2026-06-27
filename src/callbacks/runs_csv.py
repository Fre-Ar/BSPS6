"""
runs.csv logging callback.

A single source-of-truth CSV with one row per (cell, seed) appended at the end
of each training run. The schema is fixed in `RUNS_CSV_SCHEMA` — adding or
removing columns should be treated as a preregistration amendment, since
downstream analysis scripts depend on a stable column set.

Row contents:
  * Identifiers — dataset / ce / arch / act / pe / seed + JSON-encoded
    encoding_kwargs (so SH L_max=13 H5a runs are distinguishable from the
    main-grid L_max=32 runs).
  * Hyperparameters — every locked value from the preregistration.
  * Metrics — overall + polar + equatorial + per-channel for both
    reconstruction (training pixels, INR-Bench-comparable) and held-out
    (offset grid, generalization). Set by ImgRegCoordSystem.on_train_end
    (main.py) before this callback fires.
  * Bookkeeping — status, parameter_count, wall_clock_seconds,
    peak_gpu_mem_mb, epochs_run, timestamps, log_dir, mitigation_note.
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Optional

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:                               # pragma: no cover — non-POSIX
    _HAVE_FCNTL = False

import torch
from pytorch_lightning.callbacks import Callback


# Stable column schema. Order matters for CSV readability — never reorder.
# Add new columns at the END and treat as a preregistration amendment.
RUNS_CSV_SCHEMA: tuple[str, ...] = (
    # ---- Identifiers ----
    'dataset', 'ce', 'act', 'pe', 'seed', 'encoding_kwargs_json',
    # ---- Hyperparameters ----
    'lr', 'batch_size', 'num_epochs',
    'mlp_num_layers', 'mlp_layer_width',
    'sine_w0', 'gaussian_a',
    'ffn_scale', 'mapping_input', 'omega',
    'sh_lmax',
    # ---- Reconstruction (training-pixel) metrics ----
    'reconstruction_psnr',
    'reconstruction_psnr_polar', 'reconstruction_psnr_equatorial',
    'reconstruction_psnr_r', 'reconstruction_psnr_g', 'reconstruction_psnr_b',
    # ---- Held-out (offset-grid) metrics ----
    'held_out_psnr',
    'held_out_psnr_polar', 'held_out_psnr_equatorial',
    'held_out_psnr_r', 'held_out_psnr_g', 'held_out_psnr_b',
    # ---- Bookkeeping ----
    'status', 'parameter_count', 'wall_clock_seconds', 'peak_gpu_mem_mb',
    'epochs_run', 'timestamp_start', 'timestamp_end',
    'log_dir', 'mitigation_note',
)


def _safe_metric(pl_module, name: str) -> float:
    """Read a metric attribute; return NaN if it doesn't exist (failed run)."""
    v = getattr(pl_module, name, float('nan'))
    if v is None:
        return float('nan')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return ''
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(ts))


def append_row(csv_path: str, row: dict[str, Any]) -> None:
    """Append a single row to csv_path. Writes header on first append.
    Atomic under concurrent appenders via POSIX flock. Missing schema
    columns in `row` are written as empty strings (CSV-NaN)."""
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(csv_path, 'a', newline='') as f:
        if _HAVE_FCNTL:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=RUNS_CSV_SCHEMA,
                                    extrasaction='ignore')
            if os.fstat(f.fileno()).st_size == 0:
                writer.writeheader()
            normalized = {col: row.get(col, '') for col in RUNS_CSV_SCHEMA}
            writer.writerow(normalized)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if _HAVE_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class RunsCSVLogger(Callback):
    """Append one row to runs.csv summarizing the run.

    Reads end-of-training metrics from `pl_module` (set by ImgRegCoordSystem's
    `on_train_end` — see main.py) and hyperparameters from `pl_module.hparams`.

    Hook firing order in Lightning 2.x for the end of fit:

        Callback.on_train_end (e.g. log handlers)
        LightningModule.on_train_end   ← populates pl_module.reconstruction_psnr etc.
        Callback.on_fit_end            ← we write the row HERE
        LightningModule.on_fit_end

    Writing the row in `on_fit_end` (rather than `on_train_end`) means the
    LightningModule's `on_train_end` has already run, so the metrics we read
    via `_safe_metric` are populated. The earlier `on_train_end` version of
    this callback wrote a NaN-PSNR row before the module's evaluation ran —
    that's the bug this hook ordering fixes.
    """

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path: str = csv_path
        self._t_start: Optional[float] = None
        self._t_end: Optional[float] = None
        self._status: str = 'running'
        self._mitigation_note: str = ''
        self._written: bool = False

    # ----- timing & GPU mem ------------------------------------------------
    def on_train_start(self, trainer, pl_module):
        self._t_start = time.time()
        # Peak-memory tracking is CUDA-only; MPS / CPU silently skip.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_fit_end(self, trainer, pl_module):
        # Fires AFTER LightningModule.on_train_end, so end-of-training
        # evaluation metrics (reconstruction_psnr / held_out_psnr / …) have
        # been written to `pl_module` by the time we read them.
        # `_status` may have been flipped to 'eval_failed' by the module's
        # on_train_end wrapper if end-of-training evaluation raised; in that
        # case we keep that status (don't overwrite it with 'completed').
        if self._status == 'running':
            self._status = 'completed'
        self._write_row(trainer, pl_module)

    def on_exception(self, trainer, pl_module, exception):
        # Triggered by uncaught training exceptions.
        # torch 2.x exposes OutOfMemoryError as a distinct subclass of
        # RuntimeError; we match by name to stay compatible across versions.
        ex_name = type(exception).__name__
        if ex_name in ('OutOfMemoryError',) or 'CUDA out of memory' in str(exception):
            self._status = 'oom'
        else:
            self._status = f'error_{ex_name}'
        self._mitigation_note = str(exception)[:500]
        self._write_row(trainer, pl_module)

    # ----- row construction ------------------------------------------------
    def _write_row(self, trainer, pl_module) -> None:
        if self._written:
            return                              # idempotent — don't double-write
        self._written = True

        self._t_end = time.time()
        wall = (self._t_end - self._t_start) if self._t_start else None

        # CUDA exposes max_memory_allocated; MPS / CPU don't, so the column
        # is left blank on those backends.
        peak_mb: Optional[float] = None
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        h = pl_module.hparams
        ce_kwargs_json = json.dumps(
            dict(getattr(h, 'encoding_kwargs', {}) or {}),
            sort_keys=True, default=str,
        )

        param_count = sum(p.numel() for p in pl_module.parameters())

        log_dir = ''
        if trainer is not None and trainer.logger is not None:
            log_dir = getattr(trainer.logger, 'log_dir', '') or ''

        row: dict[str, Any] = {
            # ---- Identifiers ----
            'dataset':  getattr(h, 'dataset', ''),
            'ce':       getattr(h, 'ce_resolved', getattr(h, 'ce', '')),
            'act':      getattr(h, 'act', ''),
            'pe':       getattr(h, 'pe', ''),
            'seed':     int(getattr(h, 'seed', 42)),
            'encoding_kwargs_json': ce_kwargs_json,
            # ---- Hyperparameters ----
            'lr':            float(getattr(h, 'lr', float('nan'))),
            'batch_size':    int(getattr(h, 'batch_size', 0)),
            'num_epochs':    int(getattr(h, 'num_epochs', 0)),
            'mlp_num_layers':   int(getattr(h, 'mlp_num_layers', 0)),
            'mlp_layer_width':  int(getattr(h, 'mlp_layer_width', 0)),
            'sine_w0':       float(getattr(h, 'sine_w0', float('nan'))),
            'gaussian_a':    float(getattr(h, 'gaussian_a', float('nan'))),
            'ffn_scale':     float(getattr(h, 'ffn_scale', float('nan'))),
            'mapping_input': int(getattr(h, 'mapping_input', 0)),
            'omega':         int(getattr(h, 'omega', 0)),
            'sh_lmax':       int(getattr(h, 'sh_lmax', 0)),
            # ---- Reconstruction metrics ----
            'reconstruction_psnr':            _safe_metric(pl_module, 'reconstruction_psnr'),
            'reconstruction_psnr_polar':      _safe_metric(pl_module, 'reconstruction_psnr_polar'),
            'reconstruction_psnr_equatorial': _safe_metric(pl_module, 'reconstruction_psnr_equatorial'),
            'reconstruction_psnr_r':          _safe_metric(pl_module, 'reconstruction_psnr_r'),
            'reconstruction_psnr_g':          _safe_metric(pl_module, 'reconstruction_psnr_g'),
            'reconstruction_psnr_b':          _safe_metric(pl_module, 'reconstruction_psnr_b'),
            # ---- Held-out metrics ----
            'held_out_psnr':            _safe_metric(pl_module, 'held_out_psnr'),
            'held_out_psnr_polar':      _safe_metric(pl_module, 'held_out_psnr_polar'),
            'held_out_psnr_equatorial': _safe_metric(pl_module, 'held_out_psnr_equatorial'),
            'held_out_psnr_r':          _safe_metric(pl_module, 'held_out_psnr_r'),
            'held_out_psnr_g':          _safe_metric(pl_module, 'held_out_psnr_g'),
            'held_out_psnr_b':          _safe_metric(pl_module, 'held_out_psnr_b'),
            # ---- Bookkeeping ----
            'status':              self._status,
            'parameter_count':     int(param_count),
            'wall_clock_seconds':  f'{wall:.3f}' if wall is not None else '',
            'peak_gpu_mem_mb':     f'{peak_mb:.2f}' if peak_mb is not None else '',
            'epochs_run':          int(trainer.current_epoch) if trainer is not None else 0,
            'timestamp_start':     _fmt_ts(self._t_start),
            'timestamp_end':       _fmt_ts(self._t_end),
            'log_dir':             log_dir,
            'mitigation_note':     self._mitigation_note,
        }
        append_row(self.csv_path, row)
        print(f"[runs_csv] appended row to {self.csv_path} "
              f"(status={self._status})")
