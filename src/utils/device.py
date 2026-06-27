"""Cross-platform device detection.

Used to:
  * select the PyTorch Lightning accelerator string,
  * gate CUDA-only features (peak-memory tracking, CuDNN benchmark flags,
    `torch.cuda.empty_cache`, etc.) so they no-op cleanly on macOS / MPS / CPU,
  * print a human-readable banner at startup.

We prefer CUDA over MPS over CPU, in that order. CUDA is the primary target
(Windows / Linux with NVIDIA hardware); MPS is the macOS fallback for local
development (slower than CUDA but functional for the MLP/cos-sin ops used
in this benchmark); CPU is the universal fallback.
"""
from __future__ import annotations

import torch


def detect_device() -> str:
    """Return one of 'cuda', 'mps', 'cpu': the device this process will use."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def is_cuda() -> bool:
    return detect_device() == 'cuda'


def is_mps() -> bool:
    return detect_device() == 'mps'


def lightning_accelerator() -> str:
    """The argument to pass to `pytorch_lightning.Trainer(accelerator=...)`."""
    d = detect_device()
    if d == 'cuda':
        return 'gpu'          # Lightning treats 'gpu' as the generic GPU accelerator.
    if d == 'mps':
        return 'mps'
    return 'cpu'


def device_display_name() -> str:
    """Pretty banner string for logs / startup output."""
    d = detect_device()
    if d == 'cuda':
        try:
            return f'cuda ({torch.cuda.get_device_name(0)})'
        except Exception:
            return 'cuda'
    if d == 'mps':
        return 'mps (Apple Silicon / Metal)'
    return 'cpu'
