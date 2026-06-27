"""MLP activations used by the benchmark.

Only the three activations are supported:
  * 'relu'         — torch.nn.ReLU(inplace=True)
  * 'scaled-sine'  — a · sin(ω · x + b) + c with trainable (a, b, c) per layer
                     (Kazerouni et al. 2024, "INCODE"; also matches INR-Bench
                     Table III row "ScaledSine").
  * 'gaussian'     — exp(-x² / (2 σ²)) with fixed σ.

For ScaledSine, a separate 'first-sine' activation key is exposed for use in
the first layer (where the per-layer ω may differ from the hidden ω) — see
`CoordinateMLP.__init__`.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import functools
import operator as op

# Gaussian
class GaussianActivation(nn.Module):
    def __init__(self, a=1.0):
        super().__init__()
        self.register_buffer('a', a * torch.ones(1))
    def forward(self, x):
        return torch.exp(-x ** 2 / (2.0 * self.a ** 2))


# SIREN
class SineActivation(nn.Module):
    """sin(ω · x) with fixed ω. Used as the first-layer activation for
    ScaledSine when sine_w0 ≠ sine_w."""
    def __init__(self, w: float = 1.0):
        super().__init__()
        self.register_buffer('w', w * torch.ones(1))

    def forward(self, x):
        return torch.sin(self.w * x)
    
# INCODE
class ScaledSineActivation(nn.Module):
    def __init__(self, a=0.1993, b=0.0196, c=0.0588, d=0.0269, w=30.0, trainable=True):
        super().__init__()
        self.register_buffer('w', w * torch.ones(1))
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))
        self.register_parameter('c', nn.Parameter(c*torch.ones(1), trainable))
        self.register_parameter('d', nn.Parameter(d*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(self.a) * torch.sin(torch.exp(self.b) * self.w * x + self.c) + self.d


def activation_factory(act: str = 'relu', **kwargs) -> nn.Module:
    """Build an activation module by name."""
    if act == 'relu':
        return nn.ReLU(inplace=True)
    if act == 'gaussian':
        return GaussianActivation(a=kwargs['gaussian_a'])
    if act == 'scaled-sine':
        return ScaledSineActivation(w=kwargs['sine_w'])
    if act == 'first-sine':
        return SineActivation(w=kwargs['sine_w0'])
    raise ValueError(
        f"Unknown activation '{act}'. "
        f"Supported: relu, scaled-sine, gaussian (+ 'first-sine' for "
        f"ScaledSine's first layer)."
    )
