"""Weight initialization schemes for the MLP activations used here."""
from functools import partial
import math

import numpy as np
import torch
from torch import nn


def init_weights_normal(m, nonlinearity: str = 'relu') -> None:
    """Kaiming-normal init, fan-in. Used for ReLU and Gaussian activations."""
    if isinstance(m, nn.Linear) and hasattr(m, 'weight'):
        nn.init.kaiming_normal_(
            m.weight, a=0.0, nonlinearity=nonlinearity, mode='fan_in',
        )


def init_weights_sine(m, w: float = 30.0) -> None:
    """SIREN hidden-layer init: U(±√(6/in)/ω). Used for ScaledSine."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(
                -np.sqrt(6.0 / num_input) / w,
                 np.sqrt(6.0 / num_input) / w,
            )


def init_weights_sine_first(m) -> None:
    """SIREN first-layer init: U(±1/in). Applied to the first Linear when
    the activation is ScaledSine (or Sine)."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1.0 / num_input, 1.0 / num_input)


def weight_init_factory(act: str = 'relu', **kwargs):
    """Return an init function suitable for `act`, or None for no init."""
    if act in ('relu', 'gaussian'):
        return init_weights_normal
    if act == 'scaled-sine':
        return partial(init_weights_sine, w=kwargs['sine_w'])
    if act == 'first-sine':
        return init_weights_sine_first
    raise ValueError(
        f"Unknown activation '{act}' for weight_init_factory. "
        f"Supported: relu, scaled-sine, gaussian (+ 'first-sine')."
    )
