"""Minimal coordinate MLP used as the regressor in every benchmark cell.

`num_layers` counts the Linear blocks. With the default outermost-linear
convention used here, the structure is:
  Linear(in → W) → act
  Linear(W → W)  → act
  ...
  Linear(W → W)  → act    (num_layers - 1 hidden Linear+act blocks)
  Linear(W → out)         (no activation after the last layer)

For ScaledSine, the *first* layer uses the SIREN-style uniform-by-ω init
(Sitzmann et al. 2020), and a separate first-layer activation with `sine_w0`
is inserted if it differs from the hidden-layer `sine_w`.
"""
from typing import Optional

from torch import Tensor, nn

from .activations import activation_factory
from .weight_init import weight_init_factory


class CoordinateMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1,
        num_layers: int = 6,
        layer_width: int = 256,
        act: str = 'relu',
        outermost_linear: bool = True,
        out_activation: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.out_activation = out_activation
        self.outermost_linear = outermost_linear
        self.act_type = act

        weight_init = weight_init_factory(self.act_type, **kwargs)
        # SIREN-style first-layer init for sine-family activations. If the
        # first-layer ω differs from the hidden ω, insert a separate
        # first-layer activation too.
        first_sine_act: Optional[nn.Module] = None
        first_sine_init = None
        if act in ('sine', 'scaled-sine'):
            first_sine_init = weight_init_factory('first-sine', **kwargs)
            if kwargs.get('sine_w0') != kwargs.get('sine_w'):
                first_sine_act = activation_factory('first-sine', **kwargs)
                
        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            for i in range(num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dim, layer_width))
                    layers.append(
                        first_sine_act if first_sine_act is not None
                        else activation_factory(act, **kwargs)
                    )
                else:
                    layers.append(nn.Linear(layer_width, layer_width))
                    layers.append(activation_factory(act, **kwargs))
            layers.append(nn.Linear(layer_width, out_dim))
            
        if not outermost_linear:
            layers.append(
                out_activation if out_activation is not None
                else activation_factory(act, **kwargs)
            )
        
        self.layers = nn.ModuleList(layers)
            
        if weight_init is not None:
            self.layers.apply(weight_init)
        if first_sine_init is not None:
            self.layers[0].apply(first_sine_init)

    def forward(self, in_tensor: Tensor) -> Tensor:
        x = in_tensor
        for layer in self.layers:
            x = layer(x)
        return x