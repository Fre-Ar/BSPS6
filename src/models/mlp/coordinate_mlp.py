from typing import Optional, Tuple, Set
import torch
from torch import Tensor, nn
from ..normalization import norm_factory
from .activations import activation_factory
from .weight_init import weight_init_factory


class CoordinateMLP(nn.Module):
    def __init__(self,
                 in_dim=1,
                 out_dim=1,
                 num_layers=6,
                 layer_width=256,
                 act='relu',
                 act_trainable=False,
                 outermost_linear=True,
                 norm_type = 'none',
                 skip_connections: Optional[Tuple[int]] = None,
                 out_activation: Optional[nn.Module] = None,
                 **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.out_activation = out_activation
        self.outermost_linear = outermost_linear
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.norm_type = norm_type
        self.act_type = act
        self.act_trainable = act_trainable
        self.tmp_skips = []

        self.weight_init = weight_init_factory(self.act_type, **kwargs)
        
        # First layer init for Sine-type activation functions
        self.first_sine_act = None
        self.first_sine_layer_init = None
        if(act == 'sine' or act == "scaled-sine"):
            self.first_sine_layer_init = weight_init_factory(act='first-sine', **kwargs)
            if kwargs['sine_w0'] != kwargs['sine_w']:
                self.first_sine_act = activation_factory(act='first-sine', act_trainable=self.act_trainable, **kwargs)

        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                    if self.norm_type != 'none':
                        layers.append(norm_factory(self.norm_type, self.layer_width))
                    if self.first_sine_act is not None:
                        layers.append(self.first_sine_act)
                    else:
                        layers.append(activation_factory(self.act_type, self.act_trainable, **kwargs))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                    self.tmp_skips.append(len(layers)-1) # store skip layer index
                    
                    if self.norm_type != 'none':
                        layers.append(norm_factory(self.norm_type, self.layer_width))
                    layers.append(activation_factory(self.act_type, self.act_trainable, **kwargs))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
                    if self.norm_type != 'none':
                        layers.append(norm_factory(self.norm_type, self.layer_width))
                    layers.append(activation_factory(self.act_type, self.act_trainable, **kwargs))

            layers.append(nn.Linear(self.layer_width, self.out_dim))

        if not self.outermost_linear: # Output with activations
            if self.out_activation is not None:
                layers.append(self.out_activation)
            else:
                layers.append(activation_factory(self.act_type, self.act_trainable, **kwargs))
        
        self.layers = nn.ModuleList(layers)

        if self.weight_init is not None:
            self.layers.apply(self.weight_init)

        if self.first_sine_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.layers[0].apply(self.first_sine_layer_init)

    def forward(self, in_tensor):
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self.tmp_skips:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
        return x