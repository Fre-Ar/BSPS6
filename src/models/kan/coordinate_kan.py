
from torch import Tensor, nn
import torch
from models.normalization import norm_factory
from typing import Optional, Tuple, Set


def kan_layer_factory(kan_basis_type='bspline'):
    if kan_basis_type == 'bspline':
        from .bspline_kan import SplineKANLayer
        return SplineKANLayer
    elif kan_basis_type == 'fourier':
        from .fourier_kan import FourierKANLayer
        return FourierKANLayer
    else: print("Not Implemented!!!")

class CoordinateKAN(nn.Module):
    def __init__(self,
                 in_dim=1,
                 out_dim=1,
                 num_layers=6,
                 layer_width=64,
                 basis='relu',
                 outermost_linear=False,
                 norm_type = 'none',
                 input_grid_size=256,
                 hidden_grid_size=5,
                 output_grid_size=3,
                 skip_connections: Optional[Tuple[int]] = None,
                 out_activation: Optional[nn.Module] = None):
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
        self.basis_type = basis
        self.input_grid_size = input_grid_size
        self.hidden_grid_size = hidden_grid_size
        self.output_grid_size = output_grid_size
        self.tmp_skips = []

        layers = []
        if self.num_layers == 1:
            layers.append(
                kan_layer_factory(self.basis_type)(self.in_dim, self.out_dim, self.hidden_grid_size)
            )
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(
                        kan_layer_factory(self.basis_type)(self.in_dim, self.layer_width, self.input_grid_size)
                    )
                    if self.norm_type != 'none':
                        layers.append(norm_factory(self.norm_type, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(
                        kan_layer_factory(self.basis_type)(self.in_dim + self.layer_width, self.layer_width, self.hidden_grid_size)
                    )
                    self.tmp_skips.append(len(layers)-1) # store skip layer index

                    if self.norm_type != 'none':
                        layers.append(norm_factory(self.norm_type, self.layer_width))
                else:
                    layers.append(
                         kan_layer_factory(self.basis_type)(self.layer_width, self.layer_width, self.hidden_grid_size)
                    )
                    if self.norm_type != 'none':
                        layers.append(norm_factory(self.norm_type, self.layer_width))

            if self.outermost_linear:
                layers.append(nn.Linear(self.layer_width, self.out_dim))
            else:
                if self.out_activation is not None:
                    layers.append(nn.Linear(self.layer_width, self.out_dim))
                    layers.append(self.out_activation)
                else:
                    layers.append(
                        kan_layer_factory(self.basis_type)(self.layer_width, self.out_dim, self.output_grid_size)
                    )
       
        self.layers = nn.ModuleList(layers)

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
    
# class CoordinateKAN(nn.Module):
#     def __init__(self,
#                  in_features=1,
#                  hidden_features=64,
#                  hidden_layers=3,
#                  out_features=1,
#                  input_grid_size=256,
#                  hidden_grid_size=5,
#                  output_grid_size=3,
#                  outermost_linear=False,
#                  basis="fourier",
#                  norm_type="none"):
#         super().__init__()

#         KANLayer = kan_layer_factory(kan_basis_type=basis)

#         self.net = []
       
#         self.net.append(KANLayer(in_features, hidden_features, grid_size=input_grid_size))
#         if norm_type != 'none':
#             self.net.append(norm_factory(norm_type, hidden_features))
        

#         for _ in range(hidden_layers):
#             self.net.append(KANLayer(hidden_features, hidden_features, grid_size=hidden_grid_size))
#             if norm_type != 'none':
#                 self.net.append(norm_factory(norm_type, hidden_features))

#         if outermost_linear:
#             self.net.append(nn.Linear(hidden_features, out_features))
#         else:
#             self.net.append(KANLayer(hidden_features, out_features, grid_size=output_grid_size))
       

#         self.net = nn.Sequential(*self.net)

#     def forward(self, x):
#         return self.net(x)