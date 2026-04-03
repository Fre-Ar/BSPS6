import numpy as np
import torch
from torch import nn

class GaussianEncoding(nn.Module):
    def __init__(self, pos_encode_configs, in_features=1):
        super().__init__()
        
        self.scale = pos_encode_configs['scale_B']
        mapping_input = pos_encode_configs['mapping_input']
        
        self.B_gauss = torch.randn((mapping_input, in_features), device="cuda") * self.scale

        self.out_dim = mapping_input * 2
        # if in_features == 1:
        #     self.out_dim = self.out_dim * 2
        

    def forward(self, coords):
        x_proj = (2. * np.pi * coords) @ self.B_gauss.t()

        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)