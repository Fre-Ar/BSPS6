"""Coordinate-MLP wrapper with optional positional encoding.

The model is `[PE] → MLP`, where the PE is one of:
  * None  — raw coord-encoded input fed directly to the MLP.
  * RFF   — INR-Bench's Gaussian random Fourier features.
  * FKAN  — Li et al. 2025's FKAN (a single Fourier feature map with
            trainable per-(d, ω) coefficients).
"""
from torch import nn
from .mlp.coordinate_mlp import CoordinateMLP
from .pos_encoding import PosEncoding, ENCODING_DICT


def _build_pe_configs(hparams) -> dict | None:
    """Translate hparams.pe into the `pos_encode_configs` dict consumed by
    PosEncoding. Returns None if no PE is requested."""
    if hparams.pe == 'None':
        return None
    if hparams.pe == 'RFF':
        return {
            'type': 'gaussian',
            'scale_B': hparams.ffn_scale,
            'mapping_input': hparams.mapping_input,
        }
    if hparams.pe == 'FKAN':
        return {'type': 'fkan', 'omega': hparams.omega}
    raise ValueError(
        f"Unknown --pe '{hparams.pe}'. Expected one of: None, RFF, FKAN."
    )


class INR(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        in_features = hparams.in_features
        pe_configs = _build_pe_configs(hparams)
        
        if pe_configs is None:
            self.pos_encode = False
            self.positional_encoding = None
        else:
            pe_type = pe_configs['type']
            if pe_type not in ENCODING_DICT:
                raise ValueError(
                    f"PE type '{pe_type}' not in ENCODING_DICT "
                    f"({list(ENCODING_DICT)})."
                )
            self.positional_encoding = PosEncoding(pe_type).run(
                in_features=in_features, pos_encode_configs=pe_configs,
            )
            in_features = self.positional_encoding.out_dim
            self.pos_encode = True
            print(f'PE Dim: {self.positional_encoding.out_dim}')
        
        # Activation-specific kwargs forwarded to the MLP / activation factory.
        kwargs = {
            'sine_w0':    hparams.sine_w0,
            'sine_w':     hparams.sine_w,
            'gaussian_a': hparams.gaussian_a,
        }
        self.net = CoordinateMLP(
            in_dim=in_features,
            out_dim=hparams.out_features,
            num_layers=hparams.mlp_num_layers,
            layer_width=hparams.mlp_layer_width,
            act=hparams.act,
            outermost_linear=True,
            **kwargs,
        )

    def forward(self, x):
        """x: (B, D_coord) → returns {'model_in': coords_org, 'model_out': y}."""
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        
        if x.dim() == 3:
            coords = coords.squeeze(0)
            if self.pos_encode:
                coords = self.positional_encoding(coords)
            output = self.net(coords).unsqueeze(0)
        else:
            if self.pos_encode:
                coords = self.positional_encoding(coords)
            output = self.net(coords)
        
        return {'model_in': coords_org, 'model_out': output}
    