import torch.nn as nn
from .pos_encoding import PosEncoding, ENCODING_DICT
from .mlp.coordinate_mlp import CoordinateMLP
from .kan.coordinate_kan import CoordinateKAN

class INR(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        # Positional Encoding
        if hparams.pe == "None":
            pos_encode_configs = {'type': None}
        elif hparams.pe == "NeRF":
            pos_encode_configs = {'type':'frequency',
                                  'use_nyquist': True,
                                   #   For Ct recon
                                  'mapping_input': hparams.batch_size if hparams.batch_size!=1 else hparams.proj,
                                  'num_frequencies': hparams.num_frequencies}
        elif hparams.pe == "RFF":
            pos_encode_configs = {'type':'gaussian',
                                  'scale_B': hparams.ffn_scale,
                                  'mapping_input': hparams.mapping_input}

        in_features = hparams.in_features
        self.pos_encode = pos_encode_configs['type']
        if self.pos_encode in ENCODING_DICT.keys():
            self.positional_encoding = PosEncoding(self.pos_encode).run(in_features=hparams.in_features, pos_encode_configs=pos_encode_configs)
            in_features = self.positional_encoding.out_dim
        elif self.pos_encode == None:
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, gaussian]"

        if self.pos_encode:
            print("PE Dim: ", self.positional_encoding.out_dim)

        # Model
        if hparams.arch == "mlp":
            kwargs = {
                      'sine_w0': hparams.sine_w0,
                      'sine_w': hparams.sine_w,
                      'gaussian_a': hparams.gaussian_a,
                      'gaussian_b': hparams.gaussian_b,
                      'quadratic_a': hparams.quadratic_a,
                      'gabor_a': hparams.gabor_a,
                      'gabor_b': hparams.gabor_b,
                    }
            self.net = CoordinateMLP(
                        in_dim=in_features,
                        out_dim=hparams.out_features,
                        num_layers=hparams.mlp_num_layers,
                        layer_width=hparams.mlp_layer_width,
                        act=hparams.act,
                        act_trainable=hparams.act_trainable,
                        outermost_linear=True,
                        norm_type=hparams.norm,
                        skip_connections=None,
                        out_activation=None,
                        **kwargs
                    )
        elif hparams.arch == 'kan':
            self.net = CoordinateKAN(
                        in_dim=in_features,
                        out_dim=hparams.out_features,
                        num_layers=hparams.kan_num_layers,
                        layer_width=hparams.kan_layer_width,
                        basis=hparams.act,
                        norm_type=hparams.norm,
                        input_grid_size=hparams.input_grid_size,
                        hidden_grid_size=hparams.hidden_grid_size,
                        output_grid_size=hparams.output_grid_size,
                        skip_connections=None,
                        outermost_linear=False,
                        out_activation=None
                    )
        elif hparams.arch == 'kamp':
            kwargs = {
                      'sine_w0': hparams.sine_w0,
                      'sine_w': hparams.sine_w,
                      'gaussian_a': hparams.gaussian_a,
                      'gaussian_b': hparams.gaussian_b,
                      'quadratic_a': hparams.quadratic_a,
                      'gabor_a': hparams.gabor_a,
                      'gabor_b': hparams.gabor_b,
                    }
            kan_net = CoordinateKAN(
                        in_dim=in_features,
                        out_dim=hparams.kan_layer_width,
                        num_layers=hparams.kan_num_layers,
                        layer_width=hparams.kan_layer_width,
                        basis=hparams.kan_act,
                        norm_type=hparams.norm,
                        input_grid_size=hparams.input_grid_size,
                        hidden_grid_size=hparams.hidden_grid_size,
                        output_grid_size=hparams.output_grid_size,
                        skip_connections=None,
                        outermost_linear=False,
                        out_activation=None
                    )
            mlp_net = CoordinateMLP(
                        in_dim=hparams.kan_layer_width,
                        out_dim=hparams.out_features,
                        num_layers=hparams.mlp_num_layers,
                        layer_width=hparams.mlp_layer_width,
                        act=hparams.mlp_act,
                        act_trainable=hparams.act_trainable,
                        outermost_linear=True,
                        norm_type=hparams.norm,
                        skip_connections=None,
                        out_activation=None,
                        **kwargs
                    )
            self.net = nn.Sequential(kan_net, mlp_net)

    def forward(self, x):
        """
        x: [B, N]
        """
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        
        output = None
        if x.dim() == 3:
            coords = coords.squeeze(0)
            if self.pos_encode:
                coords = self.positional_encoding(coords)
            output = self.net(coords).unsqueeze(0)
        else:
            if self.pos_encode:
                coords = self.positional_encoding(coords)
            output = self.net(coords)
        
        return {'model_in': coords_org, 'model_out':  output}
    