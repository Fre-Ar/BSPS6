import argparse
from config.constants import (
    CE_CHOICES, DATASET_CHOICES, DATASET_CONFIG, ELEVATION_DATA_PATH
)

def get_opts():
    parser = argparse.ArgumentParser()
    # Model Function
    parser.add_argument('--arch', type=str, default='mlp',
                        choices=['mlp', 'kan', 'kamp'],
                        help='Model Architecture')
    parser.add_argument('--act', type=str, default='relu',
                        choices=[
                            # MLP Activation Functions
                            'relu', 'scaled-sine', 'gaussian',
                            # KAN Basis Functions
                            'bspline', 'fourier',       
                        ],
                        help='network structure')
    parser.add_argument('--kan_act', type=str, default='fourier',
                        choices=[
                            'bspline', 'fourier',     
                        ],
                        help='KAN activation function')
    parser.add_argument('--mlp_act', type=str, default='relu',
                         choices=[
                            'relu', 'scaled-sine', 'gaussian',
                        ],
                        help='MLP activation function')
    # Model Encodings
    parser.add_argument('--ce', type=str, default='angular',
                        choices=CE_CHOICES,
                        help='coordinate encoding')
    parser.add_argument('--pe', type=str, default='None',
                        choices=['NeRF', 'RFF', 'None', 'FKAN'],
                        help='positional encoding')
    
    parser.add_argument('--norm', type=str, default='none',
                        choices=['none', 'batch', 'layer', 'cross', 'global', 'instance'],
                        help='Norm Tech')
    
    # Model Shape
    parser.add_argument('--mlp_num_layers', type=int, default=6,
                        help='number of KAN layers')
    parser.add_argument('--mlp_layer_width', type=int, default=256,
                        help='number of KAN layer width')
    parser.add_argument('--kan_num_layers', type=int, default=6,
                        help='number of KAN layers')
    parser.add_argument('--kan_layer_width', type=int, default=64,
                        help='number of KAN layer width')
   
    # KAN Grid size
    parser.add_argument('--input_grid_size', type=int, default=8,
                        help='number of KAN grid zise of first layer')
    parser.add_argument('--hidden_grid_size', type=int, default=8,
                        help='number of KAN grid zise of hidden layers')
    parser.add_argument('--output_grid_size', type=int, default=8,
                        help='number of KAN grid zise of output layer')
    parser.add_argument('--degree', type=int, default=4,
                        help='number of KAN hidden layer dim')
    
    # optimizer and scheduler
    parser.add_argument('--opt', type=str, default='adam',
                        choices=['schedulefree', 'adam', 'lbfgs'],
                        help='network structure')
    
    # INR Task Shape
    parser.add_argument('--in_features', type=int, default=1,
                        help='input dim of Network')
    parser.add_argument('--out_features', type=int, default=1,
                        help='out dim of Network')
   

    parser.add_argument('--img_wh', nargs="+", type=int, default=[256, 256],
                        help='resolution (img_w, img_h) of the image')
    
    ###### Positional Encoding Parameters
    # FFN
    parser.add_argument('--ffn_scale', type=float, default=10.)
    parser.add_argument('--mapping_input', type=int, default=32) # 256
    # NeRF
    parser.add_argument('--num_frequencies', type=int, default=4)
    # parser.add_argument('--disable_use_nyquist', action='store_false', help='')

    
    parser.add_argument('--act_trainable', default=False, action='store_true',
                        help='whether to train activation hyperparameter')
    

    parser.add_argument('--sine_w0', type=float, default=30.,
                        help='omega in Sine-type activations (SIREN and INCODE)')
    parser.add_argument('--sine_w', type=float, default=30.,
                        help='omega in Sine-type activations (SIREN and INCODE)')
    
    # - When \(a = 1\), it corresponds to the standard Gaussian distribution.
    # - If \( b > 1 \), it makes the values of the Gaussian distribution smoother,
    #  while if \( 0 < b < 1 \), it further enhances the sharpness of the Gaussian.
    parser.add_argument('--gaussian_a', type=float, default=0.1)
    parser.add_argument('--gaussian_b', type=float, default=1.0)
    

    # - When \(a > 1\): The function decays more quickly, resulting in a narrower curve.
    # - When \(0 < a < 1\): The function decays more slowly, resulting in a wider curve.
    # - When \(a = 1\): The function behaves normally without scaling.
    parser.add_argument('--quadratic_a', type=float, default=3.)

    parser.add_argument('--gabor_a', type=float, default=1.)
    parser.add_argument('--gabor_b', type=float, default=1.)
    
    parser.add_argument('--sc', type=float, default=0.1, # default 10.
                        help='fourier feature scale factor (std of the gaussian)')
    

    parser.add_argument('--outermost_linear', type=bool, default=True,
                        help='outermost_linear in siren')

    # Exp
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')
    
    # Logs and Vis
    parser.add_argument('--save_vis', default=False, action='store_true',
                        help='save vis or not')
    parser.add_argument('--vis_every', type=int, default=200,
                        help='check val every n epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20,
                        help='check val every n epoch')
    parser.add_argument('--save_dir', type=str, default='logs',
                        help='experiment log dir')
    parser.add_argument('--exp_name', type=str, default='None',
                        help='experiment name')
    
    # ETOPO1 specific
    parser.set_defaults(in_features=2)
    parser.set_defaults(out_features=1)
    # parser.set_defaults(kan_hidden_layers=5)
    # parser.set_defaults(kan_hidden_features=64)
    # parser.set_defaults(input_grid_size=8)
    # parser.set_defaults(hidden_grid_size=8)
    # parser.set_defaults(output_grid_size=8)

    parser.set_defaults(batch_size=8192)

    parser.set_defaults(gaussian_a=0.1)

    parser.set_defaults(ffn_scale=10)
    parser.set_defaults(mapping_input=32)
    
    parser.set_defaults(img_wh=[1024, 512])
    parser.set_defaults(save_dir='logs/image_regression')

    parser.add_argument('--data_path', type=str, default=ELEVATION_DATA_PATH,
                        help='Path to the pre-processed .nc file. '
                             'Overrides --dataset.')
    parser.add_argument('--dataset', type=str, default='etopo1',
                        choices=DATASET_CHOICES,
                        help='Which pre-processed benchmark dataset to train on. '
                             'Sets data_path and out_features automatically '
                             '(unless --data_path / --out_features are given).')

    parser.set_defaults(arch='mlp')
    parser.set_defaults(act='scaled-sine')
    parser.set_defaults(mlp_act='scaled-sine')
    parser.set_defaults(pe='None')
    parser.set_defaults(save_vis=True)

    import sys as _sys
    hparams = parser.parse_args()
    explicit = set(a.split('=')[0] for a in _sys.argv[1:] if a.startswith('--'))
    cfg = DATASET_CONFIG[hparams.dataset]
    if '--data_path' not in explicit:
        hparams.data_path = cfg['path']
    if '--out_features' not in explicit:
        hparams.out_features = cfg['out_features']
    return hparams


import argparse
from config.constants import (
    ELEVATION_DATA_PATH, CE_CHOICES, DATASET_CHOICES, DATASET_CONFIG,
)


def get_opts():
    parser = argparse.ArgumentParser()
    # Model Function
    parser.add_argument('--arch', type=str, default='mlp',
                        choices=['mlp', 'kan', 'kamp'],
                        help='Model Architecture')
    parser.add_argument('--act', type=str, default='relu',
                        choices=[
                            # MLP Activation Functions
                            'relu', 'scaled-sine', 'gaussian',
                            # KAN Basis Functions
                            'bspline', 'fourier',       
                        ],
                        help='network structure')
    parser.add_argument('--kan_act', type=str, default='fourier',
                        choices=[
                            'bspline', 'fourier',     
                        ],
                        help='KAN activation function')
    parser.add_argument('--mlp_act', type=str, default='relu',
                         choices=[
                            'relu', 'scaled-sine', 'gaussian',
                        ],
                        help='MLP activation function')
    # Model Encodings
    parser.add_argument('--ce', type=str, default='angular',
                        choices=CE_CHOICES,
                        help='coordinate encoding')
    parser.add_argument('--pe', type=str, default='None',
                        choices=['NeRF', 'RFF', 'None', 'FKAN'],
                        help='positional encoding')
    
    parser.add_argument('--norm', type=str, default='none',
                        choices=['none', 'batch', 'layer', 'cross', 'global', 'instance'],
                        help='Norm Tech')
    
    # Model Shape
    parser.add_argument('--mlp_num_layers', type=int, default=6,
                        help='number of KAN layers')
    parser.add_argument('--mlp_layer_width', type=int, default=256,
                        help='number of KAN layer width')
    parser.add_argument('--kan_num_layers', type=int, default=6,
                        help='number of KAN layers')
    parser.add_argument('--kan_layer_width', type=int, default=64,
                        help='number of KAN layer width')
   
    # KAN Grid size
    parser.add_argument('--input_grid_size', type=int, default=8,
                        help='number of KAN grid zise of first layer')
    parser.add_argument('--hidden_grid_size', type=int, default=8,
                        help='number of KAN grid zise of hidden layers')
    parser.add_argument('--output_grid_size', type=int, default=8,
                        help='number of KAN grid zise of output layer')
    parser.add_argument('--degree', type=int, default=4,
                        help='number of KAN hidden layer dim')
    
    # optimizer and scheduler
    parser.add_argument('--opt', type=str, default='adam',
                        choices=['schedulefree', 'adam', 'lbfgs'],
                        help='network structure')
    
    # INR Task Shape
    parser.add_argument('--in_features', type=int, default=1,
                        help='input dim of Network')
    parser.add_argument('--out_features', type=int, default=1,
                        help='out dim of Network')
   

    parser.add_argument('--img_wh', nargs="+", type=int, default=[256, 256],
                        help='resolution (img_w, img_h) of the image')
    
    ###### Positional Encoding Parameters
    # FFN
    parser.add_argument('--ffn_scale', type=float, default=10.)
    parser.add_argument('--mapping_input', type=int, default=32) # 256
    # NeRF
    parser.add_argument('--num_frequencies', type=int, default=4)
    # parser.add_argument('--disable_use_nyquist', action='store_false', help='')

    
    parser.add_argument('--act_trainable', default=False, action='store_true',
                        help='whether to train activation hyperparameter')
    

    parser.add_argument('--sine_w0', type=float, default=30.,
                        help='omega in Sine-type activations (SIREN and INCODE)')
    parser.add_argument('--sine_w', type=float, default=30.,
                        help='omega in Sine-type activations (SIREN and INCODE)')
    
    # - When \(a = 1\), it corresponds to the standard Gaussian distribution.
    # - If \( b > 1 \), it makes the values of the Gaussian distribution smoother,
    #  while if \( 0 < b < 1 \), it further enhances the sharpness of the Gaussian.
    parser.add_argument('--gaussian_a', type=float, default=0.1)
    parser.add_argument('--gaussian_b', type=float, default=1.0)
    

    # - When \(a > 1\): The function decays more quickly, resulting in a narrower curve.
    # - When \(0 < a < 1\): The function decays more slowly, resulting in a wider curve.
    # - When \(a = 1\): The function behaves normally without scaling.
    parser.add_argument('--quadratic_a', type=float, default=3.)

    parser.add_argument('--gabor_a', type=float, default=1.)
    parser.add_argument('--gabor_b', type=float, default=1.)
    
    parser.add_argument('--sc', type=float, default=0.1, # default 10.
                        help='fourier feature scale factor (std of the gaussian)')
    

    parser.add_argument('--outermost_linear', type=bool, default=True,
                        help='outermost_linear in siren')

    # Exp
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')
    
    # Logs and Vis
    parser.add_argument('--save_vis', default=False, action='store_true',
                        help='save vis or not')
    parser.add_argument('--vis_every', type=int, default=200,
                        help='check val every n epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20,
                        help='check val every n epoch')
    parser.add_argument('--save_dir', type=str, default='logs',
                        help='experiment log dir')
    parser.add_argument('--exp_name', type=str, default='None',
                        help='experiment name')
    
    # ETOPO1 specific
    parser.set_defaults(in_features=2)
    parser.set_defaults(out_features=1)
    # parser.set_defaults(kan_hidden_layers=5)
    # parser.set_defaults(kan_hidden_features=64)
    # parser.set_defaults(input_grid_size=8)
    # parser.set_defaults(hidden_grid_size=8)
    # parser.set_defaults(output_grid_size=8)

    parser.set_defaults(batch_size=8192)

    parser.set_defaults(gaussian_a=0.1)

    parser.set_defaults(ffn_scale=10)
    parser.set_defaults(mapping_input=32)
    
    parser.set_defaults(img_wh=[1024, 512])
    parser.set_defaults(save_dir='logs/image_regression')

    parser.add_argument('--data_path', type=str, default=ELEVATION_DATA_PATH,
                        help='Path to the pre-processed .nc file. '
                             'Overrides --dataset.')
    parser.add_argument('--dataset', type=str, default='etopo1',
                        choices=DATASET_CHOICES,
                        help='Which pre-processed benchmark dataset to train on. '
                             'Sets data_path and out_features automatically '
                             '(unless --data_path / --out_features are given).')

    # ---- Coordinate-encoding hyperparameters ----
    parser.add_argument('--sh_lmax', type=int, default=32,
                        help='Max SH degree for --ce spherical-harmonics. '
                             'Input dim becomes (L_max+1)^2.')
    parser.add_argument('--rff_num_features', type=int, default=128,
                        help='Number of random Fourier features for '
                             '--ce spherical-rff. Input dim = 2 * num_features.')
    parser.add_argument('--rff_sigma', type=float, default=8.0,
                        help='Frequency scale σ for spherical-RFF '
                             '(ω ~ N(0, σ² I_3) on cartesian coords).')
    parser.add_argument('--rff_seed', type=int, default=42,
                        help='Seed for spherical-RFF frequency draw.')

    parser.set_defaults(arch='mlp')
    parser.set_defaults(act='scaled-sine')
    parser.set_defaults(mlp_act='scaled-sine')
    parser.set_defaults(pe='None')
    parser.set_defaults(save_vis=True)

    # Route --dataset to (data_path, out_features) unless the user set them
    # explicitly on the CLI. We inspect sys.argv to respect explicit overrides.
    # Auto-set in_features from the coordinate encoding so the model's input layer matches.
    import sys as _sys
    from datasets.coord_encodings import coord_encoding_dim
    
    hparams = parser.parse_args()
    ce_kwargs = _encoding_kwargs_from_hparams(hparams, hparams.ce)
    explicit = set(a.split('=')[0] for a in _sys.argv[1:] if a.startswith('--'))
    cfg = DATASET_CONFIG[hparams.dataset]
    if '--data_path' not in explicit:
        hparams.data_path = cfg['path']
    if '--out_features' not in explicit:
        hparams.out_features = cfg['out_features']
    if '--in_features' not in explicit:
        hparams.in_features = coord_encoding_dim(hparams.ce, **ce_kwargs)
    hparams.encoding_kwargs = ce_kwargs
    return hparams


def _encoding_kwargs_from_hparams(hparams, ce_name: str) -> dict:
    """Pull per-encoding hyperparameters off hparams into a plain dict."""
    if ce_name == 'spherical-harmonics':
        return {'L_max': hparams.sh_lmax}
    if ce_name == 'spherical-rff':
        return {
            'num_features': hparams.rff_num_features,
            'sigma': hparams.rff_sigma,
            'seed': hparams.rff_seed,
        }
    return {}