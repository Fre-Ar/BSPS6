import torch
from torch import nn
import numpy as np
import math
from functools import partial

def init_weights_normal(m, nonlinearity='relu'):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity=nonlinearity, mode='fan_in')


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))

def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))

def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)

def init_weights_uniform(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

def init_weights_sine(m, w=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w, np.sqrt(6 / num_input) / w)


def init_weights_sine_first(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input), np.sqrt(6/num_input))


def weight_init_factory(act='relu', **kwargs):
    activation_inits = \
        {
            # ReLU
            'relu': init_weights_normal,
            'prelu': init_weights_normal,
            'selu': init_weights_normal, # normal('relu') is better than normal('selu')
            'gelu': init_weights_normal,
            'elu': init_weights_elu,
            "blu": init_weights_normal,
            "pelu": init_weights_normal,
            "rrelu": init_weights_normal,
            "relu6": init_weights_normal,
            "leakyrelu": partial(init_weights_normal, nonlinearity='leaky_relu'),
            "smoothedrelu": init_weights_normal,
            'silu': init_weights_normal,
            # Sine
            'sine': partial(init_weights_sine, w=kwargs['sine_w']),
            'first-sine': init_weights_sine_first,
            'scaled-sine': partial(init_weights_sine, w=kwargs['sine_w']),
            "dualrbf": None,
            # Gaussian
            'gaussian': init_weights_normal, # # normal init is more better than xavier
            'laplacian': init_weights_normal,
            'super-gaussian': init_weights_normal,
            # Quadratic
            'quadratic': init_weights_normal,
            'multi-quadratic': init_weights_normal,
            # Exp * Sine(Cos)
            'expsin': init_weights_normal, # a=1.0 is better
            'realgabor': init_weights_normal,
            "gabor": None,
            'sinc': init_weights_normal,
            # complex need new model arch
            # 'complexgabor':(ComplexGaborActivation(trainable=False), init_weights_normal),
            # Others
            'sigmoid': partial(init_weights_normal, nonlinearity='sigmoid'),
            'tanh': init_weights_normal, # normal init is more better than xavier
            'softplus': init_weights_normal, # normal init is more better than xavier
            'swish': init_weights_normal, # normal init is more better than init_weights_selu
            'msoftplus': init_weights_normal,
            "softexp": None,
            "hardtanh": init_weights_normal, # normal is better
        }
       
    return activation_inits[act]