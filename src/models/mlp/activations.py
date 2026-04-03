import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import functools
import operator as op

class SincActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.where(x == 0, torch.tensor(1e-7), x)
        return torch.sin(torch.pi * x) / (torch.pi * x)

# Gaussian Activations
class GaussianActivation(nn.Module):
    def __init__(self, a=1.0, trainable=False):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
    def forward(self, x):
        return torch.exp(-x**2/(2* self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))

# SIREN
class SineActivation(nn.Module):
    def __init__(self, w=30., trainable=False):
        super().__init__()
        self.register_parameter('w', nn.Parameter(w*torch.ones(1), trainable))

    def forward(self, x):
        return torch.sin(self.w * x)
    
class MSoftplusActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input) - self.cst
    
class SwishActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.sigmoid(input)

# WIRE
class WireActivation(nn.Module):
    def __init__(self, a=10., b=40., trainable=True):
        super().__init__()

        self.register_parameter('omega_0', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('scale_0', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        omega = self.omega_0 * x
        scale = self.scale_0 * x
        
        return torch.exp(1j* omega - scale.abs().square())
    

class RealGaborActivation(nn.Module):
    '''
    Real Gabor Activation Function
    
    This activation function applies a Gabor transformation 
    using learnable frequency and scaling parameters.
    
    Inputs:
        omega: Frequency of Gabor sinusoid term
        scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, a=1.0, b=1.0, trainable=True):
        super().__init__()
        # Define trainable parameters
        self.omega = nn.Parameter(a*torch.ones(1), requires_grad=trainable)
        self.scale = nn.Parameter(b*torch.ones(1), requires_grad=trainable)

    def forward(self, x):
        omega_term = self.omega * x
        scale_term = self.scale * x.abs()

        return torch.cos(omega_term) * torch.exp(-(scale_term**2))
    
class ComplexGaborActivation(nn.Module):
    '''
    Complex Gabor Activation Function
    
    This activation function applies a complex Gabor transformation 
    using learnable frequency and scaling parameters.
    
    Inputs:
        omega: Frequency of Gabor sinusoid term
        sigma: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, a=1.0, b=1.0, trainable=True):
        super().__init__()
        self.omega = nn.Parameter(a*torch.ones(1), requires_grad=trainable)
        self.sigma = nn.Parameter(b*torch.ones(1), requires_grad=trainable)

    def forward(self, x):
        omega_term = self.omega * x
        scale_term = self.sigma * x.abs().square()
        
        return torch.exp(1j * omega_term - scale_term)
    
# INCODE
class ScaledSineActivation(nn.Module):
    def __init__(self, a=0.1993, b=0.0196, c=0.0588, d=0.0269, w=30., trainable=True):
        super().__init__()
        self.w = torch.tensor(w)
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))
        self.register_parameter('c', nn.Parameter(c*torch.ones(1), trainable))
        self.register_parameter('d', nn.Parameter(d*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(self.a) * torch.sin(torch.exp(self.b) * self.w * x + self.c) + self.d


"""
Elementwise nonlinear tensor operations.

code from neuromancer: https://github.com/pnnl/neuromancer
"""
def soft_exp(alpha, x):
    """
    Helper function for SoftExponential learnable activation class. Also used in neuromancer.operators.InterpolateAddMultiply
    :param alpha: (float) Parameter controlling shape of the function.
    :param x: (torch.Tensor) Arbitrary shaped tensor input
    :return: (torch.Tensor) Result of the function applied elementwise on the tensor.
    """
    if alpha == 0.0:
        return x
    elif alpha < 0.0:
        return -torch.log(1 - alpha * (x + alpha)) / alpha
    else:
        return (torch.exp(alpha * x) - 1) / alpha + alpha


class SoftExponential(nn.Module):
    """
    Soft exponential activation: https://arxiv.org/pdf/1602.01321.pdf
    """

    def __init__(self, alpha=0.0, tune_alpha=True):
        """

        :param alpha: (float) Value to initialize parameter controlling the shape of the function
        :param tune_alpha: (bool) Whether alpha is a learnable parameter or fixed
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=tune_alpha)

    def forward(self, x):
        """

        :param x: (torch.Tensor) Arbitrary shaped tensor
        :return: (torch.Tensor) Tensor same shape as input after elementwise application of soft exponential function
        """
        return soft_exp(self.alpha, x)


class BLU(nn.Module):
    """
    Bendable Linear Units: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8913972
    """

    def __init__(self, tune_alpha=False, tune_beta=True):
        """

        :param tune_alpha: (bool) Whether alpha is learnable parameter or fixed
        :param tune_beta: (bool) Whether beta is a learnable parameter of fixed
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.epsilon = 1e-7 if tune_alpha else 0.0
        self.epsilon = nn.Parameter(torch.tensor(self.epsilon), requires_grad=tune_beta)

    def forward(self, x):
        """

        :param x: (torch.Tensor) Arbitrary shaped input tensor
        :return: (torch.Tensor) Tensor same shape as input after bendable linear unit adaptation
        """
        return (
            self.beta
            * (torch.sqrt(x * x + self.alpha * self.alpha + 1e-7) - self.alpha)
            + x
        )


class APLU(nn.Module):
    """
    Adaptive Piecewise Linear Units: https://arxiv.org/pdf/1412.6830.pdf
    """

    def __init__(
        self,
        nsegments=2,
        alpha_reg_weight=1e-3,
        beta_reg_weight=1e-3,
        tune_alpha=True,
        tune_beta=True,
    ):
        """

        :param nsegments: (int) Number of segments in piecewise linear unit activation function
        :param alpha_reg_weight: (float) Strength of regularization on alpha parameter vector
        :param beta_reg_weight: (float) Strength of regularization on beta parameter vector
        :param tune_alpha: (bool) Whether to tune alpha of piecewise linear functions
        :param tune_beta: (bool) Whether to tune beta of piecewise linear functions
        """
        super().__init__()
        self.nsegments = nsegments
        self.alpha_reg_weight = alpha_reg_weight
        self.beta_reg_weight = beta_reg_weight
        self.alpha = nn.Parameter(torch.rand(nsegments), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.rand(nsegments), requires_grad=tune_beta)

    def reg_error(self):
        """
        L2 regularization on parameters of piecewise linear activation
        :return: (float) Regularization penalty
        """
        return self.alpha_reg_weight * torch.norm(
            self.alpha
        ) + self.beta_reg_weight * torch.norm(self.beta)

    def forward(self, x):
        """

        :param x: (torch.Tensor) Arbitrary shaped tensor
        :return: (torch.Tensor) Tensor same shape as input after elementwise application of piecewise linear activation
        """
        y = F.relu(x)
        for i in range(self.nsegments):
            y += self.alpha[i] * F.relu(-x + self.beta[i])
        return y


class PReLU(nn.Module):
    """
    Parametric ReLU: https://arxiv.org/pdf/1502.01852.pdf
    """

    def __init__(self, tune_alpha=True, tune_beta=True):
        """

        :param tune_alpha: (bool) Whether to tune slope on negative range elements
        :param tune_beta: (bool) Whether to tune slope on positive range elements
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=tune_beta)

    def forward(self, x):
        """

        :param x: (torch.Tensor) Arbitrary shaped input tensor
        :return: (torch.Tensor) Tensor same shape as input after parametric ReLU activation.
        """
        neg = self.alpha * -F.relu(-x)
        pos = self.beta * F.relu(x)
        return neg + pos


class PELU(nn.Module):
    """
    Parametric Exponential Linear Units: https://arxiv.org/pdf/1605.09332.pdf
    """

    def __init__(self, tune_alpha=True, tune_beta=True):
        """

        :param tune_alpha: (bool) Whether to tune alpha of parametric ELU functions
        :param tune_beta: (bool) Whether to tune beta of parametric ELU functions
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(-1.), requires_grad=tune_alpha)
        self.beta = nn.Parameter(torch.tensor(-1.), requires_grad=tune_beta)

    def forward(self, x):
        """

        :param x: (torch.Tensor) Arbitrary shaped input tensor
        :return: (torch.Tensor) Tensor same shape as input after parametric ELU activation.
        """
        posx = F.relu(x)
        negx = F.relu(-x)
        return (self.alpha / self.beta) * posx + self.alpha * (
            torch.exp(negx / self.beta) - 1
        )


class RectifiedSoftExp(nn.Module):
    """
    Mysterious unexplained implementation of Soft Exponential ported from author's Keras code:
    https://github.com/thelukester92/2019-blu/blob/master/python/activations/softexp.py
    """

    def __init__(self, tune_alpha=True):
        """

        :param tune_alpha: (bool) Whether alpha is a learnable parameter or fixed
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=tune_alpha)
        self.epsilon = 1e-7

    def forward(self, x):
        """

        :param x: (torch.Tensor) Arbitrary shaped tensor
        :return: (torch.Tensor) Tensor same shape as input after elementwise application of soft exponential function
        """
        neg_alpha = F.relu(-torch.clamp(self.alpha, -1, 1)) + self.epsilon()
        pos_alpha = F.relu(torch.clamp(self.alpha, -1, 1)) + self.epsilon()
        pos_x = F.relu(x) + self.epsilon()
        log = torch.log(neg_alpha * pos_x + 1) / neg_alpha
        exp = (torch.exp(pos_alpha * pos_x) - 1) / pos_alpha
        return log + exp


class SmoothedReLU(nn.Module):
    """
    ReLU with a quadratic region in [0,d]; Rectified Huber Unit;
    Used to make the Lyapunov function continuously differentiable
    https://arxiv.org/pdf/2001.06116.pdf
    """
    def __init__(self, d=1.0, tune_d=True):
        """

        :param d:
        :param tune_d:
        """
        super().__init__()
        self.d = nn.Parameter(torch.tensor(d), requires_grad=tune_d)

    def forward(self, x):
        alpha = 1.0 / F.softplus(self.d)
        beta = - F.softplus(self.d) / 2
        return torch.max(torch.clamp(torch.sign(x) * torch.div(alpha, 2.0) * x ** 2, min=0, max=-beta.item()), x + beta)
    

# NTK Activations
class ABRelu(nn.Module):
    def __init__(self, a: float, b: float, trainable: bool = False):
        """
        ABReLU nonlinearity, i.e. `a * min(x, 0) + b * max(x, 0)`.

        Args:
            a: Initial slope for `x < 0`.
            b: Initial slope for `x > 0`.
            trainable: If True, the slopes `a` and `b` are learnable parameters.
        """
        super(ABRelu, self).__init__()
        if trainable:
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        else:
            self.a = torch.tensor(a, dtype=torch.float32, requires_grad=False)
            self.b = torch.tensor(b, dtype=torch.float32, requires_grad=False)

    def forward(self, x):
        negative_part = torch.minimum(x, torch.tensor(0.0, device=x.device))
        positive_part = torch.maximum(x, torch.tensor(0.0, device=x.device))
        return self.a * negative_part + self.b * positive_part

# ReLU activation: Equivalent to ABRelu(0, 1)
def Relu(trainable: bool = False):
    """
    ReLU nonlinearity.
    Args:
        trainable: If True, allows the slope `b` to be trainable.
    """
    return ABRelu(0, 1, trainable)

# Leaky ReLU activation: Equivalent to ABRelu(alpha, 1)
def LeakyRelu(alpha: float, trainable: bool = False):
    """
    Leaky ReLU nonlinearity, i.e. `alpha * min(x, 0) + max(x, 0)`.
    
    Args:
        alpha: Slope for `x < 0`.
        trainable: If True, allows the slopes to be trainable.
    """
    return ABRelu(alpha, 1, trainable)

# Absolute value activation: Equivalent to ABRelu(-1, 1)
def Abs(trainable: bool = False):
    """
    Absolute value nonlinearity.
    
    Args:
        trainable: If True, allows the slopes to be trainable.
    """
    return ABRelu(-1, 1, trainable)


class Sign(nn.Module):
    def __init__(self):
        """
        Sign function.
        Returns -1 for negative numbers, 1 for positive numbers, and 0 for zeros.
        """
        super(Sign, self).__init__()

    def forward(self, x):
        return torch.sign(x)
    

class Exp(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0, trainable: bool = False):
        """
        Elementwise natural exponent function `a * exp(b * x)`.

        Args:
            a: Initial coefficient for the exponential function.
            b: Initial coefficient for the input scaling.
            trainable: If True, `a` and `b` are learnable parameters.
        """
        super(Exp, self).__init__()
        if trainable:
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        else:
            self.a = torch.tensor(a, dtype=torch.float32, requires_grad=False)
            self.b = torch.tensor(b, dtype=torch.float32, requires_grad=False)

    def forward(self, x):
        # Apply the elementwise exponential function
        return self.a * torch.exp(self.b * x)
    

# class GaussianActivation(nn.Module):
#     def __init__(self, a: float = 1.0, b: float = 1.0, trainable: bool = False):
#         super().__init__()
#         self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)
#         self.b = nn.Parameter(torch.tensor(b), requires_grad=trainable)

#     def forward(self, x):
#         return self.a * torch.exp(-self.b * x ** 2)
    

class HermiteActivation(nn.Module):
    def __init__(self, degree: int, trainable: bool = False):
        """
        Hermite polynomial activation with optional trainable coefficients.
        
        Args:
            degree (int): Degree of the Hermite polynomial.
            trainable (bool): If True, make the polynomial coefficients trainable.
        """
        super().__init__()
        if degree < 0:
            raise ValueError("`degree` must be a non-negative integer.")
        
        # Generate the Hermite polynomial coefficients
        p = np.polynomial.hermite_e.herme2poly([0] * degree + [1])[::-1]
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32), requires_grad=trainable) 
    
        # Coefficient for normalizing the output
        self.coeff = torch.sqrt(torch.tensor(float(functools.reduce(op.mul, range(1, degree + 1), 1))))

    def forward(self, x):
        # Manually evaluate the polynomial since torch.polyval doesn't exist
        result = self.p[0] * torch.ones_like(x)
        for coeff in self.p[1:]:
            result = result * x + coeff
        
        return result / self.coeff
    

class Monomial(nn.Module):
    def __init__(self, degree: int):
        """
        Monomial activation function, i.e., `x^degree`.
        
        Args:
            degree (int): An integer between 0 and 5.
        """
        super().__init__()
        if degree not in [0, 1, 2, 3, 4, 5]:
            raise NotImplementedError('The `degree` must be an integer between 0 and 5.')
        
        self.degree = degree

    def forward(self, x):
        """
        Forward pass that applies the monomial function.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying `x^degree`.
        """
        return x ** self.degree


class RectifiedMonomial(nn.Module):
    def __init__(self, degree: int):
        """
        Rectified monomial activation function, i.e., `(x >= 0) * x^degree`.
        
        Args:
            degree (int): A non-negative integer power.
        """
        super().__init__()
        if degree < 0:
            raise ValueError('`degree` must be a non-negative integer.')
        
        self.degree = degree

    def forward(self, x):
        """
        Forward pass that applies the rectified monomial function.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying `(x >= 0) * x^degree`.
        """
        return torch.where(x >= 0, x ** self.degree, torch.zeros_like(x))
    

class Sine(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.0, trainable: bool = True):
        """
        Affine transform of the sine nonlinearity, i.e., `a * sin(b * x + c)`.

        Args:
            a (float): Output scale.
            b (float): Input scale.
            c (float): Input phase shift.
            trainable (bool): If True, make the parameters trainable.
        """
        super().__init__()
        # Register parameters as trainable if specified
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=trainable)
        self.c = nn.Parameter(torch.tensor(c), requires_grad=trainable)

    def forward(self, x):
        """
        Forward pass that applies the sine function with affine transformation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the affine sine transformation.
        """
        return self.a * torch.sin(self.b * x + self.c)
    
class Erf(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.0, trainable: bool = True):
        """
        Affine transform of the Erf nonlinearity, i.e., `a * Erf(b * x) + c`.

        Args:
            a (float): Output scale.
            b (float): Input scale.
            c (float): Output shift.
            trainable (bool): If True, make the parameters trainable.
        """
        super().__init__()
        # Register parameters as trainable if specified
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=trainable)
        self.c = nn.Parameter(torch.tensor(c), requires_grad=trainable)

    def forward(self, x):
        """
        Forward pass that applies the Erf function with affine transformation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the affine Erf transformation.
        """
        return self.a * torch.erf(self.b * x) + self.c
    
def Sigmoid_like():
  """A sigmoid like function `f(x) = .5 * erf(x / 2.4020563531719796) + .5`.

  The constant `2.4020563531719796` is chosen so that the squared loss between
  this function and the ground truth sigmoid is minimized on the interval
  `[-5, 5]`; see
  https://gist.github.com/SiuMath/679e8bb4bce13d5f2383a27eca649575.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return Erf(a=0.5, b=1/2.4020563531719796, c=0.5)


class Gabor(nn.Module):
    def __init__(self):
        """
        Gabor function defined as `exp(-x^2) * sin(x)`.
        """
        super(Gabor, self).__init__()

    def forward(self, x):
        """
        Forward pass that applies the Gabor function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Gabor function.
        """
        return torch.exp(-x ** 2) * torch.sin(x)
    

class Gelu(nn.Module):
    def __init__(self, approximate: bool = False):
        """
        GELU activation function.

        Args:
            approximate (bool): If True, computes an approximation via tanh.
        """
        super(Gelu, self).__init__()
        self.approximate = approximate

    def forward(self, x):
        """
        Forward pass that applies the GELU function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the GELU function.
        """
        if self.approximate:
            # Approximation of GELU using tanh
            return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * x ** 3)))
        else:
            # Exact GELU
            return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        
class Cos(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.0, trainable: bool = False):
        """
        Affine transform of `Cos` nonlinearity, i.e. `a * cos(b * x + c)`.

        Args:
            a (float): Output scale.
            b (float): Input scale.
            c (float): Input phase shift.
            trainable (bool): If True, make the parameters trainable.
        """
        super(Cos, self).__init__()
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=trainable)
        self.c = nn.Parameter(torch.tensor(c), requires_grad=trainable)

    def forward(self, x):
        """
        Forward pass that applies the cosine transformation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the cosine transformation.
        """
        return self.a * torch.cos(self.b * x + self.c)




class DualRbf(nn.Module):
    def __init__(self, gamma: float = 16.0, trainable: bool = False):
        """
        Dual activation function for normalized RBF or squared exponential kernel.
        
        Args:
            gamma (float): Related to characteristic length-scale (l) that controls width of the kernel,
                           where `gamma = 1 / (2 l^2)`.
            trainable (bool): If True, make gamma a trainable parameter.
        """
        super(DualRbf, self).__init__()
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.gamma = torch.tensor(gamma, requires_grad=False)

    def forward(self, x):
        """
        Forward pass that applies the RBF transformation.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the RBF transformation.
        """
        return torch.sqrt(torch.tensor(2.0)) * torch.sin(torch.sqrt(2 * self.gamma) * x + (torch.pi / 4))


def activation_factory(act='relu', act_trainable=False, **kwargs):
    activation_functions = \
        {
            # ReLU
            'relu': nn.ReLU(inplace=True),
            'prelu': nn.PReLU(),
            'selu': nn.SELU(inplace=True), # normal('relu') is better than normal('selu')
            'gelu': nn.GELU(),
            'elu': nn.ELU(inplace=True),
            "blu": BLU(),
            "pelu": PELU(tune_alpha=act_trainable, tune_beta=act_trainable),
            "rrelu": nn.RReLU(inplace=True),
            "relu6": nn.ReLU6(inplace=True),
            "leakyrelu": nn.LeakyReLU(),
            "smoothedrelu": SmoothedReLU(),
            'silu': nn.SiLU(inplace=True),
            # Sine
            'sine': SineActivation(w=kwargs['sine_w'], trainable=act_trainable),
            'first-sine': SineActivation(w=kwargs['sine_w0'], trainable=act_trainable),
            'scaled-sine': ScaledSineActivation(w=kwargs['sine_w'],trainable=True),
            "dualrbf": DualRbf(trainable=act_trainable),
            # Gaussian
            'gaussian': GaussianActivation(a=kwargs['gaussian_a'], trainable=act_trainable),
            'laplacian': LaplacianActivation(a=kwargs['gaussian_a'], trainable=act_trainable),
            'super-gaussian': SuperGaussianActivation(a=kwargs['gaussian_a'], b=kwargs['gaussian_b'], trainable=act_trainable),
            # Quadratic
            'quadratic': QuadraticActivation(a=kwargs['quadratic_a'], trainable=act_trainable),
            'multi-quadratic': MultiQuadraticActivation(a=kwargs['quadratic_a'], trainable=act_trainable),
            # Exp * Sine(Cos)
            'expsin': ExpSinActivation(a=kwargs['gabor_a'], trainable=act_trainable), # a=1.0 is better
            'realgabor': RealGaborActivation(a=kwargs['gabor_a'], b=kwargs['gabor_b'], trainable=act_trainable),
            "gabor":  Gabor(),
            'sinc': SincActivation(),
            # complex need new model arch
            # 'complexgabor':(ComplexGaborActivation(trainable=False), init_weights_normal),
            # Others
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softplus': nn.Softplus(),
            'swish': SwishActivation(), 
            'msoftplus': MSoftplusActivation(),
            "softexp": SoftExponential(),
            "hardtanh": nn.Hardtanh(inplace=True),
        }
       
    return activation_functions[act]