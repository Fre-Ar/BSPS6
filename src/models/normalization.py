import torch
import torch.nn as nn

class CrossNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = (x.sum(1,keepdim=True).expand(x.size())+x.sum(0,keepdim=True).expand(x.size())-x)/(x.size()[0]+x.size()[1])
        v = (x.pow(2).sum(1,keepdim=True).expand(x.size())+x.pow(2).sum(0,keepdim=True).expand(x.size())-x.pow(2))/(x.size()[0]+x.size()[1]) - u.pow(2)
        return (x-u)/torch.sqrt(x.var(0, keepdim=True)+x.var(1, keepdim=True)+self.eps)*self.weight + self.bias
    
class GlobalNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        return ((x - torch.mean(x)) / torch.sqrt(torch.var(x) + self.eps))*self.weight + self.bias
    

def norm_factory(norm_type, dim=256):
    norm_dict = \
    {
        'batch': nn.BatchNorm1d(dim),
        'layer': nn.LayerNorm(dim),
        'instance': nn.InstanceNorm1d(dim),
        'cross': CrossNorm(dim),
        'global': GlobalNorm(dim),
    }
    return norm_dict[norm_type]