from torch import nn

from models.encodings.fkan_encoding import FKANEncoding
from .encodings.frequency_encoding import FrequencyEncoding
from .encodings.gaussian_encoding import GaussianEncoding

ENCODING_DICT = {'frequency': FrequencyEncoding,
                 'gaussian': GaussianEncoding,
                 'fkan': FKANEncoding,
                }

class PosEncoding(nn.Module):
    def __init__(self, encoding=None):
        if encoding != None:
            self.encoding = ENCODING_DICT[encoding]
        
    def run(self, *args, **kwargs):
        return self.encoding(*args, **kwargs)