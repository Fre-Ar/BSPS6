"""Positional encoding dispatch.

The PosEncoding wrapper exists to keep INR.__init__ uniform across the two
PE families this benchmark uses (RFF and FKAN). For `--pe None`, INR.py
skips PE construction entirely and feeds the raw coord-encoded input
directly to the MLP.
"""
from torch import nn
from .encodings.fkan_encoding import FKANEncoding
from .encodings.gaussian_encoding import GaussianEncoding

ENCODING_DICT = {
    'gaussian': GaussianEncoding,
    'fkan': FKANEncoding,
}

class PosEncoding(nn.Module):
    """Thin factory: `PosEncoding('fkan').run(in_features=..., pos_encode_configs=...)`."""

    def __init__(self, encoding: str):
        if encoding not in ENCODING_DICT:
            raise ValueError(
                f"Unknown PE '{encoding}'. Available: {list(ENCODING_DICT)}."
            )
        self.encoding_cls = ENCODING_DICT[encoding]

    def run(self, *args, **kwargs):
        return self.encoding_cls(*args, **kwargs)