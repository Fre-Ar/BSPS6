"""FKAN positional encoding (Li et al. 2025, INR-Bench Eq. 6/7).

For input x ∈ R^D, the FKAN positional encoding produces the 2DΩ-dim
feature map

    γ(x)_{i,ω,cos} = a_{i,ω} · cos(ω · x_i)
    γ(x)_{i,ω,sin} = b_{i,ω} · sin(ω · x_i)

for i = 1..D and ω = 1..Ω, where (a_{i,ω}, b_{i,ω}) are trainable per
(coordinate, frequency). The integer frequencies 1..Ω give a learnable
analog of NeRF's geometric (2^k) frequencies.

INR-Bench's reported "Gaussian + FKAN" (34.70 dB on image regression,
Table III) uses Ω = 1024 — see Appendix "Positional Encoding Settings":
*"For FKAN positional encoding, the maximum frequency threshold Ω is set
to 1024."* That is our default.

Note: the trainable (a, b) per-(d, ω) scales are mathematically
redundant with the downstream MLP's first layer, which can already
implement any per-feature scaling. They still have practical effects in
optimization (separate gradient flow per scale, the NTK-spectrum-tuning
interpretation in the FKAN paper). a, b are initialized to small
Gaussian noise so the MLP-first-layer + (a, b) joint parameterization
is non-degenerate from step 0.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class FKANEncoding(nn.Module):
    """Single-layer FKAN positional encoding (Eq. 7 of Li et al. 2025).

    Args:
      pos_encode_configs: dict containing key `omega` (int Ω, default 1024).
      in_features: D, the input coordinate dimensionality.

    Attributes:
      out_dim: 2 · D · Ω
    """

    def __init__(self, pos_encode_configs, in_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.omega = int(pos_encode_configs.get('omega', 32))

        # Trainable per-(d, ω) Fourier coefficients. Small Gaussian init so
        # individual PE elements have magnitude ~1/sqrt(Ω), giving an
        # overall feature L2 norm ~ sqrt(2 D) — well-matched to the MLP
        # first layer's Kaiming init.
        a = torch.empty(self.in_features, self.omega)
        b = torch.empty(self.in_features, self.omega)
        std = 1.0 / math.sqrt(self.omega)
        nn.init.normal_(a, mean=0.0, std=std)
        nn.init.normal_(b, mean=0.0, std=std)
        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)

        # Integer frequencies ω = 1, 2, ..., Ω. Registered as a buffer so it
        # moves with the module under .to(device) / Lightning device
        # placement, and lands in state_dict (so a fresh-process checkpoint
        # load reconstructs the same module).
        self.register_buffer(
            'freqs',
            torch.arange(1, self.omega + 1, dtype=torch.float32),
        )

        self.out_dim = 2 * self.in_features * self.omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute γ(x) ∈ R^{2DΩ} for a batch of D-dim coordinates.

        Args:
          x: shape (B, D). For the BSPS6 spherical setting, this is
             typically (x, y, z) on the unit sphere (D = 3) so the integer
             frequencies act on a well-behaved (non-singular) coordinate.

        Returns:
          (B, 2 · D · Ω) feature map.
        """
        # omega_x[b, d, k] = freqs[k] · x[b, d]
        omega_x = x.unsqueeze(-1) * self.freqs  # (B, D, Ω)
        cos_term = self.a * torch.cos(omega_x)  # (B, D, Ω); a broadcasts (D, Ω)→(1, D, Ω)
        sin_term = self.b * torch.sin(omega_x)  # (B, D, Ω)
        # Pair (cos, sin) along a final axis, then flatten. The exact
        # concatenation order does not matter — the MLP's first layer is
        # permutation-invariant over PE features.
        out = torch.stack([cos_term, sin_term], dim=-1)  # (B, D, Ω, 2)
        return out.flatten(start_dim=1)                  # (B, 2 D Ω)
