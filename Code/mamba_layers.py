"""Lightweight optional Mamba-style sequence blocks.

This file provides minimal drop‑in wrappers so that we can progressively
replace GRU temporal modules in `social_vae.py` without introducing a hard
dependency on external libraries. If the real `mamba_ssm` package is
available it will be used; otherwise a fallback pseudo-Mamba block (gated
1D conv + linear projection) is used so that enabling the flags does not
crash, though quality / speed benefits of true Mamba will not appear.

Design goals:
 - Same forward signature as torch.nn.GRU for (T,B,C) input when batch_first=False.
 - Return hidden state shaped (num_layers, B, D) to ease swap logic.
 - Keep parameter count roughly comparable to GRU(hidden_dim) for fair ablations.

Tiered usage (as per integration plan):
 - Tier1: Replace encoder temporal loop (rnn_fx) after attention aggregation.
 - Tier2: Replace decoder temporal loop (rnn_fy) for latent rollout.
 - Higher tiers may introduce cross‑agent fusion inside Mamba (not yet implemented).

NOTE: This is intentionally conservative; real Mamba includes selective scan
kernel optimizations and state space parameterization. For quick prototyping
we approximate with depthwise + pointwise conv + GLU gating.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

try:  # pragma: no cover
    # Attempt to import official / reference implementation.
    # Users may later install: pip install mamba-ssm
    from mamba_ssm import Mamba  # type: ignore
    _HAS_MAMBA = True
except Exception:  # pragma: no cover
    Mamba = None  # fallback to pseudo block
    _HAS_MAMBA = False


class PseudoMambaBlock(nn.Module):
    """Fallback approximation when real Mamba is unavailable.

    Architecture:
        x -> LayerNorm -> DWConv(k=5, padding=2) -> GLU -> PW Linear -> Residual
    Maintains no recurrent hidden state; we synthesize a dummy final state by
    mean‑pooling last time step (to mimic GRU interface needed by caller).
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.gate = nn.Linear(d_model, d_model * 2)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T,B,D)
        T, B, D = x.shape
        y = self.norm(x)
        y_conv = self.dw(y.permute(1,2,0)).permute(2,0,1)  # (T,B,D)
        gate_in = self.gate(y_conv)
        g, v = gate_in.chunk(2, dim=-1)
        y = torch.sigmoid(g) * v
        y = self.proj(y)
        y = self.dropout(y)
        y = y + x  # residual
        # Dummy hidden state = last timestep (1,B,D)
        h_new = y[-1:].clone()
        return y, h_new


class MambaBlock(nn.Module):
    """Wrapper that uses real Mamba if present else pseudo fallback."""
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        if _HAS_MAMBA:
            # Real implementation: single layer Mamba module.
            self.impl = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.impl = PseudoMambaBlock(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        # Delegate - unify return to (y, h_new)
        if isinstance(self.impl, PseudoMambaBlock):
            return self.impl(x, h)
        # Real Mamba expects (B,T,C) typically; adapt shapes.
        # We convert (T,B,D) -> (B,T,D)
        y = self.impl(x.transpose(0,1))  # (B,T,D)
        y = y.transpose(0,1).contiguous()
        h_new = y[-1:].clone()  # (1,B,D)
        return y, h_new


class StackedMamba(nn.Module):
    """Stack multiple Mamba (or pseudo) blocks to emulate num_layers of GRU.

    Exposed interface: forward(x, h=None) -> (y, h_all)
      x: (T,B,D)
      h_all: (num_layers, B, D)
    """
    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        hs = []
        out = x
        for i, layer in enumerate(self.layers):
            hi = None if h is None else h[i:i+1]
            out, h_new = layer(out, hi)
            hs.append(h_new)
        h_cat = torch.cat(hs, dim=0)  # (L,B,D)
        return out, h_cat


def build_mamba_or_none(use: bool, d_model: int, num_layers: int, dropout: float):
    """Factory returning a StackedMamba if use else None."""
    if not use:
        return None
    return StackedMamba(d_model=d_model, num_layers=num_layers, dropout=dropout)


__all__ = [
    'MambaBlock', 'StackedMamba', 'build_mamba_or_none'
]
