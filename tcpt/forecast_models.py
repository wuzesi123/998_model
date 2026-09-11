from __future__ import annotations

import torch
from torch import nn

from .model import CausalEncoder, ContinuousTimeEncoding


class MLPForecaster(nn.Module):
    def __init__(self, input_dim: int, short_len: int, output_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * short_len, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, short_x, **kwargs):
        return self.net(short_x)


class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 192, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, output_dim))

    def forward(self, global_x=None, short_x=None, **kwargs):
        x = global_x if global_x is not None else short_x
        y, _ = self.gru(x)
        return self.head(y[:, -1])


class TCPTForecaster(nn.Module):
    """Three-scale causal TCPT trunk with a regression head.

    This is intentionally a forecast-only head.  No weak stage, progress or transition
    targets are used anywhere in this benchmark.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 192,
        nhead: int = 6,
        layers: int = 3,
        ff: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.time_encoding = ContinuousTimeEncoding(d_model)
        self.short_encoder = CausalEncoder(d_model, nhead, layers, ff, dropout)
        self.medium_encoder = CausalEncoder(d_model, nhead, layers, ff, dropout)
        self.global_encoder = CausalEncoder(d_model, nhead, layers, ff, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, output_dim)

    def _embed(self, x, t):
        h = self.input_proj(self.input_norm(x))
        return self.time_encoding(h, t)

    def forward(
        self,
        short_x,
        medium_x,
        global_x,
        short_time=None,
        medium_time=None,
        global_time=None,
        **kwargs,
    ):
        s = self.short_encoder(self._embed(short_x, short_time))[:, -1]
        m = self.medium_encoder(self._embed(medium_x, medium_time))[:, -1]
        g = self.global_encoder(self._embed(global_x, global_time))[:, -1]
        h = self.fusion(torch.cat([s, m, g], dim=-1))
        return self.head(h)
