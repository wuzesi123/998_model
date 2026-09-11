from __future__ import annotations

import math
import torch
from torch import nn


class CausalEncoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 2, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _causal_mask(n: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._causal_mask(x.size(1), x.device)
        y = self.encoder(x, mask=mask)
        return self.norm(y)


class ContinuousTimeEncoding(nn.Module):
    """Causal continuous-time encoding using real elapsed seconds.

    Two views are encoded and averaged:
      1) log elapsed time since experiment start;
      2) log age of each token relative to the current token.

    This avoids treating irregularly sampled 1 s and 60 s gaps as identical token steps.
    It uses only timestamps available at inference time and never uses the true endpoint.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)

    def _encode_scalar(self, v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # v: [B, T], assumed finite and >= 0 after log1p.
        b, t = v.shape
        d = self.d_model
        half = d // 2
        if half == 0:
            return torch.zeros(b, t, d, device=v.device, dtype=dtype)
        freqs = torch.exp(
            torch.arange(half, device=v.device, dtype=dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        ang = v.to(dtype).unsqueeze(-1) * freqs.view(1, 1, -1)
        pe = torch.zeros(b, t, d, device=v.device, dtype=dtype)
        pe[..., 0:2 * half:2] = torch.sin(ang)[..., : pe[..., 0:2 * half:2].shape[-1]]
        pe[..., 1:2 * half:2] = torch.cos(ang)[..., : pe[..., 1:2 * half:2].shape[-1]]
        return pe

    def forward(self, x: torch.Tensor, time_s: torch.Tensor | None = None) -> torch.Tensor:
        if time_s is None:
            # Backward-compatible fallback for older callers/checkpoints.
            pos = torch.arange(x.size(1), device=x.device, dtype=x.dtype).view(1, -1).expand(x.size(0), -1)
            return x + self._encode_scalar(torch.log1p(pos), x.dtype)

        t = time_s.to(device=x.device, dtype=x.dtype)
        if t.ndim == 1:
            t = t.unsqueeze(0).expand(x.size(0), -1)
        # Dataset supplies elapsed seconds from experiment start, but clamp defensively.
        elapsed = torch.log1p(torch.clamp(t, min=0.0))
        age = torch.log1p(torch.clamp(t[:, -1:].expand_as(t) - t, min=0.0))
        pe = 0.5 * self._encode_scalar(elapsed, x.dtype) + 0.5 * self._encode_scalar(age, x.dtype)
        return x + pe


class TCPTv4(nn.Module):
    """TCPT V5.2 temporal core.

    The class name is retained for checkpoint/caller compatibility, but the temporal
    representation is upgraded to three causal time scales:
      - short: recent physical-time window;
      - medium: intermediate physical-time window;
      - global: sparse history from experiment start to the current time.

    Each branch receives real timestamps through ContinuousTimeEncoding.
    """

    def __init__(
        self,
        input_dim: int,
        num_stages: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 2,
        ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_stages = num_stages

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
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

        self.stage_head = nn.Linear(d_model, num_stages)
        self.transition_heat_head = nn.Linear(d_model, 1)
        self.transition_distance_head = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())
        self.progress_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def _embed(self, x: torch.Tensor, time_s: torch.Tensor | None) -> torch.Tensor:
        h = self.input_proj(self.input_norm(x))
        return self.time_encoding(h, time_s)

    def forward(
        self,
        short_x: torch.Tensor,
        medium_x: torch.Tensor | None = None,
        long_x: torch.Tensor | None = None,
        short_time: torch.Tensor | None = None,
        medium_time: torch.Tensor | None = None,
        long_time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Backward compatibility with the old two-argument model(sx, lx).
        if long_x is None:
            long_x = medium_x if medium_x is not None else short_x
            medium_x = short_x
            if long_time is None:
                long_time = medium_time
            medium_time = short_time
        if medium_x is None:
            medium_x = short_x

        s = self.short_encoder(self._embed(short_x, short_time))[:, -1]
        m = self.medium_encoder(self._embed(medium_x, medium_time))[:, -1]
        g = self.global_encoder(self._embed(long_x, long_time))[:, -1]
        h = self.fusion(torch.cat([s, m, g], dim=-1))
        heat_logit = self.transition_heat_head(h).squeeze(-1)
        return {
            "stage_logits": self.stage_head(h),
            "transition_heat_logit": heat_logit,
            "transition_logit": heat_logit,
            "transition_distance_norm": self.transition_distance_head(h).squeeze(-1),
            "progress": self.progress_head(h).squeeze(-1),
            "embedding": h,
        }
