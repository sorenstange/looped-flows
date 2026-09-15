"""Denoiser backbone: a TRM-style recurrent transformer (paper Appendix A).

One `Denoiser` call is one flow step. It encodes the inputs into one token sequence

    [Q] | context tokens (lookback / patch_size) | a_0 | H noisy trajectory tokens

adds a flow-time embedding to every token, and runs a shared transformer F in a two-state recurrence
`cycles` times:

    ℓ ← F(ℓ + h + e)   (inner_steps times)
    h ← F(h + ℓ)

Only the last cycle is backpropagated, and the incoming recurrent state is detached, which is the stop-gradient between
flow steps of the looped-flow objective. Level logits are decoded from h at the trajectory tokens, the confidence logit
q from h at the [Q] token.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.config import Config
from src.features import FEATURE_NAMES

SEGMENT_Q, SEGMENT_CONTEXT, SEGMENT_POSITION, SEGMENT_TRAJECTORY = range(4)


@dataclass
class RecurrentState:
    h: torch.Tensor  # (B, N, width): state the outputs are decoded from
    l: torch.Tensor  # (B, N, width): latent state used to update h


@dataclass
class DenoiserOutput:
    logits: torch.Tensor  # (B, H, num_levels) level logits of the clean trajectory
    q_logit: torch.Tensor  # (B,) confidence logit
    state: RecurrentState  # detached new recurrent state, to pass into the next flow step


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_tokens: int, base: float):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        angles = torch.outer(torch.arange(max_tokens, dtype=torch.float64), inv_freq)  # (N, head_dim / 2)
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate (B, heads, N, head_dim) queries or keys by their token position."""
        n, half = x.shape[-2], x.shape[-1] // 2
        cos, sin = self.cos[:n].to(x.dtype), self.sin[:n].to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Non-causal multi-head self-attention with rotary position embeddings."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.heads, width // self.heads).transpose(1, 3).unbind(dim=2)
        attended = F.scaled_dot_product_attention(rope(q), rope(k), v)  # (B, heads, N, head_dim)
        return self.out(attended.transpose(1, 2).reshape(batch, tokens, width))


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    """Transformer block with RMSNorm after each residual addition (post-norm, as in TRM)."""

    def __init__(self, width: int, heads: int, mlp_width: int, eps: float):
        super().__init__()
        self.attention = Attention(width, heads)
        self.mlp = SwiGLU(width, mlp_width)
        self.norm_attention = nn.RMSNorm(width, eps=eps)
        self.norm_mlp = nn.RMSNorm(width, eps=eps)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        x = self.norm_attention(x + self.attention(x, rope))
        return self.norm_mlp(x + self.mlp(x))


class Denoiser(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        model = cfg.model
        width = model.width
        self.cycles = model.cycles
        self.inner_steps = model.inner_steps
        self.horizon = cfg.oracle.horizon
        self.num_levels = cfg.oracle.num_levels
        self.context_tokens = cfg.features.lookback // model.patch_size
        self.num_tokens = 1 + self.context_tokens + 1 + self.horizon

        self.context_proj = nn.Linear(len(FEATURE_NAMES) * model.patch_size, width, bias=False)
        self.position_proj = nn.Linear(1, width, bias=False)
        self.trajectory_proj = nn.Linear(self.num_levels, width, bias=False)
        self.segment_embedding = nn.Parameter(torch.empty(4, width))
        self.time_mlp = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.rope = RotaryEmbedding(width // model.heads, self.num_tokens, model.rope_base)
        self.blocks = nn.ModuleList(Block(width, model.heads, model.mlp_width, model.norm_eps)
                                    for _ in range(model.layers))
        self.level_head = nn.Linear(width, self.num_levels, bias=False)
        self.q_head = nn.Linear(width, 1)

        segments = [SEGMENT_Q] + [SEGMENT_CONTEXT] * self.context_tokens + [SEGMENT_POSITION] \
            + [SEGMENT_TRAJECTORY] * self.horizon
        self.register_buffer("segment_ids", torch.tensor(segments), persistent=False)
        self.register_buffer("h_init", nn.init.trunc_normal_(torch.empty(width), std=1.0, a=-2.0, b=2.0))
        self.register_buffer("l_init", nn.init.trunc_normal_(torch.empty(width), std=1.0, a=-2.0, b=2.0))
        self._init_weights()

    def _init_weights(self) -> None:
        # LeCun-normal for all linear layers (as in TRM), so unit-scale inputs give unit-scale activations.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 1.0 / math.sqrt(module.in_features)
                nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # One-hot and scalar inputs act like embedding lookups: unit-scale columns.
        nn.init.trunc_normal_(self.trajectory_proj.weight, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.position_proj.weight, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.segment_embedding, std=0.02, a=-0.04, b=0.04)
        # Confidence starts near 0 so that training does not halt early (as in TRM).
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

    def initial_state(self, batch_size: int) -> RecurrentState:
        shape = (batch_size, self.num_tokens, self.h_init.shape[0])
        return RecurrentState(self.h_init.expand(shape), self.l_init.expand(shape))

    def encode(self, context: torch.Tensor, position: torch.Tensor, noisy: torch.Tensor,
               t: torch.Tensor) -> torch.Tensor:
        """Input tokens e of shape (B, N, width).

        Args:
            context: (B, lookback, F) scaled context features.
            position: (B,) current position a_0.
            noisy: (B, H, num_levels) noisy one-hot trajectory (flow state).
            t: (B,) flow time in [0, 1].
        """
        batch = context.shape[0]
        dtype = self.context_proj.weight.dtype
        tokens = torch.cat([
            torch.zeros(batch, 1, self.segment_embedding.shape[1], dtype=dtype, device=context.device),
            self.context_proj(context.to(dtype).reshape(batch, self.context_tokens, -1)),
            self.position_proj(position.to(dtype)[:, None, None]),
            self.trajectory_proj(noisy.to(dtype)),
        ], dim=1)
        return tokens + self.segment_embedding[self.segment_ids] + self.time_mlp(t.to(dtype)[:, None])[:, None, :]

    def network(self, x: torch.Tensor) -> torch.Tensor:
        """The shared transformer F."""
        for block in self.blocks:
            x = block(x, self.rope)
        return x

    def forward(self, context: torch.Tensor, position: torch.Tensor, noisy: torch.Tensor, t: torch.Tensor,
                state: RecurrentState | None = None) -> DenoiserOutput:
        e = self.encode(context, position, noisy, t)
        state = state if state is not None else self.initial_state(context.shape[0])
        h, l = state.h.detach(), state.l.detach()  # stop-gradient between flow steps
        with torch.no_grad():
            for _ in range(self.cycles - 1):
                h, l = self._cycle(h, l, e)
        h, l = self._cycle(h, l, e)
        logits = self.level_head(h[:, -self.horizon:])
        q_logit = self.q_head(h[:, 0]).squeeze(-1)
        return DenoiserOutput(logits, q_logit, RecurrentState(h.detach(), l.detach()))

    def _cycle(self, h: torch.Tensor, l: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.inner_steps):
            l = self.network(l + h + e)
        return self.network(h + l), l
