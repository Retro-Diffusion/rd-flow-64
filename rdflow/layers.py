"""Network building blocks: attention with 2-D rotary positions, conv-SwiGLU, adaLN blocks, window levels, the
OKLab colour embedding and the timestep embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def rope_tables(xy: torch.Tensor, head_dim: int, theta: float):
    """xy [N, 2] float positions (NaN = no position) -> cos, sin [N, head_dim // 2]."""
    quarter = head_dim // 4
    freqs = theta ** (-torch.arange(quarter, device=xy.device, dtype=torch.float32) / quarter)
    has = ~torch.isnan(xy[:, 0])
    pos = torch.nan_to_num(xy, nan=0.0)
    ang = torch.cat([pos[:, :1] * freqs, pos[:, 1:] * freqs], dim=-1)      # [N, head_dim/2]
    ang = ang * has[:, None]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x [..., N, Dh]; rotate pairs (first half, second half)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


def grid_xy(n: int, device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten()], -1).float()


class Attention(nn.Module):
    def __init__(self, d: int, head_dim: int):
        super().__init__()
        self.h = d // head_dim
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.qn, self.kn = RMSNorm(head_dim), RMSNorm(head_dim)

    def forward(self, x, cos, sin, mask=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.qn(q), cos, sin), apply_rope(self.kn(k), cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(B, N, D))


class SwiGLU(nn.Module):
    """Conv-SwiGLU on grids (3x3 depthwise conv on the hidden), plain SwiGLU otherwise."""

    def __init__(self, d: int, ratio: float, conv: bool):
        super().__init__()
        h = int(d * ratio)
        self.w1 = nn.Linear(d, 2 * h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.conv = nn.Conv2d(h, h, 3, padding=1, groups=h) if conv else None

    def forward(self, x, hw=None):
        a, g = self.w1(x).chunk(2, -1)
        if self.conv is not None:
            B, N, C = a.shape
            H, W = hw
            a = self.conv(a.transpose(1, 2).reshape(B, C, H, W)).reshape(B, C, N).transpose(1, 2)
        return self.w2(F.silu(g) * a)


class Block(nn.Module):
    """Pre-norm block with adaLN-single modulation (shared level MLP + per-block table)."""

    def __init__(self, d: int, cfg, conv: bool):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, cfg.head_dim)
        self.mlp = SwiGLU(d, cfg.mlp_ratio, conv)
        self.table = nn.Parameter(torch.zeros(6, d))

    def forward(self, x, mod, attn_fn, hw=None):
        sh1, sc1, g1, sh2, sc2, g2 = (mod + self.table[None]).unbind(1)          # each [B, d]
        x = x + g1[:, None] * attn_fn(self.attn, self.n1(x) * (1 + sc1[:, None]) + sh1[:, None])
        x = x + g2[:, None] * self.mlp(self.n2(x) * (1 + sc2[:, None]) + sh2[:, None], hw)
        return x


class WindowLevel(nn.Module):
    """A stack of window-attention blocks on an n x n grid (shifted windows on odd blocks)."""

    def __init__(self, d: int, n: int, depth: int, cfg):
        super().__init__()
        self.n, self.w = n, cfg.window
        self.blocks = nn.ModuleList([Block(d, cfg, conv=True) for _ in range(depth)])
        self.ada = nn.Linear(cfg.dc, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.head_dim, self.theta = cfg.head_dim, cfg.rope_theta

    def forward(self, x, c):
        B, N, D = x.shape
        n, w = self.n, self.w
        mod = self.ada(F.silu(c)).view(B, 6, D)
        xy = grid_xy(n, x.device)
        for i, blk in enumerate(self.blocks):
            shift = (w // 2) if i % 2 else 0
            cos, sin = rope_tables(self._windows(xy[None], shift).reshape(-1, 2), self.head_dim, self.theta)
            nw = (n // w) ** 2
            cos = cos.view(nw, w * w, -1)[:, None]
            sin = sin.view(nw, w * w, -1)[:, None]

            def attn_fn(attn, h, shift=shift, cos=cos, sin=sin):
                hw_ = self._windows(h, shift)                                    # [B*nw, w*w, D]
                o = attn(hw_, cos.repeat(B, 1, 1, 1), sin.repeat(B, 1, 1, 1))
                return self._unwindows(o, shift, B)

            x = blk(x, mod, attn_fn, hw=(n, n))
        return x

    def _windows(self, x, shift):
        B, N, D = x.shape
        n, w = self.n, self.w
        g = x.view(B, n, n, D)
        if shift:
            g = torch.roll(g, (-shift, -shift), (1, 2))
        g = g.view(B, n // w, w, n // w, w, D).permute(0, 1, 3, 2, 4, 5)
        return g.reshape(B * (n // w) ** 2, w * w, D)

    def _unwindows(self, x, shift, B):
        n, w = self.n, self.w
        D = x.shape[-1]
        g = x.view(B, n // w, n // w, w, w, D).permute(0, 1, 3, 2, 4, 5).reshape(B, n, n, D)
        if shift:
            g = torch.roll(g, (shift, shift), (1, 2))
        return g.reshape(B, n * n, D)


def merge(x, n):
    """[B, n*n, D] -> [B, (n/2)^2, 4D] (2x2 patches)."""
    B, _, D = x.shape
    g = x.view(B, n // 2, 2, n // 2, 2, D).permute(0, 1, 3, 2, 4, 5)
    return g.reshape(B, (n // 2) ** 2, 4 * D)


def split(x, n):
    """[B, (n/2)^2, 4D] -> [B, n*n, D]."""
    B, _, D4 = x.shape
    D = D4 // 4
    g = x.view(B, n // 2, n // 2, 2, 2, D).permute(0, 1, 3, 2, 4, 5)
    return g.reshape(B, n * n, D)


class ColourEmbed(nn.Module):
    """c(r, g, b) = MLP(Fourier(OKLab(r, g, b)))."""

    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(cfg.colour_freqs)) * math.pi, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(3 + 6 * cfg.colour_freqs, cfg.colour_hidden), nn.SiLU(),
                                 nn.Linear(cfg.colour_hidden, cfg.colour_hidden))

    @staticmethod
    def oklab(rgb):
        c = rgb.float() / 255.0
        c = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        r, g, b = c.unbind(-1)
        l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).clamp_min(0) ** (1 / 3)
        m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).clamp_min(0) ** (1 / 3)
        s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).clamp_min(0) ** (1 / 3)
        return torch.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1)

    def forward(self, rgb):
        lab = self.oklab(rgb)
        ang = lab[..., None] * self.freqs                                        # [..., 3, F]
        feats = torch.cat([lab, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)
        return self.mlp(feats)


def timestep_embed(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)
