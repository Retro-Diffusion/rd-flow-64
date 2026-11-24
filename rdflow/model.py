"""RD Flow 64's network: an hourglass transformer over the 64 x 64 pixel grid with a palette-pointer head.

    pixels 64 x 64   window attention  ----------------------------------------------- skip --+
      merge 2x2                                                                               |
    32 x 32          window attention  ------------------------------------- skip --+         |
      merge 2x2                                                                     |         |
    core 16 x 16     global attention jointly over registers, text tokens, palette  |         |
                     tokens and the image                                           |         |
      ...and back up through the skips to the x head (OKLab + alpha) and the pointer head

The x head predicts the clean image. The pointer head scores every pixel against a palette (exact colours) plus a
transparent class, which is how every output pixel ends up an exact palette colour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Block, ColourEmbed, RMSNorm, WindowLevel, grid_xy, merge, rope_tables, split, timestep_embed

CHANNELS = 4          # OKLab L, a, b + alpha, each normalised to about [-1, 1]
ATTR_VOCAB = (8, 5, 5, 4)     # optional attribute conditions (colour count, outline, dither, transparency); 0 = unknown
N_RAMPS = 16                  # palette shading-ramp ids (0 = greys)


@dataclass
class FlowConfig:
    size: int = 64
    d0: int = 128
    d1: int = 256
    d2: int = 512
    dc: int = 1024
    head_dim: int = 64
    depth0: int = 2
    depth1: int = 2
    depth2: int = 2
    depth_core: int = 16
    window: int = 8
    mlp_ratio: float = 3.0
    text_dim: int = 2560
    n_registers: int = 4
    n_styles: int = 512
    n_sources: int = 16
    dk: int = 128
    colour_freqs: int = 8
    colour_hidden: int = 256
    rope_theta: float = 100.0
    core_level: int = 2       # 2: the core runs at 16 x 16 (a quarter of the pixel grid)
    palette_tokens: bool = True
    attrs: bool = True
    text_adapter: int = 2
    self_cond: bool = True
    bg_mode: bool = True


class PixelFlowModel(nn.Module):
    def __init__(self, cfg: FlowConfig = FlowConfig()):
        super().__init__()
        self.cfg = cfg
        assert cfg.core_level == 2, "this release ships the 16 x 16 core"
        n0 = cfg.size
        self.n0, self.n1, self.nc = n0, n0 // 2, n0 // 4
        self.inp = nn.Linear(CHANNELS * (2 if cfg.self_cond else 1), cfg.d0)
        # conditioning vector
        self.t_mlp = nn.Sequential(nn.Linear(256, cfg.dc), nn.SiLU(), nn.Linear(cfg.dc, cfg.dc))
        self.style = nn.Embedding(cfg.n_styles, cfg.dc)
        self.source = nn.Embedding(cfg.n_sources, cfg.dc)
        self.text_pool = nn.Linear(cfg.text_dim, cfg.dc)
        self.text_proj = nn.Linear(cfg.text_dim, cfg.dc)
        self.null_text = nn.Parameter(torch.zeros(1, cfg.text_dim))
        self.registers = nn.Parameter(torch.zeros(cfg.n_registers, cfg.dc))
        self.modality = nn.Parameter(torch.zeros(4 if cfg.palette_tokens else 3, cfg.dc))  # register, text, pixel, palette
        if cfg.attrs:
            self.attr_emb = nn.ModuleList([nn.Embedding(v, cfg.dc) for v in ATTR_VOCAB])
        if cfg.bg_mode:
            self.bg_emb = nn.Embedding(3, cfg.dc)       # 0 unknown, 1 opaque, 2 transparent
        if cfg.text_adapter:
            self.adapter = nn.ModuleList([Block(cfg.dc, cfg, conv=False) for _ in range(cfg.text_adapter)])
        # hourglass
        self.down0 = WindowLevel(cfg.d0, self.n0, cfg.depth0, cfg)
        self.merge0 = nn.Linear(4 * cfg.d0, cfg.d1)
        self.down1 = WindowLevel(cfg.d1, self.n1, cfg.depth1, cfg)
        self.merge1 = nn.Linear(4 * cfg.d1, cfg.dc)
        self.core = nn.ModuleList([Block(cfg.dc, cfg, conv=False) for _ in range(cfg.depth_core)])
        self.core_ada = nn.Linear(cfg.dc, 6 * cfg.dc)
        self.split1 = nn.Linear(cfg.dc, 4 * cfg.d1)
        self.skip1 = nn.Linear(2 * cfg.d1, cfg.d1)
        self.up1 = WindowLevel(cfg.d1, self.n1, cfg.depth1, cfg)
        self.split0 = nn.Linear(cfg.d1, 4 * cfg.d0)
        self.skip0 = nn.Linear(2 * cfg.d0, cfg.d0)
        self.up0 = WindowLevel(cfg.d0, self.n0, cfg.depth0, cfg)
        # heads
        self.out_norm = RMSNorm(cfg.d0)
        self.x_head = nn.Linear(cfg.d0, CHANNELS)
        self.colour = ColourEmbed(cfg)
        if cfg.palette_tokens:
            # a palette colour = its OKLab (Fourier) embedding + which ramp it is on + where on the ramp
            self.ramp_emb = nn.Embedding(N_RAMPS, cfg.colour_hidden)
            self.level_proj = nn.Linear(1, cfg.colour_hidden)
            self.pal_tok = nn.Linear(cfg.colour_hidden, cfg.dc)
        self.q = nn.Linear(cfg.d0, cfg.dk)
        self.k = nn.Sequential(nn.Linear(cfg.colour_hidden, cfg.dk), nn.SiLU(), nn.Linear(cfg.dk, cfg.dk))
        self.transparent = nn.Linear(cfg.d0, 1)

    def palette_embed(self, pal):
        """pal: dict rgb [B, P, 3] long, valid [B, P] bool, optional ramp [B, P] long, level [B, P] float."""
        e = self.colour(pal["rgb"])
        if self.cfg.palette_tokens and pal.get("ramp") is not None:
            e = e + self.ramp_emb(pal["ramp"].clamp(0, N_RAMPS - 1)) + self.level_proj(pal["level"][..., None].float())
        return e

    def forward(self, x_t, t, style, source, text, text_mask, pal=None, attrs=None, sc=None, bg=None):
        """x_t [B, 64, 64, 4] noisy pixels; t [B] in [0, 1] (0 noise, 1 clean); text [B, L, D] + text_mask [B, L] (all
        False = no text); pal: palette tokens for the core (a sample whose valid row is all False has none); attrs
        [B, 4] (0 = unknown); sc [B, 64, 64, 4] the previous step's guess; bg [B] (1 opaque, 2 transparent).
        Returns the predicted clean image [B, 64, 64, 4] and the final pixel features [B, 4096, d0] (pointer head)."""
        cfg = self.cfg
        B = x_t.shape[0]
        dropped = ~text_mask.any(1)
        text = torch.where(dropped[:, None, None], torch.zeros_like(text), text)
        text = torch.cat([self.null_text[None].expand(B, -1, -1).to(text.dtype), text], 1)
        tmask = torch.cat([dropped[:, None], text_mask], 1)
        pooled = (text * tmask[..., None]).sum(1) / tmask.sum(1, keepdim=True).clamp_min(1)
        c = (self.t_mlp(timestep_embed(t, 256)) + self.style(style) + self.source(source)
             + self.text_pool(pooled.to(self.text_pool.weight.dtype)))
        if cfg.attrs and attrs is not None:
            for j, emb in enumerate(self.attr_emb):
                c = c + emb(attrs[:, j])
        if cfg.bg_mode and bg is not None:
            c = c + self.bg_emb(bg)
        xin = x_t.reshape(B, -1, CHANNELS)
        if cfg.self_cond:
            prev = sc if sc is not None else torch.zeros_like(x_t)
            xin = torch.cat([xin, prev.reshape(B, -1, CHANNELS).to(xin.dtype)], -1)
        h0 = self.down0(self.inp(xin), c)
        h1 = self.down1(self.merge0(merge(h0, self.n0)), c)
        ctx = self.text_proj(text.to(self.text_proj.weight.dtype))
        if cfg.text_adapter:
            L0 = ctx.shape[1]
            tc, ts = rope_tables(torch.full((L0, 2), float("nan"), device=x_t.device), cfg.head_dim, cfg.rope_theta)
            zmod = torch.zeros(B, 6, cfg.dc, device=x_t.device, dtype=ctx.dtype)
            amask = tmask[:, None, None, :]
            for blk in self.adapter:
                ctx = blk(ctx, zmod, lambda attn, h: attn(h, tc, ts, amask))
        hc = self.merge1(merge(h1, self.n1))
        R, L = cfg.n_registers, text.shape[1]
        parts = [self.registers[None].expand(B, -1, -1) + self.modality[0], ctx + self.modality[1]]
        masks = [torch.ones(B, R, dtype=torch.bool, device=x_t.device), tmask]
        Pn = 0
        if cfg.palette_tokens and pal is not None:
            Pn = int(pal["valid"].shape[1])
            pt = self.pal_tok(self.palette_embed(pal).to(self.pal_tok.weight.dtype))
            parts.append(pt + self.modality[3])
            masks.append(pal["valid"])
        S = R + L + Pn
        tok = torch.cat(parts + [hc + self.modality[2]], 1)
        nan = torch.full((S, 2), float("nan"), device=x_t.device)
        cos, sin = rope_tables(torch.cat([nan, grid_xy(self.nc, x_t.device)]), cfg.head_dim, cfg.rope_theta)
        keymask = torch.cat(masks + [torch.ones(B, self.nc ** 2, dtype=torch.bool, device=x_t.device)],
                            1)[:, None, None, :]
        mod = self.core_ada(F.silu(c)).view(B, 6, cfg.dc)
        for blk in self.core:
            tok = blk(tok, mod, lambda attn, h: attn(h, cos, sin, keymask))
        hc = tok[:, S:]
        u1 = self.up1(self.skip1(torch.cat([split(self.split1(hc), self.n1), h1], -1)), c)
        u0 = self.out_norm(self.up0(self.skip0(torch.cat([split(self.split0(u1), self.n0), h0], -1)), c))
        return self.x_head(u0).view(B, self.n0, self.n0, CHANNELS), u0

    def pointer_logits(self, feats, pal_rgb, pal_valid, ramp=None, level=None):
        """feats [B, N, d0]; pal_rgb [B, P, 3] long (exact sRGB); pal_valid [B, P] bool. -> [B, N, 1 + P] logits:
        0 = transparent, 1 + s = palette slot s (invalid slots -inf)."""
        keys = self.k(self.palette_embed({"rgb": pal_rgb, "valid": pal_valid, "ramp": ramp, "level": level}))
        logits = torch.einsum("bnk,bpk->bnp", self.q(feats), keys.to(feats.dtype)) / math.sqrt(self.cfg.dk)
        logits = logits.float().masked_fill(~pal_valid[:, None, :], float("-inf"))
        return torch.cat([self.transparent(feats).float(), logits], -1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
