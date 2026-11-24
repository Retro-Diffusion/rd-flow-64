"""The sampler: flow matching in OKLab + alpha pixel space, finished by the palette pointer.

Time runs from noise to data: x_t = t * x + (1 - t) * noise, t in [0, 1]. The network predicts the clean image; Euler
steps follow the implied velocity, with classifier-free guidance over most of the run. Once the image is mostly formed
the palette is fixed (a locked one, or extracted from the model's own guess), the pointer head picks a palette slot for
every pixel, and the result is an exact palette colour or transparent everywhere.
"""

from __future__ import annotations

import numpy as np
import torch

from .consts import P
from .layers import ColourEmbed

AB_SCALE = 3.0            # OKLab a, b (about +-0.3 for sRGB) -> about +-1
PALETTE_BUCKETS = (32, 64, 128, 256)
SOURCE = 6                # the source tag the model is sampled with (its cleanest training source)


def from_model(x):
    """x [..., 4] -> (OKLab [..., 3], opaque [...] bool)."""
    x = x.float()
    lab = torch.stack([(x[..., 0] + 1) / 2, x[..., 1] / AB_SCALE, x[..., 2] / AB_SCALE], -1)
    return lab, x[..., 3] > 0


def oklab_to_srgb(lab):
    """OKLab float [..., 3] -> sRGB long [..., 3] (0..255)."""
    L, a, b = lab.float().unbind(-1)
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = torch.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                       -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                       -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1).clamp(0, 1)
    rgb = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb.clamp_min(1e-12) ** (1 / 2.4) - 0.055)
    return (rgb * 255).round().long()


def bucket_palette(pal):
    """Cut a palette dict to the smallest of a few fixed lengths that holds every image's colours (None if empty)."""
    if pal is None:
        return None
    need = int(pal["valid"].sum(1).max())
    if need == 0:
        return None
    P_ = int(pal["valid"].shape[1])
    keep = min(P_, next((b for b in PALETTE_BUCKETS if b >= need), P_))
    return {k: (v[:, :keep] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == P_ else v) for k, v in pal.items()}


def extract_palette(x, max_colours: int = 64, merge: float = 0.05, bin_size: float = 0.015, iters: int = 4):
    """One image's continuous guess x [n, n, 4] -> palette sRGB long [k, 3]. Bins the opaque pixels' OKLab, keeps the
    most common bin means that are not within `merge` of one already kept, then refines with a few k-means passes."""
    lab, op = from_model(x)
    pts = lab[op].float().cpu().double().numpy()   # float64 on the CPU: Apple GPUs have none
    if not len(pts):
        return torch.zeros(0, 3, dtype=torch.long)
    keys, inv, counts = np.unique(np.round(pts / bin_size).astype(np.int64), axis=0, return_inverse=True,
                                  return_counts=True)
    inv = inv.reshape(-1)
    sums = np.zeros((len(keys), 3))
    np.add.at(sums, inv, pts)
    means = sums / counts[:, None]
    centres = []
    for i in np.argsort(-counts, kind="stable"):
        c = means[i]
        if not centres or np.min(np.linalg.norm(np.asarray(centres) - c, axis=1)) > merge:
            centres.append(c)
            if len(centres) >= max_colours:
                break
    C = np.asarray(centres)
    for _ in range(iters):
        a = np.argmin(((pts[:, None] - C[None]) ** 2).sum(-1), 1)
        keep = np.bincount(a, minlength=len(C)) > 0
        C = np.stack([pts[a == k].mean(0) for k in range(len(C)) if keep[k]])
    rgb = oklab_to_srgb(torch.from_numpy(C))
    return torch.unique(rgb, dim=0)


def _has(p):
    return p is not None and len(p) > 0


def palette_dict(colour_lists, device):
    """Per-image colour lists -> the palette dict the model takes: rgb [B, P, 3], valid [B, P], ramp [B, P], level."""
    from .palette import srgb_to_oklab
    from .ramps import ramps
    B = len(colour_lists)
    rgb = torch.zeros(B, P, 3, dtype=torch.long, device=device)
    valid = torch.zeros(B, P, dtype=torch.bool, device=device)
    ramp = torch.zeros(B, P, dtype=torch.long, device=device)
    level = torch.full((B, P), 0.5, device=device)
    for b, cols in enumerate(colour_lists):
        c = np.asarray(torch.as_tensor(cols).cpu() if torch.is_tensor(cols) else cols, dtype=np.uint8).reshape(-1, 3)[:P]
        k = len(c)
        if not k:
            continue
        r, lv = ramps(srgb_to_oklab(c))
        rgb[b, :k] = torch.as_tensor(c.astype(np.int64), device=device)
        valid[b, :k] = True
        ramp[b, :k] = torch.as_tensor(r, device=device)
        level[b, :k] = torch.as_tensor(lv, dtype=torch.float32, device=device)
    return {"rgb": rgb, "valid": valid, "ramp": ramp, "level": level}


def palette_model_space(pal):
    """[B, 1 + P, 4]: the model-space value of transparent (index 0) and of each palette slot."""
    lab = ColourEmbed.oklab(pal["rgb"])
    X = torch.stack([2 * lab[..., 0] - 1, lab[..., 1] * AB_SCALE, lab[..., 2] * AB_SCALE,
                     torch.ones_like(lab[..., 0])], -1)
    T = torch.tensor([0.0, 0.0, 0.0, -1.0], device=X.device).expand(X.shape[0], 1, 4)
    return torch.cat([T, X], 1)


@torch.no_grad()
def sample(model, text, text_mask, noise, bg, *, steps: int = 16, guidance: float = 4.0, interval=(0.0, 0.9),
           shift: float = 3.0, lock_palette=None, palette_at: float = 0.6, max_colours: int = 64,
           merge: float = 0.05, pointer_from: float = 0.35, commit_from: float = 0.55, commit_thresh: float = 0.9):
    """Generate B images. text [B, L, D] + text_mask [B, L]; noise [B, 64, 64, 4] (one seed per image, so an image
    never depends on its batch); bg [B] (1 opaque, 2 transparent); lock_palette: per-image list of (r, g, b), or None
    to let the model choose its palette at t = palette_at. Returns dict(pal [B, P, 3], tokens [B, 64, 64])."""
    cfg = model.cfg
    dev = text.device
    B = text.shape[0]
    n = cfg.size
    style = torch.zeros(B, dtype=torch.long, device=dev)
    source = torch.full((B,), SOURCE, dtype=torch.long, device=dev)
    attrs = torch.zeros(B, 4, dtype=torch.long, device=dev)
    x = noise
    ts = torch.linspace(0.0, 1.0, steps + 1, device=dev)
    ts = ts / (ts + shift * (1 - ts))           # more steps at high noise, where the layout is decided
    null_mask = torch.zeros_like(text_mask)
    locked = lock_palette is not None and all(_has(p) for p in lock_palette)
    pal = palette_dict(lock_palette, dev) if locked else None
    Xpal = palette_model_space(pal) if pal is not None else None
    feats = x_hat = sc = probs = None
    for i in range(steps):
        t, t2 = ts[i], ts[i + 1]
        tv = t.expand(B)
        if pal is None and float(t) >= palette_at and x_hat is not None:
            pal = palette_dict([extract_palette(x_hat[b], max_colours=max_colours, merge=merge) for b in range(B)], dev)
            Xpal = palette_model_space(pal)
        guide = guidance != 1.0 and interval[0] <= float(t) <= interval[1]
        kw = {}
        pb = bucket_palette(pal) if pal is not None else None
        if pb is not None:
            kw["pal"] = {k: torch.cat([v, v]) for k, v in pb.items()} if guide else pb
        kw["attrs"] = torch.cat([attrs, torch.zeros_like(attrs)]) if guide else attrs
        kw["bg"] = torch.cat([bg, bg]) if guide else bg
        if sc is not None:
            kw["sc"] = torch.cat([sc, sc]) if guide else sc
        if guide:
            xh, fe = model(torch.cat([x, x]), torch.cat([tv, tv]), torch.cat([style, torch.zeros_like(style)]),
                           torch.cat([source, source]), torch.cat([text, text]), torch.cat([text_mask, null_mask]), **kw)
            xc, xu = xh.float().chunk(2)
            x_hat, feats, feats_u = xu + guidance * (xc - xu), fe[:B], fe[B:]
        else:
            xh, feats = model(x, tv, style, source, text, text_mask, **kw)
            x_hat, feats_u = xh.float(), None
        if pal is not None and float(t) >= pointer_from:
            def expected(f):
                lg = model.pointer_logits(f, pal["rgb"], pal["valid"], pal["ramp"], pal["level"])
                lg[..., 1:] = lg[..., 1:].masked_fill(~pal["valid"].any(1)[:, None, None], float("-inf"))
                p = lg.softmax(-1)
                return p, (p @ Xpal).view(B, n, n, 4)
            probs, xp = expected(feats)
            if feats_u is not None:
                _, xpu = expected(feats_u)
                xp = xpu + guidance * (xp - xpu)
            if float(t) >= commit_from:
                conf, idx = probs.max(-1)
                exact = torch.gather(Xpal, 1, idx[..., None].expand(-1, -1, 4)).view(B, n, n, 4)
                xp = torch.where((conf >= commit_thresh).view(B, n, n, 1), exact, xp)
            x_hat = xp
        sc = x_hat
        x = x + (t2 - t) * (x_hat - x) / (1 - t).clamp_min(1e-4)
    if pal is None:
        pal = palette_dict([lock_palette[b] if lock_palette is not None and _has(lock_palette[b]) else
                            extract_palette(x_hat[b], max_colours=max_colours, merge=merge) for b in range(B)], dev)
    if probs is not None:
        tokens = probs.argmax(-1).view(B, n, n)
    else:
        logits = model.pointer_logits(feats, pal["rgb"], pal["valid"], pal["ramp"], pal["level"])
        logits[..., 1:] = logits[..., 1:].masked_fill(~pal["valid"].any(1)[:, None, None], float("-inf"))
        tokens = logits.argmax(-1).view(B, n, n)
    return {"pal": pal["rgb"], "tokens": tokens}
