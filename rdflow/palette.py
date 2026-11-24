"""sRGB <-> OKLab, palette ordering and turning (palette, pointer tokens) into an RGBA image."""

from __future__ import annotations

import numpy as np

ACHROMATIC_C = 0.04
HUE_FAMILIES = 12


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb uint8 [..., 3] -> OKLab float32 [..., 3]."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ], axis=-1).astype(np.float32)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lab = lab.astype(np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.stack([
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.round(rgb * 255).astype(np.uint8)


def ramp_order(colours: np.ndarray) -> np.ndarray:
    """Sort key order for colours uint8 [n, 3]: hue family (0 = achromatic), then lightness ascending."""
    lab = srgb_to_oklab(colours)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
    family = np.where(chroma < ACHROMATIC_C, 0, 1 + np.minimum((hue // (360 / HUE_FAMILIES)).astype(int),
                                                               HUE_FAMILIES - 1))
    return np.lexsort((colours[:, 2], colours[:, 1], colours[:, 0], lab[:, 0], family))


def decode(pal: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """(palette [P, 3], tokens [H, W]) -> RGBA uint8; exact inverse of encode."""
    t = tokens.astype(np.int64)
    out = np.zeros(t.shape + (4,), np.uint8)
    op = t > 0
    out[op, :3] = pal[t[op] - 1]
    out[op, 3] = 255
    return out
