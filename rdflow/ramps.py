"""Shading ramps of a palette: which colours belong together and where each sits from dark to light. The model takes
these with every palette colour."""

from __future__ import annotations

import math

import numpy as np

NEUTRAL_C = 0.035          # OKLab chroma under which a colour is a grey
HUE_GAP = math.radians(28)  # a ramp breaks where neighbouring hues are further apart than this
HUE_SPAN = math.radians(75)  # ...or where one group spans more than this (split at its largest gap)
MAX_RAMPS = 15              # ramp ids 1..15 (0 = neutral greys)

def ramps(pal_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OKLab palette [n, 3] -> (ramp id [n] int, level [n] float in 0..1, dark to light). Greys are ramp 0; chromatic
    colours group by hue (circular gaps), each group ordered by lightness."""
    n = len(pal_lab)
    ramp = np.zeros(n, np.int64)
    level = np.full(n, 0.5, np.float32)
    if n == 0:
        return ramp, level
    L, a, b = pal_lab[:, 0], pal_lab[:, 1], pal_lab[:, 2]
    chroma = np.hypot(a, b)
    hue = np.arctan2(b, a) % (2 * math.pi)
    chromatic = np.flatnonzero(chroma >= NEUTRAL_C)
    groups = []
    if len(chromatic):
        order = chromatic[np.argsort(hue[chromatic])]
        h = hue[order]
        gaps = np.diff(np.concatenate([h, h[:1] + 2 * math.pi]))
        start = int(np.argmax(gaps)) + 1                  # cut the circle at its widest gap
        order = np.roll(order, -start)
        h = np.unwrap(hue[order])
        cur = [order[0]]
        for i in range(1, len(order)):
            if h[i] - h[i - 1] > HUE_GAP:
                groups.append(cur)
                cur = []
            cur.append(order[i])
        groups.append(cur)
        split = []
        for g in groups:                                  # over-wide groups split at their largest internal gap
            while len(g) > 2:
                hg = np.unwrap(hue[g])
                if hg[-1] - hg[0] <= HUE_SPAN:
                    break
                k = int(np.argmax(np.diff(hg))) + 1
                split.append(g[:k])
                g = g[k:]
            split.append(g)
        groups = split
    neutral = np.flatnonzero(chroma < NEUTRAL_C)
    for rid, g in [(0, list(neutral))] + [(min(i + 1, MAX_RAMPS), g) for i, g in enumerate(groups)]:
        if not len(g):
            continue
        g = np.asarray(g)
        ramp[g] = rid
        if len(g) > 1:
            rank = np.argsort(np.argsort(L[g]))
            level[g] = rank / (len(g) - 1)
    return ramp, level
