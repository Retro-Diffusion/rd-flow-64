"""RD Flow 64 inference: a caption -> 64 x 64 RGBA pixel art, every pixel an exact palette colour or transparent.

    from rdflow import Pipeline
    pipe = Pipeline.load()                       # picks devices for this machine
    images = pipe("A small red potion bottle ...", n=4)

Two parts, placed independently: the prompt reader (Qwen3-VL-4B's first 27 layers, 3.1B parameters, bfloat16) and the
drawing model (271M parameters). On a small GPU the reader runs on the CPU and the drawing model on the GPU; with no
GPU both run on the CPU.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .flow import sample
from .model import FlowConfig, PixelFlowModel
from .palette import decode, ramp_order
from .text import TextEncoder, threads_hint

WEIGHTS_REPO = "Retro-Diffusion/rd-flow-64"
WEIGHTS_FILE = "rd-flow-64.safetensors"
GB = 1024 ** 3
# the prompt reader goes on the GPU only when it fits there with the drawing model and some room to spare
TEXT_ON_GPU_MIN_GB = 10
PROMPT_CACHE = 32          # captions kept (each is a few MB of features)
TEXT_ON_MPS_MIN_RAM_GB = 15   # a "16 GB" Mac reports a little under 16


def parse_hex(colours: Sequence[str]) -> np.ndarray:
    """["#aabbcc" | "aabbcc" | "abc", ...] -> uint8 [k, 3], deduplicated, in palette ramp order (at most 256)."""
    out = []
    for c in colours:
        h = c.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) == 8:                     # AARRGGBB (paint.net) or RRGGBBAA: keep the colour part
            h = h[2:] if c.strip().upper().startswith("FF") else h[:6]
        if len(h) != 6:
            raise ValueError(f"not a colour: {c!r}")
        out.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
    arr = np.unique(np.asarray(out, dtype=np.uint8).reshape(-1, 3), axis=0)[:256]
    return arr[ramp_order(arr)] if len(arr) else arr


def read_palette_file(path: str) -> np.ndarray:
    """A palette file from Lospec or an art program: .hex, .txt (paint.net), .gpl (GIMP), .pal (JASC)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    cols = []
    if lines and lines[0].strip() == "JASC-PAL":
        for line in lines[3:]:
            p = line.split()
            if len(p) >= 3:
                cols.append("%02x%02x%02x" % tuple(int(v) for v in p[:3]))
    elif lines and lines[0].strip() == "GIMP Palette":
        for line in lines[1:]:
            p = line.split()
            if len(p) >= 3 and all(v.isdigit() for v in p[:3]):
                cols.append("%02x%02x%02x" % tuple(int(v) for v in p[:3]))
    else:
        for line in lines:
            s = line.strip()
            if s and not s.startswith(";"):
                cols.append(s)
    return parse_hex(cols)


@dataclass
class Placement:
    text_device: str
    model_device: str
    model_dtype: torch.dtype

    def __str__(self):
        return f"prompt reader on {self.text_device}, drawing model on {self.model_device} ({str(self.model_dtype)[6:]})"


def _ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
    except (AttributeError, ValueError, OSError):
        return 0.0


def choose_placement(text_device: str = "auto", model_device: str = "auto", precision: str = "auto") -> Placement:
    if model_device == "auto":
        if torch.cuda.is_available():
            model_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model_device = "mps"
        else:
            model_device = "cpu"
    if text_device == "auto":
        text_device = "cpu"
        if model_device == "cuda" and torch.cuda.get_device_properties(0).total_memory >= TEXT_ON_GPU_MIN_GB * GB \
                and torch.cuda.is_bf16_supported():
            text_device = "cuda"
        elif model_device == "mps" and _ram_gb() >= TEXT_ON_MPS_MIN_RAM_GB:
            text_device = "mps"     # unified memory: on smaller Macs the memory-mapped CPU reader is the one that fits
    if precision == "auto":
        # bf16 where the GPU has it, else full precision (1.5 GB of VRAM for 4 images, so it still fits a 2 GB card)
        dtype = torch.bfloat16 if model_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    elif precision == "fp16":
        # trained in bf16: its activations overflow fp16's range and every image comes out empty
        raise ValueError("RD Flow 64 does not run in fp16 (its activations overflow it); use bf16 or fp32")
    else:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
    return Placement(text_device, model_device, dtype)


def _files(path: Optional[str], cache_dir: Optional[str]) -> tuple[str, Optional[str]]:
    """The weights file and its config.json: a local file (config.json beside it, if there is one) or the Hugging Face
    repo's copies, downloaded once."""
    if path:
        cfg = Path(path).with_name("config.json")
        return path, str(cfg) if cfg.is_file() else None
    from huggingface_hub import hf_hub_download
    return (hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE, cache_dir=cache_dir),
            hf_hub_download(WEIGHTS_REPO, "config.json", cache_dir=cache_dir))


def _config(path: Optional[str]) -> FlowConfig:
    """The architecture from config.json; the built-in defaults are the same model."""
    if not path:
        return FlowConfig()
    known = {f.name for f in fields(FlowConfig)}
    arch = json.loads(Path(path).read_text(encoding="utf-8")).get("config", {})
    return FlowConfig(**{k: v for k, v in arch.items() if k in known})


class Pipeline:
    def __init__(self, model: PixelFlowModel, text: TextEncoder, placement: Placement):
        self.model, self.text, self.placement = model, text, placement
        self.timings: dict = {}
        self._prompts: OrderedDict = OrderedDict()   # caption -> features: a caption is read once, then reused

    def read(self, caption: str):
        """The prompt reader's features for a caption, cached: more images or seeds of the same caption skip it."""
        if caption in self._prompts:
            self._prompts.move_to_end(caption)
            return self._prompts[caption]
        feats = self.text([caption])
        self._prompts[caption] = feats
        while len(self._prompts) > PROMPT_CACHE:
            self._prompts.popitem(last=False)
        return feats

    @classmethod
    def load(cls, weights: Optional[str] = None, *, text_device: str = "auto", model_device: str = "auto",
             precision: str = "auto", cache_dir: Optional[str] = None, compile: bool = False) -> "Pipeline":
        from safetensors.torch import load_file

        threads_hint()
        place = choose_placement(text_device, model_device, precision)
        weights_file, config_file = _files(weights, cache_dir)
        model = PixelFlowModel(_config(config_file))
        state = load_file(weights_file)
        model.load_state_dict({k: v.to(place.model_dtype) for k, v in state.items()})
        model = model.to(place.model_device, place.model_dtype).eval().requires_grad_(False)
        if compile:
            model = torch.compile(model, dynamic=True)
        text = TextEncoder(place.text_device, cache_dir=cache_dir)
        return cls(model, text, place)

    def _autocast(self):
        dev = self.placement.model_device
        half = self.placement.model_dtype in (torch.float16, torch.bfloat16)
        return torch.autocast(device_type=dev if dev != "mps" else "cpu", dtype=self.placement.model_dtype,
                              enabled=half and dev == "cuda")

    @torch.no_grad()
    def __call__(self, caption: str, n: int = 4, seed: Optional[int] = None, transparent: bool = True,
                 palette: Optional[Sequence[str] | np.ndarray] = None, steps: int = 16,
                 guidance: float = 4.0) -> list[Image.Image]:
        """n images for one caption. seed: the first image's seed (the rest follow it); random if None. palette: hex
        strings or a [k, 3] uint8 array to lock every pixel to; None lets the model choose its own."""
        dev = self.placement.model_device
        seed = int(np.random.randint(0, 2**31 - 1)) if seed is None else int(seed)
        seeds = [seed + i for i in range(n)]
        t0 = time.perf_counter()
        feats, mask = self.read(caption)
        t1 = time.perf_counter()
        feats = feats.to(dev, self.placement.model_dtype).expand(n, -1, -1)
        mask = mask.to(dev).expand(n, -1)
        noise = torch.stack([torch.randn((64, 64, 4), generator=torch.Generator().manual_seed(s)) for s in seeds]).to(dev)
        bg = torch.full((n,), 2 if transparent else 1, dtype=torch.long, device=dev)
        lock = None
        if palette is not None and len(palette):
            pal = palette if isinstance(palette, np.ndarray) else parse_hex(list(palette))
            lock = [pal] * n
        with self._autocast():
            out = sample(self.model, feats, mask, noise, bg, steps=steps, guidance=guidance, lock_palette=lock)
        pal = out["pal"].cpu().numpy().astype(np.uint8)
        tok = out["tokens"].cpu().numpy()
        images = [Image.fromarray(decode(pal[j], tok[j]), "RGBA") for j in range(n)]
        self.timings = {"prompt_s": round(t1 - t0, 2), "draw_s": round(time.perf_counter() - t1, 2), "seeds": seeds}
        return images
