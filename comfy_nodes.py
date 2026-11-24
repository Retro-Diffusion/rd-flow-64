"""ComfyUI nodes for RD Flow 64.

    RD Flow 64 Loader          the drawing model + prompt reader (downloaded once from Hugging Face)
    RD Flow 64 Generate        caption -> 64 x 64 pixel art: IMAGE, MASK (alpha), the palette and an RGBA image
    RD Flow 64 Caption Writer  a short idea -> the long caption the model needs, with any OpenAI-compatible LLM
    RD Flow 64 System Prompt   the recommended system prompt + request, for your own LLM node
    RD Flow 64 Clean Caption   the clean-up to run on an LLM's answer (background sentence, banned words)
"""

from __future__ import annotations

import os
import re

import numpy as np
import torch

from .rdflow import Pipeline, parse_hex
from .rdflow.llm import OLLAMA, write_caption
from .rdflow.prompting import finalize, messages

CATEGORY = "RD Flow 64"
SHORT_WORDS = 40
_PIPES: dict = {}


def _palette(text: str):
    cols = [c for c in re.split(r"[\s,;]+", text or "") if c.strip()]
    return parse_hex(cols) if cols else None


class RDFlow64Loader:
    @classmethod
    def INPUT_TYPES(cls):
        dev = ["auto", "cpu", "cuda", "mps"]
        return {"required": {
                    "text_device": (dev, {"default": "auto", "tooltip": "Where the 3.1B prompt reader runs. auto: the GPU "
                                          "only if it has 10 GB or more, else the CPU (it reads each caption once)."}),
                    "model_device": (dev, {"default": "auto", "tooltip": "Where the 271M drawing model runs."}),
                    "precision": (["auto", "bf16", "fp32"], {"default": "auto",
                                  "tooltip": "auto: bf16 on GPUs that have it, else fp32. fp16 is not supported."})},
                "optional": {
                    "weights_path": ("STRING", {"default": "", "tooltip": "A local rd-flow-64.safetensors (with its "
                                     "config.json beside it). Empty: download from Hugging Face."})}}

    RETURN_TYPES = ("RDFLOW64",)
    RETURN_NAMES = ("rd_flow_64",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, text_device, model_device, precision, weights_path=""):
        key = (text_device, model_device, precision, weights_path or "")
        if key not in _PIPES:
            _PIPES.clear()          # one model in memory at a time
            _PIPES[key] = Pipeline.load(weights_path or None, text_device=text_device, model_device=model_device,
                                        precision=precision)
            print(f"[RD Flow 64] loaded: {_PIPES[key].placement}")
        return (_PIPES[key],)


class RDFlow64Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "rd_flow_64": ("RDFLOW64",),
                    "caption": ("STRING", {"multiline": True, "default": "", "tooltip": "A long, plain description "
                                "(120-200 words) works best. Use the Caption Writer node to make one from a short idea."}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
                    "batch_size": ("INT", {"default": 4, "min": 1, "max": 16}),
                    "background": (["transparent", "opaque"], {"default": "transparent"}),
                    "steps": ("INT", {"default": 16, "min": 4, "max": 64}),
                    "guidance": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5,
                                 "tooltip": "3-4 is best; higher swells subjects and flattens colour."}),
                    "scale": ("INT", {"default": 8, "min": 1, "max": 16, "tooltip": "Nearest-neighbour upscale of the "
                              "64 x 64 output, so it is visible in previews. 1 = the real pixels."})},
                "optional": {
                    "palette": ("STRING", {"default": "", "tooltip": "Lock every pixel to these colours: hex codes "
                                "separated by commas or spaces. Empty: the model makes its own palette."})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "mask", "palette", "rgba")
    OUTPUT_TOOLTIPS = ("RGB on black, for other nodes", "the alpha (1 = transparent)", "the first image's colours",
                       "RGBA: Save Image writes it as a transparent PNG")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, rd_flow_64, caption, seed, batch_size, background, steps, guidance, scale, palette=""):
        caption = (caption or "").strip()
        if not caption:
            raise ValueError("RD Flow 64 needs a caption (try the Caption Writer node).")
        if len(caption.split()) < SHORT_WORDS:
            print(f"[RD Flow 64] note: {len(caption.split())} words. It draws best from 120-200 word captions.")
        images = rd_flow_64(caption, n=batch_size, seed=seed, transparent=background == "transparent",
                            palette=_palette(palette), steps=steps, guidance=guidance)
        rgba = np.stack([np.asarray(im.convert("RGBA")) for im in images]).astype(np.float32) / 255.0
        if scale > 1:
            rgba = rgba.repeat(scale, axis=1).repeat(scale, axis=2)
        image = torch.from_numpy(rgba[..., :3] * rgba[..., 3:4])
        mask = torch.from_numpy(1.0 - rgba[..., 3])          # ComfyUI's convention (as Load Image): 1 = transparent
        first = np.asarray(images[0].convert("RGBA")).reshape(-1, 4)
        cols = sorted({tuple(p[:3]) for p in first if p[3] > 0})
        return (image, mask, ", ".join("#%02x%02x%02x" % c for c in cols), torch.from_numpy(rgba))


def _llm_key(key: str) -> str:
    return key or os.environ.get("RDFLOW_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""


class RDFlow64CaptionWriter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "request": ("STRING", {"multiline": True, "default": "a red health potion"}),
                    "background": (["transparent", "opaque"], {"default": "transparent"}),
                    "base_url": ("STRING", {"default": OLLAMA, "tooltip": "Any OpenAI-compatible chat API: Ollama "
                                 "(default), LM Studio, llama.cpp, or https://openrouter.ai/api/v1."}),
                    "model": ("STRING", {"default": "", "tooltip": "The model name on that server, e.g. qwen3:8b for "
                              "Ollama or deepseek/deepseek-v4.1-flash on OpenRouter."}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True})},
                "optional": {
                    "api_key": ("STRING", {"default": "", "tooltip": "Leave empty to use the RDFLOW_LLM_API_KEY or "
                                "OPENROUTER_API_KEY environment variable (keys typed here are saved in the workflow)."}),
                    "palette": ("STRING", {"default": "", "tooltip": "The palette you will lock to, so the caption "
                                "names those colours."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(self, request, background, base_url, model, seed, api_key="", palette=""):
        if not model.strip():
            raise ValueError("Set the model name your LLM server runs (e.g. qwen3:8b for Ollama).")
        pal = [c for c in re.split(r"[\s,;]+", palette or "") if c.strip()] or None
        caption = write_caption(request, base=base_url, model=model.strip(), key=_llm_key(api_key),
                                transparent=background == "transparent", palette=pal, seed=seed)
        print(f"[RD Flow 64] caption ({len(caption.split())} words): {caption}")
        return (caption,)


class RDFlow64SystemPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"request": ("STRING", {"multiline": True, "default": "a red health potion"}),
                             "background": (["transparent", "opaque"], {"default": "transparent"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1})},
                "optional": {"palette": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "user_message")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, request, background, seed, palette=""):
        pal = [c for c in re.split(r"[\s,;]+", palette or "") if c.strip()] or None
        msgs = messages(request, background == "transparent", pal, seed)
        return (msgs[0]["content"], msgs[1]["content"])


class RDFlow64CleanCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"llm_output": ("STRING", {"multiline": True, "forceInput": True}),
                             "background": (["transparent", "opaque"], {"default": "transparent"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "clean"
    CATEGORY = CATEGORY

    def clean(self, llm_output, background):
        text = finalize(llm_output, background == "transparent")
        if not text:
            raise ValueError("The LLM's answer was too short to use as a caption.")
        return (text,)


NODE_CLASS_MAPPINGS = {
    "RDFlow64Loader": RDFlow64Loader,
    "RDFlow64Generate": RDFlow64Generate,
    "RDFlow64CaptionWriter": RDFlow64CaptionWriter,
    "RDFlow64SystemPrompt": RDFlow64SystemPrompt,
    "RDFlow64CleanCaption": RDFlow64CleanCaption,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RDFlow64Loader": "RD Flow 64 Loader",
    "RDFlow64Generate": "RD Flow 64 Generate",
    "RDFlow64CaptionWriter": "RD Flow 64 Caption Writer",
    "RDFlow64SystemPrompt": "RD Flow 64 System Prompt",
    "RDFlow64CleanCaption": "RD Flow 64 Clean Caption",
}
