"""The prompt reader: the first 27 of the 36 decoder layers of Qwen3-VL-4B-Instruct, frozen.

RD Flow 64 was trained on these features: raw caption tokens (no chat template), the decoder cut after layer 27, read
through the model's final norm and RMS-normalised per token. Only what is used is shipped, built and loaded: no vision
tower, no upper layers and no output head. That is about 3.1B parameters, 6.2 GB in bfloat16, instead of the full model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

TEXT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"  # the revision RD Flow 64 was trained with
# that model's layers RD Flow 64 uses, unchanged, ship in the RD Flow 64 repo (Apache 2.0, see text_encoder/NOTICE)
TEXT_REPO = "Retro-Diffusion/rd-flow-64"
TEXT_DIR = "text_encoder"
TEXT_LAYERS = 27
TEXT_LEN = 512
_PREFIXES = ("model.language_model.", "model.text_model.", "language_model.model.", "model.", "")


def _snapshot(cache_dir: str | None) -> Path:
    """Download (once) the prompt reader: its weights (6.2 GB), Qwen's config and the tokenizer."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(TEXT_REPO, cache_dir=cache_dir, allow_patterns=[f"{TEXT_DIR}/*"])) / TEXT_DIR


def _wanted(key: str) -> str | None:
    """The text-model key for a checkpoint key we need, or None."""
    if "visual" in key or "lm_head" in key:
        return None
    for p in _PREFIXES:
        if key.startswith(p):
            rest = key[len(p):]
            if rest.startswith("layers."):
                return rest if int(rest.split(".")[1]) < TEXT_LAYERS else None
            if rest.startswith(("embed_tokens.", "norm.")):
                return rest
    return None


_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _load_shard(path: Path, dtype: torch.dtype, device: torch.device) -> dict:
    """The tensors we use from one checkpoint file.

    On the CPU, at the checkpoint's own precision, they are memory-mapped from the file instead of copied: the pages are
    file-backed, so a machine short on RAM drops and re-reads them rather than swapping, and 8 GB machines can run it.
    Elsewhere each tensor goes straight to the device as it is read, so there is never a second full copy in RAM.
    """
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    wanted = {k: _wanted(k) for k in header}
    if device.type == "cpu" and all(_DTYPES.get(header[k]["dtype"]) == dtype for k, w in wanted.items() if w):
        try:
            storage = torch.UntypedStorage.from_file(str(path), shared=False, nbytes=os.path.getsize(path))
            out, size = {}, torch.tensor([], dtype=dtype).element_size()
            for k, name in wanted.items():
                if name is None:
                    continue
                start, end = header[k]["data_offsets"]
                t = torch.tensor([], dtype=dtype)
                t.set_(storage, (8 + n + start) // size, header[k]["shape"])
                out[name] = t
            return out
        except (RuntimeError, OSError):
            pass        # no memory mapping here: read normally
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        for k, name in wanted.items():
            if name is not None:
                out[name] = sf.get_tensor(k).to(device=device, dtype=dtype)
    return out


def load_tokenizer(cache_dir: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_snapshot(cache_dir))
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class TextEncoder(torch.nn.Module):
    """Captions -> [B, L, 2560] features (bfloat16 by default) and the [B, L] mask."""

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16,
                 cache_dir: str | None = None):
        super().__init__()
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        root = _snapshot(cache_dir)
        cfg = AutoConfig.from_pretrained(root).text_config
        cfg.num_hidden_layers = TEXT_LAYERS
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = cfg.layer_types[:TEXT_LAYERS]
        with torch.device("meta"):
            model = Qwen3VLTextModel(cfg)
        state = {}
        for f in sorted(root.glob("*.safetensors")):
            state.update(_load_shard(f, dtype, torch.device(device)))
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        missing = [m for m in missing if "rotary" not in m]
        if missing or unexpected:
            raise RuntimeError(f"text encoder weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        del state
        # buffers (rotary frequencies) were made on the meta device: rebuild them for real
        model.rotary_emb = type(model.rotary_emb)(config=cfg)
        self.m = model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.tok = load_tokenizer(cache_dir)

    @torch.no_grad()
    def forward(self, prompts: list[str]):
        e = self.tok(prompts, max_length=TEXT_LEN, truncation=True, padding="longest", return_tensors="pt")
        ids = e["input_ids"].to(self.device)
        attn = e["attention_mask"].bool().to(self.device)
        h = self.m(input_ids=ids, attention_mask=attn.long(), use_cache=False).last_hidden_state
        h = torch.nn.functional.rms_norm(h.float(), (h.shape[-1],)).to(torch.bfloat16)
        return h * attn[..., None], attn


def cpu_budget() -> int:
    """CPUs this process may really use: the affinity mask and any container (cgroup) quota, not the host's count.
    In a container, os.cpu_count() reports every core of the host machine."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]     # cgroup v2
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        try:                                                                       # cgroup v1
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except (OSError, ValueError):
            pass
    return max(1, n)


def threads_hint() -> None:
    """Keep PyTorch's CPU threads within the CPUs we really have (unless the user set OMP_NUM_THREADS): one thread per
    host core on a container's share of them makes CPU inference many times slower."""
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(min(torch.get_num_threads(), cpu_budget()))
