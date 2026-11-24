"""Writing RD Flow 64 captions with any OpenAI-compatible chat API (Ollama, LM Studio, llama.cpp, OpenRouter, ...),
using the recommended system prompt in prompting.py."""

from __future__ import annotations

import json
import random
import urllib.request
from typing import Optional, Sequence

from .prompting import finalize, messages

OPENROUTER = "https://openrouter.ai/api/v1"
OLLAMA = "http://localhost:11434/v1"


def chat(base: str, model: str, key: str, msgs: list, temperature: float = 0.6, timeout: int = 120) -> str:
    body = json.dumps({"model": model, "messages": msgs, "temperature": temperature, "max_tokens": 700}).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def write_caption(request: str, *, base: str, model: str, key: str = "", transparent: bool = True,
                  palette: Optional[Sequence[str]] = None, seed: Optional[int] = None, attempts: int = 3) -> str:
    """The full caption for a short request. Falls back to the request itself if the LLM never gives a usable answer."""
    seed = random.randrange(1 << 30) if seed is None else seed
    for i in range(attempts):
        try:
            text = finalize(chat(base, model, key, messages(request, transparent, palette, seed + i), 0.5 + 0.1 * i),
                            transparent)
        except OSError as exc:
            print(f"[RD Flow 64] LLM request failed ({exc}); is the server at {base} running?")
            continue
        if text:
            return text
    return request
