"""Turn a short idea into the long caption RD Flow 64 draws from, with any OpenAI-compatible LLM.

    python enhance.py "a red potion"                 # prints the caption
    python enhance.py "a castle on a hill" --opaque

Where the LLM runs (environment variables):
    RDFLOW_LLM_BASE_URL   default: OpenRouter if OPENROUTER_API_KEY is set, else Ollama at http://localhost:11434/v1
    RDFLOW_LLM_MODEL      default on OpenRouter: deepseek/deepseek-v4.1-flash; required for local servers
    RDFLOW_LLM_API_KEY    or OPENROUTER_API_KEY; local servers (Ollama, LM Studio) need none

The rules are in rdflow/prompting.py and PROMPTING.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request

from rdflow.prompting import finalize, messages

OPENROUTER = "https://openrouter.ai/api/v1"
OLLAMA = "http://localhost:11434/v1"


def _config() -> tuple[str, str, str]:
    key = os.environ.get("RDFLOW_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    base = os.environ.get("RDFLOW_LLM_BASE_URL") or (OPENROUTER if os.environ.get("OPENROUTER_API_KEY") else OLLAMA)
    model = os.environ.get("RDFLOW_LLM_MODEL") or ("deepseek/deepseek-v4.1-flash" if base == OPENROUTER else "")
    if not model:
        raise SystemExit("Set RDFLOW_LLM_MODEL to the model your local server runs (for Ollama, e.g. the name shown "
                         "by `ollama list`), or set OPENROUTER_API_KEY to use OpenRouter.")
    return base.rstrip("/"), model, key


def _chat(base: str, model: str, key: str, msgs: list, temperature: float) -> str:
    body = json.dumps({"model": model, "messages": msgs, "temperature": temperature, "max_tokens": 700}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def expand(request: str, transparent: bool = True, palette=None, attempts: int = 3) -> str:
    """The caption for a short request. Falls back to the request itself if the LLM never gives a usable answer."""
    base, model, key = _config()
    seed = random.randrange(1 << 30)
    for i in range(attempts):
        try:
            text = finalize(_chat(base, model, key, messages(request, transparent, palette, seed + i), 0.5 + 0.1 * i),
                            transparent)
        except OSError as exc:
            print(f"LLM request failed ({exc}); is the server at {base} running?")
            continue
        if text:
            return text
    return request


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request")
    ap.add_argument("--opaque", action="store_true", help="a full-frame picture instead of a transparent sprite")
    ap.add_argument("--palette", default="", help="comma-separated hex colours the picture is limited to")
    a = ap.parse_args()
    print(expand(a.request, transparent=not a.opaque, palette=a.palette.split(",") if a.palette else None))


if __name__ == "__main__":
    main()
