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
import os

from rdflow.llm import OLLAMA, OPENROUTER, write_caption


def _config() -> tuple[str, str, str]:
    key = os.environ.get("RDFLOW_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    base = os.environ.get("RDFLOW_LLM_BASE_URL") or (OPENROUTER if os.environ.get("OPENROUTER_API_KEY") else OLLAMA)
    model = os.environ.get("RDFLOW_LLM_MODEL") or ("deepseek/deepseek-v4.1-flash" if base == OPENROUTER else "")
    if not model:
        raise SystemExit("Set RDFLOW_LLM_MODEL to the model your local server runs (for Ollama, e.g. the name shown "
                         "by `ollama list`), or set OPENROUTER_API_KEY to use OpenRouter.")
    return base.rstrip("/"), model, key


def expand(request: str, transparent: bool = True, palette=None, attempts: int = 3) -> str:
    """The caption for a short request. Falls back to the request itself if the LLM never gives a usable answer."""
    base, model, key = _config()
    return write_caption(request, base=base, model=model, key=key, transparent=transparent, palette=palette,
                         attempts=attempts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request")
    ap.add_argument("--opaque", action="store_true", help="a full-frame picture instead of a transparent sprite")
    ap.add_argument("--palette", default="", help="comma-separated hex colours the picture is limited to")
    a = ap.parse_args()
    print(expand(a.request, transparent=not a.opaque, palette=a.palette.split(",") if a.palette else None))


if __name__ == "__main__":
    main()
