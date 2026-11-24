"""Generate 64 x 64 pixel art with RD Flow 64.

    python generate.py "A small red potion bottle with a cork stopper ..."      # a full caption
    python generate.py "red potion" --expand                                   # a short idea, expanded by an LLM
    python generate.py                                                         # interactive: load once, ask forever

RD Flow 64 was trained on long, careful descriptions (120-200 words). Short prompts work, but much worse: write a full
caption (see PROMPTING.md) or pass --expand to have an LLM write one.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from PIL import Image

from rdflow import Pipeline, parse_hex, read_palette_file

SHORT_WORDS = 40


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "image"


def save(images, caption: str, out: Path, scale: int, seeds) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    stem = slug(caption)
    paths = []
    for im, s in zip(images, seeds):
        p = out / f"{stem}-{s}.png"
        im.save(p)
        if scale > 1:
            im.resize((64 * scale, 64 * scale), Image.NEAREST).save(out / f"{stem}-{s}-x{scale}.png")
        paths.append(p)
    (out / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="a caption (or a short idea with --expand); leave out for interactive")
    ap.add_argument("--expand", action="store_true", help="have an LLM write the long caption (see enhance.py)")
    ap.add_argument("--n", type=int, default=4, help="images per prompt (default 4)")
    ap.add_argument("--seed", type=int, default=None, help="first seed (default random)")
    ap.add_argument("--opaque", action="store_true", help="a full-frame picture instead of a transparent sprite")
    ap.add_argument("--palette", default="",
                    help="lock every pixel to these colours: comma-separated hex, or a .hex/.gpl/.pal/.txt file")
    ap.add_argument("--steps", type=int, default=16, help="sampling steps (default 16; 32 gives the same images)")
    ap.add_argument("--guidance", type=float, default=4.0, help="3-4 is best; higher swells subjects and flattens colour")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--scale", type=int, default=8, help="also save an upscaled copy (nearest neighbour); 1 = off")
    ap.add_argument("--text-device", default="auto", help="auto | cpu | cuda | mps (the 3.1B prompt reader)")
    ap.add_argument("--model-device", default="auto", help="auto | cpu | cuda | mps (the 271M drawing model)")
    ap.add_argument("--precision", default="auto", help="auto | bf16 | fp32 (drawing model; auto = bf16 if the GPU has it, else fp32)")
    ap.add_argument("--weights", default=None, help="a local rd-flow-64.safetensors (default: download from Hugging Face)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the drawing model (Linux + CUDA; slow start)")
    a = ap.parse_args()

    palette = None
    if a.palette:
        palette = read_palette_file(a.palette) if Path(a.palette).is_file() else parse_hex(a.palette.split(","))
        print(f"palette locked: {len(palette)} colours")

    t0 = time.perf_counter()
    pipe = Pipeline.load(a.weights, text_device=a.text_device, model_device=a.model_device, precision=a.precision,
                         compile=a.compile)
    print(f"loaded in {time.perf_counter() - t0:.0f} s: {pipe.placement}")

    def run(prompt: str) -> None:
        caption = prompt.strip()
        if a.expand:
            from enhance import expand
            caption = expand(caption, transparent=not a.opaque, palette=palette)
            print(f"\ncaption: {caption}\n")
        elif len(caption.split()) < SHORT_WORDS:
            print(f"note: {len(caption.split())} words. RD Flow 64 draws best from 120-200 word captions; "
                  "try --expand or see PROMPTING.md.")
        images = pipe(caption, n=a.n, seed=a.seed, transparent=not a.opaque, palette=palette, steps=a.steps,
                      guidance=a.guidance)
        paths = save(images, caption, Path(a.out), a.scale, pipe.timings["seeds"])
        t = pipe.timings
        print(f"{len(paths)} images in {t['prompt_s'] + t['draw_s']:.1f} s (prompt {t['prompt_s']} s, "
              f"drawing {t['draw_s']} s) -> {paths[0].parent}")

    if a.prompt:
        run(a.prompt)
        return
    print("Type a caption (or an idea, with --expand) and press Enter. Ctrl+C to quit.")
    try:
        while True:
            line = input("\n> ").strip()
            if line:
                run(line)
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
