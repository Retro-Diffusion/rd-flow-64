"""RD Flow 64 by Retro Diffusion: a small text-to-pixel-art model that draws 64 x 64 images directly on the pixel grid.

    from rdflow import Pipeline
    pipe = Pipeline.load()
    images = pipe(caption, n=4, transparent=True)

Try the full Retro Diffusion models at https://retrodiffusion.ai
"""

from .pipeline import Pipeline, choose_placement, parse_hex, read_palette_file

__all__ = ["Pipeline", "choose_placement", "parse_hex", "read_palette_file"]
__version__ = "1.0.0"
