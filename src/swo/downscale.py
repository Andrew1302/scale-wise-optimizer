"""Aspect-preserving downscaling to a total-pixel budget."""

from __future__ import annotations

import math

from PIL import Image


def fit_to_budget(image: Image.Image, budget_px: int) -> Image.Image:
    """Scale ``image`` down until ``width * height <= budget_px``.

    Aspect ratio is preserved. Images already within budget are returned
    unchanged (the identical object, so callers can detect a no-op) — this
    never upscales, so a budget above the native resolution is a no-op rather
    than an interpolation artefact.
    """
    if budget_px < 1:
        raise ValueError(f"budget_px must be >= 1, got {budget_px}")

    width, height = image.size
    if width * height <= budget_px:
        return image

    # Flooring both sides keeps the product under budget: floor(w*s) * floor(h*s) <= w*h*s^2 = budget.
    scale = math.sqrt(budget_px / (width * height))
    new_width = max(1, math.floor(width * scale))
    new_height = max(1, math.floor(height * scale))

    # Clamping a side up to 1 can push an extreme aspect ratio back over budget
    # (10000x1 at a 2000px budget flooring to 4472x1); trim the long side to compensate.
    if new_width * new_height > budget_px:
        if new_width >= new_height:
            new_width = max(1, budget_px // new_height)
        else:
            new_height = max(1, budget_px // new_width)

    return image.resize((new_width, new_height), Image.LANCZOS)
