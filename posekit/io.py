"""Small I/O helpers shared by the pipeline steps."""

from __future__ import annotations

import re
from pathlib import Path


def natural_sort_key(path: str | Path):
    """Sort ``1.jpg, 2.jpg, 10.jpg`` in numeric instead of lexicographic order.

    ``sorted(glob("*.jpg"))`` yields ``1, 10, 11, ..., 2, 3`` which silently
    breaks the 1:1 pairing between images and rows of ``joint_angles.txt``.
    """
    name = Path(path).name.lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def list_images(directory: str | Path, extensions=(".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
    """Return the image files in *directory*, ordered the same way as the joints."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    exts = {e.lower() for e in extensions}
    return sorted((p for p in directory.iterdir() if p.suffix.lower() in exts), key=natural_sort_key)
