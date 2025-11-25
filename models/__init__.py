# models/__init__.py

"""
Model pipelines for Assignment 1.

Currently exposes v5 as the default pipeline:
- GroundingDINO + CLIP + count-aware logic.
"""

from .validate_full_pipeline_v5 import (
    IMAGES_DIR,
    META_DIR,
    OUT_DIR,
    CONFIG_PATH,
    CKPT_PATH,
    process_single_bin,
    draw_boxes,
)

__all__ = [
    "IMAGES_DIR",
    "META_DIR",
    "OUT_DIR",
    "CONFIG_PATH",
    "CKPT_PATH",
    "process_single_bin",
    "draw_boxes",
]
