"""
SynthDoG Grounding - Synthetic Document Generator with Grounding Annotations

A self-contained module for generating synthetic document images with
bounding box annotations for text lines. Based on the SynthDoG component
of the Donut project by NAVER Corp.

Copyright (c) 2022-present NAVER Corp.
MIT License
"""

from . import pillow_compat  # noqa: F401
from elements import Background, Content, Document, Paper, TextBox
from layouts import Grid, GridStack
from template import SynthDoG

__version__ = "0.1.0"
__all__ = [
    "SynthDoG",
    "Background",
    "Content",
    "Document",
    "Paper",
    "TextBox",
    "Grid",
    "GridStack",
]
