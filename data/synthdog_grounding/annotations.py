"""Dataclasses for structured text annotations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WordAnnotation:
    text: str
    bbox: list[float]
    line_id: int
    word_id: int
    quad: list[list[float]] | None = None


@dataclass
class LineAnnotation:
    text: str
    bbox: list[float]
    block_id: int
    line_id: int
    quad: list[list[float]] | None = None


@dataclass
class BlockAnnotation:
    block_id: int
    bbox: list[float]
    line_ids: list[int]
