"""Dataclasses for structured text annotations."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    words: list[WordAnnotation] = field(default_factory=list)


@dataclass
class BlockAnnotation:
    block_id: int
    bbox: list[float]
    line_ids: list[int]
