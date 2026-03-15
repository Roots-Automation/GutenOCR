"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Protocol, runtime_checkable

import numpy as np
from synthtiger import components

logger = logging.getLogger(__name__)

try:
    from layouts import GridStack, Layout
except ImportError:
    from ..layouts import GridStack, Layout

from .textbox import TextBox


def _relative_luminance(r, g, b):
    def channel(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


@runtime_checkable
class TextCursor(Protocol):
    """Protocol shared by all text readers (file-backed and streaming)."""

    def __len__(self) -> int: ...
    def __iter__(self) -> TextCursor: ...
    def __next__(self) -> str: ...
    def move(self, idx: int) -> None: ...
    def next(self) -> None: ...
    def prev(self) -> None: ...
    def get(self) -> str: ...


class TextReader:
    def __init__(self, path, cache_size=2**28, block_size=2**20):
        self.fp = open(path, encoding="utf-8")  # noqa: SIM115
        self.length = 0
        self.offsets = [0]
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.block_size = block_size
        self.bucket_size = cache_size // block_size
        self.idx = 0

        while True:
            text = self.fp.read(self.block_size)
            if not text:
                break
            self.length += len(text)
            self.offsets.append(self.fp.tell())

    def close(self):
        if self.fp and not self.fp.closed:
            self.fp.close()

    def __del__(self):
        self.close()

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        char = self.get()
        self.next()
        return char

    def move(self, idx):
        self.idx = idx

    def next(self):
        self.idx = (self.idx + 1) % self.length

    def prev(self):
        self.idx = (self.idx - 1) % self.length

    def get(self):
        key = self.idx // self.block_size

        if key in self.cache:
            text = self.cache[key]
        else:
            if len(self.cache) >= self.bucket_size:
                self.cache.popitem(last=False)

            offset = self.offsets[key]
            self.fp.seek(offset, 0)
            text = self.fp.read(self.block_size)
            self.cache[key] = text

        self.cache.move_to_end(key)
        char = text[self.idx % self.block_size]
        return char


class HuggingFaceTextReader:
    def __init__(
        self, dataset_name="HuggingFaceFW/finepdfs", split="train", streaming=True, buffer_size=1000, subset=None
    ):
        from datasets import load_dataset

        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming
        self.buffer_size = buffer_size
        self.subset = subset
        self._warned_unrecognized = False

        # Load the dataset in streaming mode
        if subset is not None:
            self.dataset = load_dataset(dataset_name, subset, split=split, streaming=streaming)
        else:
            self.dataset = load_dataset(dataset_name, split=split, streaming=streaming)

        # Initialize text buffer and position tracking
        self.text_buffer = []
        self._joined_text_cache = None
        self.idx = 0
        self.dataset_iter = iter(self.dataset)

        # Pre-load some text
        self._fill_buffer()

    def _extract_text(self, sample):
        """Extract text from a HuggingFace sample, returning None for unrecognized formats."""
        if "text" in sample:
            return sample["text"]
        if "content" in sample:
            return sample["content"]
        if not self._warned_unrecognized:
            logger.warning(
                "Skipping HuggingFace sample with no 'text' or 'content' key (keys: %s)", list(sample.keys())
            )
            self._warned_unrecognized = True
        return None

    def _fill_buffer(self):
        """Fill the buffer with text from the next few documents"""
        for _attempt in range(2):
            try:
                for _ in range(self.buffer_size):
                    sample = next(self.dataset_iter)
                    text = self._extract_text(sample)
                    if text is None:
                        continue

                    # Clean the text - remove excessive whitespace, keep only printable chars
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        self.text_buffer.append(text)
                break  # successfully filled
            except StopIteration:
                # If we run out of data, restart the iterator
                self.dataset_iter = iter(self.dataset)
                if self.text_buffer:
                    break
        self._joined_text_cache = None

    def _get_current_text(self):
        """Get current concatenated text from buffer (cached)."""
        if not self.text_buffer:
            self._fill_buffer()
        if self._joined_text_cache is None:
            self._joined_text_cache = " ".join(self.text_buffer)
        return self._joined_text_cache

    def __len__(self):
        # Return a large number since we're streaming
        return 10**8

    def __iter__(self):
        return self

    def __next__(self):
        char = self.get()
        self.next()
        return char

    def move(self, idx):
        """Move to a specific position in the text"""
        current_text = self._get_current_text()
        if idx >= len(current_text):
            # If we need more text, refresh the buffer
            self._refresh_buffer()
            current_text = self._get_current_text()
        # _refresh_buffer already clamps self.idx, but move() sets an
        # explicit target position so we override it here.
        self.idx = idx % len(current_text) if current_text else 0

    def next(self):
        """Move to next character"""
        current_text = self._get_current_text()
        if current_text:
            self.idx = (self.idx + 1) % len(current_text)
            # If we've gone through most of the current text, refresh buffer
            if self.idx > len(current_text) * 0.8:
                self._refresh_buffer()

    def prev(self):
        """Move to previous character"""
        current_text = self._get_current_text()
        if current_text:
            self.idx = (self.idx - 1) % len(current_text)

    def get(self):
        """Get current character"""
        current_text = self._get_current_text()
        if not current_text:
            return " "  # Return space if no text available
        if self.idx >= len(current_text):
            self.idx = 0  # Reset to beginning if index out of bounds
        return current_text[self.idx]

    def _refresh_buffer(self):
        """Refresh the buffer with new text and clamp idx to stay in bounds."""
        # Keep some text from current buffer and add new text
        if len(self.text_buffer) > self.buffer_size // 4:
            self.text_buffer = self.text_buffer[-self.buffer_size // 4 :]
        self._joined_text_cache = None

        try:
            for _ in range(self.buffer_size * 3 // 4):
                sample = next(self.dataset_iter)
                text = self._extract_text(sample)
                if text is None:
                    continue

                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    self.text_buffer.append(text)
        except StopIteration:
            self.dataset_iter = iter(self.dataset)

        # The buffer may have shrunk, so clamp idx to stay in bounds.
        # Position is approximate — semantic continuity is not needed.
        new_text = self._get_current_text()
        if new_text:
            self.idx = self.idx % len(new_text)
        else:
            self.idx = 0


_READER_TYPES: dict[str, type[TextCursor]] = {
    "file": TextReader,
    "huggingface": HuggingFaceTextReader,
}

_LAYOUT_TYPES: dict[str, type] = {
    "grid_stack": GridStack,
}


class Content:
    def __init__(self, config):
        self.margin = config.get("margin", [0, 0.1])

        # Choose text reader based on configuration
        text_config = config.get("text", {})
        reader_type = text_config.get("type", "file")
        reader_cls = _READER_TYPES[reader_type]  # KeyError = clear signal of bad config
        reader_kwargs = {k: v for k, v in text_config.items() if k != "type"}
        self.reader: TextCursor = reader_cls(**reader_kwargs)

        self.font = components.BaseFont(**config.get("font", {}))
        layout_config = config.get("layout", {})
        layout_type = layout_config.get("type", "grid_stack")
        layout_cls = _LAYOUT_TYPES[layout_type]
        self.layout: Layout = layout_cls(layout_config)
        self.textbox = TextBox(config.get("textbox", {}))
        self.textbox_color_config = config.get("textbox_color", {})
        self.content_color_config = config.get("content_color", {})

    def generate(self, size, bg_color=(255, 255, 255)):
        width, height = size

        lum = _relative_luminance(*bg_color)
        gray_range = [0, 64] if lum > 0.5 else [191, 255]

        tb_args = {**self.textbox_color_config.get("args", {}), "gray": gray_range}
        tb_prob = self.textbox_color_config.get("prob", 0)
        if lum < 0.5:
            tb_prob = 1.0
        textbox_color = components.Switch(components.Gray(), prob=tb_prob, args=tb_args)

        cc_args = {**self.content_color_config.get("args", {}), "gray": gray_range}
        cc_prob = self.content_color_config.get("prob", 0)
        if lum < 0.5:
            cc_prob = 1.0
        content_color = components.Switch(components.Gray(), prob=cc_prob, args=cc_args)

        layout_left = width * np.random.uniform(self.margin[0], self.margin[1])
        layout_top = height * np.random.uniform(self.margin[0], self.margin[1])
        layout_width = max(width - layout_left * 2, 0)
        layout_height = max(height - layout_top * 2, 0)
        layout_bbox = [layout_left, layout_top, layout_width, layout_height]

        text_layers, texts, block_ids, words_per_line = [], [], [], []
        textbox_total_count = 0
        textbox_null_count = 0
        layouts = self.layout.generate(layout_bbox)
        self.reader.move(np.random.randint(len(self.reader)))

        # Each (grid_idx, col_idx) pair is a distinct visual block
        col_key_to_block_id = {}
        next_block_id = 0

        for grid_idx, layout in enumerate(layouts):
            font = self.font.sample()

            for bbox, align, col_idx in layout:
                textbox_total_count += 1
                x, y, w, h = bbox
                text_layer, text, word_local_data = self.textbox.generate((w, h), self.reader, font)

                if text_layer is None:
                    textbox_null_count += 1
                    continue

                text_layer.center = (x + w / 2, y + h / 2)
                if align == "left":
                    text_layer.left = x
                if align == "right":
                    text_layer.right = x + w

                col_key = (grid_idx, col_idx)
                if col_key not in col_key_to_block_id:
                    col_key_to_block_id[col_key] = next_block_id
                    next_block_id += 1

                textbox_color.apply([text_layer])
                text_layers.append(text_layer)
                texts.append(text)
                block_ids.append(col_key_to_block_id[col_key])
                words_per_line.append(word_local_data)

        content_color.apply(text_layers)

        return text_layers, texts, block_ids, words_per_line, textbox_null_count, textbox_total_count
