"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import logging
import re
from collections import OrderedDict
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TextCursor(Protocol):
    """Protocol shared by all text readers (file-backed and streaming)."""

    def __len__(self) -> int: ...
    def __iter__(self) -> "TextCursor": ...
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

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
        self._needs_refresh = False
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
        return len(self._get_current_text())

    def __iter__(self):
        return self

    def __next__(self):
        char = self.get()
        self.next()
        return char

    def move(self, idx):
        """Move to a specific position in the text"""
        if self._needs_refresh:
            self._refresh_buffer()
            self._needs_refresh = False
        current_text = self._get_current_text()
        self.idx = idx % len(current_text) if current_text else 0

    def next(self):
        """Move to next character"""
        current_text = self._get_current_text()
        if current_text:
            self.idx = (self.idx + 1) % len(current_text)
            # Defer buffer refresh to next move() call so we never swap
            # the buffer mid-character-iteration within a single textbox.
            if self.idx > len(current_text) * 0.8:
                self._needs_refresh = True

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
