"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
from collections import OrderedDict
import re

import numpy as np
from synthtiger import components
from datasets import load_dataset

from elements.textbox import TextBox
from layouts import GridStack


class TextReader:
    def __init__(self, path, cache_size=2 ** 28, block_size=2 ** 20):
        self.fp = open(path, "r", encoding="utf-8")
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
    def __init__(self, dataset_name="HuggingFaceFW/finepdfs", split="train", streaming=True, buffer_size=1000, subset=None):
        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming
        self.buffer_size = buffer_size
        self.subset = subset
        
        # Load the dataset in streaming mode
        if subset is not None:
            self.dataset = load_dataset(dataset_name, subset, split=split, streaming=streaming)
        else:
            self.dataset = load_dataset(dataset_name, split=split, streaming=streaming)

        # Initialize text buffer and position tracking
        self.text_buffer = []
        self.current_text = ""
        self.idx = 0
        self.dataset_iter = iter(self.dataset)
        
        # Pre-load some text
        self._fill_buffer()
        
    def _fill_buffer(self):
        """Fill the buffer with text from the next few documents"""
        try:
            for _ in range(self.buffer_size):
                sample = next(self.dataset_iter)
                # Extract text content from the PDF document
                if 'text' in sample:
                    text = sample['text']
                elif 'content' in sample:
                    text = sample['content']
                else:
                    # If we can't find text directly, try to get it from other fields
                    text = str(sample)
                
                # Clean the text - remove excessive whitespace, keep only printable chars
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    self.text_buffer.append(text)
        except StopIteration:
            # If we run out of data, restart the iterator
            self.dataset_iter = iter(self.dataset)
            if not self.text_buffer:  # Only refill if buffer is empty
                self._fill_buffer()
    
    def _get_current_text(self):
        """Get current concatenated text from buffer"""
        if not self.text_buffer:
            self._fill_buffer()
        return " ".join(self.text_buffer)
    
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
            return ' '  # Return space if no text available
        if self.idx >= len(current_text):
            self.idx = 0  # Reset to beginning if index out of bounds
        return current_text[self.idx]
    
    def _refresh_buffer(self):
        """Refresh the buffer with new text"""
        # Keep some text from current buffer and add new text
        if len(self.text_buffer) > self.buffer_size // 4:
            self.text_buffer = self.text_buffer[-self.buffer_size // 4:]
        
        try:
            for _ in range(self.buffer_size * 3 // 4):
                sample = next(self.dataset_iter)
                if 'text' in sample:
                    text = sample['text']
                elif 'content' in sample:
                    text = sample['content']
                else:
                    text = str(sample)
                
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    self.text_buffer.append(text)
        except StopIteration:
            self.dataset_iter = iter(self.dataset)


class Content:
    def __init__(self, config):
        self.margin = config.get("margin", [0, 0.1])
        
        # Choose text reader based on configuration
        text_config = config.get("text", {})
        if text_config.get("use_huggingface", False):
            # Use HuggingFace dataset reader
            hf_config = {
                "dataset_name": text_config.get("dataset_name", "HuggingFaceFW/finepdfs"),
                "split": text_config.get("split", "train"),
                "streaming": text_config.get("streaming", True),
                "buffer_size": text_config.get("buffer_size", 1000),
                "subset": text_config.get("subset", None)
            }
            self.reader = HuggingFaceTextReader(**hf_config)
        else:
            # Use traditional file-based text reader
            self.reader = TextReader(**text_config)
            
        self.font = components.BaseFont(**config.get("font", {}))
        self.layout = GridStack(config.get("layout", {}))
        self.textbox = TextBox(config.get("textbox", {}))
        self.textbox_color = components.Switch(components.Gray(), **config.get("textbox_color", {}))
        self.content_color = components.Switch(components.Gray(), **config.get("content_color", {}))

    def generate(self, size):
        width, height = size

        layout_left = width * np.random.uniform(self.margin[0], self.margin[1])
        layout_top = height * np.random.uniform(self.margin[0], self.margin[1])
        layout_width = max(width - layout_left * 2, 0)
        layout_height = max(height - layout_top * 2, 0)
        layout_bbox = [layout_left, layout_top, layout_width, layout_height]

        text_layers, texts = [], []
        layouts = self.layout.generate(layout_bbox)
        self.reader.move(np.random.randint(len(self.reader)))

        for layout in layouts:
            font = self.font.sample()

            for bbox, align in layout:
                x, y, w, h = bbox
                text_layer, text = self.textbox.generate((w, h), self.reader, font)
                self.reader.prev()

                if text_layer is None:
                    continue

                text_layer.center = (x + w / 2, y + h / 2)
                if align == "left":
                    text_layer.left = x
                if align == "right":
                    text_layer.right = x + w

                self.textbox_color.apply([text_layer])
                text_layers.append(text_layer)
                texts.append(text)

        self.content_color.apply(text_layers)

        return text_layers, texts
