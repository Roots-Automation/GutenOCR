#!/usr/bin/env python3
"""
Centralized prompt extractors for Fox OCR tasks.
Re-exports existing extractors from vllm_ocr_predictor and defines missing ones.
"""

import re
from typing import Any, Optional

def create_prompt_extractor(coord_mode: str = "absolute", model_name: str = None, task: str = None):
    """Prompt extractor with explicit coordinate mode.
    
    Args:
        coord_mode: 'absolute' converts [0,1000] to pixels; 'relative' keeps as-is.
        model_name: Model identifier (reserved for future model-specific behavior).
        task: Task type (box, line, page, onbox).
    """
    # Silence unused parameter warnings; reserved for future model/task-specific prompts
    _ = model_name, task
    
    point_re = re.compile(r"\[(\d+),\s*(\d+)\]")
    rect_re = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")

    def _convert_to_absolute(text: str, image_size: Optional[tuple[int, int]]):
        if image_size is None:
            return text
        width, height = image_size

        def _clamp(v: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, v))

        # Replace all point occurrences
        def _point_sub(m: re.Match[str]) -> str:
            x = _clamp(int(m.group(1)), 0, 1000)
            y = _clamp(int(m.group(2)), 0, 1000)
            x_abs = int(x * width / 1000)
            y_abs = int(y * height / 1000)
            return f"[{x_abs}, {y_abs}]"

        # Replace all rect occurrences
        def _rect_sub(m: re.Match[str]) -> str:
            x1 = _clamp(int(m.group(1)), 0, 1000)
            y1 = _clamp(int(m.group(2)), 0, 1000)
            x2 = _clamp(int(m.group(3)), 0, 1000)
            y2 = _clamp(int(m.group(4)), 0, 1000)
            x1_abs = int(x1 * width / 1000)
            y1_abs = int(y1 * height / 1000)
            x2_abs = int(x2 * width / 1000)
            y2_abs = int(y2 * height / 1000)
            return f"[{x1_abs}, {y1_abs}, {x2_abs}, {y2_abs}]"

        text = point_re.sub(_point_sub, text)
        text = rect_re.sub(_rect_sub, text)
        return text

    def extract_prompt(ann: Any, image_size: Optional[tuple[int, int]] = None):
        user_prompt = ann["conversations"][0]["value"].replace("<image>\n", "").strip()

        if coord_mode == "absolute":
            user_prompt = _convert_to_absolute(user_prompt, image_size)
        # coord_mode == "relative" or unknown: keep normalized [0,1000] coordinates as-is
        return user_prompt
    return extract_prompt


__all__ = [
    "create_prompt_extractor",
]
