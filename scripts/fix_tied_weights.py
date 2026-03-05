#!/usr/bin/env python3
"""Fix tied lm_head weights for transformers>=5.0 compatibility.

GutenOCR checkpoints have ``tie_word_embeddings: true`` and omit
``lm_head.weight`` from the saved safetensors (it is shared with
``model.embed_tokens.weight``).  In transformers>=5.0 the weight-tying
resolution changed for ``*ForConditionalGeneration`` models with nested
configs, leaving ``lm_head.weight`` randomly initialized and producing
gibberish output.

This script:
  1. Loads a GutenOCR checkpoint.
  2. Copies ``embed_tokens.weight`` → ``lm_head.weight`` as an
     independent parameter.
  3. Sets ``tie_word_embeddings = False`` in both top-level and
     ``text_config``.
  4. Re-saves the checkpoint with ``save_pretrained()`` (safetensors).

Usage
-----
    # Save locally
    python scripts/fix_tied_weights.py \
        --model-path rootsautomation/GutenOCR-7B \
        --output-path ./GutenOCR-7B-fixed

    # Fix and push back to Hub
    python scripts/fix_tied_weights.py \
        --model-path rootsautomation/GutenOCR-7B \
        --output-path ./GutenOCR-7B-fixed \
        --push-to-hub rootsautomation/GutenOCR-7B
"""

from __future__ import annotations

import argparse
import logging

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def fix_tied_weights(model: Qwen2_5_VLForConditionalGeneration) -> bool:
    """Untie ``lm_head.weight`` so it is explicitly saved.

    Returns ``True`` if the weights were tied and have been fixed,
    ``False`` if they were already independent.
    """
    embed_w = model.model.embed_tokens.weight
    head_w = model.lm_head.weight

    if embed_w.data_ptr() == head_w.data_ptr():
        log.info("lm_head.weight is tied to embed_tokens.weight – cloning …")
        model.lm_head.weight = torch.nn.Parameter(embed_w.clone())
    else:
        log.info("lm_head.weight is already independent – no clone needed.")

    # Always ensure the config flags are correct.
    model.config.tie_word_embeddings = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.tie_word_embeddings = False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix tied lm_head weights for transformers>=5.0 compatibility."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model ID or local path to the checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Local directory to save the fixed checkpoint.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="If set, push the fixed checkpoint to this HuggingFace Hub repo ID.",
    )
    args = parser.parse_args()

    log.info("Loading model from %s …", args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    fix_tied_weights(model)

    log.info("Saving fixed checkpoint to %s …", args.output_path)
    model.save_pretrained(args.output_path)

    # Also copy the processor so the output dir is self-contained.
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.save_pretrained(args.output_path)

    log.info("Saved.")

    if args.push_to_hub:
        log.info("Pushing to Hub: %s …", args.push_to_hub)
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)
        log.info("Pushed.")


if __name__ == "__main__":
    main()
