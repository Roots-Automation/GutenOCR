"""vLLM-based inference module for OCR Vision Language Models."""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from dataset import OCREntry, TarShardDataset

logger = logging.getLogger(__name__)


def _is_deepseek_ocr(model_name: str) -> bool:
    m = (model_name or "").lower()
    return (
        "deepseek-ocr" in m
        or "deepseek_ai/deepseek-ocr" in m
        or "deepseek-ai/deepseek-ocr" in m
    )


def _format_prompt_for_deepseek(text_prompt: str, system_prompt: str) -> str:
    # vLLM recipe uses: "<image>\nFree OCR."
    sys = (system_prompt or "").strip()
    user = (text_prompt or "").strip()
    if sys:
        return f"{sys}\n<image>\n{user}"
    return f"<image>\n{user}"


class OCRVLMPredictor:
    """
    A class for batch inference with Vision Language Models using vLLM for OCR tasks.

    This class provides efficient batch prediction capabilities for VLMs,
    specifically optimized for OCR tasks with the Qwen2.5-VL series of models.

    Attributes:
        model_name: The name or path of the model to use.
        llm: The vLLM LLM instance.
        sampling_params: Default sampling parameters for generation.
        system_prompt: System prompt for OCR tasks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_model_len: int = 8192,
        limit_mm_per_prompt: dict[str, int] | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        system_prompt_path: str = "system_prompt.txt",
        **kwargs,
    ):
        """
        Initialize the OCR VLM predictor with vLLM.

        Args:
            model_name: The name or path of the model to load.
            max_model_len: Maximum sequence length for the model.
            limit_mm_per_prompt: Dictionary specifying limits for multimodal inputs.
            tensor_parallel_size: Number of GPUs to use for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0).
            system_prompt_path: Path to system prompt file.
            **kwargs: Additional arguments to pass to vLLM's LLM constructor.
        """
        self.model_name = model_name
        self.is_deepseek = _is_deepseek_ocr(model_name)

        # Set default multimodal limits if not provided
        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 1}  # OCR typically uses single images

        # Load system prompt
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, "r") as f:
                self.system_prompt = f.read().strip()
            logger.info(f"Loaded system prompt from {system_prompt_path}")
        else:
            self.system_prompt = "You are an expert OCR assistant. Please read and analyze the provided images accurately."
            logger.warning(
                f"System prompt file not found at {system_prompt_path}, using default"
            )

        logger.info(f"Initializing vLLM with model: {model_name}")
        logger.info(f"Max model length: {max_model_len}")
        logger.info(f"Multimodal limits: {limit_mm_per_prompt}")

        # Initialize vLLM
        # Build kwargs for vLLM
        llm_kwargs = dict(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            **kwargs,
        )

        if self.is_deepseek:
            # DeepSeek-OCR vLLM recipe uses these flags
            llm_kwargs.update(
                dict(
                    trust_remote_code=True,
                    enable_prefix_caching=False,
                    mm_processor_cache_gb=0,
                )
            )

            # Optional: if available in your vLLM version, enable DeepSeek's logits processor
            try:
                from vllm.model_executor.models.deepseek_ocr import (
                    NGramPerReqLogitsProcessor,
                )

                llm_kwargs["logits_processors"] = [NGramPerReqLogitsProcessor]
                logger.info("Enabled DeepSeek-OCR NGramPerReqLogitsProcessor.")
            except Exception as e:
                logger.warning(
                    f"Could not enable DeepSeek-OCR logits processor (continuing): {e}"
                )

            # IMPORTANT: do NOT pass limit_mm_per_prompt here; your vLLM will reject it
            # if the model isn't recognized as multimodal (or uses different MM config path).
        else:
            # Qwen path: keep your existing multimodal limit behavior
            llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

        self.llm = LLM(**llm_kwargs)

        # Initialize tokenizer for token-level metrics
        try:
            self.tokenizer = (
                AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if self.is_deepseek
                else AutoTokenizer.from_pretrained(model_name)
            )
            logger.info(f"Loaded tokenizer for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
            self.tokenizer = None

        # Default sampling parameters for OCR (typically want deterministic output)
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding for consistency
            max_tokens=2048,  # OCR output can be long
            top_p=1.0,
        )

        logger.info("vLLM initialization complete")

    def format_prompt_for_qwen(
        self, image_path: str, text_prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Format the prompt for Qwen2.5-VL models with correct vision tokens.

        Args:
            image_path: Path to the image file.
            text_prompt: The text instruction/question.
            system_prompt: Optional system prompt to override default.

        Returns:
            Formatted prompt string for Qwen2.5-VL.
        """
        sys_prompt = system_prompt or self.system_prompt

        if sys_prompt:
            formatted_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"

        return formatted_prompt

    def predict_batch(
        self,
        prompts: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        **sampling_kwargs,
    ) -> List[str]:
        """
        Perform batch prediction on a list of prompts.

        Args:
            prompts: List of prompt dictionaries with image paths and text.
                    Each should have 'image_path', 'text_prompt', and optionally 'system_prompt'.
            temperature: Sampling temperature (0.0 for greedy).
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            **sampling_kwargs: Additional sampling parameters.

        Returns:
            List of generated text responses.
        """
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            skip_special_tokens=False if self.is_deepseek else True,
            **sampling_kwargs,
        )

        if self.is_deepseek:
            # recipe sets extra_args for the ngram processor
            sampling_params.extra_args = dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # <td>, </td> per recipe
            )

        logger.info(f"Running batch inference on {len(prompts)} prompts")

        # Prepare prompt dictionaries for vLLM generate method
        vllm_prompts = []

        for prompt in prompts:
            # Extract components
            image_path = prompt["image_path"]
            text_prompt = prompt["text_prompt"]
            system_prompt = prompt.get("system_prompt")

            # Format the text prompt for Qwen2.5-VL
            # Format prompt (Qwen vs DeepSeek)
            if self.is_deepseek:
                formatted_text = _format_prompt_for_deepseek(
                    text_prompt=text_prompt,
                    system_prompt=(system_prompt or self.system_prompt),
                )
            else:
                formatted_text = self.format_prompt_for_qwen(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    system_prompt=system_prompt,
                )

            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                # Create a placeholder or skip
                continue

            # Create prompt dict for vLLM
            vllm_prompt = {
                "prompt": formatted_text,
                "multi_modal_data": {"image": image},
            }
            vllm_prompts.append(vllm_prompt)

        if not vllm_prompts:
            logger.error("No valid prompts to process")
            return []

        # Run inference (let vLLM handle batching automatically)
        outputs = self.llm.generate(vllm_prompts, sampling_params=sampling_params)

        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)

        logger.info(f"Batch inference complete. Generated {len(results)} responses")
        return results

    def predict_single(
        self,
        image_path: str,
        text_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **sampling_kwargs,
    ) -> str:
        """
        Convenience method for single prediction.

        Args:
            image_path: Path to the image file.
            text_prompt: The text instruction/question.
            system_prompt: Optional system prompt to override default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **sampling_kwargs: Additional sampling parameters.

        Returns:
            Generated text response.
        """
        prompt = {
            "image_path": image_path,
            "text_prompt": text_prompt,
            "system_prompt": system_prompt,
        }

        results = self.predict_batch(
            [prompt], temperature=temperature, max_tokens=max_tokens, **sampling_kwargs
        )
        return results[0] if results else ""
